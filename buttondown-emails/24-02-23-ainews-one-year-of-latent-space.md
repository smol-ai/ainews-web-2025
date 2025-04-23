---
id: ec3ace29-f799-43c3-a386-d0d0efd46c4f
title: '[AINews] One Year of Latent Space'
date: '2024-02-24T01:05:00.357006Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-one-year-of-latent-space
---

<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/22/2024. We checked **20** guilds, **317** channels, and **8875** messages for you. Estimated reading time saved (at 200wpm): **835 minutes**.

Latent Space [turned one today](https://twitter.com/latentspacepod/status/1761043241921876069). It's (of course) the #1 AI Engineering podcast, hit #10 in the generalist U.S. Tech podcast charts, and crossing 1 million unique readers on our Substack. Alessio [wrote a great reflection](https://www.alessiofanelli.com/posts/latent-space) and we hosted a great hack/demo day that is in progress as we write.

 ![image.png](https://assets.buttondown.email/images/d982f1a6-ac16-4987-b801-85789b478300.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/0ce61aa0-d3d2-4306-922e-ecb549efdb2a.png?w=960&fit=max) 



---

**Table of Contents**

[TOC] 


# PART 0: Summary of Summaries of Summaries

- **AI Ethics and Bias Discussion**:
The Gemini Image Generator controversy on TheBloke Discord highlighted challenges in AI ethics and bias, specifically how Google's Gemini 1.5 model failed to accurately represent white individuals and historic events. This sparked debates on internal biases vs. rushed implementation, as discussed in a YouTube video on Gemini's diversity issue.
- **AI-Assisted Creativity and Development**:
AI's role in creative industries, especially in game development, was emphasized across TheBloke and LM Studio Discords. Discussions revolved around using AI for artistic direction and the potential of text-to-3D tools for smaller developers, showcasing AI's growing intersection with creativity.
- **Model Fine-Tuning and Performance Optimization**:
Several Discords, including Nous Research AI and Mistral, delved into fine-tuning challenges and performance optimization of models like Gemma 7B and Mistral-next. Issues ranged from high initial loss to API access queries, with solutions involving specific learning rates and leveraging open-source tools for superior results, such as a GitHub repository for large-scale finetuning.
- **Emerging Trends in AI Development and Deployment**:
Discussions on CUDA MODE and LangChain AI Discords underscored emerging trends in AI hardware optimization and application development. Critiques of Nvidia's CUDA by Jim Keller and explorations in parallel function calls in LLMs reflect the technical community's focus on improving AI model efficiency and deployment strategies. Notably, advancements in addressing AI hallucination were teased by Richard Socher, suggesting significant progress in enhancing AI's factual accuracy, as hinted in a tweet.

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Gemini Image Generator Sparks Bias Controversy**: The community debated the apparent bias of Google's Gemini 1.5 AI image generation model after it failed to represent white individuals and historic events accurately, prompting a shutdown; some argued it was due to internal biases while others suggested rushed implementation. The controversy was discussed with references to [Gemini's diversity issue video](https://www.youtube.com/watch?v=Fr6Teh_ox-8) and articles.

- **AI-Assisted Creativity in Game Development**: The potential for AI to assist in game development surfaced, with discussions on text-to-3D tools and the benefits for smaller developers using AI for artistic direction, showcasing the growing intersection of AI and creative industries.

- **Search Engine Market Share Discussion**: Why Google continues to dominate search engine market share piqued interest; alternatives like Qwant were discussed alongside critiques of Google's corporate ethos, underlining the competition and ethics in the tech industry.

- **Opus V1 and Other Models Take the Spotlight in Roleplay and Writing**: Users in roleplay and writing channels explored model preferences, with attention on **Opus V1's** role in story-writing and character cards' influence on AI model performance in roleplaying scenarios, reflecting the significance of fine-tuning model settings for creative outputs.

- **Deep Dives into Model Merging and DPO**: Conversations on model merging explored the challenges in hybridizing non-homologous models such as **Orca-2-13b** with **Nous-Hermes-2-DPO-7b**, discussing complex techniques and potential for knowledge transfer optimization (KTO), and community input on DPO usage; one member opted to use the `trl` library's `DPOTrainer` as a starting point after viewing its [code on GitHub](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py).

- **Code Curiosity and JetBrains' Dotpeek Usage**: In the coding channel, there was a distinct curiosity for communities focused on machine learning outside of GitHub and Twitter, as well as an exchange on the use of JetBrains' Dotpeek for **vulnerability research**, indicative of the practical applications AI engineers seek from their tools.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Gemma Models on the Frits**: Users experience issues with `Gemma` models, particularly with lower quantizations breaking in `llama.cpp`. A [Hugging Face Gemma model](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) is suggested to avoid problems, while the `Gemma 7B` must be manually downloaded for LM Studio compatibility.

- **Stability and Updates in LM Studio**: An **urgent update to LM Studio v0.2.16** includes bug fixes for erratic behaviors. Users celebrate the UI improvements and fixed issues from version 0.2.15, but also critique the complexity and Comic Sans font.

- **A TESLA in Hand Worth Two in the Data Center?**: Spare TESLA K40 cards are on the market, prompting discussions about their potential use with `llama.cpp`, despite being limited to CUDA 3.5. The conversation spans adding GPUs for speed and possible disruption by AMD's MI300X in AI applications.

- **Local Models, No Internet**: LM Studio local models like `Gemma` do not have internet access, which impacts their update and improvement capabilities. Despite the limitations, the AI-assisted teaching tools and Stable Diffusion Web UI are brought up for their functionalities.

- **Visualizing Technical Troubles**: OLED monitors get a nod for their quality, affirming a trend in preference even amongst the engineer audience. On the hardware side, the Tesla K40's cost efficiency is recognized, but with reservations due to its age and limitations.

- **Fixing the Unfixable with a Classic**: When facing AutoGen package issues, a user successfully resolved them through the classic IT approach of uninstalling and reinstalling, accented by a nod to the famed "turning it off and on again" GIF [humorously shared](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069).

- **How Chunky is Your Data?**: A discussion on `chunk_size` for text preprocessing for embeddings highlights its dependency on the model used. A recommended formula is shared from [AI Stack Exchange](https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size) for calculating `num_embeddings` when `num_categories <= 1000`.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **HTML and CSS on AI's Curriculum**: There was talk about training ChatGPT with HTML and CSS, with `@ls_chicha` probing how to include programming languages in AI education. Subsequently, `@thedreamakeem` indicated a potential requirement for a .json database format when training.

- **AI Models in Crisis with PDFs and More**: Users grappled with issues such as GPTs prematurely losing the ability to read PDFs (`@arani1977`) and slow model performance raised by `@oleksandrshr`, alongside a clarification on quantized AI versions provided by `@darthgustav.` that affect model speed and precision.

- **Fine-Tuning Model Behaviors**: Discourse extended to nuances of model responses, as `@tawsif2781` and `@darthgustav.` discussed the looping glitch in ReAct prompting, and strategies for invoking improvisation even with zero temperature settings.

- **AI Conversations and Character Play**: `@link12313` proposed an app for interactions between GPT-4 and Google’s Gemini Ultra1.5, while `@eskcanta` exchanged methods and tips for managing roleplay and consistent character interactions within models, showcasing efficient Custom Instructions usage.

- **Following GPT-4's Reality Check**: There was skepticism on dramatic changes in GPT-4's abilities post-release with `_jonpo` and others debating on the model's context length and memory capabilities, while `@lugui` dispelled concerns that GPT-4 may have been "powered down."

**External Resources Discussed**:
- A link was shared to [Stability AI's announcement](https://stability.ai/news/stable-diffusion-3) about their most advanced text-to-image model, Stable Diffusion 3.
- OpenAI's foray into video generative models was highlighted through a [research link](https://openai.com/research/video-generation-models-as-world-simulators).
- Information on Google's Gemini Pro model and its anti-bias measures appeared in a [YouTube video](https://www.youtube.com/watch?v=Fr6Teh_ox-8).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Homemade Orchestration Over Enterprise Solutions**: A custom system of "crappy scripts and a database" was mentioned for worker orchestration, hinting at a pragmatic approach over sophisticated enterprise-level solutions.

- **Stable Diffusion 3 and Hiring Practices at Stability AI**: There is anticipation for the capabilities of Stable Diffusion 3, possibly featuring a base for medium resolution with an upscaling technique. Meanwhile, Stability AI seems to exhibit a hiring trend toward systems administrators pivoting to machine learning roles, seemingly due to cost-effectiveness, as well as individuals with significant YouTube followings.

- **Increased Secrecy in AI Development**: Community members voiced concerns over a trend where companies, such as Stability AI, are moving model development away from public scrutiny and contributing to a decrease in observable diversity in AI generation outputs.

- **Open-Source Models and Fine-tuning**: There is a discussion indicating the potential for open-source models like Mistral-7b, when fine-tuned, to provide superior performance compared to commercial offerings such as GPT-4, with an initiative like LoRA Land seen as leading this space.

- **Reevaluating LAION 5B's Utility and Academic Contributions**: The community contemplates whether to retire the LAION 5B dataset, while also exploring crowd-sourced captioning solutions and sharing insights into effective model training practices, such as mixed precision training with bfloat16 on TPUs. Academic contributions in the area include TinyLLaVA—on small-scale Large Multimodal Models—and INTRINSIC LoRA's exploration of generative models' capabilities.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Gemma 7B Puzzles Engineers**: AI Engineers, like `@interstellarninja`, `@teknium`, and `@gryphepadar`, reported **finetuning challenges** with the Gemma 7B model, including high initial loss and suboptimal end results compared to existing models. `@stoicbatman` found a learning rate of 5e-5 optimal in their experiments.

- **Fine-Tuning Tools and Tips Swapped**: `@alvion427` applauded `@n8programs`'s **fine-tuned Tinyllama model** for advanced multi-turn conversation capabilities. Meanwhile, `@qtnx` rectified a naming typo for a Nous-Hermes-2-Mistral model on Huggingface, and `@teknium` provided a [GitHub link](https://github.com/AblateIt/finetune-study) to shell scripts for large-scale finetuning, albeit with a need for updates.

- **Cutting-Edge LLM Integration Discussed**: Conversations spanned Microsoft's **JARVIS project** with links to its [GitHub repository](https://github.com/microsoft/JARVIS), the OpenCodeInterpreter with a blend of generation, execution, and refinement, and `@pramod8481` sharing critical analyses on human feedback and value biases in LLMs from Arxiv links.

- **AI Models Hog VRAM**: `@gryphepadar` highlighted the considerable VRAM consumption during model finetuning, indicating the necessity for planning computational resources.

- **Ethics and Mechanics of AI Meet**: Concerns were raised about how Large Language Models (LLMs) tend to favor high-value options and the possible implicit value function within, as suggested by a [study](https://arxiv.org/abs/2402.11005) discussed by `@pramod8481`.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

**Mistral-Next Sparks Anticipation and API Queries**: Engineering discussions have revealed that **Mistral-next** is outperforming previous models like Mistral-Medium, with users like `@ethux` confirming its existence but noting the absence of API access or model size details. Meanwhile, others like `@buttercookie6265` and `@louis2567` have been focusing on GPU selection for vLLMs and best practices for batch calls to vLLM servers.

**Mistral's Open-Source Commitment Questioned**: Community concerns surfaced about Mistral potentially shifting away from open-source, but users like `@casper_ai` voiced confidence in Mistral's open ethos, making parallels to Linux. With [a variety of links mentioned](https://chat.lmsys.org), it's clear that deployment methods and accessibility remain pivotal discussions.

**Frosty Feedback for Mistral's Fine-Tuning**: Newcomers to fine-tuning like `@4vis` received recommendations such as starting with [Unsloth](https://unsloth.openai.com/), while others like `@pteromaple` grappled with the intricacies of data formats and model choices for precise tuning tasks. Users discussed the practicality of fine-tuning large models on limited hardware configurations, with `@mrdragonfox` suggesting that small parameter modifications might suffice for certain style transfers.

**Mistral Data Handling Protocols Clarified**: Inquiries about the privacy of data processed through the **Mistral API** led to assurances from `@akshay_1` about non-utilization of such data in training. Additional confirmations from `@tom_lrd` and `@ethux` noted that Mistral's data and platform are hosted in Sweden, as included in their [privacy policy](https://mistral.ai/privacy-policy/), which also mentions service providers like **Azure**, **Cloudflare**, and **Stripe**.

**Mistral Community Ponders Performance and Pricing**: Model performance, serving speeds, and attractive pricing structures brought attention, with `@egalitaristen` and `@mrdragonfox` expressing positivity about Mistral's market presence. An ongoing feedback collection initiative for Mistral Next, supported by `@egalitaristen` and `@mrdragonfox`, indicates active community involvement in model improvements.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity Presents the Discover Daily Podcast**: A partnership between Perplexity and ElevenLabs brings the **Discover Daily podcast** to life, featuring AI-powered voices from ElevenLabs narrating stories from Perplexity's Discover feed. The podcast can be found on [various platforms](https://podcast.perplexity.ai).

- **No Double Discounts on Pro Subscriptions**: Clarification was offered on Perplexity Pro subscriptions; adding team members to a plan is possible, but no multi-subscription discounts are available as confirmed by a link to the [billing and subscription FAQ](https://blog.perplexity.ai/faq/billing-and-subscription).

- **Experimenting with Lightweight GPT Models**: Perplexity showcases new lightweight "Experience Gemma 2B and 7B models" through [Perplexity Labs YouTube playlist](https://www.youtube.com/playlist?list=PLKwRkjCH760ObtANfb0-Kat2XlvB5dKxf) and promotes them on Twitter, stressing their impressive performance.

- **Navigating API Issues and Gemma Integration Speculation**: Users report trouble with API credit purchases and a successful workaround for a 400 error. Curiosity arises around integrating [Google's Gemma](https://ai.google.dev/gemma) with the Perplexity API.

- **Search Insights and Potential Collaborations**: Users utilize Perplexity AI search to explore topics like the identity of `pline0`, risk analysis, and the Xiaomi 14 series, alongside discussing a potential Perplexity AI and ElevenLabs collaboration. Links directly to Perplexity AI search results are shared in conversations.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Axolotl-Dev Insights: CUDA Confusion and Gemma Optimizations**: `@casper_ai` shared advancements in optimizing the Mixtral model but struggled with crafting a compatible backward pass without CUDA expertise. They suggested precomputing token and expert ids for efficient grouped computations to enhance Mixtral's efficiency. Meanwhile, `@curiositix` recommended the [Gemma Inference Engine](https://github.com/google/gemma.cpp/) to overcome `@casper_ai`'s backward pass implementation hurdles.

- **Discussions Over Cloud and Server Costs**: In the #general channel, `@yamashi` sparked a debate on the economic trade-offs between cloud services and owning servers for long-term AI projects, considering the costs associated with ongoing cloud rentals versus one-time server purchases.

- **Inference Woes and Contribution Pleas in General Help**: `@nani1149` and `@nanobitz` discussed the alpaca inference format in the #general-help channel, where `@nanobitz` provided a [Stanford Alpaca GitHub link](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release) for reference. 'nanobitz' and `@yamashi` pondered the necessity of improved documentation to aid community members, hinting at the use of resources like Gitbooks.

- **Community Showcase of Advanced AI Storytelling**: In the #community-showcase, `@dreamgen` announced the release of new AI models for narrative creation featured on Hugging Face and shared the [Opus V1 guide](https://dub.sh/opus-v1-guide). Addressing concerns, they confirmed an oversight in updating tokenizer chat templates and promised further investigation into alleged prompt leakage. Additionally, `@finetuningllms` spotlighted their tuning of the Phi-2 model, available at [axra/phi-2-x-0.1](https://huggingface.co/axra/phi-2-x-0.1).

- **Finding the Elusive RunPod Image**: With confusion in the #runpod-help channel over a missing RunPod image, `@nanobitz` directed users to [Docker Hub](https://hub.docker.com/r/winglian/axolotl-runpod/tags) for retrieval, however, `@stoicbatman` noted discrepancies between Docker Hub and the now-misdirecting GitHub readme.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

**Aya Dataset Visualization Shared**: A [visualization of the Aya dataset](https://huggingface.co/posts/cakiki/501967924678592) intended to improve comprehension has been provided by a user.

**Innovations in Protein Research and Language Technology**: The **ProteinBERT** model and related [paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274), as well as **Fluently diffusion model** demo at [this space](https://huggingface.co/spaces/ehristoforu/Fluently), offer advancements in understanding proteins and natural language processing.

**Stable Diffusion XL Optimization Guide Released**: New article details methods for enabling image generation on less powerful GPUs, accessible through an [article by @felixsanz](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl), even as the community welcomes Stable Diffusion 3.

**Ethical Concerns Raised Over Unofficial API**: Users express concerns over the ethical and practical implications of an unofficial ChatGPT API using Selenium, highlighting potential violation of OpenAI's terms and risk of bans. [Link to GitHub Repo](https://github.com/Priyanshu-hawk/ChatGPT-unofficial-api-selenium).

**Debate Over Fine-Tuning vs. Large Model Approaches**: The community discusses whether to fine-tune a larger LLM like Mistral 7B for text classification or use an optimized BERT variant. Encoder models are suggested as a more efficient focus for classification tasks over substantial models.

**Challenges with Expanding Models and Translation Systems**: Users discuss extending the BART MNLI model beyond 10 classes and the creation of an **Interlingua-based translator** for a university project, reflecting a broader interest in model adaptation and multilingual translation systems.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **GPT-4 Falls Short of Expectations**: `@henriqueln7` tested GPT-4's ability to rewrite prompts but found it functioned like a new assistant. Extensive testing in the playground was planned to explore its capabilities further.

- **Stable Diffusion 3 Makes a Splash**: Stability AI announced an early preview of Stable Diffusion 3 with improved performance on multi-subject prompts and spelling abilities. Detailed model information was shared by `@rubenartus` through various [links](https://stability.ai/news/stable-diffusion-3).

- **Google's Gemini Pro 1.5 Revealed**: Featuring a massive 1,000,000 token context size and video input capabilities, Gemini Pro 1.5 was discussed by `@nuvic_`, with insights sourced from Google AI Studio.

- **Debating Reddit's Lucrative Data Deal**: The community, including `@guardiang` and `@pennepitstop`, debated the implications of Google's $60 million/year data agreement with Reddit and its impact ahead of Reddit's IPO.

- **Google's Gemini Image Generation Goes Awry**: After issues with Gemini's image generation feature, Google paused its function as announced in a blog post linked by `@swyxio`.

- **LLM Paper Club Takes Deep Dive into T5**: The LLM Paper Club, hosted by `@ivanleomk` and `@bryanblackbee`, provided in-depth discussions on the T5 paper, with a central [repository for notes](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4) and insights shared among participants.

- **AI Model Merging Emerges**: The technique of model merging, a cost-effective way of combining LLMs, is highlighted, with `@swyxio` sharing [Hugging Face's blog post](https://huggingface.co/blog/mlabonne/merge-models) on the subject and referencing the mergekit library.

- **Civit.ai Gallery Sparks Debate**: The Civit.ai model gallery's content, particularly images of young women, was a point of debate, emphasizing the importance of content moderation and implications for AI-generated content in `@kbal11`'s discussion.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Debate on Simulating Human Experience**: Skepticism arose around GPT-4's ability to emulate human experiences, with discussions focused on enhancing model memory layers for more realistic behavior. The discourse extended to a company, Superfocus, claiming near-perfect factual accuracy for LLMs.

- **Validity of LLM Benchmarks Questioned**: A [YouTube video](https://youtu.be/74Uo2HU8HBo) criticising the effectiveness of current LLM benchmarks spurred conversations about the benchmarks' adequacy.

- **Exploring LLM Unlearning and Chinese Contextualization**: A study titled *Survey and formalization of LLM unlearning* was shared, and training of a Chinese lens for a 13b model was reported, with an investigation into the model's uniform output behavior and tokenizer issues.

- **Concerns over Misleading Model Naming Conventions**: A debate ensued regarding naming conventions for models, with "gemma-7b" actually comprising 8.5b parameters leading to confusion and calls for consistency.

- **Optimizing Pre-training and Finetuning Techniques for GPT-NeoX**: Published work highlighting the effects of sequence composition was shared. Discussions included the appropriateness of using LoRA finetuning within the `gpt-neox` codebase, with movement away from PyTorch native FSDP for NeoX 20B finetuning under consideration.

- **Mitigating False Negatives in Multimodal Models**: Thoughts were exchanged on the significance of exact false negatives in large datasets like **datacomp** or **metaclip**. Generating unimodal embeddings or computing similarity during training might reduce the incidence of hard negatives.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Full-Stack RAG Made Easy**: `@wenqi_glantz` provided a tutorial converting a RAG notebook to a full-stack app, including an ingestion service, detailed in [her guide](https://t.co/S86B38YZQ1). The LlamaIndex release, creating a LlamaPack for advanced RAG, makes web app implementation straightforward with two lines of code, as announced [here](https://t.co/vf0aKDv1yo).

- **ColBERT Accelerates Document Re-ranking**: `@lateinteraction` introduced ColBERT, a tool for fast document re-ranking that's 100 times speedier than BERT-based models. ColBERT's improvements were confirmed by `@Haotianzh` and can be explored in [this tweet](https://t.co/kzvNPELgQ4).

- **Navigating LlamaIndex's Documentation for RAG Setup**: `@lapexer` queried about setting up a simple RAG in QueryPipeline, with `@cheesyfishes` offering the [documentation link](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting) for guidance.

- **Trouble in IngestionPipeline Town**: Issues like `ValidationError` popped up while deploying the IngestionPipeline, but were eventually resolved through community support. It was also noted that inconsistent module imports could require a reinstallation of LlamaIndex.

- **Eager for Code Invocation Models**: `@gooooooofy` seeks models adept at generating accurate code invocations and feels **Gorilla LLM** might be on the right track, despite its API call specialization.




---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Bargain GPU Power-up**: An engineer snagged three RTX 3090s for 1.7k euros to upgrade a mining rig for LLM fine-tuning and serving, highlighting the cost efficiency. They detailed the conversion process in a [two-part blog series](https://samsja.github.io/blogs/rig/part_1/).

- **CUDA Criticism from a Silicon Veteran**: Jim Keller criticized Nvidia's CUDA, describing it as a complex and inelegant solution, analogous to the x86 architecture's evolution. The criticism was featured in a [Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too).

- **Kernel Crafting and Quantization Intricacies**: There was an emphasis on the nuance of quantized model computations, as well as CUDA kernel development for deep learning. One engineer shared their [torch-bnb-fp4 repository](https://github.com/aredden/torch-bnb-fp4) for a faster alternative to bitsandbytes, and provided a [benchmark script](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L3533-L3649) to test performance improvements.

- **Exploring PyTorch Through Random Kernels**: Discussion revolved around the optimization of random kernels in PyTorch, showcasing the relevance of collaborative work on libraries such as Triton and their educational value as highlighted in a [conversation in the Triton channel](https://discord.com/channels/1189498204333543425/1189607595451895918/1210312045166198854).

- **Job Opportunity for NLP and ML Enthusiasts**: A new opportunity has arisen for an ML Engineer at SIXT in Munich, leaning towards candidates with NLP and Generative AI expertise. Prospective applicants can explore the [SIXT job listing](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472).

- **Challenges and Developments in Hardware-Accelerated Training**: Members discussed the compatibility issues of AMD GPUs with FA2 training, with a particular focus on the missing backward function/kernel for the 7900xtx. Possible solutions and ongoing work like the [flash-attention GitHub repository](https://github.com/ROCm/flash-attention/blob/b28f18350af92a68bec057875fd486f728c9f084/csrc/flash_attn_rocm/src/device_gemm_trait.hpp#L42) for better AMD GPU support were mentioned.

- **Ring Attention Draws Community Focus**: There was a flurry of activity around ring attention mechanisms, with multiple repository links provided for implementations and benchmarking. Engineers are collaborating to improve these libraries, such as [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch), and focusing on enhancements for usability and optimization.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Innovation Through Democratic Feedback**: A [research tool survey](https://forms.gle/8N4DsuCWtCXKxLSv6) has been circulated asking for community insights to improve functionalities like finding research papers and understanding complex studies.

- **LLM Enhancement Discussions**: Technical talk has revolved around optimizing **LangChain** agents, particularly through using **RunnableParallel** and **RunnablePassthrough** for improved parallel chain operations, and the integration of local models for streaming.

- **Seeking Langchain Expertise**: A community member is in search of a **Langchain** and **OpenAI's tool agent** consultant, offering compensation for guidance and expertise.

- **Debugging Tools Showcased**: The debugging and visualization capabilities of **LangSmith** were recommended for ensuring correct behavior in complex LangChain processes.

- **Explorations in Parallelism**: **Parallel function calls in LLMs** are now possible as revealed in a recent [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7166408137002962944/), expanding the technical toolkit for AI engineering applications.

- **Sharing AI-Enhanced Workflows**: Techniques for building custom chatbots with history capabilities, as well as using AI for stock portfolio summarization have been shared, powerfully demonstrating how **LLMs** can augment various business and development tasks.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Codespaces Template Boosts LLM Play**: `@derekpwillis` provided a [template repository](https://github.com/dwillis/llm-codespaces) ideal for running `orca-mini-3b` in codespaces, though there might be challenges with larger models. The template garnered positive feedback for its simplicity, though it has a noted long startup time due to on-the-fly compilation.
- **A Quirk in Codespaces Resolved**: `@simonw` detailed a workaround for an initial unavailability bug of `llm-gpt4all` in codespaces, recommending the command `llm chat -m orca-mini-3b-gguf2-q4_0` to preload the model for quicker subsequent usage.
- **Praising Prompt Craftsmanship**: `@tariqali` highlighted the nuanced benefits of traditional prompt crafting in LLMs compared to the straightforward queries now common with methods like RLHF. Traditional prompts may still hold value for specific goals like resuming chatbot conversations.
- **Large World Model's GPU Requirements**: `@simonw` showed interest in experimenting with the [Large World Model's LWM-Text-1M-Chat](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M) and discussed the necessity of a GPU instance for optimal performance due to the model's training on a substantial dataset.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **AI Hallucination Breakthrough Teased by Richard Socher**: Significant progress might have been made in addressing **AI hallucination**, as reflected in a [tweet by Richard Socher](https://x.com/RichardSocher/status/1760800655428796772?s=20) that showed error-free up-to-date references; the exact mechanism, speculated to involve state-of-the-art embeddings and a validator, was not detailed.
- **Globe Explorer's Innovations in Information Discovery**:
  - **Globe Explorer**, a tool described as a personalized Wikipedia powered by **GPT-4**, has been highlighted across discussions for symbolizing a new era in information retrieval. It was first introduced in a [tweet](https://x.com/sincethestudy/status/1761099508853944383?s=20) and further discussed in the community where it garnered viral attention even before promotional efforts.
- **Finetuning Strategies for GPT-4-Turbo Discussed**: A user with successful 1-shot data extraction from whole documents using **gpt-4-turbo** is weighing whether to include entire documents or just relevant sections in the finetuning dataset for more complex tasks.
- **Spatial Logic Prompting with LLMs Explored**: Discussion covered the challenge of writing prompts for organizing non-overlapping components in a grid, questioning the effectiveness of LLMs in spatial tasks without providing a conclusive strategy or results.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **GLAN - Next Big Thing?**: `@.benxh` surfaced a recent paper on **GLAN** (Generative Latent Nearest Neighbors), igniting a spark of interest among the community. The [paper in question](https://arxiv.org/pdf/2402.13064.pdf) was linked for those curious about this emerging technology.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1210130428321865738) (1038 messages🔥🔥🔥): 

- **Gemini Image Generator Bias Debate**: A significant discussion revolved around Google's Gemini 1.5 AI image generation model which was criticized for an inability to accurately depict white individuals or historic events, leading to its shutdown (`@coffeevampir3`, `@netrve`). Users debated whether this was due to internal biases or rushed implementation by Google (`@shanman6991`, `@netrve`), with references to a [video explaining the controversy](https://www.youtube.com/watch?v=Fr6Teh_ox-8) and several articles discussing the model (`@potatooff`).
  
- **AI-Assisted Creativity in Game Development**: Several users expressed interest in using various AI tools to generate or enhance game assets, a conversation that included methods like text to 3D (`@itsme9316`) and the potential for smaller game developers to use AI for artistic direction (`@alphaatlas1`).

- **Search Engine Market Share Puzzle**: The conversation shifted briefly to discuss why Google maintains a dominant search engine market share with suggestions for alternatives like Qwant (`@maldevide`) and critiques of Google's corporate ethos and direction (`@shanman6991`, `@selea8026`).

- **Control Vectors in AI**: `@rtyax` introduced the concept of control vectors for AI models, which was further expounded on with links to articles and research (`@selea8026`, `@rtyax`).

- **Summarization Models in AI Chat**: `@netrve` queried about good model options for summarizing chat messages within an AI platform, and discussed challenges with the current summarization pipeline within Streamlit's (ST) Transformer framework. `@itsme9316` suggested possibly using the same LLM used in ST or training a custom model.



**Links mentioned**:

- [Unexpected responses from ChatGPT](https://status.openai.com/incidents/ssg8fh7sfyz3): no description found
- [deepseek-ai/deepseek-moe-16b-chat · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat): no description found
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/): no description found
- [Paneer Paratha Recipe (Plain Layered &amp; Stuffed) - Swasthi&#039;s Recipes](https://www.indianhealthyrecipes.com/paneer-paratha/): Paneer paratha is a delicious flatbread made with paneer, wheat flour, spices and herbs. These are a great food for the entire family
- [LLM Explorer: A Curated Large Language Model Directory. LLM List. 18662 Open-Source Language Models.](https://llm.extractum.io/): Browse 18662 open-source large and small language models conveniently grouped into various categories and llm lists complete with benchmarks and analytics.
- [Gemini has a Diversity Problem](https://www.youtube.com/watch?v=Fr6Teh_ox-8): Google turned the anti-bias dial up to 11 on their new Gemini Pro model.References:https://developers.googleblog.com/2024/02/gemini-15-available-for-private-...
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [Cerebras CS-2 System Teardown](https://vimeo.com/853557623): A quick explainer video showing how Cerebras&#039; remarkable AI accelerator system is constructed. I did everything except camera, including building the props.
- [Discord | Your Place to Talk and Hang Out](https://discord.gg/YTYD3nX6)): Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
- [Scientists Claim AI Breakthrough to Generate Boundless Clean Fusion Energy](https://www.vice.com/en/article/y3w4am/scientists-claim-ai-breakthrough-to-generate-boundless-clean-fusion-energy): Princeton researchers report that a new AI model has solved one of the major roadblocks to generating fusion energy.
- [Spearman&#039;s rank correlation coefficient - Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient): no description found
- [Windows-as-a-nuisance: How I clean up a “clean install” of Windows 11 and Edge](https://arstechnica.com/gadgets/2024/02/what-i-do-to-clean-up-a-clean-install-of-windows-11-23h2-and-edge/): Tips and tricks for making Microsoft leave you alone while you use your PC.
- [Tyler Perry is so shocked by OpenAI&#x2019;s video generator Sora that he&#x2019;s pausing an $800 million studio expansion: &#x2018;A lot of jobs are going to be lost&#x2019;](https://finance.yahoo.com/news/tyler-perry-shocked-openai-video-173944787.html?guccounter=2): The movie mogul calls the AI text-to-video generator &#x22;shocking&#x22; and a &#x22;major game-changer&#x22; for TV and film workers.
- [Jurassic X Prix Finals Highlights | Extreme E | Jurassic X Prix](https://www.youtube.com/watch?v=4jkVymz8M1M): Subscribe for more Extreme E: https://bit.ly/3uj6v3zWhere to Watch Live: https://bit.ly/3ctoVbIWebsite: https://extreme-e.com Instagram: https://instagram.co...
- [American Broadcasting Cos., Inc. v. Aereo, Inc. - Wikipedia](https://en.wikipedia.org/wiki/American_Broadcasting_Cos.,_Inc._v._Aereo,_Inc.): no description found
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector): Note: Later made available as a preprint at Activation Addition: Steering Language Models Without Optimization.  Summary: We demonstrate a new scalable way of interacting with language models: adding ...
- [GitHub - amd/blis: BLAS-like Library Instantiation Software Framework](https://github.com/amd/blis): BLAS-like Library Instantiation Software Framework - amd/blis
- [240 tokens/s achieved by Groq's custom chips on Lama 2 Chat (70B)](https://old.reddit.com/r/LocalLLaMA/comments/1afm9af/240_tokenss_achieved_by_groqs_custom_chips_on/kog5l51/): Posted in r/LocalLLaMA by u/speakerknock • 238 points and 145 comments

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1210205654904737812) (438 messages🔥🔥🔥): 

- **Exploring Model Preferences and Performance**: Users are discussing their experiences and preferences among various models, including `Rogue-Rose-103b`, `miqumaid`, and `Miquella`. `@johnrobertsmith` indicated a preference for `miqu` and `@splice0001` for `Rogue-Rose-103b`, citing the writing style as a deciding factor.
  
- **Troubleshooting Model Behavior**: `@euchale` encountered issues with `EstopianMaid` acting out of character and received suggestions to check settings or character cards. After further discussion, it was determined the problem might be user-specific or related to the sequence of prompts.

- **Temperature Settings Influence on AI Models**: Users like `@splice0001` and `@dreamgen` are exchanging their experiences with temperature settings in AI models. `@dreamgen` suggested starting with a **temperature below 1** and recommended a setup with vLLM.

- **Character Card Complexity in Roleplay**: `@superking__` shares an interesting observation that giving a character the goal "survive at any cost" made it play its role more effectively in a roleplay scenario using Mixtral.

- **Opus V1 Model Guidance**: There’s been a focus on the newly published **Opus V1 models** for AI story-writing and role-playing, with `@dreamgen` publishing a guide and offering a **Colab script** for proper prompt formatting. `@splice0001` expressed positive feedback when using the model.

**Links mentioned**:

- [no title found](http://example.com)): no description found
- [dre (Kimjongeun)](https://huggingface.co/dre): no description found
- [Viralhog Grandpa GIF - Viralhog Grandpa Grandpa Kiki Dance - Discover &amp; Share GIFs](https://tenor.com/view/viralhog-grandpa-grandpa-kiki-dance-kiki-dance-dance-party-gif-12380914): Click to view the GIF
- [Angry Bender Mad GIF - Angry Bender Mad Angry - Discover &amp; Share GIFs](https://tenor.com/view/angry-bender-mad-angry-pissed-off-fist-gif-16261502): Click to view the GIF
- [LoneStriker/miqu-1-70b-sf-5.5bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/miqu-1-70b-sf-5.5bpw-h6-exl2?text=My+name+is+Merve+and+my+favorite): no description found
- [dreamgen/opus-v1-34b · Hugging Face](https://huggingface.co/dreamgen/opus-v1-34b): no description found
- [configs/lmstudio.json · dreamgen/opus-v1.2-7b at main](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/lmstudio.json): no description found
- [dreamgen/opus-v1-34b-awq · Hugging Face](https://huggingface.co/dreamgen/opus-v1-34b-awq): no description found
- [configs/opus-v1.py · dreamgen/opus-v1.2-7b at main](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/opus-v1.py): no description found
- [configs/opus-v1.py · dreamgen/opus-v1.2-7b at main](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/opus-v1.py#L163): no description found
- [Opus V1: Story-writing &amp; role-playing models - a dreamgen Collection](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31): no description found
- [DreamGen: AI role-play and story-writing without limits](https://dub.sh/opus-v1-guide): no description found
- [Models - Hugging Face](https://huggingface.co/models?search=LoneStriker/opus-v): no description found
- [Models - Hugging Face](https://huggingface.co/models?search=LoneStriker/opus-v1>): no description found
- [tokenizer : special token handling by staviq · Pull Request #3538 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3538): Special token handling, based on #1931, #3475 Works, but it&#39;s definitely not ready, just posting for feedback. Has some testing code meant to be removed before undrafting. Not optimized at all, ju...

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1210647502605258793) (4 messages): 

- **Seeking a DPO Implementation Guide**: `@cogbuji` is searching for a practical reference implementation of DPO to apply to MLX after finding the [Hugging Face alignment handbook](https://github.com/huggingface/alignment-handbook) unsatisfactory, due to a lack of implementation details beyond configuration files.
- **DPO Attempt Shared by Community Member**: Responding to `@cogbuji`, `@dirtytigerx` shared an unfinished attempt at implementing DPO, referring to the `DPOTrainer` in the `trl` library, which can be found at [huggingface/trl on GitHub](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py).
- **Extra Implementation Bits in the Mix**: `@dirtytigerx` mentions that the referenced `DPOTrainer` code includes not just DPO, but also KTO (knowledge transfer optimization) segments, which might not be directly relevant to `@cogbuji`'s needs.
- **cogbuji opts for TRL**: After the community input, `@cogbuji` decided to work with the `trl` module as a basis for implementing DPO.


**Links mentioned**:

[trl/trl/trainer/dpo_trainer.py at main · huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py): Train transformer language models with reinforcement learning. - huggingface/trl

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1210651801062215770) (25 messages🔥): 

- **In Quest for Model Hybridization**: `@jsarnecki` is considering a "frankenmerge" of **Orca-2-13b** with **Nous-Hermes-2-DPO-7b**, using Orca as the base and merging layer by layer to a 17B parameter model using [mergekit](https://github.com/arcee-ai/mergekit). However, `@maldevide` clarifies that such models are non-homologous and therefore not directly mergeable.
- **Mix-and-Match Model Merging Madness**: `@maldevide` suggests that, while direct merging is impossible, using datasets [fine-tuned on Hugging Face](https://huggingface.co/datasets/Open-Orca/OpenOrca) could be beneficial and references the complex merging techniques used in creating [SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0). They mention "SFT to clean things up" after a layered merge.
- **Homomorphic Hassles and Merging Methods**: `@alphaatlas1` and `@maldevide` discuss that for non-homologous merges like `@jsarnecki`'s project, serious issues arise with no established techniques for such merges, and recommend a homomorphic projection matrix with intensive training.
- **Curiosity Sparked by PEFT and Merge Approaches**: `@alphaatlas1` points to [a blog post](https://huggingface.co/blog/peft_merging) revealing PEFT's findings on model merges and notes DARE ties merging's adverse results on diffusion models, while it seems more suitable for LLMs according to tests in [meh on GitHub](https://github.com/s1dlx/meh).
- **Diffusion Model Merge Dilemmas**: The conversation shifts to the peculiar behavior of diffusion models with merging techniques, with `@jsarnecki` and `@alphaatlas1` noting the potential impact due to the models' density and alignment, while linear merges work well for models like SD (Stable Diffusion).

**Links mentioned**:

- [🤗 PEFT welcomes new merging methods](https://huggingface.co/blog/peft_merging): no description found
- [GitHub - s1dlx/meh: Merging Execution Helper](https://github.com/s1dlx/meh): Merging Execution Helper. Contribute to s1dlx/meh development by creating an account on GitHub.

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1210298626405044235) (8 messages🔥): 

- **Exploring Communities for MLX Enthusiasts**: `@fred.bliss` inquired about communities with a focus on machine learning and tinkering, aside from GitHub and Twitter. They expressed difficulty in finding such groups outside of those platforms.
- **Preference for Independence over Community**: `@dirtytigerx` mentioned that they do not generally seek out communities, which suggests a preference for working independently or using perhaps more established, less community-oriented platforms.
- **Dotpeek Shines for Spottyluck**: `@spottyluck` shared their use of JetBrains' Dotpeek, a .NET decompiler, mainly for **vulnerability research** rather than general programming tasks. They also added a humorous note about the abundance of poorly written system tray apps.
- **Curiosity about Dotpeek's Capabilities**: `@al_lansley` asked whether Dotpeek is limited to C# or if it has broader applications. Their message illustrates the importance of asking clarifying questions in a technical community, regardless of expertise level.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1210133216866865172) (462 messages🔥🔥🔥): 

- **Gemma Model Gossip**: Users report disappointing performance with the `Gemma` models. [@heyitsyorkie](https://lmstudio.ai) clarifies that lower than `Q8` quantizations of `Gemma` are broken in `llama.cpp`, which LM Studio uses.
- **No Picture Upload with LLava**: [@tvb1199](https://huggingface.co/) inquires about uploading images with `LLava` models. They are informed that vision capabilities require a model and a vision adapter (mmproj-model).
- **Larger Models Present Challenges**: [@wyrath](#c30) experiments with a `70b` model, finding it slow on CPUs and encountering difficulties with partial GPU offloading.
- **OLED Monitors Steal the Spotlight**: Various users praise the vivid display quality of OLED monitors, sharing their experiences and preferential shift away from traditional displays.
- **Phind-70B Curiosity**: [@pierrunoyt](https://www.phind.com/blog/introducing-phind-70b) asks about acquiring the Phind-70B model; [@heyitsyorkie](#c30) indicates it is exclusive to the Phind platform and is not available for local use.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs
- [Phind](https://www.phind.com/blog/introducing-phind-70b): no description found
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF): no description found
- [What is RAG? - Retrieval-Augmented Generation Explained - AWS](https://aws.amazon.com/what-is/retrieval-augmented-generation/): no description found
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3): Announcing Stable Diffusion 3 in early preview, our most capable text-to-image model with greatly improved performance in multi-subject prompts, image quality, and spelling abilities.
- [Phind-70B: BEST Coding LLM Outperforming GPT-4 Turbo + Opensource!](https://www.youtube.com/watch?v=v0ZN_MKYfhw): In this video, we unveil the revolutionary capabilities of Phind-70B, designed to close the code quality gap and accelerate your coding process. With up to 8...
- [Google’s NEW Open-Source Model Is SHOCKINGLY BAD](https://www.youtube.com/watch?v=1Mn0U6HGLeg): Sorry for the title. I couldn&#39;t help myself. I&#39;m proud of Google for releasing a completely open-source model to the world, but it&#39;s not good. How bad is it?...
- [dreamgen/opus-v1.2-7b-gguf · Hugging Face](https://huggingface.co/dreamgen/opus-v1.2-7b-gguf): no description found
- [dreamgen/opus-v1.2-7b · Hugging Face](https://huggingface.co/dreamgen/opus-v1.2-7b): no description found
- [configs/lmstudio.json · dreamgen/opus-v1.2-7b at main](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/lmstudio.json): no description found
- [W3C standards and drafts](https://www.w3.org/TR/?filter-tr-name=scroll): The World Wide Web Consortium (W3C) is an international community where Member organizations, a full-time staff, and the public work together to develop Web standards.
- [Designing for Web Accessibility – Tips for Getting Started](https://www.w3.org/WAI/tips/designing/): Summary
- [Web Standards](https://www.w3.org/standards/): This page introduces web standards at a high-level.
- [W3C Accessibility Standards Overview](https://www.w3.org/WAI/standards-guidelines/): Accessibility resources free online from the international standards organization: W3C Web Accessibility Initiative (WAI).

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1210136032607997973) (76 messages🔥🔥): 

- **Seeking AI-Assisted Teaching Tools**: `@therapienachdemtod` is designing an assistant to aid in teaching, looking for a model that prepares educational content and interacts with students by correcting grammar and engaging in dialogue. In response, `@thebest6337` expressed skepticism about the effectiveness of current models for such tasks, mentioning possible shortcomings and no experience with the model "gemma."
- **Gemma Model Quirks Revealed**: `@thorax7835` discussed the limitations of "mixtral" when asking for fitness tips, as it tends to censor itself, and `@nullt3r` confirmed experiencing odd behavior from "LMStudio gemma 2b model."
- **No Internet for Local Models**: `@heyitsyorkie` clarified that local models in LM Studio, like "Gemma," do not have internet access, in response to queries by `@thorax7835` about model improvements and internet capabilities.
- **Stable Diffusion Web UI Recommended**: In a discussion about image generation capabilities, `@heyitsyorkie` and `@drawingthesun` recommended using the Automatic1111's Stable Diffusion web UI for those tasks, as LLM Studio does not support them.
- **Error Troubleshooting in LM Studio**: `@macaulj` sought help with an error they were encountering with LM Studio and received advice from `@heyitsyorkie` hinting at a potential graphics card driver issue related to CUDA.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [Master Generative AI Stack: practical handbook](https://medium.com/@Naykafication/master-modern-generative-ai-stack-practical-handbook-393f446a706c?sk=731eb4d03418970b47143d1818f8c492): Yet another AI article. It might be overwhelming at times. In this comprehensive guide, I’ll simplify the complex world of Generative AI…
- [Big Code Models Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard): no description found
- [wavymulder/Analog-Diffusion · Hugging Face](https://huggingface.co/wavymulder/Analog-Diffusion): no description found
- [Models - Hugging Face](https://huggingface.co/models?search=fitness): no description found
- [macaulj@macaulj-HP-Pavilion-Gaming-Laptop-15-cx0xxx:~$ sudo &#039;/home/macaulj/Downl - Pastebin.com](https://pastebin.com/MVZmiH2Y): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [GitHub - ParisNeo/lollms-webui: Lord of Large Language Models Web User Interface](https://github.com/ParisNeo/lollms-webui): Lord of Large Language Models Web User Interface. Contribute to ParisNeo/lollms-webui development by creating an account on GitHub.
- [GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui): Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.

  

---


### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1210287136671010816) (1 messages): 

- **Urgent Update to LM Studio v0.2.16**: `@yagilb` announces that **LM Studio v0.2.16** is now available and urges users to update from v0.2.15. This update includes all features of v0.2.15 plus *important bug fixes* for erratic regenerations and erratic scrolls in chats during downloads.
  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1210172110606176298) (26 messages🔥): 

- **Gemma 7B Download Confusion Cleared**: User `@heyitsyorkie` explained to `@adtigerning` that the Gemma 7B file from [Hugging Face](https://huggingface.co/) must be downloaded manually and placed in the My Models folder for compatibility. The issue was related to access on LM Studio and Hugging Face repositories.

- **LM Studio Update to v0.2.16 Released**: `@yagilb` informed users, including `@drawingthesun` and `@heyitsyorkie`, that the scrolling bug they experienced has been fixed in the new update, version 0.2.16, which users were encouraged to download from [LM Studio](https://lmstudio.ai) or through the app's update feature.

- **Community Feedback on LM Studio v0.2.16**: `@bananatechindustries` expressed enthusiasm for the new user interface in update v0.2.16, particularly appreciating the ability to see model readmes in the search. Meanwhile, `@heyitsyorkie` confirmed that previous bugs appear to be resolved with this update.

- **Mixed Reactions to UI and Compatibility**: User `@clickclack777` critiqued the use of Comic Sans and complex UI in LM Studio v0.2.16, suggesting it added unnecessary complexity. `@woteva` raised issues with UI scalability and model folder compatibility, citing problems with screen size and incorrect RAM requirements messages.

- **New Update Receives Praise**: `@macfly` shared their positive impression of the LM Studio update's look and feel, emphasizing it with an animated fire emoji.

**Links mentioned**:

[👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1210164001020059699) (46 messages🔥): 

- **Bargain on Vintage GPUs**: `@freethepublicdebt` mentioned having spare **TESLA K40** cards for sale, indicating they have excellent VRAM/$ but limited to **CUDA 3.5**. There was a mention of interest in adapting *llama.cpp* for these cards for cheap datacenter card usage, but skepticism remains due to their age.
  
- **More GPUs, More Speed?**: `@apnea2014` asked about the benefits of adding a second GPU for inference with LM Studio, to which `@heyitsyorkie` indicated that more VRAM equals more speed, and combining two cards of the same generation can yield better results.

- **Future Competition in High VRAM GPUs**: `@nink1` shared optimism about **AMD** potentially challenging **Nvidia** with their latest earnings report surge and potential for high VRAM GPUs. `@christianazinn` and `@ptable` debated about the consumer market focus of AMD, noting the popularity of Nvidia's 4090 cards for AI applications.

- **AMD’s Enterprise Push**: Contributions from `@exio4` emphasized that while consumer Nvidia GPUs still exceed AMD's matrix throughput, AMD's latest chips like the **MI300X** might disrupt Nvidia's enterprise AI dominance with superior memory and bandwidth specs, as discussed in a [TechWireAsia article](https://techwireasia.com/12/2023/can-amd-mi300-chips-really-challenge-nvidia-ai-dominance/). `@nink1` posited AMD's potential growth in embedded AI markets, despite current CUDA compatibility issues.

- **Consumer GPU Discussions for LLMs**: Participants such as `@barduk`, `@wolfspyre`, and `@heyitsyorkie` discussed whether AMD cards like the Radeon RX 7800 XT Core Edition are suitable for running LLM models compared to Nvidia's offerings. The consensus appears to be that while AMD cards can be used, Nvidia cards are recommended for their ease of setup and broader compatibility with AI frameworks.

**Links mentioned**:

- [AMD launches MI300 chips: a challenger to Nvidia&#039;s AI dominance?](https://techwireasia.com/12/2023/can-amd-mi300-chips-really-challenge-nvidia-ai-dominance/): The latest AMD AI chips boasts over 150 billion transistors, 2.4 times the memory of Nvidia&#039;s leading H100.
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.
- [OpenCL - The Open Standard for Parallel Programming of Heterogeneous Systems](https://www.khronos.org/opencl/): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1210164712030085150) (34 messages🔥): 

- **Model Performance Report**: `@drawless111` mentions testing **Gemma 2B IT** and **7B IT** (non-supersized versions) on LM Studio version 0.2.15, indicating they perform impressively.
- **Specs Question Answered**: `@heyitsyorkie` confirms that even a system with 15 11 gen and 8 GB RAM can run **Q4_K_M** on LM Studio v0.2.15.
- **Gemma Model Struggles**: Users like `@ascrowflies` are reporting quality issues with *Lonestriker's 7B IT* quant, while `@heyitsyorkie` acknowledges it's the best available until `llama.cpp` is fixed.
- **Gemma Model Compatibility**: `@yagilb` recommends a [Gemma 2B model on Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) which resolves some issues users (`@issaminu` and `@rumpelstilforeskin`) are experiencing with the model.
- **Excitement for IQ Series Models**: `@drawless111` celebrates the successful implementation of **IQ1, IQ2, and IQ3** on LM Studio, with specific stats on performance provided for IQ1.

**Links mentioned**:

[lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF?): no description found

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1210596239087894578) (8 messages🔥): 

- **AutoGen Issues Resolved**: User `@thebest6337` encountered a ***weird problem*** with AutoGen but later **fixed the issue** by uninstalling and reinstalling all AutoGen Python packages.
- **Sharing the Solution Encouraged**: `@heyitsyorkie` suggested that sharing the fix could help others with similar issues.
- **The Classic IT Fix**: `@heyitsyorkie` humorously linked to a [Tenor GIF](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069) portraying the quintessential IT advice: "Have you tried turning it off and on again?"

**Links mentioned**:

[It Problem Phone Call GIF - It Problem Phone Call Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069): Click to view the GIF

  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1210317935202144286) (1 messages): 

- **Chunk Size Matters**: User `@simas93` discussed how the preprocessing of text for embeddings is influenced by the model's embeddings, specifically indicating that `chunk_size` should depend on the model in use. They shared a [good read on AI Stack Exchange](https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size) detailing a rule of thumb for determining embedding size and proposed a specific formula for when `num_categories <= 1000`, suggesting to set `num_embeddings` to `min(500, num_categories/2)`.

**Links mentioned**:

[How to determine the embedding size?](https://ai.stackexchange.com/questions/28564/how-to-determine-the-embedding-size)): When we are training a neural network, we are going to determine the embedding size to convert the categorical (in NLP, for instance) or continuous (in computer vision or voice) information to hidden 

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1210167903085920336) (69 messages🔥🔥): 

- **ChatGPT Training with HTML and CSS**: User `@ls_chicha` asked if it is possible to train ChatGPT with HTML and CSS files, looking for insights on incorporating coding languages into AI education.
- **GPTs Reading PDF Issues**: `@arani1977` encountered problems with GPTs that initially could read PDFs but then claimed they lost the ability, seeking an understanding of this inconsistency despite unaltered configuration settings.
- **Seeking Chat Client Recommendations for OpenAI API**: User `@oleksandrshr` inquired about chat client suggestions for the OpenAI API and further expressed concerns about the slow performance of models such as Ollama, Mistral, Phi, and Gemma:2b on Ollama.
- **Understanding "Quantized Version" in AI**: In response to `@oleksandrshr`'s question about quantized versions, `@darthgustav.` explained that such versions speed up a model by rounding the weights, which simplifies calculations but reduces precision and performance.
- **Concerns Over GPT-4's Potency Rumors**: User `@zaatuloa` brought up rumors that GPT-4 may have been powered down since its release, which was quickly debunked by user `@lugui`, who asserted that these claims are false.

**Links mentioned**:

- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3): Announcing Stable Diffusion 3 in early preview, our most capable text-to-image model with greatly improved performance in multi-subject prompts, image quality, and spelling abilities.
- [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators): We explore large-scale training of generative models on video data. Specifically, we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ...
- [Pretty Much Everywhere Steve Kornacki GIF - Pretty Much Everywhere Steve Kornacki Msnbc - Discover &amp; Share GIFs](https://tenor.com/view/pretty-much-everywhere-steve-kornacki-msnbc-all-over-the-place-all-around-gif-19744447): Click to view the GIF
- [Gemini has a Diversity Problem](https://www.youtube.com/watch?v=Fr6Teh_ox-8): Google turned the anti-bias dial up to 11 on their new Gemini Pro model.References:https://developers.googleblog.com/2024/02/gemini-15-available-for-private-...

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1210134092641734686) (67 messages🔥🔥): 

- **Qdrant and OpenAI Embeddings Query Confusion**: `@thirawat_z` shared frustrations about discrepancies in search results when using OpenAI embeddings with Qdrant compared to a tutorial, with their results being unrelated to their "modern art in Europe" query. They provided code snippets and results from both the tutorial and their own attempt for comparison.

- **Training ChatGPT with HTML and CSS**: Users `@ls_chicha`, `_jonpo`, and `@thedreamakeem` discussed the possibility of training ChatGPT with HTML and CSS files. `@thedreamakeem` mentioned that a .json database format might be required.

- **Creating AI Conversations**: `@link12313` proposed an app for GPT-4 to converse with Google’s Gemini Ultra1.5, with `@toror` commenting that a good starting point is required for engaging dialogue.

- **GPT-4 Input Prompt Inflation Issue**: `@cetacean_xx` reported an issue where input prompts with GPT-4 ballooned to over 30,000 tokens, with `@darthgustav.` suggesting it’s due to context history accumulation and recommending to remove if unnecessary.

- **ChatGPT-4 Performance and Context Limitations**: `@orbart` expressed dissatisfaction with ChatGPT-4 due to perceived nerfs affecting usage and memory capabilities, prompting a discussion on context length and token limits with `@paccer`. `@blckreaper` contributed observations that the model's available context from files may have been reduced.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1210208821356527696) (202 messages🔥🔥): 

- **Looping Logic with ReAct Prompting**: `@tawsif2781` described an issue with their chatbot agent getting stuck in a loop when using React prompting, outputting the same thought repeatedly. `@darthgustav.` suggested that this might be due to contextual inconsistencies or too much content causing retrieval issues from the model's middle context.

- **Improvisation at Zero Temperature**: In a discussion about generating independent thoughts at zero temperature, `@darthgustav.` clarified that even at zero temperature, models can follow an instruction like "improvise" and produce varied results if timestamps or slight context differences are included.

- **Avoid Negative Instructions for LLMs**: Prompt crafting advice shared by `@darthgustav.` emphasized avoiding negative instructions, as they might translate into affirmative actions due to logic gaps in transformer AI. There was also a suggestion to use redundancy by reframing instructions in the prompt for better compliance from the model.

- **Resources for Prompt Engineering**: Various users shared advice and resources for learning prompt engineering; `@darthgustav.` recommended Arxiv and Hugging Face, while `@bambooshoots` provided a direct link to OpenAI's prompt engineering guide, and `@openheroes` mentioned the usefulness of custom instructions features.

- **Custom Instructions (CI) Concerns and Usage**: Users `@jimmysapp` and `@eskcanta` discussed issues and solutions related to the usage and content policy compliance of custom instructions. `@eskcanta` provided detailed advice on effectively using CIs for roleplay and summarized conversations by incorporating consistent summaries within the conversations.

**Links mentioned**:

- [Usage policies](https://openai.com/policies/usage-policies): no description found
- [Terms of use](https://openai.com/policies/terms-of-use): no description found

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1210208821356527696) (202 messages🔥🔥): 

- **Breaking the ReAct Loop**: User `@tawsif2781` reported an issue with a ReAct prompting loop, receiving continuous identical outputs. Various techniques such as avoiding middle-context and redundant prompting, and managing temperature settings were discussed by `@darthgustav.` to troubleshoot this repetitive behavior.

- **Looping and Improvisation at Zero Temps**: `@darthgustav.` clarified that even at zero temperature, the model can improvise based on the provided context. Model behavior was explored, highlighting how factors like timestamps can influence variances in output even with a consistent prompt.

- **Graphs of Thoughts and Character Consistency**: Users engaged in a discussion about how "graph of thoughts" functions and whether it perpetuates bias. `@eskcanta` and others shared insights into maintaining character consistency and role-plays using Custom Instructions (CI) on ChatGPT.

- **Sustained AI Interactions with Roleplay Scenarios**: Through a conversation with `@cqoker`, `@eskcanta` showcased how the model can be instructed for complex interactions such as roleplay, providing examples and strategies to save and switch between different character descriptions or scenarios.

- **Concerns and Ethical Implications of AI**: `@cqoker` and `@eskcanta` reflected on the ethical concerns regarding AI-generated content and its realistic portrayal, discussing the importance of using the technology responsibly and adhering to OpenAI's usage policies.

**Links mentioned**:

- [Usage policies](https://openai.com/policies/usage-policies): no description found
- [Terms of use](https://openai.com/policies/terms-of-use): no description found

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1210147601605992488) (398 messages🔥🔥): 

- **Worker Orchestration Discussion**: `@top_walk_town` was curious about the framework used for worker orchestration. `@pseudoterminalx` revealed that they personally created the orchestration system, describing it as "crappy scripts and a database."

- **Stable Diffusion 3 Anticipation**: `@thejonasbrothers` provided insights into the upcoming Stable Diffusion 3, hypothesizing that it might utilize a similar approach to what they've been working on for months: a base for medium resolution and a flow matching upscaler. There's skepticism about the potential lack of diversity in image generation, with `@pseudoterminalx` indicating that the images already seem to lack diversity.

- **Stable AI's Employee Hiring Trends**: `@thejonasbrothers` and `@pseudoterminalx` discussed the hiring practices at Stability AI, suggesting there's preference towards hiring systems administrators who are transitioning into machine learning roles due to affordability. There's also mention of a trend of hiring individuals with YouTube followings.

- **Concerns Over Closed Model Developments**: The LAION community expressed concerns regarding the trend of companies like Stability AI moving model development further behind closed doors, away from the end-user's reach. `@thejonasbrothers` reminisced about how earlier models like LDM/SD1 had more publicly involved code and compute use.

- **Future of Fine-tuning and Open-source Models**: The discussion touched upon the profitability of open-source models and the advantages of finetuning them. `@helium__` shared a link about LoRA Land, an initiative that fine-tunes Mistral-7b models to potentially outperform GPT-4, with specialized versions for various tasks.

**Links mentioned**:

- [SDXL Lightning - by fal.ai](https://fastsdxl.ai/): Lightning fast SDXL API demo by fal.ai
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3): Announcing Stable Diffusion 3 in early preview, our most capable text-to-image model with greatly improved performance in multi-subject prompts, image quality, and spelling abilities.
- [Funny Silly GIF - Funny Silly Something Is Off - Discover &amp; Share GIFs](https://tenor.com/We7R.gif): Click to view the GIF
- [Jasper Expands by Acquiring Image Platform Clipdrop from Stability AI](https://www.jasper.ai/blog/jasper-acquires-clipdrop): Jasper enters the European market through acquisition, joins vibrant AI community in Paris
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 - Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4): LoRA Land is a collection of 25+ fine-tuned Mistral-7b models that outperform GPT-4 in task-specific applications. This collection of fine-tuned OSS models offers a blueprint for teams seeking to effi...
- [Safety Review for LAION 5B | LAION](https://laion.ai/notes/laion-maintanence/): &lt;p&gt;There have been reports in the press about the results of a research project at Stanford University, according to which the LAION training set 5B contains...
- [cc2dataset/cc2dataset/main.py at main · rom1504/cc2dataset](https://github.com/rom1504/cc2dataset/blob/main/cc2dataset/main.py#L83>): Easily convert common crawl to a dataset of caption and document. Image/text Audio/text Video/text, ... - rom1504/cc2dataset
- [WebVid 大型短视频数据集 / 数据集 / 超神经](https://hyper.ai/datasets/17289): no description found
- [Snap Video](https://snap-research.github.io/snapvideo/#title-footer): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1210148610214199337) (73 messages🔥🔥): 

- **The Retirement Debate for LAION 5B**: `@top_walk_town` pondered if **LAION 5B** should be retired due to issues like link rot and data poisoning, suggesting a community effort to create new datasets with high-quality images and annotations.
- **Community Effort for Captioning**: A "mob captioning" effort using **cogvlm** was flagged by `@twoabove`, suggesting ongoing initiatives in the community to improve datasets and annotation quality.
- **Model Training in Mixed Precision**: In a discussion on training with mixed precision, `@yoavhacohen` confirmed the effectiveness of using autocast with **bfloat16** on TPUs, while `@top_walk_town` pointed out the use of autocast and gradient scaling to address the underflow in gradients.
- **Instruct Pix2Pix State-of-the-Art**: `@twoabove` shared a [link](https://arxiv.org/abs/2402.14289) to a research paper detailing the TinyLLaVA framework, which discusses data quality, training recipes, and how smaller multimodal models compare to larger ones.
- **LoRA Receives a Humorous Examination**: `@thejonasbrothers` shared a [link](https://intrinsic-lora.github.io/) to a paper dubbed *Generative Models: What do they know? Do they know things? Let's find out!*, which uses INTRINSIC LoRA to highlight the hidden capabilities of generative models without additional layers.

**Links mentioned**:

- [OpenAI acquires Global Illumination](https://openai.com/blog/openai-acquires-global-illumination): The entire team has joined OpenAI.
- [Generative Models: What do they know?](https://intrinsic-lora.github.io/): no description found
- [Our structure](https://openai.com/our-structure): We designed OpenAI’s structure—a partnership between our original Nonprofit and a new capped profit arm—as a chassis for OpenAI’s mission: to build artificial general intelligence (AGI) that is safe a...
- [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083): While Transformers have enabled tremendous progress in various application settings, such architectures still lag behind traditional symbolic planners for solving complex decision making tasks. In thi...
- [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](https://arxiv.org/abs/2402.13929v1): We propose a diffusion distillation method that achieves new state-of-the-art in one-step/few-step 1024px text-to-image generation based on SDXL. Our method combines progressive and adversarial distil...
- [TinyLLaVA: A Framework of Small-scale Large Multimodal Models](https://arxiv.org/abs/2402.14289): We present the TinyLLaVA framework that provides a unified perspective in designing and analyzing the small-scale Large Multimodal Models (LMMs). We empirically study the effects of different vision e...
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 - Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4): LoRA Land is a collection of 25+ fine-tuned Mistral-7b models that outperform GPT-4 in task-specific applications. This collection of fine-tuned OSS models offers a blueprint for teams seeking to effi...

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1210143898249338910) (3 messages): 

- **Bubbles Galore**: `@harrisonv` posted a series of bubble emojis with no further context provided.
- **Enigmatic Mention**: `@harrisonv` tagged a user with the ID `<@644428303293349888>` but did not follow up with any additional text or context.
- **Rwkv Commentary**: `@vatsadev` responded to `@harrisonv`'s tagging of the user with a cryptic comment stating, *Rwkv goes brrr here*.
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1210223732908494898) (13 messages🔥): 

- **Spotlight on Open Source SOTA Model - Gemma**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=953U3FxHF-Q) titled "Gemma Google's open source SOTA model." The video introduces Gemma, a lightweight, state-of-the-art family of open models derived from the research behind the Gemini models.
- **Seeking AI Marketing Experts**: `@danieltkilleen` inquired about knowing any key opinion leaders (KOLs) in the AI marketing space, looking for recommendations.
- **Ski Bi Di Recognition**: `@teknium` gave a shoutout to `<@687315767208706059>`, acknowledging their expertise in skibidis and related knowledge.
- **Discussing a Zoomer-Driven LLM**: `@n8programs` pondered over the idea of training a Zoomer language model, triggering a light-hearted debate on generational work ethics with comments like *"...we are the generation born of the grind... ~~and aderall~~."*
- **Zoomers' Love for Work**: In a brief exchange, `@everyoneisgross` contested the notion that work is valuable, to which `@hexani` responded supporting `@n8programs`' perception with a one-word agreement: "Factuals."

**Links mentioned**:

[Gemma Google&#39;s open source SOTA model](https://www.youtube.com/watch?v=953U3FxHF-Q): Gemma is a family of lightweight, state-of-the-art open models built from the same research and technology used to create the Gemini models. Developed by Goo...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1210138621462454314) (18 messages🔥): 

- **OpenOrca Confirmed**: Users `@sherlockzoozoo` and `@teknium` discussed what `oo/oo2` refers to, with `@.benxh` confirming that it is indeed **Open Orca**.
- **JARVIS Connects LLMs to ML Community**: `@leonidasch2` shared [two GitHub links](https://github.com/microsoft/JARVIS) to repositories under Microsoft's **JARVIS** project, which aims to connect Large Language Models with the machine learning community, and suggested checking them out for function calling applications.
- **New Diffusion Transformer Revealed**: User `@0xevil` linked to a tweet from `@EMostaque`, discussing a new diffusion transformer similar to **Sora** that includes flow matching and other improvements. Details on multimodal inputs and transformer improvements were promised to be shared soon.
- **Challenging the Adequacy of Human Feedback**: `@pramod8481` shared an [Arxiv link](https://arxiv.org/abs/2309.16349) highlighting critical analysis on the use of human feedback for training and evaluating Large Language Models, emphasizing that preference scores may under-represent crucial aspects such as factuality.
- **Investigating Value Bias in LLMs**: A study highlighted by `@pramod8481` suggests that LLMs favor high-value options due to an implicit value function within, based on research from an [Arxiv paper](https://arxiv.org/abs/2402.11005). The study raises concerns about value bias in LLM responses.

**Links mentioned**:

- [Exploring Value Biases: How LLMs Deviate Towards the Ideal](https://arxiv.org/abs/2402.11005): Large-Language-Models (LLMs) are deployed in a wide range of applications, and their response has an increasing social impact. Understanding the non-deliberate(ive) mechanism of LLMs in giving respons...
- [TencentARC/Mistral_Pro_8B_v0.1 · Hugging Face](https://huggingface.co/TencentARC/Mistral_Pro_8B_v0.1): no description found
- [Human Feedback is not Gold Standard](https://arxiv.org/abs/2309.16349): Human feedback has become the de facto standard for evaluating the performance of Large Language Models, and is increasingly being used as a training objective. However, it is not clear which properti...
- [Tweet from Emad (@EMostaque)](https://x.com/EMostaque/status/1760660709308846135?s=20): @StabilityAI Some notes: - This uses a new type of diffusion transformer (similar to Sora) combined with flow matching and other improvements.  - This takes advantage of transformer improvements & can...
- [Bio-inspired Structure Identification in Language Embeddings](https://arxiv.org/abs/2009.02459): Word embeddings are a popular way to improve downstream performances in contemporary language modeling. However, the underlying geometric structure of the embedding space is not well understood. We pr...
- [JARVIS/taskbench at main · microsoft/JARVIS](https://github.com/microsoft/JARVIS/tree/main/taskbench): JARVIS, a system to connect LLMs with ML community. Paper: https://arxiv.org/pdf/2303.17580.pdf - microsoft/JARVIS
- [JARVIS/easytool at main · microsoft/JARVIS](https://github.com/microsoft/JARVIS/tree/main/easytool): JARVIS, a system to connect LLMs with ML community. Paper: https://arxiv.org/pdf/2303.17580.pdf - microsoft/JARVIS

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1210134881330528266) (345 messages🔥🔥): 

- **Gemma 7B Under the Microscope**: Multiple users, including `@interstellarninja`, `@teknium`, and `@gryphepadar`, shared their experiences with finetuning the Gemma 7B model. They discussed issues with loss initially starting high and ways to mitigate this, such as not adding tokens during finetuning and the end results still being less effective than desired.

- **Fine-Tuned Tinyllama Showcases Capability**: User `@alvion427` praised `@n8programs`'s fine-tuned Tinyllama model for its ability to conduct multi-turn conversations. `@n8programs` discussed using the model to produce content more efficiently.

- **OpenCodeInterpreter Sparks Interest**: Shared by `@weyaxi`, OpenCodeInterpreter integrates code generation with execution and refinement, trained on a large multi-turn interaction dataset. `@.benxh` and `@teknium` engaged in the discussion, touching on related datasets and their availability.

- **Using LLMs for Scoring and Classifications**: Users, including `@night_w0lf` and `@leontello`, examined the use of numerical scales and classification labels in giving LLMs scoring tasks. They concurred that defining scores and using classification labels yields better results.

- **LLM Fine-Tuning for Constrained Outputs**: `@cf0913` and `@mihai4256` discussed strategies for fine-tuning large language models (LLMs) for more constrained and reliable outputs such as JSON. `@teknium` and `@.interstellarninja` mentioned their ongoing work which includes structured finetuning to achieve a more predictable result.

**Links mentioned**:

- [Phind](https://www.phind.com/blog/introducing-phind-70b): no description found
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [Tweet from TokenBender (e/xperiments) (@4evaBehindSOTA)](https://fxtwitter.com/4evaBehindSOTA/status/1760512560238109167?s=20): based on my tests so far, ignoring Gemma for general purpose fine-tuning or inference.  however, indic language exploration and specific use case tests may be explored later on.  now back to building ...
- [Tweet from Xiang Yue (@xiangyue96)](https://fxtwitter.com/xiangyue96/status/1760891516107862104): 🌟With precise execution & human feedback, a 7B code model hits 90% accuracy on HumanEval! 🚀 Introducing OpenCodeInterpreter: A family of open-source code systems for generating, executing, & refinin...
- [google/gemma-7b at main](https://huggingface.co/google/gemma-7b/tree/main/examples): no description found
- [PixArt-alpha/PixArt-XL-2-1024-MS · Hugging Face](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS): no description found
- [Tweet from anton (@abacaj)](https://fxtwitter.com/abacaj/status/1760393505153679369?s=20): After trying Gemma for a few hours I can say it won’t replace my mistral 7B models. It’s better than llama 2 but surprisingly not better than mistral. The mistral team really cooked up a model even go...
- [[Regression] Yi 200K models won&#39;t load in latest release · Issue #29252 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29252): System Info transformers version: 4.38.1 Platform: Linux-5.4.0-167-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Huggingface_hub version: 0.20.3 Safetensors version: 0.4.2 Accelerate versio...
- [llama2.c/export.py at master · karpathy/llama2.c](https://github.com/karpathy/llama2.c/blob/master/export.py#L556): Inference Llama 2 in one file of pure C. Contribute to karpathy/llama2.c development by creating an account on GitHub.
- [GitHub - jxnl/instructor: structured outputs for llms](https://github.com/jxnl/instructor): structured outputs for llms . Contribute to jxnl/instructor development by creating an account on GitHub.
- [m-a-p/Code-Feedback · Datasets at Hugging Face](https://huggingface.co/datasets/m-a-p/Code-Feedback): no description found
- [LeonEricsson - Overview](https://github.com/LeonEricsson): Research Engineer | M.Sc ML, B.Sc CS . LeonEricsson has 22 repositories available. Follow their code on GitHub.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1210258605446529074) (18 messages🔥): 

- **Huggingface Error Corrected**: `@qtnx` acknowledged that a typo in the Nous-Hermes-2-Mistral-7B-DPO model name on Huggingface (mixtral -> mistral) has been corrected. Model functionality remains the same.

- **Gemma 7B Finetuning Findings Shared**: `@stoicbatman` shared results from finetuning the Gemma 7B model, indicating that a learning rate of 5e-5 yielded the best results for their experiments but did not see significant accuracy improvements.

- **Voracious VRAM Usage Noted**: `@gryphepadar` added their observation, noting that finetuning models consumes a significant amount of VRAM compared to Mistral models, which could be a factor for computational resource planning.

- **Call for Large-Scale Experiment Scripts**: `@stoicbatman` inquired about shell scripts for conducting large-scale model finetuning and evaluation experiments. `@teknium` responded by providing a link to a related [GitHub project](https://github.com/AblateIt/finetune-study) and mentioned that the initial project did not succeed, but the repository may still offer valuable insights.

- **Adjustments for Fine-Tuning and Evaluation**: In a follow-up, `@teknium` suggested that the provided GitHub script would require significant updates to meet `@stoicbatman`'s experiment requirements, as it was designed to save, upload, and evaluate models for each epoch.

**Links mentioned**:

- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [GitHub - AblateIt/finetune-study: Comprehensive analysis of difference in performance of QLora, Lora, and Full Finetunes.](https://github.com/AblateIt/finetune-study): Comprehensive analysis of difference in performance of QLora, Lora, and Full Finetunes.  - GitHub - AblateIt/finetune-study: Comprehensive analysis of difference in performance of QLora, Lora, and ...

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1210130476782723102) (311 messages🔥🔥): 

- **Model Speculations and Benchmarks**: Community members like `@shivakiran_`, `@sublimatorniq`, and others shared thoughts on the potential size and performance of **Mistral-next**, with some users suggesting it's a **larger model** than Mixtral based on its lower serving speed. Users like `@egalitaristen` and `@mrdragonfox` mentioned testing Mistral-next on lmsys, praising its capabilities in areas like mathematics, even though the specific model size remains unknown.

- **Gemma's Potential and Mistral Improvements**: `@i_am_dom` suggests that **Gemma** could be an open-source base for tiny models and hints that Mistral could improve their 7b model by rebasing from Llama2 to Gemma. Further discussions included assumptions about data recency and knowledge cutoffs.

- **Next Model Analysis**: Users such as `@gunterson` and `_._pandora_._` speculated whether Mistral-next could be an improvement or a final version of MiQu, while others like `@ethux` discussed the current limitations of Apple hardware running Mixtral due to FP16 issues. There's a general interest in the capabilities and internal details of Mistral-next, but exact details like the number of parameters are not disclosed.

- **Usage Directions and Model Access**: Inquiries about **using Mistral models locally without software** like Ollama or LM studio were addressed by `@egalitaristen`, who explained that running the code is possible with guidance from model card examples on Hugging Face. `@ethux` also discussed hardware specifics and the availability of models like Mistral-next, which is currently only available at `https://chat.lmsys.org`.

- **Open Source Concerns and Ambition**: Discussions highlighted a community concern that **Mistral** might stop open-sourcing their models, although it's mentioned that there's no clear indication of this move. Users like `@casper_ai` and `@egalitaristen` shared a belief that Mistral's commitment to open-source remains due to a stated philosophy resembling Linux's development and how it benefits safety and model improvements.

**Links mentioned**:

- [Chat with Open Large Language Models](https://chat.lmsys.org/): no description found
- [Chat with Open Large Language Models](https://chat.lmsys.org): no description found
- [ETHUX Chat](https://chat.ethux.net): Made possible by PlanetNode with ❤️
- [GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.
- [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference): Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1210168578402156567) (15 messages🔥): 

- **Mistral-next's existence confirmed**: `@ethux` confirmed that **Mistral-next** is a real development, seeming to outperform *Mistral-Medium*.
- **No API Access for Mistral-next Yet**: `@ethux` mentioned that **API access** for Mistral-next is not currently available but suggests that details about access **will be released soon**.
- **Mistral versus OpenAI**: `@paul16307` humorously notes that Mistral might be a better version of OpenAI, jokingly adding "**but French**" which prompted `_._pandora_._` to comment on Mistral being "**thrice as good**."
- **Attractive Pricing Draws Interest**: `@mrdragonfox` pointed out that Mistral's **pricing** makes it very attractive and emphasized that Mistral is pushing the boundaries of what's available outside of OpenAI.
- **Feedback Collection for Mistral Next**: `@egalitaristen` inquired about creating a **feedback thread** for Mistral Next to post extensive thoughts and screenshots, which `@mrdragonfox` supported, opening a thread for such discussions.

**Links mentioned**:

[Chat with Open Large Language Models](https://chat.lmsys.org/): no description found

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1210158434020884500) (28 messages🔥): 

- **GPU Selection for vLLM Backend**: `@buttercookie6265` inquired about a guide for selecting a GPU for hosting vLLM. `@mrdragonfox` advised that the model typically occupies 90% of the GPU and recommended doubling the VRAM that the model requires for adequate headroom.

- **Understanding vLLM GPU Consumption**: `@mrdragonfox` clarified that due to the quadratic scaling of the key-value store (kv), and the accumulation of context (ctx) in batching, more VRAM is necessary than the model size alone might indicate.

- **Batch Calls to vLLM Server**: `@louis2567` asked for the best method to call a vLLM server for batch requests. `@mrdragonfox` suggested using `async`, as vLLM does dynamic batching which can handle parallel requests, and implementation would depend on how the user chooses to handle threading/async in their code.

- **Enquiry about Maximum Tokens Per Second**: `.soulstealth` queried about the maximum tokens per second achieved with vLLM and Mistral 8x7b on 2 x H100 GPUs. No specific performance data was given.

- **Deployment Speed for Mistral 7b in fp16**: `@kiraa8415` sought advice on the fastest deployment option for Mistral 7b in fp16, and `@akshay_1` responded with an unclear "fastest matlab?", which did not seem to directly address the question.

- **Response Times for Support Inquiries**: `@fangh` reached out concerning a lack of response to their email inquiry. `@mrdragonfox` indicated that as Mistral has a small team, `@707162732578734181` or `@803073039716974593` should be contacted, but responses could be delayed.
  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1210547816334630952) (5 messages): 

- **Inquiry About Mistral Data Normalization**: `@severinodadalt` from Barcelona Supercomputing Center asked whether **Mistral data** has been normalized, and if so, which normalization was used and its implementation method. However, they couldn't find any relevant information, leading them to believe maybe no normalization has been applied.
- **Lack of Basemodel Data Normalization Info**: In response to `@severinodadalt`, `@mrdragonfox` stated that **no basemodel** will provide information regarding data normalization.
- **Questioning Inference Speed on Different VRAMs**: `@bdambrosio` questioned whether upgrading their VRAM to run Mistral 8x7B locally in full **fp16** might affect inference speed compared to the current 8-bit exl2 settings.
- **Perceived Differences Beyond Measured Metrics**: `@mrdragonfox` acknowledged that differences are noticed because **turbo**, presumably a tool or metric like "turboderp", primarily measures ppl (perplexity) and does not account for every possible improvement in performance.
- **Quantization Effects on Context Accuracy**: `@mrdragonfox` pointed out that context accuracy might degrade a bit with **quantization**, an important factor to consider when seeking to improve performance by adjusting bit depth.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1210473974245359677) (21 messages🔥): 

- **New to Fine-Tuning**: User `@4vis` expressed being new to fine-tuning and jokingly asked about fine-tuning Mistral with YouTube transcripts. `@_._pandora_._` advised starting with [Unsloth](https://unsloth.openai.com/) as it is beginner-friendly.
- **Data Doubts for Fine-Tuning**: `@pteromaple` wondered about the amount of data required for fine-tuning, asking if 4000 instances are sufficient. `@egalitaristen` suggested that sufficiency depends on the narrowness of the tuning task.
- **File Format Frenzy**: `@pteromaple` inquired about the correct data format when fine-tuning `"Mistral-7B-Instruct-v0.2"` with Unsloth, mentioning their current format, Alpaca. `@_._pandora_._` suggested fine-tuning the base model instead and advised understanding the prompt formatting section of Unsloth's notebook.
- **Instruct vs. Base Model Debate**: `@pteromaple` sought to maintain instruction-following abilities while altering output formats, expressing curiosity over whether starting with an Instruct model simplifies things. `@_._pandora_._` recommended using the base model for greater freedom and shared experiences about biases and language barriers in fine-tuning.
- **Hardware Hurdle for Hefty Models**: `@kodeurkubik` questioned the feasibility of fine-tuning Mistral 7B on a Mac with 16GB of RAM, considering swapping files as a solution. `@mrdragonfox` mentioned that significantly fewer parameters need to be modified for style transfer and clarified that 7B should fit in 16GB VRAM using fp16 and a batch size of one.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1210532718442713109) (8 messages🔥): 

- **Mistral API Privacy Clarification**: `@akshay_1` assured `@exa634` that data passing through the **Mistral API** is not used to train the model, reinforcing Mistral's robust privacy policy.
- **Models Hosted in Sweden**: Both `@tom_lrd` and `@ethux` confirmed to `@exa634` that Mistral hosts its platform and data in **Sweden**, which is mentioned in their [privacy policy](https://mistral.ai/privacy-policy/).
- **Privacy Policy Details**: `@ethux` posted an excerpt from the Mistral AI [Privacy Policy](https://mistral.ai/privacy-policy/), detailing the roles of **Data Controller** and **Data Processor**, and highlighted that **Azure** hosts the platform and associated data.
- **Comprehensive List of Providers**: In a more detailed posting, `@ethux` listed Mistral's main service providers, including **Azure**, **Cloudflare**, **Kong**, **Lago**, **Mailjet**, **Ory**, and **Stripe**, along with their roles and geographic details.

**Links mentioned**:

[Privacy Policy](https://mistral.ai/privacy-policy/): Frontier AI in your hands

  

---



### Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1210641574153822249) (1 messages): 

- **Perplexity Partners with ElevenLabs**: `@ok.alex` announced a new partnership with ElevenLabs, providing AI-powered voices for the **Discover Daily podcast**, which features episodes from Perplexity's Discover feed. The podcast is designed to fit easily into listeners' daily routines and is available on [favorite podcast platforms](https://podcast.perplexity.ai).

- **Discover Daily Podcast Launched**: The **Discover Daily podcast** offers daily dives into tech, science, and culture, using content from [Perplexity's Discover feed](https://www.perplexity.ai/discover) and narration by ElevenLabs' voices. It promises to be a fitting companion for various moments of the day, enhancing listeners' curiosity journey.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/elevenlabs): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discover Daily by Perplexity](https://podcast.perplexity.ai): We want to bring the world's stories to your ears, offering a daily blend of tech, science, and culture. Curated from our Discover feed, each episode is designed to enrich your day with insights and c...

  

---


### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1210143888266760293) (290 messages🔥🔥): 

- **Perplexity Pro Subscriptions: Sharing and Discounts**: `@irismava` inquired about adding team members to a Perplexity Pro plan, while `@rayinqo` asked about savings when subscribed to both ChatGPT and Perplexity Pro. `@tree.ai` confirmed that team members can be added under the advanced plan, and `@v01338` stated there are no discounts for holding multiple subscriptions. The official [billing and subscription FAQ](https://blog.perplexity.ai/faq/billing-and-subscription) posted by `@mares1317` clarifies that each employee requires an individual Pro account.

- **Experimental GPT Models**: The [Perplexity Labs YouTube playlist](https://www.youtube.com/playlist?list=PLKwRkjCH760ObtANfb0-Kat2XlvB5dKxf) and a tweet by `@perplexity_ai` shared by `@mares1317` highlighted the new "Experience Gemma 2B and 7B models," which are notable for their performance despite being lightweight.

- **Problems with Perplexity as Default Search Engine**: `@redhare18` experienced issues using Perplexity as the default search engine with Arc, which were resolved after `@ok.alex` provided assistance. Other users like `@shizlets` also faced difficulties with the Arc Search iOS app.

- **Multiple AI Models Discussed**: Users `@jaicraft` and `@rhysd21` discussed the performance and availability of various models on Perplexity Pro, including "Experimental" and "Gemini Advanced". The conversation touched on the functionality of models like "Gemini," "Claude 2.1," and "GPT-4 Turbo," with `@mares1317` and `@brknclock1215` confirming that GPT-4 Turbo is supported.

- **Image Generation Feature on Perplexity Pro**: There was confusion about generating images on Perplexity Pro, which `@trite8q1` sought clarity on. `@jaicraft` and `@ok.alex` explained that Pro members can create images by starting a new thread and using the generate image button; the process is detailed in a [blog post](https://blog.perplexity.ai/faq/images-media) and an [official thread](https://discord.com/channels/1047197230748151888/1194794305362071552).

**Links mentioned**:

- [Discover Daily by Perplexity](https://www.youtube.com/playlist?list=PLKwRkjCH760ObtANfb0-Kat2XlvB5dKxf): We want to bring the world&#39;s stories to your ears, offering a daily blend of tech, science, and culture. Crafted from our Discover feed, each episode is desi...
- [Hal9000 GIF - Hal9000 - Discover &amp; Share GIFs](https://tenor.com/view/hal9000-gif-22241038): Click to view the GIF
- [‎Discover Daily by Perplexity on Apple Podcasts](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427): ‎News · 2024
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1760757642987995595?s=20): Get on Perplexity Pro and try out the Experimental model!
- [Adiós Google | Hola Perplexity](https://youtu.be/NjQ8LeYfxRY?si=m32SzgylMsQPIBuQ): No te vas a creer lo que hace este buscador gracias a la Inteligencia Artificial. Aún no sabemos que sería de Perplexity de no ser por Jeff Bezos, Nvidia y D...
- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1760451622537158921?s=20): Introducing new additions to Perplexity Labs: Experience Gemma 2B and 7B models known for impressive performance despite being lightweight. Try it now on http://labs.pplx.ai.
- [Billing and Subscription](https://blog.perplexity.ai/faq/billing-and-subscription): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Images &amp; media](https://blog.perplexity.ai/faq/images-media): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1210340848089301063) (4 messages): 

- **Searching for Line0's Identity**: `@edzordzinam.ali` shared a [Perplexity AI search link](https://www.perplexity.ai/search/if-pline0-is-W1hK5gSpQW.p.c74OugrRQ?s=c) related to identifying what `pline0` is.
- **Delving into Risk Factors**: `@moonshot85` provided a [Perplexity AI search link](https://www.perplexity.ai/search/What-risks-are-xI..l2EDTeiswNVnnk76PQ?s=c#0) concerning the analysis of various risks.
- **Xiaomi 14 Series Insights**: `@icelavaman` posted a [link](https://www.perplexity.ai/search/Xiaomi-14-series-XciRF4QyTbKJZ8n8PgV8MA?s=c) to Perplexity AI's search results about the Xiaomi 14 series.
- **Perplexity AI and ElevenLabs Partnership Exploration**: `@icelavaman` also shared a [Perplexity AI search result](https://www.perplexity.ai/search/PerplexityAI-and-ElevenLabs-C.NsEuUNS4Ox6RIQwLWHxw?s=c) discussing a potential collaboration between Perplexity AI and ElevenLabs.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1210205157019615272) (11 messages🔥): 

- **Trouble with API Credit Purchase**: `@jenish_79522` is facing issues completing a transaction for API credits and seeks assistance, specifically tagging `<@752478851103326241>` for help.
- **Inquiry about Integrating Gemma with API**: `@karan01993` asked if there's any plan to integrate [Google's Gemma](https://ai.google.dev/gemma) with Perplexity API, looking for confirmation on future support.
- **Getting Started with Perplexity API**: `@brextonpham` inquired about accessing the Perplexity API as a newcomer, and `@icelavaman` directed them to the [getting started documentation](https://docs.perplexity.ai/docs/getting-started) and provided a contact for higher rate limits (api@perplexity.ai).
- **Payment Issue Escalation**: In response to `@jenish_79522`'s pending transaction issue, `@icelavaman` advised contacting support@perplexity.ai for assistance.
- **400 Error with 'Assistant' Field Resolved**: `@dogemeat_` reported an issue with a 400 error when using the 'assistant' field, and `@brknclock1215` suggested a workaround involving the message order that appeared to resolve the problem.

**Links mentioned**:

[no title found](https://ai.google.dev/gemma): no description found

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1210138982390825030) (79 messages🔥🔥): 

- **In Search of Transformers Code**: User `@qwerty_qwer` seeks **Transformers** code citing its simplicity and ease of setup; `@nanobitz` hints at considering **vLLM**.
- **Checkpoint Concerns**: `@stoicbatman` reports **checkpoint** issues with directory visible, yet faces errors possibly during merging or evaluation.
- **Cloud Costs vs. Server Ownership**: `@yamashi` questions the cost-effectiveness of cloud computing services after comparing the long-term rental costs with the one-time purchase of servers.
- **Hugging Face Issue Insights**: `@nanobitz` and `@stoicbatman` discuss a [GitHub issue](https://github.com/huggingface/transformers/issues/29157) regarding errors when saving with EarlyStoppingCallback, noting a loss of $60 due to this issue.
- **Model Storage Cleanup**: `@c.gato` seeks to clean up space from downloaded models and is directed by `@mihai4256` to use Hugging Face's CLI command `huggingface-cli delete-cache` and specifies it might work even while running another job.

**Links mentioned**:

- [Error while saving with EarlyStoppingCallback · Issue #29157 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29157): System Info transformers version: 4.38.0.dev0 (also in 4.38.0 and 4.39.0.dev0) Platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Huggingface_hub version: 0.20.3 Safete...
- [DeepSpeed Support Stage 3  · Issue #29254 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29254): System Info Does the trainer support stage 3? According to https://huggingface.co/transformers/v4.3.0/main_classes/trainer.html - it does not. Thanks, Brett Who can help? na Information The officia...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1210173805545394206) (6 messages): 

- **Mixtral Optimization Ideas in the Works**: `@casper_ai` believes they have determined effective ways to optimize the Mixtral model but lacks the skills to write a compatible backward pass, due to not being a CUDA engineer.
- **Enhancing Mixtral with Grouped Computations**: `@casper_ai` proposes a method to optimize Mixtral by concatenating and stacking experts, then precomputing token and expert ids for efficient grouped computations across all experts.
- **Significant Acceleration Achieved in AutoAWQ**: `@casper_ai` has achieved an impressive 8x increase in speed on Mixtral for prefilling and decoding when working with AutoAWQ.
- **Backward Pass Implementation Challenges**: `@casper_ai` discusses the potential need to import megablocks from another implementation as they have the backward passes for various operations.
- **Resource Suggestion - Gemma Inference Engine**: `@curiositix` suggests looking at [Gemma - a lightweight, standalone C++ inference engine](https://github.com/google/gemma.cpp/) for implementing the backward pass that could potentially assist with `@casper_ai`'s optimization challenges.

**Links mentioned**:

[GitHub - google/gemma.cpp: lightweight, standalone C++ inference engine for Google&#39;s Gemma models.](https://github.com/google/gemma.cpp/): lightweight, standalone C++ inference engine for Google&#39;s Gemma models. - google/gemma.cpp

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1210143755798188133) (165 messages🔥🔥): 

- **Understanding inference format for codellama**: `@nani1149` inquired about the format needed for inference after training a model with alpaca format, to which `@nanobitz` confirmed that the alpaca format is also used for inference, providing a [link to the stanford_alpaca GitHub repo](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release) for reference.
  
- **Discussing documentation and community contributions**: Users `@yamashi` and `@nanobitz` discussed the need for better documentation to avoid repeated questions, mentioning the potential use of gitbooks and citing the help of a large community in maintaining resources like gitbook for different projects.

- **Troubleshooting Learning Rate issues for Gemma 2B**: `@kearm` expressed difficulty in finding the right learning rate for Gemma 2B with various attempts listed, and `@stoicbatman` responded by suggesting to share loss charts and discussing their own experiences.

- **Merging mixtral performance concerns**: `@dreamgen` experienced slow merge times and GPU not being used while merging mixtral, leading to discussions with `@nanobitz` about potential solutions and whether running out of VRAM or operating on RAM was the issue.

- **Troubleshooting checkpoint saving error during model training**: `@kearm` struggled with a checkpoint saving issue during model training, which was not resolved despite trying a downgrade of deepspeed as suggested by `@stoicbatman`. The conversation involved back and forth suggestions and references to related GitHub issues.

**Links mentioned**:

- [Docker](https://hub.docker.com/r/winglian/axolotl-cloud/tags): no description found
- [nottlespike](https://wandb.ai/nottlespike/Gemma/runs/hhkez6fn?workspace=user-nottlespike): Weights & Biases, developer tools for machine learning
- [monk1337](https://wandb.ai/monk1337/gemma_results): Weights & Biases, developer tools for machine learning
- [Error while saving with EarlyStoppingCallback · Issue #29157 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29157): System Info transformers version: 4.38.0.dev0 (also in 4.38.0 and 4.39.0.dev0) Platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Huggingface_hub version: 0.20.3 Safete...
- [fine tune gemma model checkpoint save error · Issue #1320 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1320): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior should work Current behaviour this error comes when...
- [GitHub - tatsu-lab/stanford_alpaca: Code and documentation to train Stanford&#39;s Alpaca models, and generate the data.](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release): Code and documentation to train Stanford&#39;s Alpaca models, and generate the data. - tatsu-lab/stanford_alpaca

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1210262728535711784) (9 messages🔥): 

- **DreamGen's New AI Models Released**: `@dreamgen` announced the launch of new AI models for story-writing and role-playing, trainable with Axolotl and Unlosth, and detailed on Hugging Face with a collection at [dreamgen/opus-v1-story-writing-and-role-playing-models](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31). They feature ~100M tokens of human-generated data and are based on an extended version of ChatML, with further instructions available in the [Opus V1 guide](https://dub.sh/opus-v1-guide).
- **Prompt Template Oversight Corrected**: `@nanobitz` noticed that `@dreamgen` seemed to have forgotten to update the tokenizer's chat template for the new models; `@dreamgen` acknowledged the issue, confirming that version 7b did not update as intended.
- **Possible Prompt Leakage in Opus V1.2-7b**: 'nanobitz' reported an issue when testing `@dreamgen`'s new models, suggesting that the prompts might be leaking user and assistant roles during conversation starts in chat mode. `@dreamgen` responded with a link to the prompting format code to clarify the setup, [prompt formating code](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/opus-v1.py).
- **Further Review Needed on Formatting Issue**: `@dreamgen` is looking into the earlier mentioned "leak" reported by `@nanobitz`, who indicated the need to investigate more after noticing user/assistant content in the final assistant message.
- **Phi-2 Model Fine-tuned With Axolotl**: `@finetuningllms` shared a link to their fine-tuning of the Phi-2 model, noting high performance and promising to soon add a model card including an image, available at [axra/phi-2-x-0.1](https://huggingface.co/axra/phi-2-x-0.1).

**Links mentioned**:

- [axra/phi-2-x-0.1 · Hugging Face](https://huggingface.co/axra/phi-2-x-0.1): no description found
- [configs/opus-v1.py · dreamgen/opus-v1.2-7b at main](https://huggingface.co/dreamgen/opus-v1.2-7b/blob/main/configs/opus-v1.py): no description found
- [Opus V1: Story-writing &amp; role-playing models - a dreamgen Collection](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31): no description found
- [DreamGen: AI role-play and story-writing without limits](https://dub.sh/opus-v1-guide): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1210177857465229363) (6 messages): 

- **RunPod Image Availability Concerns**: `@stoicbatman` inquired if the **RunPod image** was deleted as they were unable to find it.
- **Helpful Direction to Docker Hub**: In response, `@nanobitz` shared a direct [link to Docker Hub](https://hub.docker.com/r/winglian/axolotl-runpod/tags) where the RunPod image tags can be found.
- **Confusion Over GitHub Readme**: `@stoicbatman` followed up to mention that the **GitHub readme** is no longer redirecting to the actual RunPod image.
- **Seeking the Latest Link**: `@nanobitz` asked `@stoicbatman` if they have the latest link, attempting to address the redirection issue mentioned.
- **Reliance on Docker Hub Over GitHub**: `@stoicbatman` confirmed using the image from Docker Hub but expressed confusion as the GitHub readme previously redirected to the RunPod image, which is no longer the case.

**Links mentioned**:

[Docker](https://hub.docker.com/r/winglian/axolotl-runpod/tags): no description found

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1210336265468117014) (1 messages): 

- **Visualizing Aya Dataset**: User `@416019758492680203` provides a [visualization of the Aya dataset](https://huggingface.co/posts/cakiki/501967924678592) for better insights and understanding.
- **Image Generation Upgrade**: With the new release of **Proteus V0.4**, `@1093866142608670772` enhances the capabilities of image generation, available at [Proteus V0.4 space](https://huggingface.co/spaces/FumesAI/Proteus-V0.4).
- **Interactive Text-to-Image RAG Prompts**: User `@942079288952381461` created an interactive demo to play with over 1.4 million text2image prompts using RAG, accessible [here](https://c6548e7f4c4e5a6d00.gradio.live/).
- **Serverless Hosted API for Inference**: `@319141699605626881` shares a serverless inference solution hosted on a free Colab environment, with details on [GitHub](https://github.com/groloch/LocalLlm).
- **Innovating with ProteinBERT and Fluently Models**: Links to **ProteinBERT** model weights by `@403280164433297409` and the accompanying [paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274) were shared, along with the **Fluently diffusion model** demo by `@1056663454519406652`, available at [Fluently space](https://huggingface.co/spaces/ehristoforu/Fluently).
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1210140027649269801) (149 messages🔥🔥): 

```html
<ul>
  <li><strong>Seeking Performance Clarity</strong>: User <code>@0ldgranpa</code> inquires about optimal model types and performance fixes for his hardware specifications. There are no responses to guide them yet.</li>
  <li><strong>GPU Memory Workarounds</strong>: <code>@alifthi</code> asks for solutions to run large models like Mistral with limited GPU memory, and <code>@typoilu</code> suggests using llama.cpp or accelerate for CPU offloading.</li>
  <li><strong>Hardware Curiosity</strong>: <code>@zorian_93363</code> compares ASIC mining machines' capabilities to potential uses for running models, and <code>@vipitis</code> explains the difference between computational tasks and discusses current hardware such as Google's TPU and Graphcore's IPU.</li>
  <li><strong>Exploring GPT Alternatives</strong>: <code>@amirgame197</code> asks why GPT 3.5 is unlimited and free on chat.openai.com but paid on api.openai.com, suggesting he’s seeking free alternatives for API usage, without receiving a direct answer.</li>
  <li><strong>Accidental Template Confusion</strong>: In a coding issue, <code>@levisco</code> initially struggles with using the create_sample feature from the transformers QuestionAnsweringPipeline, but discovers it was only a typo in their code.</li>
</ul>
```

**Links mentioned**:

- [Groq](https://groq.com): no description found
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [3rd Rock GIF - 3rd Rock From - Discover &amp; Share GIFs](https://tenor.com/view/3rd-rock-from-the-sun-gif-5973311): Click to view the GIF
- [On-device training in TensorFlow Lite &#8212; The TensorFlow Blog](https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html): no description found
- [Use custom models](https://huggingface.co/docs/transformers.js/custom_usage): no description found
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3): Announcing Stable Diffusion 3 in early preview, our most capable text-to-image model with greatly improved performance in multi-subject prompts, image quality, and spelling abilities.
- [Deer GIF - Deer - Discover &amp; Share GIFs](https://tenor.com/view/deer-gif-22652112): Click to view the GIF
- [🌌 Analysis of Spaces in Hugging Face](https://huggingface.co/blog/Weyaxi/huggingface-spaces-analysis): no description found
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1761042421243093164): 🎉 New blogpost in @huggingface   🌌 Analysis of Spaces in Hugging Face  I scraped 20K spaces&#39; code files and combined them into one dataset, showcasing meaningful statistics 📶  📝 Blogpost: http...
- [GitHub - SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2](https://github.com/SYSTRAN/faster-whisper): Faster Whisper transcription with CTranslate2. Contribute to SYSTRAN/faster-whisper development by creating an account on GitHub.
- [GitHub - kuangliu/pytorch-cifar: 95.47% on CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar): 95.47% on CIFAR10 with PyTorch. Contribute to kuangliu/pytorch-cifar development by creating an account on GitHub.
- [Phind-70B: BEST Coding LLM Outperforming GPT-4 Turbo + Opensource!](https://www.youtube.com/watch?v=v0ZN_MKYfhw): In this video, we unveil the revolutionary capabilities of Phind-70B, designed to close the code quality gap and accelerate your coding process. With up to 8...
- [Pipelines](https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline.create_sample): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1210206360055193600) (4 messages): 

- **Flutter Game Inquiry**: User `@.konoh` inquired about a flutter game, but provided no further context or details.
- **Hugging Face Open Sources "DoReMi"**: User `@neuralink` shared a [link](https://github.com/huggingface/nanotron/tree/main/examples/doremi) to an open-sourced Hugging Face project on GitHub named **DoReMi**, part of the **nanotron** repository.
- **User Feels Overwhelmed by Complexity**: `@cursorop` expressed feeling overwhelmed by the complexity of the project shared by `@neuralink`, using the `:blobsweat:` emoji to convey their sentiment.
- **Seeking Advice on Imitation Learning for Robotics**: `@alefram` asked the community for tips or resources for learning about imitation learning as it applies to robotics, but no responses were provided within the given messages.

**Links mentioned**:

[nanotron/examples/doremi at main · huggingface/nanotron](https://github.com/huggingface/nanotron/tree/main/examples/doremi): Minimalistic large language model 3D-parallelism training - huggingface/nanotron

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1210202983397068800) (5 messages): 

- **Benchmarking AI Models**: User `@ryzxl` announced a comprehensive benchmarking initiative for AI models, comparing platforms like **gpt-3.5-turbo-instruct** and **Mistral.** The initiative covered key datasets including ASDiv, BBQ, BigBench, and more. Full details and the leaderboard can be found in their [LinkedIn post](https://lnkd.in/gxUHqwNp).

- **Reminding About Posting Etiquette**: User `@cakiki` reminded `@ryzxl` to avoid cross-posting the same message multiple times to prevent spam.

- **Deep Unsupervised Learning Course Announcement**: User `@omrylcn.` shared information about the Spring 2024 offering of Berkeley's **Deep Unsupervised Learning** course, covering Deep Generative Models and Self-Supervised Learning, similar to [previous offerings](https://sites.google.com/view/berkeley-cs294-158-sp20/home).

- **Large Action Models (LAMs)**: User `@fernando_cejas` shared a [blog post](https://blog.finxter.com/large-action-models-lams-a-new-step-in-ai-for-understanding-and-doing-human-tasks/) discussing **Large Action Models (LAMs)**, which are AI systems that perform human-like tasks within digital environments through neural networks and symbolic reasoning.

- **Warp Dev Referral**: User `@gjyotin305` posted a referral link to [Warp Dev](https://app.warp.dev/referral/59MJGK), but provided no additional context or information about the link.

**Links mentioned**:

- [CS294-158-SP24 Deep Unsupervised Learning Spring 2024](https://sites.google.com/view/berkeley-cs294-158-sp24/home): About: This course will cover two areas of deep learning in which labeled data is not required: Deep Generative Models and Self-Supervised Learning.  Recent advances in generative models have made it ...
- [Warp](https://app.warp.dev/referral/59MJGK): no description found
- [Large Action Models (LAMs): A New Step in AI for Understanding and Doing Human Tasks &#8211; Be on the Right Side of Change](https://blog.finxter.com/large-action-models-lams-a-new-step-in-ai-for-understanding-and-doing-human-tasks/): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1210211136520065034) (26 messages🔥): 

- **Unofficial ChatGPT API via Selenium Raises Concerns**: `@.infinityhawk` shared a link to an unofficial ChatGPT API created with Selenium ([Github Repo](https://github.com/Priyanshu-hawk/ChatGPT-unofficial-api-selenium)). Both `@myg5702` and `@cakiki` raised potential ethical and practical concerns, such as contravening OpenAI's terms of service and risking IP or RP bans.

- **Optimization Techniques for Stable Diffusion XL**: `@felixsanz` published a comprehensive article detailing optimization methods for Stable Diffusion XL, enabling image generation on GPUs with just 6 GB memory ([Read the Article](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl)). Despite timing the release coinciding with the announcement of Stable Diffusion 3, `@paccer` commended the educational value and efforts.

- **Cheaper Access to OpenAI GPT-4 Models via New API**: `@exrew` introduced an API offering affordable access to OpenAI GPT-4 models, with a free plan for trial and a flexible credit system for various models ([Find the API here](https://rapidapi.com/NextAPI/api/cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api)).

- **Real-time Text Streaming Chat Interface with Gemma**: `@not_lain` created a text streaming chat interface utilizing the new Gemma AI model, promising fast performance ([Experience it here](https://huggingface.co/spaces/not-lain/text-streaming-with-gemma-2b-it)).

- **Browser-Based Speaker Embeddings with WavLMForXVector**: `@davidre95` has contributed to `transformers.js` by submitting a pull request to support WavLMForXVector, enabling running speaker embeddings models directly in the browser ([PR on GitHub](https://github.com/xenova/transformers.js/pull/603); [Model on HuggingFace](https://huggingface.co/D4ve-R/wavlm-base-plus-sv)).

**Links mentioned**:

- [Proteus V0.4 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Proteus-V0.4): no description found
- [Text-Streaming - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/text-streaming-with-gemma-2b-it): no description found
- [xVASynth TTS - a Hugging Face Space by Pendrokar](https://huggingface.co/spaces/Pendrokar/xVASynth?refreshed=1): no description found
- [D4ve-R/wavlm-base-plus-sv · Hugging Face](https://huggingface.co/D4ve-R/wavlm-base-plus-sv): no description found
- [Cheapest GPT-4 Turbo, GPT 4 Vision, ChatGPT OpenAI AI API API Documentation (NextAPI) | RapidAPI](https://rapidapi.com/NextAPI/api/cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api): no description found
- [GitHub - Priyanshu-hawk/ChatGPT-unofficial-api-selenium: This is unofficial ChatGPT API totally written by me in python with selenium](https://github.com/Priyanshu-hawk/ChatGPT-unofficial-api-selenium): This is unofficial ChatGPT API totally written by me in python with selenium - Priyanshu-hawk/ChatGPT-unofficial-api-selenium
- [lo-fi ableton speedrun with musicgen, max4live and acoustic guitar - captains chair 15](https://youtu.be/3YzlC1kafW8): this week&#39;s episode we&#39;re using the acoustic again with  @veryVANYA  &#39;s fine-tunewe&#39;re using  @matttytel9056  &#39;s helm vstwhich is better than the ones you ha...
- [Add support for WavlmForXVector by D4ve-R · Pull Request #603 · xenova/transformers.js](https://github.com/xenova/transformers.js/pull/603): Adding support for wavlm with xvector head on top. The onnx version of microsoft/wavlm-base-plus-sv can be found at D4ve-R/wavlm-base-plus-sv. Aims to be as close to the python implementation as po...
- [Ultimate guide to optimizing Stable Diffusion XL](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl): Discover how to get the best quality and performance in SDXL with any graphics card.

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1210474517600665640) (5 messages): 

- **Neural Circuit Diagrams Presentation Scheduled**: `@chad_in_the_house` confirmed there **will be a recording** of the neural circuit diagrams presentation.
- **Time Confirmation for a Live Event**: `@chad_in_the_house` mentioned the presentation will take place at **7pm EST** today.
- **Consideration for Time Zones**: `@gschwepp_84093` noted that the presentation time translates to **00:00 UTC**, expressing potential difficulty in attending due to the late hour.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1210188041344389190) (5 messages): 

- **Inquiring about Interlingua-Based Translators**: User `@hobojesus6250a` expressed interest in finding or creating an **Interlingua-based translator** on Hugging Face and discussed the potential need to extend an existing model due to time constraints.
- **Looking for Ways to Expand Class Limit**: `@agusschmidt` asked how to run the [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model with more than 10 classes, referencing a previous discussion that suggested it was possible when running the model locally.
- **Friendly Caution from HuggingMod**: The automated moderation bot `@HuggingMod` reminded users `<@345587852052267018>` and `<@745207885201539072>` to slow down their posting, indicating they might be sending too many messages in a short span of time.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1210215877396668457) (3 messages): 

- **Multi-label Image Classification Tutorial Drops**: User `@nielsr_` shared a [tutorial notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb) for multi-label image classification using **SigLIP**, a strong vision backbone, although any vision model from the Transformers library can be substituted.

- **Too Much Zeal from HuggingMod**: `@745207885201539072` received a gentle warning from HuggingMod to slow down their message posting speed on the server.

- **Forge Ahead with Emotion Recognition**: `@rodricota_` began a discussion on building an emotion recognition model and expressed a desire to troubleshoot some issues.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1210283284194656266) (49 messages🔥): 

- **Peft's Persistent Problem**: `@grimsqueaker` mentioned a significant bug where peft does not save the right heads for non-auto-configured architectures. The workaround involved random parameter adjustments until finding a config that works but compromises had to be made.

- **Reformer Research Ruminations**: `@devbravo` shared their current research focus on developing *smaller, more memory-efficient models* with Reformer architecture to run on edge devices. A reminder to keep it slow appeared from `@HuggingMod`, prompting `@devbravo` to slow down their rapid posting.

- **GPT Length Logistics**: `@vipitis` corrected `@nrs9044` by stating that *Transformers are not recurrent but fully parallel*, and confirmed that the size of self-attention matrices in GPT indeed scales quadratically with sequence length.

- **Generating Positive and Negative Sentiments**: `@jimmyfromanalytics` inquired about fine-tuning Flan T5 for creating synthetic data for *sentiment analysis*. Discussions revolved around prompt engineering and potentially exploring decoder-only models for better performance.

- **Fine-Tuning vs. Large Model Dilemma**: `@arkalonman` sought insights regarding whether to fine-tune a larger LLM like Mistral 7B for text classification, versus sticking with a BERT variant. The conversation with `@lavi_39761` led to a consensus that efficient encoder models might be a better focus than more substantial models for classification purposes.

**Links mentioned**:

- [climatebert (ClimateBert)](https://huggingface.co/climatebert): no description found
- [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/html/2312.05934v3): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1210188041344389190) (5 messages): 

- **Exploring Interlingua-Based Translators**: `hobojesus6250a` raised a question about whether anyone has experimented with creating or tweaking an **Interlingua-based translator on Hugging Face**. They expressed interest in extending an existing model for a university project due to time constraints.
- **Expanding Classes for BART MNLI Model**: `agusschmidt` inquired about how to run the [BART-large-mnli model](https://huggingface.co/facebook/bart-large-mnli) with more than 10 classes, suggesting they are aware of the possibility when running it locally and seeking guidance on how to implement this.
- **Friendly Bot Reminders to Avoid Spam**: **HuggingMod**, the Hugging Face moderator bot, issued reminders to `@345587852052267018` and `@745207885201539072` to **slow down their message posting** as they were sending messages too rapidly.
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1210224135704420403) (52 messages🔥): 

<ul>
  <li><strong>Prompt Engineering with GPT-4:</strong> `@henriqueln7` expressed disappointment as GPT-4 failed to effectively rewrite prompts as per their request, instead generating responses akin to a new assistant's. They plan to test further in the playground.</li>
  <li><strong>Announcement of Stable Diffusion 3:</strong> `@rubenartus` shared [an announcement](https://stability.ai/news/stable-diffusion-3) about Stable Diffusion 3's early preview with enhanced multi-subject prompt performance and spelling abilities. They also provided a [link](https://twitter.com/EMostaque/status/1760660709308846135) to more model details.</li>
  <li><strong>Google's New Model Gemini Pro 1.5:</strong> `@nuvic_` discussed the capabilities of Gemini Pro 1.5 highlighting its 1,000,000 token context size and ability to use video as input, as explored through Google AI Studio.</li>
  <li><strong>Assessing Reddit's Data Deal with Google:</strong> Users like `@guardiang` and `@pennepitstop` provided perspectives on the financial and strategic implications of [Google’s reported $60 million/year data deal](https://news.ycombinator.com/item?id=39471964) with Reddit ahead of its IPO.</li>
  <li><strong>Gemini Image Generation Paused:</strong> `@swyxio` posted a [link](https://blog.google/products/gemini/gemini-image-generation-issue/) to a Google blog where the SVP took responsibility for issues with Gemini's image generation feature, which resulted in a temporary pause of the function.</li>
</ul>

**Links mentioned**:

- [Google cut a deal with Reddit for AI training data | Hacker News](https://news.ycombinator.com/item?id=39471964): no description found
- [SDXL Lightning - by fal.ai](https://fastsdxl.ai/): Lightning fast SDXL API demo by fal.ai
- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/): Last week Google introduced Gemini Pro 1.5, an enormous upgrade to their Gemini series of AI models. Gemini Pro 1.5 has a 1,000,000 token context size. This is huge—previously that …
- [Things I Don&#x27;t Know About AI](https://blog.eladgil.com/p/things-i-dont-know-about-ai): The more I learn about AI markets, the less I think I know. I list questions and some thoughts.
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3): Announcing Stable Diffusion 3 in early preview, our most capable text-to-image model with greatly improved performance in multi-subject prompts, image quality, and spelling abilities.
- [Interconnects Audio | Google ships it: Gemma open LLMs and Gemini backlash](https://podcast.interconnects.ai/episodes/google-ships-it-gemma-open-llms-and-gemini-backlash): Audio format of posts on interconnects.ai -- generated with AI from the author.
- [Gemini image generation got it wrong. We&#x27;ll do better.](https://blog.google/products/gemini/gemini-image-generation-issue/): An explanation of how the issues with Gemini’s image generation of people happened, and what we’re doing to fix it.
- [Tweet from Shu (@shuding_)](https://x.com/shuding_/status/1761085838174175379?s=46&t=90xQ8sGy63D2OtiaoGJuww):   ↘️ Quoting Guillermo Rauch (@rauchg)   AG(UI) has been achieved internally
- [Is the AI Boom Real?](https://youtu.be/J-BvkmNtgAM?si=W6XSJocA6odM9kqS): Notes: 7:50 - TPUs are in their fifth iteration. Messed up. Links:- The Asianometry Newsletter: https://www.asianometry.com- Patreon: https://www.patreon.com...
- [OpenAI’s Sora: How to Spot AI-Generated Videos | WSJ](https://youtu.be/XllmgXBQUwA?si=p9): OpenAI just revealed Sora – an AI video generator that creates hyper-realistic scenes and animated worlds in moments. But the tech isn’t perfect. There are a...
- [Tweet from Jim Fan (@DrJimFan)](https://x.com/drjimfan/status/1761052023821369639?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Career update: I am co-founding a new research group called &#34;GEAR&#34; at NVIDIA, with my long-time friend and collaborator Prof. @yukez. GEAR stands for Generalist Embodied Agent Research.  We be...
- [Demis Hassabis on Chatbots to AGI | EP 71](https://youtu.be/nwUARJeeplA?si=V09X6h7iqucrh4af): This week’s episode is a conversation with Demis Hassabis, the head of Google’s artificial intelligence division. We talk about Google’s latest A.I. models, ...
- [[AINews] Google AI: Win some (Gemma, 1.5 Pro), Lose some (Image gen)](https://buttondown.email/ainews/archive/ainews-google-ai-win-some-gemma-15-pro-lose-some/): AI Discords for 2/20/2024. We checked 20 guilds, 313 channels, and 8555 messages for you. Estimated reading time saved (at 200wpm): 836 minutes. Google is at...

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1210525808612085812) (6 messages): 

- **LLM Paper Club T5 Discussion**: `@ivanleomk` announced a session of the **LLM Paper Club** discussing the T5 paper with `@bryanblackbee`. The event was scheduled to happen in 5 minutes with a link to join the discussion: [Join LLM Paper Club](https://discord.gg/wjrQxPpW).
- **Regretting Missing the Paper Club**: `@swyxio` expressed regret for missing the LLM Paper Club on T5 led by `@bryanblackbee`, hinting at the need for a recording of the session.
- **AI in Action Event**: `@kbal11` promoted an **AI in Action** event with `@yikesawjeez` focusing on local models. A link to the session was provided: [Learn About Local Models](https://discord.gg/QCPSP7bv).
- **Praise for AI Event Management**: `@swyxio` complimented `@kbal11` on the successful management of the AI in Action session led by `@yikesawjeez`.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/QCPSP7bv): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Join the Latent Space (née /dev/invest) Discord Server!](https://discord.gg/wjrQxPpW): Check out the Latent Space (née /dev/invest) community on Discord - hang out with 2980 other members and enjoy free voice and text chat.

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1210526022488039447) (16 messages🔥): 

- **LLM Paper Club Asia Edition Kicks Off**: `@ivanleomk` invites participants to join the discussion, offering a platform for anyone to ask questions or discuss topics by joining the stage as a speaker or chatting if they're more comfortable with that.
- **Central Repository for Notes and Insights**: `@bryanblackbee` provides a [link to notes](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4), which is a central repository for the discussions taking place in the LLM Paper Club.
- **Inquiries to the Community About Model Vocabularies and Constraints**: `@mattoshimasu` is curious about whether new models they're discussing have a smaller set of vocabulary and also asks about the text length and verb count constraints.
- **Fine-Tuning Mechanisms in NLP Explained**: In response to `@healthymonkey`'s question, the community discusses fine-tuning in NLP tasks like T5 for sentiment classification, touching on whether the head/linear layer is replaced, like in computer vision.
- **Technical Comparison of Encoder-Decoder vs. Decoder-Only Architectures**: `@hanzo4958` sparks a discussion about the differences between encoder-decoder and decoder-only architectures in traditional NLP tasks, noting the rising popularity of decoder-only models.
- **Parting Gratitude and Positive Feedback on the Session**: Several participants, including `@healthymonkey`, `@hanzo4958`, `@thehippoguy`, `@edwin_75513_08956`, and `@lord_idiot`, express their thanks and appreciation for the detailed session and notes before leaving the discussion.

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1210693465269338193) (136 messages🔥🔥): 

- **Local Models and LoRA Discussion**: Users discussed their experiences with local AI models and the LoRA (Low-Rank Adaptation) technique. `@markredito` clarified that LoRA is an adapter placed on top of a generative model to influence its output and is commonplace in platforms like Stable Diffusion.

- **Latent Space Final Frontiers Event**: `@kbal11` shared details about the [Latent Space Final Frontiers](https://lu.ma/latent-space-final-frontiers) event, which focuses on pushing AI boundaries and features a research/startup competition with notable judges from companies like GitHub, Replit, and LlamaIndex.

- **ComfyUI for Stable Diffusion**: `@markredito` provided a [GitHub link](https://github.com/comfyanonymous/ComfyUI) to ComfyUI, which is described as a powerful and modular GUI, API, and backend for Stable Diffusion with a graph/nodes interface.

- **AI Model Merging Trend**: `@swyxio` shared a [Hugging Face blog post](https://huggingface.co/blog/mlabonne/merge-models) discussing the emerging technique of model merging that allows combination of multiple LLMs to create state-of-the-art models for cheap, highlighting the use of the mergekit library.

- **Civit.ai Model Gallery Concerns**: `@kbal11` pointed out the prevalence of stylized and sexualized images of young women in the Civit.ai model gallery, sparking a lighthearted but poignant discussion about the content generated by AI and shared within the community.

**Links mentioned**:

- [Twitch](https://twitch.tv/yikesawjeez): no description found
- [SDXL Lightning - by fal.ai](https://fastsdxl.ai/): Lightning fast SDXL API demo by fal.ai
- [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models): no description found
- [Smol Talk](https://buttondown.email/ainews): We summarize AI discords, and send you a roundup each day!
- [Latent Space: Final Frontiers · Luma](https://lu.ma/latent-space-final-frontiers): We&#x27;re excited to host the second annual Latent Space demo day 🚀 Enough chatting with PDFs. Let&#x27;s see some Science Fiction-level AI. This year&#x27;s theme is Final Frontiers: who are the te...
- [GitHub - deforum-art/deforum-stable-diffusion](https://github.com/deforum-art/deforum-stable-diffusion): Contribute to deforum-art/deforum-stable-diffusion development by creating an account on GitHub.
- [GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.](https://github.com/comfyanonymous/ComfyUI): The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1210135297304559617) (107 messages🔥🔥): 

- **Skeptical Takes on Simulating Personhood**: `@sparetime.` expressed *skepticism* about the claim that GPT-4 and a scratchpad can simulate a human, questioning the model's ability to faithfully generate realistic experiences. `@rallio.` responded with a detailed explanation that the simulation would include creating a set of fake memories and layers to emulate human behavior and perspective, even noting recent improvements in memory consistency.
  
- **Discord Member Shares Benchmark Critique Video**: `@cahya.wirawan` shared a [YouTube video link](https://youtu.be/74Uo2HU8HBo) titled "Everything WRONG with LLM Benchmarks (ft. MMLU)!!!" which criticizes benchmarks for large language models, sparking a conversation about the validity and effectiveness of current LLM benchmarks.

- **Eleuther Community Discusses Improving LLM Consistency**: In a technical discussion, `@rallio.` suggested that the issues related to consistency in simulating memories for Large Language Models (LLMs) have been potentially mitigated according to recently published research such as Google's TrueTeacher and Propsegment.

- **The Hallucination Debate**: `@rallio.` mentioned a [company called Superfocus](https://superfocus.ai/#about) which claims to have achieved near 100% factual accuracy for LLMs, implying a solution to the hallucination problem. This sparked a debate with `@fern.bear` over the veracity of these claims and the nature of solving the hallucination issue with LLMs.

- **Creating Lifelike NPCs in Virtual Worlds**: `@rallio.` discussed their ambition to create persistent NPCs that could interact with humans in virtual worlds without revealing their artificial nature. They explained this would utilize the formulated approach for consistency and memory simulation in conjunction with fine-tuning and context. 

- **Community Shout-Out for Collaboration**: `@hawk1399` prompted the community to consider a project based on a paper outlining the use of diffusion models to generate high-performing neural network parameters, inviting others to contribute to continued research in the field.

**Links mentioned**:

- [Neural Network Diffusion](https://arxiv.org/abs/2402.13144): Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also \textit{generate high-performing neural network parameters}...
- [SuperFocus](https://superfocus.ai/#about): no description found
- [Everything WRONG with LLM Benchmarks (ft. MMLU)!!!](https://youtu.be/74Uo2HU8HBo?si=D9bHCZZrnIRX9skj): 🔗 Links 🔗When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboardshttps://arxiv.org/pdf/2402.01781.pdf❤️ If you want to s...
- [PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition](https://arxiv.org/abs/2212.10750): The widely studied task of Natural Language Inference (NLI) requires a system to recognize whether one piece of text is textually entailed by another, i.e. whether the entirety of its meaning can be i...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1210243156453625926) (70 messages🔥🔥): 

- **Model Naming Ethics Debated**: `@thooton_` [expressed frustration](https://discord.com) over misleading model naming practices, suggesting that a model named "7b" should not exceed 7.99b parameters. They highlighted the inconsistency with "gemma-7b" actually having 8.5b parameters, while "gemma-2b" is closer to its stated size with 2.5b parameters.
- **Clarifications on Embedding Sizes**: In a discussion with `@catboy_slim_`, it was clarified that "gemma-7b" includes 8.5 billion parameters with the embedding size considered, but the number matches the correct leading digit when embeddings are excluded.
- **New Paper on Minimizing Data Loss**: `@jckwind` shared excitement for a new paper on data efficiency and minimizing information loss during layer transmissions, advocating its novelty and potential usefulness.
- **Searchformer Beats Traditional Planners**: `@jckwind` highlighted "Searchformer", a Transformer that outperforms traditional symbolic planners by solving Sokoban puzzles while utilizing fewer search steps than A* search.
- **Simplicity in AI Alignment with Reinforce**: Discussion around [a paper](https://arxiv.org/pdf/2402.14740.pdf) suggested that simpler REINFORCE-style optimization could be more effective for RLHF (Reinforcement Learning from Human Feedback) compared to the canonical PPO method, which `@canadagoose1` mentioned discussing extensively.

**Links mentioned**:

- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740): AI alignment in the shape of Reinforcement Learning from Human Feedback (RLHF) is increasingly treated as a crucial ingredient for high performance large language models. \textsc{Proximal Policy Optim...
- [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083): While Transformers have enabled tremendous progress in various application settings, such architectures still lag behind traditional symbolic planners for solving complex decision making tasks. In thi...
- [xVal: A Continuous Number Encoding for Large Language Models](https://arxiv.org/abs/2310.02989): Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding sche...
- [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616): Today&#39;s deep learning methods focus on how to design the most appropriate objective functions so that the prediction results of the model can be closest to the ground truth. Meanwhile, an appropri...
- [Uncovering mesa-optimization algorithms in Transformers](https://arxiv.org/abs/2309.05858): Transformers have become the dominant model in deep learning, but the reason for their superior performance is poorly understood. Here, we hypothesize that the strong performance of Transformers stems...
- [Spectral State Space Models](https://arxiv.org/abs/2312.06837): This paper studies sequence modeling for prediction tasks with long range dependencies. We propose a new formulation for state space models (SSMs) based on learning linear dynamical systems with the s...

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1210281621278691358) (17 messages🔥): 

- **Training of Chinese Lens Underway**: `@mrgonao` mentioned that the training of the Chinese lens is in progress and will finish in a few hours. Additionally, there is an issue with the 13b model showing uniform output which will be checked alongside the Chinese lens comparison.
- **Unlearning in Language Models**: `@millander` highlighted a recent academic publication titled *Survey and formalization of LLM unlearning*. It can be accessed [here](https://arxiv.org/abs/2402.08787) for detailed insights on unlearning processes in Large Language Models (LLMs).
- **Identical Tokenizer Across Models?**: Such issue prompted `@mrgonao` to inquire whether the tokenizer used for the 13b model is the same as that of the 7b model, which may associate with the model's odd behavior of "thinking in Chinese" when applied with a Chinese lens.
- **Lens Training Could Lead to Intra-Translation**: `@butanium` contributed a hypothesis that training the tuned lens exclusively on Chinese content might impel the lens to translate from English to Chinese, expecting an inclination toward the presence of Chinese tokens in an English setting.
- **Troubleshooting Dataset Anomalies**: `@mrgonao` is experiencing unexpected dataset behaviors in translation tasks and is seeking to rectify potential issues, mentioning that the wrong languages are paired with words. The related GitHub repository can be found [here](https://github.com/SrGonao/llm-latent-language/tree/tuned-lens/visuals/translation).

**Links mentioned**:

- [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787): We explore machine unlearning (MU) in the domain of large language models (LLMs), referred to as LLM unlearning. This initiative aims to eliminate undesirable data influence (e.g., sensitive or illega...
- [llm-latent-language/visuals/translation at tuned-lens · SrGonao/llm-latent-language](https://github.com/SrGonao/llm-latent-language/tree/tuned-lens/visuals/translation): Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;. - SrGonao/llm-latent-language

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1210133882809225216) (4 messages): 

- **False Negatives in Scaled Datasets not a Concern**: `@_.hrafn._` opined that exact false negatives are unlikely at the scale of current datasets like **datacomp** or **metaclip**, particularly with balanced datasets. They suggested generating unimodal embeddings or computing similarity scores on the fly to mitigate concerns.
- **Creating Own Model to Exclude Hard Negatives**: `@_.hrafn._` further proposed the idea of using one's own model during training to compute similarity scores in order to exclude particularly hard negatives.
- **Irrelevance of Solution for Non-Image-Text Projects**: `@tz6352` responded that the discussed issue and solutions are not applicable for them as they are not working on Image-Text projects.
- **Loss Masking as a Viable Solution**: `@.solux` discussed the possibility of masking the loss for samples that are too close, suggesting it as a potential solution when there's no good way to identify false negatives in the training process.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1210167779106357279) (8 messages🔥): 

- **Exploration of Pre-training Techniques**: `@pminervini` shared an [arXiv paper](https://arxiv.org/abs/2402.13991) which discusses the impact of sequence composition and causal masking during pre-training of language models. The findings suggest that intra-document attention can significantly improve performance on various tasks.

- **PR Gratitude and Prompting Protocol**: `@tastybucketofrice` thanked `@441658587404697600` for their pull requests and encouraged future pings for faster merges.

- **LoRA Finetuning Inquiry**: `@norabelrose` inquired about the feasibility of using LoRA finetuning with the `gpt-neox` codebase, signaling their current use of Hugging Face and PyTorch Lightning for similar tasks.

- **Potential Shift to NeoX Codebase for Finetuning**: Faced with issues using PyTorch native FSDP, `@norabelrose` contemplated using the `gpt-neox` repository for a NeoX 20B finetune. 

- **Resolution Acknowledgment**: `@80melon` acknowledged the resolution of a previously unstated issue brought up by `@norabelrose`.

**Links mentioned**:

[Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991): Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use causal masking to compute the likelihood of each token given its context; this strategy i...

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1210267978294108310) (3 messages): 

- **Turn RAG into Full-Stack**: `@wenqi_glantz` shared a tutorial on converting RAG notebook into a full-stack app with ingestion and inference services. She explains the setup of an ingestion service and more in [her guide](https://t.co/S86B38YZQ1). ![Tweet Screenshot](https://t.co/fpt7OSurs8)
- **Rapid Re-ranking with ColBERT**: ColBERT, highlighted by `@lateinteraction`, provides a faster alternative for document re-ranking at 100x the speed of BERT-based models. `@Haotianzh` is credited for its efficiency and improved performance over dense retrieval in [this tweet](https://t.co/kzvNPELgQ4). ![Tweet Screenshot](https://t.co/G4K2cJbY9C)
- **Advanced RAG becomes Easily Accessible**: The create-llama release now includes LlamaPack for advanced RAG concepts, allowing full-stack web app implementation in just two lines of code. Community-contributed modules simplify the integration of advanced RAG features as highlighted in [the announcement](https://t.co/vf0aKDv1yo). ![Tweet Screenshot](https://t.co/xxU3r8IAgg)
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1210131181463408680) (150 messages🔥🔥): 

- **Querying RAG in QueryPipeline**: `@lapexer` asked about implementing a simple RAG in QueryPipeline with modules `prompt`, `retriever`, and `llm`. `@cheesyfishes` directed them to the documentation for guidance and an overview ([How to write a simple RAG](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting)).

- **IngestionPipeline Integration Challenges**: `@emmepra` faced `ValidationError` issues while deploying the IngestionPipeline using ChromaVectorStore and TextEmbeddingsInference in Docker services. After multiple iterations and community support, especially from `@whitefang_jr` and `@cheesyfishes`, they resolved the issue by using consistent import paths between core and legacy modules.

- **LlamaIndex Import Inconsistencies**: Users like `@pymangekyo` and `@oopskapootz` reported inconsistencies and errors relating to module imports in LlamaIndex's new version. It was suggested by `@whitefang_jr` and `@cheesyfishes` to reinstall LlamaIndex and create a new environment if prior versions were installed to resolve import issues (e.g., `pip uninstall llama-index` and `pip install llama-index`).

- **LlamaParse Enterprise Deployment Possibilities**: `@self.1` inquired about the possibility of LlamaParse being open-source or self-hostable considering privacy concerns. `@cheesyfishes` pointed out that enterprise deployments are being considered but are not yet available.

- **Strategies for RAG Response Consistency**: `@a3lita` sought advice to improve the reliability of responses in RAG, specifically questioning the settings around LLM temperature. `@kapa.ai` explained several techniques such as prompt optimization, evaluation and benchmarking, context augmentation, and multi-modal evaluation to address this issue.

**Links mentioned**:

- [no title found](http://localhost:8001',): no description found
- [no title found](http://localhost:8000",>): no description found
- [no title found](http://localhost:8000">): no description found
- [Loading Data (Ingestion) - LlamaIndex 🦙 v0.10.12](https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html#using-readers-from-llamahub),): no description found
- [no title found](https://llamahub.ai/l/readers/llama-index-readers-database): no description found
- [no title found](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers): no description found
- [LangChain Embeddings - LlamaIndex 🦙 v0.10.12](https://docs.llamaindex.ai/en/stable/examples/embeddings/Langchain.html): no description found
- [LlamaIndex 🦙 0.9.15.post2](https://docs.llamaindex.ai/en/v0.9.15.post2/): no description found
- [TikTokLive v6.0.0](https://isaackogan.github.io/TikTokLive/): no description found
- [An Introduction to LlamaIndex Query Pipelines - LlamaIndex 🦙 v0.10.12](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting): no description found
- [Survey on your Research Journey](https://forms.gle/8N4DsuCWtCXKxLSv6): In an effort to revolutionize academic and business research, EurekAI seeks your insights to tailor our tool to your needs. Whether you&#39;re immersed in research or engage with it sporadically, your...

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1210430742207795200) (3 messages): 

- **In Search of Code Invocation Models**: `@gooooooofy` inquired about models or finetunes capable of generating code invocations like python scripts or shell commands with the correct arguments.
- **Gorilla LLM Almost Fits the Bill**: `@gooooooofy` mentioned that **Gorilla LLM** is similar to what they need but noted that it specializes in API calls and appears to be a smaller model.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1210490370270429245) (16 messages🔥): 

- **A Steal on GPUs for Deep Learning**: `@andreaskoepf` scored three RTX 3090s for 1.7k euros, aiming to convert a mining rig for deep learning tasks, especially for fine-tuning and serving large language models (LLMs). They outlined the specifications and considered a significant deal given the prevailing prices. [Part 1](https://samsja.github.io/blogs/rig/part_1/) and [Part 2](https://samsja.github.io/blogs/rig/part_2/) of their blog detail the transformation process to a deep learning rig.
  
- **Jim Keller’s Critique of CUDA**: User `@itali4no` shared a [link](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too) where Jim Keller criticized Nvidia's CUDA, calling it "a swamp" like x86, for being cumbersome and not beautifully constructed, evolving through the addition of multiple functionalities.
  
- **The Preferable GPU for Deep Learning**: A discussion about choosing GPUs for deep learning had `@iron_bound` pointing out the advantages of used 3090s over the new 4060 ti, mainly due to better memory bandwidth and PCIe support. Meanwhile, `@cropinky.` mentioned that the 4060 ti's 16GB VRAM is usually insufficient for LLM tasks.

- **Quantized Model Computations Explained**: `@andreaskoepf` explained that quantized models perform matrix multiplications at a higher internal resolution, and provided [GitHub links](https://github.com/TimDettmers/bitsandbytes) illustrating the dequantization process.

- **Guidance on Buying Second-Hand GPUs**: In response to queries on purchasing second-hand GPUs for deep learning, `@cropinky.` advised that it's a risk but suggested stress testing the GPU, checking for fan wear, and replacing thermal components as necessary for maintaining performance.

**Links mentioned**:

- [Jim Keller criticizes Nvidia's CUDA, x86 &mdash; 'Cuda&rsquo;s a swamp, not a moat. x86 was a swamp too'](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too): Jim Keller is not exactly a fan of Nvidia's CUDA.
- [Building a deep learning rig | part-1 - Samsja](https://samsja.github.io/blogs/rig/part_1/): no description found
- [Building a deep learning rig | part-2 - Samsja](https://samsja.github.io/blogs/rig/part_2/): no description found
- [Posts - Samsja](https://samsja.github.io/blogs): no description found
- [GitHub - TimDettmers/bitsandbytes at 5d6dfe6fb43e5aae277ec86cba20a002b34df705](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a0): Accessible large language models via k-bit quantization for PyTorch. - GitHub - TimDettmers/bitsandbytes at 5d6dfe6fb43e5aae277ec86cba20a002b34df705
- [bitsandbytes/bitsandbytes/functional.py at 5d6dfe6fb43e5aae277ec86cba20a002b34df705 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a002b34df705/bitsandbytes/functional.py#L1686-L1691): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes
- [bitsandbytes/csrc/kernels.cu at 5d6dfe6fb43e5aae277ec86cba20a002b34df705 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a002b34df705/csrc/kernels.cu#L3597-L3604): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1210312045166198854) (2 messages): 

- **Triton's Role in Education and Deployment**: `@_hazler` inquired whether integrating with Triton offers any advantages in speed or deployment platforms. `@srush1301` answered that it was primarily an educational undertaking, although it also enables Jax support via Pallas and offers a simplified version for researchers to modify.
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1210141090028720129) (17 messages🔥): 

- **Catching Up with CUDA Profiling**: `@dvruette` battled through installation errors and is now exploring `ncu` to delve into *low-level CUDA profiling*.

- **Open Repository for CUDA-Accelerated BnB**: `@zippika` announced their new GitHub repository [torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4), which hosts their faster alternative to bitsandbytes with minor differences in output and requires **cuda compute >= 8.0**.

- **Touting Token Speed Triumphs**: `@zippika` highlighted a significant speed boost achieved by their library, showcasing a performance jump from **24 tokens/s to a max of 29 tokens/s**.

- **Test Script to Benchmark BnB Performance**: `@zippika` shared a detailed [Python script](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L3533-L3649) for comparing the performance of the default bitsandbytes and their own torch-bnb-fp4 library; to execute the test, users need to toggle `USE_LINEAR_HIJACK` and have at least **12.8GB of VRAM** available.

- **Code Improvements and Community Engagements**: `@zippika` referencedmodifications made to CUDA 'gemv' kernels for optimization and expressed a commitment to enrich the repository with more examples and thorough documentation; meanwhile, `@_t_v_i_` expressed enthusiasm for the work done.

**Links mentioned**:

- [GitHub - aredden/torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4): Contribute to aredden/torch-bnb-fp4 development by creating an account on GitHub.
- [bitsandbytes/csrc/kernels.cu at e820409c095ea7cbb5ce156992307b84352cbf90 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L832-L896): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes
- [bitsandbytes/csrc/kernels.cu at e820409c095ea7cbb5ce156992307b84352cbf90 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L3533-L3649): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1210690041974624266) (2 messages): 

- **Exploring the World of Random Kernels**: `@hdcharles_74684` discussed the challenges of making random kernels accessible, particularly the clunky release of `int_mm` through `out_dtype`. They referenced the [pytorch/_higher_order_ops/out_dtype.py](https://github.com/pytorch/pytorch/blob/ed0ea2f30b2f31be7534a7fdafbed90d247f76b5/torch/_higher_order_ops/out_dtype.py#L107) and their work on a 4-bit triton kernel in [torch/_inductor/fx_passes/post_grad.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py#L241-L274).

- **Torch Compile's Kernel Integration Limits**: `@hdcharles_74684` pointed out that `torch.compile` struggles with operations needing custom kernels that differ from existing ones, particularly for GPUs. They mentioned an intention to improve kernel access, such as adding weight-only int8 quantization for batch sizes larger than one.

**Links mentioned**:

- [pytorch/torch/_higher_order_ops/out_dtype.py at ed0ea2f30b2f31be7534a7fdafbed90d247f76b5 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/ed0ea2f30b2f31be7534a7fdafbed90d247f76b5/torch/_higher_order_ops/out_dtype.py#L107)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/torch/_inductor/fx_passes/post_grad.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py#L241-L274): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1210135471230033990) (1 messages): 

- **Gemini 1.5 Discussion Session Announced**: `@shashank.f1` is hosting a discussion on **Gemini 1.5**, welcoming everyone to join live. The link to a past session titled "A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms" is provided with a [YouTube video](https://youtu.be/FgcN62LFzIU).

**Links mentioned**:

[A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms](https://youtu.be/FgcN62LFzIU): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1210136760097439784) (1 messages): 

- **SIXT is hiring ML Engineer in Munich**: `@ppeter0480` posted a job opening for an ML Engineer at **SIXT in Munich**. The role requires knowledge and skills in **NLP and Generative AI**, and solid engineering abilities. Interested parties can apply through the provided [SIXT job listing](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472).

**Links mentioned**:

[Apply now: Senior Machine Learning Engineer (m/f/d) | Munich](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472): The job of your dreams in Munich: Senior Machine Learning Engineer (m/f/d). Join the SIXT team! We are looking forward to your application!

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1210142923404873729) (3 messages): 

- **Politeness Reigns Supreme**: Users `@0ut0f0rder` and `@dpearson` exchanged pleasantries, appreciating each other's helpfulness and agreeing on the importance of learning.
- **Seeking Help with OpenCV in Google Colab**: `@dpearson` is utilizing **Google Colab's GPUs** to run C/C++ code with 'nvcc4jupyter' but is facing issues with not being able to include `<opencv2/opencv.hpp>`. They are looking for a solution or an alternative to test their `colorToGrayscaleConverter` function on an image.
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 messages): 

marksaroufim: Lecture 6 on youtube
https://www.youtube.com/watch?v=hIop0mWKPHc
  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1210229022051536926) (11 messages🔥): 

- **AMD GPUs Lack Support for FA2**: `@mrrational` and `@iron_bound` both reported issues running FA2 training on their AMD GPUs, specifically the 7900xtx, even with the triton version for `@iron_bound`. The backward function/kernel appears to be missing, causing failures.

- **Potential Solution for Backward Function**: `@_t_vi_` suggested using [Triton-autodiff on GitHub](https://github.com/srush/triton-autodiff) to help `@iron_bound` get the backward kernel for FA2 training on an AMD GPU; however, `@srush1301` clarified it would still require adjustments as it mainly differentiates mathematical functions.

- **Limited AMD PyTorch Support for FAv2**: `@drisspg` informed the channel that AMD has added some limited FAv2 support in PyTorch's nightly builds, but `@iron_bound`'s subsequent error message indicates that the 7900xtx GPU isn't supported yet, as it's expecting gpu architecture gfx90a and not gfx11.

- **Further Clarification on GPU Architecture**: `@iron_bound` explained the architectural differences between the AMD GPUs, noting that the 7900 series targets "wave32" while data-center cards support "wave64." He also mentioned that AMD developers are currently focused on their mi300 product, signaling that lower-priority support issues may not be addressed promptly.

- **Exploring Code for Having Wave Matrix Multiplication (WMMA)**: `@iron_bound` shared a goal to potentially create a kernel targeting WMMA by referencing the code [from the flash-attention GitHub repository](https://github.com/ROCm/flash-attention/blob/b28f18350af92a68bec057875fd486f728c9f084/csrc/flash_attn_rocm/src/device_gemm_trait.hpp#L42), as RDNA architecture supports WMMA in contrast to data-center cards that use XDL.

**Links mentioned**:

- [flash-attention/csrc/flash_attn_rocm/src/device_gemm_trait.hpp at b28f18350af92a68bec057875fd486f728c9f084 · ROCm/flash-attention](https://github.com/ROCm/flash-attention/blob/b28f18350af92a68bec057875fd486f728c9f084/csrc/flash_attn_rocm/src/device_gemm_trait.hpp#L42): Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.
- [GitHub - srush/triton-autodiff: Experiment of using Tangent to autodiff triton](https://github.com/srush/triton-autodiff): Experiment of using Tangent to autodiff triton. Contribute to srush/triton-autodiff development by creating an account on GitHub.
- [GitHub - ROCm/flash-attention at howiejay/navi_support](https://github.com/ROCm/flash-attention/tree/howiejay/navi_support/): Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1210168146623725588) (46 messages🔥): 

- **Exploration of Facebook's Xformers**: `@jeremyhoward` provided a link to **Xformers**' FMHA initializations on GitHub with a particular focus on [line 417](https://github.com/facebookresearch/xformers/blob/99ad1723b0b80fb21c5e4dc45446e93752f41656/xformers/ops/fmha/__init__.py#L417), spotlighting their repository as a subject of interest.

- **PyTorch Forum Discussion on Equivalence of JAX Lax Scan**: `@andreaskoepf` shared a PyTorch forum discussion that appears to be asking about an equivalence in PyTorch for JAX's `lax.scan`. The link includes CSS styling details, likely extracted from the webpage.

- **Introducing Ring Attention PyTorch Implementations**: `@ericauld` and `@iron_bound` introduced GitHub repositories for ring attention, [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) and [exists-forall/striped_attention](https://github.com/exists-forall/striped_attention), which explore the Ring Attention concept from Berkeley AI and striped attention codes respectively.

- **Benchmarks and Implementation Discussions**: `@iron_bound` presented their own [ring-flash-attention benchmarks](https://github.com/Iron-Bound/ring-flash-attention/blob/349ea8c41d430d28810dd5419ebdca51e9f57e64/benchmark.py#L135), including performance figures for different settings, while `@zhuzilin96`, the author of one of the discussed repos, joined the conversation, offering insights and mentioning the need for testing and enhancements such as support for returning fp32 outputs and arbitrary mask handling.

- **Collaboration Offers and Ongoing Improvements**: `@andreaskoepf` and others offered to team up with `@zhuzilin96` to further develop and optimize the ring attention implementations, with specific focus on testing, striped attention, and handling issues such as arbitrary masking for better flexibility of the models. All the while, `@zhuzilin96` has been pushing commits for improvements like `zigzag_ring_flash_attn_varlen_qkvpacked_func`.

**Links mentioned**:

- [Is there an equivalent of jax.lax.scan (eg in torch.func)?](https://discuss.pytorch.org/t/is-there-an-equivalent-of-jax-lax-scan-eg-in-torch-func/177088): I would like to translate the following jax code (that implements a Kalman filter) to torch.  def kf(params, emissions, return_covs=False):     F, Q, R = params[&#39;F&#39;], params[&#39;Q&#39;], para...
- [flash-attention/flash_attn/flash_attn_triton.py at main · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [flash-attention/flash_attn/flash_attn_triton.py at 87a1277653fc55cd615f5341255e00c69d5c00a1 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/87a1277653fc55cd615f5341255e00c69d5c00a1/flash_attn/flash_attn_triton.py#L211C13-L211C43): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [ring-attention/ring-transformer at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/tree/main/ring-transformer): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [ring-flash-attention/benchmark.py at 349ea8c41d430d28810dd5419ebdca51e9f57e64 · Iron-Bound/ring-flash-attention](https://github.com/Iron-Bound/ring-flash-attention/blob/349ea8c41d430d28810dd5419ebdca51e9f57e64/benchmark.py#L135): Ring attention implementation with flash attention - Iron-Bound/ring-flash-attention
- [xformers/xformers/ops/fmha/__init__.py at 99ad1723b0b80fb21c5e4dc45446e93752f41656 · facebookresearch/xformers](https://github.com/facebookresearch/xformers/blob/99ad1723b0b80fb21c5e4dc45446e93752f41656/xformers/ops/fmha/__init__.py#L417): Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [ring-attention/ring-transformer/main.py at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/ring-transformer/main.py): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [GitHub - exists-forall/striped_attention](https://github.com/exists-forall/striped_attention): Contribute to exists-forall/striped_attention development by creating an account on GitHub.
- [[Feature Request] Balancing computation with zigzag blocking · Issue #2 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/issues/2): Currently the implementation will split the input sequence into n blocks, e.g. 4 gpu will split into: b0 | b1 | b2 | b3 however, this will result in uneven calculation, where the gpu that has b3 wi...
- [GitHub - bigscience-workshop/petals: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading](https://github.com/bigscience-workshop/petals): 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading - bigscience-workshop/petals
- [transformer_nuggets/transformer_nuggets/flash/flash_attention.py at 4036b4385feaf610edf35b09b97cd14cba4ce701 · drisspg/transformer_nuggets](https://github.com/drisspg/transformer_nuggets/blob/4036b4385feaf610edf35b09b97cd14cba4ce701/transformer_nuggets/flash/flash_attention.py#L52.): A place to store reusable transformer components of my own creation or found on the interwebs - drisspg/transformer_nuggets
- [Add Striped Attention extension. · exists-forall/striped_attention@0c3ef0f](https://github.com/exists-forall/striped_attention/commit/0c3ef0f02541f7004c6cfb51ad305e92f1e01d29): no description found
- [Custom attention bias by b-albar · Pull Request #617 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/pull/617): This PR is an attempt to add custom (additive) attention biases. This is still very much a work in progress to say the least but I though to make my code available as there may be a lot of interest...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1210168890349584404) (70 messages🔥🔥): 

- **Feedback Request for Research Tool**: `@d97tum` shared a [survey link](https://forms.gle/8N4DsuCWtCXKxLSv6) to gather feedback for a product he is developing that addresses common research problems, such as finding relevant research papers and comprehending complex studies. He hopes the community's insights will shape the product's features.
- **Need for Langchain Consultant**: `@cybersmiths` is looking for a consultant skilled in **Langchain** and **OpenAI's tool agent** to assist with their efforts, and is willing to offer compensation for the help. This opportunity is directed to the LangChain AI Discord community.
- **Technical Discussions on Optimizing Chains**: `@b0otable` initiated a deep dive into how to better optimize chains in LangChain, focusing on using **RunnableParallel** and **RunnablePassthrough** to maintain the input query while running multiple chains in parallel and retaining the outputs at the root level of a dict-like output.
- **API Calls and Streaming in LangChain**: `@critical3645`, `@saita_ma_`, and `@edartru.` brought up questions about implementing streaming in **agent_supervisor**, calling local models like **OpenHermes**, and the applicability of certain tools with streams, highlighting the technical nuances of working with LangChain tools and integrations.
- **LangSmith Debugging and Visualization Tool**: `@b0otable` shares his experience with **LangSmith** for debugging complex LangChain processes, recommending it as a way to ensure chains behave as expected and offering a brief guide on setting it up for new users.

**Links mentioned**:

- [🦜🕸️LangGraph | 🦜️🔗 Langchain](https://python.langchain.com/docs/langgraph): ⚡ Building language agents as graphs ⚡
- [langgraph/examples/multi_agent/agent_supervisor.ipynb at main · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb): Contribute to langchain-ai/langgraph development by creating an account on GitHub.
- [Create Chat UI Using ChainLit, LangChain, Ollama &amp; Gemma 🧠](https://youtu.be/n9AMtXLveMs): In this video, I am demonstrating how you can create a simple ChatGPT like UI locally in your computer. You can follow along with me by cloning the repo loca...
- [Survey on your Research Journey](https://forms.gle/8N4DsuCWtCXKxLSv6): In an effort to revolutionize academic and business research, EurekAI seeks your insights to tailor our tool to your needs. Whether you&#39;re immersed in research or engage with it sporadically, your...
- [LangGraph: Intro](https://www.youtube.com/watch?v=5h-JBkySK34&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg): In this video we will introduce LangGraph - a way to more easily create agent runtimes.GitHub Repo: https://github.com/langchain-ai/langgraph
- [Self-reflective RAG with LangGraph: Self-RAG and CRAG](https://www.youtube.com/watch?v=pbAd8O1Lvm4): Self-reflection can greatly enhance RAG, enabling correction of poor quality retrieval or generations. Several recent RAG papers focus on this theme, but imp...

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1210205242927611975) (3 messages): 

- **Parallel Function Calls Now Available**: `@gokusan8896` announced a method to enable **parallel function calls in any Large Language Model (LLM)**. The details were shared in a [LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7166408137002962944/).
- **Seeking Feedback on Aggregate Query Platform**: `@rogesmith` is developing a platform/library for **aggregate document data queries** and is considering making it public, soliciting community feedback on its usefulness.
- **Guide to Building Custom Chatbots**: `@deadmanabir` released a comprehensive guide on how to create custom chatbots incorporating chat history using OpenAI, Qdrant DB, and Langchain JS/TS SDK. For more information and feedback opportunities, check out their [Twitter post](https://twitter.com/ItsDutta99/status/1761064358321525235).
  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1210600621854425099) (3 messages): 

- **Introducing Chat UI with ChainLit and Friends**: A YouTube video demonstrating how to create a ChatGPT-like UI locally using **ChainLit, LangChain, Ollama & Gemma** was shared. The video can be watched [here](https://youtu.be/n9AMtXLveMs), where viewers can clone the repository and follow along to set up their own chat interface.

- **Stock Analysis via LLMs**: `@rito3281` published an article discussing how **Large Language Models (LLMs)** can assist in understanding a company's quarterly reports to predict future growth, risk, and market opportunities. The detailed post and a demonstration of a Stock Portfolio Summarizer app can be found [here](https://rito.hashnode.dev/daily-portfolio-summarizer-with-langchain-qdrant-and-mistral-ai).

- **Google Colab and Ollama Meet**: `@schimazing` announced an adaptation that uses **Ollama's** new embeddings, fully hosted on Google Colab without the need for API keys. More information is available in the linked [Twitter post](https://twitter.com/theReedTard/status/1761107453465252120?s=19).

**Links mentioned**:

- [Daily Portfolio Summarizer with Langchain, Qdrant, and Mistral AI](https://rito.hashnode.dev/daily-portfolio-summarizer-with-langchain-qdrant-and-mistral-ai): Today&#x27;s Investors are bombarded with news, reports, statistics, and more information. AI cuts through this noise, analyzes vast datasets to unearth hidden patterns and trends, and offers insights...
- [Create Chat UI Using ChainLit, LangChain, Ollama &amp; Gemma 🧠](https://youtu.be/n9AMtXLveMs): In this video, I am demonstrating how you can create a simple ChatGPT like UI locally in your computer. You can follow along with me by cloning the repo loca...

  

---



### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1210275499209662494) (19 messages🔥): 

- **Codespaces for LLM Playground**: `@derekpwillis` shared a [template repository](https://github.com/dwillis/llm-codespaces) to run LLM in codespaces, finding it effective for `orca-mini-3b` while expressing concern about the support for larger models.
- **Positive Feedback on Codespaces Configuration**: `@simonw` praised the barebones `.devcontainer` configuration in the codespace template and found it to be highly useful as an example. The same user also noted a long startup time, which seemed to involve compiling many components from scratch.
- **Untangling a Codespaces Quirk**: `@simonw` encountered a bug where `llm-gpt4all` was not recognized as available initially but worked after running `llm models`. He suggested using `llm chat -m orca-mini-3b-gguf2-q4_0` to keep the model in memory for faster follow-up messages.
- **Prompt Crafting Versus Direct Query**: `@tariqali` compared an old-school prompt crafting that gives more control to modern, more direct queries with LLM (like RLHF), noting the usefulness of the former approach in specific circumstances, such as resuming conversations with new chatbot instances.
- **Exploring Larger World Model Integration**: `@simonw` expressed interest in running the [Large World Model's LWM-Text-1M-Chat](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M), which may require a GPU instance for pytorch models considering it's trained on a large dataset.

**Links mentioned**:

- [Large World Models](https://largeworldmodel.github.io/): no description found
- [LargeWorldModel/LWM-Text-Chat-1M · Hugging Face](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M): no description found
- [llm-codespaces/.devcontainer/devcontainer.json at main · dwillis/llm-codespaces](https://github.com/dwillis/llm-codespaces/blob/main/.devcontainer/devcontainer.json): A template repository for using the Python llm library in codespaces - dwillis/llm-codespaces
- [GitHub - dwillis/llm-codespaces: A template repository for using the Python llm library in codespaces](https://github.com/dwillis/llm-codespaces): A template repository for using the Python llm library in codespaces - dwillis/llm-codespaces

  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1210624176927281162) (5 messages): 

- **Richard Socher hints at a solution to AI hallucination**: `@res6969` shared a [tweet by @RichardSocher](https://x.com/RichardSocher/status/1760800655428796772?s=20) suggesting that significant progress might have been made in addressing the issue of AI hallucination, showing up-to-date references with no errors.

- **Wondering about the Wizardry Behind Non-hallucinatory AI**: `@res6969` speculated that to prevent hallucinations, the AI might be utilizing some **state-of-the-art embeddings** along with an instructional validator.

- **Globe Explorer: Your Personalized Wikipedia**: `@sincethestudy` shared a [tweet](https://x.com/sincethestudy/status/1761099508853944383?s=20) about a new platform called **Globe Explorer**, which acts as an on-demand custom Wikipedia page powered by GPT-4, marking an evolution in how we discover information. Visit the tool at [explorer.globe.engineer](http://explorer.globe.engineer/).

- **GPT-4 Powers New Discovery Engine**: `@sincethestudy` announced the launch of **Globe Explorer**, a discovery engine that uses **GPT-4** as its backend, paving the way for enhanced information discovery experiences.

**Links mentioned**:

- [Tweet from brian-machado-finetuned-7b (e/snack) (@sincethestudy)](https://x.com/sincethestudy/status/1761099508853944383?s=20): Globe Explorer is kinda like a custom wikipedia page on anything you want.  We are entering a new age of information discovery.  go try it: http://explorer.globe.engineer/
- [Tweet from Richard Socher (@RichardSocher)](https://x.com/RichardSocher/status/1760800655428796772?s=20): Did we solve the hallucination problem? It is starting to look like it here and in any other example I&#39;ve tried in research mode - all with tons of up-to-date references.  Query: Reddit S-1

  

---


### LLM Perf Enthusiasts AI ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/1210641609704865793) (1 messages): 

- **Finetuning Dilemma with GPT-4-Turbo**: User `@pantsforbirds` has successfully been embedding entire documents for 1-shot data extraction using **gpt-4-turbo** and is contemplating finetuning for more complex tasks. They inquire whether the finetuning dataset should include entire example documents or just relevant sections.
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1210697163684978788) (4 messages): 

- **Globe Explorer Sparks Information Discovery Excitement**: `@joshcho_` shared a link to [Globe Explorer](http://explorer.globe.engineer/), highlighting it as akin to a custom Wikipedia page. They remarked on entering a **new age of information discovery**.
- **Discovery Spreads Beyond Original Post**: `@nosa_` followed up by pointing to a previous [Discord conversation](https://discord.com/channels/1168579740391710851/1168579740391710855/1210667324995145728) where `@sincethestudy` had already introduced Globe Explorer.
- **Viral Before Official Spread Attempt**: `@joshcho_` humorously noted that the Globe Explorer already **went viral** before the call to spread the word was even seen.

**Links mentioned**:

[Tweet from brian-machado-finetuned-7b (e/snack) (@sincethestudy)](https://x.com/sincethestudy/status/1761099508853944383?s=46): Globe Explorer is kinda like a custom wikipedia page on anything you want.  We are entering a new age of information discovery.  go try it: http://explorer.globe.engineer/

  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1210434355323871292) (2 messages): 

- **Setting Up Max Token Limit**: `@ayushsharma` mentioned the need to **set the max_token_limit in the constructor**, but provided no further details or context around this request.
- **Prompting LLMs for Non-Overlapping Grid Components**: `@firefox8975` inquired about writing prompts to organize different-sized components into a grid without overlap and questioned the LLM’s effectiveness at such spatial tasks. They sought advice on ensuring components do not overlap within an **X by Y grid**.
  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1210174238657486928) (2 messages): 

- **GLAN Paper Spotted**: User `@.benxh` shared a link to a paper on **GLAN** (Generative Latent Nearest Neighbors), asking if anyone is working with it. They included the [research paper](https://arxiv.org/pdf/2402.13064.pdf) for reference.
- **Interest in GLAN Expressed**: `@entropi` responded with interest to the mention of GLAN, indicating that they found the shared paper on the [Generative Latent Nearest Neighbors](https://arxiv.org/pdf/2402.13064.pdf) algorithm intriguing.
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=953U3FxHF-Q
  

---




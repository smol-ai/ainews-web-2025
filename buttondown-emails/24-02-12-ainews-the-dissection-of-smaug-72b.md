---
id: 28549af9-0301-4f3f-a4d8-369a8359538c
title: '[AINews] The Dissection of Smaug (72B)'
date: '2024-02-13T01:40:29.456403Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-to-be-named-3759
---

<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/8-10/2024. We checked **20** guilds, **311** channels, and **6143** messages for you. Estimated reading time saved (at 200wpm): **528 minutes**.

---

It's now the Chinese year of the Dragon, and Abacus AI appropriately rung it in [making a lot of noise](https://twitter.com/bindureddy/status/1754665925834690907) about Smaug 72B, their latest and largest finetune of Qwen (1.0... badly timed since 1.5 just came up, but you can be sure they will update it with more noise)

 ![image.png](https://assets.buttondown.email/images/0c797830-2124-4784-8192-0d9ac26d35d1.png?w=960&fit=max) 

Typical skepticism aside, it is still standing unchallenged after a week on [the HF Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and with [published contamination results](https://huggingface.co/abacusai/Smaug-72B-v0.1#contamination-results), which is a good sign. However the Nous people are skeptical:

 ![image.png](https://assets.buttondown.email/images/6bbd8c06-2005-4897-bfb0-04014e81e659.png?w=960&fit=max) 

In other news, LAION popped up with [an adorably named local voice assistant model](https://laion.ai/blog/bud-e/) with a great demo.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Size Matters in Model Performance**: Community members debated the cost versus performance of large models like GPT-4 and explored alternatives like Janitor AI for cost-effective chatbot solutions. The potential effectiveness of smaller models quantized at higher levels was also discussed, but consensus suggests larger models may handle heavy quantization better, though this relationship isn't strictly linear.

- **Good AI Turns Goody-Two-Shoes**: Conversations noted the safety-focused AI model "Goody-2," acknowledging its extreme caution against controversial content, which sparked some playful ideas about challenging the model's stringency.

- **Web UI Woes and Wins**: Collaborative discussions featured progress in UI development projects such as PolyMind and others, with members discussing the intricacies of web development and prompt engineering.

- **Missing Models and Merging Mysteries**: Queries for fine-tuned Mistral-7b models for merging projects surfaced, alongside musings on the conspicuous absence of a member likened to awaiting updates, humorously compared to the expectation for Llama3 to support vision.

- **Model Merging Muscles Flexed**: The community saw lively exchanges on model merges like Miquella and the potential performance of ERP-focused models like MiquMaid, showcasing enthusiasm for fine-tuning these AI models to specific tasks, while members also sought advice on setup configurations, such as context length and memory allocation across GPUs.

- **Injected Instructiveness**: Interest in fine-tuning methodologies was evident with discussions around converting base models into instruct models by utilizing datasets like WizardLM_evol_instruct_V2_196k and OpenHermes-2.5. This process enriches models with added knowledge and alters the chatbot's tone, with members pointing to resources like [unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning.

- **Coding Corner Collaborations**: A shared Python script for Surya OCR hinted at the continued development and application of optical character recognition tools among members. Debugging help was sought for a webaudiobook repository with a possible service issue tied to ElevenLabs' platform performance highlighted as a potential culprit.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Model Conversion Mishap**: It was noted that an error occurred while running TheBloke's models, which is likely due to a broken quant during the PyTorch to gguf conversion process. Users also exchanged experiences and advice on hardware components and setups for running Large Language Models efficiently, with mentions of Intel Xeon CPUs, Nvidia P40 GPUs, and the importance of VRAM and CUDA cores.

- **LM Studio's Technological Underpinnings**: Discussions in the LM Studio community clarified that LM Studio uses llama.cpp and potentially Electron for its tech stack. There were also privacy concerns with LMStudio data usage, but it was indicated that the platform prioritizes privacy, only sharing data from updates and model downloads.

- **Image Generation Limitations**: Users shared their difficulty in using LM Studio for image generation, prompting some to consider VPN usage to circumvent ISP blockages affecting access to necessary resources like huggingface.co.

- **Web UI Woes**: The community sought but confirmed the absence of a web UI for LM Studio, adding a layer of complexity for some users.

- **Challenges with Small Model Classification**: Users `'@speedy.dev'` faced challenges with classification tasks using 13B and 7B Llama models. A comparison of story writing capabilities between various sized models showed Goliath 120B capturing emotions better while Mixtral 7B outperformed in speed.

- **Metadata Management and a 'model.json' Standard Proposed**: The value of proper model metadata management was highlighted, recommending a categorization system for different parameters, and a proposal for a `model.json` standard was posted on GitHub.

- **Intel's AVX-512 Decision Bewilders**: Intel's choice to drop AVX-512 support in a powerful i9 processor while maintaining it in the less powerful i5-11600k sparked confusion among users.

- **Preference for CrewAI Over AutoGen**: There was a brief mention indicating a preference for CrewAI over AutoGen due to ease of management concerns.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **HF Hub Temporarily Offline**: [@lunarflu](https://discord.com/channels/879548962464493619/897387888663232554/1205945710164574279) reported that the **Hugging Face Hub** is experiencing downtime, with git hosting affected. The issue is being addressed.

- **Discord Authentication and API Changes**: On Discord verification, user @d3113 solved a bot authentication error by enabling "Allow messages from server members." Meanwhile, [@miker1662](https://discord.com/channels/879548962464493619/879548962464493622/1205799064541593631) encountered an API bug related to finding conversational models on Hugging Face, with @osanseviero confirming the API's shift away from `conversational` towards `text-generation`.

- **Large Language Models (LLMs) in Focus**: Conversations included discussions on the **PEFT for LLM Fine-Tuning** versus full fine-tuning approaches, with @samuelcorsan choosing the latter, while @akvnn inquired about leveraging a Raspberry Pi for computer vision tasks and @pradeep1148 shared a [YouTube video on zero shot object detection](https://www.youtube.com/watch?v=W4T7zHluzaM).

- **Choice of Hardware for Local LLMs**: Dual NVIDIA 4080 Supers versus a single 4090 for coding LLMs was debated by @lookingforspeed and @tddammo, with older generation pro cards like the A4500 or A4000 suggested for better efficiency and NVLink support.

- **API Action with Library Selection and Innovation**: [@subham5089](https://discord.com/channels/879548962464493619/898619964095860757/1205807574042148894) shared insights on choosing the right Python library for API calling, highlighting **Requests, AIOHTTP, and HTTPX**. Websockets in Python were addressed with a recommendation for [httpx-ws](https://frankie567.github.io/httpx-ws/).

- **Developments in Computer Vision and NLP**: From discussions on tools like `Dinov2ForImageClassification` for simplifying multi-label image classification, to NLP-related issues such as the downgrading to **PEFT 0.7.1** for saving LoRA weights, the community engagement is rich with shared solutions and knowledge exchanges.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **AI Financial Advice Comes with Caution**: The limitations of AI in generating **trading strategies** were discussed with a sense of frustration due to AI's cautious responses and memory constraints when dealing with financial queries.

- **Innovation in AI-Powered Compression**: A tool named `ts_zip` was shared, which leverages **Large Language Models** for text file compression, sparking interest in AI-driven compression technologies. The tool can be examined at [ts_zip: Text Compression using Large Language Models](https://bellard.org/ts_server/ts_zip.html).

- **Deliberations on AI's Resource Extraction and Impact**: A mention of **5-7 trillion** investment for **AI chips** led to debates around the impact on AI research and development and societal risks linked with autonomous robotics.

- **Google's Gemini AI Draws Attention**: Google's **Gemini** was discussed with regards to its anticipated improvements and current pros and cons, capturing attention for its coding and interpreter functionalities.

- **ChatGPT Token Context Limitations Explored**: Issues with **ChatGPT's attention span** and context retention were addressed, noting limitations like the full 128K token context available only to API and Enterprise users, contrasting with the 32K token limit for Plus and Team users.

- **GPT-4 Subscription Details Clarified**: Clarifications were made around subscription-sharing where it was noted that all GPT versions now utilize GPT-4, and thus a Plus or higher subscription is necessary for usage.

- **Navigating GPT-4's Conversational Flow and State Awareness**: Users discussed GPT-4's handling of "state changes" and noted its effectiveness in managing dynamic conditions and conversational flow, which is important when providing prompts that require multi-step problem-solving.

- **Effective Prompt Engineering Tactics Shared**: In the #prompt-engineering channel, strategies to instruct **ChatGPT** to perform complex tasks like converting rules of evidence into a spreadsheet and translating text into simple language were discussed. Simplifying tasks and using stepwise prompting were some of the recommended approaches.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Elon Musk's Alleged AI Antics**: Elon Musk was humorously linked to two separate AI-related events: one where he's rumored to be on a call with Alex Jones and one where he boasts about creating an AI with 200 GPUs. `@fullstack6209` shared a snippet that can be found [here](https://twitter.com/i/status/1756105704786817130) and quoted Musk's comment from February 9th, 2024. Moreover, Hugging Face (HF) was reported to be offline by `@gabriel_syme`.

- **Quantization and Merge Innovations in AI**: Senku 70B model quantized to 4-bit achieves significant scores without calibration, and `@carsonpoole` suggests mixing formats for better results. Meanwhile, `@nonameusr` introduced the QuartetAnemoi-70B model, and `@.benxh` discussed the potential of 1.5-bit quantization that fits a 70b model into less than 18GB of RAM. Links to QuartetAnemoi-70B model [here](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001), and 1.5-bit quantization [here](https://github.com/ggerganov/llama.cpp/pull/5453).

- **Datasets and Models Galore**: The engineering community discussed numerous models and datasets: UNA-SimpleSmaug-34B showed improvement over Smaug-34B and was trained on [SimpleMath](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta). Lilac Processed OpenHermes-2.5 dataset is available [here](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5). TheProfessor-155b, a model using [mergekit](https://github.com/cg123/mergekit), was also mentioned.

- **Community Exchanges on Finetuning and Hosting AI Models**: `@natefyi_30842` looked for simple fine-tuning methods, with `@teknium` recommending together.ai and noting the need for hyperparameter tuning. `@nonameusr` sought advice for hosting a model on Hugging Face for API inference.

- **Discussion on Autonomous LLMs and Platform Features**: Interest in autonomous large language models (LLMs) was expressed by `@0xsingletonly`, who is looking forward to Nous's SDK development mentioned by `@teknium`. Additionally, confusion about feature inclusions on the roadmap was clarified, with the Llava team's independent integration work pointed out by `@qnguyen3`.

- **Humor and Services Down**: Other notable mentions include a humorous reluctance from `@.ben.com` to take advice from individuals lacking emoji expertise, and the brief outage of Hugging Face's services as noted by `@gabriel_syme`.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **AI Newcomers Tackle EleutherAI and More**: A member with a background in software development and research papers showed interest in contributing to AI and GPT-J, amidst discussions of The Pile dataset and CUDA programming. The community explored Prodigy's VRAM-heavy performance, AI merging practices, and expressed concerns about the rise of questionable AI practices, along with a reference to OpenAI's new release potentially timed with the Super Bowl. Relevant resources include [The Prodigy optimizer](https://github.com/konstmish/prodigy), [miqu-1-120b on Hugging Face](https://huggingface.co/wolfram/miqu-1-120b), and [Microsoft's Copilot commercial](https://youtu.be/SaCVSUbYpVc?t=40). 

- **Nested Networks and Vector Confusions**: Enthusiasm was shown for nested networks in JAX, with helpful resources like a [Colab notebook](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq) for experimentation. Discussions also delved into the confusion over vector orientations in mathematics, while help was offered for implementing diffusion paper methods, with code shared in a [nbviewer gist](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57). For further reading, an aggregate of UniReps research was shared via [GitHub](https://github.com/UniReps/UniReps-resources).

- **Debating the Merits of Machine Unlearning Benchmarks**: Skepticism arose regarding the significance of the "TOFU" benchmark for unlearning sensitive data as detailed in the [TOFU paper](https://arxiv.org/abs/2401.06121). Concerns were raised about its efficacy and real-world applications, with participants also discussing a related [neuron pruning paper](https://arxiv.org/abs/2401.01814) which may illuminate the conversation.

- **Model Evaluation and Hallucination Tracking**: Questions about the MedQA benchmark suggested that Pythia models might struggle with multiple-choice formats. A search for comparative model API data, spanning OpenAI to Anthropic and the Open LLM Leaderboard, was highlighted. For tasks involving the GPQA dataset, warnings against potential data leakage were noted, seeking manual downloads [GPQA dataset paper](https://arxiv.org/abs/2311.12022). Clarifications were requested for evaluating Big Bench Hard tasks using GPT-4 models. A call for participation was made to a new hallucinations leaderboard explained on [Hugging Face's blog post](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations) and the associated [Harness space](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Circuit Integration Trumps Cascading in Voice Assistants**: The opinion was voiced that **Cascaded ASR + LLM + TTS** is less impressive compared to end-to-end training for voice AI, with the **BUD-E voice assistant** showcased as an example of integrated ASR, LLM, and TTS in a single PyTorch model, promoting end-to-end trainability ([Learn More About BUD-E](https://speechbot.github.io/spiritlm/index.html)).

- **Legal Tangles for AI Art Creators**: A recent court ruling saw *U.S. District Judge William Orrick* deny Midjourney and StabilityAI's motion for early dismissal under a First Amendment defense, sparking debate among users about the case's broader implications.

- **AI Community Grapples with Open-Source Ethics**: The AI community discussed the `sd-forge` project, which combines code from *diffusers*, *automatic1111*, and *comfyui*, yet tries to keep a distance from these projects amidst the evolving landscape for Stable Diffusion models and their open-source UI counterparts.

- **Creative Frontiers: AI-Generated DnD Maps**: Users have successfully used neural networks to create Dungeons and Dragons maps, reflecting the expanding capabilities of AI in creative endeavors.

- **Hugging Face Faces Hurdles**: There were reports of **Hugging Face**'s services experiencing downtimes. The conversation focused on the challenges of relying on external APIs and the need for robust alternatives to maintain smooth AI development operations.

- **An Open Voice Evolves**: **BUD-E** was introduced as an open-source, low-latency voice assistant designed to operate offline, with an invitation extended to the community to contribute to its further development ([Contribute to BUD-E](https://laion.ai/blog/bud-e/), [Join Discord](https://discord.com/invite/MDTpftKbpv)).

- **The Science of Loss in AI**: There was a query on Wasserstein loss in one of Stability AI's projects with a link to the GitHub repository, although no direct code pertaining to the claim was identified ([Discriminator Loss Mystery](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py)).

- **Stacking Talents and Scientific Insights**: A user showcased their full stack design and development skills, while another shared a scientific article without additional context. Additionally, there was a request for guidance on reproducing the **MAGVIT V2 model**, indicative of active research and development efforts within the community ([Shinobi Portfolio](https://shinobi8894.onrender.com/), [Check out MAGVIT V2](https://github.com/lucidrains/magvit2-pytorch)).

- **Introducing Agent Foundation Models**: The community was alerted to a paper on "**An Interactive Agent Foundation Model**" available on arXiv, suggesting a shift towards dynamic, agent-based AI systems ([Read the Abstract](https://arxiv.org/abs/2402.05929)).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Compare AIs Head-to-Head**: Users discussed **comparing different AI models** by opening multiple tabs; one such site for comparison includes [AIswers](https://www.aiswers.com/), where **Perplexity's performance** can be tested against others.

- **App Interaction Oversensitivity**: **Perplexity's iPad app** received criticism for overly sensitive thread exit functionality, and an inquiry about a **developers-oriented channel for Perplexity's API** resulted in a redirection to an existing Discord channel.

- **API Rate Limiting Quirks**: Some users faced a **429 HTTP status error** when using the Perplexity API through an App script, initially mistaking it for an OpenAI-related issue. The problem was resolved by **adding a millisecond delay** in the script; credits and limits on Perplexity can differ from those on OpenAI.

- **Model Features and Functions Inquiry**: There was a request for an update on **Mistral's 32k context length availability** from the [feature roadmap](https://docs.perplexity.ai/docs/feature-roadmap), as well as clarification that the **messages field** is required for all inputs with `mistral-8x7b-instruct` and that function calling isn't supported.

- **Ensure Search Results are Public**: **Perplexity users were reminded** to make sure threads are publicly viewable before sharing in the channel, which is designated for notable results obtained using Perplexity.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **H100 GPU as a Stepping Stone to AGI**: `@andreaskoepf` discussed the potential of the new **H100** GPU in achieving Artificial General Intelligence (AGI) when combined with the appropriate models and sufficient numbers of GPUs, referencing an [AI Impacts article](https://aiimpacts.org/brain-performance-in-flops/) with FLOPS estimates for human brain activity.

- **Learning from Stanford's AI Hardware Experts**: The community highlighted a [Stanford MLSys seminar by Benjamin Spector](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86) for engineers, which offers insights into AI hardware that might be relevant to the discourse on engineering forums.

- **Serverless Triton Kernel Execution**: `@tfsingh` announced the launch of [Accelerated Computing Online](https://acceleratedcomputingonline.com), a serverless environment for executing Triton kernels on a T4 GPU, and mentioned the project's GitHub repository ([tfsingh/aconline](https://github.com/tfsingh/aconline)) for further exploration.

- **CUDA Development Deep Dives**: Discussions centered around CUDA programming involved memory coalescing for performance enhancements, NVIDIA NPP for image operations, the nuances of numerical stability in fp16 matmuls, and best practices for independent development of CUDA-compatible extensions.

- **Multi-GPU Troubleshooting and FAISS Challenges**: `@morgangiraud` faced issues with incorrect data during direct device-to-device tensor copying in distributed matrix multiplication and sought collaborators having multi-GPU setups for verification, while `@akshay_1` dealt with errors embedding vectors in FAISS(colbert) that might stem from distributed worker timeouts.

- **CUDA MODE Lecture Sessions and Announcements**: Upcoming and past educational events such as "CUDA MODE Lecture 5: Going Further with CUDA for Python Programmers" sparked interest, with link sharing on platforms such as [Discord](https://discord.gg/6UQXQYZp) for community engagement and learning.

- **Engagement with Educational Content**: Community members, particularly `@smexy3`, showed eagerness for future instructional video content, especially those that will teach how to analyze optimization opportunities in reports, with the next video scheduled to be released on **March 2**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Awq Gguff Converter: Lightning Fast**: Users praised the **awq gguff converter** for its swift performance, calling it a "10/10 fast" conversion tool without specifying further details or links.

- **HuggingFace Troubles Spark Community Solutions**: During a HuggingFace outage, which affected even local training jobs, members discussed workarounds including downgrading to **version 0.7.1** and considering the use of alternative inference solutions like **TensorRT** for local inference.

- **Mixtral Quandary Resolved with Peft Upgrade**: An issue with **Mixtral's quantization process** was resolved by upgrading from **peft version 0.7.1** to **0.8.0**, with confirmation that the upgrade remedied the initial problems. LlamaFactory's adoption of ShareGPT format was noted, and discussions about naming conventions ensued without further conclusion.

- **Fine-Tuning Techniques and Efficiency in Focus**: The community exchanged tips on fine-tuning strategies, including generating Q/A pairs from historical datasets for chat models and seeking cost-effective methods such as using quantized Mixtral. Practical insights into training configurations for Mistral-7b-instruct were also shared with references to configuration files from **Helix** and the **axolotl GitHub repo**.

- **Resource Quest for Fine-Tuning Newbies Goes Unanswered**: A request for learning resources on fine-tuning went unanswered in the message history, highlighting a potential area for community support and knowledge sharing.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **LLMs Master Tabular Traversing**: A [new video tutorial](https://www.youtube.com/watch?v=L1o1VPVfbb0) released by `@jerryjliu0` highlights advanced **text-to-SQL orchestration**, essential for navigating and querying tabular data with language models.

- **Enhancing Understanding of Tabular Data**: Recent advancements have been shared around RAG systems, with a significant emphasis on multi-hop query capabilities, detailed in Tang et al.'s dataset for [benchmarking advanced RAG models](https://t.co/Hqx1KKOqYv). Alongside this, a [mini-course](https://t.co/BS0VkZjbZI) is available covering the construction of query pipelines that blend text-to-SQL with RAG, amplifying the QA over tabular data framework.

- **Innovating Video Content Interaction**: A **Multimodal RAG architecture** that synergizes OpenAI GPT4V with LanceDB VectorStore is enhancing video content interaction. *Video Revolution: GPT4V and LlamaIndex Unleashed* discusses this innovation and its potential, a must-read for those interested in the field, available [here](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30).

- **Explorations and Solutions in AI Context Management**: LlamaIndex community members have discussed practical applications such as using LlamaIndex for generating SQL metadata and the need for solutions like `SimpleChatStore` for maintaining chat continuity across multiple windows. The resolution for extracting keywords from text was suggested to involve prompt engineering.

- **Pricing and Availability Clarifications**: Questions about LlamaIndex's free and open-source nature and availability led to clarifications that it is indeed open source, with more details accessible on their [official website](https://www.llamaindex.ai/).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Scktlearn Struggles Call for Voice Support**: `@vithan` sought assistance with **scktlearn and pandas**, indicating the limitations of text communication and requesting a voice call with `@himalaypatel` for more effective troubleshooting.

- **LangChain Video Tutorial Drops**: A YouTube tutorial "Unlock the Power of LangChain: Deploying to Production Made Easy" was shared by `@a404.eth`, detailing the deployment process of a PDF RAG using LangChain and **UnstructuredIO** to **DigitalOcean** for production use. The video is accessible at [this link](https://youtu.be/CbBIwVxjdP8).

- **Open Source Selfie Project Needs Your Pics**: `@dondo.eth` introduced **Selfie**, an open source project working to improve text generation by utilizing personal data via an OpenAI-compatible API, with contributions and testing welcomed on their [GitHub](https://github.com/vana-com/selfie).

- **Intellifs Setting the Standard**: `@synacktra` announced the creation of **Intellifs**, a tool for local semantic search based on the aifs library, currently open for contributions on [GitHub](https://github.com/synacktraa/intellifs).

- **Your Art, AI’s Touch**: `@vansh12344` launched **ArtFul - AI Image Generator**, an app that uses AI models such as Kandinsky and DALL-E to create unique art, free to use with ad support, available on the [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful).

- **Merlinn's Magic Aid in Incident Resolution**: `@david1542` presented **Merlinn**, intended to aid teams in quickly resolving production incidents through support from an LLM agent and LangChain integration. More details can be found on the [Merlinn website](https://merlinn.co/).

- **Triform Appeals for Beta Test Cooks**: **Triform**, a new platform for hosting and orchestrating Python scripts with LangChain integration, was announced by `@igxot`. Users are invited to obtain a free permanent account through beta participation, with a sign-up link [here](https://triform.ai) and documentation [here](https://triform-docs.readthedocs.io/).

- **Automatic Object Detection Made Easy**: `@pradeep1148` shared a [YouTube tutorial](https://www.youtube.com/watch?v=W4T7zHluzaM) on using the MoonDream Vision Language Model for zero-shot object detection.

- **Chatting up Documents with AI Tools**: `@datasciencebasics` posted a [video guide](https://youtu.be/2IL0Sd3neWc) explaining the creation of a Retrieval Augmented Generation UI using ChainLit, LangChain, Ollama, & Mistral.
  
- **Playground Disabled in Production**: `@gitmaxd` discussed the possibility of **disabling the playground** on deployed LangChain AI endpoints using a specific code snippet, but received no responses to the inquiry.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

**One Size Fits All with Mistral's Subscription**: Users discussed the subscription model for the Mistral Discord chatbot, confirming it is a unified model with payment per token and scalable deployment, highlighted by @mrdragonfox; quantized models, such as those found on [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF), were also mentioned as requiring less RAM.

**GPU Economics: Rent vs. Own**: @i_am_dom analyzed the cost-effectiveness of Google GPU rentals versus owning hardware like **A100s 40GB**, suggesting that after 70000 computational units or about half a year of use, owning GPUs could be more economical.

**Docker Deployment Discussion**: A request for `docker_compose.yml` for deploying Mistral AI indicates ongoing discussions about streamlining Mistral AI setups as REST APIs in Docker environments.

**Fine-Tuning for Self-Awareness and Personal Assistants**: Fine-tuning topics ranged from installation success on Cloudfare AI maker to a lack of self-awareness in models, as noted by @dawn.dusk in relation to GPT-4 and Mistral; a [Datacamp tutorial](https://www.datacamp.com/tutorial/mistral-7b-tutorial) was recommended for learning use cases and prompts.

**Showcasing Mistral’s Capabilities**: @jakobdylanc’s Discord chatbot with collaborative LLM prompting feature supports multiple models including Mistral with a lean 200-line implementation, available on [GitHub](https://github.com/jakobdylanc/discord-llm-chatbot); additionally, Mistral 7b's note-taking prowess was spotlighted in an article at [Hacker Noon](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk), outperforming higher-rated models.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **TurboPuffer Soars on S3**: A new [serverless vector database called TurboPuffer](https://turbopuffer.com/) was discussed for its efficiency, highlighting that warm queries for 1 million vectors take around 10 seconds to cache. The conversation compared TurboPuffer and LanceDb, noting that TurboPuffer leverages S3, while LanceDb is appreciated for its open-source nature.

- **Podcast Ponders AI and Collective Intelligence**: An interview with Yohei Nakajima on the [Cognitive Revolution podcast](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/) was shared, discussing collective intelligence and the role of AI in fostering understanding across cultures.

- **AI as Google's Achilles' Heel**: A 2018 internal Google memo shared via [TechEmails](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww) indicating that the company viewed AI as a significant business risk sparked discussion, with its concerns continuing to be relevant years later.

- **ChatGPT's Impact on College Processes**: The trend of using ChatGPT for college applications was analyzed, citing a [Forbes article](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/) which pointed out potential red flags that may arise, such as the use of specific banned words that alert admissions committees.

- **Avoiding Academic Alert with Banned Words**: There was a suggestion to program ChatGPT with a list of banned words to prevent its misuse in academic scenarios, relating back to the discussion on college admissions and the overuse of AI detected via such words.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Hugging Face Service Disruption Ignites Community Debate**: Discussions arose as **Hugging Face** experienced downtime, with community members such as `_jp1_` recognizing the platform's integral role and revealing past considerations to switch to **Amazon S3** for hosting model weights and datasets, yet the convenience of HF's free services prevailed. `_jp1_` and `@philipmay` also pondered HF's long-term sustainability, floating concerns about possible future monetization and the impact on the AI research community.
  
- **Considerations on HF's Role as Critical Infrastructure**: The debate initiated by `@philipmay` questioned whether **Hugging Face** qualifies as critical infrastructure for the AI community, highlighting how pivotal external platforms have become in maintaining model operations.
  
- **Prospects of Pliable Monetization Plans**: `@philipmay` speculated on a scenario where **Hugging Face** might begin charging for model access or downloads, triggering a need for preemptive financial planning within the community.

- **A Whisper of Sparse Efficiency**: Without details, `@phantine` dropped hints about an algorithm leveraging sparsity for efficiency, with an intended link for further details which failed to resolve.

- **SPINning Around With German Language Models**: `@philipmay` brought up applying the **SPIN method** (self-play) to a **Mixtral model** in German, sharing the [SPIN technique's official GitHub repository](https://github.com/uclaml/SPIN) to spark additional conversation or perhaps experimentation.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Whispers of Upcoming OpenAI Launch**: `@res6969` dropped hints about a potential new **OpenAI release**, creating anticipation with a vague announcement expecting news **tomorrow or Tuesday**. Conversations sparked with `@.psychickoala` playfully inquiring, **"What is it haha"** but no concrete details were shared.

Please note that the other message from rabiat did not contain sufficient context or information relevant for a technical, detail-oriented engineer audience and thus was omitted from the summary.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Curiosity About Colleague's Activities**: @teknium inquired about the current endeavors of `<@748528982034612226>`.
- **Status Update on Mysterious Member**: @atlasunified informed that `<@748528982034612226>` has gone **off grid**, without further elaboration on their status.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1205802028987715644) (1251 messages🔥🔥🔥): 

- **Concerns About Model Sizes and Preferences**: Members like `@dao_li` shared their experiences with various AI models, discussing the costs and effectiveness of GPT-4 and alternatives like Janitor AI for chatbots. As they found GPT-4 expensive, other users suggested trying various small models for more cost-effective solutions.
- **Discussions on Quantization**: `@immortalrobot` asked about the trade-offs between low quantized larger models versus higher quantized smaller ones. The consensus, including input from `@kalomaze` and `@superking__`, seemed to be that larger models might handle heavy quantization better, but the relationship is not linear.
- **Jokes About "Goody-2"**: The discussion touched upon the safe AI model "Goody-2," with users like `@selea` remarking on its stringency, as it rejects anything that could be controversial. The conversation playfully explored the idea of challenging the model.
- **User Interface Development**: `@itsme9316` and `@potatooff` shared progress on their respective UI development projects with PolyMind and a new UI being built. They discussed the complexities and challenges of web development and prompt engineering.
- **Curiosities on Model Absence and Updates**: `@rombodawg` sought fine-tuned Mistral-7b models for a merge project, while `@kaltcit` humorously remarked on the cat-like absence of a user named turbca, likening it to waiting for Llama3 to support vision.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1115976636400148562/1206157809042063380): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [no title found](https://www.independent.co.uk/news/world/americas/daniel-beckwitt-trial-askia-khafra-death-nuclear-bunker-fire-bethesda-maryland-a8963161.html): no description found
- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/chat): Introducing a new AI model with next-gen ethical alignment. Chat now.
- [Unbabel/TowerInstruct-13B-v0.1 · Hugging Face](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1): no description found
- [Aligning LLMs with Direct Preference Optimization](https://www.youtube.com/watch?v=QXVCqtAZAn4): In this workshop, Lewis Tunstall and Edward Beeching from Hugging Face will discuss a powerful alignment technique called Direct Preference Optimisation (DPO...
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2): no description found
- [MrDragonFox/apple-ferret-13b-merged · Hugging Face](https://huggingface.co/MrDragonFox/apple-ferret-13b-merged): no description found
- [Rick Astley - Never Gonna Give You Up (Official Music Video)](https://www.youtube.com/watch?v=dQw4w9WgXcQ>): The official video for “Never Gonna Give You Up” by Rick Astley. The new album &#39;Are We There Yet?&#39; is out now: Download here: https://RickAstley.lnk.to/AreWe...
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1): no description found
- [Answer Overflow - Search all of Discord](https://www.answeroverflow.com): Build the best Discord support server with Answer Overflow. Index your content into Google, answer questions with AI, and gain insights into your community.
- [GitHub - apple/ml-ferret](https://github.com/apple/ml-ferret): Contribute to apple/ml-ferret development by creating an account on GitHub.
- [GitHub - daswer123/xtts-api-server: A simple FastAPI Server to run XTTSv2](https://github.com/daswer123/xtts-api-server): A simple FastAPI Server to run XTTSv2. Contribute to daswer123/xtts-api-server development by creating an account on GitHub.
- [GitHub - mzbac/mlx-llm-server: For inferring and serving local LLMs using the MLX framework](https://github.com/mzbac/mlx-llm-server): For inferring and serving local LLMs using the MLX framework - mzbac/mlx-llm-server
- [GitHub - Tyrrrz/DiscordChatExporter: Exports Discord chat logs to a file](https://github.com/Tyrrrz/DiscordChatExporter): Exports Discord chat logs to a file. Contribute to Tyrrrz/DiscordChatExporter development by creating an account on GitHub.
- [metavoiceio/metavoice-1B-v0.1 · Hugging Face](https://huggingface.co/metavoiceio/metavoice-1B-v0.1): no description found
- [nvidia/canary-1b · Hugging Face](https://huggingface.co/nvidia/canary-1b): no description found
- [Piper Voice Samples](https://rhasspy.github.io/piper-samples/): no description found
- [GitHub - metavoiceio/metavoice-src: Foundational model for human-like, expressive TTS](https://github.com/metavoiceio/metavoice-src): Foundational model for human-like, expressive TTS. Contribute to metavoiceio/metavoice-src development by creating an account on GitHub.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/111269072853191): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112690728531918948): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [GitHub - daveshap/Reflective_Journaling_Tool: Use a customized version of ChatGPT for reflective journaling. No data saved for privacy reasons.](https://github.com/daveshap/Reflective_Journaling_Tool/tree/main): Use a customized version of ChatGPT for reflective journaling. No data saved for privacy reasons.  - GitHub - daveshap/Reflective_Journaling_Tool: Use a customized version of ChatGPT for reflective...
- [GitHub - LAION-AI/natural_voice_assistant](https://github.com/LAION-AI/natural_voice_assistant): Contribute to LAION-AI/natural_voice_assistant development by creating an account on GitHub.
- [LoneStriker/HamSter-0.2-8.0bpw-h8-exl2 · Hugging Face](https://huggingface.co/LoneStriker/HamSter-0.2-8.0bpw-h8-exl2): no description found
- [Simpsons Homer Simpson GIF - Simpsons Homer simpson - Discover &amp; Share GIFs](https://tenor.com/view/simpsons-homer-simpson-gif-13518564799186478937): Click to view the GIF
- [GitHub - LostRuins/koboldcpp: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI](https://github.com/LostRuins/koboldcpp?tab=readme-ov-file#osx-and-linux-manual-compiling): A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI - LostRuins/koboldcpp
- [Everything WRONG with LLM Benchmarks (ft. MMLU)!!!](https://www.youtube.com/watch?v=74Uo2HU8HBo): 🔗 Links 🔗When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboardshttps://arxiv.org/pdf/2402.01781.pdf❤️ If you want to s...
- [Cheat Sheet: Mastering Temperature and Top_p in ChatGPT API](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683): Hello everyone!  Ok, I admit had help from OpenAi with this. But what I “helped” put together I think can greatly improve the results and costs of using OpenAi within your apps and plugins, specially ...
- [Andrew Garfield Andrew Garfield Moonlight Meme GIF - Andrew garfield Andrew Garfield Moonlight meme Andrew Garfield Moonlight trend - Discover &amp; Share GIFs](https://tenor.com/view/andrew-garfield-andrew-garfield-moonlight-meme-andrew-garfield-moonlight-trend-andrew-garfield-meme-gif-18154541527070698780): Click to view the GIF
- [GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros?tab=readme-ov-file#lmoe): Customizable implementation of the self-instruct paper. - jondurbin/airoboros
- [brucethemoose/Yi-34B-200K-RPMerge · Hugging Face](https://huggingface.co/brucethemoose/Yi-34B-200K-RPMerge): no description found
- [Doctor-Shotgun/Nous-Capybara-limarpv3-34B · Hugging Face](https://huggingface.co/Doctor-Shotgun/Nous-Capybara-limarpv3-34B): no description found
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - Haidra-Org/AI-Horde-Worker: This repo turns your PC into a AI Horde worker node](https://github.com/Haidra-Org/AI-Horde-Worker): This repo turns your PC into a AI Horde worker node - Haidra-Org/AI-Horde-Worker
- [Adjust VRAM/RAM split on Apple Silicon · ggerganov/llama.cpp · Discussion #2182](https://github.com/ggerganov/llama.cpp/discussions/2182): // this tool allows you to change the VRAM/RAM split on Unified Memory on Apple Silicon to whatever you want, allowing for more VRAM for inference // c++ -std=c++17 -framework CoreFoundation -o vra...
- [k-quants by ikawrakow · Pull Request #1684 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1684): What This PR adds a series of 2-6 bit quantization methods, along with quantization mixes, as proposed in #1240 and #1256. Scalar, AVX2, ARM_NEON, and CUDA implementations are provided. Why This is...
- [nextai-team/apollo-v1-7b · Hugging Face](https://huggingface.co/nextai-team/apollo-v1-7b): no description found
- [mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): no description found
- [Intel/neural-chat-7b-v3-3 · Hugging Face](https://huggingface.co/Intel/neural-chat-7b-v3-3): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ancmf2/yet_another_awesome_roleplaying_model_review/?rdt=58656): no description found
- [Reddit - Dive into anything](https://reddit.com/r/LocalLLaMA/comments/190pbtn/shoutout_to_a_great_rp_model/): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1205810274955952128) (535 messages🔥🔥🔥): 

- **Discussing the Versatility of Miqu Models**: Users have been sharing their insights on models such as **Miqu**, including their strength in performance compared to model merges like **Miquella**. There was also a mention of the potential performance of a **MiquMaid** model, which is fine-tuned for ERP, with links to [MiquMaid-v2-70B](https://huggingface.co/NeverSleep/MiquMaid-v2-70B) and [MiquMaid-v2-70B-DPO](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO) provided by `@soufflespethuman`.

- **Model Configuration and Setup Queries**: Users like `@netrve` and `@johnrobertsmith` shared details and experiences on setting up models, debating the effects of context length, repetition penalty, and memory splits across GPUs. `@lonestriker` provided detailed information about **exl2 models** and their quant sizes, with raw sizes ranging from 34GB to 110GB.

- **Lively Debate on Tokenizer Use in ST**: `@stoop poops` suggested `@netrve` read the docs when there was a question about ST's use of tokenizers. The discussion highlighted some confusion around the purpose and functionality of the tokenizer setting in ST (Silly Tavern).

- **Technical Tips for AMD Users**: `@spottyluck` offered advice on using AMD's AOCL for improved CPU performance on inference with llama.cpp, suggesting specific compile options that make use of AMD's AVX512 extensions and better kernels.

- **Implementing Custom Scripting for Character States**: `@johnrobertsmith` expressed an interest in creating scripts to manage character states in storytelling scenarios using STscript and lorebooks, looking for assistance and ideas to turn his theoretical knowledge into a practical implementation.

**Links mentioned**:

- [Neko Atsume Cat GIF - Neko Atsume Cat Kitty - Discover &amp; Share GIFs](https://tenor.com/view/neko-atsume-cat-kitty-neko-atsume-vr-speech-bubble-gif-25743386): Click to view the GIF
- [NeverSleep/MiquMaid-v2-2x70B-DPO · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-2x70B-DPO?not-for-all-audiences=true): no description found
- [Homer Simpsons GIF - Homer Simpsons Audacity - Discover &amp; Share GIFs](https://tenor.com/view/homer-simpsons-audacity-lisa-marge-gif-16591769): Click to view the GIF
- [The Chi GIF - The Chi - Discover &amp; Share GIFs](https://tenor.com/view/the-chi-gif-11090171): Click to view the GIF
- [Cat Kitten GIF - Cat Kitten Speech Bubble - Discover &amp; Share GIFs](https://tenor.com/view/cat-kitten-speech-bubble-speech-discord-gif-25192162): Click to view the GIF
- [What The Fuck Wtf Is Going On GIF - What The Fuck Wtf Is Going On What The - Discover &amp; Share GIFs](https://tenor.com/view/what-the-fuck-wtf-is-going-on-what-the-gif-16853392): Click to view the GIF
- [GitHub - yule-BUAA/MergeLM: Codebase for Merging Language Models](https://github.com/yule-BUAA/MergeLM?tab=readme-ov-file): Codebase for Merging Language Models. Contribute to yule-BUAA/MergeLM development by creating an account on GitHub.
- [Answering questions with data](https://www.crumplab.com/statistics/): A free textbook teaching introductory statistics for undergraduates in psychology, including a lab manual, and course website. Licensed on CC BY SA 4.0
- [Cats Cat GIF - Cats Cat Cucumber - Discover &amp; Share GIFs](https://tenor.com/view/cats-cat-cucumber-scared-gif-10226870): Click to view the GIF
- [Boxing Day GIF - Cats Cats In Boxes Armor - Discover &amp; Share GIFs](https://tenor.com/view/cats-cats-in-boxes-armor-gif-3294499): Click to view the GIF
- [Did You Pray Today Turbulence GIF - Did you pray today Turbulence - Discover &amp; Share GIFs](https://tenor.com/view/did-you-pray-today-turbulence-gif-6688607547865508186): Click to view the GIF
- [Nexesenex/abacusai_Smaug-Yi-34B-v0.1-iMat.GGUF at main](https://huggingface.co/Nexesenex/abacusai_Smaug-Yi-34B-v0.1-iMat.GGUF/tree/main): no description found
- [Skinner Homer GIF - Skinner Homer Drag Net - Discover &amp; Share GIFs](https://tenor.com/view/skinner-homer-drag-net-nod-plan-gif-4964903): Click to view the GIF
- [Catzilla 😅 | Do not do this to your cat, the street friends will laugh at him 👀](https://www.youtube.com/shorts/7wS5Nmvpjp8): &quot;Copyright Disclaimer under section 107 of the Copyright Act of 1976, allowance is made for &#39;fair use&#39; for purposes such as criticism, comment, news reportin...
- [Good Heavens GIF - OMG Shocked Surprised - Discover &amp; Share GIFs](https://tenor.com/view/omg-shocked-surprised-oh-my-god-gasp-gif-5189017): Click to view the GIF
- [NeverSleep/MiquMaid-v2-70B · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B): no description found
- [NeverSleep/MiquMaid-v2-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-GGUF): no description found
- [NeverSleep/MiquMaid-v2-70B-DPO · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO): no description found
- [NeverSleep/MiquMaid-v2-70B-DPO-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO-GGUF): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1205878821065334915) (30 messages🔥): 

- **Fine-tuning Chatbot Models**: `@maldevide` highlighted the basic steps to convert a base model to an instruct model involve fine-tuning with a good instruct dataset for two epochs. Further discussion by `@jondurbin` and `@starsupernova` delved into details like dataset sources, such as [bagel datasets](https://github.com/jondurbin/bagel?tab=readme-ov-file#sft-data-sources), and the actual process of fine-tuning which can add knowledge to the model.
  
- **Instruct Dataset Recommendations**: `@mr.userbox020` inquired about which datasets are best for creating an instruct model, and `@maldevide` recommended considering datasets like **WizardLM_evol_instruct_V2_196k**, **OpenHermes-2.5**, and others for their proven broad base, while also mixing in any specific specializations needed.

- **Understanding the Impact of Fine-tuning**: `@skirosso` asked about the purpose of fine-tuning, leading to a clarification that it can change a chatbot model's tone and also add knowledge, especially when pretraining is continued across all layers, as explained by `@starsupernova`. `@mr.userbox020` agreed, noting that the best instruct models, like mixtral, would already be capable of telling dragon stories instructed by a user.

- **The Future of Fine-tuning Speed and Efficiency**: Sharing a resource, `@mr.userbox020` brought attention to a GitHub repository named [unsloth](https://github.com/unslothai/unsloth), which claims faster and more efficient QLoRA fine-tuning for models like Mistral. `@starsupernova` confirmed its performance improvements, citing 2.2x speed-up and 70% VRAM reduction.

- **Fine-tuning compared to Training**: `@wolfsauge` added depth to the fine-tuning discussion by differentiating between training and fine-tuning, emphasizing resource savings and stability. They also mentioned that staying up-to-date with current fine-tuning trends is crucial and recommended further exploration of specific fine-tuning methods like RHLF with PPO and SFT with DPO.

**Links mentioned**:

- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel?tab=readme-ov-file#sft-data-sources): A bagel, with everything. Contribute to jondurbin/bagel development by creating an account on GitHub.
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1205881885905915954) (10 messages🔥): 

- **Surya OCR Script Shared**: `@cybertimon` provided a **Python script** for optical character recognition using the Surya OCR tool and mentioned the required installation of the dev branch of Surya with the command `pip install git+https://github.com/VikParuchuri/surya@dev`.
- **Code Snippet Appreciation**: `@bartowski1182` expressed their admiration for the shared Surya OCR code, calling it awesome.
- **GitHub Repo Suggestion**: `@mr.userbox020` suggested `@cybertimon` create a **GitHub repository** to share the Surya OCR code, but `@cybertimon` clarified that it was merely an example script, not a full project. They later shared a [Gist link](https://gist.github.com/CyberTimon/85ca0e797a95e3d5562dd6018f4e2131) to the code.
- **Request for Debugging Assistance**: `@ninyago` asked for help with a bug in `user.html` from their [GitHub repository](https://github.com/Ninyago53/webaudiobook.git), where the recording function does not always start after ElevenLabs finishes speaking.
- **Potential Third-Party Issue Highlighted**: In response to `@ninyago`'s request, `@falconsfly` suggested that the problem might be related to **ElevenLabs**, sharing an experience of a job stalling on their platform, indicating a potential issue not with the code but with the ElevenLabs service itself.

**Links mentioned**:

- [Surya OCR](https://gist.github.com/CyberTimon/85ca0e797a95e3d5562dd6018f4e2131): Surya OCR. GitHub Gist: instantly share code, notes, and snippets.
- [GitHub - Ninyago53/webaudiobook](https://github.com/Ninyago53/webaudiobook.git): Contribute to Ninyago53/webaudiobook development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1205790688667770940) (399 messages🔥🔥): 

- **Model Troubles and Discussions**: Users `@lacrak27` and `@heyitsyorkie` discussed issues running TheBloke's models, concluding the error likely resulted from a broken quant during the conversion of the original PyTorch model to gguf.
- **Tech Stack Queries**: User `@jitterysniper` inquired about the tech stack of LM Studio, clarified by `.ben.com` as llama.cpp, and the discussion later expanded to the app's specifics, surmising it might be built with Electron.
- **Image Generation Woes**: Users `@sunboy9710` and `@heyitsyorkie` discussed the difficulty and limitations of using LM Studio for image generation tasks.
- **VPNs and ISP Blockages**: User `@stevecnycpaigne` had issues accessing huggingface.co from different locations, leading `@heyitsyorkie` to suggest trying a VPN as it might be an ISP-related problem.
- **Privacy and Usage Data Concerns**: User `@f0xa` compared GPT4All and LM Studio, seeking clarity on data privacy with LMStudio, and `@fabguy` indicated LM Studio's privacy by default, with shared data coming from updates and model downloads from Huggingface.
- **Web UI for LM Studio Sought After**: User `@laststandingknight` queried about the availability of a web UI for LM Studio chats, confirmed to be unavailable by `@fabguy`.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1187757393556295681): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204759902548000769): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Continue](https://continue.dev/): no description found
- [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF): no description found
- [Twochoices Funny GIF - Twochoices Funny Two - Discover &amp; Share GIFs](https://tenor.com/view/twochoices-funny-two-choices-divorce-gif-5018698): Click to view the GIF
- [chilliadgl/RG_fake_signatures_ST at main](https://huggingface.co/chilliadgl/RG_fake_signatures_ST/tree/main): no description found
- [no title found](https://news.ycombinator.com/item?id=39326165): no description found
- [GitHub - b4rtaz/distributed-llama: Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage.](https://github.com/b4rtaz/distributed-llama): Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage. - b4rtaz/distributed-llama
- [examples/how-to-run-llama-cpp-on-raspberry-pi.md at master · garyexplains/examples](https://github.com/garyexplains/examples/blob/master/how-to-run-llama-cpp-on-raspberry-pi.md): Example code used in my videos. Contribute to garyexplains/examples development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1205856822226067456) (92 messages🔥🔥): 

- **Struggling with Small Model Classification**: `@speedy.dev` experienced issues with classification tasks using 13B and 7B Llama models, particularly the uncensored variants, and contemplated [fine-tuning as a solution](https://discourse.devontechnologies.com/t/experimenting-with-llama2-llm-for-local-file-classification-renaming-summarizing-analysing/76945).
- **Goliath vs. Goat - The Model Battle for Story Quality**: `@goldensun3ds` ran tests comparing story writing between the Bloke Goat Storytelling 70B Q6, Bloke Goliath 120B Q6, and Mixtral 7B Q6 models, noting that Goliath captured emotions better, but Mixtral was faster, and offering insight into combating repetitive loops.
- **Local Chat with Docs Still in Limbo**: `@dr.nova.` joined the community looking for a local alternative to GPT-4 for chatting with PDFs and received input that while LMStudio has no such feature yet, GPT-4 is the reigning solution for document-based interactions.
- **Selecting the Best Model for Task Delegation? A Hypothetical Approach**: `@binaryalgorithm` pondered the idea of a meta-model that could choose the best model for a given task, and `@.ben.com` mentioned that [openrouter.ai](https://openrouter.ai) has a basic router for model selection.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1186031214189084783/1186031214189084783): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=document-question-answering&sort=trending): no description found
- [Experimenting with llama2 LLM for local file classification (renaming, summarizing, analysing)](https://discourse.devontechnologies.com/t/experimenting-with-llama2-llm-for-local-file-classification-renaming-summarizing-analysing/76945): Follow-up from OpenAI ChatGPT for automatic generation of matching filenames - #3 by syntagm  ChatGPT works extremely well to get some logic into OCRed documents and PDFs, but would be nice to do this...

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1206231358398009344) (1 messages): 

- **Model Metadata Management Insights Shared**: `@wolfspyre` highlighted the importance of model metadata management using a [HackMD post](https://hackmd.io/@janhq/HJezIhu4T), which suggests the need for better categorization of model "parameters" among various chat platforms. They drew attention to the differences in `init/load params`, `run params`, and `server/engine params`.
- **Potential for a Model.json Standard**: The HackMD document discusses the possibility of establishing a `model.json` standard by Jan and provides a [link to a Github repository](https://github.com/janhq/model.json) which includes schema and example files for different versions.

**Links mentioned**:

[Model Object Teardowns - HackMD](https://hackmd.io/@janhq/HJezIhu4T): Model File Formats

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1205791295545671710) (197 messages🔥🔥): 

- **Seeking Advice on AVX Support**: `@guest7_25187` was pleased to find Intel Xeon E5-2670 v3 CPUs on eBay that support AVX2, which would be compatible with their server. `@heyitsyorkie` suggested that if combined with sufficient RAM and Nvidia P40 GPUs, `@guest7_25187` would see significant speed improvements.
  
- **GPU Decisions for Model Performance**: Discussions about hardware for running LLMs (Large Language Models) highlighted `@nink1` emphasizing the benefit of having more VRAM and CUDA cores, noting particularly the 3090's advantage in terms of VRAM to core ratio and NVLINK capability. `@konst.io` was advised that adding another 64GB of RAM wouldn’t hurt but the priority should be to maximize VRAM first.

- **Mac vs. Custom Builds for LLM Inferencing**: `@heyitsyorkie` shared their experience with running Goliath 120b 32k model on an M3 Max 128gb, clocking it faster than their 4090 setup. Meanwhile, `@wildcat_aurora` discussed their setup involving P40 GPUs and Xeon processors which was repurposed from Apple's Siri service, delivering effective results at a lower power consumption.

- **Market Speculations and Hardware Strategies**: There was a mix of speculation and desires for future hardware developments with `@nink1`, `@christianazinn`, and others debating Nvidia's strategic decisions about VRAM on consumer GPUs and looking forward to potential enterprise solutions by Apple with more RAM.

- **Troubleshooting and Setup for LLM Inferencing**: Members like `@therealril3y`, `@lardz90`, and `@the_yo_92` sought assistance with issues of LLM Studio not utilizing extra GPUs, RAM not being correctly detected after upgrades, and encountering JavaScript errors on M1 Mac. `@heyitsyorkie` and `@speedy.dev` provided quick fixes and suggested creating a support thread or updating software for resolution.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/11105981831443): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1111649100518133842/1205933827999010827): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1153759714082033735/1205652225435631667): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1111649100518133842/1204400922948673546): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (1 messages): 

ramendraws: :1ski_smug:
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1206457972767723520) (2 messages): 

- **Preference for CrewAI**: `@docorange88` expressed a distinct preference for **CrewAI** over AutoGen, suggesting that AutoGen is more difficult to manage.
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1206295896581480490) (1 messages): 

- **Intel Dropping AVX-512 Raises Eyebrows**: `@technot80` expressed confusion over **Intel**'s decision to drop **AVX-512** support on a powerful i9 processor, especially since their less powerful i5-11600k includes it. They found the move *truly weird*.
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1205945710164574279) (1 messages): 

- **HuggingFace Hub Experiencing Downtime**: User `@lunarflu` announced that **HF Hub** and git hosting are temporarily down. They mentioned that the team is currently working on resolving the issue and asked for community support.
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1205799064541593631) (603 messages🔥🔥🔥): 

- **Verifying on Discord Can Be Tricky**: User `@d3113` faced an error when trying to authenticate with a bot on Discord, which was resolved by enabling "Allow messages from server members" in their settings. User `@lunarflu` advised this solution and mentioned that the token used for verification can be deleted afterwards.
- **Keeping It Conversational**: User `@miker1662` was encountering issues with finding conversational models on Hugging Face due to an API bug, which `@osanseviero` confirmed and linked to an ongoing issue and changes in API usage. Hugging Face is deprecating `conversational` and merging it into `text-generation`.
- **The Dilemma of PEFT for LLM Fine-Tuning**: User `@samuelcorsan` debated whether to use PEFT (Parameter Efficient Fine-Tuning) when fine-tuning a conversational chatbot LLM, but eventually decided to go for full fine-tuning, removing PEFT from their code. They were given advice by `@vipitis` to use the full sequence length during fine-tuning to avoid the model learning positional embeddings, chunking the dataset for efficiency.
- **Raspberry Pi for CV on the Edge**: User `@akvnn` inquired whether a Raspberry Pi could handle running a computer vision (CV) model continuously for a multi-camera system. And user `@yamatovergil89` confirmed its viability but did not specify if it could handle multiple cameras simultaneously.
- **Hardware Queries for Running LLMs Locally**: `@lookingforspeed` sought advice on whether dual NVIDIA 4080 Supers would be more suitable than a single 4090 for locally coding LLMs like Mixtral. `@tddammo` recommended older generation pro cards like the A4500 or A4000 for better electrical efficiency and support for NVLink, and subsequently explained the benefits over consumer cards.

**Links mentioned**:

- [mistralai/Mixtral-8x7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1): no description found
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206181148976222268): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206246780950544405): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [transformers/src/transformers/configuration_utils.py at 58e3d23e97078f361a533b9ec4a6a2de674ea52a · huggingface/transformers](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/src/transformers/configuration_utils.py#L677C8-L677C10): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
- [Cry Animecry GIF - Cry Animecry Anime - Discover &amp; Share GIFs](https://tenor.com/view/cry-animecry-anime-gif-27695618): Click to view the GIF
- [Free Hugs Shack GIF - Free Hugs Shack Forest - Discover &amp; Share GIFs](https://tenor.com/view/free-hugs-shack-forest-scary-weird-gif-23469977): Click to view the GIF
- [Anime Cry GIF - Anime Cry Sad - Discover &amp; Share GIFs](https://tenor.com/view/anime-cry-sad-gif-14080503): Click to view the GIF
- [M3 max 128GB for AI running Llama2 7b 13b and 70b](https://www.youtube.com/watch?v=jaM02mb6JFM): In this video we run Llama models using the new M3 max with 128GB and we compare it with a M1 pro and RTX 4090 to see the real world performance of this Chip...
- [
Hugging Face status
](https://status.huggingface.co/): no description found
- [Tweet from Ross Wightman (@wightmanr)](https://x.com/wightmanr/status/1742637301346508929?s=20): @bhutanisanyam1 @jamesbower NVLINK does make a difference, even on 2-GPUs but impact varies with distributed workload. Unfortunately, it&#39;s not a concern on hobby machines now, 40x0 and RTX6000 Ada...
- [andreasjansson/codellama-7b-instruct-gguf – Run with an API on Replicate](https://replicate.com/andreasjansson/codellama-7b-instruct-gguf): no description found
- [Whisper Large V3 - a Hugging Face Space by hf-audio](https://huggingface.co/spaces/hf-audio/whisper-large-v3): no description found
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884): Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented...
- [Error when calling `InferenceClient.conversational` · Issue #2023 · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/issues/2023): Describe the bug Calling InferenceClient.conversational according to the docs results in a 400 Client Error. Reproduction from huggingface_hub import InferenceClient InferenceClient().conversationa...
- [DSPy PROMPT Engineering w/ ICL-RAG (How to Code Self-improving LLM-RM Pipelines)](https://www.youtube.com/playlist?list=PLgy71-0-2-F00lrRr2EzzTdnbJXax6sn2): Advanced Prompt Engineering. From human prompt templates to self-improving, self-config prompt pipelines via DSPy. Advanced Techniques in Pipeline Self-Optim...
- [Models - Hugging Face](https://huggingface.co/models?other=conversational): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/17kcgjv/how_does_apples_new_m3_128gb_ram_macbook_pro/): no description found
- [Amazon EC2 G5 Instances | Amazon Web Services](https://aws.amazon.com/ec2/instance-types/g5/): no description found
- [Pricing | Cloud AI Meets Unbeatable Value](https://rundiffusion.com/pricing): With pay as you go pricing up to full on monthly enterprise stable diffusion plans, we have you covered. Empower your vision with unbeatable costs.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/17kcgjv/how_does_): no description found
- [What Is NVLink?](https://blogs.nvidia.com/blog/what-is-nvidia-nvlink/): NVLink is a high-speed interconnect for GPU and CPU processors in accelerated systems, propelling data and calculations to actionable results.
- [GitHub - huggingface/transformers at 58e3d23e97078f361a533b9ec4a6a2de674ea52a](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - GitHub - huggingface/transformers at 58e3d23e97078f361a533b9ec4a6a2de674ea52a
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1205807574042148894) (8 messages🔥): 

- **Back in Action on LinkedIn**: `@aiman1993` has resumed posting on LinkedIn after a 6-week pause, sharing a [post on upgrading CUDA Toolkit and NVIDIA Driver](https://www.linkedin.com/posts/isham-rashik-5a547711b_upgrading-cuda-toolkit-and-nvidia-driver-activity-7161933041349640192-6c9W?utm_source=share&utm_medium=member_desktop).
- **TokenClassification Fine-tuning**: `@kamama2127` is learning the ropes of fine-tuning for token classification.
- **Choosing the Right Python Library for APIs**: `@subham5089` discussed the importance of understanding different Python libraries for API calling, like **Requests, AIOHTTP, and HTTPX**, in a [LinkedIn post](https://www.linkedin.com/posts/subham-kundu-2746b515b_python-api-generativeai-activity-7162345296587284480-mzuB?utm_source=share&utm_medium=member_desktop).
- **Websockets Meet HTTPX**: In response to `@dastardlydoright`'s questions about websockets in Python, `@subham5089` recommended [httpx-ws](https://frankie567.github.io/httpx-ws/), a library for WebSockets support in HTTPX, in addition to providing a [link to the source code](https://github.com/frankie567/httpx-ws).
- **GPT Insights from Karpathy**: `@wonder_in_aliceland` mentioned enjoying a video by Andrej Karpathy about **nanogpt**, which explores the inner workings and ideas behind GPT.

**Links mentioned**:

[HTTPX WS](https://frankie567.github.io/httpx-ws/): no description found

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1205918004282527774) (5 messages): 

- **Deep Dive into Deep Learning Breakthroughs**: `@branchverse` shared an [article](https://link.springer.com/article/10.1007/s10489-022-04278-6) highlighting the progress in deep learning since the 2010s. The article notes innovations driven by open source tools, hardware advancements, and availability of labeled data.
  
- **Normcore LLM Reads on GitHub**: `@husainhz7` linked a [GitHub Gist](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e) titled "Normcore LLM Reads," which is a collection of code, notes, and snippets related to LLMs.
  
- **AI-Infused Genetic Algorithm for Greener Gardens**: `@paccer` discussed an article that features a genetic algorithm combined with LLM for gardening optimization. The AI-powered tool GRDN.AI seeks to improve companion planting and is documented in a [Medium post](https://medium.com/@dheymann314/ai-infused-optimization-in-the-wild-developing-a-companion-planting-app-357e5da29d10).
  
- **Unveiling Computer Vision Techniques**: `@purple_lizard` posted a link to the [Grad-CAM research paper](https://arxiv.org/pdf/1610.02391.pdf), which introduces a technique for making convolutional neural network (CNN) decisions transparent via visual explanations.

- **Exploring AI Research**: `@kamama2127` pointed out a recent AI research paper on [arXiv](https://arxiv.org/abs/2402.00838) with a list of authors contributing to the field. The paper discusses new findings and advancements in artificial intelligence.

**Links mentioned**:

- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838): Language models (LMs) have become ubiquitous in both NLP research and in commercial product offerings. As their commercial importance has surged, the most powerful models have become closed off, gated...
- [AI-Infused Optimization in the Wild: Developing a Companion Planting App](https://medium.com/@dheymann314/ai-infused-optimization-in-the-wild-developing-a-companion-planting-app-357e5da29d10): Key to a thriving garden, companion planting offers natural pest control, promotes healthy growth, and leads to more nutrients in soil. GRDN.AI applies this concept using an AI-infused genetic…
- [Normcore LLM Reads](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e): Normcore LLM Reads. GitHub Gist: instantly share code, notes, and snippets.
- [Front-end deep learning web apps development and deployment: a review - Applied Intelligence](https://link.springer.com/article/10.1007/s10489-022-04278-6): Machine learning and deep learning models are commonly developed using programming languages such as Python, C++, or R and deployed as web apps delivered from a back-end server or as mobile apps insta...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1205917374147199016) (10 messages🔥): 

- **Data Structures Demystified**: User `@kamama2127` created a Streamlit page that aggregates **27 data structures**, their implementation in Python, and associated problems. Check out the interactive learning tool [here](https://datastructurewithaipy-dhmexr7ukaudutklwjiawp.streamlit.app/).

- **Zero Shot to Object Detection**: `@pradeep1148` shared a YouTube video illustrating automatic object detection using zero shot methods with the MoonDream vision language model. Watch the tutorial [here](https://www.youtube.com/watch?v=W4T7zHluzaM).

- **DALL-E and Midjourney's Imagery Dataset**: User `@ehristoforu` introduced two Hugging Face datasets consisting of images generated by DALL-E 3 and Midjourney models. Explore the DALL-E dataset [here](https://huggingface.co/datasets/ehristoforu/dalle-3-images) and Midjourney dataset [here](https://huggingface.co/datasets/ehristoforu/midjourney-images).

- **Animation Made Easy with FumesAI**: `@myg5702` shared a link to FumesAI's Hugging Face space, featuring the **text-to-Animation-Fast-AnimateDiff** tool. Jump into animating your text [here](https://huggingface.co/spaces/FumesAI/text-to-Animation-Fast-AnimateDiff).

- **AI-Enhanced Music Creation Workflow**: `.bigdookie` discussed how the integration of the MusicGen tool into Ableton has improved, likening it to playing a slot machine. Watch the creative process [here](https://youtu.be/wCh-ug1475Q?si=R5_jeeBM5aEK7de2).

**Links mentioned**:

- [Text To Animation Fast AnimateDiff - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/text-to-Animation-Fast-AnimateDiff): no description found
- [Prometheus - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/prometheus): no description found
- [Quiz Maker - a Hugging Face Space by narra-ai](https://huggingface.co/spaces/narra-ai/quizmona): no description found
- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM): We are going to see how we can do automatic object detetction using zero shot object detection and moondream vison langugae model#llm #ml #ai #largelanguagem...
- [another song from scratch with ableton and musicgen - captain&#39;s chair 12](https://youtu.be/wCh-ug1475Q?si=R5_jeeBM5aEK7de2): In this episode we use  use @CradleAudio &#39;s god particle to make much big loudness on our acousticand @Unisonaudio &#39;s drum monkey, which kinda bummed me out,...
- [ehristoforu/dalle-3-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/dalle-3-images): no description found
- [ehristoforu/midjourney-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/midjourney-images): no description found
- [Spatial Media Converter](https://www.spatialmediaconverter.com): Convert RGB Images to Spatial Photos for Apple Vision Pro.
- [no title found](https://datastructurewithaipy-dhmexr7ukaudutklwjiawp.streamlit.app/): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1206085334224478218) (20 messages🔥): 

- **Flash Attention Fires Up Interest**: `@ericauld` expressed interest in presenting on **Flash Attention**, inquiring if it had been discussed in relation to **Mamba** and **S4**. Both `@chad_in_the_house` and `@ericauld` discussed the mathematical identities that allow softmax calculations to be done blockwise rather than globally.
- **Curiosity About RWKV**: `@typoilu` showed an eagerness to give a presentation about the **RWKV** model, which has not yet been presented, and discussed scheduling for a future date.
- **Scheduling Mamba Presentation**: `@ericuald` and `@chad_in_the_house` coordinated to find a mutually convenient time for a **Mamba** presentation, using [When2meet](https://www.when2meet.com/?23627290-1hbtA) for scheduling.
- **Google Calendar Consideration**: `@chad_in_the_house` suggested starting a Google Calendar to manage presentation scheduling, with `@lunarflu` supporting the idea for its usefulness.
- **Time Zone Coordination for Presentations**: `@typoilu` noted their **UTC+1** time zone, and `@chad_in_the_house` mentioned the time difference with EST (Eastern Standard Time), looking to find a suitable time for US audience attendance for the RWKV presentation.

**Links mentioned**:

[Mamba Presentation - When2meet](https://www.when2meet.com/?23627290-1hbtA): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1205911769072009286) (1 messages): 

- **Simplifying Multi-label Image Classification**: User `@nielsr_` explained how to **fine-tune an image classifier** for multi-label classification with **Hugging Face Transformers**. They provided a code snippet using `Dinov2ForImageClassification` for easy instantiation.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1205835648653598741) (20 messages🔥): 

- **PEFT Package**: User `@开饭噜` reported issues with saving LoRA weights locally due to inability to connect to HuggingFace and received a solution from `@nruaif` recommending to **downgrade to PEFT 0.7.1**, which resolved the problem.
- **Troubleshooting Saving Models with PEFT**: `@vipitis` suggested that the `.save_pretrained()` method may attempt to create a repository when a full path is not provided. They recommended trying a **`Path` object instead of a `str`** to bypass connectivity issues.
- **Seeking JSON-aware LLM for Local Use**: `@nic0653` inquired about a robust Large Language Model (LLM) that interprets language and JSON schemas to output JSON. The discussion pointed to **Claude** excelling in this task but challenges remain for **local deployment** options.
- **In Search of Profanity Capable 15b+ LLM**: `@wavy_n1c9` sought recommendations for a 15 billion+ parameter LLM capable of generating dialogue or text containing **profanity** for local use, yet no suggestions were made within the chat history.
- **Small Model for Local Code Generation**: `@adolfhipster1` asked for advice on a small model fit for **code generation**, expressing concerns that GPT-2 was insufficient and looking for alternatives to downloading **LLaMA**.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1205797164911955998) (234 messages🔥🔥): 

- **Discord Community Wonders About AI's Financial Advice**: Members like `@azashirokuchiki` discuss the limitations of AI in generating trading strategies, expressing frustration with the cautious responses and lack of memory in the bot regarding financial matters.
- **AI-Centric Tool `ts_zip` Shared**: Link to a tool for compressing text files with Large Language Models shared by `@lugui`, sparking interest in AI-powered compression technologies.
- **Curiosity about OpenAI's Funding Ambitions**: The extraction of 5-7 trillion mentioned for AI chips sparked a debate, with users like `@thedasenqueen` and `@1015814` weighing in on the feasibility and impact on AI research and development.
- **AI's Impact on Societal Risks Discussed**: `@1015814` delves into the societal risks of improperly integrated autonomous robotics and the debate around AI's potential for both societal good and potential danger.
- **Google Gemini Takes Center Stage in Conversations**: Several users, including `@jaicraft` and `@thedreamakeem`, discuss the pros and cons of Google's Gemini AI product, its features, and anticipated improvements in coding and interpreter capabilities.

**Links mentioned**:

- [Building an early warning system for LLM-aided biological threat creation](https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation): We’re developing a blueprint for evaluating the risk that a large language model (LLM) could aid someone in creating a biological threat. In an evaluation involving both biology experts and students, ...
- [Tweet from Jack Krawczyk (@JackK)](https://fxtwitter.com/JackK/status/1756114082632220717?s=20): Ok - Gemini day 2 recap: things people like, things we gotta fix. Keep your feedback coming. We&#39;re reading it all.  THINGS PEOPLE LIKE (♥️♥️♥️) - Writing style - Creativity for helping you find th...
- [ts_zip: Text Compression using Large Language Models](https://bellard.org/ts_server/ts_zip.html): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1205824112862306304) (37 messages🔥): 

- **ChatGPT Suffers from Attention Issues**: `@.nefas` expressed frustration with ChatGPT's inconsistent responses during role-playing sessions, which diverge from the discussion abruptly. Others like `@satanhashtag` and `@a1vx` referred to context limits, explaining that the full 128K token context can be utilized in the API and by Enterprise users, but Plus and Team are limited to 32K tokens.
  
- **GPT-4 Subscription Sharing Confusion**: `@nickthepaladin` inquired if non-subscribers could use their GPT, to which `@solbus` clarified that a Plus or higher subscription is essential since all GPTs utilize GPT-4.

- **@ Mentions Feature Inconsistency**: `@rudds3802` raised an issue about not having access to the @ mentions feature, and `@solbus` and `@jaicraft` discussed its possible limited rollout and problems on mobile chromium browsers.

- **Flagged GPTs Create Confusion**: `@eligump` and `@yucareux` discussed their experiences with their GPT content being flagged and the appeal process, suggesting that compliance with the academic honesty policy might play a role in reapproval.

- **Strategies for Effective ChatGPT Interactions**: `@blckreaper` recommended providing ChatGPT with multiple narrative options for action-adventure scenarios to improve response flow and save tokens, while `@airahaerson` lamented the necessity of repeating instructions despite detailed formatting efforts.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1203945630561599518/1204282794529005599): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1205906395073880064) (63 messages🔥🔥): 

- **Assistance with ChatGPT Spreadsheet Creation**: `@crosscx` seeks help for instructing ChatGPT to convert the Midlands rules of evidence into a spreadsheet. However, facing Python errors, they run out of ChatGPT 4 messages. Users `@madame_architect` and `@darthgustav` suggest simplifying the task, perhaps starting without the spreadsheet aspect or breaking the task into manageable chunks.

- **Plain Language Transformation Trouble**: `@mondsuppe` discusses difficulties in getting ChatGPT to translate regular text into simple language adhering to specific plain language rules. `@darthgustav` and `@madame_architect` advise using examples and stepwise prompting to achieve better results, while `@eskcanta` shared a successful strategy using a two-step process. 

- **Understanding 'Conversational Cadence'**: In a discussion about GPT-4's capabilities, `@drinkoblog.weebly.com` observes the model's understanding of conversational cadence, which they argue implies an understanding of time in a specific context. `@bambooshoots` clarifies that it's about conversational flow rather than time awareness, leading to a further exchange on the definition of "cadence" with `@beanz_and_rice`.

- **GPT-4's Handling of 'State Changes'**: `@beanz_and_rice` elaborates on GPT-4’s ability to process "state changes," mentioning its competence in managing dynamic conditions and recommending prompt strategies that allow the model to adapt and respond effectively over multiple messages. 

- **Video Command Interjection**: An interjection in the conversation by `@zen_dove_40136` with "/video" is humorously countered by `@beanz_and_rice` with "/denied", continuing the trend of message commands in the discussion.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1205906395073880064) (63 messages🔥🔥): 

- **Chasing the Right Formula for Midlands Rules of Evidence**: `@crosscx` sought advice on getting ChatGPT to format the Midlands Rules of Evidence into a spreadsheet without altering content, facing a Python error and exhausting ChatGPT messages. `@madame_architect` suggested simplifying by not asking for a spreadsheet, while `@darthgustav.` recommended breaking down the task and later clarified that the issue is a memory error in the CI environment.

- **Plain Language Translation Conundrum**: `@mondsuppe` shared challenges translating text into simple language for individuals with learning difficulties, discussing specific rules such as short sentences and clear structures. Although `@darthgustav.` suggested templates might help, they expressed doubt about ChatGPT-3.5's capabilities, suggesting GPT-4 might fare better.

- **GPT-4 and Temporal Awareness**: Users discussed the nature of GPT-4's responses, with `@drinkoblog.weebly.com` observing a perception of time when allowing it to respond over multiple cycles, leading to a conversation about cadence and conversational flow with `@beanz_and_rice` and others.

- **Exploring Multi-Step Problem Solving with GPT-3.5**: `@eskcanta` described a two-step process to simplify complex explanations into child-friendly language using GPT-3.5, revealing its potential in breaking down complicated tasks into manageable stages.

- **Conversations on Definitions and Capabilities**: The channel included a discussion on the definition of "cadence" and its relation to GPT-4's conversational capabilities, with various interpretations provided by `@beanz_and_rice` and `@drinkoblog.weebly.com`. This highlighted the nuances in understanding how AI processes and delivers information.
  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1205844411842433064) (11 messages🔥): 

- **Elon Musk Goes Incognito?**: `@fullstack6209` spotted an individual who resembles Elon Musk in a video, claiming it's **AdrianDittmann** and mentioning a surprise call-in by **Alex Jones**. They shared [this snippet](https://twitter.com/i/status/1756105704786817130) from a recorded conversation that supposedly includes Musk, asking viewers to skip to timestamps **1:13** and **1:20** for the key moments, and attempted to summarize the event [here](https://twitter.com/DavidFSWD/status/1756287867167584671).
- **Musk's AI Ambitions Exposed**: Keeping up with the Elon Musk theme, `@fullstack6209` shares a quote of Musk stating, ***"Hey I've got 200 GPUs in the back of my pickup, and I'm going to make an AI faster than you can, and they do"***, reportedly from February 9th, 2024.
- **Emoji Advice Dismissed**: User `@.ben.com` humorously comments on not taking advice from someone who lacks emoji skills, though the context of the advice is not provided.
- **Heavyweight Framework Offline**: `@gabriel_syme` briefly laments that **"HF"** (likely referring to Hugging Face) is still down, indicating an ongoing issue with the service.
- **Sam Altman Tweet Shared**: `@teknium` shares a tweet from Sam Altman without further comment, viewers can check the tweet [here](https://twitter.com/sama/status/1756547355556598170).
- **YouTube on Automatic Object Detection**: `@pradeep1148` provides a link to a [YouTube video](https://www.youtube.com/watch?v=W4T7zHluzaM) titled **"Automatic Object Detection"**, but doesn't include any additional commentary on the content.

**Links mentioned**:

- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM): We are going to see how we can do automatic object detetction using zero shot object detection and moondream vison langugae model#llm #ml #ai #largelanguagem...
- [The Truth About Building AI Startups Today](https://www.youtube.com/watch?v=TwDJhUJL-5o&t=49s): In the first episode of the Lightcone Podcast, YC Group Partners dig into everything they have learned working with the top founders building AI startups tod...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1205851665140424704) (22 messages🔥): 

- **QuartetAnemoi-70B Unveiled**: `@nonameusr` shared a new sequential merge model dubbed [QuartetAnemoi-70B-t0.0001](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001), a combination of four distinct models using a NearSwap algorithm, showcasing storytelling prowess without relying on typical story-ending clichés.
- **Senku 70B Scores in TruthfulQA and ARC-Challenge**: `@carsonpoole` reports that [Senku 70B model](https://huggingface.co/senku), quantized to 4-bit without any calibration, achieves 62.3 in TruthfulQA and 85.75 in the ARC-Challenge, noting that the results are influenced by a bespoke prompt format.
- **Mixing Formats May Boost Senku Performance**: Continuing the discussion on Senku 70B, `@carsonpoole` mentions advice from the trainer suggesting that using chatml could potentially improve the model's performance, although it's not currently implemented in the testing format.
- **Tiny Model Training on OpenHermes**: `@euclaise` shared a [Twitter thread](https://twitter.com/WuMinghao_nlp/status/1756307170512248985) discussing a small model trained on OpenHermes, which sparked a side conversation about the smallest models members have trained, with `@teknium` revealing a 7B model as their smallest.
- **1.5 Bit Quantization Breakthrough**: `@.benxh` highlighted a GitHub pull request for 1.5 bit quantization, noting that this state-of-the-art quantization allows a 70b model to fit in less than 18GB of RAM, and expressed the intent to benchmark these new quants on the Miqu model.

**Links mentioned**:

- [alchemonaut/QuartetAnemoi-70B-t0.0001 · Hugging Face](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001): no description found
- [Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary](https://arxiv.org/abs/2402.00236): This study discusses the effects of positional encoding on recurrent neural networks (RNNs) utilizing synthetic benchmarks. Positional encoding &#34;time-stamps&#34; data points in time series and com...
- [1.5 bit quantization by ikawrakow · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453): This draft PR is a WIP that demonstrates 1.5 bits-per-weight (bpw) quantization. Only CUDA works, there is no implementation for the other supported back-ends. CUDA, AVX2 and ARM_NEON are implement...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1205795195593957467) (228 messages🔥🔥): 

- **Model Making Skills Inquiry**: User `@copyninja_kh` asked about the necessary skills for creating datasets, and after further prompting about technical specifics, `@teknium` suggested looking at how others like wizardevol and alpaca have done it.
- **UNA-SimpleSmaug-34B on Hugging Face**: User `@fblgit` shared a link to the UNA-SimpleSmaug-34B model on Hugging Face, describing its superior scoring over the original Smaug-34B model and noting its training on the [SimpleMath dataset](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta) with an emphasis on improving mathematical and reasoning capabilities.
- **Exploring Lilac Processed Hermes**: User `@nikhil_thorat` engaged in a discussion about the utility of UMAP projections in clustering datasets, offering options to share embeddings and projections. Later, he shared a link to the [Lilac Processed OpenHermes-2.5 dataset](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5) and mentioned that he'd add a column with 2D coordinates to the same dataset in the future.
- **Hosting Hugging Face Models with APIs**: User `@nonameusr` sought advice on the easiest way to host a Hugging Face model for API inference, leading to suggestions like using Flask and looking at platforms like Runpod that offer pay-by-the-second GPU services.
- **Mergekit Usage in Model Creation**: User `@weyaxi` prominently posted about TheProfessor-155b, a merged model using [mergekit](https://github.com/cg123/mergekit), designed for broad skills in conversational, reasoning, and scientific domains. Skepticism about its performance statistics such as `0.69 MMLU` and `0.4284 GSM8K` was expressed by `@nonameusr`.

**Links mentioned**:

- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b): no description found
- [fblgit/UNA-SimpleSmaug-34b-v1beta · Hugging Face](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta): no description found
- [typeof/miqu-70b-6 · Hugging Face](https://huggingface.co/typeof/miqu-70b-6): no description found
- [Buffer Overflow in Mixture of Experts](https://arxiv.org/abs/2402.05526): Mixture of Experts (MoE) has become a key ingredient for scaling large foundation models while keeping inference costs steady. We show that expert routing strategies that have cross-batch dependencies...
- [Doctor-Shotgun/Nous-Capybara-limarpv3-34B · Discussions](https://huggingface.co/Doctor-Shotgun/Nous-Capybara-limarpv3-34B/discussions): no description found
- [Herika - SafeAI in Skyrim](https://youtu.be/W9-uYzVEDuM?si=2DU7bvNUPKJiHu3F): THE most aligned AI Skyrim companion
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO): no description found
- [lilacai/lilac-OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5): no description found
- [abacusai/TheProfessor-155b · Hugging Face](https://huggingface.co/abacusai/TheProfessor-155b): no description found
- [Tweet from Eric Hartford (@erhartford)](https://fxtwitter.com/erhartford/status/1756747509186338956): https://huggingface.co/abacusai/TheProfessor-155b  TheProfessor-155b is a special model I made in partnership with @abacusai using @chargoddard&#39;s MergeKit - its purpose is interactive brainstormin...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1205806529291354142) (30 messages🔥): 

- **Seeking Straightforward Finetuning Path**: `@natefyi_30842` inquired about the simplest way to fine-tune models like **Nous: Hermes 2 Mixtral 8x7B DPO** with limited GPU resources and if there's a service for uploading prompt/answer pairs for fine-tuning. They mentioned previous experiences with **Axolotl** but found it cumbersome.
- **Tool Recommendation for Easy Finetuning**: `@teknium` suggested **together.ai** as a possible platform for `@natefyi_30842`'s requirements and mentioned the inevitable need for hyperparameter tuning to achieve a good model.
- **Yarn Repo 7B Training Troubles**: `@yoelvis8576` shared their failed attempt to fine-tune the **Mistral 7B** model using **FSDP** and **Flash Attn-2** due to CUDA out of memory errors despite various configurations.
- **User Interface Framework Woes**: `@tempus_fugit05` discussed the challenges of creating prompts with the correct structure when switching between models, and `@.ben.com` proposed **ollama** as a solution, though `@tempus_fugit05` prefers to continue developing a personal framework.
- **Exploration of Autonomous LLMs**: `@0xsingletonly` expressed interest in autonomous large language models (LLMs) and shared their intention to participate in Nous's upcoming SDK that `@teknium` mentioned was under development for enabling easy use of autonomous LLMs.
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1205805344165462066) (8 messages🔥): 

- **Brief Confusion about Roadmap Features**: `@gabriel_syme` expressed disappointment regarding the absence of a feature from the roadmap.
- **Clarification on Llava Team's Integration Work**: Responding to `@gabriel_syme`, `@qnguyen3` informed that the **Llava team** integrated the feature themselves but it has not been merged and is available in their repository.
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1205793352289751100) (162 messages🔥🔥): 

- **First Steps into AI**: `@dev2c2j` has experience in programming and has developed software from the ground up, starting from research papers. They have expressed a desire to understand and potentially contribute to AI, showing interest in EleutherAI and GPT-J, but facing initial hurdles understanding where to begin.
  
- **Community Engagement and Guidance**: `@paganpegasus` and others discussed the size of The Pile dataset for model training, with recommendations to use rsync for faster data transfer. `@alexanderrgriffing` shared their interest in learning CUDA programming and noted other communities focused on small systems and emergent AI.

- **Optimization Woes and Wonders**: `@zippika` discussed the benefits and challenges of the Prodigy optimizer, particularly its high VRAM usage but exceptional performance in training diffusion models. They also lamented the misuse of AI model merging and storage space on Hugging Face repositories.

- **Exploring AI's Fringes**: `@dev2c2j` challenged the notion that AI development requires brute force computing power, suggesting that combinations of tiny and larger networks could be more resource-efficient. Their statements sparked conversations about choosing the right AI problems and methods, with `@catboy_slim_` and `@alexanderrgriffing` underscoring the importance of scale in current AI development.

- **Rogue Machine Learning Models**: The community expressed concerns about the rise of questionable AI practices and the potential "grift era" of open-source LLMs, as mentioned by `@canadagoose1`. `@rallio.` brought up rumors of an impending advertisement for OpenAI's new release potentially coinciding with the Super Bowl, with links to the actual Microsoft commercial shared by `@clockrelativity2003`.

**Links mentioned**:

- [XXIIVV &mdash; uxn](https://wiki.xxiivv.com/site/uxn.html): no description found
- [Join the Learn AI Together Discord Server!](https://discord.com/invite/learnaitogether): Learn &amp; build AI. Technical Q&amp;A, tutorials, collabs, events, model bots.. ML, NLP, Generative AI (Midjourney, ChatGPT).. | 57384 members
- [Join the emergence Discord Server!](https://discord.gg/YA2eCJ2P): Check out the emergence community on Discord - hang out with 346 other members and enjoy free voice and text chat.
- [wolfram/miqu-1-120b · Hugging Face](https://huggingface.co/wolfram/miqu-1-120b): no description found
- [Microsoft Game Day Commercial | Copilot: Your everyday AI companion](https://youtu.be/SaCVSUbYpVc?t=40): With Microsoft Copilot and the power of AI, ideas become action, the impossible becomes possible, and hopes become reality. Copilot is available to anyone, a...
- [Nvidia CUDA Compiler - Wikipedia](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler): no description found
- [Crystal Nights &#x2014; Greg Egan](https://www.gregegan.net/MISC/CRYSTAL/Crystal.html): no description found
- [GitHub - konstmish/prodigy: The Prodigy optimizer and its variants for training neural networks.](https://github.com/konstmish/prodigy): The Prodigy optimizer and its variants for training neural networks. - konstmish/prodigy
- [GitHub - konstmish/prodigy: The Prodigy optimizer and its variants for training neural networks.](https://github.com/konstmish/prodigy?tab=readme-ov-file#diffusion-models): The Prodigy optimizer and its variants for training neural networks. - konstmish/prodigy

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1205787426212159508) (66 messages🔥🔥): 

- **Nested Networks Spark Interest**: `@thatspysaspy` was enthused by a tweet about nested networks in JAX and shared the [inspiration](https://twitter.com/aicrumb/status/1756156655836770504) within the channel.
- **Transformers within Transformers**: `@thatspysaspy` shared access to a [Colab notebook](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq) for experimenting with nested transformer networks, confirming that the implementation is backpropagatable and JIT compatible.
- **Discussion of Vector Orientation Confusion**: `@alexanderrgriffing` sparked a debate on conventions in math and machine learning, particularly regarding row versus column vector orientations, where `@thatspysaspy` highlighted how numpy treats them effectively as row vectors due to its broadcasting semantics.
- **Helping Hand for Diffusion Paper Implementation**: `@Nipsu` sought assistance on implementing a method for prompt-aware adjustment from a diffusion paper, sharing code in a [gist](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57) for Section 3.3 visualization, while `@johnryan465` offered to help debug the issue.
- **Collection of UniReps Research Papers Shared**: `@digthatdata` provided a link to a helpful GitHub repository, [UniReps-resources](https://github.com/UniReps/UniReps-resources), which contains a compilation of research papers for the community's benefit.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/794042109048651818/1205251974912811029): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [A phase transition between positional and semantic learning in a solvable model of dot-product attention](https://arxiv.org/abs/2402.03902): We investigate how a dot-product attention layer learns a positional attention matrix (with tokens attending to each other based on their respective positions) and a semantic attention matrix (with to...
- [Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning](https://arxiv.org/abs/2402.06619): Datasets are foundational to many breakthroughs in modern artificial intelligence. Many recent achievements in the space of natural language processing (NLP) can be attributed to the finetuning of pre...
- [Image Inpainting via Tractable Steering of Diffusion Models](https://arxiv.org/abs/2401.03349): Diffusion models are the current state of the art for generating photorealistic images. Controlling the sampling process for constrained image generation tasks such as inpainting, however, remains cha...
- [Fixed-point Inversion for Text-to-image diffusion models](https://arxiv.org/abs/2312.12540): Text-guided diffusion models offer powerful new ways to generate and manipulate images. Several applications of these models, including image editing interpolation, and semantic augmentation, require ...
- [GitHub - UniReps/UniReps-resources](https://github.com/UniReps/UniReps-resources): Contribute to UniReps/UniReps-resources development by creating an account on GitHub.
- [Google Colaboratory](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq): no description found
- [Jupyter Notebook Viewer](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57): no description found

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1205879335924535366) (7 messages): 

- **TOFU Benchmark Sparks Debate**: `@hailey_schoelkopf` linked the paper titled "TOFU" regarding a benchmark for unlearning sensitive data from trained models ([TOFU paper](https://arxiv.org/abs/2401.06121)). `@stellaathena` expressed skepticism about the paper's significance as an effective Machine Unlearning (MUL) benchmark.
- **Looking for Clarity**: `@stellaathena` has read the TOFU paper multiple times but remains unconvinced about its value as a meaningful MUL benchmark, questioning its implications.
- **A Potential Resource Shared**: `@aidan5513` shared another paper which may shed light on the conversation, discussing the relearning of concepts in models after neuron pruning ([neuron pruning paper](https://arxiv.org/abs/2401.01814)).
- **Papers as Great Primers**: `@millander` found value in the papers recommended, stating both were insightful as introductory material on the topics discussed.
- **Seeking Insight on TOFU Criticism**: `@millander` queried `@193204646687408129` (possibly stellaathena's unique user ID) directly, asking for specific disagreements with the TOFU paper, including if the concern was about its limited real-world application in a Q&A setting.

**Links mentioned**:

- [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121): Large language models trained on massive corpora of data from the web can memorize and reproduce sensitive or private data raising both legal and ethical concerns. Unlearning, or tuning models to forg...
- [Large Language Models Relearn Removed Concepts](https://arxiv.org/abs/2401.01814): Advances in model editing through neuron pruning hold promise for removing undesirable concepts from large language models. However, it remains unclear whether models have the capacity to reacquire pr...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1205797366825877504) (11 messages🔥): 

- **MedQA Benchmark Scrutiny**: `@johnnysands` raised a hypothesis that the **MedQA benchmark** might be a multiple-choice format leading Pythia models to perform at almost random chance levels if they are inherently poor at this type of question.

- **Search for Model API Comparative Data**: `@matthiaslau` is on the lookout for detailed results using `log_samples` for Model APIs, in order to conduct a thorough **comparison** between various models from **OpenAI, Anthropic, and the Open LLM Leaderboard**.

- **Call for GPQA Dataset Tasks**: `@hailey_schoelkopf` acknowledged new task PRs and suggested adding tasks for the **GPQA dataset**, warning that manual download may be necessary due to the dataset authors' concerns about **data leakage**. The dataset can be found in [this academic paper](https://arxiv.org/abs/2311.12022).

- **Clarification on Big Bench Hard Task Evaluation**: `@scrungle.tech` sought advice on **evaluating tasks** with GPT-4 models, particularly on response formatting choices, with `@hailey_schoelkopf` advising that their current approach involves searching the whole response (option A).

- **Contributors Needed for Hallucinations Leaderboard**: `@pminervini` extended an open invitation to contribute to a new **hallucinations leaderboard**, involving several novel tasks, which can be found on [Hugging Face's blog post](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations) and detailed within the Harness space on [Hugging Face's platform](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks).
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1205854690735292466) (166 messages🔥🔥): 

- **Cascading ASR+LLM+TTS Strategy Deemed Inadequate**: User `@donjuan5050` voiced his opinion that employing Cascaded ASR + LLM + TTS for speaking bots is unimpressive. Instead, he favors end2end training using a conversation dataset, and `@wielandbrendel` pointed out that the [Bud-e voice assistant](https://speechbot.github.io/spiritlm/index.html) adheres to such an approach, integrating ASR, LLM, and TTS in one Pytorch model for end-to-end trainability.

- **Legal Troubles for AI Art**: Discussion surfaced around a court ruling by *U.S. District Judge William Orrick*, rejecting Midjourney and StabilityAI's claim for an early dismissal under a First Amendment defense. Users debate the case's implication while `@pseudoterminalx` shares his view that he "rebuffed their claims" without granting their motion.

- **OpenAI's Diffusers and Stable Diffusion Drama**: Within the community, there's chatter about `sd-forge`, which includes code from diffusers, automatic1111, and comfyui, yet aims to avoid associations with the mentioned projects. `@astropulse` and others gave opinions on the chaotic nature of developments within the open-source UI sector for Stable Diffusion models, highlighting the line: "i prefer diffusers, where the developers are stable and the codebase is unstable."

- **DnD Map Generation with Neural Networks**: `@thejonasbrothers` showcases successful use of recent checkpoints to create detailed Dungeons and Dragons maps, with `@pseudoterminalx` providing images as examples of the generated artwork.

- **Technical Difficulties in the AI Development Space**: Users report issues with Hugging Face's services being down and discuss the pitfalls of reliance on external APIs, which can lead to operational problems when those services experience downtime. There are also mentions of technical intricacies within Diffusers and alternative approaches to model scheduling.

**Links mentioned**:

- [no title found](https://speechbot.github.io/spiritlm/index.html): no description found
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [Thumbs Up Double Thumbs Up GIF - Thumbs Up Double Thumbs Up Like - Discover &amp; Share GIFs](https://tenor.com/rbmfd590vsb.gif): Click to view the GIF
- [Section 230 - Wikipedia](https://en.m.wikipedia.org/wiki/Section_230): no description found
- [Atlas Struts](https://youtube.com/shorts/SFKM-Rxiqzg?si=Dgl7Ey38F4Mu6V5B): Can&#39;t trip Atlas up! Our humanoid robot gets ready for real work combining strength, perception, and mobility.
- [TheBloke/deepseek-coder-33B-instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF): no description found
- [GitHub - Ninyago53/webaudiobook](https://github.com/Ninyago53/webaudiobook.git): Contribute to Ninyago53/webaudiobook development by creating an account on GitHub.
- [Forge Is Not Using ComfyUI as A Backend · lllyasviel/stable-diffusion-webui-forge · Discussion #169](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/169#discussioncomment-8431103): Recently some people begin to spread misinformation about Forge using ComfyUI as a backend. This is false, harmful to the community, and harmful to the efforts of our engineering team. The backend ...

  

---


### LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1205853975589429293) (1 messages): 

- **Meet BUD-E, an Open-Source Voice Assistant**: `@spirit_from_germany` announced **BUD-E**, a low-latency, naturally sounding voice assistant that operates fully offline on a standard gaming laptop. They encourage everyone to join and contribute, sharing details in a [blog post](https://laion.ai/blog/bud-e/) and inviting people to their [Discord community](https://discord.com/invite/MDTpftKbpv) to help develop BUD-E further.
- **Tweet Alert for BUD-E Launch**: LAION took to [Twitter](https://fxtwitter.com/laion_ai/status/1756293407855485002?t=HzWRilPFjLu0Cc9DaRjEmw&s=19) to announce the launch of **BUD-E**, emphasizing its natural voice and offline capabilities, while seeking collaborators to join the Discord and assist in the project.

**Links mentioned**:

- [BUD-E: Enhancing AI Voice Assistants’ Conversational Quality, Naturalness and Empathy | LAION](https://laion.ai/blog/bud-e/): &lt;p&gt;AI voice assistants have revolutionized our interaction with technology, answering queries, performing tasks, and making life easier. However, the stilted...
- [Tweet from LAION (@laion_ai)](https://fxtwitter.com/laion_ai/status/1756293407855485002?t=HzWRilPFjLu0Cc9DaRjEmw&s=19): We present BUD-E:   A naturally sounding open-source voice assistant that runs on a standard gaming laptop with low latency, without requiring an internet connection.  Join our Discord & help us build...
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/invite/MDTpftKbpv): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1206127763128459334) (5 messages): 

- **Wasserstein Loss Mystery**: `@yoavhacohen` inquired about the claim that the discriminator in a project uses Wasserstein loss, but couldn't find evidence in the [GitHub repository's code](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py).
- **Full Stack Talent on Display**: `@fashionista8894` offered their services as a full stack designer and developer, sharing their [online portfolio](https://shinobi8894.onrender.com/) with the community.
- **Scientific Article Shared**: `@helium__` provided a [link to a scientific article](https://www.science.org/doi/10.1126/sciadv.adl4000), though no context or discussion followed the post.
- **Seeking MAGVIT Reproduction Guidance**: `@lostneko` seeks technical advice for reproducing the MAGVIT V2 model, referencing the [GitHub repository](https://github.com/lucidrains/magvit2-pytorch) as their starting point.
- **Paper on Agent Foundation Models Introduced**: `@vrus0188` shared a link to an arXiv paper titled "**An Interactive Agent Foundation Model**" listing the [authors and the abstract](https://arxiv.org/abs/2402.05929) for those interested in the topic.

**Links mentioned**:

- [Shinobi](https://shinobi8894.onrender.com/): no description found
- [An Interactive Agent Foundation Model](https://arxiv.org/abs/2402.05929): The development of artificial intelligence systems is transitioning from creating static, task-specific models to dynamic, agent-based systems capable of performing well in a wide range of application...
- [generative-models/sgm/modules/autoencoding/losses/discriminator_loss.py at main · Stability-AI/generative-models](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py): Generative Models by Stability AI. Contribute to Stability-AI/generative-models development by creating an account on GitHub.

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1205841083536253009) (77 messages🔥🔥): 

- **Chatbot Model Comparison**: User `@mares1317` suggests opening two tabs to **compare different AI models** directly.
- **iPad App Sensitivity Issues**: `@tylersavage` critiques the **Perplexity iPad app** for its overly sensitive thread exit feature when holding the iPad's sides.
- **Perplexity's API Discussion Space**: `@boyn_` inquires about a channel dedicated to **developers using Perplexity's API**; `@mares1317` directs them to an existing channel on Discord.
- **Perplexity vs. Other AI Models**: `@tauist.` introduces [AIswers](https://www.aiswers.com/), a site where users can compare **Perplexity's performance** against other AIs.
- **Beta Testing for iOS App Closed**: Users discuss the **Perplexity iOS app**; `@icelavaman` confirms no new iOS beta testers are currently being accepted.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1176526177050054766/1193673004811563158): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1176526177050054766/1176526177050054766): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1197511473749032981): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1161802929053909012): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1205951344876326913): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1111786888626438245/1205936844613619712): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1198438847877480629): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1107686562357063883): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Sponge Bob Imagination GIF - Sponge Bob Imagination Rainbow - Discover &amp; Share GIFs](https://tenor.com/view/sponge-bob-imagination-rainbow-gif-6957879033178108845): Click to view the GIF

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1205924093652766750) (11 messages🔥): 

- **Channel Guidance by me.lk**: `@actisenergy` expressed uncertainty, and `@me.lk` clarified that the “sharing” channel is for sharing notable results obtained using Perplexity.
- **Clearing Up Confusion about Discord Bots**: When `@emullie` used the `/dream` command, `@me.lk` informed them that the **discord bots are discontinued**.
- **Sharing Perplexity Search Results**: Users shared links to their Perplexity.ai search results, including `@nejjad`, `@deepanshumehta`, `@buttros`, `@bioforever`, `@austind1313_49718`, and `@w00dh3n`, presenting various topics from ethics in AI to generating images.
- **Reminder to Make Threads Public**: In response to `@bioforever` sharing a link, `@me.lk` reminded to **make sure the thread is public** by pressing the share button for visibility.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1205933768225984564) (9 messages🔥): 

- **Clarification on model function calls**: `@sourya4` sought clarification on whether to use the **messages field** for all inputs with `mixtral-8x7b-instruct`, and confirmed that **function calling is not supported** for this model. They also inquired about **latency** compared to GPT-4 and the possibility of using Mistral for **latency reduction**.
  
- **Inquiry about Mistral 32k Context Length**: `@sourya4` referenced the Perplexity AI feature roadmap [link](https://docs.perplexity.ai/docs/feature-roadmap) and asked for an **update on the availability** of **Mistral 32k context length**, last updated 2 months ago.

- **Handling API Rate Limit Errors**:  `@johang_11693` encountered a **429 HTTP status error** alleging a request rate limit exceeded while using the API through App script. The error message mentioned OpenAI despite using Perplexity.

- **Possible Causes for Rate Limit Error**: `@clay_ferguson` shared that encountering such an error might be due to **running out of credits** on OpenAI, while acknowledging this might not be the same on Perplexity and could actually mean exceeding the genuine rate limit.

- **Rate Limit Error Resolution with Delay**: `@johang_11693` resolved the **rate limit error** by **adding a millisecond delay** to the App script after verifying enough credits were available and giving a nod to **GPT-4 for the fix**.

**Links mentioned**:

[Feature Roadmap](https://docs.perplexity.ai/docs/feature-roadmap): no description found

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1205993757657268304) (9 messages🔥): 

- **The Power of H100 for AGI**: User `@andreaskoepf` discussed the computational capabilities of the H100, suggesting that with the right model and a sufficient number of these GPUs, it might be possible to achieve AGI. They referenced an [AI Impacts article](https://aiimpacts.org/brain-performance-in-flops/) which offers a range of FLOPS estimated to replicate human brain activity.
- **Celebrating GPU-Powered Supercomputing**: `@andreaskoepf` admired the "GPU church" photo depicting the MareNostrum supercomputer in Barcelona that houses V100 and MI50 GPUs. The photo was originally posted by `<@745592110245478452>` on [Twitter](https://twitter.com/johannes_hage), highlighting the impressive architecture housing advanced computing nodes.
- **Stanford's MLSys Seminars for Machine Learning Enthusiasts**: `@ericauld` shared a link to a YouTube playlist featuring [seminars from Stanford's MLSys group](https://www.youtube.com/playlist?list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq), which covers a variety of topics in machine learning systems.
- **AI Hardware Insights from Stanford MLSys**: Following up, `@iss_llm` highlighted a specific seminar, [Notes on AI Hardware - Benjamin Spector | Stanford MLSys #88](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86), as being particularly pertinent and possibly interesting to the forum's discussions.
- **Anticipation for CUDA-MODE 5 Session Recording**: In response to `@freakgoy` asking about a recorded version of the "`CUDA-MODE 5: Going Further with CUDA for Python Programmers`" event, `@jeremyphoward` replied that the recording should be available by Monday. The event details were shared by `<@neurosp1ke>` on [Twitter](https://x.com/neurosp1ke/status/1756340500116754448?s=20).

**Links mentioned**:

- [MLSys Seminars](https://www.youtube.com/playlist?list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq): no description found
- [Tweet from Andreas Köpf (@neurosp1ke)](https://x.com/neurosp1ke/status/1756340500116754448?s=20): CUDA-MODE 5: Going Further with CUDA for Python Programmers  Writing tiled kernels that leverage shared memory and thread synchronization 🚀.  Speaker: @jeremyphoward Sat, Feb 10 12:00 PM PST / 9:00 P...
- [Notes on AI Hardware - Benjamin Spector | Stanford MLSys #88](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86): Episode 88 of the Stanford MLSys Seminar Series!Notes on AI HardwareSpeaker: Ben SpectorAbstract:This week, one of our hosts -- Ben Spector -- is subbing in ...
- [Brain performance in FLOPS](https://aiimpacts.org/brain-performance-in-flops/): The computing power needed to replicate the human brain&#039;s relevant activities has been estimated by various authors, with answers ranging from 1012 to 1028 FLOPS. Details Notes We have not invest...

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1205861739275747378) (1 messages): 

- **Introducing Accelerated Computing Online**: User `@tfsingh` announced the creation of [Accelerated Computing Online](https://acceleratedcomputingonline.com), an online environment to execute Triton kernels serverlessly. The project, hosted on GitHub ([tfsingh/aconline](https://github.com/tfsingh/aconline)), allows users to run code on a T4 GPU and is available as a lighter alternative to the robust Lightning platform.

**Links mentioned**:

[ACO](https://acceleratedcomputingonline.com): no description found

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1205942870222053427) (38 messages🔥): 

- **Experiments Reveal Simple and Fast Solutions**: `@artste` achieved **the same output** with a new `experiment_A2`, applying a function to constants as suggested by `@719599526448463933`, which now aligns closely with `experiment_M`. The user expressed realization and satisfaction that the simplest solution turned out to be the fastest, detailed in [their updated notebook](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb).
  
- **Memory Coalescing Recommended**: `@cudawarped` suggested coalescing memory reads/writes for performance improvements and shared a [relevant notebook](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.ipynb) on the topic.

- **NVIDIA NPP Discussions**: `@zippika` shared their preference for using NVIDIA NPP for CUDA operations and mentioned creating a Torch C++ extension for it, including functions like remap, dilate, erode, etc. `@cudawarped` commented on the potentially questionable performance of NPP, while `@morousg` considered the idea of comparing NPP with their own library for performance benchmarks.

- **Numerical Stability in CUDA MatMul**: `@andreaskoepf` inquired about the numerical stability and minor differences in results when using simple fp16 CUDA matmul in batched-inference versus individual sequences. `@_tvi_` and `@cudawarped` discussed the non-determinism and order of computations, highlighting the impact on repeated runs with the same inputs.

- **Ownership of Independently Developed Extensions**: `@zippika` is unsure about the public release of their Torch C++ extension developed in their free time. `@morousg` outlined conditions under which code may typically be considered personal property and expressed interest in the potential to learn from `@zippika`'s experiences to make their library Python-accessible.

**Links mentioned**:

- [lecture2/lecture3/cuda_rgb_to_gray_refactor.ipynb at cuda_rgb_to_gray_refactor_notebook · artste/lecture2](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb): lecture 2 - 2024-01-20. Contribute to artste/lecture2 development by creating an account on GitHub.
- [cuda_mode_lectures/lecture3/rgb_to_grey.ipynb at rgb_to_grey · cudawarped/cuda_mode_lectures](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.ipynb): Material for cuda-mode lectures. Contribute to cudawarped/cuda_mode_lectures development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1205853420758638612) (11 messages🔥): 

- **CUDA Environments, Modules, and Device Properties**: `@morgangiraud` tested CUDA availability and device properties using Torch, displaying details such as device count, device names, and device capabilities. They showed two available NVIDIA Graphics Devices supporting peer access and the output of the device properties, including the P2P connectivity matrix from the Nvidia cuda samples p2pBandwidthLatencyTest.
  
- **Distributed Matrix Multiplication Issue**: In a multi-GPU environment, `@morgangiraud` identified an issue where copying a tensor from one GPU to another resulted in incorrect data. The code demonstrates distributed matrix multiplication that works when copying via CPU but fails with direct device-to-device transfer, as highlighted by the incorrect output in the shared results.

- **Looking for Multi-GPU Testers**: `@morgangiraud` requested anyone with access to a machine with two or more GPUs to test the provided distributed matrix multiplication code to verify if the issue occurs on other setups as well.
  
- **FAISS Embedding Vector Error Inquiry**: `@akshay_1` mentioned encountering an error while embedding vectors in FAISS(colbert) and noted that finding a solution could be costly due to the need for trial and error.

- **CUDA Graph Issue Speculation**: In response to a shared tweet by `@jxmnop` about an issue with Torch compile, `@marksaroufim` speculated it could be related to a CUDA graph issue but required a reproducible example to confirm.

- **Distributed Worker Timeout Debug Suggestion**: Addressing `@akshay_1`’s FAISS error, `@uwu1468548483828484` suggested the error might be due to a distributed worker not reaching an allreduce call, causing a timeout. To debug, they recommended running with GDB to inspect which worker hangs.

**Links mentioned**:

[Tweet from jack morris (@jxmnop)](https://x.com/jxmnop/status/1755397683471143172): welp. this is what happened when i tried to use torch compile

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1205962994186461254) (2 messages): 

- **Deep Dive into CUDA**: `@andreaskoepf` announced **CUDA MODE Lecture 5: Going Further with CUDA for Python Programmers**, inviting everyone to join the informative session shortly.
- **Lecture Link Provided**: `@jeremyhoward` shared the [lecture's Discord link](https://discord.gg/6UQXQYZp) for participants to access the CUDA MODE Lecture 5.

**Links mentioned**:

[Join the CUDA MODE Discord Server!](https://discord.gg/6UQXQYZp): CUDA reading group | 4068 members

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/) (1 messages): 

ericauld: Very interested, though I just realized I'm like a month late
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1206339445746442250) (3 messages): 

- **User requests tutorial on report analysis**: `@smexy3` commented on `<@325883680419610631>`'s video, suggesting it would be more helpful if it included a guide on **how to read the report and identify fusion/optimization opportunities**.
- **New instructional video incoming**: In response to `@smexy3`, `@marksaroufim` confirmed that the **next video**, addressing the mentioned topic, **will be released on March 2**.
- **Anticipation for the upcoming content**: `@smexy3` expressed excitement about the announcement of the **future video**, gratefully acknowledging `@marksaroufim`'s update.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1205802595331870770) (28 messages🔥): 

- **Speedy Conversion Praise**: `@dangfutures` expressed satisfaction with the **awq gguff converter**, rating it as **"10/10 fast"**.
- **HuggingFace Outages Concern**: `@c.gato` experienced an app crash that appeared related to a HuggingFace outage, despite it being a **local training job**. `@nanobitz` and `@nruaif` responded with advice, suggesting a **downgrade to version 0.7.1** and discussing potential causes like an open socket.
- **HuggingFace Downtime Frustrations**: Users `@noobmaster29`, `@rtyax`, and `@c.gato` commented on HuggingFace's server outages, with `@rtyax` noting the service **came back up briefly before going down again**.
- **Alternate Solutions for Model Inference**: `@noobmaster29` inquired about using vllm for local inference and mentioned **TensorRT**, seeking feedback on the fastest solution.
- **Exploring Extended Model Parameters**: `@xzuyn` asked the community if anyone has experimented with **qlora** on the Mistral model, specifically with a 16k max length, and wondered about the VRAM usage for such a setup.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1206096778823213096) (10 messages🔥): 

- **Mixtral Quantization Quandary**: `@jaredquek` identified an issue with an outdated **peft version (0.7.1)**, which was resolved by upgrading to **0.8** to support quantized Mixtral. `@casper_ai` confirmed that with the upgrade, things started to work again.
- **LlamaFactory Adopts ShareGPT Format**: `@faldore` pointed out how LlamaFactory has adapted the ShareGPT format in its [repository documentation](https://github.com/hiyouga/LLaMA-Factory/blob/91d09a01ac3b5da29d284b8d51cdfe4252b391e0/data/README.md?plain=1#L89), suggesting it as a potential enhancement for other projects.
- **Discussion on Naming Conventions**: While discussing LlamaFactory's adaptation of the ShareGPT format, `@le_mess` expressed a preference for not using the ShareGPT name directly for such solutions.
- **Clarifying Tools Description**: `@nanobitz` inquired about the purpose of the "tools description," to which `@faldore` responded that it's intended for functions within the context of LlamaFactory's documentation.

**Links mentioned**:

- [Standards](https://xkcd.com/927/): no description found
- [LLaMA-Factory/data/README.md at 91d09a01ac3b5da29d284b8d51cdfe4252b391e0 · hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/91d09a01ac3b5da29d284b8d51cdfe4252b391e0/data/README.md?plain=1#L89): Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM) - hiyouga/LLaMA-Factory

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1206028366994481232) (27 messages🔥): 

- **Fine-Tuning Chat Models with Historical Data**: `@smithclay` inquired about fine-tuning chat models using large volumes of historical newspaper articles, questioning the necessity of a conversational dataset for fine-tuning. `@yamashi` advised that a Q/A pair dataset can be generated from the historical data and instructed to match the old style.

- **Cost Concerns When Generating Quality Models**: User `@noobmaster29` reacted with "Rip wallet" to the suggestion of generating a fine-tuning dataset, implying concerns about cost, to which `@yamashi` responded, hinting at the financial commitment required for working with Large Language Models (LLMs).

- **Alternative Approaches for Cost-Effectiveness**: `@nafnlaus00` suggested using a quantized version of Mixtral on consumer-grade hardware to economize, describing a method to fine-tune using a supplied prompt which imbues the style of the 1800s into the model's responses.

- **Fine-Tuning Practicalities and Cost Efficiency Debate**: As `@smithclay` clarified their understanding of the suggested multistage fine-tuning process, `@dangfutures` seconded `@yamashi`'s earlier point on not compromising on model quality for cost savings.

- **Seeking Guidance for Local Server Configurations**: `@siafu7795` asked for assistance on how to use a specific configuration from Helix for training a locally run server of Mistral-7b-instruct, with `@le_mess` eventually confirming that following the axolotl GitHub repo instructions should work.

- **Fine-tuning Learning Resources Request**: `@formidoboi` expressed their newness to fine-tuning and asked the community for resources to learn more about the process, though no responses were provided within the given history.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1180827321704390657/1180827321704390660/1206197887575531561): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [helix/api/pkg/dataprep/qapairs/qapair_config.yaml at main · helixml/helix](https://github.com/helixml/helix/blob/main/api/pkg/dataprep/qapairs/qapair_config.yaml): Create your own AI by fine-tuning open source models - helixml/helix
- [axolotl/helix-mistral-instruct-v1.yml at new-long-running · lukemarsden/axolotl](https://github.com/lukemarsden/axolotl/blob/new-long-running/helix-mistral-instruct-v1.yml): Go ahead and axolotl questions. Contribute to lukemarsden/axolotl development by creating an account on GitHub.

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1206045637657759794) (1 messages): 

- **Weekend Learning with Text-to-SQL**: `@jerryjliu0` announced a new **video tutorial** on advanced text-to-SQL orchestration. The [YouTube video](https://www.youtube.com/watch?v=L1o1VPVfbb0), titled *"LLMs for Advanced Question-Answering over Tabular/CSV/SQL Data (Building Advanced RAG, Part 2)"*, guides viewers through composing a simple-to-advanced query pipeline over tabular data.

**Links mentioned**:

[LLMs for Advanced Question-Answering over Tabular/CSV/SQL Data (Building Advanced RAG, Part 2)](https://www.youtube.com/watch?v=L1o1VPVfbb0): In the second video of this series we show you how to compose an simple-to-advanced query pipeline over tabular data. This includes using LLMs to infer both ...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1205920247379660800) (5 messages): 

- **Exploring Multi-Hop Queries for RAG**: The ability to answer multi-hop queries is a significant step in advanced Retrieval-Augmented Generation (RAG) systems. The work by Tang et al. introduces the first [dataset for multi-hop queries](https://t.co/Hqx1KKOqYv) to aid in the benchmarking of advanced RAG models.
- **Fine-tuning Mistral-7B**: @lmarsden from @helixml discussed the potential of fine-tuning [Mistral-7B to memorize knowledge](https://t.co/J6JP1gycJm), which could enable the model to reason about complex questions without relying on RAG, a topic that has recently gained attention on Hacker News (HN).
- **Mini Course on QA over Tabular Data**: LlamaIndex's new mini-course offers a detailed overview of building query pipelines that combine text-to-SQL with RAG. This course presents three levels of complexity for constructing [simple-to-advanced query pipelines](https://t.co/BS0VkZjbZI).
- **Implementing Guardrails in Advanced RAG**: For user-facing applications, setting up advanced RAG involves additional layers for content moderation, topic guidance, and hallucination prevention. These [input/output filters](https://t.co/THRzAAFtF0) are crucial for maintaining quality and safety.
- **Webinar on Advanced Techniques for Tabular Data Understanding**: The latest webinar focuses on advanced tabular data understanding with LLMs, featuring two papers and authors, including the one on [Chain-of-Table](https://t.co/MpKxqK33AN) with a comprehensive list of contributing researchers and their associated work.

**Links mentioned**:

[Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://t.co/MpKxqK33AN): Table-based reasoning with large language models (LLMs) is a promising direction to tackle many table understanding tasks, such as table-based question answering and fact verification. Compared with g...

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1205951630323879976) (53 messages🔥): 

- **Exploring Text to SQL Capabilities**: User `@ottomation` inquired about experimenting with LlamaIndex to generate metadata and data dictionaries for undocumented SQL tables. They are looking for assistance on generating column descriptions using column names, schema, and values.
- **Seeking Chat Session Continuity Solutions**: `@tbhaxor.com` asked how to maintain the context of the chat session across multiple windows, similar to ChatGPT. `@cheesyfishes` provided a solution with a link to documentation on [chat stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores) and suggested using `SimpleChatStore`.
- **Clarifying LlamaIndex's Pricing and Availability**: Queries were raised about whether LlamaIndex is free and open source, with `@cheesyfishes` confirming it is open source and pointing to the official website for [more information](https://www.llamaindex.ai/).
- **Efficient Keyword Extraction Tactics**: User `_shrigmamale` sought assistance on extracting keywords such as "last years," "excels," "sales" from texts. `@bin4ry_d3struct0r` recommended prompt engineering for such tasks.
- **Mock Objects for Testing Vector Stores**: `@7leven` was on the lookout for dummy objects for testing vector stores. `@cheesyfishes` offered a code snippet using `Document.example()` to create a static document for testing operations.

**Links mentioned**:

- [LlamaIndex - Data Framework for LLM Applications](https://www.llamaindex.ai/): LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).
- [Chat Stores - LlamaIndex 🦙 v0.10.1](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores): no description found
- [LlamaIndex - Features, Pricing &amp; Use Cases](https://www.toolsforhumans.ai/ai-tools/llamaindex): LlamaIndex is a data management software aimed at enhancing Large Language Model (LLM) applications. It streamlines data ingestion, indexing, and data analysis through its user-friendly query interfac...

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1205939325133393940) (2 messages): 

- **Dive into Video Processing Innovations**: `@andysingal` shared an article titled *Video Revolution: GPT4V and LlamaIndex Unleashed*, discussing the breakthrough **Multimodal RAG architecture** that fuses OpenAI GPT4V with LanceDB VectorStore. The piece heralds a new wave of efficiency and versatility in how we interact with video content. [Read more](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30)
- **Whisper Gets Supercharged**: `@denverbitch` developed a technique to significantly increase the speed of Whisper and is open to collaborating on writing or answering questions related to their enhancement.

**Links mentioned**:

[Video Revolution: GPT4V and LlamaIndex Unleashed](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30): Ankush k Singal

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1205811906057871390) (23 messages🔥): 

- **Seeking Solutions**: User `@vithan` requested assistance with **scktlearn and pandas**, expressing difficulty in explaining the issue via text and asked `@himalaypatel.` for a voice call to provide better help.
- **Tutorial Time**: `@a404.eth` shared a **YouTube tutorial** titled "Unlock the Power of LangChain: Deploying to Production Made Easy", showing how to deploy a PDF RAG using LangChain and **UnstructuredIO** to **DigitalOcean** for production. The standalone link was provided for mobile users: [Watch the tutorial here](https://youtu.be/CbBIwVxjdP8).
- **Infinite Loop Query**: `@vvm2264` described a challenge with an essay generator **agent** seemingly reusing a tool infinitely, potentially running up against OpenAI rate limits, and asked for advice on how to prevent this behavior.
- **Coding with a Cast**: After `@_adjekofori` revealed a broken leg, `@johnny2x2` humorously suggested that it leaves more time for coding and shared that they are currently learning **AWS**.
- **Implementation Inquiry**: User `@damianj5489` asked if there's a repository with notebooks including examples from the **LangChain** Python documentation, aiming for interactive learning rather than straight copy-pasting of examples.

**Links mentioned**:

- [🦜️🔗 Langchain](https://python.langchain.com): no description found
- [LangChain](https://www.langchain.com/): LangChain’s flexible abstractions and extensive toolkit unlocks developers to build context-aware, reasoning LLM applications.
- [Unlock the Power of LangChain: Deploying to Production Made Easy](https://youtu.be/CbBIwVxjdP8): In this tutorial, Austin Vance, CEO and co-founder of  @FocusedLabs , will guide you through deploying a PDF RAG with LangChain to production! In this captiv...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1206033805270585415) (1 messages): 

- **Inquiry about Disabling Playground Feature**: User `@gitmaxd` sought advice on **disabling the playground** on deployed endpoints using the code snippet: `add_routes(app, my_app_chain, disabled_endpoints=["playground"])`. No responses to the query were provided in the given messages.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1205916966230167583) (8 messages🔥): 

- **Selfie Experiment Launched**: `@dondo.eth` introduced an open source project named **Selfie** aiming to enhance text generation with personal data via an OpenAI-compatible API, emphasizing context-rich outputs. The project repository can be explored for contributing or testing at [Selfie on GitHub](https://github.com/vana-com/selfie).

- **Intellifs Python Library Debuts**: `@synacktra` created **Intellifs**, a new python library/tool inspired by the aifs library, enabling local semantic search. The tool is open for contribution on GitHub at [Intellifs Repository](https://github.com/synacktraa/intellifs).

- **ArtFul App Launches**: `@vansh12344` announced the launch of **ArtFul - AI Image Generator**, an app providing access to various AI models like Kandinsky and DALL-E for generating original AI art, without sign-up or usage limits, and entirely free with ad viewing. The app is available on the Google Play Store at [ArtFul App Link](https://play.google.com/store/apps/details?id=com.projecthit.artful).

- **Merlinn Product Unveiled**: `@david1542` shared the launch of **Merlinn**, a product aimed at helping teams swiftly resolve production incidents with the assistance of an LLM agent and leveraging LangChain behind the scenes. More information is available on their website [Merlinn](https://merlinn.co/).

- **Triform Platform Beta Testing**: `@igxot` announced the early beta of **Triform**, a platform for hosting and orchestrating Python scripts, integrated with LangChain, and invited users to sign up for a free permanent account for production use via beta testing. Getting started with Triform is outlined at [Triform Sign Up](https://triform.ai) and their documentation can be accessed at [Triform Docs](https://triform-docs.readthedocs.io/).

**Links mentioned**:

- [100% Local Tiny Vision Model - Very Quick](https://youtu.be/jToWjbhdoEg): In this video I&#39;m going over Moondream1 a 1.6b Small Vision and Text Gen Model.Github Links:▸ https://github.com/vikhyat/moondreamMore Content from me:▸ http...
- [GitHub - BCG-X-Official/agentkit: Starter-kit to build constrained agents with Nextjs, FastAPI and Langchain](https://github.com/BCG-X-Official/agentkit): Starter-kit to build constrained agents with Nextjs, FastAPI and Langchain - BCG-X-Official/agentkit
- [GitHub - synacktraa/intellifs: Content-Aware File System.](https://github.com/synacktraa/intellifs): Content-Aware File System. Contribute to synacktraa/intellifs development by creating an account on GitHub.
- [GitHub - vana-com/selfie: Enhance text generation with personal data via an OpenAI-compatible API, seamlessly integrating with local or hosted LLMs for context-rich outputs.](https://github.com/vana-com/selfie): Enhance text generation with personal data via an OpenAI-compatible API, seamlessly integrating with local or hosted LLMs for context-rich outputs. - vana-com/selfie
- [Merlinn - Resolve incidents fast using AI](https://merlinn.co/): Investigate production incidents efficiently using AI; Empower your team by an AI agent that knows your environment.
- [Triform - Unleashing AI Potential](https://triform.ai): no description found
- [Welcome to Triform Documentation &mdash; Triform 0.1 documentation](https://triform-docs.readthedocs.io/): no description found

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1206167818341584907) (2 messages): 

- **Spotlight on Automatic Object Detection**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=W4T7zHluzaM) titled "Automatic Object Detection", highlighting how to use zero-shot object detection with the MoonDream Vision Language Model.
- **Tutorial on Chatting with Documents Using Various Tools**: `@datasciencebasics` posted a [tutorial video](https://youtu.be/2IL0Sd3neWc) that provides a guide on creating a Retrieval Augmented Generation UI using ChainLit, LangChain, Ollama, & Mistral.

**Links mentioned**:

- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM): We are going to see how we can do automatic object detetction using zero shot object detection and moondream vison langugae model#llm #ml #ai #largelanguagem...
- [Chat With Documents Using ChainLit, LangChain, Ollama &amp; Mistral 🧠](https://youtu.be/2IL0Sd3neWc): In this video, I am demonstrating how you can create a simple Retrieval Augmented Generation UI locally in your computer. You can follow along with me by clo...

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1205870669716267020) (18 messages🔥): 

- **Subscription Model Explained**: `@djioliat` queried about the Mistral Discord chatbot's subscription model. `@mrdragonfox` clarified that it is **one model for all users**, you pay per token used, and the deployment is scaled as needed without custom deployments for individual users.

- **Resource Requirements for Mistral 7B**: `@mihail2132` asked about the RAM requirements for running Mistral 7B 0.2, stating that 40GB RAM was insufficient on a laptop. `@donjuan5050` suggested using a quantized model like the one provided on [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF), which only needs a few GBs of RAM.

- **Discussion on Quantized Models**: `@mrdragonfox` responded with the point that **quantized models are not ideal for all use cases**. They emphasized that small models like 7b can run in fp16 and still provide decent performance.

- **MistralAI's Provision for Structured Output**: `@mrdomoo` inquired about how MistralAI plans to handle structured output. `@mrdragonfox` responded with a reference to a previous message for information, suggesting that the question has been addressed earlier.

- **Clarifying RAM Usage**: `@mihail2132` sought clarification on the expected RAM usage for the standard model, while `@sublimatorniq` suggested that **40GB of RAM should be enough**, with `@mrdragonfox` adding that it depends on both the model and batch size.

**Links mentioned**:

[TheBloke/Mistral-7B-Instruct-v0.2-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1205855711851057153) (2 messages): 

- **GPU Investment vs. Rental for Production**: `@i_am_dom` expressed that using GPUs from Google is not a feasible strategy for production due to cost effectiveness. They explain that owning the hardware like **A100s 40GB** could be more economical in the long run.

- **Breaking Down the Cost of GPU Ownership**: `@i_am_dom` continued their analysis by breaking down the cost of GPU ownership, explaining after **70000 computational units**, buying a GPU would pay for itself, excluding electricity. This equates to around **half a year** of continuous use.
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1205985547772756060) (1 messages): 

- **Inquiry about Docker Setup for Mistral AI**: User `@norphiil` asked the community if anyone has created a `docker_compose.yml` to simplify the deployment of **Mistral AI** as a Docker REST API. They requested assistance and thanked in advance anyone who could provide help.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1206238280480325682) (3 messages): 

- **Mistral Chatbot Successfully Installed**: User `@1mbc` reported successful installation of **Mistral** on Cloudfare AI maker, but observed that the model couldn't recognize its own origin.
- **ChatGPT's Self-Awareness Compared**: In response, `@dawn.dusk` reassured `@1mbc` that it's normal for models like **GPT-4 and Mistral** not to be self-aware, similar to how GPT-4 doesn't know its own identity.
- **Learning to Use Mistral**: For `@1mbc`'s question about first steps towards building a personalized assistant, `@dawn.dusk` provided a [Datacamp tutorial link](https://www.datacamp.com/tutorial/mistral-7b-tutorial) on using **ChatGPT** which includes writing prompts and exploring use cases.
- **Personal Assistant Development Advice**: `@tom_lrd` suggested that creating a "personal assistant" with Mistral is complex and recommended starting with simpler tasks, hinting at considering the **Retrieval Augmented Generation (RAG)** for data integration into models.

**Links mentioned**:

[Mistral 7B Tutorial: A Step-by-Step Guide to Using and Fine-Tuning Mistral 7B](https://www.datacamp.com/tutorial/mistral-7b-tutorial): The tutorial covers accessing, quantizing, fine-tuning, merging, and saving this powerful 7.3 billion parameter open-source language model. 

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1205902151155454012) (4 messages): 

- **Collaborative Chatbot Experience on Discord**: User `@jakobdylanc` introduced a collaborative LLM prompting feature for Discord that allows users to talk to LLMs such as OpenAI, Mistral, and more, alongside friends. The [bot's GitHub page](https://github.com/jakobdylanc/discord-llm-chatbot) features support for various LLMs, vision support, streamed responses, and boasts a succinct implementation in just 200 lines of code.
- **Mistral 7b Outshines Its Peers**: `@cognitivetech` shared an article showcasing how *Mistral 7b Instruct v0.2 Q8 GGUF* outperforms other models that are rated higher on leaderboards specifically in creating *comprehensive bulleted notes*. The details can be found in the write-up at [Hacker Noon](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk).
- **Enhanced Web Search Feature Acknowledged**: `@miscend` acknowledged the superior web search feature provided by the solution shared by @jakobdylanc, comparing it favorably to LibreChat and inquired about setting up a different API key specifically for Mistral to use both OpenAI and Mistral models.
- **Cross-Language Source Mapping Exploration**: `@sublimatorniq` made a brief mention indicating interest or activity in cross-language source mapping, although the context and specifics of the discussion were not provided.

**Links mentioned**:

- [GitHub - jakobdylanc/discord-llm-chatbot: Collaborative prompting • Supports OpenAI, Mistral, ollama, oobabooga and more • Vision support • Streamed responses • 200 lines of code 🔥](https://github.com/jakobdylanc/discord-llm-chatbot): Collaborative prompting • Supports OpenAI, Mistral, ollama, oobabooga and more • Vision support • Streamed responses • 200 lines of code 🔥 - jakobdylanc/discord-llm-chatbot
- [no title found](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk>): no description found

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1206217223010656268) (2 messages): 

- **Greeting Exchange**: User `@elpo_55` simply said "hi".
- **API Timeout Issue**: `@oumar7842` is seeking assistance with an issue where the API is generating a very long output that results in a timeout. They are inquiring if there is something that can be done to resolve this.
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1205936226750693427) (10 messages🔥): 

- **Innovative Serverless Vector DB**: `@nuvic_` shared [TurboPuffer](https://turbopuffer.com/), a cost-effective, serverless vector database based on S3, highlighting its efficiency and simplicity with warm queries for 1 million vectors taking about 10 seconds to cache. Comparisons were made with LanceDb, but `@nuvic_` clarified that TurboPuffer's main selling point is its S3 base, while LanceDb's is its open-source nature and ease of management.

- **Interview with Untapped Capital GP**: `@btdubbins` pointed users to an interesting interview on [Cognitive Revolution podcast](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/) featuring Yohei Nakajima, discussing collective intelligence and AI's role in enhancing mutual understanding. 

- **Google's AI Disruption Concern**: `@swyxio` found a farsighted 2018 Google memo that referred to AI as a serious business risk, which seems prescient in retrospect. The content can be accessed through the shared [TechEmails](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww) tweet.

- **ChatGPT's Influence in College Admissions**: Sharing a [Forbes article](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/), `@swyxio` discussed the trend of students using ChatGPT for college applications and the resulting use of banned words that may alert admissions committees. 

- **Banned Words for ChatGPT**: `@lightningralf` humorously suggested giving ChatGPT the list of banned words to avoid its overuse in academic settings as noted in the previous Forbes article shared by `@swyxio`.

**Links mentioned**:

- [turbopuffer](https://turbopuffer.com/): turbopuffer is a vector database built on top of object storage, which means 10x-100x cheaper, usage-based pricing, and massive scalability
- [Tweet from Internal Tech Emails (@TechEmails)](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww): Google engineer: AI is a serious risk to our business  Dec 26, 2018
- [Did You Use ChatGPT On Your School Applications? These Words May Tip Off Admissions](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/): Students who’ve turned to ChatGPT for help writing their school applications are turning back to people to make that work sound more human—and schools just can’t keep up.
- [AI &amp; Identity, from East &amp; West, with Yohei Nakajima GP at Untapped Capital and BabyAGI Creator](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/): In today&#x27;s episode Yohei Nakajima, GP at Untapped Capital and Creator of BabyAGI, returns to the show to discuss collective intelligence, identity, and how AI can h…

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1205956859459018822) (8 messages🔥): 

- **Hugging Face Experiences Downtime**: `@_jp1_` reported that **Hugging Face (HF)** is down, indicating potential reliance on HF's services within the community.
- **Debating HF's Role as Critical Infrastructure**: `@philipmay` questioned if **Hugging Face** could be considered critical infrastructure, sparking a discussion on the reliance on external platforms for model storage and operations.
- **Considering Alternatives to Hugging Face**: `@_jp1_` brought up past attempts to shift infrastructure to store weights, results, and datasets on **S3** but found HF's free integrated services more convenient despite potential reliability concerns.
- **Future Monetization Concerns for Hugging Face**: `@philipmay` speculated about a future where **Hugging Face** may start charging for access to models or for downloads, indicating a need for the community to consider financial implications.
- **Phantine Shares a Thought on Algorithms**: `@phantine` mentioned an algorithmic idea without specifics, referencing an efficient use of sparsity, and pointed to a conversation for further justification; however, the provided link was not retrievable (`<<<null>>>`).

**Links mentioned**:

[‎Gemini - chat to supercharge your ideas](https://gemini.google.com/app/d208129c63f63536): Bard is now Gemini. Get help with writing, planning, learning, and more from Google AI.

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1205879938717327480) (1 messages): 

- **Spin the Wheel with German**: `@philipmay` questions whether the SPIN (self-play) method applies to German on a **Mixtral model**. They shared the [official GitHub link](https://github.com/uclaml/SPIN) for the SPIN technique's implementation.

**Links mentioned**:

[GitHub - uclaml/SPIN: The official implementation of Self-Play Fine-Tuning (SPIN)](https://github.com/uclaml/SPIN): The official implementation of Self-Play Fine-Tuning (SPIN) - uclaml/SPIN

  

---



### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages): 

rabiat: Interesting thought 🙂
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1206310598732812408) (3 messages): 

- **OpenAI Release Teaser**: `@res6969` hinted at a potential new **OpenAI release** that could be announced soon, suggesting the timeframe to be **tomorrow or Tuesday**.
- **Sources Cited at a Gathering**: The information about the upcoming release was shared by `@res6969` who **heard from people at a party**.
- **Anticipation Builds Among Users**: In response to the news, `@.psychickoala` expressed curiosity, playfully asking **"What is it haha"** to learn more about the speculated release.
  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1205796795171610684) (2 messages): 

- **What's Up with <@748528982034612226>?**: User `@teknium` expressed curiosity about what `<@748528982034612226>` might be doing currently.
- **<@748528982034612226> Goes Off Grid**: In response, `@atlasunified` mentioned that `<@748528982034612226>` has been **off grid**, with no further details provided.
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=W4T7zHluzaM
  

---



---




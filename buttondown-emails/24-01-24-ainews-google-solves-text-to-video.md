---
id: 80cf371a-cbd9-42db-ba0a-b43efaba411e
title: Google Solves Text to Video
date: '2024-01-25T05:36:26.994654Z'
original_slug: ainews-google-solves-text-to-video
description: >-
  **Google Research** introduced **Lumiere**, a text-to-video model featuring
  advanced inpainting capabilities using a Space-Time diffusion process,
  surpassing previous models like Pika and Runway. **Manveer** from
  UseScholar.org compiled a comprehensive list of code evaluation benchmarks
  beyond HumanEval, including datasets from **Amazon Science**, **Hugging
  Face**, and others. The **TheBloke** Discord community discussed API usage and
  deployment of models such as **Mistral-7B**, **LLava**, and the concept of
  "frankenmerging" multimodal models. The **Nous Research AI** Discord
  highlighted fine-tuning challenges, learning rate strategies, and the emerging
  **RestGPT** model for RESTful API control of LLMs. The **Mistral** Discord
  focused on the design of **Mistral Instruct** as an autocomplete system.
companies:
  - google-research
  - amazon-science
  - huggingface
  - mistral-ai
  - together-ai
  - useScholar.org
models:
  - mistral-7b
  - llava
  - mistral-instruct
  - restgpt
topics:
  - text-to-video
  - inpainting
  - space-time-diffusion
  - code-evaluation
  - fine-tuning
  - learning-rates
  - api
  - model-deployment
  - multimodality
  - autocomplete
people:
  - manveer
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 1/23/2024. We checked **19** guilds, **291** channels, and **4199** messages for you. Estimated reading time saved (at 200wpm): **348 minutes**.

## Lumiere - text to video

Enter [Lumiere from Google Research](https://lumiere-video.github.io/). Every part of this video is computer generated:

https://www.youtube.com/watch?v=wxLr02Dz2Sc

In particular I would draw your attention to [their inpainting capabilities](https://lumiere-video.github.io/#w-node-_3f4320ee-e950-98f9-f142-c6b14cdf4fd4-47edbc52) - watch the syrup pour on the cake and *stay there*:

 ![image.png](https://assets.buttondown.email/images/e85d9a0c-37c2-4e0e-b9f3-e417e071fd6c.png?w=960&fit=max) 

This is a step above anything we've yet seen coming out of Pika and Runway. This seems to come from a Space-Time diffusion process:

![image.png](https://assets.buttondown.email/images/6153a6fc-c233-4bc8-b412-b2c6d0a219c5.png?w=960&fit=max) 

which we think Einstein would particularly enjoy.

## Code Evals beyond HumanEval

In other news, Manveer of UseScholar.org is collating a comprehensive list of all evals, including some code ones we haven't heard of:

- [https://github.com/amazon-science/cceval](https://github.com/amazon-science/cceval)
- [https://infi-coder.github.io/inficoder-eval/](https://infi-coder.github.io/inficoder-eval/)
- [https://evalplus.github.io/leaderboard.html](https://evalplus.github.io/leaderboard.html)
- [https://leaderboard.tabbyml.com/](https://leaderboard.tabbyml.com/)
- [https://huggingface.co/datasets/mbpp](https://huggingface.co/datasets/mbpp)
- [https://huggingface.co/datasets/nuprl/CanItEdit](https://huggingface.co/datasets/nuprl/CanItEdit)




--

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Reheating the Old Flamewar**: A fiery debate erupted around AI and roleplay, spurred by `@frankenstein0424` and `@kalomaze`, steering the conversation towards moderation and the lighter side of things with humor.
  
- **The Rat Pack's Digital Cards**: Rat-themed character cards flooded `#general`, enticing a range of reactions from `@rogue_variant`, `@mrdragonfox`, and `@.justinobserver`.

- **Model Mayhem: Seeking API Assistance**: `@priyanshuguptaiitg` navigated the labyrinth of running models like **Mistral-7B** through an API, aided by `@itsme9316` and `@kalomaze`, among others. Missing tokenizers and fine-tuning generated much buzz in the hive.

- **Need for Speed: GPUs & Ai Model Rentals**: The community dove deep into GPUs, discussing storage on platforms like runpod.io and pondering over NVIDIA's A100 GPU end-of-life, alongside practical nuances of renting GPUs for AI processing.

- **Discussion of LLM Deployment and API Usage**: `@frankenstein0424` opened a dialogue seeking insights on deploying AI models for bot hosting, with responses pointing towards utilizing `SillyTavern` for connecting LLM APIs such as [together.ai](https://together.ai) and [mistral.ai](https://mistral.ai).

- **Probing Mergekit's Possibilities**: `@222gate` investigated the potential of **mergekit** to connect gguf models while also seeking wisdom on the daring endeavor of fusing vision and non-vision AI models.

- **Frankenstein's Model**: Conversations center on **LLava**'s multimodal abilities and the aspiration of **frankenmerging**—`@selea` clarifies LLava's working and discusses the challenges inherent in model cross-integration and training.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Antique Data for Modern Testing**: One user is testing models using Q&As from antique school books, questioning the reliability of models on current datasets.
- **Learning Rates for LLMs Discussed**: A consensus is suggesting to start with learning rates around 1e-5, referencing rates from previous architecture papers, and adjusting based on early epoch observations.
- **RestGPT Earns Recognition**: [RestGPT](https://github.com/Yifan-Song793/RestGPT) is gaining attention as an example of LLMs controlling applications via RESTful APIs, showcasing the expanding capabilities of LLMs in real-world interfacing.
- **Inference and Fine-Tuning Quirks**: Various users have noted issues with inference methods, including sporadic `EOS` token inclusions, as well as OOM issues during LLM fine-tuning, with suggestions to sort data by decreasing length to troubleshoot.
- **Code-Related Evaluations for LLMs**: Benchmarks like `HumanEval`, `MBPP`, and `DeepSeek` are being used for LLM evaluations, along with a shared concern about the maintenance of the Hugging Face code and open PRs.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral Instructs on Autocomplete Design**: `@i_am_dom` provided clarification that **Mistral Instruct** is designed as an autocomplete system which would suggest omitting tags for plain text input. Sophia Yang's association with **Mistral** garnered attention in the community, confirmed by her responsive emoji.

- **Deployment & docker with vLLM**: **vLLM** supports using a local model file with the `--model` flag, and for docker users, the HuggingFace cache directory is bind-mounted which eliminates the need to redownload the model on Docker rebuilds.

- **Mixtral 8x7b Summarization Issues**: Users reported unpredictable response cutoffs in summarization tasks with **Mixtral 8x7b**, despite stable VRAM, and changing prompt syntax was suggested as a partial fix. Meanwhile, **JSON-formatted API responses** remain a challenge and Mistral's **7B models are now deployable through Amazon SageMaker JumpStart**.

- **Finetuning Challenges and Recommendations**: Attempting to finetune the **Mixtral** model appears costly and complex with mixed success, while **ColbertV2** is recommended for training embeddings models. Both prompt optimization and fine-tuning were also discussed as methods to improve results.

- **Sharing Code for Debugging and Client Package Discussions**: `@jakobdylanc` shared a link to their code on GitHub to debug an issue with the "openai" python package, which led to a discussion about the comparability of **Mistral's client library**. The conversation included the possibility of transitioning to Mistral’s package for its lightness but raised concerns about compatibility with vision models.

- **Philosophical and Math-Transformers Engagement**: A user’s philosophical inquiry did not garner traction, while another suggested combining mathematical theory with transformer models and the A* algorithm to produce new mathematical concepts, which reflects the community's creative and theoretical discussions.

**Key links mentioned**:

- [Mistral 7B models on SageMaker](https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/)
- [Discord LLM Chatbot code snippet](https://github.com/jakobdylanc/Discord-LLM-Chatbot/blob/ec908799b21d88bb76f4bafd847f840ef213a689/llmcord.py#L166)
- [Mistra - Overview on GitHub](https://github.com/mistra)
- [OpenAI python package](https://github.com/openai/openai-python)



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Ubuntu Users Overcome libclblast.so.1 Hurdle**: Ubuntu 22.04 users faced a missing `libclblast.so.1` error in LM Studio, which was tackled by creating symbolic links as a fix.
- **Apple Neural Engine Integration in LM Studio**: Discussions in LM Studio probed the utilization of Apple's Neural Engine via Metal API, and the "asitop" tool was suggested for monitoring.
- **Mixing and Matching AI Models with LM Studio**: An inquisitive approach to integrate the Retrieval-Augmented Generation (RAG) with LM Studio was met with suggestions pointing to third-party applications and setup help.
- **Model Behavior and Performance Variance**: LM Studio users grappled with model inconsistency issues, sharing tips like reducing "Randomness" or "Temperature" settings, and queried about model difference, such as between Dolphin versions 2.5 and 2.7, although the Discord link with specifics was not accessible in the provided content.
- **CodeShell's GPU Acceleration Conundrum**: Users reported that GPU acceleration was greyed out for CodeShell in LM Studio, with a possible workaround involving renaming the model file to insert "llama," but with uncertain results.
- **Hardware Enthusiasts Wrestle with VRAM Display Errors**: One user's Nvidia 3090 displaying "0Bytes" of VRAM kicked off discussions on hardware specifications, budget-fitting setups for running models like Mixtral, and stability configurations for offloading workloads to GPU.
- **Enticing Intel GPU Support on the Horizon?**: A GitHub pull request hinted at upcoming support for Intel GPUs in llama.cpp, potentially boosting LM Studio's hardware compatibility.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

**Logit Distillation's Progress and Voice Synthesis Challenge**: Discussions revealed progress in **logit distillation** using GPT-4 logits with success in backfilling strategies. However, adding custom tokens for voice synthesis to LLMs, as high as 8k, would require extensive pretraining, as shared by participants like `@ex3ndr`, `@le_mess`, and `@stefangliga`.

**Jupyter SSL Woes and Self-Rewarding Language Models**: SSL issues with Jupyter in the Latitude container surfaced without a solution, leading `@dctanner` to utilize SSH port forwarding. Interest in **Self-Rewarding Language Models** sparked discussion, with a [PyTorch implementation](https://github.com/lucidrains/self-rewarding-lm-pytorch) shared by `@caseus_`.

**DPO Dataset Loading Success, Strategy Struggles, and Local Dataset Queries**: Members discussed overcoming DPO dataset loading issues using a PR, with `@dangfutures` using a micro batch size of 1 amidst out-of-memory errors. There was a collaborative effort to address prompt strategies and finetuning with llava models, indicating the **Axolotl framework's** flexibility, referenced by `@caseus_`, `@noobmaster29`, and `@gameveloster`.

**Insight into Optimal LoRA Hyperparameters and Dataset Overlap Confirmation**: A shared [Lightning AI article](https://lightning.ai/pages/community/lora-insights/) provided insights on effective LoRA hyperparameter usage, as `@noobmaster29` and `@c.gato` discussed alpha, rank, and batch size variations. Dataset overlap concerns between dolphin and openorca datasets were confirmed, signaling data redundancy awareness.

**YAML Configuration and Prompt Tokenization for RLHF**: RLHF projects encountered a `KeyError` within YAML configurations, but a resolution via new `type` formats (`chatml.argilla` and `chatml.intel`) was found and shared by `@alekseykorshuk`. Configurations for local datasets and prompt tokenization strategy updates were also discussed, emphasizing the evolving nature of these components.

**Cog Configurations for ML Containers**: `@dangfutures` shared a **Cog configuration guide** detailing the use of CUDA "12.1", Python "3.11", and Python packages installations for machine learning containers, as per the [Cog's documentation](https://github.com/replicate/cog/blob/main/docs/yaml.md). This practical snippet demonstrates active community guidance on infrastructure setup.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

**Byte-Level BPE Enables Multilingual LLM Responses**: The **Llama 2 model** generates responses in multiple languages using *byte-level BPE*, which supports Hindi, Tamil, and Gujarati.

**Mamba's Scalability Questioned**: Enthusiastic debate unfolded over **Mamba's** potential to scale and replace Transformers, with a lack of evidence concerning its performance at larger scales provoking skepticism among technical users.

**Google Steals the Show with Lumiere**: Google Research's space-time diffusion model for video generation, **Lumiere**, attracted attention, despite concerns over dataset size and data advantages.

**First-of-its-kind Conference on Language Modeling**: Excitement buzzed around the announcement of the inaugural **Conference on Language Modeling** at the University of Pennsylvania, promising to bring deep insights into language modeling research.

**MoE Implementation Challenges and Parallelism**: A developer shared a [pull request](https://github.com/EleutherAI/gpt-neox/pull/1129) to implement **Mixture of Experts (MoE)** in **GPT-NeoX**, voicing conundrums on validating MoE with single GPU limits and seeking insights into parallelism optimizations, while another [pull request](https://github.com/EleutherAI/gpt-neox/pull/1105) scrutinizes the potential of fused layernorm in performance enhancements.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Quick Model Training Estimates**: A training time inquiry from `@astropulse` regarding a tiny model of size 128x128 led `@nodja` to estimate a couple of days' runtime on a dual 3090 rig, referencing Appendix E of an unspecified paper for further details.
- **GPT-4 Caps Confusion**: User `@helium__` discussed reduced token caps for GPT-4 with others like `@astropulse` confirming they have encountered similar constraints.
- **Implications of Image Scale on Model Performance**: According to `@thejonasbrothers`, ImageNet models with resolutions under 256x256 tend to underperform, advocating for larger image resolutions despite increased training times.
- **Safe Multimodal Dataset Discussions**: A conversation led by `@irina_rish` regarding the safety and integrity of datasets for multimodal model training saw participation and solution-seeking by `@thejonasbrothers`, `@progamergov`, and others.
- **First Language Modeling Conference Promoted**: `@itali4no` informed about the Conference on Language Modeling (CoLM) that is scheduled to happen at the University of Pennsylvania. A pertinent [tweet](https://fixupx.com/colm_conf/status/1749881065055924363) provides more details about the event.

- **Innovations in Reward Modeling**: A paper shared by `@thejonasbrothers` suggests Weight Averaged Reward Models (WARM) as a solution for reward hacking in LLMs, find the details in the linked [paper](https://arxiv.org/abs/2401.12187).
- **Advancing Unsupervised Video Learning**: `@vrus0188` showcased a paper on VONet, an unsupervised video object learning framework outperforming contemporary techniques, with the corresponding code available on [GitHub](https://github.com/hn).




---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **HuggingFace Introduces New Perks and Releases**: Community members are engaging with new activities and opportunities highlighted by HuggingFace, including a new channel for high-level contributors, the second chapter of the *Making games with AI* course, and a performance breakdown on how Gradio was optimized. Furthermore, `transformers v4.37` features new models and a 4-bit serialization, while Transformers.js now supports running Meta's SAM model in the browser.

- **Enthusiasm and Challenges in Open-Sourcing and Machine Learning**: Open-source contributions remain a vibrant part of the community's spirit. Users report difficulties with ONNX model exports and seek starter guides for learning machine learning -- being directed to a useful guide on Hugging Face. Another user is creating a privacy-conscious transcription tool for sensitive audio consultations, intending to use Hugging Face's transformers and **pyannote.audio 3.1**, combining it with Go and protobuf definitions.

- **AI Innovation and Collaboration Showcase**: The channel features a variety of AI projects and intellectual discussions, including *InstantID* for identity-preserving generation and Yann LeCun's endorsement. Users also share information about Hedwig AI's new video platform and inquire about AI background effects used in a YouTube video. Open-source contributions like QwenLM's journey with Large Language Models are presented with related resources.

- **Creators Flaunt Their Latest AI Tools and Studies**: Community members show off enhancements such as a faster `all-MiniLM-L6-v2`, scripts for PCA in embedding comparisons, and projects for detecting fakes visually. An enhancement of the Open LLM Leaderboard with Cosmos Arena gets a nod, and tools like *HF-Embed-Images* for easy image dataset embedding, 3LC for ML training and data debugging, and Gabor Vecsei's GitHub repositories are highlighted.

- **Peering into the Diffuser-Discussions**: Karras's improvements in DPM++ generate anticipation among users, who also share reflections on diffusion scheduling and its origins, citing [k-diffusion](https://github.com/crowsonkb/k-diffusion) and a paper about diffusion-based generative models. One user is working on diagrams for diffusion models to gain a comprehensive understanding for potential reimplementation.

- **Exploring NLP and Diffusion Models**: The NLP channel discusses the nuances of model parallel training, with a guide being shared to help transition from single GPU to multi-GPU training setups. Curiosity arises about multilingual models generating responses without direct language tokens, while slow inference issues with DeciLM-7B models prompt users to seek speed optimization solutions.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **RPA AI Efficiency and Dark Mode Dominance**: Discussions touched on the efficiency of RPA AI, while users lamented the lack of a light mode in labs, noting only dark mode availability. Frustrations also surfaced with Android microphone permission settings lacking adequate options. 

- **Dream Bot Discontinued Creating Channel Confusion**: Confirmations were made regarding channel closures, with notably the Dream bot being no longer available, leading to user confusion and a suggestion for more regular news summaries on channel updates.

- **GPT-4 vs. Gemini Pro Clarified**: Users sought to distinguish between GPT-4 and Gemini Pro models within Perplexity AI's pro version, receiving guidance on model selection settings, and prompting community managers to encourage community recognition of helpful contributions.

- **Feature Inquiry and Credit Support**: Questions arose about a potential teams feature and issues with credit support, alongside speculation regarding future app support for Wear OS in light of a potential collaboration with Rabbit.

- **Extended API and VSCode Integration Hints**: There were requests for information on increasing API rate limits for product integration, recommendations for the Continue.dev extension to integrate with VSCode, and light-hearted encouragement for Pro subscribers to donate credits to an imaginary "church of the God prompt".




---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Hackathon Heats Up with LlamaIndex**: IFTTT is hosting an *in-person* hackathon from February 2-4, featuring $13,500 in prizes, including $8,250 in cash, with the objective to **build projects that solve real problems**. The excitement is palpable and expertise is guaranteed with access to mentors. [Hackathon announcement tweet](https://twitter.com/llama_index/status/1749890655323898344).

- **Meet MemGPT for Memorable Chatbots**: MemGPT is a new OSS project, highlighted by `@charlespacker`, designed for creating chat experiences with enriched capabilities like **long-term memory and self-editing**, leveraging LlamaIndex technology for advanced AI chat solutions. It can be installed via pip, paving a path to a personalized AI experience. [OSS project spotlight tweet](https://twitter.com/llama_index/status/1749935634033348693).

- **SQLite Meets Llama-Index**: `@pveierland` asked about any existing **sqlite-vss integrations** for llama-index but no documentation or solutions could be identified during the discussions.

- **Pandas Query Engine Pandemonium**: Members discussed issues related to the **PandasQueryEngine** with open-source LLMs like **Zephyr 7b**, shedding light on the complexity of query pipelines in large language models. Documentation was shared but the pressing CSV file issue in RAG chatbot building remained largely unresolved.

- **Enhancing RAG Chatbots with Dynamic Knowledge**: `@sl33p1420` provided insights through their Medium article on how to augment RAG chatbots by integrating dynamic knowledge sources. The comprehensive guide walks readers through the nuances of model selection and server setup to chat engine construction for creating a robust LLM-powered RAG QA chatbot. [Empowering Your Chatbot article](https://medium.com/@marco.bertelli/empowering-your-chatbot-unveiling-dynamic-knowledge-sources-with-advanced-integration-e8353e85099c).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **AI Community Connections Fall Short**: A user sought more **LLM discord servers**, but received no recommendations, highlighting a potential gap in community resource sharing.
- **AI Behavior Benchmark**: The comparison of AI diversity to human complexity prompted discourse, underlining the notion that a variety of AI behaviors might emerge from distinct designs and environments.
- **Seeking the Best Tools**: Queries for effective **LLM evaluation & monitoring tools** for a **GPT-4**-based chatbot were raised but went unanswered, indicating a demand for such resources.
- **Image AI Scrutiny**: Questions about **Dall.E's** image handling capabilities were asked, however, there was no conclusive discussion on the specifics or causes of the issues.
- **AGI Control Argument**: Control over AGI dominated conversations with questions around who will maintain authority over such technology and considerations about its potential uses.

- **File Upload Constraints Clarity**: Clarifications were made about limits for file uploads in Custom GPT (up to **20 files**, **512MB** each, and **2 million tokens** for text files), while discussing strategies to bypass the restrictions, such as merging documents.
- **GPTs Marketplace Vanishing Act**: An inquiry was made about missing CustomGPTs on GPTs Marketplace, which remained unresolved, signaling a possible need for transparency or technical support.
- **Word Processing with Grimoire GPT**: The development of a word processor using Grimoire GPT within ten minutes was shared, showcasing the rapid implementation capabilities of GPT-based applications.
- **Custom GPT Network Troubles**: Network errors following responses from a custom GPT were reported with no solutions presented, highlighting continuing technical concerns within the community.
- **Thread Intricacies in GPT**: Confusion surfaced when a file from one GPT thread seemingly influenced another, suggesting possible file handling issues across threads, with the community awaiting confirmation on expected behavior.

- **Enhanced Context Management for AI**: A concern regarding the handling of chat logs for extracting information was mentioned, specifically about the significant impact the size and format of logs can have on the AI's performance in this area.
- **Refining AI Assistance via Prompts**: Suggestions for prompt ideas suitable for organization or executive assistance roles were discussed, with emphasis on tailoring Custom GPT prompts to include user background info for enhanced performance.
- **Championing Clear Communication Goals**: Advice on refining tasks within the description fields of a custom GPT was shared, with the recommendation to articulate clear objectives and desired outcomes, whether seeking AI or human collaboration.
- **Simplicity vs. Effectiveness in AI Command**: A user advised focusing on clear goals over the 'most effective' language when guiding AI, suggesting a pragmatic approach to achieve desired outcomes.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Prompt Perplexity? Template Tinkering To The Rescue**: Community collaboration identifies a discrepancy in templates between a local and demo instance of a model, suggesting **f-strings** and newline formatting for better compatibility with DiscoLM models. Guidance includes reference to [DiscoLM German 7b v1](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1) and community gratitude for support in navigating LLM intricacies.

- **Translation Evaluation and Predictive Musings**: The implementation of **Lilac** for translation quality and [Distilabel](https://github.com/argilla-io/distilabel) for filtering bad translations is discussed, though GPT-4 costs are mentioned as a concern. **Llama-3** predictions emphasize a 5-trillion-token pretraining focused on multilingual capability, with a hat tip to [advanced context chunking research](https://arxiv.org/abs/2310.10638), and a new **German LM** with a 1-trillion-token dataset announces its impending debut, hinting at significant compute demands.

- **Mistral Molds New Paths**: A project similar to the **Mistral embedding model** is launched on [GitHub](https://github.com/SebastianBodza/Embedding_Training) utilizing Quora data, with discussions around hosting on Hugging Face or GitHub and whether to craft a BigGraph or Table Embedding model. Also, Voyage's **new code embedding model**, `voyage-code-2`, is spotlighted for its advancements in semantic code retrieval, detailed in their [blog post](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/).

- **Axolotl Adoption Anecdotes**: Troubleshooting for **Axolotl** includes advice on dataset integration using [supported formats](https://github.com/OpenAccess-AI-Collective/axolotl#dataset) and referencing in the [Axolotl config](https://github.com/OpenAccess-AI-Collective/axolotl#config), managing GPU recognition in Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and a hint to seek specialized help on the [Axolotl Discord](https://discord.gg/2zHuDM2C). A problematic newline issue with DiscoLM German model prompts a community chipset fix, [amending the `config.json`](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/commit/560f972f9f735fc9289584b3aa8d75d0e539c44e) to resolve output glitches as discussed on [Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/discussions/4).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Karpathy Sheds Light on Tech's Human Impact**: Andrej Karpathy's new [blog post](https://huggingface.co/posts/clem/970025506569107) discusses the difficulty those outside the tech industry face in adapting to rapid technological changes. Anxiety and discomfort are common emotional responses to the pace of innovation.

- **Perplexity's Complex Progress Visualized**: A [tweet](https://x.com/madiator/status/1749501859151405124?s=61) from `@madiator` shows the non-linear development trajectory of the AI model Perplexity over a span of three months.

- **Scaling Down Model Size While Keeping Cognition Intact**: Research on training smaller language models (LMs) reveals potential to maintain grammar and reasoning capabilities, as discussed in the [TinyStories paper](https://arxiv.org/abs/2305.07759).

- **Discord Enlists AI for Smarter Notifications**: Discord has begun using large language models (LLMs) for summarizing community messages to create notification titles, signaling a potential shift in privacy policy considerations.

- **Breakthrough in Image Generation by Stability AI**: [Stability AI](https://buttondown.email/ainews/archive/ainews-rip-latent-diffusion-hello/) has developed a diffusion model capable of generating megapixel-scale images outright, which could signal the end of traditional latent diffusion techniques.

- **Lucidrains Set to Tackle SPIN and Meta Paper**: `@lucidrains` is preparing implementations of SPIN and a new Meta paper approach in separate projects, with [self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch) being the repository to watch for progress updates.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChainJS Experimental Foray**: `@ilguappo` has shared an in-development project on [GitHub](https://github.com/anaxios/langchainjs-workers) entitled LangChainJS Workers which controversially strays from best web API practices but explores a novel endpoint for emoji reactions in Discord messages. They are also tackling the steep learning curve of TypeScript and its integration into the current project.

- **Teaming Up for RAG Systems**: An interest in end-to-end **Retrieval-Augmented Generation (RAG)** solutions has been voiced by `@alvarojauna`, seeking collaborations or precedents, while `@allenpan_36670` has sparked a clarifying discussion on GPT chat completion's handling of message lists, with `@lhc1921` alluding to **ChatML's** prompt structures as a method for handling such data.

- **Initiating Intelligent PDF Dialogues**: `@a404.eth` broadcasted the launch of a tutorial series with Part 1 [available on YouTube](https://youtu.be/UwgZmrRAgQ4), guiding users through the creation of **Full Stack RAG systems** enabling conversations with PDF documents leveraging PGVector, unstructured.io, and semantic chunker technologies.

- **LLaMA Outshines Baklava in Artistic Judgement**: In a comparison battle of AI models, `@dwb7737` posted findings on a [GitHub Gist](https://gist.github.com/donbr/5273d55a089c644d35e8e019370c1902) showcasing **LLaMA**'s superior performance over **Baklava** in art analysis tasks.

- **Engineers Beware of Mischievous Links**: A cautionary note regarding a potential spam message posted by `@eleuss` in the langserve that included a sequence of bars, underscores, and a suspicious Discord invite link, implying the need for vigilance against such behaviors in technical communities.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **New Horizons for LLM Perf Guild**: `@jeffreyw128` kicked off 2024 with an energizing welcome and revealed intentions to expand the Discord guild through a new wave of select invitations and member referrals.
- **Eyeing the State-of-the-Art in Document Layout**: Discourse in the guild highlighted the **Vision Grid Transformer** as a cutting-edge model for understanding document layouts, particularly excel at identifying charts within PDFs as shared by `@res6969`, with the GitHub repository available [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery).
- **#share Your Knowledge**: A new channel named *#share* emerged from community collaboration, ready to house mutual knowledge exchanges, as decided by `@degtrdg` and `@jeffreyw128`.
- **Synergy Through LLM Activities**: `@yikesawjeez` touched on the vibrancy of the LLM space, pointing out engaging happenings such as paper clubs, implementation sessions, and codejams which are quite the nexus for the LLM performance aficionados.
- **Infiltrate with Intelligence**: In a light-hearted tone, `@yikesawjeez` proposed that members expand their reach and influence by bringing their LLM performance expertise to outside events.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Based on the provided messages, there isn't sufficient context or substantial technical content relevant to an engineer audience to generate a summary. Both messages appear to be informal communications without any discernible technical discussion or key points.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

- **Quest for Cloud Independence**: @floriannoell asked about **on-premise AI solutions** that do not rely on major cloud providers like AWS, GCP, or Azure, mentioning [watsonx.ai](https://watsonx.ai) as a point of reference for desired capabilities.
- **Tailoring AI to Fit the Mold**: In the process of discussing on-premise solutions, @spillai suggested @floriannoell elucidate specific AI requirements such as **pretraining, finetuning,** or **classification**, to guide the search towards a more fitting on-premise AI system.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Catch the Slim Orca Dataset on Hugging Face**: **Slim Orca dataset** is now hosted on Hugging Face, boasting **~500k GPT-4 completions** with enhanced quality through GPT-4 refinements. This dataset is noted for needing only 2/3 the computational power for performance comparable to larger data slices ([Slim Orca](https://huggingface.co/datasets/Open-Orca/SlimOrca)).
- **Training Made Efficient with Slim Orca Models**: Two models, [jackalope-7b](https://huggingface.co/openaccess-ai-collective/jackalope-7b) and [Mistral-7B-SlimOrca](https://huggingface.co/Open-Orca/Mistral-7B-SlimOrca), demonstrate the high efficiency and performance of practice on the Slim Orca subset. This advancement was shared by `@222gate` in the community chat, spotlighting the dataset's reduced computational requirement without compromising output quality.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Offline LLM Enhancement Unveiled**: The [`llm-gpt4all` version 0.3](https://github.com/simonw/llm-gpt4all/releases/tag/0.3) has been released, featuring improvements including offline functionality for models and the ability to adjust model options such as `-o max_tokens 3`. The release also incorporates fixes from community contributors.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199266845174267925) (1398 messages🔥🔥🔥): 

- **AI and RP community unite**: Users `@frankenstein0424` and `@kalomaze` engage in a heated discussion about AI and roleplay, leading to calls for moderation and jokes about the situation.
- **Obsession with rats**: The chat room `#general` has been spammed with rat-related character cards, inciting mixed reactions from users such as `@rogue_variant`, `@mrdragonfox`, and `@.justinobserver`.
- **Mistral 7B and coding with AI**: User `@priyanshuguptaiitg` seeks help running models like Mistral-7B through an API and receives directions from `@itsme9316`, `@kalomaze`, and others. They discuss difficulties with the API, mentioning issues like missing tokenizers and fine-tuning options.
- **Exploring MoEs and Mergekit**: A serious technical discussion unfolds between users `@kquant`, `@sanjiwatsuki`, and `@kalomaze`, focusing on the nuances of fine-tuning Mixture of Experts (MoE) models, their performance, and their unique challenges.
- **GPU talk and renting for AI**: The chat delves into the world of GPUs, discussing storage options on platforms like runpod.io, and the End-of-Life announcement for NVIDIA's A100 GPU. They also touch on the practical aspects of using rented GPUs for large language models.

**Links mentioned**:

- [No Way GIF - Stunned Wow Omg - Discover &amp; Share GIFs](https://tenor.com/view/stunned-wow-omg-lego-batman-lego-batman-movie-gif-7647770): Click to view the GIF
- [Tweet from Jon Durbin (@jon_durbin)](https://x.com/jon_durbin/status/1749851101812150473?s=20): Working on an RP-enhancing DPO dataset using cinematika data, meaning the responses are human-written (but still llm augmented).  Let&#39;s see if this works 🤞🏻
- [Screenshot to HTML - a Hugging Face Space by HuggingFaceM4](https://huggingface.co/spaces/HuggingFaceM4/screenshot2html): no description found
- [openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projector · Hugging Face](https://huggingface.co/openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projector): no description found
- [Brain GIF - Brain - Discover &amp; Share GIFs](https://tenor.com/view/brain-gif-338783462287106063): Click to view the GIF
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF): no description found
- [All Your Base Are Belong To Us Cats GIF - All Your Base Are Belong To Us Cats Cat - Discover &amp; Share GIFs](https://tenor.com/view/all-your-base-are-belong-to-us-cats-cat-all-your-base-meme-gif-15689149): Click to view the GIF
- [Create a Network Volume](https://docs.runpod.io/docs/create-a-network-volume): no description found
- [Prompt-Engineering for Open-Source LLMs](https://www.youtube.com/live/f32dc5M2Mn0): Turns out prompt-engineering is different for open-source LLMs! Actually, your prompts need to be engineered when switching across any LLM — even when OpenAI...
- [Mad Men Conversing GIF - Mad Men Conversing Feel Bad For You - Discover &amp; Share GIFs](https://tenor.com/view/mad-men-conversing-feel-bad-for-you-i-dont-think-about-you-at-all-gif-11888203): Click to view the GIF
- [What Do You Mean By That GIF - What Do You Mean By That - Discover &amp; Share GIFs](https://tenor.com/view/what-do-you-mean-by-that-gif-10559980082154008380): Click to view the GIF
- [DIE ANTWOORD - RATS RULE [Music Video]](https://www.youtube.com/watch?v=lATimFFLOZ8): Die Antwoord - Mount Ninji and Da Nice Time Kid - Rats Rule/ Featuring JACK BLACK !More of these comming soon, subscribe to dont miss out!Used videos:/ DIE A...
- [Speaking with Angry Rats Baldur&#39;s Gate 3](https://www.youtube.com/watch?v=GiMvWtsMRtQ): Speaking with Angry Rats Baldur&#39;s Gate 3. You can see Baldur&#39;s Gate III Speaking with Angry Rats Scene following this video guide. Baldur&#39;s Gate III is a rol...
- [yahma/alpaca-cleaned · Datasets at Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned): no description found
- [GitHub - openai/consistencydecoder: Consistency Distilled Diff VAE](https://github.com/openai/consistencydecoder): Consistency Distilled Diff VAE. Contribute to openai/consistencydecoder development by creating an account on GitHub.
- [Neil deGrasse Tyson Explains the Simulation Hypothesis](https://www.youtube.com/watch?v=pmcrG7ZZKUc): Neil deGrasse Tyson and comic co-host Chuck Nice are here (or are they?) to investigate if we&#39;re living in a simulation. We explore the ever-advancing comput...
- [GitHub - deep-floyd/IF](https://github.com/deep-floyd/IF): Contribute to deep-floyd/IF development by creating an account on GitHub.
- [Samsung 870 QVO 8TB SSD Memory Storage | Samsung UK](https://www.samsung.com/uk/memory-storage/sata-ssd/ssd-870-qvo-sata-3-2-5-inch-8tb-mz-77q8t0bw/): Discover incredible storage with a Samsung SSD. Enjoy improved performance, easy management with Samsung Magician and awesome reliability.
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Mod...
- [Mac | Jan](https://jan.ai/install/mac/): Jan is a ChatGPT-alternative that runs on your own computer, with a local API server.
- [Port of self extension to server by Maximilian-Winter · Pull Request #5104 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5104): Hi, I ported the code for self extension over to the server. I have tested it with a information retrieval, I inserted information out of context into a ~6500 tokens long text and it worked, at lea...
- [Nature](https://www.nature.com): Nature is the foremost international weekly scientific journal in the world and is the flagship journal for Nature Portfolio. It publishes the finest ...
- [Solving olympiad geometry without human demonstrations - Nature](https://www.nature.com/articles/s41586-023-06747-5): A new neuro-symbolic theorem prover for Euclidean plane geometry trained from scratch on millions of synthesized theorems and proofs outperforms the previous best method and reaches the performance of...
- [main : add Self-Extend support by ggerganov · Pull Request #4815 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4815): continuation of #4810 Adding support for context extension to main based on this work: https://arxiv.org/pdf/2401.01325.pdf Did some basic fact extraction tests with ~8k context and base LLaMA 7B v...
- [YT Industries](https://www.yt-industries.com/products/bikes/decoy/core-3/666/decoy-mx-core-3/): Decoy MX CORE 3

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1199287771416776734) (427 messages🔥🔥🔥): 

- **Model Comparison and Usage Queries**: Users discussed their experiences with various models like `Nous Hermes` and `SanjiWatsuki/Lelantos-Maid-DPO-7B` for specific roleplay tasks. `@ks_c` found `Kunoichi dpo v2` to be the best for character interpretation.

- **Frontend Features and Lorebooks**: `@animalmachine` shared insights on the value of a lorebook feature for roleplay chats and pointed to the relevant documentation on [SillyTavern's usage of World Info](https://docs.sillytavern.app/usage/core-concepts/worldinfo/).

- **Automating Data Collection for Model Training**: `@frankenstein0424` is scripting to automate the creation of training data for their bot from website messaging, planning to gather a dataset for a highly specialized task.

- **Quantization and Model Performance**: `@keyboardking` discussed the difficulty in getting grammatically correct output from sub 10gb models and voiced concerns about whether deeper quantization renders models like `7B` suboptimal.

- **Deployment and API Choices for Bots**: `@frankenstein0424` sought advice for hosting and using models like `Mixtral AI` through external APIs, with suggestions including using `SillyTavern` frontend to connect to various LLM APIs such as [together.ai](https://together.ai/) and [mistral.ai](https://mistral.ai/).

**Links mentioned**:

- [Mistral AI | Open-weight models](https://mistral.ai/): Frontier AI in your hands
- [World Info | docs.ST.app](https://docs.sillytavern.app/usage/core-concepts/worldinfo/): World Info (also known as Lorebooks or Memory Books) enhances AI's understanding of the details in your world.
- [NeverSleep/Noromaid-13B-0.4-DPO-GGUF · Hugging Face](https://huggingface.co/NeverSleep/Noromaid-13B-0.4-DPO-GGUF): no description found
- [LoneStriker/Noromaid-13B-0.4-DPO-3.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/Noromaid-13B-0.4-DPO-3.0bpw-h6-exl2): no description found
- [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch): no description found
- [Rentry.co - Markdown Paste Service](https://rentry.org/): Markdown paste service with preview, custom urls and editing.
- [Models - Hugging Face](https://huggingface.co/models?sort=trending&search=lonestriker+Air-Striker>): no description found
- [Another LLM Roleplay Rankings](https://rentry.co/ALLMRR): (Feel free to send feedback to AliCat (.alicat) and Trappu (.trappu) on Discord) We love roleplay and LLMs and wanted to create a ranking. Both, because benchmarks aren't really geared towards rolepla...
- [TheBloke/CaPlatTessDolXaBoros-Yi-34B-200K-DARE-Ties-HighDensity-GGUF · Hugging Face](https://huggingface.co/TheBloke/CaPlatTessDolXaBoros-Yi-34B-200K-DARE-Ties-HighDensity-GGUF): no description found
- [text-generation-webui/modules/sampler_hijack.py at 837bd888e4cf239094d9b1cabcc342266fee11c0 · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/837bd888e4cf239094d9b1cabcc342266fee11c0/modules/sampler_hijack.py#L386): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui
- [How to mixtral](https://rentry.org/HowtoMixtral): Updated 12/22 Have at least 20GB-ish VRAM / RAM total. The more VRAM the faster / better. Grab latest Kobold: https://github.com/kalomaze/koboldcpp/releases Grab the model Download one of the quants a...
- [text-generation-webui/modules/logits.py at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/modules/logits.py): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui
- [The Sarah Test](https://rentry.org/thesarahtest): (by #theyallchoppable on the Ooba and SillyTavern Discord servers) See also: https://rentry.org/thecelltest The Sarah Test is a simple prompt to test a model's coherency, logical consistency, whatever...
- [Intro](https://rentry.org/thecelltest/#appendix-b-effect-of-instruct-template): Intro Changelog Performing the cell test Checking logits Effect of samplers Model Results Summary Model Results Table Closing thoughts Future improvements Appendix A: Potential problem with the prompt...

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1199289166882689024) (5 messages): 

- **Eager to Train Locally**: `@superking__` expressed enthusiasm about starting to experiment with training locally to save on compute costs before becoming proficient.
- **Credit for Implementation Queried**: A user asked `@superking__` if they were responsible for implementing a method for loading a single model, recalling their name from a pull request (PR).
- **Acknowledgement of Contribution**: `@jondurbin` confirmed he implemented the 2-adapter method, clarifying the existing code for null reference was already present.
- **Seeking Solution for a Shared Issue**: `@carlito.88` inquired about a previous issue, wondering if a resolution had been found.
- **Query on LLMS and TensorRT Conversion**: `@bycloud` questioned the community for experiences on training/finetuning large language models (LLMs) and converting them into TensorRT optimized models.
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199545474194096128) (2 messages): 

- **Inquiry on Merging gguf Models**: `@222gate` asked if anyone knows whether merging gguf's is possible using **mergekit**, expressing interest in experimenting with it despite assumptions of infeasibility.
- **Fusion of Vision and Non-Vision Models**: `@222gate` queried the community for any attempts or documentation on merging a vision model with a non-vision model, indicating a need for guidance on such cross-modality merges.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1199397063985745970) (13 messages🔥): 

- **Understanding LLava's Composition**: `@selea` explained that **LLava** models incorporate **CLIP** for image recognition; tokens from CLIP are appended before a text prompt in a language model. There's a necessity to train the model to interpret CLIP embeddings correctly.
- **Adding Multimodality is Complex**: For `@lordofthegoons` enquiry on adding multimodality, `@selea` indicated the challenge lies in training the text model to understand and utilize image embeddings efficiently.
- **Frankenmerging Models**: `@lordofthegoons` pondered about partially extracting layers from **LLava** to merge them with another model, while `@selea` admitted to not knowing much about frankenmerges but speculated on the possibility of adding CLIP understanding to another model.
- **Glitches in Frankenmerged Models**: `@selea` mentioned that even if a frankenmerge were successful, the resultant model would likely operate glitchily due to the inherent complexities of combining different systems.
- **Improving Model Training with LLava**: `@selea` proposed the idea of using **LLava** to distill accurate image descriptions and retrain the text-processing part of models like **Stable Diffusion** which currently employs a "wacky machine code" for prompting.
  ,

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1199281738669621248) (18 messages🔥): 

- **Charming Old School Data**: User `@everyoneisgross` mentioned that they purchase antique school books to manually input Q&As for model testing, expressing some skepticism with models that perform too well on modern datasets.
- **Fine-Tuning AI Models**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=n3gkZ_IRwCI) titled "Finetuning TinyLlama using Unsloth," which includes sections on data preparation, training, inference, and model saving.
- **Turning "Machine Learning" into "Money Laundering"**: `@euclaise` posted a funny tweet about a Chrome extension that replaces "machine learning" with "money laundering" and shared the [GitHub link](https://github.com/stong/ml-to-ml) for the extension.
- **Cuda Kernels Allow Non-Power-of-Two Configurations**: `@carsonpoole` discussed the advantages of writing **kernels in CUDA over Triton**, noting that configurations can be set to non-powers-of-two, sometimes yielding better performance.
- **Satirical Spin on AI Company Expectations**: `@sumo43` joked about the lofty expectations set by companies calling themselves AI companies, suggesting a play on words with "token companies" that merely generate tokens.

**Links mentioned**:

- [Tweet from cts🌸 (@gf_256)](https://fxtwitter.com/gf_256/status/1749739346150719610): twitter is more funny if you replace &#34;machine learning&#34; with &#34;money laundering&#34;. So i made a chrome extension that does this  https://github.com/stong/ml-to-ml
- [Finetuning TinyLlama using Unsloth](https://www.youtube.com/watch?v=n3gkZ_IRwCI): You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), &amp; [how to save it](#Save) (eg for Llama.cpp).**[NOTE...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1199292286236889138) (38 messages🔥): 

- **Prompt Lookup Revolution**: `@leontello` promoted efficient prompt lookup for input-grounded tasks by sharing a [mention of its significance](https://twitter.com/abacaj/status/1749612925973680426), stating it's a "free lunch" that should be utilized more.
- **Control Applications with LLMs**: `@mikahdang` highlighted [RestGPT](https://github.com/Yifan-Song793/RestGPT), a project showcasing an LLM-based autonomous agent that can control real-world applications via RESTful APIs.
- **Function Calling as the Future**: `@mikahdang` and `@teknium` held a passionate agreement on the importance of function calling for reasoning and planning as integral to the future of LLMs integration with APIs.
- **Unraveling Non-determinism in GPT-4**: A conversation led by `@burnytech` linked articles discussing GPT-4's non-determinism due to Sparse MoE, with contributions from `@stefangliga`, `@stellaathena`, and `@betadoggo` on the challenges and implications.
- **Diffusion Model Considerations**: `@mikahdang` shared research on Contrastive Preference Learning and scalability of diffusion language models, sparking a debate about their underestimation in NLP. Different views were expressed by `@_3sphere`, `@betadoggo`, and `@manojbh` regarding the potential and challenges of merging autoregressive and diffusion models for various tasks.

**Links mentioned**:

- [Are Diffusion Models Vision-And-Language Reasoners?](https://openreview.net/forum?id=fmJv8Hj0yo): Text-conditioned image generation models have recently shown immense qualitative success using denoising diffusion processes. However, unlike discriminative vision-and-language models, it is a...
- [Diffusion Language Models Can Perform Many Tasks with Scaling and...](https://openreview.net/forum?id=Qn4HEhezKW): The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it...
- [Contrastive Preference Learning: Learning from Human Feedback...](https://openreview.net/forum?id=iX1RjVQODj): Reinforcement Learning from Human Feedback (RLHF) has emerged as a popular paradigm for aligning models with human intent. Typically RLHF algorithms operate in two phases: first, use human...
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1749771913545609661): I published BagelHermes-2x34B, a Mixture of Experts model, combining @jon_durbin&#39;s bagel 🥯 and @NousResearch&#39;s Hermes2 📨  Hermes excels in math, while Bagel is superior in QA and science. So...
- [GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs](https://github.com/Yifan-Song793/RestGPT): An LLM-based autonomous agent controlling real-world applications via RESTful APIs - GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs
- [Non-determinism in GPT-4 is caused by Sparse MoE](https://152334h.github.io/blog/non-determinism-in-gpt-4/): It&rsquo;s well-known at this point that GPT-4/GPT-3.5-turbo is non-deterministic, even at temperature=0.0. This is an odd behavior if you&rsquo;re used to dense decoder-only models, where temp=0 shou...
- [Tweet from Maksym Andriushchenko 🇺🇦 (@maksym_andr)](https://fxtwitter.com/maksym_andr/status/1749546209755463953): GPT-4 is inherently not reproducible, most likely due to batched inference with MoEs (h/t @patrickrchao for the ref!): https://152334h.github.io/blog/non-determinism-in-gpt-4/  interestingly, GPT-3.5 ...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1199301000964620348) (271 messages🔥🔥): 

- **Batch Size Example Disclosed**: `@leontello` shared an **example of batched inference** for machine learning models, providing a code snippet to illustrate how to run batched prompts using a model and tokenizer from the `transformers` library on a GPU.

- **Helpful Tools for Chat Templating**: `@osanseviero` mentioned a [helpful resource](https://twitter.com/xenovacom/status/1749756308641697946) for those working with chat templates, pinpointing potential usefulness for developers.

- **Discussion on OpenAI's Logit Distillation**: "Is anyone doing logit (soft) distillation of GPT4?" `@dreamgen` queried, sparking a conversation on the availability of logits from OpenAI's API and the feasibility and methods of distilling large language models. Users debated the value and strategies of distillation, noting it as a potentially unexplored area.

- **RUGPULL Visualization App Development**: `@n8programs` is working on an application called **RUGPULL**, intended for exploring UMAP representations of corpora, with an ability to see the distance and relevance between chunks, all in an engaging, interactive graph format.

- **Qwen 72B Base vs Llama 2 70B Base Discussed**: The conversation turned towards comparing the Qwen 72B base and Llama 2 70B base models regarding their usability for fine-tuning. Some users like `@intervitens` and `@s3nh1123` mentioned issues like VRAM consumption and the advantages of multilingual support, respectively; however, the consensus seemed elusive due to a lack of extensive experimentation with Qwen.

**Links mentioned**:

- [Introducing Qwen](https://qwenlm.github.io/blog/qwen/): 4 months after our first release of Qwen-7B, which is the starting point of our opensource journey of large language models (LLM), we now provide an introduction to the Qwen series to give you a whole...
- [Brain GIF - Brain - Discover &amp; Share GIFs](https://tenor.com/view/brain-gif-338783462287106063): Click to view the GIF
- [Cat Explode GIF - Cat Explode Explosion - Discover &amp; Share GIFs](https://tenor.com/view/cat-explode-explosion-kitty-shitpost-gif-15585170105969644973): Click to view the GIF
- [Tweet from anton (@abacaj)](https://fxtwitter.com/abacaj/status/1749963250064662551): I got some concrete numbers on phi-2 DPO. You can see clear jump in model capabilities first turn and second turn for MT-bench using DPO. More epochs does not really help overall, my model was overfit...
- [GitHub - KillianLucas/aifs: Local semantic search. Stupidly simple.](https://github.com/KillianLucas/aifs): Local semantic search. Stupidly simple. Contribute to KillianLucas/aifs development by creating an account on GitHub.
- [HuggingFaceH4/open_llm_leaderboard · Discussions](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions): no description found
- [HuggingFaceH4/open_llm_leaderboard · Flagging models with incorrect tags](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/540): no description found
- [01-ai/Yi-34B · Hugging Face](https://huggingface.co/01-ai/Yi-34B): no description found
- [cognitivecomputations/dolphin · Datasets at Hugging Face](https://huggingface.co/datasets/cognitivecomputations/dolphin): no description found

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1199264792930693181) (57 messages🔥🔥): 

- **Finding the Optimal Learning Rate (LR)**: Users `@alyosha11` and `@bozoid.` discussed finding the best LR for LLMs, suggesting to use evaluation datasets or benchmarks. `@bozoid.` mentioned that learning rates from previous architectures' papers could yield decent results, while `@teknium` advised starting with a ballpark LR of around 1e-5 and adjust after observing results at 3 epochs.

- **Inference Tricks for Mistral and Llama**: User `@blackl1ght` questioned about alternative inference methods for `mistral` over `llama.cpp`, and `@.ben.com` recommended `exllamav2`. In connection, `@blackl1ght` reported an issue with the `EOS` token being included in streaming responses, which `@max_paperclips` confirmed happens sporadically, depending on the model.

- **OOM Issues and Sequence Length in Fine-tuning LLMs**: User `@besiktas` described an out-of-memory (OOM) issue encountered when fine-tuning, even when previous forward/backward passes were successful. `@yonta0098` recommended checking if longer sequences are causing the issues and maybe sorting the data by decreasing length to trigger OOM early if that's the case.

- **Discussions on LLM Evaluation and Fine-tuning**: `@rememberlenny` initiated a discussion on code-related evaluations for LLMs, and various users, including `@manveerxyz` and `@besiktas`, mentioned benchmarks such as `HumanEval`, `MBPP`, and `DeepSeek`. `@besiktas` also raised concerns about the quality of some parts of the Hugging Face code and mentioned PRs that haven't been addressed.

- **Fine-tuning Challenges and Hugging Face Problems**: `@besiktas` provided a link to a test implementation to diagnose fine-tuning memory leaks by gradually increasing the context length and described difficulties with the Hugging Face `FuyuProcessor`. This sparked a discussion about the challenges of contributing to such large-scale collaborative projects.

**Links mentioned**:

- [pretrain-mm/tests/test_model.py at 4159505915d5e15952957aa5607eadf9fc6c70cd · grahamannett/pretrain-mm](https://github.com/grahamannett/pretrain-mm/blob/4159505915d5e15952957aa5607eadf9fc6c70cd/tests/test_model.py#L129): Contribute to grahamannett/pretrain-mm development by creating an account on GitHub.
- [nuprl/CanItEdit · Datasets at Hugging Face](https://huggingface.co/datasets/nuprl/CanItEdit): no description found
- [mbpp · Datasets at Hugging Face](https://huggingface.co/datasets/mbpp): no description found
- [FuyuProcessor broken and causes infinite loop · Issue #27879 · huggingface/transformers](https://github.com/huggingface/transformers/issues/27879#issuecomment-1892579789): transformers/src/transformers/models/fuyu/processing_fuyu.py Line 618 in 75336c1 while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != ( I am not sure exa...
- [GitHub - amazon-science/cceval: CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion (NeurIPS 2023)](https://github.com/amazon-science/cceval): CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion (NeurIPS 2023) - GitHub - amazon-science/cceval: CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-Fil...
- [InfiCoder-Eval: Systematically Evaluating Question-Answering
    for Code Large Language Models](https://infi-coder.github.io/inficoder-eval/): no description found
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [Coding LLMs Leaderboard](https://leaderboard.tabbyml.com/): no description found

  ,

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1199272375431204865) (225 messages🔥🔥): 

- **Mistral Instruct Autocomplete Clarification**: User `@i_am_dom` has clarified that Mistral Instruct is autocomplete by design and suggested skipping tags for plain text input, which will prompt the model to predict completion to the input.
- **Sophia Yang Confirmed as Mistral**: User `@jarsalfirahel` expressed surprise at learning **Sophia Yang**, known from YouTube, is associated with **Mistral**. Sophia acknowledged with a thank you emoji.
- **Mistral Knowledge Base File Uploads**: User `@vivacious_gull_97921` inquired if **Mistral** supports uploading files to the knowledge base, to which `@sophiamyang` responded that it's not currently supported, suggesting the use of Mistral with other RAG tools.
- **Mistral 7B Foundation Models on Amazon SageMaker**: `@sophiamyang` shared a [blog on SageMaker](https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/), announcing the availability of **Mistral 7B models for deployment via Amazon SageMaker JumpStart**. The post illustrates how to discover and deploy the model.
- **Moderation on Mistral Discord and Future Plans**: `@sophiamyang` confirms that moderators are set up on the Mistral Discord after `@ethux` suggested the need for them due to scams. They welcomed recommendations for better moderation setups.

**Links mentioned**:

- [Cat Berg Cat GIF - Cat Berg Cat Orange Cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-berg-cat-orange-cat-swimming-gif-25177582): Click to view the GIF
- [Mistral 7B foundation models from Mistral AI are now available in Amazon SageMaker JumpStart | Amazon Web Services](https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/.): Today, we are excited to announce that the Mistral 7B foundation models, developed by Mistral AI, are available for customers through Amazon SageMaker JumpStart to deploy with one click for running in...
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [GPT-4](https://openai.com/research/gpt-4): We’ve created GPT-4, the latest milestone in OpenAI’s effort in scaling up deep learning. GPT-4 is a large multimodal model (accepting image and text inputs, emitting text outputs) that, while less ca...

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1199317600686051400) (81 messages🔥🔥): 

- **Summarization Shenanigans with Mixtral 8x7b**: User `@atom202300` reported issues with **Mixtral 8x7b** and the **Huggingface Text Generation Interface (TGI)** wherein the model would unpredictably cut off responses during summarization tasks. Despite a stable VRAM, the problem persisted for specific examples, suggesting sensitivity to prompt structure.
- **Prompt Patterns Prompt Problems**: `@sublimatorniq` and `@atom202300` discussed the effect of altering prompt syntax on summarization performance, finding that changing from square brackets to parentheses reduced premature stopping, while `@mrdragonfox` mentioned that **Mixtral** requires careful increment adjustments to prevent spamming.
- **Troubleshooting Model Stops with Seeds**: `@sublimatorniq` theorized that the use of different seeds could induce variable stopping behaviors—with one seed causing early stops and another leading to extended responses.
- **Open-Weights and Finetuning Frustrations**: `@wayne_deng` inquired about the availability of source code and finetuning possibilities for the **Mixtral** model. `@mrdragonfox` stated that many have attempted finetuning without success due to high costs and the model's complexity.
- **Mistral API JSON Response Request**: User `@madmax____` sought help with forcing the Mistral API to return responses in **JSON format**, but found the `response_format` parameter ineffective. `@akshay_1` shared a link, possibly as a solution to the issue but the context of the link was not provided.

**Links mentioned**:

- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started): no description found
- [GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference): Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1199328463606251650) (21 messages🔥): 

- **Local Model Use in vLLM**: `@shihard_85648` asked for a way to use a local model file with vLLM, having downloaded the "raw" Mistral file, and `@mrdragonfox` advised to use the `--model` flag followed by the full local path. This bypasses the need to download the model from HuggingFace, which is the usual process as explained by `@vhariational`.
- **vLLM Docker Image Clarification**: `@vhariational` referenced the vLLM documentation, indicating that the Huggingface cache directory is bind-mounted at runtime, meaning you wouldn't need to redownload the model when rebuilding the Docker image.
- **Model Path Configuration Instructions**: `@mrdragonfox` provided a detailed link to the vLLM documentation, elaborating on the engine arguments including how to specify paths for both model and tokenizer.
- **Efficient Mistral MoE Setup Inquiries**: User `@yoann_b` inquired about an economical setup for running Mistral MoE at 12token/s on hardware less than $3,000, leading to `@mrdragonfox` suggesting a 32GB M1 Mac or using two 3090 GPUs for higher performance.
- **Technical Specs and Performance Metrics**: In the discussions about configurations, `@mrdragonfox` mentioned running mistral 8x7b on an `A6000` (a 48GB VRAM GPU) with '6bpw' (bits per weight), achieving about 60 tokens/second with `exllamav2`.

**Links mentioned**:

[Engine Arguments &#8212; vLLM](https://docs.vllm.ai/en/latest/models/engine_args.html): no description found

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1199269254772555796) (7 messages): 

- **Prompt Optimizations Proposed**: `@akshay_1` suggested optimizing the prompts as an easier solution for the issues encountered with the Retriever-Augmented Generator (RAG) application.
- **Embedding Model Training vs. Cost**: Training an embedding model might not be cost-effective at a small scale, according to `@akshay_1`.
- **ColbertV2 for Embedding Training**: For those looking into training an embedding model, `@akshay_1` recommended checking out **ColbertV2**.
- **Identifying RAG App Limitations**: `@mrdragonfox` highlighted that an embedding model might need training when dealing with unique terminologies as they might not cluster effectively with intended meanings.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1199460114244051066) (2 messages): 

- **Tips for Enhanced Function Calling**: `@akshay_1` recommends using `dspy`, `SGLang`, `outlines`, and `instructor` for better function calling, stating it **works really good**.
- **Advocating for Fine-tuning**: In a follow-up, `@akshay_1` mentions that **fine-tuning on a dataset** will yield better results if one is not satisfied with the initial solutions suggested.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1199279637939884042) (11 messages🔥): 

- **Code Sharing for Error Investigation**: `@sophiamyang` asked `@jakobdylanc` to share their code because of an unfamiliar error. Jake provided a link to a specific part of their GitHub [repository](https://github.com/jakobdylanc/Discord-LLM-Chatbot/blob/ec908799b21d88bb76f4bafd847f840ef213a689/llmcord.py#L166), describing the issue with the "openai" python package used in their Discord LLM Chatbot.
- **Package Compatibility Discussion**: `@sophiamyang` questioned `@jakobdylanc` about the python client package being used, linking to Mistral's client ([Mistra - Overview](https://github.com/mistralai/client-python)), and expressed uncertainty about reproducing the error. Jakob confirmed the use of the [OpenAI python package](https://github.com/openai/openai-python) and contemplated switching to Mistral’s package for its lightweight nature despite potential issues with vision models.
- **Error Reproduction Troubles**: `@jakobdylanc` admits difficulty in reproducing the error but promises to report back to `@sophiamyang` if it occurs again.
- **Philosophical Query Lacks Response**: User `@jrffvrr` posed an existential question about the most beautiful person in the world and followed with a test message, seemingly checking functionality with no further discussion on the topic.
- **Intersection of Transformers and Mathematics**: `@stefatorus` proposed the idea of training transformer models on mathematics and using an A* algorithm to generate potentially fruitful mathematical ideas worthy of exploration.

**Links mentioned**:

- [Discord-LLM-Chatbot/llmcord.py at ec908799b21d88bb76f4bafd847f840ef213a689 · jakobdylanc/Discord-LLM-Chatbot](https://github.com/jakobdylanc/Discord-LLM-Chatbot/blob/ec908799b21d88bb76f4bafd847f840ef213a689/llmcord.py#L166): Multi-user chat | Choose your LLM | OpenAI API | Mistral API | LM Studio | GPT-4 Turbo with vision | Mixtral 8X7B | And more 🔥 - jakobdylanc/Discord-LLM-Chatbot
- [Mistra - Overview](https://github.com/mistra): Mistra has 29 repositories available. Follow their code on GitHub.
- [GitHub - openai/openai-python: The official Python library for the OpenAI API](https://github.com/openai/openai-python): The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.

  ,

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199272089702633512) (172 messages🔥🔥): 

- **Ubuntu Users Encounter libclblast.so.1 Error**: Ubuntu 22.04 users, including `@d0mper` and `@josemanu72`, were struggling with an error when opening LM Studio related to a missing `libclblast.so.1` file. After much discussion, the creation of symbolic links resolved the issue.

- **Performance Questions on Apple Silicon Neural Engine**: LM Studio's utilization of the Apple Silicon Neural Engine was a question posed by `@crd5`, where `@Aqualiteking` helped clarify that the neural engine might be used indirectly via Apple's Metal API and suggested monitoring with "asitop" tool.

- **Queries on AI Modeling and Setup**: Various users, including `@golangorgohome`, `@cloakedman`, and `@christianazinn`, exchanged info on the suitable hardware for LM Studio and alternative setups for different AI applications, such as image-to-text and local hosting of model implementations.

- **LM Studio Model Compatibility and Troubleshooting**: Users like `@bright_chipmunk_28966` and `@yagilb` discussed issues with loading certain models in LM Studio, leading to advice on updating to newer versions and checking for compatibility on platforms like HuggingFace.

- **Exploration of RAG with LM Studio**: `@elevons` inquired about integrating RAG with LM Studio, and though there were no straightforward solutions within LM Studio itself, `@heyitsyorkie` and `@thelefthandofurza` provided guidance on third-party apps and setup assistance.

**Links mentioned**:

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found
- [CLBlast/doc/installation.md at master · CNugteren/CLBlast](https://github.com/CNugteren/CLBlast/blob/master/doc/installation.md): Tuned OpenCL BLAS. Contribute to CNugteren/CLBlast development by creating an account on GitHub.
- [GitHub - HeliosPrimeOne/ragforge: Crafting RAG-powered Solutions for Secure, Local Conversations with Your Documents - V2 Web GUI 🌐 Product of PrimeLabs](https://github.com/HeliosPrimeOne/ragforge): Crafting RAG-powered Solutions for Secure, Local Conversations with Your Documents - V2 Web GUI 🌐 Product of PrimeLabs - GitHub - HeliosPrimeOne/ragforge: Crafting RAG-powered Solutions for Secure,.....
- [GitHub - john-rocky/CoreML-Models: Converted CoreML Model Zoo.](https://github.com/john-rocky/CoreML-Models): Converted CoreML Model Zoo. Contribute to john-rocky/CoreML-Models development by creating an account on GitHub.
- [Core ML Tools &#8212; Guide to Core ML Tools](https://apple.github.io/coremltools/docs-guides/): no description found

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199459310804160512) (15 messages🔥): 

- **Model Error Mystery**: User `@alex_m.` presented an issue with LM Studio where the model fails regardless of configuration, showing a JSON error with Exit code: 0. `@gustavo_60030` responded, suggesting checking a different Discord channel for possible solutions.
- **Channel Direction Confusion**: After `@alex_m.` was directed to one support channel, `@heyitsyorkie` intervened to recommend another as the appropriate place for discussing model errors.
- **AI's Unpredictable Personality**: `@cloakedman` commented on the unpredictability of AI models, remarking how the same model can provide different responses. `@fabguy` suggested that reducing the temperature setting can increase consistency in the AI's responses.
- **AI Consistency Tips**: `@cloakedman` inquired about what `@fabguy` meant by reducing temperature. `@fabguy` replied, clarifying that setting "Randomness" or "Temperature" to zero can yield consistent answers given the same seed is used.
- **Dolphin Version Differences**: `@cloakedman` asked for insights on the differences between Dolphin 2.5 and 2.7 AI versions. Although `@fabguy` provided a Discord link for detailed comparison, the link was not accessible in the summary provided.
  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1199424358419025970) (16 messages🔥): 

- **GPU Acceleration Greyed Out for CodeShell**: `@czkoko` reported that CodeShell was listed as not supported for GPU acceleration despite being supported in version 0.2.11. `@yagilb` confirmed that the architecture is not currently considered supported by the LM Studio app.
- **Potential Workaround for GPU Support**: `@yagilb` suggested a workaround by renaming the model file to include "llama" which might enable GPU acceleration, but `@czkoko` followed up saying the workaround did not change the greyed-out GPU acceleration or RoPE.
- **Conservative App Behavior Regarding GPU Acceleration**: `@yagilb` pointed out that the app errs on the side of caution by graying out options for unsupported architectures, [mentioning a previous discussion](https://discord.com/channels/1110598183144399058/1199424358419025970/1199426064720613516).
- **User Feedback on GPU Acceleration**: `@heyitsyorkie` confirmed that the GPU acceleration remains grayed out even after trying the suggested workaround and commented that the current UI state, which indicates "not supported," is clear enough.
- **Discussion Invitation**: `@yagilb` extended an invitation to continue the conversation about the GPU acceleration issue for CodeShell in a different thread, providing a [Discord channel link](https://discord.com/channels/1110598183144399058/1199424358419025970/1199425015339294750) for further discussion.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1199297985587200052) (29 messages🔥): 

- **VRAM Woes Amidst Hardware Talk**: User `@cheerful_panda_16252` reported an issue with VRAM capacity showing as "0Bytes" despite owning a Nvidia 3090 with 24GB VRAM. `@cloakedman` also expressed concern that this might affect the recommended settings of the software.
- **Call for Hardware Specs**: `@yagilb` directed users to provide their hardware specifications in order to address the VRAM capacity problem, guiding them to a specific Discord channel with a [posted link](https://discord.com/channels/1110598183144399058/1199088851973578883/1199088851973578883).
- **Potential Boost for Intel GPU Users**: A GitHub pull request shared by `@heyitsyorkie` suggests that Intel GPU users might soon see support in llama.cpp [View Pull Request](https://github.com/ggerganov/llama.cpp/pull/2690). However, `@goldensun3ds` showed skepticism regarding the timeline of this update.
- **Exploring Budget Configurations for Mixtral**: `@yoann_b` inquired about the cheapest hardware configuration capable of running Mixtral at 12t/s, mentioning the potential of an M1 Pro. `@rugg0064` contributed by clarifying the bandwidth differences between M2 Pro, M1, and high-end GPUs like the RTX 4090.
- **Hardware Compatibility Discussions and Recommendations**: `@cloakedman` shared difficulties with system crashes when offloading to a GPU, and `@bobzdar` offered troubleshooting tips including layer adjustments and prompt compression. The discussion evolved into `@cloakedman` finding a stable setting for their system.

**Links mentioned**:

- [Part 1:Building and Optimizing a High-Performance Proxmox Cluster On a Budget.](https://medium.com/@contact_45426/part-1-building-and-optimizing-a-high-performance-proxmox-cluster-on-a-budget-4fdabebe1260): In our guide for building a Proxmox cluster, we’ve primarily focused on utilizing second-hand components to cater to small producers or…
- [Feature: Integrate with unified SYCL backend for Intel GPUs  by abhilash1910 · Pull Request #2690 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/2690): Motivation: Thanks for creating llama.cpp. There has been quite an effort to integrate  OpenCL runtime for AVX instruction sets. However for running on Intel graphics cards , there needs to be addi...

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1199334019570081933) (2 messages): 

- **Thumbs Up for WhiteRabbit 33B**: User `@johntdavies` mentioned successful testing with the **WhiteRabbit 33B model (Q8)**, giving positive feedback for its performance.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (1 messages): 

senecalouck: Try it using 127.0.0.1 in the script.
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 

gciri001: Is it possible to use Langchain and MySql with LLAMA 2 withouts openAI api?
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1199378283410755666) (4 messages): 

- **Model Loading Failure Frustration**: User `@ferrolinga` encountered an error with the message "unknown (magic, version) combination" when trying to load a model. The error report included system diagnostics indicating sufficient RAM and VRAM but a potential issue with the model file itself.
- **Incorrect File Format Diagnosis**: `@draco9900` quickly identified that `@ferrolinga`'s issue stemmed from using a model that is not in GGUF format—a necessary format for LM Studio.
- **Solution Suggestion**: In response to the loading issue, `@heyitsyorkie` advised `@ferrolinga` to use GGUF files specifically and recommended searching for "TheBloke - GGUF" in the Model Explorer for optimal results, as LM Studio requires GGUF model formats, not pytorch/ggml or .bin files.
  ,

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1199349862957858816) (50 messages🔥): 

- **Logit Distillation Chat**: `@dreamgen` discusses **logit distillation** using GPT-4 logits and techniques like backfilling with open-source models and masking loss. `@nruaif` expresses interest in the topic and considers further discussion.
- **Voice Synthesis Adaptation Inquiry**: `@ex3ndr` queries about **adding custom tokens** for voice synthesis to LLMs and learns from `@le_mess` and `@stefangliga` that adding a large number of tokens, like 8k, would necessitate extensive pretraining.
- **Challenges with QLoRA Finetuning**: Several participants including `@stefangliga`, `@noobmaster29`, and `@c.gato` discuss the limitations of **QLoRA finetuning**, especially with a significant number of new tokens, stressing that simple auxiliary networks won't suffice and full embedding layer finetuning must be considered.
- **Model Finetuning Tips and Loss Evaluation**: `@ex3ndr` shares their experience with **finetuning custom tokens**, highlighting concerns with unusually high loss figures, while `@noobmaster29` and `@c.gato` provide insights on what loss metrics to aim for in different scenarios.
- **Using Special Tokens in Axolotl Finetuning**: Guidance on how to **configure special tokens** for finetuning is shared by `@faldore`, including code snippets for embedding `lm_head` and specifying `eos_token`. This was in response to `@dreamgen` pointing to `@faldore`'s success with a project like Dolphin on Hugging Face.

**Links mentioned**:

- [Magic GIF - Magic - Discover &amp; Share GIFs](https://tenor.com/view/magic-gif-26166638): Click to view the GIF
- [axolotl/src/axolotl/utils/lora_embeddings.py at dc051b861d4d0f20c673ad55ac93b2a43fa56fc4 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/dc051b861d4d0f20c673ad55ac93b2a43fa56fc4/src/axolotl/utils/lora_embeddings.py#L14): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [cognitivecomputations/dolphin-2.6-mixtral-8x7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b/tree/main): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1199302220773068800) (4 messages): 

- **Jupyter issues in Latitude container**: `@dctanner` raised a problem encountering SSL issues with Jupyter running in the Latitude container due to Cloudflare's tunneling for port forwarding. No solution was provided yet.
- **SSH Port Forwarding as a Band-Aid**: `@dctanner` is currently using SSH port forwarding as a workaround for the Jupyter issue in the Latitude container.
- **Intriguing Idea: Self-Rewarding Language Models**: `@dctanner` shares interest in incorporating the concept of [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020.pdf) into the axolotl framework.
- **Self-Rewarding Model Implementation**: `@caseus_` responds to the idea with a link to a PyTorch implementation on GitHub, [lucidrains/self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch), which is an implementation of the training framework proposed in Self-Rewarding Language Model from MetaAI.

**Links mentioned**:

[GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI](https://github.com/lucidrains/self-rewarding-lm-pytorch): Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI - GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1199349661417361499) (116 messages🔥🔥): 

- **DPO Dataset Mysteries Unraveled**: `@dangfutures` and `@c.gato` discussed issues with loading datasets for DPO, finding success with a prior pull request ([PR #1137](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1137)). `@dangfutures` mentioned overcoming an out-of-memory error by using a micro batch size of 1.
  
- **Struggling with Strategy?**: `@c.gato` helped `@dangfutures` navigate prompt strategies for datasets, sharing code snippets and a GitHub [file link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompt_strategies/dpo/chatml.py). To fix a persistent error, they advised to implement a fix from the DPO fixes branch on GitHub.
  
- **CI-CD Goodness or Local Frustration?**: `@caseus_` highlighted an automated ci-cd sanity check for remote datasets, which doesn't cover local datasets. Meanwhile, `@dangfutures` and `@c.gato` discovered that reverting to a previous commit allowed for using local datasets, despite initial errors.
  
- **LoRA Hyperparameter Head-Scratchers**: `@noobmaster29` inquired about optimal settings for LoRA's alpha and rank hyperparameters, prompting `@c.gato` to share [an article](https://lightning.ai/pages/community/lora-insights/) providing insights into their effective usage. Discussion included varying alpha and rank as well as batch size considerations during training.
  
- **Branching out for LLAMA**: `@gameveloster` and `@noobmaster29` explored finetuning with llava models, pointing to a [specific branch of axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/tree/llava) and considering whether existing configs could be adapted for this purpose. The conversation highlights the community's collaborative effort in sharing knowledge and resources.

**Links mentioned**:

- [axolotl/src/axolotl/prompt_strategies/dpo/chatml.py at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompt_strategies/dpo/chatml.py): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [axolotl/tests/e2e/test_dpo.py at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/tests/e2e/test_dpo.py): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [GitHub - OpenAccess-AI-Collective/axolotl at llava](https://github.com/OpenAccess-AI-Collective/axolotl/tree/llava): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments - Lightning AI](https://lightning.ai/pages/community/lora-insights/): LoRA is one of the most widely used, parameter-efficient finetuning techniques for training custom LLMs. From saving memory with QLoRA to selecting the optimal LoRA settings, this article provides pra...
- [configs/pretrain-llava-mistral.yml · openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projector at main](https://huggingface.co/openaccess-ai-collective/mistral-7b-llava-1_5-pretrained-projector/blob/main/configs/pretrain-llava-mistral.yml): no description found
- [Feat: Add sharegpt multirole by NanoCode012 · Pull Request #1137 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1137): Description: Allow multiple roles for input and output. NOTE: Beta and hardcoded values for now! How to use: - type: sharegpt + type: sharegpt.load_multirole Only supports conversation: (chatml|zep...
- [DangFutures/DPO_RAG · Datasets at Hugging Face](https://huggingface.co/datasets/DangFutures/DPO_RAG): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1199574838101168138) (4 messages): 

- **Dolphin Data Doubts**: `@noobmaster29` inquired if there was any overlap between the [dolphin dataset](https://huggingface.co/datasets/cognitivecomputations/dolphin) and the openorca dataset. `@dangfutures` expressed belief that there is sure to be overlap.
- **Overlap Confirmation**: Upon hearing `@dangfutures`' belief, `@noobmaster29` sought clarification and confirmed understanding that the two datasets do indeed overlap.

**Links mentioned**:

[cognitivecomputations/dolphin · Datasets at Hugging Face](https://huggingface.co/datasets/cognitivecomputations/dolphin): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1199304448703475722) (13 messages🔥): 

- **Configuration Key Error for `argilla_apply_chatml`**: `@alekseykorshuk` encountered a `KeyError: 'prompt'` when using `argilla_apply_chatml` in their YAML configuration for a project with **Reinforcement Learning Hub (RLHF)**. They initially sought assistance with this configuration issue.

- **Solution Identified in Unittests**: Later, `@alekseykorshuk` resolved the issue by discovering new `type` formats (`chatml.argilla` and `chatml.intel`) within the unittests on the main branch, and confirmed that this solution worked for them, prompting them to share the update with the community.

- **Clarification Sought on Branch Usage**: After the solution was shared, `@dangfutures` sought clarification on which branch `@alekseykorshuk` used to find the successful new `type` formats. `@alekseykorshuk` clarified they used the main branch.

- **Config Doubt for Local Datasets**: `@matanvetzler` inquired if the same configuration would apply to local datasets and requested to see the config setup. `@alekseykorshuk` assumed local datasets should work similarly by just changing the `type`.

- **Prompt Tokenization Strategy Issue and Solution**: `@pierrecolombo` reported a `ValueError: unhandled prompt tokenization strategy: intel_apply_chatml`, to which `@c.gato` responded advising to update to `chatml.intel` if using the latest commit due to breaking changes. `@pierrecolombo` acknowledged the solution with thanks.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1199469092776710174) (1 messages): 

- **Cog Configuration Guide Shared**: `@dangfutures` provided a snippet defining a **configuration for Cog**, referencing documentation on their [GitHub page](https://github.com/replicate/cog/blob/main/docs/yaml.md). The configuration is set up for GPU usage with CUDA "12.1", uses Python "3.11", and includes installation of various Python packages like `aiohttp[speedups]`, `megablocks`, `autoawq`, and more via a custom `pip install` command referencing multiple package URLs.

**Links mentioned**:

- [cog/docs/yaml.md at main · replicate/cog](https://github.com/replicate/cog/blob/main/docs/yaml.md): Containers for machine learning. Contribute to replicate/cog development by creating an account on GitHub.
- [no title found](https://download.pytorch.org/whl/cu121): no description found

  ,

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1199271969036718171) (56 messages🔥🔥): 

- **Byte-Level BPE's Multilingual Abilities**: `@synquid` explained that the **Llama 2 model** can generate responses in languages like Hindi, Tamil, and Gujarati using *byte-level BPE (Byte Pair Encoding)*, which does include tokens for these languages.

- **Seeking Code for Mistral 7b Fine-Tuning**: User `@aslawliet` requested assistance for code to fine-tune **Mistral 7b** for token classification, but did not receive a direct response within the provided messages.

- **Skepticism About Mamba Replacing Transformers**: Users, including `@stellaathena`, `@stefangliga`, and `@mrgonao`, expressed skepticism regarding **Mamba** scaling and replacing Transformers, noting the absence of evidence that it will maintain its performance at larger scales. Discussions centered on the engineering challenges and the need for more research to validate Mamba's scalability.

- **Finetuning as a Service Inquiry**: User `@kh4dien` reached out for recommendations on finetuning large language models, expressing a preference for full supervision tuning rather than methods like QLORA. `@stellaathena` suggested that running finetuning personally on rented GPUs might be a simple out-of-the-box solution.

- **Evaluating Impact of Fine-Tuning LLMs**: `@everlasting_gomjabbar` queried the community about comprehensive studies highlighting the benefits of fine-tuning Large Language Models (LLMs), suggesting that the real-world justifications for the investment in fine-tuning are often unclear. No direct response catered to this query was provided in the discussion.

**Links mentioned**:

[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020): We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human prefer...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1199288061578707014) (65 messages🔥🔥): 

- **Exploring Cryptographic Hiding in LLMs**: `@ai_waifu` shared a [paper](https://arxiv.org/abs/2401.10360) that introduces a cryptographic method to hide a secret payload in a Large Language Model's response, requiring a key for extraction and remaining undetectable without it. `@fern.bear` questioned the claim that the method doesn't modify the response distribution, arguing that some distribution must change to convey information.

- **Weight Averaged Reward Models (WARM) Introduced by Google DeepMind**: `@jacquesthibs` highlighted a [paper](https://huggingface.co/papers/2401.12187) that discusses WARM, a strategy to combat reward hacking in LLMs aligning with human preferences through RLHF, by averaging fine-tuned reward models in weight space, and shared an [author's thread](https://x.com/ramealexandre/status/1749719471806157304?s=61&t=ryK3X96D_TkGJtvu2rm0uw) for further insights.

- **Google's Realistic Video Generation Research**: `@pizza_joe` linked to a demonstration of Google Research's space-time diffusion model for video generation, with accompanying [paper](https://arxiv.org/abs/2401.12945). `@thatspysaspy` and `@ad8e` discussed the significance, with `@ad8e` noting Google's data advantage while `@main.ai` countered with information on dataset size in comparison to other models.

- **Exploration of Model Constraints in Unsupervised Learning**: `@rybchuk` asked about research regarding models learning the correct constraints to minimize loss in an unsupervised manner, with several users, including `@fern.bear`, discussing empirical risk minimization and the nature of constraints in learning.

- **First Conference on Language Modeling Announced**: `@stellaathena` shared an announcement regarding the first Conference on Language Modeling, set to be held at the University of Pennsylvania. The conference will be a gathering for those interested in language modeling research and advancements.

**Links mentioned**:

- [Tweet from Conference on Language Modeling (@COLM_conf)](https://fixupx.com/colm_conf/status/1749881065055924363): We are pleased to announce that the first Conference on Language Modeling will be held at the University of Pennsylvania in Philadelphia at the Zellerbach Theatre.     Thanks so much to UPenn CS as we...
- [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945): We introduce Lumiere -- a text-to-video diffusion model designed for synthesizing videos that portray realistic, diverse and coherent motion -- a pivotal challenge in video synthesis. To this end, we ...
- [Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding](https://arxiv.org/abs/2401.12954): We introduce meta-prompting, an effective scaffolding technique designed to enhance the functionality of language models (LMs). This approach transforms a single LM into a multi-faceted conductor, ade...
- [Active Inference as a Model of Agency](https://arxiv.org/abs/2401.12917): Is there a canonical way to think of agency beyond reward maximisation? In this paper, we show that any type of behaviour complying with physically sound assumptions about how macroscopic biological a...
- [Excuse me, sir? Your language model is leaking (information)](https://arxiv.org/abs/2401.10360): We introduce a cryptographic method to hide an arbitrary secret payload in the response of a Large Language Model (LLM). A secret key is required to extract the payload from the model&#39;s response, ...
- [Lumiere - Google Research](https://lumiere-video.github.io/): Space-Time Text-to-Video diffusion model by Google Research.
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1749646258245927405?s=61&t=ryK3X96D_TkGJtvu2rm0uw): Google Deepmind presents WARM  On the Benefits of Weight Averaged Reward Models  paper page: https://huggingface.co/papers/2401.12187  Aligning large language models (LLMs) with human preferences thro...
- [Tweet from Stella Biderman (@BlancheMinerva)](https://fixupx.com/BlancheMinerva/status/1749641148979876185)): @Wetassprior @daphneipp In the Pythia paper we explore the effect of term frequency on fact learning over the course of training. If you squint at Fig. 4, it seems like there is weak evidence that the...
- [Tweet from Alexandre Ramé (@ramealexandre)](https://x.com/ramealexandre/status/1749719471806157304?s=61&t=ryK3X96D_TkGJtvu2rm0uw): Introducing DeepMind&#39;s Weight Averaged Reward Model (WARM) for alignment via RLHF! We merge multiple reward models into one that&#39;s more reliable and robust. WARM efficiently captures the best ...
- [k-diffusion/k_diffusion/models/image_transformer_v2.py at master · crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py#L567>): Karras et al. (2022) diffusion models for PyTorch. Contribute to crowsonkb/k-diffusion development by creating an account on GitHub.

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1199417300655276032) (3 messages): 

- **Scaling Doubts on relora**: `@joey00072` raised a concern that **relora** might not scale beyond **1 billion** parameters based on hearsay from Twitter or Discord, although the original reLoRA paper does not make this claim.
- **reLoRA Paper's Limits**: `@joey00072` mentioned that the reLoRA models were only tested up to a size of **350 million** parameters, seeking further studies or feedback on scaling beyond this point.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1199413819970174996) (2 messages): 

- **Exploring Truth Representation Interventions**: User `@80melon` discussed a technique involving **intervening or negating a truth representation** in language model outputs without using patching. They then observed changes in the language model's output based on various completions.
- **Seeking Maximally Causal Truth Direction**: `@80melon` mentioned the goal of their experiment was to identify a **truth direction** that had a direct and significant impact on the language model's various outputs.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1199487242826158160) (3 messages): 

- **Working Hard on MoE Implementation**: `@xyzzyrz` is contributing to the implementation of **MoE (Mixture of Experts)** with a [pull request](https://github.com/EleutherAI/gpt-neox/pull/1129), seeking tips and feedback for deeper validation. Concerns about `mpu.get_model_tensor_parallelism_world_size()` and its relation to *PipeModelDataParallelTopology* and *axonn 3D tensor parallelism* were raised, with uncertainty about how to proceed with a single-GPU setup.

- **Single-GPU Limits Parallelism Testing**: `@xyzzyrz` expressed difficulty in testing **parallelism enhancements** due to only having access to a single-GPU node. They provided initial numbers showing a significant slowdown when increasing *num-experts* and referenced an existing MoE branch with overlapping work.

- **Fused Layernorm Pull Request Scrutiny**: They are also following the progress on another pull request regarding **fused layernorm**, found [here](https://github.com/EleutherAI/gpt-neox/pull/1105), and noted there wasn't much time difference detected which might suggest limited room for further improvements.

- **Interest in Improving Deepspeed Inference**: `@xyzzyrz` indicated willingness to work on **Deepspeed Inference** to potentially improve performance as discussed in [Issue #845](https://github.com/EleutherAI/gpt-neox/issues/845), pending availability of compute resources.

**Links mentioned**:

- [fused layernorm by yang · Pull Request #1105 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1105): Add simple util for timings Add fused layernorm kernel from Megatron
- [Add MoE by yang · Pull Request #1129 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1129): Closes #479
- [Build software better, together](https://github.com/EleutherAI/gpt-neox/pull/1086.): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [Investigate DeepSpeed Inference · Issue #845 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/845): DeepSpeed wins most inference benchmarks I see. We should test their claims on neox models. EleutherAI spends a significant amount of compute running inference, so any improvement in inference perf...

  ,

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199311393678168065) (124 messages🔥🔥): 

- **Quick Training Time Inquiry**: `@astropulse` was curious about the time required to train a tiny model in the 128x128 range. `@nodja` responded that it should take no more than a couple of days on a dual 3090 system, and suggested checking appendix E of a mentioned paper.

- **GPT-4 Token Allocations Adjusted**: User `@helium__` raised a concern over reduced caps for GPT-4, with other users like `@astropulse` confirming similar experiences.

- **Cautionary Notes on Tiny ImageNet Models**: `@thejonasbrothers` pointed out that ImageNet models with resolutions below 256x256 did not perform well, and recommended better performance is seen at larger scales with significant training time.

- **Discussion on Crafting Safe Multimodal Datasets**: `@irina_rish` initiated a conversation about ensuring the safety of datasets utilized for multimodal model training, stressing the need for solutions guaranteeing data integrity. Challenges and potential solutions were discussed by various users, including `@thejonasbrothers`, `@progamergov`, and `@.undeleted`.

- **Upcoming Conference on Language Modeling Announcement**: `@itali4no` shared the venue announcement for the first Conference on Language Modeling (CoLM), which will be held at the University of Pennsylvania, along with an invite to another user, `@chad_in_the_house`, who anticipates taking them to food trucks.

**Links mentioned**:

- [Tweet from Conference on Language Modeling (@COLM_conf)](https://fixupx.com/colm_conf/status/1749881065055924363): We are pleased to announce that the first Conference on Language Modeling will be held at the University of Pennsylvania in Philadelphia at the Zellerbach Theatre.     Thanks so much to UPenn CS as we...
- [no title found](https://news.ycombinator.com/news>): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1199370029007896687) (2 messages): 

- **Combating LLM Reward Hacking with WARM**: User `@thejonasbrothers` shared a research paper addressing reward hacking in large language models (LLMs) through a method called Weight Averaged Reward Models (WARM). The [paper](https://arxiv.org/abs/2401.12187) proposes fine-tuning multiple reward models (RMs) and averaging them to improve robustness and reliability under distribution shifts and human preference inconsistencies.

- **VONet Breaks Ground in Unsupervised Video Learning**: `@vrus0188` introduced a paper on VONet, an unsupervised video object learning framework using parallel U-Net attention and object-wise sequential VAE. The [method outperforms existing techniques](https://arxiv.org/abs/2401.11110) for decomposing video scenes across multiple datasets with its novel approach, and the code is accessible on [GitHub](https://github.com/hn).

**Links mentioned**:

- [WARM: On the Benefits of Weight Averaged Reward Models](https://arxiv.org/abs/2401.12187): Aligning large language models (LLMs) with human preferences through reinforcement learning (RLHF) can lead to reward hacking, where LLMs exploit failures in the reward model (RM) to achieve seemingly...
- [VONet: Unsupervised Video Object Learning With Parallel U-Net Attention and Object-wise Sequential VAE](https://arxiv.org/abs/2401.11110): Unsupervised video object learning seeks to decompose video scenes into structural object representations without any supervision from depth, optical flow, or segmentation. We present VONet, an innova...

  ,

### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1199460151208460288) (1 messages): 

- **HuggingFace Levels Up!**: Introducing a new channel `<#1197143964994773023>` to showcase the highest-leveled community members. Activities such as posting, reacting, and creating repositories or papers on the Hub will earn members experience points.
  
- **Making Games with AI - Chapter 2 Released**: The second chapter of the *Making games with AI* course is now available. Interested parties can celebrate and learn more on [Thomas Simonini's Twitter post](https://twitter.com/ThomasSimonini/status/1748018680787616219).
  
- **Gradio Performance Optimizations Revealed**: The Gradio team shared insights into making Gradio faster by *"...slowing it down!"* after a lag issue was reported by `@oobabooga`. The full story and technical breakdown can be followed on [this HuggingFace post](https://huggingface.co/posts/abidlabs/528535313265224).
  
- **Transformers.js Introduces Meta's SAM Model in-browser**: Version 2.14 of Transformers.js runs Meta’s SAM model completely within your browser. Users can experiment with this using npm and details on usage can be found in [the linked HuggingFace post](https://huggingface.co/posts/Xenova/240458016943176).
  
- **transformers v4.37 Packed with Innovations**: The latest release of `transformers v4.37` introduces several new models and features including Qwen2, Phi-2, SigLIP, and 4-bit serialization. More details can be found on [GitHub](https://github.com/huggingface/transformers/releases/tag/v4.37.0).

**Links mentioned**:

- [@abidlabs on Hugging Face: &quot;𝗛𝗼𝘄 𝘄𝗲 𝗺𝗮𝗱𝗲 𝗚𝗿𝗮𝗱𝗶𝗼 𝗳𝗮𝘀𝘁𝗲𝗿 𝗯𝘆... 𝘀𝗹𝗼𝘄𝗶𝗻𝗴 𝗶𝘁…&quot;](https://huggingface.co/posts/abidlabs/528535313265224): no description found
- [@Xenova on Hugging Face: &quot;Last week, we released 🤗 Transformers.js v2.14, which added support for SAM…&quot;](https://huggingface.co/posts/Xenova/240458016943176): no description found
- [Release v4.37 Qwen2, Phi-2, SigLIP, ViP-LLaVA, Fast2SpeechConformer, 4-bit serialization, Whisper longform generation · huggingface/transformers](https://github.com/huggingface/transformers/releases/tag/v4.37.0): Model releases Qwen2 Qwen2 is the new model series of large language models from the Qwen team. Previously, the Qwen series was released, including Qwen-72B, Qwen-1.8B, Qwen-VL, Qwen-Audio, etc. Qw...
- [@philschmid on Hugging Face: &quot;What&#39;s the best way to fine-tune open LLMs in 2024? Look no further! 👀 I am…&quot;](https://huggingface.co/posts/philschmid/542513204804942): no description found
- [@abidlabs on Hugging Face: &quot;There&#39;s a lot of interest in machine learning models that generate 3D objects…&quot;](https://huggingface.co/posts/abidlabs/986151192658340): no description found
- [GitHub - Vaibhavs10/open-tts-tracker](https://github.com/Vaibhavs10/open-tts-tracker): Contribute to Vaibhavs10/open-tts-tracker development by creating an account on GitHub.
- [Tweet from Xenova (@xenovacom)](https://x.com/xenovacom/status/1749756308641697946): Introducing the 🛝 Jinja Playground: Design LLM chat templates directly in your browser with instant feedback.  Built with `@​huggingface/jinja`, a minimalistic JavaScript implementation of the Jinja ...
- [@merve on Hugging Face: &quot;Explaining the 👑 of zero-shot open-vocabulary object detection: OWLv2 🦉…&quot;](https://huggingface.co/posts/merve/182364809187009): no description found
- [Fine-Tune W2V2-Bert for low-resource ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-w2v2-bert): no description found
- [PatchTSMixer in HuggingFace](https://huggingface.co/blog/patchtsmixer): no description found
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): no description found
- [Tweet from Niels Rogge (@NielsRogge)](https://x.com/NielsRogge/status/1747631048941252878): New @YouTube video in my ChatGPT at home series: fine-tuning Mistral-7B on a GPU rented on @runpod_io   Involves chat templates, QLoRa, packing, Flash Attention 2, bfloat16... a lot of things to expla...

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199281004976807976) (49 messages🔥): 

- **Morning Cheer for Open-Sourcing**: `@osanseviero` kicks off the day ready for open-source contributions. Replies like `@Cubie | Tom` keeps the spirit high with a resounding "Always 💪".

- **ONNX Export Woes**: `@blahblah6407` is facing issues exporting a fine-tuned model to ONNX, encountering an `[ONNXRuntimeError]` regarding `Trilu(14)` node implementation.

- **Looking for Guidance in Machine Learning**: `@𝐀𝐌𝐎𝐔𝐑` inquires about starting points for learning machine learning. They receive assistance from `_mad.haven_` who shares a [useful guide on Hugging Face](https://discord.com/channels/879548962464493619/1199265839489220628/1199265839489220628) to grasp the basics.

- **AI Video Background Inquiry**: `@omniroots` posts a [YouTube video](https://www.youtube.com/watch?v=L8JVzu7S784) titled "THE FUTURE OF AI IS PURE IMAGINATION" and asks the community for information on the AI background effects used.

- **Leveling Up on HuggingFace**: Discussions by `@realmrfakename` and `@lunarflu` regarding the integration of Discord and Hub activities influence the level displayed by LevelBot, and potential future feature of displaying levels on user profiles is also considered.

**Links mentioned**:

- [Dreamoving - a Hugging Face Space by jiayong](https://huggingface.co/spaces/jiayong/Dreamoving): no description found
- [THE FUTURE OF AI IS PURE IMAGINATION](https://www.youtube.com/watch?v=L8JVzu7S784): Shrinking the lag time between what we imagine and what we create. The future of AI. #ai #jasonsilva #singularity #possibility #awe #tech
- [GitHub - dreamoving/dreamoving-project: Official implementation of DreaMoving](https://github.com/dreamoving/dreamoving-project): Official implementation of DreaMoving. Contribute to dreamoving/dreamoving-project development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1199364658046451853) (6 messages): 

- **Getting the Hang of It**: `@not_lain` mentioned that although a task may be challenging at first, it becomes easier with practice, implying a learning curve in their recent endeavors.

- **Privacy-Conscious Transcription Tool Development**: `@n278jm` is building a tool that transcribes and summarizes **audio recordings of consultations**. The objective is to operate entirely locally to maintain the **privacy** of the data due to the sensitive nature of the consultations.

- **Tool Specifications Unveiled**: `@n278jm` detailed the tool's workflow: "Meeting audio upload as input -> transcription with speaker diarization -> meeting summary as output". They plan to use **Hugging Face's transformers** for the audio transcription, **pyannote.audio 3.1** for speaker diarization, and a DistilBERT model for summarization, all tied together with a simple HTML front end.

- **Python as a Blast from the Past**: `@n278jm` expressed some challenges with the Python language due to a long absence from using it, particularly when parsing lists and dictionaries to integrate diarization data with transcription results.

- **Considering Protobuf for Pipeline Communications**: `@n278jm` is contemplating writing **protobuf definitions** to bridge pipelines and possibly use Go for processing the results of the transformers. This suggests they're considering alternatives to Python for parts of their project workflow.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1199281113097572432) (9 messages🔥): 

- **New Identity-Preserving AI Demo**: `@osanseviero` highlighted **InstantID**, a tool for identity-preserving generation within seconds, and shared a positive tweet from Yann LeCun that endorses the **InstantID demo** on Gradio Spaces. [Try it here](https://hf.co/spaces/InstantX/InstantID).
  
- **Hedwig AI Video Intro on YouTube**: `@forestwow7397` shared a [YouTube video](https://www.youtube.com/watch?v=ClBbg-STXVU) introducing **Hedwig AI**, a platform aimed at transforming the utilization and understanding of video data through AI.
  
- **Enigmatic AI Backgrounds**: `@omniroots` requested information on the AI-generated background effect used in a [YouTube video](https://www.youtube.com/watch?v=L8JVzu7S784) by Jason Silva, sparking curiosity among other users.

- **QwenLM Opensource Journey**: `@osanseviero` shared an update from **QwenLM**, presenting an overview of their opensource journey with Large Language Models, along with important links to their academically published [paper](https://arxiv.org/abs/2309.16609), [GitHub repository](https://github.com/QwenLM/Qwen), and [Hugging Face models](https://huggingface.co/Qwen).

- **PhotoMaker Offers Efficient Personalized Generation**: `@ggabe_2` provided an [arXiv paper link](https://arxiv.org/abs/2312.04461v1) discussing **PhotoMaker**, a novel method for efficient and personalized text-to-image generation, promising in identity fidelity and text controllability.

- **Fine-Tuning LLMs with VPGs**: `@andysingal` shared a study on fine-tuning multimodal large language models (MLLMs) to follow zero-shot demonstrative instructions, tackling issues related to current visual prompt generators (VPGs) biasing. The study introduces a new module called **Visual Prompt Generator Complete** (VPG-C) that improves the models' understanding of such instructions. [Read the study](https://openreview.net/forum?id=BXY6fe7q31).

**Links mentioned**:

- [Introducing Qwen](https://qwenlm.github.io/blog/qwen/): 4 months after our first release of Qwen-7B, which is the starting point of our opensource journey of large language models (LLM), we now provide an introduction to the Qwen series to give you a whole...
- [PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding](https://arxiv.org/abs/2312.04461v1): Recent advances in text-to-image generation have made remarkable progress in synthesizing realistic human photos conditioned on given text prompts. However, existing personalized generation methods ca...
- [Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative...](https://openreview.net/forum?id=BXY6fe7q31): Recent advancements in Multimodal Large Language Models (MLLMs) have been utilizing Visual Prompt Generators (VPGs) to convert visual features into tokens that LLMs can recognize. This is achieved...
- [THE FUTURE OF AI IS PURE IMAGINATION](https://www.youtube.com/watch?v=L8JVzu7S784): Shrinking the lag time between what we imagine and what we create. The future of AI. #ai #jasonsilva #singularity #possibility #awe #tech
- [Youtube Video Intro hedwigAI](https://www.youtube.com/watch?v=ClBbg-STXVU): Welcome to the world of seamless streaming with Hedwig AI, where we&#39;re transforming the way video data is utilized and understood. In this video, we showcase...
- [h94/IP-Adapter-FaceID at main](https://huggingface.co/h94/IP-Adapter-FaceID/tree/main): no description found
- [Tweet from Omar Sanseviero (@osanseviero)](https://fxtwitter.com/osanseviero/status/1749481521788907651): InstantID: Identity-Preserving Generation in Seconds  Try it at  https://hf.co/spaces/InstantX/InstantID
- [Tweet from Yann LeCun (@ylecun)](https://fxtwitter.com/ylecun/status/1749564695454085558): Yay, I&#39;m a Marvel superhero! Where&#39;s my Iron Man suit?  ↘️ Quoting Gradio (@Gradio)   🔥InstantID demo is now out on Spaces.  Thanks @Haofan_Wang et al, for building a brilliant Gradio demo fo...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1199284665010356224) (14 messages🔥): 

- **Quick Inference with "Tiny" Model**: `@Cubie` affirmed the efficiency of an enhancement to `all-MiniLM-L6-v2` which **maintains fast inference** by only adding a Dense layer, potentially expediting comparisons of embeddings.
  
- **PCA Script Adoption in sbert Library**: `@stroggoz` mentioned the **utilization of PCA** for comparisons of embeddings and credited their script to the one in the sbert library.

- **Visualizing Suspected Fakes**: In response to `@lunarflu`'s query about a heatmap's purpose, `@not_lain` and `@thenameless7741` seemed to confirm its use in **detecting potential fakes** without providing additional context.

- **Upgrade and Migration of Open LLM Leaderboard**: `@thenameless7741` notified about the **migration of their project**, Cosmos Arena, which includes various features such as model types, weight types, and licenses, and mentioned it contains a few regressions and bugs post-update. `@osanseviero` reacted positively, considering sharing it with their team.

- **Zero-Effort Image Embedding with Hugging Face**: `@tony_assi` introduced their GitHub project, [HF-Embed-Images](https://github.com/TonyAssi/HF-Embed-Images), which provides a one-line solution for generating embeddings for image datasets with the Hugging Face ecosystem.

- **Debugging Machine Learning Training with 3LC**: `@paulend76` teased the upcoming public Beta for 3LC, an **ML training and data debugger/editor**, integrating with PyTorch and free for non-commercial purposes; they provided links to a [YouTube video](https://www.youtube.com/shorts/o_5XStvSF18) and [blog post](https://3lc.ai/3lc-machine-learning-data-quality-enhancement/) for more insights and invited feedback for their Beta.

- **Exploring Gabor Vecsei's GitHub Repos**: `@ggabe_2` shared a link to `@gaborvecsei`'s [GitHub profile](https://github.com/gaborvecsei), encouraging others to explore the repositories, and highlighted a recent project on [live transcription with Whisper](https://github.com/gaborvecsei/whisper-live-transcription).

**Links mentioned**:

- [Cosmos Arena](https://thenameless.net/cosmos-arena): no description found
- [GitHub - TonyAssi/HF-Embed-Images: Generates image embeddings for 🤗 Datasets](https://github.com/TonyAssi/HF-Embed-Images): Generates image embeddings for 🤗 Datasets. Contribute to TonyAssi/HF-Embed-Images development by creating an account on GitHub.
- [Imbalance in Balance](https://www.youtube.com/shorts/o_5XStvSF18): In this demo, we see how 3LC helps Data Scientists quickly and efficiently pinpoint and address issues in their model. By adding weight to underrepresented s...
- [Introducing 3LC](https://3lc.ai/3lc-machine-learning-data-quality-enhancement/): With our innovative approach to data quality, 3LC paves the way for more accurate model training, without changing where your data lives.
- [gaborvecsei - Overview](https://github.com/gaborvecsei): I push my boundaries as far as I can.  Also I love chocolate. 😎 - gaborvecsei
- [GitHub - gaborvecsei/whisper-live-transcription: Quick and dirty example for live transcription with Whisper](https://github.com/gaborvecsei/whisper-live-transcription): Quick and dirty example for live transcription with Whisper - GitHub - gaborvecsei/whisper-live-transcription: Quick and dirty example for live transcription with Whisper

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199524353864384622) (5 messages): 

- **DPM++ Updates on the Horizon**: `@keturn` expressed anticipation for improvements in the DPM++ with Karras after issues addressed in a recent PR, hoping to see these enhancements in the **next release**.
- **Inquiry into Diffusion Scheduling**: `@vtabbott_` brought up questions about HuggingFace's diffusion implementation, specifically the `scheduling_euler_distance.py` file and its origins.
- **Karras's Paper Sheds Light on Diffusion**: In response, `@keturn` explained that the "euler" calculations follow the naming from [k-diffusion](https://github.com/crowsonkb/k-diffusion), which implements concepts from Karras's paper titled *[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)*. They further clarified that the Euler scheduler is likely derived from the DDIM paper, and mentioned differences between "variance preserving" and "variance exploding" without certainty on their mathematical equivalence.
- **Diving Deep into Diffusion Models**: `@vtabbott_` mentioned working on diagrams for diffusion models to ensure a thorough understanding strong enough to reimplement the processes.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1199266684998000722) (4 messages): 

- **Model Parallel Training Explained**: `@mr_nilq` highlighted that the current approach is **inference-only** and recommended using the Trainer with **Accelerate** for model parallel training. The user shared a helpful [HuggingFace guide](https://huggingface.co/docs/transformers/perf_train_gpu_many) explaining when and how to transition from single GPU to multi-GPU training setups.
- **Llama 2's Multilingual Mysteries**: `@dhruvbhatnagar.0663` inquired how **Llama 2 model** can generate responses in languages like Hindi, Tamil, or Gujarati without specific tokens in its vocabulary. `@vipitis` replied with a suggestion to observe the token ids generated and to investigate further.
- **DeciLM-7B Slow Inference Troubles**: `@kingpoki` is experiencing slow inference times with **Deci/DeciLM-7B** on an Nvidia 3060, taking **about 23 seconds for 50 tokens**. The user tried quantizing the model but encountered issues and is seeking advice on improving inference speed without impacting their ability to finetune the model.

**Links mentioned**:

[Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199524353864384622) (5 messages): 

- **Anticipation for Improved DPM++**: `@keturn` is looking forward to enhancements in DPM++ as mentioned by Karras, hoping that the issues addressed in a recent PR will show significant improvement in the next release.
  
- **Euler's Place in HuggingFace Schedulers**: In response to `@vtabbott_`'s query about the `scheduling_euler_distance.py`, `@keturn` explains that it aligns with the naming from [k-diffusion](https://github.com/crowsonkb/k-diffusion), and eventually with Karras's [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364). They mention that Euler scheduler is related to the DDIM paper and there’s a variance distinction with `diffusers`'s `scheduling_ddim` which might relate to being "variance preserving" or "variance exploding".
  
- **Digging Deep into Diffusion**: `@vtabbott_` expresses a keen interest in deeply understanding diffusion models to the extent of being able to reimplement them, demonstrating a commitment to crafting accurate explanatory diagrams.
  ,

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1199295465875837000) (57 messages🔥🔥): 

- **Discussions on Model Adjustments**: Users discussed various topics including the efficiency of RPA AI (`@moyaoasis`), light mode availability on labs (`@hvci` and `@icelavaman` responding with only dark mode available), and frustration with using the microphone feature on Android where the app permissions lacked options for the mic (`@.cryptobadger`).
- **Channel Navigation Confusion**: People admitted confusion over channels existing or being removed, particularly related to generating pictures and the Dream bot that is no more, with `@ok.alex` confirming a channel's closure and `@oscar_010` expressing a need for news summaries on Perplexity.
- **Clarification on Model Usage**: `@iprybilovych` had questions about discerning between GPT-4 and Gemini Pro models on Perplexity AI's pro version, which were addressed by `@icelavaman`, explaining model selection settings, with follow-up engagement by community managers like `@Dyno` encouraging community recognition for helpful posts.
- **Feature Inquiries and Assistance**: A user inquired about a potential teams option and support issues with credit (`@umyong`), while another user `@zwaetschgeraeuber` speculated about future app support for Wear OS, following a perceived collaboration with Rabbit.
- **API Enquiries and Extensions Discussed**: `@generativeai_strategy_44986` requested information about increasing the rate limit of the API for product integration, `@dogemeat_` recommended the Continue.dev extension for integration with vscode, and `@speedturkey` humorously encouraged Pro subscribers to donate credits to "the church of the God prompt".

**Links mentioned**:

- [Continue&#32;-&#32;CodeLlama,&#32;GPT-4,&#32;and&#32;more&#32;-&#32;Visual&#32;Studio&#32;Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue): Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Open-source&#32;autopilot&#32;for&#32;software&#32;development&#32;-&#32;bring&#32;the&#32;power&#32;of&#32;ChatGPT&#32;to&#32;your&#32;IDE
- [no title found](https://api.perplexity.ai",): no description found

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1199265040637898862) (12 messages🔥): 

- **Troubleshooting Application Errors**: `@_marcos75` reported issues accessing Perplexity shares due to an "Application error." After receiving advice, they shared an example link that was causing trouble: [BMW Humanoid Factory](https://www.perplexity.ai/search/BMW-humanoid-factory-ToZqMRZMT0uvwpeU2jP2fQ?s=c). `@icelavaman` suggested that it is likely a browser issue rather than a perplexity error.

- **Perplexity as a Research Powerhouse**: `@nicknalbach` shared a positive review of using Perplexity, emphasizing its ability to quickly consolidate information from multiple sources, as demonstrated when searching for new phone and cell phone plans.

- **Perplexity Praise and Recommendation**: `@zenrobot.eth` responded to `@nicknalbach`'s feedback, endorsing the impressive speed and accuracy of Perplexity's search capability.

- **Workflow Enhancement through Perplexity**: `@joe_heller` mentioned discovering how Perplexity can complement workflow, sparking curiosity from `@icelavaman` who asked for a public example. `@ivibudh` humorously inferred the response might be AI-generated.

- **Real-World Assistant for Mechanical Engineering**: `@n_667` testified to using Perplexity as an assistive tool for mechanical engineering courses and personal projects, noting time-saving benefits.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1199369629336862761) (3 messages): 

- **Confusion Over API Usage**: User `@tpsk12345` expressed difficulty in using the API and requested an update from Alex, with no additional context provided.

- **Inquiry About Online Model Citations**: `@donvitoph` tested the `pplx-70b-online` model via the API and raised a question about whether the online models return citation URLs. The exact usage was shown with a code snippet for calling the `chat.completions.create` function.
  ,

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1199450104973754451) (2 messages): 

- **Hackathon Excitement with Cash Prizes**: IFTTT announces an in-person hackathon from February 2-4 with prizes totaling $13,500, including $8,250 in cash. The event encourages participants to *build projects that solve real problems* and offers expert feedback. [Hackathon announcement tweet](https://twitter.com/llama_index/status/1749890655323898344).

- **Introducing MemGPT for Advanced Chat Experiences**: MemGPT, spotlighted by `@charlespacker`, is an OSS project for creating chat experiences featuring long-term memory, self-editing memory, and infinite context windows, built with LlamaIndex. Installation is quick through pip, offering a personalized AI experience. [OSS project spotlight tweet](https://twitter.com/llama_index/status/1749935634033348693).

**Links mentioned**:

[MemGPT](https://t.co/COUh9LAKzR): no description found

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1199280038835650561) (65 messages🔥🔥): 

- **SQLite-VSS Integration for Llama-Index**: `@pveierland` inquired about any existing **sqlite-vss integrations** for llama-index but received no documented responses or solutions.
- **Pandas Query Engine Troubles**: `@techexplorer0` pointed out that there are issues when using the **PandasQueryEngine** with open-source LLMs like **Zephyr 7b**. A documentation link with an example for building a [query pipeline with Pandas](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_pandas.html) was shared but did not directly address working with CSV files in building a RAG chatbot.
- **Open Source LLMs with RAG**: `@zaesar` seeking how to implement a RAG with open-source LLMs like **dolphin mistral** in the Llama-Index course on deeplearning.ai received a suggestion from `@emanuelferreira` to use **LlamaIndex's base LLM module**. `@cheesyfishes` suggested using **olama** for easy local setups and **vLLM** for production.
- **SubQuestionQueryEngine in Streamlit**: `@matthews_38512` mentioned using **SubQuestionQueryEngine** at Streamlit but no solution was present. Later a conversation between `@kapa.ai`, presumed to be the chatbot and `@matthews_38512`, offered detailed steps and links to [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html) for setting up a SubQuestionQueryEngine.
- **Implementing Chatbots with Memory Over CSV**: `@techexplorer0` sought to create a conversational chatbot with memory leveraging RAG over multiple CSV files using an open-source LLM for data aggregations and received guidance about structured workflows from `@kapa.ai`, including a link to the official [LlamaIndex documentation](https://docs.llamaindex.ai/en/latest/getting_started/concepts.html). There was also a request on setting a **Query Pipeline** to use the maximum number of workers, where `@kapa.ai` provided a potential solution involving `os.cpu_count()`.

**Links mentioned**:

- [Query Pipeline over Pandas DataFrames - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_pandas.html): no description found
- [Step-wise, Controllable Agents - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_runner.html): no description found
- [Customizing LLMs within LlamaIndex Abstractions - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html#example-using-a-custom-llm-model-advanced): no description found
- [OpenAI Agent with Query Engine Tools - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_with_query_engine.html): no description found
- [Query Pipeline - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1199510662418804776) (1 messages): 

- **New Insights on Dynamic Knowledge for RAG Chatbots**: User `@sl33p1420` shared their latest Medium article about [integrating dynamic sources](https://medium.com/@marco.bertelli/empowering-your-chatbot-unveiling-dynamic-knowledge-sources-with-advanced-integration-e8353e85099c) in RAG chatbots. The article focuses on enhancing chatbots by harnessing advanced knowledge integration techniques.
- **Detailed RAG Chatbot Development Guide**: The same user also highlighted their previous works that form a series, starting with [An In-depth Journey](https://medium.com/@marco.bertelli/build-a-complete-opensource-llm-rag-qa-chatbot-an-in-depth-journey-introduction-c630b16c330c), moving through model selection, server setup, and finally the chat engine construction.

**Links mentioned**:

[Empowering Your Chatbot: Unveiling Dynamic Knowledge Sources with Advanced Integration](https://medium.com/@marco.bertelli/empowering-your-chatbot-unveiling-dynamic-knowledge-sources-with-advanced-integration-e8353e85099c): Explore the next frontier in chatbot development adding how dynamic knowledge sources are harnessed.


### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1199294661831962664) (7 messages): 

- **Looking for AI Community Spaces**: User `stranger_54257` asked for links to more **LLM discord servers** but did not receive any responses.
- **AI Diversity Mirrors Human Complexity**: `@bambooshoots` discussed how a broad spectrum of behaviors in superintelligent AI might emerge due to various designs, objectives, and learning environments, similar to the diversity seen in humanity.
- **Seeking LLM Evaluation and Monitoring Tools**: `@coderindajungle` asked for suggestions regarding **LLM evaluation & monitoring tools** for a chatbot application built on **GPT-4**, but no suggestions were given.
- **Concerns about Dall.E's Image Handling**: `@brnst` questioned why **Dall.E** struggles with recreating or editing images when provided with an example, but the conversation did not yield an answer.
- **Control Over AGI Dominates the AI Ethics Debate**: `@catcapitol` stated that with regard to artificial general intelligence (AGI), the main concern is about **who controls it** and their uses of the technology.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1199372403139154021) (28 messages🔥): 

- **File Upload Facts Uncovered**: `@solbus` clarified for `@mrbr2023` that up to **20 files** of up to **512MB each** can be uploaded to a Custom GPT as knowledge, while text files are limited to **2 million tokens** each. They discussed the potential strategy of combining documents into single files to bypass the 20-file limit, but also considered possible downsides like inefficiency in searching large files.
- **GPTs Marketplace Mystery**: `@sstrader29` inquired about missing CustomGPTs on the GPTs Marketplace, but no specific answer was provided in the summarized conversation.
- **Grimoire GPT for Word Processing**: `@eligump` excitedly shared their creation of a word processor using the **Grimoire GPT** in under ten minutes.
- **Custom GPT Network Error Enigma**: `@valikami` requested advice regarding persistent network errors after their custom GPT provides responses, but no resolution was mentioned.
- **Intertwined GPT Threads**: `@snowmkr_jk` expressed confusion when a file from one thread seemed to impact another, presenting a potential issue with file handling across separate threads, awaiting community confirmation on whether that should happen.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1199265163556159518) (10 messages🔥): 

- **Looking for Better Context Handling in Answers**: `@ArianJ` wants to answer additional questions based on chat logs concerning career topics, but finds that directly inputting the logs and questions doesn't yield good results. `@darthgustav.` inquires about the size of the logs and the number of questions asked simultaneously to diagnose the issue.

- **Executive Assistant Prompt Suggestions**: `@jdf.wwp` asks for prompt ideas suited for organization or executive assistants. `@darthgustav.` responds by asking about their goal and suggests using his Custom GPT for Plus subscribers or guiding ChatGPT with a user's background information.

- **Custom GPT's Descriptive Language**: `@archipelagic` is seeking advice on the most effective language to use for refining tasks within the description field of a custom GPT. `@eskcanta` recommends defining clear objectives and communicating those goals effectively, whether seeking help from the AI or human collaborators.

- **Simplicity Over Effectiveness for AI Guidance**: In response to concerns about refining prompts, `@eskcanta` advises focusing on deciding what one wants from the AI and finding ways to achieve that rather than worrying about the 'most effective' language.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1199265163556159518) (10 messages🔥): 

- **Prompt Engineering Enthusiasm**: `darthgustav.` supports the idea that a good prompt engineering plan works well every time, implying a consistent strategy for successful AI interactions.

- **Improving Contextual Understanding**: `@ArianJ` is facing issues with the AI not answering questions based on chat logs. The user seeks advice on handling chat logs for extracting additional information, but faces challenges with the AI understanding the context.

- **Inquiry About Prompt Size for Chat Logs**: `darthgustav.` inquires about the size in bytes of the chat logs that `@ArianJ` is using, which might influence the AI's ability to process and answer the questions.

- **Efficient Language for Custom GPT Configurations**: `@archipelagic` asks about the most effective language to use when refining descriptions for a custom GPT model. `eskcanta` responds, emphasizing the importance of clarity in communicating goals, both to the AI and when seeking help.

- **Solving for Desired AI Outcomes**: Instead of focusing on the 'most effective' language, `eskcanta` advises to work towards clear outcomes with the AI, suggesting an approach of adapting to achieve specific goals rather than seeking a universally 'best' method.
  ,

### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1199268415026774076) (10 messages🔥): 

- **Template Troubles for `theBloke Q6`**: `@eric_73339` revealed struggles with different outcomes between a local model and a demo version. They suspected the issue was with the **prompt template** format.
  
- **Mixtral Template Clarification**: `@sebastian.bodza` shared the correct **Mixtral chat template** as per the Hugging Face tokenizer, highlighting discrepancies with the readme.md documentation.
  
- **Better Formatting with f-strings**: `@bjoernp` suggested using **f-strings** and adding newlines after each role to correct `@eric_73339`'s **ChatML template** for compatibility with DiscoLM models.
  
- **Demo Site Model & Template Usage**: `@bjoernp` pointed out that https://demo.discoresearch.org uses **[DiscoLM German 7b v1](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1)** and provided a [link](https://huggingface.co/docs/transformers/main/chat_templating) to avoid issues with chat templates.
  
- **Gratitude for Community Support**: `@eric_73339` thanked the **community** for assistance with adapting to large language models (LLMs) and fixing template issues.

**Links mentioned**:

- [DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1): no description found
- [Templates for Chat Models](https://huggingface.co/docs/transformers/main/chat_templating): no description found
- [DiscoLM German 7b Demo](https://demo.discoresearch.org?): no description found

  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1199267109331533894) (11 messages🔥): 

- **Evaluating Translation Quality**: `@bjoernp` discussed the challenges in evaluating translations and recommended using **Lilac** for manual checks. They mentioned trying to create a method to filter out bad translations using [Distilabel](https://github.com/argilla-io/distilabel), but recognized its potential cost with GPT-4.

- **DiscoLM German Poised to Shine**: `@_jp1_` expressed interest in comparing the DiscoLM German performance to other models, acknowledging the availability of the English messages for this purpose.

- **Predictions for Llama-3 Unveiled**: `@bjoernp` shared detailed predictions for the upcoming **Llama-3** release, which included extensive pretraining on 5 trillion tokens, a focus on multilingualism and code, and [advanced context chunking as illuminated by a recent paper](https://arxiv.org/abs/2310.10638).

- **Delve into Data Multiple Epochs**: In response to `@maxidl` suggesting more than one epoch for training, `@rasdani` questioned if this has been attempted post-Datablations paper, with `@bjoernp` replying it's currently not seen as necessary.

- **German LM On the Horizon**: `@maxidl` revealed an upcoming **German LM** with about 1 trillion tokens in the dataset and discussed the implications for compute resources, hoping for simplifications after establishing a solid first checkpoint. They linked a [Twitter post](https://twitter.com/BlancheMinerva/status/1721380386515669209) regarding their approach.

**Links mentioned**:

[In-Context Pretraining: Language Modeling Beyond Document Boundaries](https://arxiv.org/abs/2310.10638): Large language models (LMs) are currently trained to predict tokens given document prefixes, enabling them to directly perform long-form generation and prompting-style tasks which can be reduced to do...

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1199332394952568942) (4 messages): 

- **Embarking on a Mistral-like Journey**: `@sebastian.bodza` announced the creation of a dataset similar to the **Mistral embedding model** and shared their project on GitHub: [Embedding Training](https://github.com/SebastianBodza/Embedding_Training). The aim is to first generate diverse **Topics** leveraging Quora topics data.
  
- **Size Matters for Hosting**: `@sebastian.bodza` mentioned the plan to host the generated data on **Hugging Face** or **GitHub**, depending on the dataset's size.

- **To BGE or GTE? That Is the Question**: `@sebastian.bodza` speculated on potentially training a **BigGraph Embedding (BGE)** or **Graph 'n' Table Embedding (GTE)** model.

- **Voyage to Enhanced Code Retrieval**: `@sebastian.bodza` brought to attention **Voyage's new code embedding model**, `voyage-code-2`, which has shown a significant improvement in semantic code retrieval tasks. The model's details are shared in a [Voyage AI blog post](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/).

**Links mentioned**:

- [voyage-code-2: Elevate Your Code Retrieval](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/): TL;DR – We are thrilled to introduce voyage-code-2, our latest embedding model specifically tailored for semantic retrieval of codes and related text data from both natural language and code querie…
- [GitHub - SebastianBodza/Embedding_Training](https://github.com/SebastianBodza/Embedding_Training): Contribute to SebastianBodza/Embedding_Training development by creating an account on GitHub.

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199319747611856926) (19 messages🔥): 

- **Axolotl Usage Inquiry**: `@thomasrenkert` was trying to use **Axolotl** and managed to get it running in Docker but struggled with how to include a personal dataset. `@rasdani` provided assistance, suggesting to convert the dataset into [a supported format](https://github.com/OpenAccess-AI-Collective/axolotl#dataset) and referencing it in the [configuration](https://github.com/OpenAccess-AI-Collective/axolotl#config).
- **Navigating Axolotl GPU Quirks**: `@thomasrenkert` faced an issue where **Axolotl** did not recognize GPUs from within the Docker container, to which `@devnull0` recommended setting up GPU access as per the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) guide and installing necessary CUDA libraries, possibly using Conda.
- **Referral to Axolotl Discord for Support**: As questions about **Axolotl** debugging arose, `@bjoernp` referred users to the [Axolotl Discord](https://discord.gg/2zHuDM2C) for more specialized assistance.
- **DiscoLM German Model Newline Issue**: `@thewindmom` reported an issue when running `discolm_german_7b_v1.Q4_K_M.gguf` with **Ollama** where the output consisted of endless newlines, a problem shared among users according to a [Hugging Face discussion thread](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/discussions/4).
- **Fix for DiscoLM Newline Glitch**: `@_jp1_` noted a fix for the aforementioned newline issue in the DiscoLM_German model's `config.json`, changing `"eos_token_id": 2` to `32000` ([commit details](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/commit/560f972f9f735fc9289584b3aa8d75d0e539c44e)). They called for community members to test if the issue was resolved with this update.

**Links mentioned**:

- [Installing the NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.14.4 documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): no description found
- [Join the OpenAccess AI Collective Discord Server!](https://discord.gg/2zHuDM2C): Check out the OpenAccess AI Collective community on Discord - hang out with 1546 other members and enjoy free voice and text chat.
- [DiscoResearch/DiscoLM_German_7b_v1 · Endless Spaces](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/discussions/4): no description found
- [Fix wrong EOS token in config.json · DiscoResearch/DiscoLM_German_7b_v1 at 560f972](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/commit/560f972f9f735fc9289584b3aa8d75d0e539c44e): no description found
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl#dataset)): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl#config).): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  ,

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1199268258952511518) (28 messages🔥): 

- **Karpathy's Perspective on Technology Change**: `@swyxio` shared a [new blog post by Andrej Karpathy](https://huggingface.co/posts/clem/970025506569107) highlighting the challenges faced by those outside the tech industry in adapting to change. They cited personal examples to emphasize the anxiety and fear that technological change can induce in people.
- **Perplexity’s Non-Linear Progress**: `@swyxio` linked to a [tweet](https://x.com/madiator/status/1749501859151405124?s=61) from `@madiator` visualizing the non-linear three-month journey of development for the AI model Perplexity.
- **Exploring Grammar and Reasoning in Smaller Models**: `@nuvic_` discussed a paper on training smaller LMs with hopes of retaining their grammar and reasoning capabilities, pointing to the possibility that coherent language might be achievable without massive parameter counts ([TinyStories paper](https://arxiv.org/abs/2305.07759)).
- **Discord Applies AI to Create Notification Titles**: `@.onacomputer` commented on Discord's new approach of using LLMs to summarize and generate notification titles based on community messages, which `@vcarl` and `@youngphlo` found surprising considering past privacy policies.
- **Stability's New Diffusion Model Unveiled**: `@swyxio` announced a notable diffusion model development by Stability AI that generates megapixel scale images directly, bypassing latent diffusion processes, and also shared an [AI news summary service](https://buttondown.email/ainews/archive/ainews-rip-latent-diffusion-hello/).

**Links mentioned**:

- [@clem on Hugging Face: &quot;Re-posting @karpathy&#39;s blogpost here because it&#39;s down on…&quot;](https://huggingface.co/posts/clem/970025506569107): no description found
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759): Language models (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GP...
- [JSONalyze Query Engine - LlamaIndex 🦙 0.9.36](https://docs.llamaindex.ai/en/latest/examples/query_engine/JSONalyze_query_engine.html#): no description found
- [Tweet from Mahesh Sathiamoorthy (@madiator)](https://x.com/madiator/status/1749501859151405124?s=61): In case you thought Perplexity&#39;s journey was straightforward and linear.
- [Torvalds Speaks: Impact of Artificial Intelligence on Programming](https://youtu.be/VHHT6W-N0ak?si=KA7betIkfPSmKxch): 🚀 Torvalds delves into the transformative influence of Artificial Intelligence on the world of coding.🚀 Key Topics:* Evolution of programming languages in ...
- [[AINews] RIP Latent Diffusion, Hello Hourglass Diffusion](https://buttondown.email/ainews/archive/ainews-rip-latent-diffusion-hello/): AI Discords for 1/22/2024. We checked 19 guilds, 291 channels, and 4368 messages for you. Estimated reading time saved (at 200wpm): 436 minutes. Katherine...

  

---


### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1199371078896717974) (5 messages): 

- **SPIN Implementation on the Horizon**: User `@swyxio` announced that `@lucidrains` is planning to implement SPIN, sharing a link to the GitHub repository [self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch).
- **Clarification on SPIN and Meta Paper Implementation**: `@ivanleomk` initially thought the implementation was for the new Meta paper, but `@swyxio` clarified that `@lucidrains` will implement both SPIN and Meta's approach separately.
- **Check the ReadMe for More Info**: For further details, `@swyxio` directed users to the bottom of the repository's readme file.

**Links mentioned**:

[GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI](https://github.com/lucidrains/self-rewarding-lm-pytorch): Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI - GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in...

  ,

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1199265059457736734) (16 messages🔥): 

- **LangChainJS Workers Project Peek**: `@ilguappo` shared [their work-in-progress project](https://github.com/anaxios/langchainjs-workers), despite hesitations due to it being standard and not reflecting best web API practices. It includes a fun endpoint for emoji reactions in Discord messages.
- **Typescript: The Learning Curve**: `@ilguappo` is learning TypeScript, which is currently implemented in their project from a template, and `@lolis0518` acknowledges the difficulty of the language but is willing to give it a try.
- **Seeking End-to-End RAG Solutions**: `@alvarojauna` inquired about projects related to end-to-end **Retrieval-Augmented Generation (RAG)** systems, looking for potential collaborations or examples.
- **GPT Chat Completion Clarifications**: `@allenpan_36670` questioned the dynamics of how the GPT chat completion endpoint processes a list of messages, which `@lhc1921` addressed by referring to **ChatML's** prompt template that handles such lists.
- **Chat With Your PDFs Tutorial**: `@a404.eth` announced a tutorial covering the creation of a full stack RAG using **PGVector**, **unstructured.io**, and the semantic chunker, aiming to enable chatting with PDF documents through OpenAI. The first part of the tutorial is available on [YouTube](https://youtu.be/UwgZmrRAgQ4).

**Links mentioned**:

- [Chat With Your PDFs: An End to End LangChain Tutorial For Building A Custom RAG with OpenAI. Part 1](https://youtu.be/UwgZmrRAgQ4): A common use case for developing AI chat bots is ingesting PDF documents and allowing users to ask questions, inspect the documents, and learn from them. In ...
- [GitHub - anaxios/langchainjs-workers](https://github.com/anaxios/langchainjs-workers/): Contribute to anaxios/langchainjs-workers development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1199304103348666420) (2 messages): 

- **Mischievous Link Spam**: User `@eleuss` posted a message with a long sequence of bars and underscores, followed by what appears to be a Discord invite `discord.gg/pudgys` and tagged `@everyone`, which resembles spam behavior or an attempt to playfully clutter the chat.

- **Insight on Feedback Mechanism**: `@georgeherby` explained the process for handling feedback, mentioning that users can provide written comments as follow-up to any feedback given, positive or negative. This feedback gets added to the same record for consistency.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1199489296781029437) (1 messages): 

- **LLaMA Outperforms Baklava in Art Analysis**: `@dwb7737` shared a TLDR that **LLaVA** seems to be more effective at analyzing and classifying artwork than **Baklava**. They provided a [GitHub Gist](https://gist.github.com/donbr/5273d55a089c644d35e8e019370c1902) for more details on the comparison.

**Links mentioned**:

[Ollama models - Image Summarization](https://gist.github.com/donbr/5273d55a089c644d35e8e019370c1902): Ollama models - Image Summarization. GitHub Gist: instantly share code, notes, and snippets.

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1199291475951898705) (2 messages): 

- **CrewAI gets Obsidian Note-Taking Power**: `business24.ai` shared [a YouTube tutorial](https://youtu.be/Iqp6mE1xHOw) titled **"Use crewAI and add a custom tool to store notes in Obsidian."** The tutorial demonstrates how to create a custom tool for crewAI that enables users to add search results as a note in Obsidian using OpenAI's ChatGPT 4 and ChatGPT 3.

- **Launching Tutorial Series for Building RAG with PDFs**: `a404.eth` posted the first video of a three-part tutorial series, titled ["Chat With Your PDFs: An End to End LangChain Tutorial For Building A Custom RAG with OpenAI. Part 1"](https://youtu.be/UwgZmrRAgQ4). The tutorial guides viewers through developing AI chatbots capable of interacting with PDF documents, using tools like PGVector, unstructured.io, and the semantic chunker.

**Links mentioned**:

- [Chat With Your PDFs: An End to End LangChain Tutorial For Building A Custom RAG with OpenAI. Part 1](https://youtu.be/UwgZmrRAgQ4): A common use case for developing AI chat bots is ingesting PDF documents and allowing users to ask questions, inspect the documents, and learn from them. In ...
- [Use crewAI and add a custom tool to store notes in Obsidian](https://youtu.be/Iqp6mE1xHOw): In this Tutorial, we create a custom tool for crewAI to add search results as a note in Obsidian. We use it with OpenAI ChatGPT 4 and ChatGPT 3 and Multiple ...

  ,

### LLM Perf Enthusiasts AI ▷ #[announcements](https://discord.com/channels/1168579740391710851/1168760950803931136/1199411943849590815) (1 messages): 

- **Ringing in 2024 with Quality LLM Talk**: `@jeffreyw128` welcomed everyone to 2024 and praised the Discord as a hub for top-tier LLM discussions. He announced his plans to inject fresh energy into the group with a new round of hand-selected invites and encouraged current members to send referrals.
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1199411854326366230) (2 messages): 

- **Document Layout SOTA Unveiled**: User `@res6969` inquired about the **state-of-the-art in document layout understanding**, particularly for identifying bounding boxes around charts in PDF reports. They later updated with the **Vision Grid Transformer** as a potential solution, linking to its GitHub repository [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery).

**Links mentioned**:

[GitHub - AlibabaResearch/AdvancedLiterateMachinery: A collection of original, innovative ideas and algorithms towards Advanced Literate Machinery. This project is maintained by the OCR Team in the Language Technology Lab, Alibaba DAMO Academy.](https://github.com/AlibabaResearch/AdvancedLiterateMachinery): A collection of original, innovative ideas and algorithms towards Advanced Literate Machinery. This project is maintained by the OCR Team in the Language Technology Lab, Alibaba DAMO Academy. - Git...

  

---


### LLM Perf Enthusiasts AI ▷ #[feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549/1199412128176681101) (6 messages): 

- **New Channel Creation**: `@jeffreyw128` announced the creation of a new channel, initially without a specific name.
- **Channel Naming Brainstorm**: `@degtrdg` suggested the name *#share* for the newly created channel.
- **Channel Renamed to #share**: Following `@degtrdg`'s suggestion, `@jeffreyw128` agreed and renamed the channel to *#share*.
- **Discussing LLM Perf-Related Activities**: `@yikesawjeez` mentioned ongoing activities such as Swyx's paper club, Skunkworks paper implementation sessions, and weekly codejams/challenges that are potentially relevant to the LLM Perf community.
- **LLM Perf Infiltration Idea**: `@yikesawjeez` humorously proposed the idea of LLM Perf community members joining other events to bring their expertise on training, architecture, tuning, and performance.
  ,

### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (1 messages): 

far_el: good lad
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=n3gkZ_IRwCI
  ,

### YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/1199427667280277624) (2 messages): 

- **In Search of On-Premise AI Solutions**: User `@floriannoell` inquired about on-premise AI solutions that work independently of major cloud providers like AWS, GCP, or Azure, specifically mentioning watsonx.ai as a reference for what they're looking for.
- **Custom Needs for AI Solutions**: In response, `@spillai` prompted `@floriannoell` to clarify the intended use case for the AI technology, suggesting that identifying specific needs such as pretraining, finetuning, inference, or classification could lead to a more targeted on-premise solution.
  ,

### Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1199580481092595785) (1 messages): 

- **Slim Orca Dataset Now Available on HF**: `@222gate` informed that the **Slim Orca dataset** is accessible on Hugging Face and provided instructions for finetuning using this dataset. The dataset boasts **~500k GPT-4 completions**, and [Slim Orca](https://huggingface.co/datasets/Open-Orca/SlimOrca) has been refined using GPT-4 to remove inaccuracies, requiring 2/3 the compute for similar performance to larger data slices.
- **Efficient High-Performance Training with Slim Orca**: The Open Orca team has curated a subset to enable efficient training, which has been corroborated by demo models such as [jackalope-7b](https://huggingface.co/openaccess-ai-collective/jackalope-7b) and [Mistral-7B-SlimOrca](https://huggingface.co/Open-Orca/Mistral-7B-SlimOrca), showcasing the dataset's effectiveness.

**Links mentioned**:

[Open-Orca/SlimOrca · Datasets at Hugging Face](https://huggingface.co/datasets/Open-Orca/SlimOrca): no description found

  ,

### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1199557401746079924) (1 messages): 

- **`llm-gpt4all` Release 0.3 Goes Live**: `@simonw` announced the new release of [`llm-gpt4all` version 0.3](https://github.com/simonw/llm-gpt4all/releases/tag/0.3) which includes a fix by `<@461550757901107221>` and several other improvements. The update notably allows access to model options like `-o max_tokens 3` and enables models to work without an internet connection.

**Links mentioned**:

[Release 0.3 · simonw/llm-gpt4all](https://github.com/simonw/llm-gpt4all/releases/tag/0.3): Now provides access to model options such as -o max_tokens 3. Thanks, Mauve Signweaver. #3 Models now work without an internet connection. Thanks, Cameron Yick. #10 Documentation now includes the l...

  


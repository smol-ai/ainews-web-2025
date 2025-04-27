---
id: abdcdec1-8bb9-4db6-9518-d728096e9274
title: Qwen 1.5 Released
date: '2024-02-06T23:40:32.776669Z'
original_slug: ainews-qwen-15-released
description: >-
  **Chinese models Yi, Deepseek, and Qwen** are gaining attention, with **Qwen
  1.5** showing strong performance against **Mistral** and **Llama2** models,
  supporting up to **32k token context** and offering multilingual, RAG, agent
  planning, and code generation capabilities. The Qwen release includes
  **Hugging Face transformers compatibility** and official **AWQ/GPTQ 4/8bit
  quantized models**. On Discord, **TheBloke** community discussed quantization
  of a **70B LLM**, the introduction of the **Sparse MoE model Sparsetral**
  based on Mistral, debates on model merging versus fine-tuning, and techniques
  like **Direct Preference Optimization (DPO)** for character generation. The
  **Nous Research AI** Discord highlighted challenges in Japanese Kanji
  generation, AI scams on social media, and Meta's VR headset prototypes with
  retinal resolution and light field passthrough presented at **SIGGRAPH 2023**.
  Discussions also covered fine-tuning frozen networks and new models like
  **bagel-7b-v0.4**, **DeepSeek-Math-7b-instruct**, and **Sparsetral-16x7B-v2**
  on Hugging Face.
companies:
  - qwenlm
  - deepseek
  - mistral-ai
  - hugging-face
  - meta-ai-fair
  - thebloke
  - nous-research
models:
  - qwen-1.5
  - mistral-7b
  - sparsetral-16x7b-v2
  - bagel-7b-v0.4
  - deepseek-math-7b-instruct
topics:
  - quantization
  - multilinguality
  - rag
  - agent-planning
  - code-generation
  - fine-tuning
  - direct-preference-optimization
  - model-merging
  - memory-augmentation
  - ascii-art
  - character-generation
  - stable-diffusion
  - vr
  - retinal-resolution
  - light-field-passthrough
  - frozen-networks
people:
  - lorenzoroxyolo
  - nonameusr
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/5/2024. We checked **20** guilds, **308** channels, and **5078** messages for you. Estimated reading time saved (at 200wpm): **418 minutes**.

The Chinese models (Yi, Deepseek, and Qwen, to a lesser extent Zhipu) have been quietly cooking up a storm. [Qwen's release this week](https://qwenlm.github.io/blog/qwen1.5/) claims strong performance vs Mistral and Llama2 equivalents:

 ![image.png](https://assets.buttondown.email/images/91b07fee-1f9a-4d08-9570-20f936d388fd.png?w=960&fit=max) 

with up to 32k token context. The technical report also discusses a number of evals made on multilingual, RAG, agent planning, and code generation capabilities. The Qwen team are also showing serious dedication to the downstream ecosystem, releasing with HF transformers compatibility and official AWQ/GPTQ 4/8bit quantized models.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Quantization Quest for 70B LLM**: An exploration of quantizing a 70B LLM on vast.ai was shared with suggestions like creating a large swap file and potentially using USB-attached SSDs to circumvent powerful GPU requirements.

- **GPTZero Faces Scrutiny**: Debates over the effectiveness of AI content detection tools like GPTZero sparked discussions, highlighting its potential unreliability in detecting subtly augmented prompts.

- **Introducing Sparsetral**:
    A new Sparse MoE model based on Mistral was introduced, emphasizing its efficient operation and selective weight application during forward passes, garnering interest and sparking inquiries into its training intricacies.

- **Merging Vs. Fine-Tuning Dilemma**: There's an ongoing debate on whether it's more efficient to individually fine-tune models for separate datasets or to combine datasets and fine-tune a single model, with the community generally leaning towards the latter for coherence.

- **Knowledge Sharing in AI**: Community members discussed a range of topics involving LLM performance and handling, including strategies to augment memory capabilities and sharing of problem-solving tactics, reinforcing the collaborative spirit within the guild.

- **Deep Dive into DPO and Character Generation**: Insights were exchanged on merging adapters in the training process using Direct Preference Optimization (DPO), with a focus on role-playing character generation, as well as tactics to prevent overfitting in such models.

- **Eldritch ASCII Art Conversations**: Attempts at creating ASCII art using various models prompted discussions on the evolving capabilities of language models in creative endeavors.

- **Enigmas of Model Merging**: A desire to comprehend model merging led to sharing resources that delve into the tensor operations involved, accompanied by recommendations of tools like ComfyUI-DareMerge for the task.

- **Coding Discussions Span Character Memory and 3D Generation**: Inquiries about setting up ChromeDB for character-specific long-term memory were seen alongside promotions of a text-to-3D gen AI project, solutions for OpenAI costs, and shared links to code-generating LLMs with detailed explanations.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Kanji Generation's Complex Quest**: The community discussed the challenges in training a model for Japanese Kanji generation, with `@lorenzoroxyolo` referencing a [Stable Diffusion experiment](https://x.com/hardmaru/status/1611237067589095425?s=61) for inspiration. The use of a controlnet was suggested by `.ben.com` as an alternative method for nuanced tasks such as this.
  
- **AI Scams Surface on Social Media**: A surge in AI-related scams on platforms like Facebook prompted community members to discuss the importance of awareness and the detrimental impact fictional narratives can have on AI's perception.

- **Meta's Pioneering VR Prototypes at SIGGRAPH**: Meta's advancement in VR technology, specifically the development of new headset prototypes with retinal resolution and advanced light field passthrough, was presented at SIGGRAPH 2023 and shared by `@nonameusr` with relevant articles from [Road to VR](https://www.roadtovr.com/meta-prototype-vr-retinal-resoltion-light-field-passthrough/) and a [developer blog post](https://www.meta.com/en-gb/blog/quest/reality-labs-research-display-systems-siggraph-2023-butterscotch-varifocal-flamera/).

- **Finetuning Frozen Networks and New AI Models**: Discussions revolved around the effectiveness of fine-tuning normalization layers in frozen networks, as described in [an arXiv paper](https://arxiv.org/abs/2302.07937), and the sharing of information on new models like **bagel-7b-v0.4**, **DeepSeek-Math-7b-instruct**, and **Sparsetral-16x7B-v2** across various Hugging Face repositories, each with unique capabilities and suggested improvements.

- **Performance Review and Anticipated Release**: The community scrutinized the performance of different models, including Qwen 1.5’s release which some found underwhelming compared to its predecessor. Additionally, an announcement was made about an unspecified release happening in 23.9 hours; [fblgit unveiled a new model-similarity analysis tool](https://github.com/fblgit/model-similarity) on GitHub for community contribution.

- **Transformer Matrices and LLM Conversation Memory Debate**: Practical engineering advice was shared, such as fusing QKV matrices in transformers for efficiency. Users also explored techniques for managing conversation history with LLMs, noting the potential use of **langchain** and the benefits of summarizing history or utilizing long-context models to navigate context size limitations. Concerns were raised over licensing changes between various Hermes datasets for commercial use.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Initiative to Build Foundational Models Kindles Interest**: `@pratikk10` seeks collaboration on creating foundational models, acknowledging diverse applications like text-to-text/image/video. However, `@_3sphere` highlights the **prohibitive costs** of such models, discussing the matter in the context of the newly released Qwen1.5, a 72B parameter chat model detailed in their [blog post](http://qwenlm.github.io/blog/qwen1.5/) and [repository](https://github.com/QwenLM/Qwen1.5).

- **Grapple with Interpreting LLMs**: Debates on the effectiveness of interpretability in large language models (LLMs) ensue, drawing parallels to the human genome project and questioning the relationship between interpretability and intelligence. There's a critical assessment of AGI claims and model capabilities, particularly the authenticity of performance on benchmarks such as the MMLU.

- **Scaling Law Reconnaissance**: `@stellaathena` discusses possible efficiency improvements in scaling laws research, referencing Kaplan et al. and Hoffman et al. with interest in reducing the necessity for numerous runs. Contributions from `@clashluke` and others ponder the application of **PCGrad** in multi-task loss handling, the role of hypernetworks in generating LoRA weights, and the use of varying activation functions like polynomials, substantiated by a neural-style [Python file](https://github.com/ProGamerGov/neural-style-pt/blob/master/neural_style.py#L404) and [facebookresearch's code](https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py).

- **Bootstrapping Pooled Variance for Rigorous Model Evaluation**: `@hailey_schoelkopf` updates the **lm-evaluation-harness** to use pooled variance for standard error, an optimally chosen approach over combined variance [as detailed here](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390), prompting `@stellaathena` to recommend preserving both methods with expert-use warnings.

- **Discern Vector Semantics in Activation Functions**: `@digthatdata` and `@norabelrose` dissect the conception of vectors as directions, operators, and Euclidean representations in the context of deep learning, complemented by the introduction of `[model-similarities](https://github.com/fblgit/model-similarity)`, a tool for layer-wise parameter space analysis of different models.

- **Remedying LM Pre-training with Paraphrased Web Data**: New initiatives like **Web Rephrase Augmented Pre-training (WRAP)**, documented in an [arxiv paper](https://arxiv.org/abs/2401.16380), aim to augment large model pre-training by improving data quality, with `@elliottdyson` suggesting a comparative study to gauge WRAP's advantage over mere fine-tuning.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **AI Journey from Novice to Pro**: @dykyi_vladk discussed their ML learning curve, focusing on specific models and techniques, while @lunarflu highlighted the importance of *building demos and sharing them* as a vital step to advance as an ML professional.

- **A100G Launch May Cause Server Hiccups**: Server downtime raised questions about its relation to the A100G launch; @lunarflu offered to escalate the issue. For full AI model utilization, @meatfucker recommended distributing tasks across multiple GPUs.

- **Experts in Computer Vision Sought**: @danielsamuel131 invited computer vision specialists to share their expertise with the community.

- **Papillon Flaunts NER & Sentiment Tool**: An NER and sentiment analysis tool developed by @8i8__papillon__8i8d1tyr, based on Flair and FLERT, was shared; find it on [GitHub](https://github.com/CodeAKrome/bootcupboard/blob/main/flair/SentimentalNERD.py).

- **LLaMA-VID Debuts for Long Videos**: The **LLaMA-VID** model, designed to support hour-long videos, was introduced by @tonic_1. However, user concerns over empty model cards and lack of details may hinder usage. An [arXiv paper](https://arxiv.org/abs/2312.08361)  on cost-efficient LLM strategies was also shared by @jessjess84.

- **Boosting Conversational AI with Fine-Tuning**: @joeyzero seeks resources on conversational datasets for fine-tuning a chatbot. Meanwhile, @denisjannot struggled with fine-tuning Mistral 7b for YAML generation and eyed the Instruqt model for improvement. @meatfucker recommended few-shot learning techniques for making precise YAML modifications.

- **Ankush's Finetuning Mastery**: Ankush Singal's finetuned model, based on previous work from OpenPipe, earned community kudos. The model is available at [Hugging Face](https://huggingface.co/Andyrasika/mistral-ft-optimized-dpo).

- **Schedule Flex for Reading Group**: @ericauld may need to postpone a planned talk due to jury duty, with @chad_in_the_house expressing support. Keep an eye on the events calendar for potential changes.

- **The Meme Highlighting AI Progress**: @typoilu presented an article detailing AI advancements through the lens of a meme in the "Mamba Series LLM Enlightenment," but community engagement on the content remains to be seen. Read the article [here](https://www.marktechpost.com/2024/02/03/a-memes-glimpse-into-the-pinnacle-of-artificial-intelligence-ai-progress-in-a-mamba-series-llm-enlightenment/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Model Selection Mayhem**: Amidst numerous models, `@hades____` sought advice on picking the right one, with suggestions pointing to use-case focus over a universal solution.
- **DeepSeek-Math-7B Launches**: **DeepSeek-Math-7B** was unveiled by `@czkoko` as the latest entry in the model arena, catered specifically to mathematical problem-solving and research, and available on [GitHub](https://github.com/deepseek-ai/DeepSeek-Math).
- **LM Studio Polishes Performance**: **LM Studio v0.2.13** brings new features such as **Qwen 1.5 support**, pinning models and chats, and quality of life updates, downloadable at [LM Studio's website](https://lmstudio.ai) with open-sourced Qwen1.5 models on [Hugging Face](https://huggingface.co/Qwen).
- **Hardware Discourse Heats Up**: Operating system compatibility, GPU utilization, and speed optimizations in token generation dominated discussions, as users shared experiences with LMStudio on different hardware configurations; recommendations included using Ubuntu 22 and quantization methods, as illustrated in comparisons on [YouTube](https://youtu.be/Eaz-H-3FkZg).
- **Beta and Feature Beckoning**: A beta preview of **LM Studio v0.2.13** dropped inviting feedback, while users appealed for a model leaderboard in-app but meanwhile can reference rankings on [Hugging Face Leaderboard](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard) and [OpenRouter](https://openrouter.ai/rankings); a GUI for server access and improved chat interfaces were hot topics.
- **RAG System Question Creation Quest**: `@varelaseb` sought techniques for generating questions in a **RAG system**, inquiring about **Tuna** without much background info available on it.




---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **LangChain's Uncertain Future**: Discord users expressed concerns over the longevity of **LangChain's** utility, suggesting a need for stabilization after a week. Meanwhile, the deterministic nature of **Mistral 8x7B** faced scrutiny, establishing its probabilistic behavior adjustable by a temperature parameter, yet no definitive resolution was reached on the deterministic inquiry.

- **Mistral's Emoji-Terminating Quirk**: A peculiar **terminating behavior** was observed in Mistral models, where responses ended with the "stop" finish_reason but still contained an **emoji**. The issue was noted across all three Mistral API models, signaling a potential area for debugging or insight into response construction.

- **AI's Philosophical Conundrum**: Philosophical implications of LLMs were deliberated, with participants promoting a deeper understanding of AI's foundational principles. This discussion underscores the evolving complexity of AI impacts on broader intellectual fields.

- **Prompt Precision Practices**: Discussion around refining prompts to improve accuracy was active, with the conversion of a PHP class to a JSON schema being one method shared. However, methods for synthetic dataset generation, despite being a hot topic, remained close-guarded due to its value as a source of income.

- **Fine-Tuning Pad Pains**: Concerns about padding during fine-tuning were voiced, pointing out issues with models not generating end-of-sentence tokens using the common `tokenizer.pad = tokenizer.eos` practice. This suggests a need for optimized fine-tuning approaches to enhance model performance.

- **Lean Chatbots and Starry Success**: A **Discord chatbot** capable of multi-user interaction and supporting multiple LLMs, including Mistral, made headlines with its lean 200-line code and functionality like vision support and streamed responses, attracting attention with over 69 GitHub stars. [GitHub - jakobdylanc/discord-llm-chatbot](https://github.com/jakobdylanc/discord-llm-chatbot)

- **Flags and Fast GIFs**: Among lighter interactions, users shared flag emojis and humorous GIFs, hinting at a casual and engaging community dynamic. Specifically mentioned was a **Sanic the Hedgehog** GIF from Tenor, celebrated for its humor in the context of a language settings discussion. [Sanic The Hedgehob GIF - Tenor](https://tenor.com/view/sanic-the-hedgehob-running-gotta-go-fast-fast-gif-4964355)



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Enthusiastic Engineering for Dual GPU Builds**: Community member `@morgangiraud` is assembling a dual build with 2 x **4070 TI SUPER** GPUs and considering VRAM trade-offs between newer and older cards. They shared their build costing 4k in total through [PCPartPicker](https://pcpartpicker.com/user/morgangiraud/saved/VTZRFT).

- **Library Launch to Lighten LLM Load**: `@andreaskoepf` highlighted [FlashInfer](https://github.com/flashinfer-ai/flashinfer), an open-source library aimed at boosting performance of LLM serving, by optimizing Self-Attention and other key operations.

- **Precision Predicament Perplexes PyTorch Programmer**: `@zippika` encountered inaccuracies with dequantize and linear operations in PyTorch and sought the cause, speculating it may be rounding issues or related to disabled C++ flags like `"__CUDA_NO_HALF_OPERATORS__"`.

- **GPU Kernel Conundrums in JAX**: `@stefangliga` introduced **Pallas**, an experimental JAX extension for writing custom GPU/TPU kernels, while `@nshepperd` shared insights on using pure CUDA kernels within JAX. `@marvelousmit` inquired about methods to print Triton kernels and the code JAX executes for kernel profiling.

- **Recorded Resourcefulness for CUDA Coders**: Mark Saroufim assured users that despite technical delays, Lecture 4's recording has been uploaded to [YouTube](https://www.youtube.com/watch?v=lTmYrKwjSOU), promising HD quality shortly after.

- **Fast.ai Favoritism Flows Freely**: Users `@einstein5744` and `@joseph_en` signal satisfaction with fast.ai's educational resources, particularly regarding a course on diffusion models and the **DiffEdit paper**.

- **Weighing Words of Wisdom for March 9th Meet-up**: `@jku100` is tentatively set to speak on March 9th about their work with the `torch.compile` optimizations, which has shown promise in AI acceleration techniques.

- **Preparing for PMPP's Fifth Lecture**: `@jeremyhoward` scheduled lecture 5 for the weekend, linking the [Discord event](https://discord.gg/pBhQAAvB?event=1204175111633113168), while `@lancerts` queried about the 'swizzled order' concept discussed in a [PyTorch blog](https://pytorch.org/blog/accelerating-triton/) and its coverage in the PMPP book.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **ChatGPT File Upload Fiasco**: Users, including `@sherlyleta`, `@guti_310`, and `@lugui`, have brought up ongoing issues with the **ChatGPT file upload feature**, which has been glitchy since the previous week. `@lugui` mentioned that resolution is on the horizon.
- **Firmware Fix Frenzy in Manufacturer Talk**: Debates heated on manufacturer responsibility for technical issues, with `@aipythonista` advocating for firmware updates as solutions instead of relying on content like Louis Rossman's, citing potential brand bias.
- **Mistral GPT Alternatives Gather Steam**: Amidst talks of local GPT-3.5 instances and the infeasibility pointed out by `@elektronisade`, `@riaty` recommended the open-source **Mistral 8x7b** for homelab diagnostics.
- **Trademark Tangles Trouble GPT Customizer**: `@zurbinjo` faced trademark obstacles when naming a GPT, with clarification from `@solbus` about the prohibition due to OpenAI's [branding guidelines](https://openai.com/brand#gpts-in-chatgpt).
- **Perfecting PDF Presentations to AI**: A brief exchange prompted by `@wazzldorr` explored whether AIs perform better processing **PDFs** or **extracted text** for scientific articles, with `@lugui` assuring AI's capability to handle PDFs effectively.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **RunPod and BGE-M3 Catch Engineers' Attention**: In the **general** channel, `@lhc1921` highlighted **RunPod** for competitively priced GPU nodes and introduced the **BGE-M3** multi-functional embedding model, providing the GitHub [repository](https://github.com/FlagOpen/FlagEmbedding) and research [paper](https://arxiv.org/pdf/2402.03216.pdf). `@kapa.ai` detailed using **OpenAIEmbedder** with **LangChain**, citing [LangChain's JavaScript documentation](https://js.langchain.com/docs/integrations/text_embedding/openai) and [Python documentation](https://python.langchain.com/docs/integrations/text_embedding/openai).

- **Bearer Token and Setup Woes in LangServe Discussions**: The **langserve** channel saw a sharing of tips on AzureGPT.setHeader with bearer token by `@veryboldbagel`, referencing [Configurable Runnables documentation](https://github.com/langchain-ai/langserve/blob/main/examples/configurable_chain/server.py) and [APIHandler examples](https://github.com/langchain-ai/langserve/blob/main/examples/api_handler_examples/server.py). `@gitmaxd` provided a [guide](https://medium.com/@gitmaxd/your-first-a-i-api-endpoint-with-langserve-deeb65e750b1) for Hosted LangServe setup, while `@lucas_89226` and `@veryboldbagel` engaged in troubleshooting discussions, with a suggestion to use [LangServe GitHub discussions page](https://github.com/langchain-ai/langserve/discussions) for further help.

- **Showcasing Innovations and Job Openings in Share-Your-Work Channel**: `@siddish` introduced AI Form Roast by WorkHack on [Product Hunt](https://www.producthunt.com/posts/ai-form-roast-by-workhack) for online form optimization, and `@shving90` highlighted TweetStorm Express Flow for crafting tweets via a [Twitter post](https://x.com/OranAITech/status/1754461373466042527?s=20). The **Dewy** knowledge base for RAG applications was presented by `@kerinin` with a [blog post](https://dewykb.github.io/blog/introducing-dewy/), while `@hinayoka` announced job opportunities in a crypto project. `@felixv3785` showcased a [Backlink Outreach Message Generator](https://www.backlinkgpt.com/free-seo-tools/backlink-outreach-message-generator) tool for SEO.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI Team Formation Tactics**: Discussions about setting up an internal AI engineering team suggested starting **solo** to showcase value before scaling up. [Eugene Yan's articles](https://eugeneyan.com/writing/real-time-recommendations/#how-to-design-and-implement-an-mvp) on real-time ML and team configurations were recommended, illustrating various organizational tactics including centralization and embedding data scientists into product teams.

- **DSPy Series Simplified**: A video series on DSPy prompted requests for a more digestible explanation, indicating the community's readiness to collaborate on grasping its concepts. The [DSPy explained video](https://youtu.be/ycfnKPxBMck?feature=shared) was shared as a starting point for those interested in learning more.

- **GPT-4, the Procrastinator?**: GPT-4’s perceived initial laziness became a topic of amusement, supported by [Sam Altman's tweets](https://x.com/sama/status/1754172149378810118?s=46&t=90x) and several Reddit discussions that ultimately confirmed GPT-4 should now be "much less lazy," according to the shared community feedback.

- **Philosophical Digital Library Envisioned**: The concept of a digital library with AI philosophical agents led to suggestions of leveraging tools like [Botpress](https://botpress.com/) and [WorkAdventure](https://github.com/workadventure/workadventure) for development, indicating an interest in merging philosophical discourse with AI technology.

- **Technical Setup Exchanges**: Engineers like `@ashpreetbedi` shared their technical setups, which involve tools such as PyCharm and ScreenStudio, reflecting a shared interest in the practical aspects of engineering environments and tooling.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

**Call for Collaboration in Foundational Models**: A discussion initiated by `@pratikk10` invites interested parties to contribute to the creation of foundational models across different media, including text, image, and video, seeking exchanges with serious creators.

**Bias Watch in Reinforcement Learning**: RLHF's introduction of significant biases is debated, with `@pseudoterminalx` and `@astropulse` noting the potentially counterproductive effect on base model development, while also observing a distinctive style in Midjourney's images potentially rooted in such biases.

**Tackling Textual Bias in Pixart**: Conversations reveal challenges in unlearning textual biases from datasets, specifically version 5.1 of pixart. Critique is directed at the use of the JourneyDB dataset, with suggestions to find more robust alternatives for unbiased text modalities.

**Innovative Reading of Ancient Texts**: The **Vesuvius Challenge 2023 Grand Prize** announcement highlighted a successful method for reading 2000-year-old scrolls without unrolling them, using a TimeSformer model and a particle accelerator, although at a high cost of $40,000 per scroll.

**Chinese Machine Learning Thrives Despite Restrictions**: Discussions ponder the success of Chinese ML entities in light of GPU restrictions, noting their preemptive procurement of NVIDIA's H100s and A100s before restrictions came into play, questioning the overall impact on technological progress.

**Critique of Hugging Face's OWLSAM**: `@SegmentationFault` shared and commented on the performance of **OWLSAM** in a [Hugging Face Space](https://huggingface.co/spaces/merve/OWLSAM), indicating that the model lacked coverage in visual representation and accuracy in object detection.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **LlamaIndex Gears Up for Big Release**: A significant release of **LlamaIndex** is due this week, with cleanups signaling an important update for users planning to upgrade their LlamaIndex systems.

- **Boosting Multi-modal Applications on MacBooks**: LlamaIndex's recent integration allows building multi-modal applications on a MacBook, enhancing image reasoning. The related announcement and developments were shared in a [tweet](https://twitter.com/llama_index/status/1754545663155793972).

- **Home AI Triumphs at Hackathon with PDF Search Innovation**: Home AI's unique implementation of a **RAG-powered search engine** for home filtering won Best Use of PDF Parser at an in-person hackathon, details of which can be found in this [tweet](https://twitter.com/llama_index/status/1754601626688749755).

- **Hackathon Spurs LlamaIndex Enhancement**: The hackathon saw participation from nearly 200 people, offering feedback for the LlamaIndex team, and a [resource guide](https://t.co/Oe5l44bSdl) catering to developers was circulated.

- **Building Engineer-Conscious Chatbots**: One discussion focused on creating a chatbot for engineers to interact with standards documents using LlamaIndex, supported by a GitHub project at [GitHub - imartinez/privateGPT](https://github.com/imartinez/privateGPT). 

- **Navigating Vector Search Challenges**: Users discussed methods to improve vector search results with Qdrant, sharing insights into embedding and score analysis, and highlighted the usage of TypeScript code example for generating and comparing embeddings with `Ollama`.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **New Player in Town - Qwen1.5**: The OpenAccess AI community is abuzz with the release of **Qwen1.5**, promising higher performance with quantized versions. The release was accompanied by a [comprehensive blog post](https://qwenlm.github.io/blog/qwen1.5/) and various development resources. However, some users already see room for improvement, noting the absence of a **30b model** and the need for including standard deviation in benchmarks to account for noise.

- **GCP's Competitive Edge**: GCP is extending olive branches to enterprise customers with A100 instances available at **$1.5-2** per hour on demand. This rate is significantly lower than what non-enterprise customers pay, spotlighting strategies employed by cloud providers to manage their ecosystems of users and resellers, including spotlight deals like GCP's L4 spot price at **$0.2** per hour.

- **Bridging the Quantization Gap**: Within the **Axolotl development** community, there's discussion around Hugging Face's suggestion to quantize before merging models. [`@dreamgen`](https://github.com/jondurbin/qlora/blob/main/qmerge.py#L42) suggests that this approach could benefit performance, sparking conversation about the potential of Bayesian optimization within the Axolotl framework and reports of a smoother implementation process.

- **Axolotl's Growing Pains**: Axolotl users are reporting installation woes, noting dependency conflicts, specifically with `torch` and `xformers`. A suggested fix involves using `torch 2.1.2`. Also, there's a call to simplify YAML configurations in Axolotl, indicating that ease of use and accessibility are high on the developer wishlist, along with creating a Hugging Face Spaces UI for a more beginner-friendly setup.

- **Crickets in RLHF**: A lone message in the #rlhf channel, seemingly directed to a specific user, asks for configurations related to **zephyer**, leaving much to the readers’ imagination regarding context or importance, and offers too little to chew on for the tech-hungry engineer audience.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Pro Payment Problems**: Users like `@officialjulian` experienced payment issues with the Pro upgrade—funds were deducted without service activation, suggesting an error with Stripe's payment system, while `@yuki.ueda` faced unresponsive customer support over billing inquiries. It was recommended to reach out to support@perplexity.ai for assistance.

- **AI Ethics in Education Debated**: `@worriedhobbiton` highlighted the need for AI to offer unbiased and culturally sensitive support, especially in educational contexts like Pasco High School, reflecting the challenges in serving diverse student populations.

- **Mismatched AI Research Responses**: User experiences like `@byerk_enjoyer_sociology_enjoyer`'s underscore the limitations of AI in delivering relevant search outcomes, as shown in the shared [Perplexity AI search result](https://www.perplexity.ai/search/42bbb721-0450-47eb-bd01-f4f303e62d79), which did not match the research query about source validation.

- **Speedy Summary Solutions Sought**: `@sid.jjj` expressed the need to improve API response times when generating summaries, noting that current processes take about 10 seconds for three parallel links, underlining a performance benchmark concern for AI engineers.

- **Discrepancies Detected in AI Usage**: Concerns were raised by `@jbruvoll` about inconsistencies between interactive and API use of Perplexity AI, directing to a [specific Discord message](https://discord.com/channels/1047197230748151888/1161802929053909012/1189372086658011237) for further detail, which highlights the importance of aligning AI behavior across different interfaces.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Danish Delight in Model Merging**: A **Danish language model** utilizing the *dare_ties merge* method achieved 2nd place on the Mainland Scandinavian NLG leaderboard, introduced by `@johannhartmann` and detailed [here](https://huggingface.co/RJuro/munin-neuralbeagle-7b).

- **Merge Models Minus Massive Machines**: `@sebastian.bodza` noted that **LeoLM models** can be merged without GPUs, highlighting alternatives like Google Colab for performing model merging tasks.

- **German Giant - Wiedervereinigung-7b-dpo-laser**: `@johannhartmann` unveiled a 7b parameter German model combining top German models, named [Wiedervereinigung-7b-dpo-laser](https://huggingface.co/mayflowergmbh/Wiedervereinigung-7b-dpo-laser), which has high MT-Bench-DE scores.

- **Merging Models More than a Score Game**: The conversation between `@johannhartmann` and `@bjoernp` suggested an improvement in actual use-cases, like chat functions, after merging models, beyond just achieving high scores.

- **Code Cross-Language Searchability Soars**: Jina AI released new code embeddings that support English and 30 programming languages with an impressive sequence length of **8192**, as shared by `@sebastian.bodza`. The model is available on [Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) and is optimized for use via Jina AI's [Embedding API](https://jina.ai/embeddings/).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Taming Llama2's Training Loss**: An engineer faced an unexpected training loss curve with **LLama2** using SFT which might be due to a high learning rate. A peer recommended switching to [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) and provided specific configuration examples that suggest increasing the learning rate.

- **Ankr Networking for Node Collaboration**: `@anastasia_ankr` reached out to discuss node infrastructure with the team and was directed to contact `@748528982034612226` for further dialogue.

- **Community Engagement**: Users `@xterthy` and `@aslawliet` kept the community active with brief greetings, contributing to a friendly atmosphere.

- **Awaiting Direct Communications**: `@mizzy_1100` flagged a direct message for `@748528982034612226`'s attention, indicating important pending communications.

- **Celebrating Collaborative Spirit**: `@rusch` compared the Discord server to an "amazing discordian circus," highlighting its dynamic and entertaining nature for knowledge sharing and innovation.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Audacity Echoes with Intel's AI**: **Audacity**'s integration of **Intel's AI tools** adds powerful local features: [noise suppression](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/noise_suppression/README.md), [transcription](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/whisper_transcription/README.md), [music separation](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/music_separation/README.md), and experimental music generation, presenting a challenge to expensive subscription services.

- **Thesis Enhancement with LLM**: On **LLM integration**, `@kiloton` seeks advice for handling **PDFs and web searches**, and whether chat histories can be transferred and stored across different models.

- **SQL, Simplified with Hugging Face**: `@dbreunig` points to the integration potential of `llm` with **Hugging Face's transformers**, highlighting the [Natural-SQL-7B](https://huggingface.co/chatdb/natural-sql-7b) model for its advanced Text-to-SQL capabilities and deep comprehension of complex questions.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Qwen1.5 Debuts with Open Source Fanfare**: **Qwen1.5** has been introduced and open-sourced, offering base and chat models across six sizes. Resources shared by `@potrock` include a [blog post](https://qwenlm.github.io/blog/qwen1.5/), [GitHub repository](https://github.com/QwenLM/Qwen1.5), a presence on [Hugging Face](https://huggingface.co/Qwen), [Modelscope](https://modelscope.cn/organization/qwen), a user-friendly [demo](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat), and an invitation to join the [Qwen Discord community](https://discord.gg/yPEP2vHTu4).
- **Efficiency Breakthrough with Qwen1.5**: The **0.5B Qwen1.5 model** has shown promise by exhibiting performance on par with the much larger Llama 7B model, signaling a new wave in model efficiency optimization as shared by `@potrock`.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1203981148812738580) (1293 messages🔥🔥🔥): 

- **Model Quantization Guidance**: `@xmrig` explored quantizing the 70B LLM model on vast.ai due to local resource limitations. Advice offered included creating a large swap file and potentially using external USB-attached SSDs, avoiding the need for a powerful GPU during the quantization process (`@spottyluck`, `@rtyax`, `@stoop poops`).

- **GPTZero Analysis**: Users debated the effectiveness of AI content detection tools such as GPTZero, with suggestions that it may not reliably detect more subtly augmented prompts and that it's seen as a student's tool, making it far from product-ready (`@mr.userbox020`, `.meathead`, `@kaltcit`, `@righthandofdoom`, `@itsme9316`).

- **Sparsetral, a New Sparse MoE Model**: `@morpheus.sandmann` shared a Sparse MoE model based on Mistral, underscoring efficient operation on high-end hardware. The sparse model uses only a portion of weights during forward passes, applying adapters selectively through a router, which sparked the community's interest but also raised questions on the intricacies of its training and functionality (`@netrve`, `@itsme9316`).

- **Merging vs. Single Model Fine-tuning**: `@givan_002` inquired about the efficiency of fine-tuning separate models for different datasets versus fine-tuning one model on a combined dataset. The consensus leaned towards using one comprehensive set for coherence and optimization (`@amogus2432`).

- **Community Assistance with LLM Tasks**: Users `@kaltcit`, `@potatooff`, and others discussed various topics from LLM performance to practical advice on using LLMs and related technologies, like swapping VRAM to augment memory, showcasing collaborative problem-solving and knowledge sharing within the community.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1111984430945402960/1204096739318177955): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [no title found](https://www.apa.org/news/podcasts/speaking-of-psychology/attention-spans): no description found
- [Realtime Colors](https://www.realtimecolors.com/?colors=fddbfd-250222-f97ae9-8f9b07-33f40c&fonts=Poppins-Poppins): Visualize your color palettes on a real website.
- [Screenshot to HTML - a Hugging Face Space by HuggingFaceM4](https://huggingface.co/spaces/HuggingFaceM4/screenshot2html): no description found
- [Realtime Colors](https://www.realtimecolors.com/?colors=fddbfd-250222-f97ae9-8f9b07-33f40c&font): Visualize your color palettes on a real website.
- [Qwen1.5 - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524): no description found
- [Qwen/Qwen1.5-14B-Chat-GGUF · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF): no description found
- [Huang Jensen Nvidia Ceo GIF - Huang Jensen Nvidia Ceo - Discover &amp; Share GIFs](https://tenor.com/view/huang-jensen-nvidia-ceo-gif-19751265): Click to view the GIF
- [budecosystem/code-millenials-13b · Hugging Face](https://huggingface.co/budecosystem/code-millenials-13b): no description found
- [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/#basic-capabilities): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...
- [Bing GIF - BING - Discover &amp; Share GIFs](https://tenor.com/view/bing-gif-25601964): Click to view the GIF
- [TheBloke/Llama-2-70B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Llama-2-70B-GGUF): no description found
- [NousResearch/Nous-Hermes-Llama2-13b · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b): no description found
- [Social Credit GIF - Social Credit - Discover &amp; Share GIFs](https://tenor.com/view/social-credit-gif-23976170): Click to view the GIF
- [dataautogpt3/miqu-120b · Hugging Face](https://huggingface.co/dataautogpt3/miqu-120b): no description found
- [GitHub - TheBlokeAI/dockerLLM: TheBloke&#39;s Dockerfiles](https://github.com/TheBlokeAI/dockerLLM): TheBloke&#39;s Dockerfiles. Contribute to TheBlokeAI/dockerLLM development by creating an account on GitHub.
- [wolfram/miquliz-120b · Hugging Face](https://huggingface.co/wolfram/miquliz-120b): no description found
- [v0 by Vercel](https://v0.dev/): Generate UI with simple text prompts. Copy, paste, ship.
- [Swap on video RAM - ArchWiki](https://wiki.archlinux.org/title/Swap_on_video_RAM): no description found
- [Why No One Feels Like They Can Focus Anymore](https://time.com/6302294/why-you-cant-focus-anymore-and-what-to-do-about-it/): And what to do about it 
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/o4RNEdHYpk): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1203973968973930497) (457 messages🔥🔥🔥): 

- **Dynamic Adapter Merging in Training**: `@jondurbin` recommends merging the adapter from SFT only after DPO, rather than before, continuing with the adapter from SFT throughout the process. Insights came while discussing the [Direct Preference Optimization (DPO) Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) as detailed in the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer).
  
- **Discussions on DPO and Adapter Loading**: `@dreamgen` shares information that the [DPO Trainer expects a specific dataset format](https://huggingface.co/docs/trl/main/en/dpo_trainer#using-option-3---load-the-adapter-twice), citing examples from the `Anthropic/hh-rlhf` dataset. Issues such as the attribute name `train` conflicting with the latest transformers version are addressed by `@jondurbin`.

- **CharGen v2 - A Model for Roleplaying Creatives**: `@kalomaze` unveils CharGen v2, a model designed to generate character descriptions for role playing, featured on [Hugging Face](https://huggingface.co/kubernetes-bad/chargen-v2) with a [live version available here](https://chargen.kubes-lab.com). The model creates character descriptions in a dialogue format, generating one field at a time to allow for partial re-rolls and reduce repetition.

- **Fine-Tuning Role Play Models with Diverse Data**: Users discuss strategies for preventing overfitting in role play models, suggesting a mix of RP data with varied datasets like The Pile or MiniPile at the start of each epoch (`@kalomaze`). `@stoop poops` and `@flail_.` exchange views on enhancement tactics like incorporating assistant data with RP data to avoid dumb outputs.

- **Eldritch ASCII Art Endeavors**: `@c.gato` and others experiment with generating ASCII art using various models like Mixtral, Miqu, and GPT-4. The conversation showcases attempts at creating simple ASCII art with varying degrees of success, highlighting the limited but improving abilities of language models in this creative task.

**Links mentioned**:

- [kubernetes-bad/chargen-v2 · Hugging Face](https://huggingface.co/kubernetes-bad/chargen-v2): no description found
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf): no description found
- [G-reen (G)](https://huggingface.co/G-reen): no description found
- [GitHub - MeNicefellow/Intelligent_RolePlaying_Sandbox](https://github.com/MeNicefellow/Intelligent_RolePlaying_Sandbox): Contribute to MeNicefellow/Intelligent_RolePlaying_Sandbox development by creating an account on GitHub.
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#using-option-3---load-the-adapter-): no description found
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#using-option-3---load-the-adapter-twice): no description found
- [bigscience/sgpt-bloom-7b1-msmarco · Hugging Face](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco): no description found
- [Norquinal/claude_multiround_chat_30k · Datasets at Hugging Face](https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k): no description found
- [ASCII Art Archive](https://www.asciiart.eu/): A large collection of ASCII art drawings and other related ASCII art pictures.

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1204186695474356255) (2 messages): 

- **Fine-tuning Llama2-7b-chat with website content**: `@gabrielelanzafame` inquired about the possibility of fine-tuning **Llama2-7b-chat** using text scraped from a website. They want to train the model to generate copy in the brand's tone or evaluate the brand tone in given copy.
- **Strategies for Fine-tuning with Multiple Datasets**: `@givan_002` asked whether it is more effective to fine-tune a separate model for each individual dataset—**airoboros, hermes, limarp**—and then merge them, or to combine all datasets and fine-tune a single model.
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1204191511005237289) (3 messages): 

- **Seeking Wisdom on Model Merging**: `@noobmaster29` expressed a desire to understand **model merging** at a deeper level, beyond just demo notebooks, seeking sources for reading or video explanations.
- **Diving Deep into Model Merging**: `@maldevide` referenced their own **gist** as a thorough breakdown of model merging at the tensor operation level, aimed at providing a deeper understanding.
- **Tool Suggestion for Model Merging**: `@maldevide` suggested using [ComfyUI-DareMerge](https://github.com/54rt1n/ComfyUI-DareMerge), a tool that facilitates **model merging** for SD1.5 and SDXL, as a convenient resource already present in their notebook.

**Links mentioned**:

[GitHub - 54rt1n/ComfyUI-DareMerge: ComfyUI powertools for SD1.5 and SDXL model merging](https://github.com/54rt1n/ComfyUI-DareMerge): ComfyUI powertools for SD1.5 and SDXL model merging - GitHub - 54rt1n/ComfyUI-DareMerge: ComfyUI powertools for SD1.5 and SDXL model merging

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1203973253371002962) (4 messages): 

- **Creating Character Memory with ChromeDB**: `@vishnu_86081` inquired about implementing long-term memory for each character in a chatbot app by using **ChromeDB**. They're currently using **ooba web UI API** for text generation and **MongoDB** for storing messages, and they seek guidance on setting up ChromeDB to separate messages of each character.

- **Seeking Shared Links for neThing.xyz**: `@rawwerks` requested the re-sharing of links for their **text-to-3D gen AI project** called [neThing.xyz](https://nething.xyz), expressing concerns over their OpenAI costs while offering free user trials.

- **Code-13B and Code-33B Links Reshared**: `@london` shared links to **Code-13B** and **Code-33B**, two Large Language Models (LLMs) trained to generate code with detailed explanations, available on the Hugging Face platform. These models were trained using the datasets [Python-Code-23k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Python-Code-23k-ShareGPT) and [Code-74k-ShareGPT](https://huggingface.co/datasets/ajibawa-2023/Code-74k-ShareGPT), with the former taking 42 hours and the latter 6 days & 5 hours to train.

**Links mentioned**:

- [neThing.xyz - AI Text to 3D CAD Model](https://nething.xyz): 3D generative AI for CAD modeling. Now everyone is an engineer. Make your ideas real.
- [ajibawa-2023/Code-13B · Hugging Face](https://huggingface.co/ajibawa-2023/Code-13B): no description found
- [ajibawa-2023/Code-33B · Hugging Face](https://huggingface.co/ajibawa-2023/Code-33B): no description found

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1203976333437313064) (28 messages🔥): 

- **Quest for Kanji Mastery**: User `@lorenzoroxyolo` expressed frustration with their attempts to train a model on Japanese Kanji generation, referring to a [Stable Diffusion experiment](https://x.com/hardmaru/status/1611237067589095425?s=61) by `@hardmaru` for inspiration. User `.ben.com` recommended considering a controlnet instead of stable diffusion for nuanced tasks like rendering complex images.
  
- **Model Training Challenges Discussed**: Amidst the concerns about unsatisfactory kanji generation models, `.ben.com` suggested `@lorenzoroxyolo` could learn from the [IDS repository](https://github.com/cjkvi/cjkvi-ids) to understand the structure of characters better and potentially optimize model training.

- **AI Breeding Game Theory Emerges**: `@bananawalnut69` speculated about an "AI breeding" game to generate child models using a method akin to the GAN approach, with `@Error.PDF` responding that reinforcement learning might align with the concept.

- **Scams Awareness Raised**: Discussion about the prevalence of AI-related scams on Facebook led `@Error.PDF` to lament the misinformation and shallow perceptions fostered by fictional narratives, which results in references to Skynet or WALL·E in serious AI talks.

- **Meta Pushes VR Boundaries**: `@nonameusr` shared a link to a [Road to VR article](https://www.roadtovr.com/meta-prototype-vr-retinal-resoltion-light-field-passthrough/) and an accompanying [developer blog post](https://www.meta.com/en-gb/blog/quest/reality-labs-research-display-systems-siggraph-2023-butterscotch-varifocal-flamera/) about Meta's new VR headset prototypes with retinal resolution and advanced light field passthrough capability presented at SIGGRAPH 2023.

**Links mentioned**:

- [Huh Cat Huh M4rtin GIF - Huh Cat Huh M4rtin Huh - Discover &amp; Share GIFs](https://tenor.com/view/huh-cat-huh-m4rtin-huh-huh-meme-what-cat-gif-27377993): Click to view the GIF
- [Skeleton Skeleton Laugh GIF - Skeleton Skeleton laugh Laugh - Discover &amp; Share GIFs](https://tenor.com/view/skeleton-skeleton-laugh-laugh-skull-skull-explosion-gif-6339936494140884475): Click to view the GIF
- [Tweet from hardmaru (@hardmaru)](https://x.com/hardmaru/status/1611237067589095425?s=61): A #StableDiffusion model trained on images of Japanese Kanji characters came up with “Fake Kanji” for novel concepts like Skyscraper, Pikachu, Elon Musk, Deep Learning, YouTube, Gundam, Singularity, e...
- [Meta Reveals New Prototype VR Headsets Focused on Retinal Resolution and Light Field Passthrough](https://www.roadtovr.com/meta-prototype-vr-retinal-resoltion-light-field-passthrough/): Meta unveiled two new VR headset prototypes that showcase more progress in the fight to solve some persistent technical challenges facing VR today. Presenting at SIGGRAPH 2023, Meta is demonstrating a...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1204055209437954058) (14 messages🔥): 

- **Boosting Frozen Networks via Fine-Tuning**: @euclaise discussed the potential of fine-tuning only the normalization layers of frozen networks, hinting it could be a promising approach as an alternative to LoRA. This concept is based on the findings from [a recent arXiv paper](https://arxiv.org/abs/2302.07937).

- **New Flavor of Bagel Unleashed**: @nonameusr shared a link to the non-DPO version of the Mistral-7b model fine-tuned, known as **bagel-7b-v0.4** on Hugging Face. It's reported that this version is *better for roleplay usage*, and the [model card](https://huggingface.co/jondurbin/bagel-7b-v0.4) outlines compute details and data sources, with a DPO variant expected soon.

- **DeepSeek Unveils Math-Savvy Model**: @metaldragon01 introduced the **DeepSeek-Math-7b-instruct** model, alongside links to use cases and a paper detailing the model's capabilities for mathematical reasoning using chain-of-thought prompts. [Model details can be found on Hugging Face](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct).

- **Sparse Modeling with Sparsetral**: @mister_poodle provided a link to **Sparsetral-16x7B-v2**, a model trained with QLoRA and MoE adapters. The [Hugging Face model card](https://huggingface.co/serpdotai/sparsetral-16x7B-v2) supplies key information on training, prompt format, and usage.

- **Sparsetral Ramblings on Reddit**: @dreamgen pointed to a Reddit post about Sparsetral, a sparse MoE model derived from Mistral, alongside several links to resources and papers. The discussion also suggests improving Sparsetral by initializing experts from Mixtral, with the [model available on Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2).

**Links mentioned**:

- [deepseek-ai/deepseek-math-7b-instruct · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct): no description found
- [serpdotai/sparsetral-16x7B-v2 · Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2): no description found
- [jondurbin/bagel-7b-v0.4 · Hugging Face](https://huggingface.co/jondurbin/bagel-7b-v0.4): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ajwijf/model_release_sparsetral/): no description found
- [The Expressive Power of Tuning Only the Normalization Layers](https://arxiv.org/abs/2302.07937): Feature normalization transforms such as Batch and Layer-Normalization have become indispensable ingredients of state-of-the-art deep neural networks. Recent studies on fine-tuning large pretrained mo...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1203975647131729940) (514 messages🔥🔥🔥): 

- **IPO Performance Revisited**: `@teknium` commented on IPO paper recommendations improving IPO, but `@dreamgen` noted DPO still outperforms IPO in tests atop open hermes. `@teknium` acknowledged outdated information and a missed update.
- **Quantization Sensitivity Discussed**: `@dreamgen` highlighted that DPO is sensitive to beta settings, potentially problematic for users unable to run extensive beta sweeps. `@teknium` replied with Hermes Mixtral sensitivity to beta, indicating the issue is broader.
- **Upcoming Release Anticipation**: `@main.ai` announced that 23.9 hours remain until an unspecified release based on AoE time, linking to a tweet for confirmation.
- **Introduction of model-similarity Tool by fblgit**: `@fblgit` presented a new tool for analyzing model similarities, capable of understanding weight differences and parameter alignment between various models. The tool is open for contributions on GitHub.
- **Quality Concerns over Qwen 1.5 Release**: Amidst the Qwen 1.5 release, `@nonameusr` expressed underwhelmed sentiment, pointing to minimal benchmark improvements over Qwen 1. Meanwhile, `@euclaise` and `@metaldragon01` discussed the prospects of smaller Qwen models and a potential Qwen-Miqu model merge.

**Links mentioned**:

- [Qwen1.5 72B Chat - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat): no description found
- [HuggingChat - Assistants](https://huggingface.co/chat/assistants): Browse HuggingChat assistants made by the community.
- [Social Credit GIF - Social Credit - Discover &amp; Share GIFs](https://tenor.com/view/social-credit-gif-23165146): Click to view the GIF
- [You Naughty Naughty Pointing GIF - You Naughty Naughty Pointing Smile - Discover &amp; Share GIFs](https://tenor.com/view/you-naughty-naughty-pointing-smile-you-gif-17657303): Click to view the GIF
- [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...
- [Qwen/Qwen1.5-7B-Chat-GGUF · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF): no description found
- [CausalLM/34b-beta · Hugging Face](https://huggingface.co/CausalLM/34b-beta): no description found
- [wolfram/miquliz-120b · Hugging Face](https://huggingface.co/wolfram/miquliz-120b): no description found
- [Tweet from qnguyen3 (@stablequan)](https://x.com/stablequan/status/1754679410773619003?s=20): Introducing Quyen, our first flagship LLM series based on the Qwen1.5 family with 6 different sizes: Quyen-SE (0.5B) Quyen-Mini (1.8B) Quyen (4B) Quyen-Plus (7B) Quyen-Pro (14B) Quyen-Pro-Max (72B) Al...
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://fxtwitter.com/reach_vb/status/1754263336642867493): It’s done! 🤯  miqudev merged the PR from @arthurmensch  ↘️ Quoting Vaibhav (VB) Srivastav (@reach_vb)   leak/ acc.
- [Tweet from Awni Hannun (@awnihannun)](https://fxtwitter.com/awnihannun/status/1754542587678220536?s=20): Qwen1.5 is out, and already works with MLX !  pip install -U mlx-lm  Models from 0.5B to 72B, all super high quality.  0.5B runs fast with MLX on my laptop, high quality, hardly any RAM:  ↘️ Quoting J...
- [Kind request for updating MT-Bench leaderboards with Qwen1.5-Chat series · Issue #3009 · lm-sys/FastChat](https://github.com/lm-sys/FastChat/issues/3009): Hi LM-Sys team, we would like to present the generation results and self-report scores of Qwen1.5-7B-Chat, Qwen1.5-14B-Chat, and Qwen1.5-72B-Chat on MT-Bench. Could you kindly help us verify them a...
- [GitHub - fblgit/model-similarity: Simple Model Similarities Analysis](https://github.com/fblgit/model-similarity): Simple Model Similarities Analysis. Contribute to fblgit/model-similarity development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1204118828217405440) (36 messages🔥): 

- **Fusing QKV Matrices Optimizes Performance**: `@carsonpoole` clarified to `@sherlockzoozoo` that fusing the Query, Key, and Value (QKV) matrices in transformer models is **mathematically identical** but slightly faster, as it reduces the number of operations and memory loads required.
- **Understanding LLMs for Conversation Memory**: In a conversation initiated by `@lushaiagency`, `@4biddden` mentioned using **langchain** to handle conversation history with LLMs, while `.ben.com` and `@samuel.stevens` discussed issues around context size and history breakdown, suggesting that summarizing history or using long-context models could be solutions.
- **Licensing Questions on Hermes 2.5 Dataset**: `@tculler91` expressed concerns about the licensing change between the OpenHermes and Hermes 2.5 datasets, looking for clarification for commercial use.
- **Configuring Special Tokens for OpenHermes**: `@gabriel_syme` sought advice on token configurations for fine-tuning an OpenHermes model, and `@teknium` advised including `"
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1204051927290224660) (363 messages🔥🔥): 

- **Creating Foundational Models Discussion**: User `@pratikk10` expressed an interest in connecting with anyone considering creating their own foundational models for various applications, including text-to-text/image/video.
- **High Cost of Foundational Models Highlighted**: In response to `@pratikk10`, user `@_3sphere` pointed out the expensive nature of developing foundational models.
- **Qwen1.5 Model Release and Details**: User `@johnryan465` shared links to Qwen1.5, a 72B parameter chat model, including its introduction, repositories, and demos ([Qwen1.5 Blog Post](http://qwenlm.github.io/blog/qwen1.5/), [Qwen GitHub](https://github.com/QwenLM/Qwen1.5), [Hugging Face Space](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat)).
- **Interpreting Large Language Models**: Discussions spanning various users including `@rami4400`, `@_3sphere`, and `@fern.bear` debated the effectiveness and potential of interpretability in neural networks like LLMs, with comparisons to the human genome project and skepticism about whether interpretability aligns with the nature of intelligence.
- **Concerns About AGI Claims and Model Capabilities**: Skepticism was expressed by `@fern.bear`, `@vara2096`, and `@worthlesshobo` regarding the performance claims of some models, like Qwen 1 and 2 on the MMLU benchmark, and the potential of overfitting or "cheating" on test sets.

**Links mentioned**:

- [Teslas Have a Minor Issue Where the Wheels Fly Off While Driving, Documents Show](https://futurism.com/tesla-flaws-failures-blame-drivers): Despite knowing about chronic &quot;flaws,&quot; Tesla reportedly blamed drivers for glaring defects like collapsed suspensions and breaking axles.
- [Cavemanspongebob React GIF - Cavemanspongebob Caveman Spongebob - Discover &amp; Share GIFs](https://tenor.com/view/cavemanspongebob-caveman-spongebob-react-whatreact-gif-20206670): Click to view the GIF
- [Qwen1.5 72B Chat - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat): no description found
- [Join the Self-Play Language Models Discord Server!](https://discord.gg/aAa7JJ2s): Check out the Self-Play Language Models community on Discord - hang out with 15 other members and enjoy free voice and text chat.
- [Troy Community GIF - Troy Community Room - Discover &amp; Share GIFs](https://tenor.com/view/troy-community-room-fire-pizza-gif-5612111): Click to view the GIF
- [Introducing Qwen1.5](http://qwenlm.github.io/blog/qwen1.5/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...
- [simple ai - chat](https://simple-ai.io/): no description found
- [File:Clock 10-30.svg - Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Clock_10-30.svg): no description found
- [Minimum description length - Wikipedia](https://en.wikipedia.org/wiki/Minimum_description_length): no description found
- [GitHub - idiap/nvib](https://github.com/idiap/nvib): Contribute to idiap/nvib development by creating an account on GitHub.
- [GitHub - SimonKohl/probabilistic_unet: A U-Net combined with a variational auto-encoder that is able to learn conditional distributions over semantic segmentations.](https://github.com/SimonKohl/probabilistic_unet): A U-Net combined with a variational auto-encoder that is able to learn conditional distributions over semantic segmentations. - GitHub - SimonKohl/probabilistic_unet: A U-Net combined with a variat...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1204025456039305246) (61 messages🔥🔥): 

- **Clarification on Multi-task Loss Handling**: `@clashluke` discussed PCGrad's role in improving estimates over manual per-gradient reweighting. They raised concerns about magnitude preservation in gradients using this method, referencing the official code from [facebookresearch/encodec](https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py).

- **Scaling Studies Query**: `@stellaathena` initiated a discussion on whether a post-hoc correlation could convert Kaplan et al.-style scaling laws study to Hoffman et al.-style, potentially reducing the number of training runs needed. They shared a link to the related Twitter conversation and [blog post](https://fixupx.com/BlancheMinerva/status/1754559343058726930) for further exploration.

- **Hypernetworks for LoRA Weights**: Dialogue between `@.rend`, `@thatspysaspy`, and others covered the idea of using hypernetworks to generate LoRA weights for pretrained language models tailored to specific contexts. An [issue on `davisyoshida/lorax`](https://github.com/davisyoshida/lorax/issues/6) detailed a related interest, and `@thatspysaspy` offered a code sample for experimentation.

- **Discussing CNN Training Methodology**: `@jstephencorey` voiced confusion about a video's explanation of CNN training, sparking debate on how layers learn and the effectiveness of pruning. Users like `@Hawk` and `@xylthixlm` discussed the accuracy of the video's claims, although opinions differed on the technicalities.

- **Polyquant Activation Function Debate**: In a conversation about substituting traditional activation functions with alternatives like polynomials, `@clashluke` and others scrutinized a novel architecture's performance on ImageNet without typical activation functions. `@fern.bear` questioned the nonlinearity of the proposed product function, while `@clashluke` highlighted its optimization potential.

**Links mentioned**:

- [Tweet from Grigoris Chrysos (@Grigoris_c)](https://x.com/Grigoris_c/status/1754537124693504320): Proud for our new #ICLR2024 paper attempting to answer: Are activation functions required for all deep networks?  Can networks perform well on ImageNet recognition without activation functions, max po...
- [neural-style-pt/neural_style.py at master · ProGamerGov/neural-style-pt](https://github.com/ProGamerGov/neural-style-pt/blob/master/neural_style.py#L404>): PyTorch implementation of neural style transfer algorithm - ProGamerGov/neural-style-pt
- [encodec/encodec/balancer.py at main · facebookresearch/encodec](https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py): State-of-the-art deep learning based audio codec supporting both mono 24 kHz audio and stereo 48 kHz audio. - facebookresearch/encodec
- [Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based): no description found
- [Tweet from Stella Biderman (@BlancheMinerva)](https://fixupx.com/BlancheMinerva/status/1754559343058726930): @Wetassprior @daphneipp Is there a post-hoc correlation that can be applied to a scaling laws study done Kaplan et al.-style to get one done Hoffman et al.-style? Note that this would be very high val...
- [Predicting LoRA weights · Issue #6 · davisyoshida/lorax](https://github.com/davisyoshida/lorax/issues/6): I would like to use a separate neural network to predict LoRA weights for a main neural network, while training both neural networks at the same time. How can I manipulate the pytrees or to achieve...

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1204026963749244999) (2 messages): 

- **Exploring the Tensor Programs Issue**: `@lucaslingle` explained that `@.johnnysands`' mention of "wrong init" refers to the findings from Tensor Programs 4 and 5 papers, which show that the "standard parameterization" can cause infinite logits when the model width increases indefinitely.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1204020063753998388) (10 messages🔥): 

- **Simple Model Analysis Tool Launched**: `@fblgit` introduced [model-similarities](https://github.com/fblgit/model-similarity), a tool for computing per-layer cosine similarities in parameter space of different models.
- **Comparison with CCA/CKA**: `@xa9ax` inquired about the insights offered by the model-similarity tool compared to Canonical Correlation Analysis (CCA) or Centered Kernel Alignment (CKA), and `@digthatdata` clarified that the tool specifically contrasts the parameter space.
- **Directions in Vector Space Explained**: `@norabelrose` and `@pinconefish` discussed the notion of a "direction," and `@norabelrose` clarified it's a 1D subspace with orientation in vector space rather than just any unit vector.
- **Understanding Vectors Beyond Coordinates**: `@digthatdata` outlined that in deep learning, a vector is understood as both a direction and a magnitude, not just a position in space, and further elaborated on how cosine similarity measures the angle between two vectors.
- **Vectors as Directions and Operators**: `@digthatdata` continued to explain that a vector can function as an "operator," illustrating with the vectors representing `KING`, `QUEEN`, `MAN`, and `WOMAN`, how the difference vector `z` between `WOMAN` and `MAN` acts semantically to alter the meaning from `KING` to `QUEEN`.

**Links mentioned**:

[GitHub - fblgit/model-similarity: Simple Model Similarities Analysis](https://github.com/fblgit/model-similarity): Simple Model Similarities Analysis. Contribute to fblgit/model-similarity development by creating an account on GitHub.

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1203986214538190908) (16 messages🔥): 

- **Statistics Snafu Solved**: `@hailey_schoelkopf` clarified that bootstrapping for standard error on MMLU matches the pooled variance formula, not the combined variance. The [proper formula for pooled variance](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390) is now selected for use in their project over the current combined variance formula.
  
- **Statistical Dragons Ahead**: `@stellaathena` suggested keeping both the old and new statistical formulas in their codebase for expert use, with a humorous warning comment added by `@hailey_schoelkopf`: `# here there be dragons`.

- **Harnessing Language Model Evaluation**: `@jbdel.` prepared a clean fork containing updates for the lm-evaluation-harness in anticipation of a meeting, with all changes available in a [commit on GitHub](https://github.com/jbdel/lm-evaluation-harness-multi/commit/83209a8ac6ecc671cade709dabd05351ef434399?diff=split&w=1). The updated harness allows evaluation of language models with specific arguments and task settings.

**Links mentioned**:

- [Use Pooled rather than Combined Variance for calculating stderr of task groupings by haileyschoelkopf · Pull Request #1390 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390/commits/911c725e1259aec705fbd108ff2664da654aca5d): This PR updates the formula we use for aggregating stderrs / sample std. deviations across groups of tasks. In this PR: formula:  result: hf (pretrained=mistralai/Mistral-7B-v0.1), gen_kwargs: (Non...
- [Use Pooled rather than Combined Variance for calculating stderr of task groupings by haileyschoelkopf · Pull Request #1390 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390#issuecomment-1924217036): This PR updates the formula we use for aggregating stderrs / sample std. deviations across groups of tasks. In this PR: formula:  result: hf (pretrained=mistralai/Mistral-7B-v0.1), gen_kwargs: (Non...
- [GitHub - jbdel/lm-evaluation-harness-multi: A framework for few-shot evaluation of language models.](https://github.com/jbdel/lm-evaluation-harness-multi): A framework for few-shot evaluation of language models. - GitHub - jbdel/lm-evaluation-harness-multi: A framework for few-shot evaluation of language models.
- [VLM · jbdel/lm-evaluation-harness-multi@83209a8](https://github.com/jbdel/lm-evaluation-harness-multi/commit/83209a8ac6ecc671cade709dabd05351ef434399?diff=split&w=1): no description found

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1204129742190542929) (3 messages): 

- **Seeking Pre-tokenized Validation/Test Datasets**: `@pietrolesci` expressed dissatisfaction with the current validation set created using the [pile-uncopyrighted dataset](https://huggingface.co/datasets/monology/pile-uncopyrighted) from Hugging Face. They inquired about the availability of pre-tokenized validation/test splits in a manner akin to the pre-tokenized training set on Hugging Face.

- **Introducing Web Rephrase Augmented Pre-training (WRAP)**: `@elliottdyson` shared an [arxiv paper](https://arxiv.org/abs/2401.16380) proposing **WRAP**, a method to improve large language model pre-training by paraphrasing web data into higher quality formats, which could potentially reduce compute and data requirements. 

- **Comparing WRAP Efficacy to Fine-Tuning**: `@elliottdyson` pondered if using WRAP would be more beneficial compared to just fine-tuning models on the same data and suggested that a comparative study on data processed with and without WRAP followed by fine-tuning could shed some light on its efficacy.

**Links mentioned**:

[Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...

  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1203983170735833138) (321 messages🔥🔥): 

- **AI Training and Career Progression Tips**: User `@dykyi_vladk` shared their learning journey in ML over the past year mentioning specific models and techniques they've studied. `@lunarflu` recommended *building demos and sharing them* as a next step to becoming a professional.

- **Server Troubles Amid A100G Launch**: `@lolskt` reported the server being down for nearly 12 hours and queried if it was related to the A100G launch. After a discussion, `@lunarflu` directed to send details to email and offered to forward the issue to the team.

- **Modifying Model Responses**: User `@tmo97` sought advice on prompting models to stop giving warnings. `@lunarflu` suggested techniques to modify prompts and discussed the balance of safety and user control in AI responses.

- **HuggingFace Fellowship & Model Uploading Queries**: `@not_lain` answered multiple questions about using the HuggingFace platform, including the process to join the fellowship program and details about uploading custom pipeline instances and models.

- **Inference Performance and Hardware Utilization**: User `@prod.nova` inquired why their 4 RTX A5000 GPUs weren't being utilized to their full potential during generation. `@meatfucker` clarified that inference usually utilizes a single GPU and suggested running an instance on each card to distribute the task.


**Links mentioned**:

- [Templates for Chat Models](https://huggingface.co/docs/transformers/main/chat_templating): no description found
- [GitHub - Significant-Gravitas/AutoGPT: AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters.](https://github.com/Significant-Gravitas/AutoGPT): AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters. - GitHub - Significant-Gravitas/AutoGPT: Aut...
- [GitHub - Sanster/tldream: A tiny little diffusion drawing app](https://github.com/Sanster/tldream): A tiny little diffusion drawing app. Contribute to Sanster/tldream development by creating an account on GitHub.
- [Add `push_to_hub( )`  to pipeline  by not-lain · Pull Request #28870 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28870): What does this PR do? this will add a push_to_hub( ) method when working with pipelines. this is a fix for #28857 allowing for easier way to push custom pipelines to huggingface Fixes # (issue) #28...
- [TheBloke (Tom Jobbins)](https://huggingface.co/TheBloke): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1204070331296391198) (1 messages): 

- **Pyannote Praised for Performance**: `@marc.casals.salvador` expressed admiration for **Pyannote**, a tool for speaker diarization, citing its excellent performance.
- **Introduction of Diarizationlm**: `@marc.casals.salvador` brought to attention **Diarizationlm**, a module that simultaneously trains Automatic Speech Recognition (ASR) and diarization to improve annotations by correcting diarizations through the speech itself.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1204079391077040128) (5 messages): 

- **Empty Model Cards on HuggingFace**: `@tonic_1` pointed out that [model cards are empty](http://103.170.5.190:7864/) on HuggingFace, which can be a hindrance for users seeking model information.
  
- **Interactive Gradio Demo Lacks Visibility**: `@tonic_1` shared excitement about a [cool gradio demo](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-long-video) but expressed concerns over the lack of details provided for this potentially overlooked model.

- **Innovative LLaMA-VID Model Introduced**: A model card for **LLaMA-VID** was shared by `@tonic_1`, detailing an open-source chatbot that supports hour-long videos by using an extra context token. The model and more information can be found [here](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-long-video).

- **DV Lab's Under-the-Radar Model Hosted on HuggingFace**: `@tonic_1` mentioned discovering another compelling but underappreciated model by DV Lab on HuggingFace, expressing hope to be able to serve it.

- **Exploring Cost-Efficient LLM Usage**: `@jessjess84` shared an [arXiv paper](https://arxiv.org/abs/2312.08361) describing research on cost-efficient strategies for inference and fine-tuning of large language models (LLMs), including distributed solutions leveraging consumer-grade networks.

**Links mentioned**:

- [Distributed Inference and Fine-tuning of Large Language Models Over The Internet](https://arxiv.org/abs/2312.08361): Large language models (LLMs) are useful in many NLP tasks and become more capable with size, with the best open-source models having over 50 billion parameters. However, using these 50B+ models requir...
- [YanweiLi/llama-vid-7b-full-224-long-video · Hugging Face](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-long-video): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1204003484526059582) (3 messages): 

- **Ankush's Finetuning Feat**: User `@andysingal` announced a finetuned [model](https://huggingface.co/Andyrasika/mistral-ft-optimized-dpo) developed by Ankush Singal. The model is an optimization based on a previous model from [OpenPipe](https://huggingface.co/OpenPipe/mistral-ft-optimized-1227) and comes with an installation and usage guide.

- **Community Applause for New Finetuned Model**: `@osanseviero` congratulated `@andysingal` for the creation of the new finetuned model, hailing it as **very cool** and *fiery* with a 🔥 emoji.

**Links mentioned**:

[Andyrasika/mistral-ft-optimized-dpo · Hugging Face](https://huggingface.co/Andyrasika/mistral-ft-optimized-dpo): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1204147217225617418) (3 messages): 

- **Discover a Memetastic AI Milestone**: User `typoilu` shared an article titled "A Meme's Glimpse into the Pinnacle of Artificial Intelligence (AI) Progress in a Mamba Series LLM Enlightenment" claiming it contains insightful information about AI progress. There was no further discussion on the content. [Read Here](https://www.marktechpost.com/2024/02/03/a-memes-glimpse-into-the-pinnacle-of-artificial-intelligence-ai-progress-in-a-mamba-series-llm-enlightenment/)
  
- **Scheduled Talk May Need Rescheduling**: `@ericauld` might have jury duty on Friday, suggesting postponing the planned talk to Friday the 16th. An adjustment in the event calendar may be needed.

- **Chad Brings Support with a Dash of Luck**: In response to the potential rescheduling, `@chad_in_the_house` showed understanding and wished `@ericauld` good luck with jury duty if it happens.

**Links mentioned**:

[no title found](https://www.marktechpost.com/2024/02/03/a-memes-glimpse-into-the-pinnacle-of-artificial-intelligence-ai-progress-in-a-mamba-series-llm-enlightenment/): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1204110029431635968) (1 messages): 

- **Invitation to Share Computer Vision Expertise**: User `@danielsamuel131` has made an open call for those with expertise in **computer vision** to come forward and share their knowledge. Interested individuals are encouraged to drop him a direct message.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1204061285134630963) (11 messages🔥): 

- **Fine-Tuning Frenzy**: `@joeyzero` is looking to **fine-tune** a chatbot with noisy old data and seeks resources on conversational data sets and tools for managing data sets easily. The user is open to suggestions including datasets ready for conversational use, robust editing tools, and any tutorials or tips for such NLP projects.
  
- **Papillon's NLP Tool Shared**: `@8i8__papillon__8i8d1tyr` shared a tool for **Named Entity Recognition** (NER) and **Sentiment Analysis** based on Flair and FLERT with a [link to the GitHub repository](https://github.com/CodeAKrome/bootcupboard/blob/main/flair/SentimentalNERD.py).

- **YAML Fine-Tuning Challenges**: `@denisjannot` experienced issues with fine-tuning **Mistral 7b** for YAML generation, where unintended parts of YAML are modified upon a second request to alter specific parts.

- **Trajectory for YAML Fine-Tuning**: `@denisjannot` mentions plans to train the **Instruqt model** to see if it improves the YAML modification issue. There's a request for ideas to enhance the fine-tuning process without including modification examples in the training dataset.

- **Few-Shot Learning Suggestion**: `@meatfucker` advised including examples of the desired modification in the prompt to induce **one-shot or few-shot learning**, which should work well even if the examples aren't in the training dataset. This is intended to guide the model for YAML modifications.

**Links mentioned**:

[bootcupboard/flair/SentimentalNERD.py at main · CodeAKrome/bootcupboard](https://github.com/CodeAKrome/bootcupboard/blob/main/flair/SentimentalNERD.py): It&#39;s bigger on the inside than the outside! Contribute to CodeAKrome/bootcupboard development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1203982648364630047) (103 messages🔥🔥): 

- **Confusion in Choosing the Right Model**: User `@hades____` expressed feeling overwhelmed by the hundreds of available models, looking for guidance on how to select the best one for their needs. Suggestions pointed to resources and the idea of focusing on specific use-cases rather than seeking a "one size fits all" model.

- **Model Compatibility Queries**: Various users, including `@dicerx` and `@foxwear`, inquired about connecting LM Studio with specific applications and models. The conversation involved potential compatibility with iOS apps and multimodal models, seeking clarity on integration possibilities.

- **Resource and Template Requests**: Users like `@Jonatan`, `@funapple`, and `@ayelwen` requested resources for code examples, prompt templates, and explanations of model differences, highlighting a community need for easily accessible and straightforward documentation.

- **Technical Assistance Sought for LM Studio**: Participants `@plaraje`, `@perkelson`, `@ts9718`, and `@joelthebuilder` asked for technical help with issues ranging from prompt generation quirks to software functionalities in LM Studio, such as changes in zoom behavior and server connection guidance.

- **Jokes and Light-Hearted Comments Flow**: Amidst technical discussions, users like `@sica.rios`, `@rugg0064`, and `@wildcat_aurora` infused humor into the conversation, joking about AI's capabilities and making light of language misunderstandings while interacting with the AI models.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1138544400771846174/1201187492414619791): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...
- [liuhaotian/llava-v1.5-7b · gguf variant availability](https://huggingface.co/liuhaotian/llava-v1.5-7b/discussions/4): no description found
- [GitHub - FriendofAI/LM_Chat_TTS_FrontEnd.html: LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, making it suitable for a wide range of users interested in exploring voice interactions with AI models.](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html): LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, makin...
- [GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型](https://github.com/thudm/cogvlm): a state-of-the-art-level open visual language model | 多模态预训练模型 - GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/164dbip/how_to_run_or_convert_pytorch_model_with_llamacpp/): no description found

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1203994893551271987) (3 messages): 

- **Brief Mention of Channel Reference**: `@egalitaristen` mentioned a channel with the code `<#1167546635098804284>`, but provided no context or further details.
- **Inquiry About a Model Usage**: `@delfi_r_88002` inquired if anyone is using the model **llava-v1.6-34b.Q4_K_M.gguf**, but did not offer further information or context.
- **New Model DeepSeek-Math-7B Released**: `@czkoko` announced the release of **DeepSeek's** new model, **DeepSeek-Math-7B**. The model can be found on [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) with the corresponding metadata provided from the page.

**Links mentioned**:

[GitHub - deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math): Contribute to deepseek-ai/DeepSeek-Math development by creating an account on GitHub.

  

---


### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1204117720065515541) (1 messages): 

- **LM Studio v0.2.13 Drops with Qwen 1.5**: LM Studio announced `@yagilb` the release of **LM Studio v0.2.13**, featuring support for **Qwen 1.5** across a range of model sizes (0.5B, 1.8B, 4B, 7B, 72B). Users can download the new version directly from [https://lmstudio.ai](https://lmstudio.ai) or update through the app.

- **Pin Your Favorites in LM Studio**: The new LM Studio update allows users to **pin models and chats** to the top of their lists, making favorite tools more accessible.

- **Qwen's New Models Now Open Source**: Qwen1.5 models have been released and open-sourced, with sizes ranging from 0.5B to 72B, including base, chat, AWQ, GPTQ, GGUF models available on various platforms including [Hugging Face](https://huggingface.co/Qwen) and [LM Studio](https://lmstudio.ai).

- **Integration and Accessibility Enhancements**: Qwen1.5 features quality improvements and integrates into Hugging Face transformers, removing the need for `trust_remote_code`. With APIs offered on DashScope and Together, the recommended model to try is Qwen1.5-72B-chat on [https://api.together.xyz/](https://api.together.xyz/).

- **Performance Tuning in LM Studio App**: The latest **LM Studio release eliminates subprocesses** used to measure CPU and RAM on Windows, leading to improvements in the app's performance.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs
- [Tweet from LM Studio (@LMStudioAI)](https://x.com/LMStudioAI/status/1754547632972738978?s=20): LM Studio v0.2.13 is available now!  What&#39;s New:  - 🚀 Support for Qwen 1.5!  (0.5B, 1.8B, 4B, 7B, 72B)  And: - 🤖📌 Pin models to the top of the list - 💬📌 Pin chats, too  Download from https://...
- [Qwen1.5 GGUF - a lmstudio-ai Collection](https://huggingface.co/collections/lmstudio-ai/qwen15-gguf-65c110cf444ff44cb6dd5ec4): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1204102960888418314) (9 messages🔥): 

- **Shoutout for LM Studio's Simplicity**: `@drawless111` shared a link to a [Medium blog post](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed) praising **LM Studio** for allowing anyone to experience the power of large language models with a simple user interface and no technical expertise needed.
- **Feature Request Acknowledgment**: `@heyitsyorkie` responded to `@foobar8553`'s call for a resume feature by pointing to a [current feature request](https://discord.com/channels/1110598183144399058/1193271374375043133) and asked to show support for it.
- **Appreciation for LM Studio App**: `@gli7ch.com` expressed gratitude towards LM Studio app developers for making work with LLMs easier, especially in terms of integrating them with automation tasks.
- **Inquiry About Investment Opportunities**: `@ahakobyan.` asked about investment opportunities which led to a witty exchange between `@fabguy` and `@ptable`, both humorously claiming they'd accept money for their fabricated and competing "foundations."
- **Questioning Odd Zoom Shortcuts**: `@perkelson` questioned the logic behind LM Studio using non-standard zoom shortcuts, differing from those commonly used in browsers.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1193271374375043133): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Elio And Lea GIF - Elio and Lea - Discover &amp; Share GIFs](https://tenor.com/view/elio-and-lea-gif-16841311517970792125): Click to view the GIF
- [LM Studio: experience the magic of LLMs with Zero technical expertise](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed): Your guide to Zero configuration Local LLMs on any computer.

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1204072523969470544) (40 messages🔥): 

- **Troubleshooting LMStudio on Linux**: `@heyitsyorkie` advised `@aswarp` to use the latest Ubuntu 22 to avoid glibc errors when running LMStudio, adding that the Linux build still has compatibility issues with older Ubuntu versions.
- **GPU Utilization Mastery for Chatbots**: `@heyitsyorkie` reassured `@shylor` that near-full utilization of GPU memory without spilling into shared RAM is good, as too much use of shared RAM can degrade performance over time with LLM chatbot interactions.
- **Speeding Up Token Generation**: `@roscopeko` discussed ways to reduce the time to the first token, mentioning a 6-second delay with a powerful setup; members like `@aswarp` and `@alastair9776` suggested different approaches including picking another model, quantization, or trying out the AVX beta.
- **Shadow PC Machinery Under the Microscope**: `@goldensun3ds` shared a [YouTube video](https://youtu.be/Eaz-H-3FkZg) detailing a comparison of running LLMs on a Shadow PC compared to a powerful local PC setup, noting the Shadow PC's slower performances in loading models.
- **Grappling with Hardware Resources & Model Configurations**: `@robert.bou.infinite` detailed available GPU configurations for running LMStudio across various Nvidia offerings in data centers, outlining the implications of NVLINK technology and Kubernetes Pods compatibility, suggesting that multiple GPU setups can be leveraged effectively for LLM workloads.

**Links mentioned**:

- [Nvidia LHR explained: What is a ‘Lite Hash Rate’ GPU?](https://www.pcworld.com/article/395041/nvidia-lhr-explained-what-is-a-lite-hash-rate-gpu.html): Nvidia&#039;s Lite Hash Rate technology is designed to foil Ethereum miners and get more GeForce graphics cards in the hands of gamers. Here&#039;s what you need to know.
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [Testing Shadow PC Pro (Cloud PC) with LM Studio LLMs (AI Chatbot) and comparing to my RTX 4060 Ti PC](https://youtu.be/Eaz-H-3FkZg): I have been using Chat GPT since it launched about a year ago and I&#39;ve become skilled with prompting, but I&#39;m still very new with running LLMs &quot;locally&quot;. Whe...

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1204092411198967828) (14 messages🔥): 

- **New Beta Update Rolls Out**: `@yagilb` announces **LM Studio version 0.2.13 Preview - Build V3**, featuring the ability to pin models and chats and performance improvements. The update is available for download with Mac and Windows links provided, and feedback is encouraged through a specified Discord channel. [Download here](https://discord.com/channels/1110598183144399058/1204092056897716304/1204092056897716304).

- **Leaderboard for Best LLMs Request**: `@kyucilow` requests a leaderboard tab for the best LLMs to make selection easier, `@minorello` suggests referencing that in a certain channel, while `@heyitsyorkie` and `@re__x` provide links to external resources featuring LLM rankings. [Hugging Face LLM Leaderboard](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard) and [OpenRouter Rankings](https://openrouter.ai/rankings).

- **GUI for Server Access Desired**: `@_jayross` expresses a wish for a web server GUI for remote access of the LM Studio server component; `@goldensun3ds` offers a workaround using Parsec for remote access and inquires about Intel ARC GPU support in LM Studio.

- **Chat Interface Issues and Error Reporting**: `@wolfspyre` encounters a potential issue where, after ejecting a model in LMS' chat interface and modifying settings, the chat system seems to stop responding. The problem is being investigated for bugs or unexpected behavior.

- **Dependency Installation Tips Shared**: `@greg0403` suggests installing the **blast library** to address a potential issue, recommending the use of `sudo apt-get install ncbi-blast+`.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1110598183144399061/1202679898024452096): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204092056897716304/1204092056897716304): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Yet Another LLM Leaderboard - a Hugging Face Space by mlabonne](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard): no description found
- [OpenRouter](https://openrouter.ai/rankings): Language models ranked and analyzed by usage across apps

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (1 messages): 

lowkey9920: Try autogen studio . It's two commands to get started with a ui
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1204134089851142175) (1 messages): 

- **Inquiry into Methods for Question Creation**: `@varelaseb` expressed interest in improving a **RAG system** and asked for advice on methods for question creation over a dataset. Specifically, they mentioned a lack of information on **Tuna** despite hearing good things about it.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1203985817287004230) (118 messages🔥🔥): 

- **LangChain Lamentations**: Discord users `@akshay_1` and `@mrdragonfox` express dissatisfaction with LangChain, with the former suggesting its utility may only last a week before components need solidifying.
- **Search for Old Mac Compatible Interpreters**: `@zhiyyang` seeks a model interpreter for Mac OSX versions under 11, with suggestions ensuing but no specific solutions provided.
- **Mistral 8x7B Determinism Discussed**: `@zaragatungabumbagumba_59827` queries about the deterministic nature of Mixtral 8x7B during inference, and `@mrdragonfox` explains that the behavior is probabilistic, influenced by an adjustable temperature parameter.
- **Philosophical Perspectives on AI**: A philosophy student `@zaragatungabumbagumba_59827` inquires into the philosophical implications of LLMs, receiving recommendations from `@mrdragonfox` and others to explore fundamental AI concepts.
- **Synthetic Data Generation Secrets Stay Secret**: In a discussion about dataset generation, `@mrdragonfox` mentions the utility of GitHub repositories like [airoboros](https://github.com/jondurbin/airoboros) and [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit), but declines to share specific methods, highlighting synthetic data generation as a crucial income source.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1144547040454508606/1204003259367424020): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets
- [GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros): Customizable implementation of the self-instruct paper. - GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1204004695002325023) (6 messages): 

- **Prompt Enhancement Queries**: `@drprimeg1` inquired about the additional information `@gbourdin` includes in prompts to improve accuracy.
- **PHP to JSON Schema for Better Prompts**: `@gbourdin` described their method of converting a PHP class to a JSON schema to refine prompts and offered to share the code privately.
- **Code Sharing Offer Accepted**: `@drprimeg1` expressed interest in seeing `@gbourdin`'s code structure for their prompts, prompting a private message with the details from `@gbourdin`.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1204127319460028496) (1 messages): 

- **Padding Dilemmas in Fine-Tuning**: User `@ramin2024` is seeking advice on how to properly set padding times for fine-tuning, mentioning that the common practice of setting `tokenizer.pad = tokenizer.eos` sometimes results in the fine-tuned model not generating an end-of-sentence (eos) token. They shared that they and others have encountered this issue.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1204282900514869290) (1 messages): 

- **Discord Chatbot Hits Star Milestone**: User `@jakobdylanc` announced their Discord chatbot now has over 69 stars on GitHub. The bot features support for multiple LLMs such as **Mistral** and offers multi-user chat, vision support, streamed responses, all in just 200 lines of code [check out the repository](https://github.com/jakobdylanc/discord-llm-chatbot).

**Links mentioned**:

[GitHub - jakobdylanc/discord-llm-chatbot: Supports OpenAI, Mistral, ollama, oobagooba and more • Multi-user chat • Vision support • Streamed responses • 200 lines of code 🔥](https://github.com/jakobdylanc/discord-llm-chatbot): Supports OpenAI, Mistral, ollama, oobagooba and more • Multi-user chat • Vision support • Streamed responses • 200 lines of code 🔥 - GitHub - jakobdylanc/discord-llm-chatbot: Supports OpenAI, Mistr.....

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1204057386491584512) (5 messages): 

- **Expression through Flags**: `@jujuderp` posted an emoji of the **North Korean flag** 🇰🇵, while `@matmatgamer` shared the emoji of the **French flag** 🇫🇷.
- **Shoutout to a Fire Playlist**: `@jakobdylanc` gave a shoutout to `<@421429282804465666>` for playing some great tunes, describing it as *bumping fire*.
- **The Need for Speed**: `@bam4d` responded with a sense of urgency, replying with *gotta go fast*.
- **Fast and Furry-ous**: `@bam4d` shared a humorous GIF from [Tenor](https://tenor.com/view/sanic-the-hedgehob-running-gotta-go-fast-fast-gif-4964355) featuring a quick, parody version of Sonic the Hedgehog known as Sanic, along with a note about language settings on Tenor's site.

**Links mentioned**:

[Sanic The Hedgehob GIF - Sanic The Hedgehob Running - Discover &amp; Share GIFs](https://tenor.com/view/sanic-the-hedgehob-running-gotta-go-fast-fast-gif-4964355): Click to view the GIF

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1204003610556502016) (2 messages): 

- **Emoji Terminator Causes Unexpected Behavior**: `@jakobdylanc` identified a peculiar issue where **Mistral models** terminate with finish_reason as "stop" but still include an emoji in the **content** of the response. They provided a detailed **snippet of the chat response**, highlighting this anomaly across all three Mistral API models.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1203996622577606666) (29 messages🔥): 

- **Enthusiasm for Dual GPU Builds**: `@morgangiraud` is excited about assembling a new dual build with 2 x 4070 TI SUPER to gain experience with dual GPU training. They also consider the trade-off between more VRAM with older models like the 3090 and the desire for brand new cards.
  
- **Choosing the Right Components**: Community members, including `@__boatbuilder__` and `@morgangiraud`, are discussing the advantages of using [PCPartPicker](https://pcpartpicker.com/) and resources like the /r/selfhosted and r/buildapc subreddits to assist in building their setups.
  
- **European vs US Pricing**: `@morgangiraud` shares that there weren't significant deals available for their PC components, attributing it to higher prices in Europe, and reveals a total cost of 4k for the setup. A complete parts list was provided on [PCPartPicker](https://pcpartpicker.com/user/morgangiraud/saved/VTZRFT).
  
- **Multi-GPU Setup Considerations**: `@jeremyhoward` and `_tvi_` weigh in on the multi-GPU conversation, discussing that a good motherboard can overcome the lack of P2P communication in consumer cards and the strategic advantage of upgrading with a second big card later.
  
- **Accelerating LLM Serving**: `@andreaskoepf` highlights [FlashInfer](https://github.com/flashinfer-ai/flashinfer), an open-source library designed to improve the performance of Large Language Model serving, which targets optimizing Self-Attention and other transformative operations critical to LLMs.

**Links mentioned**:

- [C-Payne PCB Design](https://c-payne.com/): C-Payne PCB Design
- [Accelerating Self-Attentions for LLM Serving with FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html): Introduce Techniques to accelerate Large Language Model Deployment
- [GitHub - flashinfer-ai/flashinfer: FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer): FlashInfer: Kernel Library for LLM Serving. Contribute to flashinfer-ai/flashinfer development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1204047348393578507) (27 messages🔥): 

- **Floating Point Precision Troubles**: `@zippika` reports inaccuracies when they dequantize and then apply an `nn.Linear` operation in PyTorch, finding that results differ significantly from expected fp16/fp32 calculations. The dequantize function reportedly works well, but subsequent operations lead to errors, and this is suspected to be connected to rounding differences between fp32 and fp64.
  
- **CUDA Synchronization or C++ Flag Issues?**: In their debugging process, `@zippika` contemplates whether CUDA synchronization issues or the disabling of standard PyTorch C++ flags might be causing unexpected behavior when performing linear operations after dequantization. The mentioned disabled flags include `"__CUDA_NO_HALF_OPERATORS__"`, `"__CUDA_NO_HALF_CONVERSIONS__"`, `"__CUDA_NO_HALF2_OPERATORS__"`, and `"__CUDA_NO_BFLOAT16_CONVERSIONS__"`.

- **Quantization Error Monologue**: `@zippika` expresses frustration over their implementation details in a CUDA `.cu` file, trying to understand why the function `dequantizeBlockwise_impl_stream` results in significant numerical discrepancies when used as a precursor to matrix multiplication operations. They suggest that perhaps transposing the weight matrix or a mistaken switch of the `K/N` dimension parameters might be contributing to the issue.
  
- **Dequantize Function Under Scrutiny**: Debugging efforts by `@zippika` focus on the `qlinear_impl` function, where the `dequantizeBlockwise_impl_stream` function is used. `@zippika` notes that the dequantize function appears to function perfectly until utilized within their custom linear operation, after which the results diverge sharply from expectations.
  
- **Stability Across Functions Questioned**: `@zippika` details differences between two PyTorch functions, `mm_normal` and `mm_qlinear`, with the former producing stable and expected results, while the latter is faster but appears unstable and is suspected to contain synchronization issues or other errors. The user shares snippets of the contrasting functions in search of feedback on the discrepancies.

**Links mentioned**:

[Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors](https://arxiv.org/abs/2206.02874): Tensor Cores have been an important unit to accelerate Fused Matrix Multiplication Accumulation (MMA) in all NVIDIA GPUs since Volta Architecture. To program Tensor Cores, users have to use either leg...

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1204074385254912010) (18 messages🔥): 

- **Calling All AI Innovators**: `@vim410` shared an exciting opportunity for developers to participate in NVIDIA's Generative AI [RTX Developer Contest](https://www.nvidia.com/en-us/ai-data-science/generative-ai/rtx-developer-contest/) to win a GeForce RTX 4090 GPU and other prizes. The contest encourages building innovative AI projects using NVIDIA's technologies.
  
- **Optimizing GPTQ with Toro and Triton**: `@jku100` discussed enhancing `gptq` performance using `torch.compile` and domain knowledge, leading to a kernel that matches customized CUDA performance. They also provided a [benchmarking discussion](https://github.com/AutoGPTQ/AutoGPTQ/pull/530) and expressed that their work shows the potential of combining Torch and Triton tools.

- **Invitation to Share Innovation**: `@marksaroufim` responded positively to `@jku100`'s achievements, offering an opportunity to give a talk about their work, highlighting the significant potential of `torch.compile` for AI optimization.

- **Potential Talk Scheduled for March 9th**: `@jku100` tentatively agreed to present their work on March 9th, upon confirmation, to discuss further optimizations in `torch.compile` and the implications on AI acceleration techniques.

- **Skepticism on Contest Incentives**: `@naisdi` commented critically on NVIDIA's contest, implying the prize to cost ratio may not be worthwhile for participants, and comparing it unfavorably with NVIDIA's marketing practices involving influencers.

**Links mentioned**:

- [NVIDIA Gen AI on RTX PCs Developer Contest](https://www.nvidia.com/en-us/ai-data-science/generative-ai/rtx-developer-contest/): Enter to win a GeForce RTX 4090 GPU, a GTC event pass, and more.
- [AutoGPTQ/auto_gptq/nn_modules/triton_utils/kernels.py at be78af8d4fd80b5afa0a8a7df7b0e0ec44420003 · AutoGPTQ/AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ/blob/be78af8d4fd80b5afa0a8a7df7b0e0ec44420003/auto_gptq/nn_modules/triton_utils/kernels.py#L15)).): An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm. - AutoGPTQ/AutoGPTQ

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1204291815499370566) (3 messages): 

- **Software Engineer Seeks MLOps Wisdom**: `@einstein5744`, a software engineer from an ML company, is looking to enhance skills in MLOps and is currently experimenting with **fast.ai** and **Diffusers library**. They are open to suggestions and advice on delving deeper into machine learning operations.

- **Is it the fast.ai Diffusion Course?**: `@jeremyhoward` inquired if `@einstein5744` is engaged with the fast.ai **diffusion course** to gain a deeper understanding of machine learning techniques.

- **Course Gratitude from a Fast Learner**: `@joseph_en` expressed gratitude for the fast.ai course and specifically mentioned finishing a review of the **DiffEdit paper** in chapter 11, appreciating the availability of such resources.
  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1204216449527316560) (2 messages): 

- **Lecture 5 Event Scheduled**: `@jeremyhoward` announced the addition of the event for lecture 5, which will be occurring this weekend, with a [Discord event link](https://discord.gg/pBhQAAvB?event=1204175111633113168) provided for members to join.

- **Inquiry About Swizzled Order**: `@lancerts` referenced the [PyTorch blog on accelerating Triton](https://pytorch.org/blog/accelerating-triton/) and inquired whether the concept of *swizzled order* was covered in the PMPP book, suggesting they are yet to encounter it but may not have reached that part of the book.
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1204051941794127882) (4 messages): 

- **Lecture 4 Recording Inquiry**: `@zonepg` inquired about the availability of a recording for Lecture 4, as they were unable to attend the live stream due to timezone constraints.
- **Typical Upload Schedule Shared**: `@morgangiraud` informed that recordings are usually uploaded at the beginning of the week.
- **Technical Issues Delay Lecture 4 Recording**: `@marksaroufim` mentioned that there were technical issues which delayed the recording, but assured that the video would be posted that day.
- **Lecture 4 Recording Now Live**: `@marksaroufim` announced the upload of the [Lecture 4 recording](https://www.youtube.com/watch?v=lTmYrKwjSOU) and added that HD quality should be available in about an hour.
  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1204034634061713448) (5 messages): 

- **Exploring Pallas Extension for JAX**: `@stefangliga` shared a [Pallas Quickstart Guide](https://jax.readthedocs.io/en/latest/pallas/quickstart.html), highlighting that **Pallas** is an experimental extension which simplifies writing custom kernels for GPU and TPU within JAX. Pallas operates at a lower level of abstraction requiring consideration of memory access and computations across hardware accelerators, translating to Triton on GPUs and to Mosaic on TPUs.
  
- **User Contemplation**: `@stefangliga` posted a thinking face emoji (🤔), which may indicate a moment of reflection or a need for further clarification about the discussed material.

- **Using Pure CUDA Kernels in JAX**: `@nshepperd` provided a link to a resource explaining how to use pure CUDA kernels in JAX, offering additional flexibility when working with custom operations ([Custom Operations for GPUs](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)).

- **Inquiry About Triton Kernels and JAX Execution Details**: `@marvelousmit` asked if there's a method to print out Triton kernels and the actual code JAX executes, expressing an interest in understanding the underlying calls when profiling, especially related to sgemm kernel launches.

- **Seeking JAX Reverse Engineering Tools**: Further, `@marvelousmit` mentioned being accustomed to reverse engineering kernel code using Triton+Torch with inductor and inquired about the equivalent process for JAX.

**Links mentioned**:

[Pallas Quickstart &#8212; JAX  documentation](https://jax.readthedocs.io/en/latest/pallas/quickstart.html): no description found

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1204009917661577246) (31 messages🔥): 

- **Troubles with ChatGPT File Uploads**: Users `@sherlyleta`, `@guti_310`, and `@lugui` reported that the ChatGPT file upload feature has been malfunctioning since the end of last week, with `@lugui` assuring that it will be resolved soon.
- **Debating Manufacturer Responsibility**: `@aipythonista` countered `@johnnyslanteyes`'s suggestion to view Louis Rossman's content for a list of issues, arguing that firmware updates should address these concerns and that claims may be biased as the brand in question is considered to be of higher quality than competitors.
- **Localizing GPT-3.5 Discussion**: `@czarcast` inquired about hosting a local instance of GPT-3.5 for homelab diagnostics, `@elektronisade` noted that GPT-3.5 is not available for such use, prompting `@riaty` to recommend the open-source alternative, Mistral 8x7b.
- **Debating Model Efficiency and Smarts**: `@kotykd` and `@riaty` discussed the viability of running the open-source model Mistral 8x7b locally, acknowledging its high resource requirements but arguing its potential utility depending on the user's needs.
- **Consideration of Novel 3D Language Model**: `@red_code` proposed a concept of a 3D vector language model to represent words and characters, sparking a brief reaction from `@darthgustav.` questioning the feasibility in terms of performance.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1204034836692598784) (50 messages🔥): 

- **Trademark Troubles for Custom GPT**: `@zurbinjo` encountered issues using the term "Midjourney" in a GPT name, and `@solbus` clarified that trademark rights prevent this usage according to OpenAI's [branding guidelines](https://openai.com/brand#gpts-in-chatgpt).
- **Persistent Technical Turbulence**: Numerous users including `@Aleks`, `@sherlyleta`, `@thatjay_`, and `@realspacekangaroo` experienced persistent issues with file uploads and ChatGPT functionalities across various browsers, with some success after switching browsers.
- **Users Grapple with GPT-4 Glitches**: `@dexmeighan` and `_odaenathus` reported that GPT-4 is getting stuck during conversations, affecting usage across different web browsers, with the situation gradually improving for some.
- **Customization Confusion for GPT Builders**: Users expressed difficulties with custom GPT features, such as setting a voice (`@diphon`), enforcing instruction sequence (`@woodenrobot`), and changing the identity displayed (`@scootmandu_ai`).
- **Seeking Feedback and Exposure for Custom GPTs**: `@_loier` looked for ways to test and obtain feedback for a custom GPT designed for role-playing games, leading to discussions about the best methods and platforms to accomplish this.

**Links mentioned**:

[Brand guidelines](https://openai.com/brand#gpts-in-chatgpt>): Language and assets for using the OpenAI brand in your marketing and communications.

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1204086867734302761) (2 messages): 

- **PDF vs Extracted Text for AI**: `@wazzldorr` asked for advice on the best method to provide a scientific article to the AI, pondering between a **PDF format** or extracting the text and inputting it **directly into the chat**. In response, `@lugui` assured that the AI **_should_ be able to handle the PDF without any issues**.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1204086867734302761) (2 messages): 

- **PDF vs. Extracted Text Question**: User `@wazzldorr` asked about the best way to address a scientific article, questioning whether to provide the whole document in PDF format or to extract the text and provide it in the chat directly. 
- **PDF Handling Capability**: In response, `@lugui` assured that the AI **should be able to handle the PDF** with no problems.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1203989126530207794) (28 messages🔥): 

- **Cloud GPU Nodes at Competitive Prices**: User `@lhc1921` mentioned that **RunPod** provides good offering on GPU nodes in the cloud, suggesting it may be a resource to consider for those requiring cloud computing services.

- **Introducing BGE-M3 Multi-Functional Embedding Model**: `@lhc1921` shared a link to a new embedding model known as **BGE-M3** that excels in multi-functionality, multi-linguality, and multi-granularity. For more details, they provided the [GitHub repository](https://github.com/FlagOpen/FlagEmbedding) and the accompanying [paper](https://arxiv.org/pdf/2402.03216.pdf).

- **How to Use OpenAIEmbedder for Embeddings**: In response to `@natuto_uzumaki1808`'s inquiry, `@kapa.ai` provided detailed instructions on generating embeddings after declaring embedder, including code examples in both JavaScript and Python, and referred users to [LangChain's JavaScript documentation](https://js.langchain.com/docs/integrations/text_embedding/openai) and [Python documentation](https://python.langchain.com/docs/integrations/text_embedding/openai).

- **Comparing Data Preprocessing in Llama Index and LangChain**: User `@arrmlet` sought insights on the differences in data preprocessing between **llama-index** and **LangChain** and shared a [Stack Overflow question](https://stackoverflow.com/questions/77941814/how-do-llamaindex-and-langchain-differ-in-terms-of-data-preprocessing-for-llm-ap) for community response.

- **Translating HTML with LangChain While Preserving Styles**: `@o3omoomin` raised a query about translating HTML into another language using LangChain, emphasizing the importance of maintaining the original style and formatting.

**Links mentioned**:

- [BAAI/bge-m3 · Hugging Face](https://huggingface.co/BAAI/bge-m3): no description found
- [How do LlamaIndex and LangChain Differ in Terms of Data Preprocessing for LLM Applications?](https://stackoverflow.com/questions/77941814/how-do-llamaindex-and-langchain-differ-in-terms-of-data-preprocessing-for-llm-ap): I&#x27;ve been exploring frameworks to integrate large language models (LLMs) into my applications, specifically focusing on data preprocessing, ingestion, and query capabilities. I&#x27;ve come acros...
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/1560>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1204112546911756298) (33 messages🔥): 

- **Bearer Token Troubles**: `@peterlandis` inquired about how to pass in request headers using a bearer token when working with AzureGPT. In response, `@veryboldbagel` provided guidance referring to examples from the [Configurable Runnables documentation](https://github.com/langchain-ai/langserve/blob/main/examples/configurable_chain/server.py) and the [APIHandler examples](https://github.com/langchain-ai/langserve/blob/main/examples/api_handler_examples/server.py) for full endpoint customization.
  
- **LangServe Learning Curve**: A conversation between `@lucas_89226`, a self-described jaded former enthusiast, and `@veryboldbagel` focused on issues with setting up LangServe. `@lucas_89226` faced errors when trying to use the sample code and the playground, which led to troubleshooting that included checking the client code, considering whether the LangServe server code had been altered, and verifying the correct OpenAI API endpoint configuration.

- **LangServe Setup Guide Announcement**: `@gitmaxd` shared their experience with Hosted LangServe and offered a [helpful guide](https://medium.com/@gitmaxd/your-first-a-i-api-endpoint-with-langserve-deeb65e750b1) complete with a deployment video and walkthrough for setting up a simple LangServe template.

- **Experiencing Errors with LangServe**: `@lucas_89226` reported encountering an error message while attempting to invoke a LangServe endpoint. In contrast, `@veryboldbagel` requested the full traceback and suggested that `@lucas_89226` confirm if the error persists with the original unmodified server code.

- **Support Offer for Troubleshooting LangServe**: As `@lucas_89226` continued to troubleshoot their LangServe setup, `@veryboldbagel` remained engaged, offering additional support and advising `@lucas_89226` to open an issue or start a discussion on the [LangServe GitHub discussions page](https://github.com/langchain-ai/langserve/discussions) if needed.

**Links mentioned**:

- [no title found](https://...```): no description found
- [Your first A.I. API endpoint with 🦜LangServe](https://medium.com/@gitmaxd/your-first-a-i-api-endpoint-with-langserve-deeb65e750b1): Deploying your first A.I. Rest API Endpoint with LangServe is EASY! We’ll walk through everything to get your first project online.
- [langchain-ai/langserve · Discussions](https://github.com/langchain-ai/langserve/discussions): Explore the GitHub Discussions forum for langchain-ai langserve. Discuss code, ask questions &amp; collaborate with the developer community.
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/configurable_chain/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/configurable_chain/server.py): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/api_handler_examples/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/api_handler_examples/server.py): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1204009968337166388) (6 messages): 

- **AI Form Roast Promises Form Optimization**: `@siddish` showcased **AI Form Roast by WorkHack**, an AI tool aimed at analyzing online forms and providing feedback to enhance completion rates and user experience. They're inviting feedback and support on [Product Hunt](https://www.producthunt.com/posts/ai-form-roast-by-workhack).

- **OranScribe TweetStorm Express Flow Launch**: `@shving90` shared a tweet about OranScribe's new feature that aids users in generating multiple versions of a tweet for diverse audience engagement, promising it only takes 10 minutes with their **TweetStorm Express Flow**. More can be learned from their [Twitter post](https://x.com/OranAITech/status/1754461373466042527?s=20).

- **Introducing Dewy - Simplified RAG Applications**: `@kerinin` introduced a knowledge base platform called **Dewy**, which simplifies getting Retrieval Augmented Generation (RAG) applications up and running. They provided a [blog post](https://dewykb.github.io/blog/introducing-dewy/) for more details on Dewy's abilities to automate document extraction, indexing, and retrieval.

- **Crypto Project Seeking Passionate Professionals**: `@hinayoka` is on the lookout for professionals to fill various roles in an exciting crypto project, with vacancies including Web3 Developer, Game Developer, Web Developer, Moderator, and UI/UX Designer. Interested applicants are encouraged to reach out with their resume and portfolio.

- **Create Personalized AI Backlink Outreach Messages**: `@felixv3785` introduced a tool built with Vercel AI SDK and Langchain that generates personalized messages for backlink outreach. To test the **free tool**, visit [Backlink Outreach Message Generator](https://www.backlinkgpt.com/free-seo-tools/backlink-outreach-message-generator).

**Links mentioned**:

- [Introducing Dewy | Dewy](https://dewykb.github.io/blog/introducing-dewy/): Today we&#x27;re releasing the first version of Dewy, a knowledge base built for the specific needs of Gen AI applications.
- [ AI Form Roast by WorkHack - Free AI tool to audit and optimize online forms | Product Hunt](https://www.producthunt.com/posts/ai-form-roast-by-workhack): AI Form Roast by WorkHack is a free AI tool that analyzes online forms and provides feedback to improve completion rates and user experience. Trained on 1000+ forms, it generates insights into key are...
- [Tweet from Adi Oran (@OranAITech)](https://x.com/OranAITech/status/1754461373466042527?s=20): OranScribe New Flow Just Released! 🚀   Creativity is the art of echoing the same message in a thousand different ways.  OranScribe TweetStorm Express Flow.    Steps:  1. Set your audience and seo  2....
- [Backlink Outreach Message Generator](https://www.backlinkgpt.com/free-seo-tools/backlink-outreach-message-generator): Elevate your SEO strategy with our Backlink Outreach Message Generator. Craft personalized, compelling messages to secure valuable backlinks effortlessly.

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1203994042090782720) (66 messages🔥🔥): 

- **Sharing Setups & Tools**: `@ashpreetbedi` shared details about their recording and development setup involving pycharm and screenstudio.
- **Strategizing AI Team Structures**: `@30sleeps` discussed the idea of setting up an internal AI engineering team, possibly starting as a solo venture to demonstrate value before scaling up. `@quicknick123` and `@eugeneyan` offered insights into team formations and considerations for scaling ML projects, with `@eugeneyan` recommending reading material on real-time ML and team configurations.
- **DSPy Series Breakdown Interest**: `@lightningralf` sought a more accessible breakdown of a video series on DSPy, with `@kbal11` and `@coffeebean6887` expressing interest and suggesting the community could assist with understanding.
- **Lighthearted Banter on GPT-4**: Members like `@swyxio` and `@coffeebean6887` humorously discussed the perceived laziness of GPT-4, with links to relevant tweets and Reddit threads shared to highlight community feedback on the model’s performance.
- **Building Philosophical AI Agents**: `@dereklomas` proposed creating an AI-powered digital library with philosophical agents interacting with each other, with `@fanahova` suggesting tools like Botpress and WorkAdventure for development.

**Links mentioned**:

- [DSPy explained: No more LangChain PROMPT Templates](https://youtu.be/ycfnKPxBMck?feature=shared): DSPy explained and coded in simple terms. No more LangChain or LangGraph prompt templates. A self-improving LLM-RM pipeline! Plus automatic prompt engineerin...
- [AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback](https://arxiv.org/abs/2402.01469): The notable success of large language models (LLMs) has sparked an upsurge in building language agents to complete various complex tasks. We present AMOR, an agent framework based on open-source LLMs,...
- [Tweet from Sam Altman (@sama)](https://x.com/sama/status/1754172149378810118?s=46&t=90x): gpt-4 had a slow start on its new year&#39;s resolutions but should now be much less lazy now!
- [PromptHub Blog: How To Protect Against Prompt Hacking](https://www.prompthub.us/blog/how-to-protect-against-prompt-hacking): Learn everything you need to know about prompt hacking. Methods, defenses, and implications. See how PromptHub can help you ensure your prompts act as designed.
- [GitHub - copilot-us/chatgpt-plugins: Official ChatGPT Plugins🧩](https://github.com/copilot-us/chatgpt-plugins/): Official ChatGPT Plugins🧩. Contribute to copilot-us/chatgpt-plugins development by creating an account on GitHub.
- [Tweet from Sam Altman (@sama)](https://x.com/sama/status/1754172149378810118?s=46&t=90xQ8sGy63D2OtiaoGJuww): gpt-4 had a slow start on its new year&#39;s resolutions but should now be much less lazy now!
- [GitHub - DefinitelyTyped/DefinitelyTyped: The repository for high quality TypeScript type definitions.](https://github.com/DefinitelyTyped/DefinitelyTyped): The repository for high quality TypeScript type definitions. - GitHub - DefinitelyTyped/DefinitelyTyped: The repository for high quality TypeScript type definitions.
- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/186r89x/devs_aware_that_gpt_is_too_lazy_now_and_are/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/1aj6lrz/damned_lazy_ai/): no description found
- [Welcome to Flowise - FlowiseAI](https://docs.flowiseai.com/?): no description found
- [Botpress | the Generative AI platform for ChatGPT Chatbots](https://botpress.com/): Build ChatGPT chatbots faster with Botpress. An intuitive building experience powered by the latest in LLMs and GPT by OpenAI. Get started for free
- [GitHub - workadventure/workadventure: A collaborative web application (virtual office) presented as a 16-bit RPG video game](https://github.com/workadventure/workadventure): A collaborative web application (virtual office) presented as a 16-bit RPG video game - GitHub - workadventure/workadventure: A collaborative web application (virtual office) presented as a 16-bit ...
- [Real-time Machine Learning For Recommendations](https://eugeneyan.com/writing/real-time-recommendations/#how-to-design-and-implement-an-mvp).): Why real-time? How have China & US companies built them? How to design & build an MVP?
- [What is the most effective way to structure a data science team?](https://towardsdatascience.com/what-is-the-most-effective-way-to-structure-a-data-science-team-498041b88dae): From 2012 to 2017, I had the privilege to build the Data and Analytics organization at Coursera from scratch. Over that period of time, we…
- [Designing a data science organization](https://medium.com/data-science-at-microsoft/designing-a-data-science-organization-ab53a80b1d15): Data Science continues to be a growing and evolving field. Given this, there are multiple approaches in the industry for how to structure Data Science roles and organizations. In this post, I’ll…
- [The debate is over: Centralize your data science team | Prolego](https://www.prolego.com/blog/the-debate-is-over-centralize-your-data-science-team): Should you centralize your data science team and embed data scientists into product teams? Or should you hire data scientists directly into product teams?

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1204052397706584074) (61 messages🔥🔥): 

- **AI Foundational Models: A Call to Collaborate**: `@pratikk10` is reaching out to individuals interested in creating foundational models, such as text to text/image/video. Conversations with serious thinkers and creators in this area are welcomed.
- **Bias in Reinforcement Learning**: `@pseudoterminalx` discusses the concern that RLHF (Reinforcement Learning from Human Feedback) introduces significant bias into model weights, which is counterproductive for developing a base model. `@astropulse` echoes this sentiment, noting the distinctive style apparent in Midjourney's v3-4 images, which could be a result of such biases.
- **Challenges in Unlearning Textual Bias**: `@thejonasbrothers` shares the trials faced in attempting to unlearn text biases within pixart, especially stemming from version 5.1 datasets. `@pseudoterminalx` criticizes the utilization of the JourneyDB dataset, advocating that it should be discarded for more robust alternatives.
- **Revelations in Ancient Text**: `@itali4no` shares details about the unveiling of the Vesuvius Challenge 2023 Grand Prize winners, who have successfully developed methods to read 2000-year-old scrolls without opening them. `@nx5668` praises the transformative use of a TimeSformer model to detect ink in scroll scans, while noting the high cost of $40k per scroll for scanning using a particle accelerator.
- **Discussion on Chinese ML Expertise Amid Restrictions**: `@qwerty_qwer` questions how Chinese entities are exceeding in machine learning despite GPU restrictions, which leads to a brief discussion about the sufficiency of GPU access and the implications of recent export restrictions. `@kenjiqq` informs that major Chinese corporations had procured substantial quantities of NVIDIA's H100s and A100s before restrictions were enforced.

**Links mentioned**:

- [Vesuvius Challenge 2023 Grand Prize awarded: we can read the scrolls!](https://scrollprize.org/grandprize): The 2000-year-old scroll discusses music, food, and how to enjoy life’s pleasures.
- [Jinbo Xing](https://doubiiu.github.io): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1204197548877684808) (5 messages): 

- **Hugging Face's OWLSAM Spotlight**: User `@SegmentationFault` shared a [Hugging Face Space](https://huggingface.co/spaces/merve/OWLSAM) dedicated to **OWLSAM**, which combines **OWLv2** with **SAM** optimization.
- **Visual Representation Lacks Coverage**: `@SegmentationFault` noted disappointment in OWLSAM's performance stating it *“did not capture a lot of things in some tests I made”*.
- **OWLSAM Misidentifies Objects**: `@SegmentationFault` also pointed out that during their tests, OWLSAM tended to *“capture wrong objects too”* indicating a potential issue with the model's object detection accuracy.

**Links mentioned**:

[OWLSAM - a Hugging Face Space by merve](https://huggingface.co/spaces/merve/OWLSAM): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1204106198584926298) (1 messages): 

- **Big LlamaIndex Release Incoming**: User `@jerryjliu0` announced that a substantial **LlamaIndex release** is expected this week, featuring many cleanups. Users planning to upgrade their LlamaIndex version should anticipate this upcoming update.
  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1204104498436513813) (6 messages): 

- **LlamaIndex Enhances Multimodal Capabilities**: The integration with `@llama_index` enables the building of complete multi-modal applications that can run on a MacBook, taking image reasoning to the next level. Find out more in this [tweet](https://twitter.com/llama_index/status/1754545663155793972).
- **Home AI Wins Best Use of PDF Parser at Hackathon**: The first in-person hackathon highlighted winning projects like Home AI, which incorporates a **RAG-powered search engine** to filter homes by innovative criteria. Check out the winners in this [tweet](https://twitter.com/llama_index/status/1754601626688749755).
- **Hackathon Cultivates Innovation and Feedback**: Nearly 200 people participated in the hackathon, forming teams and providing real-time feedback to the LlamaIndex team on the user experience. More details can be found in this [announcement](https://twitter.com/llama_index/status/1754602472910520358).
- **Valuable Hackathon Resource Guide Shared**: LlamaIndex's resource guide was well received at the hackathon, catering to a range from beginners to experts. The guide is available [here](https://t.co/Oe5l44bSdl).
- **New RAG CLI Tool Unveiled**: A new CLI tool powered by `@llama_index`, Mistral-7B, and bge-m3 offers an LLM-powered grep for on-device file search and customization. Learn about this tool and its capabilities in this [tweet](https://twitter.com/llama_index/status/1754678983881621595).

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://t.co/Oe5l44bSdl): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1204008459478245396) (33 messages🔥): 

- **Integrating Knowledge Graphs with Azure**: User `@senshi2904` asked if there is a way to store and access knowledge graphs created by llama knowledge graph index using Azure CosmosDB Gremlin. There were no suggestions or responses provided in the messages.
- **Improving Prompt Effectiveness for Llama 2**: `@wrapdepollo` inquired about techniques to prevent Llama 2 from forgetting instructions when prompted with too much text, seeking tips and custom prompts from the community. No solutions were directly offered in the conversation.
- **Replit Bounties Inquiry**: `@d.j147` queried where to contact for Replit bounties, but no guidance was offered within the given messages.
- **Internal Document Summarization Challenge**: `@mysterious_avocado_98353` sought advice on how to summarize the most recent document regarding AAPL, with `created_time` as metadata, while `@akshay_1` mentioned that a query expansion via a large language model (llm) may be required, offering to DM resources later.
- **Updating Vector Stores in LlamaIndex**: `@ramihassanein` pointed out that vector stores like MongoDB and DeepLake might need updating to BasePydanticVectorStore, similar to the recent update of AstraDB. `@cheesyfishes` responded by encouraging contributions through pull requests (PRs) and updated that they address updates as they are pointed out.

**Links mentioned**:

- [GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex](https://github.com/run-llama/sec-insights): A real world full-stack application using LlamaIndex - GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex
- [llama_index/llama_index/vector_stores/mongodb.py at 61011d7721c5c95b15abfb840630be4b98a9beb5 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/61011d7721c5c95b15abfb840630be4b98a9beb5/llama_index/vector_stores/mongodb.py#L35): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama_index/vector_stores/deeplake.py at 61011d7721c5c95b15abfb840630be4b98a9beb5 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/61011d7721c5c95b15abfb840630be4b98a9beb5/llama_index/vector_stores/deeplake.py#L30): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama_index/vector_stores/astra.py at 61011d7721c5c95b15abfb840630be4b98a9beb5 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/61011d7721c5c95b15abfb840630be4b98a9beb5/llama_index/vector_stores/astra.py#L39): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [LlamaIndex RAG Hackathon &quot;SOLD OUT!&quot; -- (join waitlist)](https://rag-a-thon.devpost.com/project-gallery): Think Beyond Chatbots: Unleashing the Potential of AI Agents

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1204102419307298828) (25 messages🔥): 

- **Seeking Assistance for Chatbot Creation**: `@cay7man` expressed interest in building a chatbot for engineers to query standards documents. `@rick_03848` responded with a link to a GitHub project that uses LlamaIndex for document querying, which can be found at [GitHub - imartinez/privateGPT](https://github.com/imartinez/privateGPT).

- **Vector Search Woes Uncovered**: `@gavmor` reported having issues with vector search results using Qdrant, finding non-relevant nodes returned higher than expected nodes. `@cheesyfishes` replied, suggesting the review of `response.source_nodes` for debugging.

- **Retrieval Tips and Code Sharing**: `@gavmor` inquired about printing vector scores and comparing node similarity, eventually sharing TypeScript code for generating and comparing embeddings using `Ollama` and the query "How old is John?" as an example.

- **Architecture Quirks and Best Practices**: `@gavmor` showed concern about the proper usage of embeddings and LLM objects in their retrieval setup, leading to a discussion about the implementation. The conversation included a critique by `@cheesyfishes` on the LlamaIndex's LLM object containing embeddings.

- **Planning for Future Retrieval Experiments**: `@gavmor` mentioned their plans for further retrieval experiments, including pulling more nodes for reranking but expressed reluctance to change their chunking strategy.

**Links mentioned**:

- [Class: Ollama | LlamaIndex.TS](https://ts.llamaindex.ai/api/classes/Ollama): Unified language model interface
- [GitHub - imartinez/privateGPT: Interact with your documents using the power of GPT, 100% privately, no data leaks](https://github.com/imartinez/privateGPT): Interact with your documents using the power of GPT, 100% privately, no data leaks - GitHub - imartinez/privateGPT: Interact with your documents using the power of GPT, 100% privately, no data leaks

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1204096883212157029) (41 messages🔥): 

- **Qwen1.5 Model Released!**: `@bratao` announced the release of **Qwen1.5**, providing links to the blog post, GitHub, Hugging Face, ModelScope, a demo, and the Discord server. The model comes in multiple sizes and includes quantized versions for better developer experience.
  
- **GCP Offers Competitive Rates for Enterprises**: Users discussed Google Cloud Platform's (GCP) pricing, with `@nruaif` revealing that enterprise customers can access A100 instances for approximately **$1.5-2** per hour on demand, which is cheaper than the market rate for non-enterprise users.

- **Comparison and Benchmarks of Qwen 1.5**: `@yamashi` mentioned that **Qwen 1.5** appears better than Mistral based on benchmarks, expressing disappointment at the lack of a 30b model.

- **Complaints about Inconsistent Metrics in Benchmarks**: `@dreamgen` and `@yamashi` discussed the benchmarks associated with the new models, suggesting that the inherent noise in benchmarks should be acknowledged in results, and that standard deviation should be reported, although often omitted.

- **Insight into Reseller Dynamics and Pricing**: The conversation revealed how large cloud providers like AWS and GCP set higher non-enterprise prices, presumably to encourage resellers like RunPod, with further mention of spot price deals, such as the GCP L4 spot price at **$0.2** per hour, as shared by `@nruaif`.

**Links mentioned**:

[Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1203979767741943839) (14 messages🔥): 

- **Quantizing Before Merging**: `@dreamgen` highlighted that Hugging Face suggests quantizing the base model before merging, as [`qlora`](https://github.com/jondurbin/qlora/blob/main/qmerge.py#L42) does, which differs from the Axolotl script's approach.
- **Axolotl Possibility For Bayesian Optimization**: `@mihai4256` inquired about including abstractions for Bayesian optimization in the Axolotl framework for hyperparameter tuning, to which `@nruaif` responded that it seems easy to implement and referenced [Hugging Face's documentation](https://huggingface.co/docs/transformers/en/hpo_train) that supports such optimization for training.
- **Dependency Conflicts in Axolotl Installation**: `@nanobitz` reported dependency conflicts during a clean installation of Axolotl, involving package versions of `torch` where `xformers` had to be commented out. `@dctanner` proposed a workaround using `torch 2.1.2`.
- **Ideas for Axolotl UI Experience in Hugging Face**: `@dctanner` proposed the idea of a user interface within Hugging Face Spaces for Axolotl that could provide a beginner-friendly experience of configuring and running training using user's own Hugging Face account's GPU resources.
- **Axolotl YAML Configuration Simplification Request**: `@nanobitz` noted an anomaly with `ds2` saving potentially corrupt `model.safetensors` and also asked for contributions in removing redundant `is_*_derived_model` fields from all example YAML configurations in the Axolotl project.

**Links mentioned**:

- [Hyperparameter Search using Trainer API](https://huggingface.co/docs/transformers/en/hpo_train): no description found
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer#downsides-to-merging-qlora-before-dpo-approach-2): no description found
- [qlora/qmerge.py at main · jondurbin/qlora](https://github.com/jondurbin/qlora/blob/main/qmerge.py#L42): QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to jondurbin/qlora development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (1 messages): 

dangfutures: does anyone know how to the configs for zephyer <@257999024458563585>
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1203975144784138251) (25 messages🔥): 

- **Pro Upgrade Payment Issue Reported**: `@officialjulian` experienced a problem where funds were deducted for a Pro upgrade, but they received a "card has been declined" message. `@mares1317` suggested contacting support@perplexity.ai for assistance.
- **Billing Support Sought on Discord**: `@yuki.ueda` expressed trouble getting a response from support regarding a billing inquiry after two weeks. `@ok.alex` responded promptly, asking for a DM with the email to check the issue.
- **Confusion on Stripe's Declined Payments**: In response to `@officialjulian`'s issue, `@icelavaman` explained that the "card declined" message from Stripe implies no charge should have occurred.
- **Exploring AI Prompt Lengths in Collections**: `@twelsh37` inquired about prompt character limits as they had a prompt that was over by 2500 characters when adding to Collections.
- **Feedback on Claude's Performance**: `@Catto` mentioned that an earlier version of Claude was still available on Anthropic's site and suggested Perplexity might consider reverting to it, citing dissatisfaction with Claude 2.1's capabilities.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1176526177050054766): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1194794305362071552/1194794305362071552): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Dj Khaled Another One GIF - DJ Khaled Another One - Discover &amp; Share GIFs](https://tenor.com/view/dj-khaled-another-one-gif-26093316): Click to view the GIF
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1201430522493141073): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1204126551201816626) (7 messages): 

- **Ethical AI in Diverse Educational Contexts**: `@worriedhobbiton` raised concerns about the application of AI at Pasco High School, questioning if AI can provide unbiased and culturally sensitive support for students, especially given their LatinX identity and socioeconomic challenges. They asked for strategies to ensure that AI offers ethical guidance.

- **Irrelevant AI Research Query Result**: `@byerk_enjoyer_sociology_enjoyer` experienced an issue with the AI performing unrelated research when asked to assess the validity and authority of a source. They shared a [Perplexity AI search result](https://www.perplexity.ai/search/42bbb721-0450-47eb-bd01-f4f303e62d79) that did not meet expectations.

- **Seven Principles Inquiry**: `@fafu_10` provided a [Perplexity AI search link](https://www.perplexity.ai/search/7-principles-of-YQGmgm8eRlWxYXAdERakCg?s=u) without additional context or commentary.

- **AI Tools Showcase**: User `@vipinpg` shared a [Perplexity AI link](https://www.perplexity.ai/search/best-ai-tools-XM6zJ40PS5aCdNnxGYyJ0g) about the best AI tools, however, no further context or discussion was provided.

- **Contemplating Digital vs. Biological Intelligence**: `@mares1317` posted a [YouTube video](https://www.youtube.com/watch?v=iHCeAotHZa4) featuring Geoffrey Hinton discussing if digital intelligence will replace biological intelligence, hosted by the University of Toronto and associated institutes.

- **A Moment with Master Yoda**: `@mares1317` shared a [Yoda Star Wars GIF](https://tenor.com/view/yoda-star-wars-gif-8063259) without further discussion or context.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1203759744217653278): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Geoffrey Hinton | Will digital intelligence replace biological intelligence?](https://www.youtube.com/watch?v=iHCeAotHZa4): The Schwartz Reisman Institute for Technology and Society and the Department of Computer Science at the University of Toronto, in collaboration with the Vect...
- [Yoda Star Wars GIF - Yoda Star Wars - Discover &amp; Share GIFs](https://tenor.com/view/yoda-star-wars-gif-8063259): Click to view the GIF

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1204031720832376893) (4 messages): 

- **Seeking Assistance**: User `@aiistheonlyway` requested help, but did not provide details about the issue they're facing.
- **Link to Potential Solution**: `@icelavaman` shared a link, but the content or context of the link was not specified in the message.
- **Speeding Up Summaries**: `@sid.jjj` is looking for ways to reduce the API response time for generating summaries from long transcripts. They note that summary generation for three links in parallel takes approximately 10 seconds.
- **Discrepancy Issue Raised**: `@jbruvoll` asked for insights regarding a behavior discrepancy between interactive use and API usage, linking to a specific Discord message for reference.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1161802929053909012/1189372086658011237): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1118264005207793674/1204011306512945152): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1203974066554540093) (27 messages🔥): 

- **Dare_Ties Merge Achieves High Scores**: `@johannhartmann` shared that they created a high scoring Danish language model using a **dare_ties merge** method, now ranked 2nd on the Mainland Scandinavian NLG leaderboard. Details can be found at [munin-neuralbeagle-7b](https://huggingface.co/RJuro/munin-neuralbeagle-7b).

- **Merging Models Without GPUs**: `@sebastian.bodza` mentioned that **LeoLM models** can be merged as desired without the need for GPUs, and that this is possible even on platforms like Google Colab.

- **Wiedervereinigung-7b-dpo-laser Unveiled**: `@johannhartmann` introduced [Wiedervereinigung-7b-dpo-laser](https://huggingface.co/mayflowergmbh/Wiedervereinigung-7b-dpo-laser), a 7b parameter German model that combines some of the best German models available, and discussed its high scores on MT-Bench-DE.

- **Analyzing the Effectiveness of Model Merging**: In a discussion between `@johannhartmann` and `@bjoernp`, they explored whether high scores from model merging reflected an actual improvement in real-world use cases. `@johannhartmann` confirmed seeing improvements using chat functions.

- **Laser Treatment for Language Models Discussed**: `@johannhartmann` experimented with **laserRMT** on their model, but found that the mtbench scores were higher before using the procedure. The discussion hinted at the complexity of model merging and the effects of different training processes on model performance.

**Links mentioned**:

- [RJuro/munin-neuralbeagle-7b · Hugging Face](https://huggingface.co/RJuro/munin-neuralbeagle-7b): no description found
- [mayflowergmbh/Wiedervereinigung-7b-dpo-laser · Hugging Face](https://huggingface.co/mayflowergmbh/Wiedervereinigung-7b-dpo-laser): no description found

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1204023431558070302) (1 messages): 

- **Jina AI Launches New Code Embeddings**: `@sebastian.bodza` shared a [link to the new code embeddings from Jina AI](https://huggingface.co/jinaai/jina-embeddings-v2-base-code), which supports English and 30 programming languages for **neural search applications**. The model boasts **8192** sequence length and is best utilized through Jina AI's [Embedding API](https://jina.ai/embeddings/).

**Links mentioned**:

[jinaai/jina-embeddings-v2-base-code · Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v2-base-code): no description found

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1204034889318670366) (14 messages🔥): 

- **LLama2 SFT Training Loss Issues resolved?**: `@jellyroger5505` encountered a problem with a strange training loss curve while using SFT on LLama2. `@ufghfigchv` suggested that the issue might be due to a high learning rate, but upon checking `@jellyroger5505`'s config, the suggested resolution was to switch to [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) and possibly increase the learning rate based on their config examples.

- **Ankr Seeks Connection for Growth and Development**: `@anastasia_ankr` from Ankr reached out to connect with the engineering and/or business development team to discuss node infrastructure. `@ufghfigchv` pointed Ankr towards `@748528982034612226` as the contact person.

- **A Simple Hello Can Mean a Lot**: Both `@xterthy` and `@aslawliet` dropped brief greetings into the chat, fostering a welcoming community vibe.

- **Direct Message Awaiting a Response**: `@mizzy_1100` indicated the desire to communicate further, tagging `@748528982034612226` and advising them to check their Direct Messages (DMs) for more information.

- **Encouraging the Circus of Ideas**: `@rusch` playfully referred to the Discord server as an "amazing discordian circus," promoting a sense of fun and collaboration amongst its participants.

**Links mentioned**:

[axolotl/examples/llama-2/fft_optimized.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1204144296953913446) (1 messages): 

- **Audacity Integrates Intel's AI Tools**: `@dbreunig` announced that **Audacity**, the free audio editor, now includes a suite of free AI tools from **Intel**, challenging expensive subscription services. Key features are [noise suppression](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/noise_suppression/README.md), [transcription](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/whisper_transcription/README.md) using Whisper.cpp, [music separation](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/feature_doc/music_separation/README.md), and experimental generative features like music generation, all running locally on users' computers.

**Links mentioned**:

[Audacity now has free AI-powered sound tools from Intel - CDM Create Digital Music](https://cdm.link/2024/02/audacity-free-ai-tools-from-intel/): The free audio editor now gets a suite of free AI tools from Intel, some competing with expensive paid subscription services. That covers useful stuff like noise suppression and transcriptions and mus...

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1204010611806310401) (2 messages): 

- **Exploring `llm` for Thesis Work**: `@kiloton` has been using chatGPT for thesis brainstorming and is pleased with the results when combining file uploads and web search features. They seek advice on working with **PDFs and web searches** through `llm`, pondering if conversation histories can be ported to other models and retained locally.

- **Hugging Face Integration Interest**: `@dbreunig` expressed interest in integrating `llm` with [Hugging Face's transformers](https://huggingface.co/chatdb/natural-sql-7b), sharing a **Natural-SQL-7B** model that delivers strong Text-to-SQL performance and complex question understanding. The model reportedly surpasses peers in its category.

**Links mentioned**:

[chatdb/natural-sql-7b · Hugging Face](https://huggingface.co/chatdb/natural-sql-7b): no description found

  

---



### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1204136060964048916) (2 messages): 

- **Qwen1.5 Unleashes a New Model Generation**: `@potrock` shared a comprehensive introduction of **Qwen1.5**, announcing the open-sourcing of base and chat models with six different sizes. Various resources were provided such as the [blog post](https://qwenlm.github.io/blog/qwen1.5/), links to [GitHub](https://github.com/QwenLM/Qwen1.5), [Hugging Face](https://huggingface.co/Qwen), [Modelscope](https://modelscope.cn/organization/qwen), a [demo](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat), and a [Discord community](https://discord.gg/yPEP2vHTu4).
- **Llama's Potential Challenger**: `@potrock` highlighted that the **0.5B Qwen1.5 model** closely rivals the performance of Llama 7B. The revelation hints at significant efficiency improvements in the new model iteration.

**Links mentioned**:

[Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In recent months, our focus has been on developing a &ldquo;good&rdquo; model while optimizing the developer experience. As we progress towards...

  

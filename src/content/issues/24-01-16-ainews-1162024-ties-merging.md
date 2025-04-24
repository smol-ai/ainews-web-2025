---
id: 5321cce1-4789-4c89-a93b-314a98edf4b5
title: '1/16/2024: TIES-Merging'
date: '2024-01-16T20:51:01.991703Z'
original_slug: ainews-1162024-ties-merging
description: >-
  The **TheBloke Discord** community is actively discussing advancements in
  **Mixture of Experts (MoE) models**, including experiments with random gate
  routing layers and model merging strategies, alongside debates on quantization
  methods like **GPTQ** versus **EXL2** for performance optimization.
  Additionally, **Nous Research** announced the release of **Nous Hermes 2**, a
  new RLHF-trained model based on **Mixtral 8x7B** that reportedly outperforms
  Mixtral Instruct on many benchmarks, while the use of the **Frontier
  supercomputer** at Oak Ridge National Laboratory for training a
  trillion-parameter LLM has sparked conversations about open access to
  government-funded AI research.
tags:
  - thebloke
  - nous-research
  - mistral
  - google
  - oak-ridge-national-laboratory
  - vast
  - runpod
  - mixtral-8x7b
  - nous-hermes-2
  - mixtral-instruct
  - gptq
  - exl2
  - frankendpo-4x7b-bf16
  - academicat
  - mixture-of-experts
  - model-merging
  - quantization
  - rl-hf
  - fine-tuning
  - supercomputing
  - trillion-parameter-models
  - government-funded-research
  - ghost-attention
  - model-benchmarks
companies:
  - thebloke
  - nous-research
  - mistral
  - google
  - oak-ridge-national-laboratory
  - vast
  - runpod
models:
  - mixtral-8x7b
  - nous-hermes-2
  - mixtral-instruct
  - gptq
  - exl2
  - frankendpo-4x7b-bf16
  - academicat
topics:
  - mixture-of-experts
  - model-merging
  - quantization
  - rl-hf
  - fine-tuning
  - supercomputing
  - trillion-parameter-models
  - government-funded-research
  - ghost-attention
  - model-benchmarks
---


<!-- buttondown-editor-mode: plaintext -->> We checked **19** guilds, **284** channels, and **4372** messages for you. Estimated reading time saved (at 200wpm): **460 minutes**. Notice a jump? We added TheBloke's Discord today... and it's ultra active. We'll have to figure out how to balance this. We've also tweaked the prompts to make the summaries more informative.

As highlighted in recent issues, model merging is top of everyone's minds. We featured [Maxime Labonne's writeup](https://huggingface.co/blog/mlabonne/merge-models#2-ties) 2 days ago, and [the TIES paper](https://arxiv.org/abs/2306.01708) is now making the rounds again.

 ![image.png](https://assets.buttondown.email/images/e04fbd23-615e-476b-938b-a6eb77fe4528.png?w=960&fit=max) 

Digging into the details, the results are encouraging but not conclusive.

 ![image.png](https://assets.buttondown.email/images/a4a2aea0-8e57-4abe-a7e6-213ab2370338.png?w=960&fit=max) 

--

**Table of Contents**

[TOC] 


## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

**MoE Model Mixology**: Discussions circled around creating **efficient MoE (Mixture of Experts) models**, with experiments in **random gate routing layers** for training and the potential of merging top models from benchmarks. [@sanjiwatsuki](https://discord.com) posited that while beneficial for training, random gate layers may not be ideal for immediate model usage.

**Quantize with Caution**: A robust debate ensued over the **efficacy of various quantization methods**, comparing **GPTQ** and **EXL2 quants**. There was a general consensus that EXL2 might offer faster execution on specialized hardware, but the full scope of trade-offs requires further exploration.

**The Narrative Behind Model Fine-Tuning**: [@superking__](https://discord.com) flagged potential, undisclosed complexities in **finetuning Mixtral models**, citing recurring issues across finetunes. Additionally, a mention was made of a **frankenMoE model**, presumably optimized and performing better in certain benchmarks, available at [FrankenDPO-4x7B-bf16 on Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16).

**Training Anomalies and Alternatives**: The perplexing occurrence of a **model's loss dropping to near zero** sparked discussions about possible exploitation of the reward function. Alternatives to **Google Colab Pro for cost-effective fine-tuning** were discussed, with **vast.ai** and **runpod** recommended as potential options.

**Supercomputing in the Name of AI**: The community was abuzz about Oak Ridge National Laboratory's **Frontier supercomputer** used to train a trillion-parameter LLM, stirring debates on the openness of government-funded AI research. Meanwhile, [@kaltcit](https://discord.com) boasted about incorporating **ghost attention** within their 'academicat' model, eliciting both skepticism and curiosity from peers.

**TheBloke Channel Summaries**

### ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/) (1786 messages🔥🔥🔥): 
        
- **Exploring MoE Training and Performance**: Users like `@sanjiwatsuki` and `@rombodawg` are discussing strategies for creating efficient MoE (Mixture of Experts) models, experimenting with tactics like using random gate router layers for training and merging top models from benchmarks to potentially improve leaderboard scores. Sanjiwatsuki mentions that random gate layer is good for training but not for immediate use, while Rombo is experimenting to challenge the leaderboard.

- **Discussion on the Efficiency of Quantization**: Participants are trying to understand the benefits and trade-offs of different quantization methods. They're debating on the speed and performance gains when moving from GPTQ to EXL2 quants, with consensus that EXL2 can lead to faster execution on high-performance hardware.

- **New Model Release by Nous Research**: `@mrdragonfox` announced a new model called Nous Hermes 2 based on Mixtral 8x7B, which has undergone RLHF training and claims to outperform Mixtral Instruct in many benchmarks. However, `@_dampf` found during a short test on together.ai that Hermes 2 showed some inconsistencies compared to Mixtral Instruct.

- **AI Supercomputer for LLM Training**: Users discuss a news piece about Oak Ridge National Laboratory's supercomputer called Frontier, used for training a trillion-parameter LLM with a requirement of 14TB RAM. The conversation turned towards whether such government-funded models need to be open-sourced, with `@kaltcit` arguing that they should be according to usual requirements for government-funded research.

- **Focus on Application of Ghost Attention in Models**: `@kaltcit` claims to have recreated ghost attention in a model they're calling academicat, with the model able to handle complex prompted instructions across multiple turns. There is a hint of skepticism and curiosity from other users like `@technotech` about other models employing this technique, with `@kaltcit` noting academicat is the only one besides llama chat that they've seen it in.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discordapp.com/channels/1053877538025386074/1145143867818119272/1196552788205908048): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Chat with Open Large Language Models](https://chat.lmsys.org/)
- [Kquant03/FrankenDPO-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)
- [Mistral AI - Implementation specialist](https://jobs.lever.co/mistral/84f45658-6bd5-4c61-b3ce-4d6e7cc3bc90): Mistral AI is looking for an Implementation Specialist to drive adoption of its products with its early customers. The Implementation Specialist will be an integral part of our team, dedicated to driv...
- [
First Token Cutoff LLM sampling - &lt;antirez&gt;
](http://antirez.com/news/142)
- [Curly Three Stooges GIF - Curly Three Stooges 81C By Phone - Discover &amp; Share GIFs](https://tenor.com/view/curly-three-stooges-81c-by-phone-gif-20798723): Click to view the GIF
- [Takeshi Yamamoto GIF - Takeshi Yamamoto Head Scratch Head Scratching - Discover &amp; Share GIFs](https://tenor.com/view/takeshi-yamamoto-head-scratch-head-scratching-my-fault-oops-gif-5312570): Click to view the GIF
- [jbochi/madlad400-10b-mt · Hugging Face](https://huggingface.co/jbochi/madlad400-10b-mt)
- [Most formidable supercomputer ever is warming up for ChatGPT 5 &mdash; thousands of 'old' AMD GPU accelerators crunched 1-trillion parameter models](https://www.techradar.com/pro/most-formidable-supercomputer-ever-is-warming-up-for-chatgpt-5-thousands-of-old-amd-gpu-accelerators-crunched-1-trillion-parameter-models): Scientists trained a GPT-4-sized model using much fewer GPUs than you'd ordinarily need
- [moreh/MoMo-70B-lora-1.8.4-DPO · Hugging Face](https://huggingface.co/moreh/MoMo-70B-lora-1.8.4-DPO)
- [SanjiWatsuki/tinycapyorca-8x1b · Hugging Face](https://huggingface.co/SanjiWatsuki/tinycapyorca-8x1b)
- [turboderp/Mixtral-8x7B-instruct-exl2 at 3.5bpw](https://huggingface.co/turboderp/Mixtral-8x7B-instruct-exl2/tree/3.5bpw)
- [clibrain/mamba-2.8b-instruct-openhermes · Hugging Face](https://huggingface.co/clibrain/mamba-2.8b-instruct-openhermes)
- [240105-(Long)LLMLingua-AITime.pdf](https://drive.google.com/file/d/1fzK3wOvy2boF7XzaYuq2bQ3jFeP1WMk3/view)
- [Mili - world.execute(me); 【cover by moon jelly】](https://www.youtube.com/watch?v=wFXK4osifXw): execution♡♡♡♡♡♡e-girlfriend momentSOUNDCLOUD: https://soundcloud.com/moonjelly0/worldexecuteme~CREDITS~Vocals, Mix, Animation : moon jelly (Me!)(https://www....
- [Robocop Smile GIF - Robocop Smile Robocop smile - Discover &amp; Share GIFs](https://tenor.com/view/robocop-smile-robocop-smile-robocop-rogue-city-robocop-happy-gif-3488842367248583764): Click to view the GIF
- [Tweet from Nous Research (@NousResearch)](https://fxtwitter.com/NousResearch/status/1746988416779309143): Introducing our new flagship LLM, Nous-Hermes 2 on Mixtral 8x7B.   Our first model that was trained with RLHF, and the first model to beat Mixtral Instruct in the bulk of popular benchmarks!  We are r...
- [Build software better, together](https://github.com/ggerganov/llama.cpp/pull/4930).): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2): A fast inference library for running LLMs locally on modern consumer-class GPUs - GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs
- [豆包](https://www.doubao.com/): 豆包是你的AI 聊天智能对话问答助手，写作文案翻译情感陪伴编程全能工具。豆包为你答疑解惑，提供灵感，辅助创作，也可以和你畅聊任何你感兴趣的话题。
- [A study of BERT for context-aware neural machine translation - Machine Learning](https://link.springer.com/article/10.1007/s10994-021-06070-y): Context-aware neural machine translation (NMT), which targets at translating sentences with contextual information, has attracted much attention recently. A key problem for context-aware NMT is to eff...
- [Add ability to use importance matrix for all k-quants by ikawrakow · Pull Request #4930 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4930): TL;DR See title I see improvements in perplexities for all models that I have tried. The improvement is most significant for low-bit quantization. It decreases with bits-per-weight used, and become...


### ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/) (43 messages🔥): 
        
- **Mistral Finetuning Challenges**: `@superking__` suggests that finetuning Mixtral may have unknown complexities, as most finetunes seem to have issues, hinting at a possible secret aspect not disclosed by MistralAI.
- **Repeated Expressions in Roleplay**: Regarding the use of Yi for roleplay, `@superking__` observes that it tends to latch onto certain expressions, repeating them across multiple messages.
- **Finetuning FrankenMoE Adventures**: `@kquant` shares the creation of a frankenMoE made from "DPOptimized" models which perform better on GSM8k and Winogrande benchmarks than Mixtral Instruct 8x 7B. Also, [Kquant's frankenMoE at Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16) was noted as a redemption for a previous flawed ERP model.
- **Mixtral Trix Not MoE Material**: `@kquant` learns that Mixtral Trix models do not serve well as material for MoE (Mixture of Experts) models, a finding that might impact future frankenMoE development.
- **Dynamic Audio for Evocative Settings**: `@netrve` and `@kquant` discuss the possibility of having dynamic audio that changes based on story location, envisioning a system resembling a Visual Novel which could script automatic scene changes for enhanced immersion.

**Links mentioned**:

- [Kquant03/FrankenDPO-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-bf16)
- [Kquant03/FrankenDPO-4x7B-GGUF · Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-GGUF)
- [Most formidable supercomputer ever is warming up for ChatGPT 5 &mdash; thousands of 'old' AMD GPU accelerators crunched 1-trillion parameter models](https://www.techradar.com/pro/most-formidable-supercomputer-ever-is-warming-up-for-chatgpt-5-thousands-of-old-amd-gpu-accelerators-crunched-1-trillion-parameter-models): Scientists trained a GPT-4-sized model using much fewer GPUs than you'd ordinarily need


### ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/) (24 messages🔥): 
        
- **Optimal Model Combination for Scaling**: `@sao10k` recommends using **qlora with Mistral** when planning to scale up data, suggesting it as the best case scenario.
- **A Weird Reward Function Anomaly**: `@nruaif` pointed out an abnormality where their model's loss dropped to near zero, which could imply the model found a way to **cheat the reward function**.
- **Finetuning Format Confusion**: `@joao.pimenta` seeks advice on the proper format for finetuning a chat model using **auto-train** and is unsure how to implement chat history and enforce single responses from the model. They provided a structure based on information from ChatGPT but expressed doubts about its correctness.
- **Epoch Jumps in Training Revealed**: `@sanjiwatsuki` questioned the unusual jumping of epochs in their model's training, later attributing the issue to **Packing=True** being enabled.
- **Cloud Fine-tuning Alternatives Explored**: `@jdnvn` asked for cheaper cloud alternatives to **Google Colab Pro** for fine-tuning models, with `@sao10k` suggesting **vast.ai** or **runpod** depending on the specific requirements of the model and dataset size.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Embeddings on a Budget**: Embeddings are described as "really cheap," with **window chunking** suggested for sentences. Discussion highlighted the need for optimal chunking, suggesting overlapping chunks might improve retrieval accuracy, esp. for smaller models. Local models were noted for their time-saving embedding creation, and a hierarchical strategy is currently being tested for its effectiveness.

- **Multimodal Mergers and Efficient GPT Hopes**: Reddit talks about a homemade multimodal model combining Mistral and Whisper, signaling community innovation. Twitter reflects a preference for a more efficient "GPT-5 with less parameters," which aligns with a chat focus on techniques and architectures for AI progression, like OpenAI's InstructGPT, Self-Play Preference Optimization (SPO), and discussions on whether simply scaling up models is still the right approach.

- **Introducing Nous-Hermes 2**: Nous-Hermes 2, a model surpassing Mixtral Instruct in benchmarks, was released with SFT and SFT+DPO versions. The **DPO model** is available on [Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO), and **Together Compute** offers a live model playground to try Nous-Hermes 2 firsthand at [Together's model playground](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO).

- **Model Training and Generalization Discussed**: Community members debated Nous-Hermes-2's benchmarks, with SFT+DPO outperforming other models. Possibilities for models to generalize beyond training data distributions were explored, and the usage of synthetic GPT data in training Mistral models was confirmed. MoE and DPO strategies were also lightly touched on.

- **UI and Training Challenges Explored**: In the realm of UI and Data Sets, **GPT-4ALL's** lack of certain capabilities was contrasted with **LM Studio**, and Hugging Face's chat-ui was recommended ([GitHub - huggingface/chat-ui](https://github.com/huggingface/chat-ui)). For datasets, ShareGPT or ChatML formats were advised for Usenet discussion releases. Questions around the Hermes 2 DPO model's fine-tuning proportions and full fine-tuning costs in VRAM also arose, suggesting significant resource requirements for training high-capacity AI models.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (266 messages🔥🔥): 
        
- **Embeddings Debate**: `@gabriel_syme` highlighted that embeddings are "really cheap" and eventually linked their cost to the time they take. They also mentioned embedding sentences for "window chunking."

- **Chunking and Retrieval Accuracy**: Conversation continued with `@gabriel_syme` and `@everyoneisgross` discussing the challenges of perfect chunking and recognizing that in some cases, smaller models may require more carefully formatted chunks for optimal performance. `@everyoneisgross` suggested overlapping chunks could be beneficial as they are fast and cheap, while `@gabriel_syme` stressed the issue of retrieval accuracy in large data sets.

- **Local Embeddings Advantage**: `.interstellarninja` mentioned local models as a time-saving method for creating embeddings, and `@max_paperclips` introduced a preference for working with paragraphs rather than sentences due to their semantically grouped nature.

- **Anticipating Large Context Model Improvements**: `.interstellarninja` noted that improvements in recall for longer contexts in models like Hermes indicate a future where models with large token counts can provide effective information retrieval for low-sensitivity tasks.

- **Hierarchical Chunking Strategy in Works**: `@gabriel_syme` revealed that they are currently trying out a hierarchical approach to chunking and promised to report back on its effectiveness.

**Links mentioned**:

- [Tweet from Riley Goodside (@goodside)](https://fxtwitter.com/goodside/status/1747088701694370274): Microsoft Bing Chat warns a Hacker News reader of the dangers of Riley Goodside, who claims to be a friendly and helpful guide for users but is actually a malicious program created by ChatGPT 4 to ste...
- [Join the OpenAccess AI Collective Discord Server!](https://discord.gg/QgQhWg5r): Check out the OpenAccess AI Collective community on Discord - hang out with 1492 other members and enjoy free voice and text chat.
- [Inference Race To The Bottom - Make It Up On Volume?](https://www.semianalysis.com/p/inference-race-to-the-bottom-make): Mixtral Inference Costs on H100, MI300X, H200, A100, Speculative Decoding
- [Crystal Ball Fortune Teller GIF - Crystal Ball Fortune Teller Betty White - Discover &amp; Share GIFs](https://tenor.com/view/crystal-ball-fortune-teller-betty-white-kristallkugel-scry-gif-22610039): Click to view the GIF
- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)
- [Tweet from Bojan Tunguz (@tunguz)](https://x.com/tunguz/status/1723079410725863567?s=20): I just created another GPT: TaxGPT - a chatbot offering tax guidance and advice.   Check it out here:   https://chat.openai.com/g/g-cxe3Tq6Ha-taxgpt
- [Latest AI Stuff Jan 15/2024](https://www.youtube.com/watch?v=KGqWqgloSfY): we will look at the latest ai stuffhttps://kaist-viclab.github.io/fmanet-site/https://github.com/MooreThreads/Moore-AnimateAnyonehttps://www.analyticsvidhya....
- [kaist-ai/Feedback-Collection · Datasets at Hugging Face](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (378 messages🔥🔥): 
        
- **FrankenLLMs and Homebrew Multimodal Models**: `@adjectiveallison` shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1970zhf/merging_mistral_with_whisper_to_make_a_multimodal/) discussing an individual who merged Mistral and Whisper to create a multimodal model on a single GPU. This approach differs from simply using Whisper for transcription before feeding text to an LLM and could lead to more integrated audio-text model interactions.

- **Public Interest in Efficient GPT-5**: `.interstellarninja` conducted a [Twitter poll](https://fxtwitter.com/intrstllrninja/status/1746840644151087422?s=20) about AI progress, where "GPT-5 with less parameters" was most favored, suggesting a public desire for more efficient models over larger ones with more tokens. The poll aligns with sentiments in the chat about advancements beyond just increasing parameter counts.

- **InstructGPT’s Impact on Model Training**: `@ldj` discussed how OpenAI's InstructGPT methodology allowed a 6B parameter model to perform with higher human preference than a 175B GPT-3 model with the same pretraining. This illustrates that improved training techniques, architecture changes, better handling of the data, and implementation of newer models like Alpaca can potentially lead to significant performance improvements without increasing parameter count.

- **Self-Play and Reinforcement Learning Advances**: `@ldj` brought attention to [research](https://arxiv.org/abs/2401.04056) on Self-Play Preference Optimization (SPO), an algorithm for reinforcement learning from human feedback that simplifies training without requiring a reward model or adversarial training. This type of algorithm could play a role in future advancements by enhancing the ability of models to learn from interactions with themselves, likely improving robustness and efficiency in training.

- **Is Scaling Still King?**: Throughout the conversation, `@giftedgummybee` and `@ldj` debated whether OpenAI will continue to scale parameters up for GPT-5 or focus on new architectures and training techniques. The discussion highlighted differing opinions on the best path for advancement in AI, with `@giftedgummybee` expressing skepticism about moving away from transformers, given their current success and potential for incorporating new modalities.

**Links mentioned**:

- [Let&#39;s Verify Step by Step](https://arxiv.org/abs/2305.20050): In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. T...
- [A Minimaximalist Approach to Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2401.04056): We present Self-Play Preference Optimization (SPO), an algorithm for reinforcement learning from human feedback. Our approach is minimalist in that it does not require training a reward model nor unst...
- [Listening with LLM](https://paul.mou.dev/posts/2023-12-31-listening-with-llm/#background): Overview This is the first part of many posts I am writing to consolidate learnings on how to finetune Large Language Models (LLMs) to process audio, with the eventual goal of being able to build and ...
- [Tweet from interstellarninja (@intrstllrninja)](https://fxtwitter.com/intrstllrninja/status/1746840644151087422?s=20): what would progress in AI look like to you?  ████████████████████ GPT-5 w/ less parameters  (62.5%) ████ GPT-5 w/ more parameters  (12.5%) ██████ GPT-5 w/ less tokens  (18.8%) ██ GPT-5 w/ more tokens ...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1970zhf/merging_mistral_with_whisper_to_make_a_multimodal/)


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (1 messages): 
        
- **Nous-Hermes 2 Dethrones Mixtral Instruct**: `@teknium` announces the new **Nous-Hermes 2** model, the first model trained with RLHF and surpassing Mixtral Instruct in benchmarks, with both the SFT only and SFT+DPO versions released, along with a qlora adapter for the DPO. Check out the [DPO model on Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO).

- **SFT Version Unleashed**: The supervised finetune only version of Nous Hermes 2 Mixtral 8x7B (SFT) is now available. For SFT enthusiasts, the version aimed at providing an alternative to the SFT+DPO model can be found on [Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT).

- **DPO Adapter Now Ready**: The QLoRA Adapter for the DPO phase of Nous-Hermes-2 Mixtral 8x7B has been made public. For developers looking to utilize the DPO phase more seamlessly, visit the [Hugging Face repository](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-adapter).

- **GGUF Versions Roll Out**: GGUF versions of Nous-Hermes-2 are compiled and ready in all quantization sizes. Access the [DPO GGUF](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF) and [SFT only GGUF](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT-GGUF) on their respective pages.

- **AI Playground on Together Compute**: To experience Nous-Hermes 2 firsthand, head over to Together Compute's API. The Model Playground is now live with the DPO model at [Together's model playground](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO).

**Links mentioned**:

- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-adapter · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-adapter)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT-GGUF · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT-GGUF)
- [TOGETHER](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (321 messages🔥🔥): 
        
- **Cloning Woes for Copyninja_kh**: `@copyninja_kh` faced an error when cloning and running Axolotl; a long filename error caused a failed checkout from a `git clone` command, and subsequent messages suggest confusion on whether they needed to fork a repository first for their operations.
- **DPO vs SFT Model Evaluation**: `@n8programs` and `@teknium` contributed to discussions about the new Nous-Hermes-2-Mixtral model's performance, especially the SFT + DPO version, which reportedly scores higher on certain benchmarks than other models, beating a Mixtral-instruct on a benchmark with 73 vs 70.
- **Generalization Beyond Model Training**: `@n8programs` pointed out that it's possible for models to generalize beyond the distribution of their original training data, potentially leading to performance that surpasses that of GPT-4 when trained with synthetic data from it. This idea was contested by `@manojbh`, who differentiated between generalizing within the data distribution and scaling beyond it.
- **Preferences in Model Announcements**: `@manojbh` and `@makya` discussed how Mistral base models use synthetic GPT data, and `@teknium` confirmed that models like Nous-Hermes-2-Mixtral are trained using outputs from GPT models. There was also mention of a Misral v0.2, but it was clarified that v0.1 is the latest.
- **Light Discussion on MoE and DPO**: Gating mechanisms and domain specialization were briefly discussed by `@baptistelqt` and `@teknium`, with a mention of looking at different gating strategies and how MoE stabilizes training without necessarily pushing domain specialization. `@yikesawjeez` referred to research exploring multiple gating strategies for MoE models.

**Links mentioned**:

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised ...
- [Weak-to-strong generalization](https://openai.com/research/weak-to-strong-generalization): We present a new research direction for superalignment, together with promising initial results: can we leverage the generalization properties of deep learning to control strong models with weak super...
- [Fine-Tuning Llama-2 LLM on Google Colab: A Step-by-Step Guide.](https://medium.com/@csakash03/fine-tuning-llama-2-llm-on-google-colab-a-step-by-step-guide-cf7bb367e790): Llama 2, developed by Meta, is a family of large language models ranging from 7 billion to 70 billion parameters. It is built on the Google…
- [Cat Cats GIF - Cat Cats Cat meme - Discover &amp; Share GIFs](https://tenor.com/view/cat-cats-cat-meme-meme-meme-cat-gif-14470917232397934693): Click to view the GIF
- [HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System](https://arxiv.org/abs/2203.14685): As giant dense models advance quality but require large amounts of GPU budgets for training, the sparsely gated Mixture-of-Experts (MoE), a kind of conditional computation architecture, is proposed to...
- [mistralai/Mixtral-8x7B-Instruct-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Do It GIF - Do It Get - Discover &amp; Share GIFs](https://tenor.com/view/do-it-get-to-work-gif-21630516): Click to view the GIF
- [one-man-army/UNA-34Beagles-32K-bf16-v1 · Hugging Face](https://huggingface.co/one-man-army/UNA-34Beagles-32K-bf16-v1)
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF)
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/180p17f/new_claude_21_refuses_to_kill_a_python_process/?utm_source=share&utm_medium=mweb3x&utm_name=mweb3xcss&utm_term=1&utm_content=share_button)
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [HuggingFaceH4/open_llm_leaderboard · [FLAG] fblgit/una-xaberius-34b-v1beta](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/444)


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (96 messages🔥🔥): 
        
- **GPT-4ALL and LM Studio UI Capabilities**: `@manojbh` pointed out that **GPT-4ALL** does not support vision and function calling, while **LM Studio** does but only for local models. They recommended an alternative UI with support for web browsing, by sharing **Hugging Face's chat-ui**: [GitHub - huggingface/chat-ui](https://github.com/huggingface/chat-ui).

- **Data Formatting for Dialogue Mining in AI**: `@.toonb` sought advice on the best data format for releasing a mined dataset of Usenet discussions for AI training. **Max_paperclips** recommended the ShareGPT or ChatML format for its compatibility with libraries and its suitability for multi-turn conversations.

- **Training Semantic Proportions for Hermes 2 DPO Model**: `@teknium` clarified to `@samin` that the ratio of SFT to DPO fine-tuning for the **Hermes 2 DPO model** is closer to 100:5, indicating a significantly higher proportion of SFT examples than DPO examples.

- **Curiosity Around Hermes Mixtral**: `@jaredquek` thanked for the new **Hermes Mixtral** and inquired if it's a full fine-tune, while also mentioning that 8bit LoRA doesn't seem to work with it. `@teknium` confirmed it's a full fine-tune.

- **Cost of Fine-Tuning on GPU**: `@jaredquek` and `@n8programs` discussed the high VRAM cost of full fine-tuning (FFT), with `@teknium` mentioning it costs around 14 times more VRAM, whereas `@n8programs` noted that using alternatives like qLoRA or float16 precision can save on VRAM.

**Links mentioned**:

[GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **AI Impersonation Challenges Content Creators**: The ongoing discussions on the impact of AI-generated content on legal rights highlighted a case where a YouTube channel was taken down for using David Attenborough's voice AI-generated narrations. The conversations around copyright and privacy implications for AI underlined the importance of understanding laws concerning impersonation and likeness for AI engineers.

- **Data Handling Tips for RAG Accuracy**: The recommendation of [SuperDuperDB](https://github.com/SuperDuperDB/superduperdb) to `@liberty2008kirill` in response to questions about improving RAG application accuracy while handling CSV data points engineers toward possible solutions that integrate AI applications with existing data infrastructure.

- **Service Quality Concerns Following GPT Store Launch**: Engineers noted a correlation between the rollout of the GPT store and service quality issues such as lagging and network errors. This observation prompts discussions on the impact of new features and services on the reliability and performance of GPT-4.

- **Prompt Engineering and Attachments in GPT**: Members shared tactics to increase the efficacy of prompt engineering and improve GPT's interactions with attachments, including embodying specific command phrases like "Analyze the attached" and adopting structured data for enhanced retrieval and generation.

- **Exploring Modularity with Lexideck Technologies**: The engineering implications of Lexideck were discussed, identifying it as a potential tool for testing various prompt optimization models. The adaptability and modularity of such frameworks were of particular interest in the context of improving AI's agentic behaviors.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (113 messages🔥🔥): 
        
- **Copyright Takedown Precedent**: User `.dooz` discussed an example of AI-generated content being restricted legally, highlighting a YouTube channel using David Attenborough's voice to narrate Warhammer 40k videos that was shut down. This instance demonstrates that laws concerning impersonation and likeness could impact AI-generated content.

- **SuperDuperDB Suggested for RAG with CSV Data**: In response to `@liberty2008kirill` seeking help on RAG application accuracy with CSV data, `@lugui` recommended checking out [SuperDuperDB](https://github.com/SuperDuperDB/superduperdb), a project that might help in building and managing AI applications directly connected to existing data infrastructure.

- **Context Size and Role-play Capabilities in AI**: The OpenAI Discord channel had a detailed discussion, including `@i_am_dom_ffs` and `@darthgustav.`, about the role of context size in AI's ability to maintain character during role-play. Users debated whether a larger context size improves the AI's consistency or if attention and retrieval mechanisms are more significant factors.

- **Link Sharing and Permissions**: Users like `@mrcrack_` and `@Cass of the Night` discussed the ability to share links within the Discord channel, with suspicions that some sources might be whitelisted to bypass immediate muting, which is the general policy for most links shared.

- **ChatGPT Downtime and Issues Discussion**: Several users, including `@die666die666die` and `@kazzy110`, reported potential downtimes and errors with ChatGPT. `@solbus` provided troubleshooting advice, while `@satanhashtag` directed users to check [OpenAI's status page](https://status.openai.com/) for updates.

**Links mentioned**:

- [Dead Internet theory - Wikipedia](https://en.wikipedia.org/wiki/Dead_Internet_theory)
- [Welcome to Life: the singularity, ruined by lawyers](https://youtu.be/IFe9wiDfb0E?feature=shared>): http://tomscott.com - Or: what you see when you die.If you liked this, you may also enjoy two novels that provided inspiration for it: Jim Munroe&#39;s Everyone ...
- [GitHub - SuperDuperDB/superduperdb: 🔮 SuperDuperDB: Bring AI to your database! Build, deploy and manage any AI application directly with your existing data infrastructure, without moving your data. Including streaming inference, scalable model training and vector search.](https://github.com/SuperDuperDB/superduperdb): 🔮 SuperDuperDB: Bring AI to your database! Build, deploy and manage any AI application directly with your existing data infrastructure, without moving your data. Including streaming inference, scal.....


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (82 messages🔥🔥): 
        
- **GPT Chat Modifications Bewilder Users**: `@csgboss` voiced frustration after teaching GPT to handle conversation starters, only to have the chatbot replace them with ineffective ones. `@pietman` advised manual configuration instead of using the chat feature to prevent overwriting.
  
- **Users Face Lag and Network Problems with GPT-4**: Multiple users, including `@blacksanta.vr`, `@kemeny`, and `@shira4888` reported lagging issues and error messages indicating network problems with GPT-4, which intensified after the introduction of the GPT store.

- **Troubles with Hyperlinks in Custom GPT Outputs**: Users `@thebraingen` and `@kemeny` discussed challenges with GPT not generating clickable hyperlinks, necessitating workarounds like building an API to fix the problem, as mentioned by `@kemeny`.

- **AI Teaching Approach Simulating Human Learning Suggested**: `@chotes` and `@d_smoov77` proposed that GPT should follow the development model of a human student, starting from a base language and progressively building expertise through a curated curriculum.

- **The Advent of GPT Store Appears to Impact Service Quality**: Users like `@blacksanta.vr` and `@pixrtea` noticed a decline in GPT’s performance coinciding with the GPT store rollout, leading to broader discussion on the current issues and the potential for growth in the GPT's service quality.

**Links mentioned**:

- [Hyperlinks in Custom GPT not linking?](https://community.openai.com/t/hyperlinks-in-custom-gpt-not-linking/565252/31): I still have same problem. tried all fixes in comments still same.  Friday, January 12, 2024 11:50:13 PM
- [Custom GPT Bug - Hyperlinks not clickable](https://community.openai.com/t/custom-gpt-bug-hyperlinks-not-clickable/565499): It looks like hyperlinks produced by custom GPTs are not working. Here is my GPT which provides links to research papers: https://chat.openai.com/g/g-bo0FiWLY7-researchgpt.  However, I noticed that th...


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (159 messages🔥🔥): 
        
- **Trouble Training ChatGPT to Remember Preferences**: `@henike93` is having issues with ChatGPT not remembering changes, particularly after uploading a pdf and wanting a different response than what's in the document. `@darthgustav.` suggests using more specific language, such as: *"Always use the example(s) in your knowledge to improvise original, unique responses based on the current context and the examples provided."* and also notes that structured data is easier for retrieval and generation (RAG).
  
- **GPT Gets Attachment Amnesia**: `@madame_architect` observes that attaching a file with a prompt doesn't guarantee the GPT will read the attached document, a behavior that can be corrected by specifically referring to "the attached paper" in the prompt. `@darthgustav.` recommends stating "Analyze the attached" in the prompt to direct attention to the file.

- **Contrastive Conundrums: Challenges in Generalizing CCOT**: `@madame_architect` is grappling to find generalized natural language prompts for Contrastive Chain of Thought (CCOT) that don't resemble grade school tests. `@darthgustav.` theorizes that contrastive conditions in the main prompt can effectively provoke the desired contrasts.
 
- **Prompt Engineering Battlebots**: `@madame_architect` and `@darthgustav.` discuss the possibility of creating a framework, like darthgustav.'s Lexideck, to test various prompt optimization models against each other under controlled conditions. `@darthgustav.` explains how his system of Lexideck can adapt and emulate almost any software from documentation.

- **Prompt Engineering is Not a Walk in the Park**: `@electricstormer` expressed frustration at getting GPT to follow instructions consistently, noting that it often ignores parts of the input. `@darthgustav.` responded by asking for more details to help and acknowledging that prompt engineering can indeed be challenging and requires fine-tuning for consistency.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (159 messages🔥🔥): 
        
- **In Search of Continuous Text**: User `@eligump` inquired about how to make the "continue generating" prompt appear continuously. `@samwale_` advised them on adding specific instructions to the prompt to achieve this, such as "add during every pause in your response please resume immediately."

- **Navigating ChatGPT's Memory**: `@henike93` faced challenges with ChatGPT not retaining information as expected. `@darthgustav.` explained the issue could be due to a retrieval gap and suggested using more specific language in their instructions.

- **All About Attachment Perception**: `@madame_architect` shared successful prompting adjustments that improved GPT's interaction with file attachments. `@darthgustav.` recommended explicit commands like "Analyze the attached" for better results.

- **Contrastive CoT Prompting Discussed**: `@madame_architect` sought assistance in designing natural language prompts using Contrastive CoT (CCOT) prompting. `@darthgustav.` suggested avoiding negative examples and focusing on using conditions in the main prompt for better outcomes.

- **Lexideck Technologies Explored**: Conversation around `@darthgustav.`'s Lexideck Technologies revealed it as a modular, agent-based framework with potential future implications for agentic behavior in AI models. Its capability to adapt and prompt itself was highlighted.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral AI Office Hours Announced**: Scheduled office hours for **Mistral** will take place, with community members encouraged to join via this [Office Hour Event](https://discord.gg/mistralai?event=1196389318160306226) link.

- **Mistral on Azure & API Economics**: Technical discussions highlight that **Mistral runs on Sweden/Azure** as per [privacy policy](https://mistral.ai/privacy-policy/), and that its **API pricing** is competitive, charging based on the sum of prompt and completion tokens, detailed in the [API docs](https://docs.mistral.ai/api/#operation/createChatCompletion).

- **Finessing Fine-tuning for Mistral Models**: The community expresses frustration over the challenges and expense in fine-tuning **Mistral**'s 8x7B model, with experts attempting various techniques, including "clown car merging" referenced from [an academic paper](https://arxiv.org/abs/2306.01708), and a need for clearer guidance from Mistral noted.

- **Deployment Dilemmas**: Recommendations for Mistral deployments suggest that **API usage** fits non-intense usage, and [quantized versions of Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) may be effective for local runs, while hosting locally is needed for handling multiple parallel queries free from API rate limits.

- **Navigating Model and UI Implementations**: Users share solutions and challenges while implementing **Mistral AI** in various interfaces, including a UI adaptation ([mistral-ui](https://github.com/irony/mistral-ui)) and ways to configure API keys with environmental variables, highlighting practical implementation hurdles for engineers.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (75 messages🔥🔥): 
        
- **Scheduled Office Hours for Mistral**: `@sophiamyang` announced an upcoming office hour for Mistral community members, which can be attended through this link: [Office Hour Event](https://discord.gg/mistralai?event=1196389318160306226).
- **Inquiry About Unfiltered Chatbot Responses**: `@akali` questioned whether the chat completion API, such as mistral-tiny, can generate uncensored responses.
- **Affiliate Program Interest**:
  - `@swarrm777` expressed interest in a potential affiliate program for Mistral AI due to their French website that garners significant traffic discussing ChatGPT.
  - `@sophiamyang` responded to `@swarrm777` by asking for clarification on the function of the proposed affiliate program.
- **Hardware Requirements for Mistral AI**:
  - `@mrdragonfox` advised `@mrhalfinfinite` that running Mistral 7b is feasible on CPU, but using Mixtral requires a GPU with at least 24 GB VRAM.
  - For virtualization on Windows, `@mrdragonfox` recommended WSL2 over Hyper-V for `@mrhalfinfinite`.
- **Tokenization Clarifications**:
  - Discussions around token costs included tips on how to calculate the numbers of tokens using a Python snippet and the differences between tokens and words. `@i_am_dom` clarified that emojis can potentially equate to an approximate 30 tokens each.
- **Model Choice for Structured Data from Local DB**: `@refik0727` sought advice on selecting an LLM model for handling structured data sourced from a local database, to which `@sophiamyang` recommended Mistral.

**Links mentioned**:

- [Join the Mistral AI Discord Server!](https://discord.gg/mistralai?event=1196389318160306226): Check out the Mistral AI community on Discord - hang out with 9538 other members and enjoy free voice and text chat.
- [Byte-Pair Encoding tokenization - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Mistral AI API | Mistral AI Large Language Models](https://docs.mistral.ai/api/#operation/createChatCompletion))): Chat Completion and Embeddings APIs
- [llama.cpp/grammars/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md): Port of Facebook&#39;s LLaMA model in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [Huggingface AutoTokenizer cannot be referenced when importing Transformers](https://stackoverflow.com/questions/68481189/huggingface-autotokenizer-cannot-be-referenced-when-importing-transformers/68486285#68486285)): I am trying to import AutoTokenizer and AutoModelWithLMHead, but I am getting the following error:&#xA;ImportError: cannot import name &#x27;AutoTokenizer&#x27; from partially initialized module &#x27...


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (3 messages): 
        
- **Short and Sweet Approval**: `@sophiamyang` expressed that something (unspecified) **works pretty well**, although the context is not provided.
- **Robolicious about to take off**: `@robolicious` acknowledged the positive feedback with "[Yes it works pretty well]" and shared their excitement about starting, noting their experience is with other LLMs and inquiring about how it compares to GPT-4 for few-shot prompting.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **API vs Local Hosting**: `@vhariational` suggested that for non-intense usage, using the API is the easiest and most cost-effective method, but for local runs, they recommended [quantized versions of Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) with a trade-off in quality for infrastructure constraints.
- **Parallel Processing Needs Local Models**: `@richardclove` argued that despite the API's rate limit of 2 requests per second, hosting the model locally is beneficial for handling multiple parallel queries without such restrictions.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (31 messages🔥): 
        
- **Frustrations in Fine-tuning Mistral Models**: `@sensitron` is curious about the process and expected time for fine-tuning the 8x7B model while `@mrdragonfox` points out the difficulty and expense the community faces in approximating the original Mistral Instruct, with experts spending significant amounts without success.
- **The Quest for Mistral's Secret Sauce**: Both `@mrdragonfox` and `@canyon289` discuss the lack of clear guidance from Mistral on fine-tuning its models, with experts such as Eric Hardman ("dolphin") and Jon ("airoboros") trying to crack the code without official hints, leading to what `@mrdragonfox` calls "brute force" efforts.
- **Clown Car Merging - A Potential Method**: `@mrdragonfox` introduces the concept of "clown car merging," referencing [an academic paper](https://arxiv.org/abs/2306.01708) on model merging as a potential technique, and suggests that the community has not yet cracked the nuances of this method as it applies to the 8x7B model.
- **Misconceptions about MOE Models Clarified**: Clarifying `@sensitron`'s misunderstanding, `@mrdragonfox` explains that the 8x7B Mixture of Experts (MoE) model operates differently: the expertise is distributed across the model rather than being isolated in specific sections, serving primarily as an inference speed optimization rather than an expertise focusing mechanism.
- **Learning Resources for LLM Novices**: Newcomers like `@sensitron` seeking to understand and work with large language models are advised by `@mrdragonfox` to turn to YouTube content and academic papers to keep up with the fast-moving industry, given that even industry professionals find it challenging to stay informed.

**Links mentioned**:

[TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708): Transfer learning - i.e., further fine-tuning a pre-trained model on a downstream task - can confer significant advantages, including improved downstream performance, faster convergence, and better sa...


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (1 messages): 
        
- **Summarization Output Range Issues**: User `@ykshev` is seeking advice on how to make a model, **mistralai/Mixtral-8x7B-Instruct-v0.1**, produce outputs within a specific character range for a summarization task. They're incentivizing a solution with a **$200 tip** but express frustration that most outputs do not meet the expected length.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (74 messages🔥🔥): 
        
- **Mistral's Consumer-Facing Products Still Uncertain**: `@mercercl` expressed hope that Mistral might remain focused and not develop their own chatbot/assistant product. `@sublimatorniq` suggested a versatile model like OpenAI's GPT would be interesting for various applications.

- **Mistral Runs on Azure**: Users `@olivierdedieu` and `@sublimatorniq` discussed La Plateforme's cloud provider, with `@sublimatorniq` mentioning that Mistral uses **Sweden/Azure** as specified on the [privacy policy page](https://mistral.ai/privacy-policy/).

- **Mistral's API Pricing**: User `@vhariational` explained that **Mistral's API pricing** is based on the sum of prompt and completion tokens, with extensive [documentation](https://docs.mistral.ai/api/#operation/createChatCompletion) provided. The related `@akali` noted Mistral's competitive pricing compared to ChatGPT 3.5 Turbo API.

- **Third-Party UI Solutions for Mistral**: User `@clandgren` shared a UI adaptation for Mistral (https://github.com/irony/mistral-ui), originally designed for OpenAI, which functions well and is open source for community feedback and use. Addressed issues include setting `OPENAI_API_HOST` correctly and dealing with Docker environment variables.

- **Access to Mistral and API Key Configuration Challenges**: Users discussed how to gain access to Mistral AI, with `@fhnd_` querying about the waitlist process, while `@arduilex` and `.elekt` shared troubleshooting experiences with configuring Mistral API keys and environmental variables in a third-party UI, sometimes resulting in runtime errors and infinite loading issues.

**Links mentioned**:

- [Privacy Policy](https://mistral.ai/privacy-policy/): Frontier AI in your hands
- [Chatbot UI](https://mistral-ui.vercel.app)
- [HoloViz Blog - Build a Mixtral Chatbot with Panel](https://blog.holoviz.org/posts/mixtral/#build-a-panel-chatbot.): With Mistral API, Transformers, and llama.cpp
- [Mistral AI API | Mistral AI Large Language Models](https://docs.mistral.ai/api/#operation/createChatCompletion).): Chat Completion and Embeddings APIs
- [HuggingChat](https://huggingface.co/chat)
- [GitHub - irony/mistral-ui](https://github.com/irony/mistral-ui): Contribute to irony/mistral-ui development by creating an account on GitHub.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Pile v2 Remains a Mystery**: The existence of **The Pile v2** was debunked by `@stellaathena`, stating it as a work-in-progress and informing about a subset released by CarperAI. Meanwhile, **Minipile** was highlighted by `@giftedgummybee` as a cost-effective alternative for grad students, and a GitHub repository named [Awesome-Multilingual-LLM](https://github.com/y12uc231/Awesome-Multilingual-LLM/tree/main) was shared as a resourceful link for multilingual dataset information.

- **Innovation in Multilingual Model Training**: `@philpax` shared an article from [Tensoic Blog on Kannada LLAMA](https://www.tensoic.com/blog/kannada-llama/), while `@xylthixlm` discussed how models trained to forget their embeddings could be more adaptable to new languages, as described in an Arxiv paper on [Learning to Learn for Language Modeling](http://arxiv.org/abs/2307.01163).

- **Byte-Level Tokenization for LLMs Examined**: Discussions around fine-tuning LLMs for byte-level tokenization included a suggestion to re-use bytes embeddings from the original vocabulary, and the concept of activation beacons potentially improving byte-level LLMs' ability to self-tokenize was introduced.

- **Comparing Across Models and Seeking Codes**: `@jstephencorey` sought model suites like T5, OPT, Pythia, BLOOM, and Cerebras to assess embeddings for retrieval, prompting sharing of accessible codes and data publications, particularly for **BLOOM** and **T5**.

- **Handling GPT-NeoX Development Issues**: OOM errors occurring consistently at 150k training steps were resolved by `@micpie` using `skip_train_iteration_ranges`. A question regarding gradient storage in mixed-precision training referenced [Hugging Face's Model Training Anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy), and `@catboyslimmer` grappled with failing tests, casting doubt on the reliability of the test or system-specific issues.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (95 messages🔥🔥): 
        
- **New Dataset Release Speculation**: In response to `@lrudl`'s question about The Pile v2 release date, `@stellaathena` clarifies that Pile v2 is a work-in-progress and doesn't officially exist, although a subset is available from another direction by CarperAI.
- **Minipile as a Pile Alternative**: `@giftedgummybee` points out the existence of Minipile, a smaller version of the Pile dataset, which might fit the budget constraints of a grad student as mentioned by `@sk5544`.
- **Exploring Multilingual Datasets**: `@stellaathena` suggests datasets such as mT5, ROOTS, and multilingual RedPajamas for improving non-English generations of LLMs. `@sk5544` shares the [Awesome-Multilingual-LLM](https://github.com/y12uc231/Awesome-Multilingual-LLM/tree/main) GitHub repository as a resource for related papers.
- **CIFARnet Dataset Introduced**: `@norabelrose` shares a link to CIFARnet, a 64x64 resolution dataset extracted from ImageNet-21K which can be found on [Hugging Face datasets](https://huggingface.co/datasets/EleutherAI/cifarnet). The dataset is discussed in relation to label noise and possible experimental uses.
- **ImageNet Label Noise Discussed**: `@ad8e` and `@norabelrose` engage in a conversation about the labeling issues within ImageNet and the CIFARnet dataset, including the presence of grayscale images and potentially mislabeled items.


**Links mentioned**:

- [Re-labeling ImageNet: from Single to Multi-Labels, from Global to Localized Labels](https://arxiv.org/abs/2101.05022): ImageNet has been arguably the most popular image classification benchmark, but it is also the one with a significant level of label noise. Recent studies have shown that many samples contain multiple...
- [Know Your Data](https://knowyourdata-tfds.withgoogle.com/#dataset=cifar10&tab=ITEM&sort_ids_by=default_segment.cifar10.label.value&select=__none__)
- [CarperAI/pile-v2-small-filtered · Datasets at Hugging Face](https://huggingface.co/datasets/CarperAI/pile-v2-small-filtered)
- [HPLT](https://hplt-project.org/datasets/v1.2)
- [uonlp/CulturaX · Datasets at Hugging Face](https://huggingface.co/datasets/uonlp/CulturaX)
- [EleutherAI/cifarnet · Datasets at Hugging Face](https://huggingface.co/datasets/EleutherAI/cifarnet)
- [GitHub - y12uc231/Awesome-Multilingual-LLM: Repo with papers related to Multi-lingual LLMs](https://github.com/y12uc231/Awesome-Multilingual-LLM/tree/main): Repo with papers related to Multi-lingual LLMs. Contribute to y12uc231/Awesome-Multilingual-LLM development by creating an account on GitHub.


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (62 messages🔥🔥): 
        
- **Exploring Pretraining and Fine-Tuning for New Languages**: `@philpax` shared an article about a Continually LoRA PreTrained & FineTuned 7B Indic model, indicating its effectiveness ([Tensoic Blog on Kannada LLAMA](https://www.tensoic.com/blog/kannada-llama/)). `@xylthixlm` noted a paper suggesting that Language Models trained to "learn to learn" by periodically wiping the embedding table could be easier to fine-tune for another language ([Learning to Learn for Language Modeling](http://arxiv.org/abs/2307.01163)).

- **Causal vs Bidirectional Models in Transfer Learning**: `@grimsqueaker` posed a question about the comparative performance in transfer learning between causal models and bidirectional ones, especially for sub 1B sized models. `@.solux` suggested that causality provides substantial performance improvements for transformers, making an equal number of parameters not practically equivalent.

- **Fine-Tuning Language Models to Use Raw Bytes**: `@carsonpoole` inquired about the possibility of fine-tuning a model for byte-level tokenization, suggesting that transformer block representations might carry over during such a process. In follow-up discussions, `@the_sphinx` recommended to re-use the bytes embeddings from the original vocab when fine-tuning as bytes, to ease the process and avoid disastrous results.

- **Activation Beacons Could Alter Byte-Level LLM Potential**: `@carsonpoole` mentioned that the concept of activation beacons has influenced his view on the potential of byte-level Large Language Models (LLMs). `@xylthixlm` described activation beacons as allowing the model to tokenize itself by compressing multiple activations into one.

- **Comparing Embeddings Across Different Model Suites**: `@jstephencorey` queried for suites of models with a wide range of sizes to evaluate model embeddings for retrieval, noting that quality peaks differed between Pythia and OPT models. `@stellaathena` provided a list of model suites that meet the criteria, including T5, OPT, Pythia, BLOOM, and Cerebras, with `@catboyslimmer` expressing interest in accessible code and data for these models, to which `@stellaathena` responded that BLOOM and T5 have published runnable code and data.

**Links mentioned**:

- [Kannada LLAMA | Tensoic](https://www.tensoic.com/blog/kannada-llama/)
- [Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon](https://arxiv.org/abs/2401.03462): The utilization of long contexts poses a big challenge for large language models due to their limited context window length. Although the context window can be extended through fine-tuning, it will re...
- [Turing Complete Transformers: Two Transformers Are More Powerful...](https://openreview.net/forum?id=MGWsPGogLH): This paper presents Find+Replace transformers, a family of multi-transformer architectures that can provably do things no single transformer can, and which outperforms GPT-4 on several challenging...
- [Improving Language Plasticity via Pretraining with Active Forgetting](http://arxiv.org/abs/2307.01163): Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a ba...
- [The Unreasonable Effectiveness of Easy Training Data for Hard Tasks](https://arxiv.org/abs/2401.06751): How can we train models to perform well on hard test data when hard training data is by definition difficult to label correctly? This question has been termed the scalable oversight problem and has dr...
- [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796): Probabilistic weather forecasting is critical for decision-making in high-impact domains such as flood forecasting, energy system planning or transportation routing, where quantifying the uncertainty ...
- [HandRefiner: Refining Malformed Hands in Generated Images by Diffusion-based Conditional Inpainting](http://arxiv.org/abs/2311.17957): Diffusion models have achieved remarkable success in generating realistic images but suffer from generating accurate human hands, such as incorrect finger counts or irregular shapes. This difficulty a...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (4 messages): 
        
- **Seeking RLHF Interpretability Insights**: `@quilalove` inquired about any findings or insights from the **rlhf interpretability group**. They mentioned the context being a channel titled #rlhf-interp on the mechanistic interpretability discord.
- **Request for Context by @stellaathena**: In response to `@quilalove`'s query, `@stellaathena` asked for more context in order to provide relevant information regarding RLHF interpretability.
- **Clarification Provided by @quilalove**: After being prompted, `@quilalove` clarified their interest in any knowledge on the effects of RLHF experienced by the group in the #rlhf-interp channel.


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (16 messages🔥): 
        
- **Training Troubles**: User `@micpie` experienced an out-of-memory (OOM) error after 150k steps, consistently at the same step. They resolved the issue by using the `skip_train_iteration_ranges` feature, skipping more batches around the problematic step.

- **Understanding Gradient Precision**: `@afcruzs` raised a question about gradients always being stored in fp32 even when training with mixed precision, citing [Hugging Face's documentation](https://huggingface.co/docs/transformers/model_memory_anatomy). `@micpie` provided an EleutherAI guide explaining that gradients are computed in fp16 with the weight update being done in fp32, which is normal for mixed precision.

- **Tests Yielding Errors**: User `@catboyslimmer` has been encountering failing tests while running with pytest, with discrepancies depending on whether the `--forked` flag is used. They consider that the tests might be broken or there might be an issue specific to their system.

- **Exploring Train/Packing Resources**: `@cktalon` shared a link to [MeetKai's functionary GitHub repository](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing) for a chat language model that interprets and executes functions/plugins. `@butanium` thanked `@cktalon` and was encouraged to share any interesting findings.

**Links mentioned**:

- [Model training anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)
- [functionary/functionary/train/packing at main · MeetKai/functionary](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing): Chat language model that can interpret and execute functions/plugins - MeetKai/functionary
- [Jupyter Notebook Viewer](https://nbviewer-org.translate.goog/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/08_zero_redundancy_optimization.ipynb?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=de&_x_tr_pto=wapp#Mixed-Precision의-동작방식)


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Creative Bookmarking Strategies Explored**: `@api_1000` and `@dagbs` discussed bookmarking Discord posts with a potential solution involving creating a new server to store message links. Meanwhile, `@heyitsyorkie` mentioned the traditional copy/paste for offline backup, providing alternatives for resource management.

- **Challenges and Solutions in Dynamic Model Loading**: Users `@nyaker.` and `@nmnir_18598` reported issues with loading **Mixtral Q3** and image processing errors, respectively. Potential causes suggested by members like `@heyitsyorkie` and `@fabguy` include version incompatibility and clipboard errors, with remedies pointing towards updates and system checks.

- **Navigating Hardware Constraints for Advanced AI models**: Insights from users like `@heyitsyorkie` and `@pefortin` emphasized the heavy VRAM requirements of **Mixtral 8×7b** and potential bandwidth bottlenecks of mixed GPU setups. Discussions included advice on tensor splitting and monitoring proper GPU utilization for model operations.

- **Local Model Optimizations for Creative Writing**: Recommendations for using **OpenHermes** and **dolphin mixtral models** were offered for fiction worldbuilding tasks, with community members guiding on optimizing GPU settings. Utility tools like **World Info** from [SillyTavern](https://docs.sillytavern.app/usage/core-concepts/worldinfo/) were shared to enhance the AI's understanding of narrative details.

- **Feature Requests and Humor in Feedback**: The feedback section saw a tongue-in-cheek remark by `@fabguy`, suggesting that a bug could be considered a feature, and a user-driven request by `@blackflagmarine` to improve the search capabilities of the **LLM search** with a ***contains*** function, aimed at enhancing user experience.

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (77 messages🔥🔥): 
        
- **DIY Bookmarking Tips**: `@api_1000` got creative advice from `@dagbs` on bookmarking useful Discord posts by creating a new server and pasting message links there. Alongside `@heyitsyorkie`, who also suggested the traditional copy/paste method for offline backups.
- **Model Loading Troubles**: `@nyaker.` voiced their inability to load Mixtral Q3 with or without GPU acceleration and received input from `@heyitsyorkie` and `@fabguy`, suggesting version incompatibility and available system resources as potential issues. They recommended upgrading to later versions and checking system requirements.
- **Mysterious Vision Error**: `@nmnir_18598` encountered an error with image processing in the chat window, which `@heyitsyorkie` linked to clipboard content. The issue was resolved by `@fabguy` who recommended starting a new chat and advised on potentially editing the JSON file to remove the erroneous content.
- **Installation Assistance**: Newcomers like `@duncan7822` and `@faradomus_74930` inquired about installing LM Studio on Ubuntu Linux, and `@heyitsyorkie` provided guidance, including the necessary condition of having an updated glibc for compatibility on Ubuntu 22.
- **Feature Functionality and Resource FAQs**: `@meadyfricked` sought help regarding function calling with autogen, prompting responses from `@heyitsyorkie` and `@dagbs` on current limitations and workarounds. Additionally, `@heyitsyorkie` posted a link to an unofficial LMStudio FAQ for community reference.

**Links mentioned**:

- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [GitHub - microsoft/lida: Automatic Generation of Visualizations and Infographics using Large Language Models](https://github.com/microsoft/lida): Automatic Generation of Visualizations and Infographics using Large Language Models - GitHub - microsoft/lida: Automatic Generation of Visualizations and Infographics using Large Language Models


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (59 messages🔥🔥): 
        
- **LM S Struggles with Newer gguf Models**: `@coolbreezerandy6969` experienced issues loading newer gguf models with LM S (Linux LM+Studio), clarified by `@fabguy` who explained that new architectures like Mixtral require updates, and version 0.2.10 might resolve these issues.

- **Mixtral Confined to Local Use**: `@pinso` asked about TheBloke's dolphin-2.5-mixtral-8x7b-GGUF model having internet search capabilities, which `@heyitsyorkie` refuted, confirming that LMStudio does not support function calling for web searches.

- **Hefty VRAM Required for Mixtral 8×7b**: `@heyitsyorkie` mentioned that running Mixtral 8×7b at q8 requires 52 GBs of VRAM. Consequently, `@madhur_11` noted poor performance with just 16 GB of RAM on a laptop, to which `@heyitsyorkie` responded that LM Studio's system for Mixtral models carries bugs.

- **Understanding VRAM and Shared GPU Memory**: A conversation between `@nikoloz3863` and `@heyitsyorkie` helped clarify that VRAM is dedicated memory on the graphics card, while shared GPU memory includes a combination of VRAM and CPU RAM.

- **Recommendations for Local Models Aiding Fiction Writing**: `@rlewisfr` sought model recommendations for worldbuilding and was directed by `@ptable` to try OpenHermes and dolphin mixtral models. Further discussions led to `@heyitsyorkie` offering advice on optimizing GPU layer settings and referencing SillyTavern for leveraging World Info for interactive story generation.

**Links mentioned**:

- [dagbs/laserxtral-GGUF · Hugging Face](https://huggingface.co/dagbs/laserxtral-GGUF)
- [liminerity/Blur-7B-slerp-v0.1 · Hugging Face](https://huggingface.co/liminerity/Blur-7B-slerp-v0.1)
- [World Info | docs.ST.app](https://docs.sillytavern.app/usage/core-concepts/worldinfo/): World Info (also known as Lorebooks or Memory Books) enhances AI's understanding of the details in your world.
- [222gate/Blur-7B-slerp-v0.1-q-8-gguf · Hugging Face](https://huggingface.co/222gate/Blur-7B-slerp-v0.1-q-8-gguf)


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (2 messages): 
        
- **The Feature Debate**: User `@fabguy` humorously commented that an aspect of the chatbot, which might be perceived negatively, should be considered a **feature, not a bug**.
- **Search Enhancement Request**: `@blackflagmarine` requested an addition of a ***contains*** functionality to the **LLM search** to improve the search capabilities.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (6 messages): 
        
- **Franken-PC Experiments Reveal Bandwidth Bottlenecks**: In his franken-PC setup with mixed GPUs, `@pefortin` shared some [experimental performance results](https://discord.link.to.experiment) with **Mixtral 8x7B** and different configurations. The combination of a **3090** with a **3060ti** led to the best performance at *1.7 tokens/second*, while adding slower GPUs and PCIe lanes decreased throughput.

- **Tensor Split Needs Investigation**: `@dagbs` suggested testing the tensor split performance with a **3060ti versus 2x 1660**, hinting at possible issues with tensorsplit's workings. `@pefortin` responded, clarifying that the model layers were proportionally split not evenly distributed, implying the splitting mechanism functioned with the **GGUF and llamacpp** framework.

- **Exploring GPTQ/exl2 for Possible Performance Gains**: `@pefortin` mentioned plans to conduct tests using **GPTQ/exl2** formats to see if they alter performance outcomes in the model setup.

- **Sanity Check for Model Splits via GPU Monitoring Advised**: `@ben.com` recommended monitoring the "copy" graph in Task Manager's GPU tab to ensure there are no hidden inefficiencies during model splits. `@pefortin` assured he keeps an eye on GPU memory usage and compute activity, confirming all looked normal.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Merge Dilemma and the Quest for the Right Model**: Engineers discussed dataset merging strategies, where `_michaelsh` brought up a query about combining **85 GB of audio samples** with **186 MB of associated texts**. The conversation pivoted to the best **Large Language Model (LLM)** for a local database, considering models like Mistral, Llama, Tapas, and Tapex, with `refik0727` spearheading the discussion.

- **Tackling Env Issues and Enhancing Chatbots**: There was an exchange on resolving environment-related errors in model packaging, specifically concerning `package_to_hub` functionality with a non-gymnasium environment, as articulated by `boi2324` and `doctorpangloss`. Additionally, strategies to improve **chatbot responses using TinyLLaMa** were discussed, proposing an array-based structuring of user/assistant messages to guide model comprehension.

- **Learning and Adaptation in AI**: `bluebug` shared accomplishments of labeling over **6k datasets** and creating a new **image-to-text labeler tool**. Insights into **MiniLLM**, a method for distilling LLMs developed by Microsoft, were highlighted, featuring reinforcement learning language techniques for efficiently running LLMs on consumer-grade GPUs.

- **Tools and Papers Unveiled**: The community brought to light academic resources linking **Transformers to RNNs** and a GitHub repository named **UniversalModels** designed to act as an adapter between Hugging Face transformers and different APIs.

- **Innovations and Implementations in AI Showcased**: Creations spanned from **Midwit Studio**, an AI-driven text-to-video generator, to articles detailing **Stable Diffusion's** inner workings. New models like **e5mistral7B** were introduced, and tools like a fast-paced data annotation tool and **Dhali**, a platform for monetizing APIs, were demonstrated.

- **Image Editing Advances and Issue Management**: `sayakpaul` encouraged an issue thread for clarification accompanied by a reproducible snippet and presented **Emu Edit**, a multi-task oriented image editing tool, distinct from standard inpainting thanks to its task-specific approach.

- **AI Mimicry and Human-Like NPCs**: An **AI agent** that interacts with **ChatGPT-4v** for block manipulation tasks to achieve human-like behavior was shared by `harsh_xx_tec_87517`, indicating potential applications in **NPC behavior** with a demonstrated process shared through LinkedIn.

- **Model Insights and NER Efficiency**: Parameter count strategies using **safetensors model files** and Python functions were debated, leading to a confirmed utility of a parameter estimation function across models like Mistral, LLaMA, and yi-34. An innovative lasso selector tool boasted the ability to label 100 entities in 2 seconds, and model embeddings within LLMs were discussed, focusing on tokenizer origins and training methodologies.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (62 messages🔥🔥): 
        
- **Merge Strategy Mysteries**: User `@_michaelsh` asked about the best method to merge two large datasets, one consisting of 85 GB of audio samples and the other of 186 MB of associated texts. `@moizmoizmoizmoiz` requested further details to provide an accurate suggestion.
- **Choosing the Right LLM for Local Database**: `@refik0727` inquired about the most suitable Large Language Model (LLM) for structured data from a local database, considering models like Mistral, Llama, Tapas, and Tapex.
- **Gym vs Gymnasium Environment for Model Packaging**: `@boi2324` encountered an error when attempting to use `package_to_hub` with a non-gymnasium environment, discussing this with `@doctorpangloss` who ultimately recommended using environments supported by Hugging Face to avoid major issues.
- **Improving Chatbot Responses**: `@mastermindfill` discussed optimizing chatbot responses using TinyLLaMa after observing suboptimal output. `@cappuch__` advised appending messages to an array with a user/assistant format and using username prompts to direct model comprehension.
- **Concerns Over Model Safety Labels**: `.ehsan_lol` expressed confusion about models being labeled as "unsafe" on Hugging Face, with a specific interest in understanding why this might be for the purpose of downloading the model.

**Links mentioned**:

[Bingsu/adetailer at main](https://huggingface.co/Bingsu/adetailer/tree/main)


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **Productivity Unleashed with Custom Tool**: `@bluebug` has successfully labeled a significant amount of data, boasting about having labeled over **6k datasets**.
- **Homebrew Image to Text Tool Completion**: Created by `@bluebug`, a new **image to text labeler** tool has been completed to assist with data labeling tasks.
- **Discovering Mini LLM - A Leap in LLM Distillation**: `@frequesny` learned about **MiniLLM**, a state-of-the-art method developed by Microsoft for distilling large language models (LLMs) using reinforcement learning language. The method boasts impressive results in comparison to existing baselines, and `@frequesny` shared the GitHub repository: [MiniLLM on GitHub](https://github.com/kuleshov/minillm).

**Links mentioned**:

[GitHub - kuleshov/minillm: MiniLLM is a minimal system for running modern LLMs on consumer-grade GPUs](https://github.com/kuleshov/minillm): MiniLLM is a minimal system for running modern LLMs on consumer-grade GPUs - GitHub - kuleshov/minillm: MiniLLM is a minimal system for running modern LLMs on consumer-grade GPUs


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (2 messages): 
        
- **"Transformers meet RNNs" Paper Shared**: User `@doodishla` shared an academic paper linking **Transformers to RNNs**, which can be found on [arXiv](https://arxiv.org/pdf/2401.06104.pdf).
- **Universal Adapters for Transformers**: `@andysingal` found a nice GitHub repository named **UniversalModels** which acts as an adapter between HuggingFace transformers and several different APIs, available at [GitHub - matthew-pisano/UniversalModels](https://github.com/matthew-pisano/UniversalModels).

**Links mentioned**:

[GitHub - matthew-pisano/UniversalModels: An adapter between Huggingface transformers and several different APIs](https://github.com/matthew-pisano/UniversalModels): An adapter between Huggingface transformers and several different APIs - GitHub - matthew-pisano/UniversalModels: An adapter between Huggingface transformers and several different APIs


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (13 messages🔥): 
        
- **Introducing Midwit Studio**: User `@ajobi882` shared a link to **Midwit Studio**, an AI-driven text-to-video generator designed for simplification, teasingly suggested for "midwits". Check it out here: [Midwit Studio](https://midwitstudio.com).
- **Diving Deep into Stable Diffusion**: `@felixsanz` published a detailed two-part article series on **Stable Diffusion**: The first explains its working without code, while the second part tackles implementation with Python. Read about it [here](https://www.felixsanz.dev/articles/how-to-implement-stable-diffusion).
- **Tonic Spotlights E5 Mistral**: `@tonic_1` announces the availability of **e5mistral7B** on GPUZero and describes it as a new Mistral model with merged embeddings capable of creating embeddings from the right prompts. Explore the model on [HuggingFace Spaces](https://huggingface.co/spaces/Tonic/e5).
- **Speedy Data Annotation Tool**: `@stroggoz` introduces an alpha-stage data annotation tool for NER/text classification, boasting the ability to label around 100 entities every 2 seconds. The tool's preview is available [here](https://gyazo.com/29c6e3487eedca343c0a31e3d255761a).
- **Monetize APIs with Dhali**: `@dsimmo` presents **Dhali**, a platform that allows users to monetize their APIs within minutes, using a Web3 API Gateway and offering low overhead and high throughput without the need for subscriptions. For more details, visit [Dhali](https://dhali.io).

**Links mentioned**:

- [Gyazo Screen Video](https://gyazo.com/29c6e3487eedca343c0a31e3d255761a):  
- [Midwit Video Studio](https://midwitstudio.com)
- [Dhali](https://dhali.io)
- [E5 - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/e5)
- [How to implement Stable Diffusion](https://www.felixsanz.dev/articles/how-to-implement-stable-diffusion): After seeing how Stable Diffusion works theoretically, now it's time to implement it in Python
- [How Stable Diffusion works](https://www.felixsanz.dev/articles/how-stable-diffusion-works): Understand in a simple way how Stable Diffusion transforms a few words into a spectacular image.


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 messages): 
        
annorita_anna: I would love to see this happen too!🤍


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **Invitation to Create Issue Thread**: `@sayakpaul` encourages the opening of an issue thread for further discussion and clarifies the need for a reproducible snippet. A specific user is cc'ed for visibility.

- **Emu Edit's Approach to Image Editing**: `@sayakpaul` differentiates **Emu Edit**, an image editing model, from inpainting by highlighting its multi-tasking ability across a range of editing tasks. He provides a brief explanation and a link to [Emu Edit](https://emu-edit.metademolab.com/) for further information.

- **Assurance on Issue Logging**: In response to a link posted by `@felixsanz`, `@sayakpaul` agrees that even if it's not a bug, having the issue logged is helpful.

- **Clarification on "Not a Bug"**: `@felixsanz` clarifies that the prior issue under discussion is not a bug.

**Links mentioned**:

[Emu Edit](https://emu-edit.metademolab.com/): Precise Image Editing via Recognition and Generation Tasks


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **AI Agent Mimics Human Task Management**: `@harsh_xx_tec_87517` developed an **AI agent** that captures screenshots and interacts with **ChatGPT-4v** for block manipulation tasks, iterating this process until a specific state is reached. The agent aims to replicate human-like behavior for potential future use in **NPCs** and a [video demonstration and LinkedIn post](https://www.linkedin.com/posts/harsh-nigam-096b67133_i-built-an-ai-agent-that-looks-at-the-screen-activity-7152894742886817792-obl1) provide additional insights.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (12 messages🔥): 
        
- **Parameter Count Without Model Download**: `@robert1` inquired about obtaining the parameter count of a model without downloading it. `@vipitis` responded that the parameter count can be seen if there is a **safetensors model file** on the model page.

- **Estimating Parameter Count from `config.json`**: `@robert1` mentioned the possibility of writing a function to calculate parameter count using `config.json`, and `@vipitis` noted that would require in-depth knowledge of the model's hyperparameters.

- **LLaMA Model Python Function Shared**: `@robert1` shared a Python function `_get_llama_model_parameter_count` that calculates the parameter count for LLaMA-based models using information from the `config.json`.

- **Utility of Parameter Count Function Confirmed**: `@robert1` confirmed that the provided Python function correctly estimates the parameter count across various models like Mistral, LLaMA, and yi-34 after testing.

- **Innovative Lasso Selector for NER**: `@stroggoz` shared a [7-second gif](https://gyazo.com/29c6e3487eedca343c0a31e3d255761a) demonstrating a lasso selector tool that can be used to label 100 named entities or spans in just 2 seconds.

- **Embedding Models in LLMs Discussed**: `@pix_` asked about the type of embedding used in large language models (LLMs) with positional encoding. `@stroggoz` clarified that embeddings typically derive from a tokenizer and pre-trained transformer base architecture, with random initialization being a possibility for training from scratch.

**Links mentioned**:

[Gyazo Screen Video](https://gyazo.com/29c6e3487eedca343c0a31e3d255761a):


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **Invitation to Open an Issue Thread**: `@sayakpaul` encouraged users to open an issue thread and include a reproducible snippet for discussion, tagging another user with `Cc: <@961114522175819847>`.
- **Emu Edit Demonstrates Inpainting Capabilities**: `@sayakpaul` shared a link to [Emu Edit](https://emu-edit.metademolab.com/) and described its distinct approach to image editing, which involves multi-task training and learned task embeddings to steer generation processes.
- **Inpainting Requires Binary Mask**: In the context of discussing image editing techniques, `@sayakpaul` noted that inpainting, unlike other methods, requires a binary mask to indicate which pixels in an image should be modified.
- **Clarification That an Issue Is Not a Bug**: `@felixsanz` stated that although there's a situation at hand, it does not constitute a bug. This was followed by a reassurance from `@sayakpaul` that logging the issue would still be beneficial.

**Links mentioned**:

[Emu Edit](https://emu-edit.metademolab.com/): Precise Image Editing via Recognition and Generation Tasks


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Mistral Medium Disconnection Drama**: Users `@moyaoasis` and `@me.lk` highlighted an issue with **Mistral Medium** disconnecting in **MS Edge**. The problem was recognized and noted for a fix as per a prior community [message](https://discord.com/channels/1047197230748151888/1131087959026835457/1196217734028071032).

- **Voice for Bots in Question**: Curiosity arose about voice conversation features for chatbots as user `@financers` inquired about such capabilities in Perplexity, resembling those in ChatGPT. Though uncertain about Perplexity's adoption of the feature, user `@mares1317` suggested [pi.ai/talk](https://pi.ai/talk) as an alternative for voice interaction.

- **Exploring PPLX API's Potential**: Discussion occurred about the new **pplx-api**, particularly about whether it could include source links in responses. A [blog post](https://blog.perplexity.ai/blog/introducing-pplx-api) shared by `@mares1317` described the API's features, indicating a future capability for fact and citation grounding.

- **Pro Member Plunges Into Perplexity**: Newly minted Pro member `@q7xc` is delving into the features and benefits of the platform, as mentioned in the `#sharing` channel.

- **pplx-7b-online Model Suffers Setback**: User `@yueryuer` reported experiencing a **500 internal server error** while using the `pplx-7b-online` model, raising concerns about server stability at the time of the incident.

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (88 messages🔥🔥): 
        
- **Mistral Medium Disconnection Issue Raised**: User `@moyaoasis` reported experiencing problems with Mistral Medium disconnecting while other models worked fine after switching to MS Edge from Brave. The issue was confirmed by `@me.lk` as known and to be fixed as indicated in a community [message](https://discord.com/channels/1047197230748151888/1131087959026835457/1196217734028071032).

- **Curiosity About Voice Features for Chatbots**: `@financers` inquired if Perplexity would implement voice conversation features like ChatGPT. `@mares1317` doubted Perplexity would adopt that feature but suggested a third-party alternative, [pi.ai/talk](https://pi.ai/talk), for vocal interaction.

- **PPLX API Introduction and Limitations**: Users `@d1ceugene` and `@mares1317` discussed the new pplx-api, with questions regarding its capability to provide source links in responses. `@mares1317` shared a [blog post](https://blog.perplexity.ai/blog/introducing-pplx-api), detailing the API features and hinting at future support for fact and citation grounding with Perplexity RAG-LLM API.

- **Perplexity Access and Performance Issues**: Several users including `@louis030195`, `@zoka.16`, and `@nathanjliu` encountered issues with the API, app responsiveness, and logins across various devices. `@mares1317` and `@ok.alex` responded with troubleshooting suggestions, and `@icelavaman` later confirmed that Perplexity should be working again.

- **App Login and Account Migration Queries**: Users `@.mergesort` and `@leshmeat.` sought assistance with account login issues, specifically related to Apple account migration and lost email access. `@ok.alex` and `@me.lk` responded with possible login steps and support contact for subscription transfers, but no history transfer was confirmed.

**Links mentioned**:

- [Anime Star GIF - Anime Star - Discover &amp; Share GIFs](https://tenor.com/view/anime-star-gif-20269661): Click to view the GIF
- [Moon (Dark Mode)](https://docs.perplexity.ai)
- [Introducing pplx-api ](https://blog.perplexity.ai/blog/introducing-pplx-api): Perplexity Lab's fast and efficient API for open-source LLMs


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (4 messages): 
        
- **Perplexity Android Widget Now Available**: User `@mares1317` shared a tweet from `@AravSrinivas` announcing the release of a **widget** for Perplexity Android users. The tweet, [Perplexity Android Users: Thanks for waiting patiently for the widget! Enjoy!](https://x.com/AravSrinivas/status/1746760200550539759?s=20), expresses gratitude for users' patience.

- **Channel Etiquette Reminder for Project Sharing**: `@ok.alex` reminded `<@935643161504653363>` to share project-related content in the specific channel for such posts, directing them to `<#1059504969386037258>`.

- **New User Praises Perplexity**: `@pablogonmo` joined the chat to share their initial positive impressions, calling Perplexity a "very solid alternative."

- **Pro Membership Exploration**: New Pro member `@q7xc` mentioned they are in the process of figuring out the platform.

**Links mentioned**:

[Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1746760200550539759?s=20): Perplexity Android Users: Thanks for waiting patiently for the widget! Enjoy!


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (4 messages): 
        
- **Model Misclassifies Companies**: `@eggless.omelette` reported issues with a model classifying companies into specific categories, receiving responses that included a repetition of the company name, a verbose Google-like search result, or a message stating no results found.
- **Intriguing 'related' Model Mentioned**: `@dawn.dusk` hinted at the existence of a "related" model, expressing curiosity and seeking confirmation by tagging `<@830126989687914527>`.
- **Server Error Hurdles for pplx-7b-online Model**: `@yueryuer` encountered a **500 internal server error** when calling the API with the `pplx-7b-online` model, questioning the stability of the server at that time.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Axolotl Adventures in DPO**: `@c.gato` expressed **gratitude** for the ease of utilizing **Axolotl's Dynamic Performance Optimizer (DPO)**, calling the experience immensely *FUN*. `@casper_ai` and `@xzuyn` provided advice on creating DPO datasets which consist of **chosen/rejected pairs,** confirming that these are designed differently than SFT datasets based on the desired model behavior.

- **RLHF Update is Imminent**: An update regarding **Reinforcement Learning from Human Feedback (RLHF)** is to be shared **soon**, as teased by `@caseus_`.

- **Empowering the Dataset Formats**: **Hugging Face MessagesList** format is being considered for chat message formatting, as discussed by `@dctanner`. To align with this effort, Axolotl **Pull Request #1061** will have updates to support this new 'messageslist' format, as proposed in the [Hugging Face Post](https://huggingface.co/posts/dctanner/975913831192894).

- **Optimization Talk around Model Packing**: An interest has been shown in the optimized solution for model packing from [MeetKai functionary](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing), with focus on efficiency and potential implementation in a collator.

- **Technicalities and Troubleshootings in Bot Land**: `@mrfakename_` highlighted potential downtime of a bot after it failed to respond to prompts, `@noobmaster29` confirmed the online status but shared similar unresponsiveness concerns. In **runpod-help**, `@baptiste_co` successfully installed `mpi4py` using **Conda**, while `@tnzk` encountered a `RuntimeError` after installation, suggesting a possible bug report to **PyTorch**.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (16 messages🔥): 
        
- **The Joy of Axolotl DPO**: `@c.gato` expressed excitement and **gratitude** for the ease of running DPO jobs with **Axolotl**, having immense *FUN* in the process.
- **Upcoming RLHF News Teaser**: `@caseus_` hinted that updates regarding **RLHF** will be shared **soon**.
- **Details on Training Phases Clarified**: In a discussion about training methods, `@caseus_` and `@casper_ai` clarified that SFT should be done first, followed by **DPO**. `@dangfutures` engaged in the conversation seeking clarity on the process.
- **Guidance on DPO Dataset Creation**: `@casper_ai` and `@xzuyn` advised `@dangfutures` that DPO datasets typically consist of chosen/rejected pairs and are designed based on **desired model behavior**, which can be quite different from SFT datasets.
- **Inquiry About Continual Pretraining**: `@jinwon_k` questioned the success of continual pretraining with **Axolotl**, to which `@nanobitz` responded confirming successful usage, although it's been a while since implemented.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (31 messages🔥): 
        
- **Converging on a Chat Dataset Standard**: `@dctanner` discussed formalizing chat message formats and introduced the **Hugging Face MessagesList** format as a clean and simple structure. The [Hugging Face Post](https://huggingface.co/posts/dctanner/975913831192894) explains the proposed standard.
- **Refining Axolotl PRs for Dataset Formats**: `@dctanner` intends to update a [Pull Request #1061](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1061) to support the newly suggested 'messageslist' format, moving away from overloading the sharegpt format.
- **DPO Templates Need Global System Prompts**: `@dctanner` suggested adding support for a global system prompt in DPO templates, citing an ongoing [Pull Request #935](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935/files) and questioning why `apply_chat_template` isn't used in DPO as [alignment-handbook does](https://github.com/huggingface/alignment-handbook/blob/c74ed111710d57f563cfbf1806cfb8f07dd3dc67/src/alignment/data.py#L55).
- **Issue with Incorrect Token Generation Post-DPO**: `@caseus_`, `@dctanner`, and `@teknium` discussed a baffling issue where models generate `im_start` and `im_end` tokens incorrectly, leading to endless responses, with `@teknium` noting they had to regenerate multiple times to prompt this error.
- **Functionary's Approach to Model Packing**: `@le_mess` shared a potential optimized solution for packing models from [MeetKai functionary](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing), with `@casper_ai` expressing interest in the packing efficiency and `@caseus_` considering implementation in a collator.

**Links mentioned**:

- [@dctanner on Hugging Face: &quot;As the amount of datasets for fine tuning chat models has grown, there&#39;s been…&quot;](https://huggingface.co/posts/dctanner/975913831192894)
- [alignment-handbook/src/alignment/data.py at c74ed111710d57f563cfbf1806cfb8f07dd3dc67 · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/blob/c74ed111710d57f563cfbf1806cfb8f07dd3dc67/src/alignment/data.py#L55): Robust recipes for to align language models with human and AI preferences - huggingface/alignment-handbook
- [functionary/functionary/train/packing at main · MeetKai/functionary](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing): Chat language model that can interpret and execute functions/plugins - MeetKai/functionary
- [Add support to sharegpt strict: false for more formats by dctanner · Pull Request #1061 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1061): Expanding on the strict option for sharegpt format, I&#39;ve added support to sharegpt strict: false for more formats like those used in HuggingFaceH4/no_robots.
- [[WIP] RL/DPO by winglian · Pull Request #935 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935/files)


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (15 messages🔥): 
        
- **Yi 34b Finetuning Clarifications**: User `@c.gato` inquired about finetuning Yi 34b models, specifically about the differences between a normal version and a 200k model. `@nanobitz` clarified that the 200k model can be used as-is, since its model configuration handles the context.

- **Understanding Yi 34b's Max Context**: `@c.gato` needed confirmation on setting max context in the yml for the 200k model and was reassured by `@nanobitz` that setting `max_seq_len` should suffice to get started.

- **DPO Scheduling Quirks**: `@c.gato` reported issues with setting cosine and constant learning rate schedules in the Dynamic Performance Optimizer (DPO), speculating that its beta status might be the reason for the settings being ignored.

- **Request for Axolotl Config YML**: `@thinking_butterfly` sought the configuration `.yml` or hyperparameters for Open-Orca/Mistral-7B-SlimOrca. `@xzuyn` shared a link to the config for a related model, Mistral-7B-OpenOrca, but acknowledged the mix-up regarding the specific request for SlimOrca settings.


### ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/) (4 messages): 
        
- **Testing Bot Responsiveness**: User `@mrfakename_` pinged `@1163482975883772027` with a test message but received no reply.
- **Agent Search Functionality Questioned**: Following a lack of response, `@mrfakename_` asked if the agent search was down. `@noobmaster29` responded, indicating that it seems online but is not responding as expected.
- **Bot Might Be Down**: `@mrfakename_` suggested that the bot could be down due to the unresponsiveness.


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (3 messages): 
        
- **Conda Solves mpi4py Installation Issue**: `@baptiste_co` encountered a problem but resolved it by using Conda to install `mpi4py`: `conda install --name py3.10 mpi4py`.
- **Consultation on Runpod Image Setup**: `@caseus_` inquired whether `mpi4py` should be a standard installation on runpod/cloud images, considering `@baptiste_co`'s success with it.
- **RuntimeError After Installing mpi4py**: `@tnzk` followed the installation advice for `mpi4py` but encountered a `RuntimeError` related to PyTorch's grad accumulator, prompting a suggestion to report the bug to PyTorch.


        

---

## [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) Discord Summary

**LLMs Query Tables with Style**: A new paper showcasing **Language Models**' abilities to query tabular data using **textual and symbolic reasoning** was highlighted, indicating the current state and potential of LLMs in this domain. Details and discussions can be found at [this link](https://t.co/b36ufH9YMi) and an accompanying image is available [here](https://t.co/XyrJh5vSUq).

**Vector Search Goes Multi-Tenant**: The complexities of implementing **multi-tenancy** in vector search, particularly in the context of private data and retrieval-augmented generation applications, was dissected in a recent blog post. Insights and full content, as well as a visual aid, are available [here](https://t.co/jsGipOyauq) and [here](https://t.co/0yGIXfC1XJ), respectively.

**Collaborate on LlamaIndex Publishings**: *LlamaIndex blog* openings for authors was a hot topic, with members discussing who to contact and how to get involved; **@493606302971592747** was mentioned as a key contact. For those interested, an informative **compatibility report** to aid in selecting the appropriate LLM for local datasets was shared, [LlamaIndex compatibility report link](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#open-source-llms).

**Data Storage Choices Clarified**: LlamaIndex's data storage policy was clarified wherein data embedding and responses default through OpenAI, but storage is user's choice as no dedicated cloud is offered. Additionally, role assignment in GPT mimicking OpenAI's capabilities was touched upon, with **SimpleChatEngine** documentation provided for guidance.

**AI Propels Dynamic Databases and Data Querying**: Enthusiasm was shown for a **Chain-of-Table** framework aimed at enhancing data interpretation through LlamaIndex, explained in detail in a [Medium article](https://medium.com/technology-hits/harmony-unleashed-llamaindexs-guided-symphony-with-chain-of-table-d866247a72d2). A Twitter post introduced the **fluid database** concept meant for AI agents that dynamically updates its schema, further information is available on [GitHub](https://github.com/TheMind-AI/fluid-db). Querying capabilities integrating tables with LlamaIndex's technology was also discussed, with an illustrative [Medium article](https://medium.com/ai-advances/unlocking-insights-harnessing-table-extraction-and-advanced-data-querying-with-llamaindexs-pandas-f7200ef07771) on the procedure.

**LlamaIndex Discord Channel Summaries**

### ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/) (2 messages): 
        
- **LLMs Finessing Tabular Data**: A new paper explores the use of **textual and symbolic reasoning** with Language Model-based systems for querying tabular data, revealing strengths and weaknesses of each method. The tweet links to further discussions and paper details at [https://t.co/b36ufH9YMi](https://t.co/b36ufH9YMi) and includes an illustrative image at [https://t.co/XyrJh5vSUq](https://t.co/XyrJh5vSUq).
- **Multi-Tenancy Challenges in Vector Search**: The latest blog post tackles the challenges of **multi-tenancy** in retrieval-augmented generation applications, focusing on private data storage and vector search benefits. Additional insights and complete blog content are available at [https://t.co/jsGipOyauq](https://t.co/jsGipOyauq), accompanied by a visual snippet at [https://t.co/0yGIXfC1XJ](https://t.co/0yGIXfC1XJ).


### ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/) (48 messages🔥): 
        
- **LlamaIndex Writers, Assemble!**: User `@mouhannad1` is writing a Medium article series about LlamaIndex and inquires about publishing on the LlamaIndex blog. `@whitefang_jr` advises `@493606302971592747` as the go-to contact for this endeavor.
  
- **Choosing the Right LLM for Local Deployment**: `@refik0727` seeks advice on choosing the right LLM model for using a structured local DB dataset. `@whitefang_jr` provides a helpful [LlamaIndex compatibility report link](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#open-source-llms) to assist in selecting the most suitable LLM.

- **Storing LlamaIndex Data - A Clarification**: `@dp9075` asks whether LlamaIndex data is stored on a personal or LlamaIndex cloud. `@cheesyfishes` clarifies that LlamaIndex does not have its cloud, so data storage is at the user's discretion, but notes that by default, data traverses OpenAI for embeddings and responses.

- **LLM Lingua's Impressive Performance in Summarization**: `.assets.` shares a success story about implementing LLM Lingua in their pipeline, specifically citing significant speed improvements while maintaining quality. `@cheesyfishes` inquires about evaluation methods, and `.assets.` describes a practical approach using known-answer questions to assess performance.

- **Role Play with LlamaIndex**: `@pansocrates` inquires about the possibility of adding roles to GPT without modifying the query, similar to OpenAI. `@desk_and_chair` responds with a guide, referring to the documentation for [SimpleChatEngine](https://docs.llamaindex.ai/en/stable/api_reference/query/chat_engines/simple_chat_engine.html#llama_index.chat_engine.simple.SimpleChatEngine.chat_history) in LlamaIndex.

**Links mentioned**:

- [Llama Hub](https://llamahub.ai/?tab=llama_datasets)
- [Using LLMs - LlamaIndex 🦙 0.9.31](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#open-source-llms)
- [Simple Chat Engine - LlamaIndex 🦙 0.9.31](https://docs.llamaindex.ai/en/stable/api_reference/query/chat_engines/simple_chat_engine.html#llama_index.chat_engine.simple.SimpleChatEngine.chat_history)
- [Discover LlamaIndex: Ask Complex Queries over Multiple Documents](https://www.youtube.com/watch?v=GT_Lsj3xj1o): In this video, we show how to ask complex comparison queries over multiple documents with LlamaIndex. Specifically, we show how to use our SubQuestionQueryEn...
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.


### ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (5 messages): 
        
- **Harmony Unleashed with LlamaIndex**: `@andysingal` introduces Chain-of-Table through LlamaIndex, highlighting a transformative framework for data interpretation. They shared an article titled "Harmony Unleashed: LlamaIndex’s Guided Symphony with Chain-of-Table" available on [Medium](https://medium.com/technology-hits/harmony-unleashed-llamaindexs-guided-symphony-with-chain-of-table-d866247a72d2).
  
- **Fluid DB, AI's next frontier**: `@anakin.xyz` talks about a fluid database concept which updates its schema dynamically using AI, potentially to be used for AI agents. Further explanation is available in a tweet linked to [Twitter](https://x.com/adamzvada/status/1747002314106282007?s=20) and the project can be found on [GitHub](https://github.com/TheMind-AI/fluid-db).

- **Extraction and Querying Revolutionized**: `@sandeepsangole` inquires whether tables embedded in confluence pages are compatible with SimpleDirectoryReader and GPTVectorStoreIndex. `@andysingal` responds by referencing an article on how to extract and query tables using LlamaIndex's tech, titled "Unlocking Insights: Harnessing Table Extraction and Advanced Data Querying with LlamaIndex's Pandas" on [Medium](https://medium.com/ai-advances/unlocking-insights-harnessing-table-extraction-and-advanced-data-querying-with-llamaindexs-pandas-f7200ef07771).

- **Awaiting Resolution**: `@andysingal` awaits feedback on whether the solution offered was successful in addressing `@sandeepsangole`'s query.

**Links mentioned**:

- [Harmony Unleashed: LlamaIndex’s Guided Symphony with Chain-of-Table](https://medium.com/technology-hits/harmony-unleashed-llamaindexs-guided-symphony-with-chain-of-table-d866247a72d2): Ankush k Singal
- [Unlocking Insights: Harnessing Table Extraction from Unstructured Data and Querying with…](https://medium.com/ai-advances/unlocking-insights-harnessing-table-extraction-and-advanced-data-querying-with-llamaindexs-pandas-f7200ef07771): Ankush k Singal
- [Tweet from Adam Zvada (@adamzvada)](https://x.com/adamzvada/status/1747002314106282007?s=20): if you&#39;ve been thinking about agents and gen interfaces, you need to about hear about this fluid database.  LLMs will be rendering interfaces but they need proper data grounding otherwise they won...
- [GitHub - TheMind-AI/fluid-db: Fluid Database](https://github.com/TheMind-AI/fluid-db): Fluid Database. Contribute to TheMind-AI/fluid-db development by creating an account on GitHub.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **NVIDIA Outshines in Speed and Efficiency**: In comparisons of GPUs for deep learning tasks, the **NVIDIA RTX 4090** is highlighted for being more energy-efficient than the 3090 and the **Mac Studio M2 Ultra**. A detailed [GPU guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) and [LoRA examples](https://github.com/ml-explore/mlx-examples/tree/main/lora) provide resources for AI engineers considering hardware for deep learning applications.

- **Merging MoEs Sparks Interest and Debate**: Discussion around merging fine-tuned **Llama2 models** with tools like **Mergekit MoE** opened up conversations about the feasibility and techniques involved in merging models to achieve domain adaptation. Shared insights and documents, such as [*Perfecting Mergekit MoEs*](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit), contribute to the exploration of future model development strategies.

- **Maximizing Inference Throughput**: AI engineers shared insights on memory bandwidth's influence on inference speed, theoretical throughput capacity of the **RTX 3090**, and the recommendation to use **Nvidia hardware** over Mac for high-throughput tasks, including fine-tuning and inference in deep learning.

- **Mixtral Training Insights and Embedding Developments**: The **Mixtral** model's training progress was shared, with training halted at 86% and results posted on [Hugging Face](https://huggingface.co/datasets/SebastianBodza/wikipedia-22-12-de-dpr). Discussions also revolved around prompt design's impact on query specificity and "raw query" inputs, while **Jina AI's bilingual embedding model** was announced with an extensive 8k token length and a new benchmark suite [available on GitHub](https://github.com/jina-ai/mteb-de).

- **Contemplating Extended Context Lengths in Embedding Models**: Skepticism was expressed over the benefits of extended context lengths in embedding models like **M2-BERT**, with reference to an opinion warning against poor performance for context sizes larger than 300 tokens. The efficacy of embedding dimensions and token length in top models was discussed, touching upon the trust in industry opinions.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (21 messages🔥): 
        
- **Deep Dive into GPU Selection for Deep Learning**: `@thewindmom` shared a comprehensive [GPU guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) for selecting the best GPU for deep learning tasks. They provided a comparison of GPUs by speed for llama inference, noting **NVIDIA's 4090 is faster than both 3090 and Mac Studio M2 Ultra**, and also shared a link to [LoRA examples](https://github.com/ml-explore/mlx-examples/tree/main/lora) on the MLX framework for Mac.

- **Exploring MoE Merges for Fine-Tuned Llama2 Models**: `@philipmay` raised a question about merging two fine-tuned Llama2 models using Mergekit MoE and linked to the [Mergekit on GitHub](https://github.com/cg123/mergekit/blob/mixtral/moe.md). They inquired whether merging a business domain-specific model with a RAG prompting model was sensible since LORA targets the self-attention layers.

- **Adaptation via Merging or Stacking**: In response to `@philipmay`, `@bjoernp` noted that LoRA usually targets all linear layers including FFN layers. `@philipmay` considered either merging for domain adaptation or stacking models, while `@bjoernp` mentioned the trade-off between memory requirements and throughput when using dual MoEs.

- **Skepticism Regarding the Effectiveness of Domain Expert Notion in MoEs**: `@bjoernp` and `@sebastian.bodza` discussed the preliminary nature of MoE merges and the misconception about the "domain experts" in MoEs being too fine-grained to represent specific domains effectively.

- **Practical Considerations for Training and Merging MoEs**: `@philipmay` saw potential in scaling MoE models by having teams develop them independently, and `@bjoernp` acknowledged this as an interesting approach for future large-team production. They further touched upon the possibility of training merged MoEs with Axolotl, to which `@bjoernp` responded that it should work well. 

- **Additional Mergekit MoEs Insights**: `@devnull0` shared a link to a document titled *Perfecting Mergekit MoEs* posted by `@Teknium1` on Twitter, which might be of interest to `@philipmay` and others considering MoE merges. The document can be found [here](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit).

**Links mentioned**:

- [Tweet from Teknium (e/λ) (@Teknium1)](https://fxtwitter.com/Teknium1/status/1746774307383157042?t=QoiD2dRLhYD0ZlzcUtGyMg&s=19): .@DudeMan6790 in @NousResearch discord shared a document he wrote about mergekit MoEs, if anyone&#39;s interested  &#34;Perfecting Mergekit MoEs&#34;  https://docs.google.com/document/d/1_vOftBnrk9NRk...
- [mergekit/moe.md at mixtral · cg123/mergekit](https://github.com/cg123/mergekit/blob/mixtral/moe.md): Tools for merging pretrained large language models. - cg123/mergekit
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.
- [mlx-examples/lora at main · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/lora): Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (9 messages🔥): 
        
- **GPU Showdown: 4090 vs 3090 Energy Efficiency**: `@thewindmom` claimed that the **RTX 4090** is much more energy-efficient than the 3090, offering **fp8 training** and vastly superior performance in inference. These observations suggest that choosing the right hardware is crucial for optimization and efficiency based on a model's specific needs.
  
- **Memory Bandwidth's Role in Inference Speed**: In a theoretical assessment of memory bandwidth's effect on inference speed, `@thewindmom` calculated that **RTX 3090** could potentially feed a model through its system almost **44.56 times** per second, suggesting that memory bandwidth may significantly influence performance.

- **Discussing Mac's Compute Limitation**: `@bjoernp` noted that inference on Macs, regardless of optimizations, is still compute-bound and significantly slower than the RTX 4090, particularly in high-throughput scenarios.

- **Local vs High-Throughput Inference Preferences**: `@_jp1_` recommended **Nvidia hardware** for deep learning tasks that require high throughput, finetuning, or inference, suggesting that a fully equipped Mac may be better suited for local, smaller-scale tasks.

- **Potential for Custom Benchmarks**: `@sebastian.bodza` responded to a shared [ArXiv paper](https://arxiv.org/abs/2311.03687), expressing a willingness to set up custom benchmarks to compare **RTX 4090** and **3090**, particularly concerning how quantization might affect performance.

**Links mentioned**:

- [Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models](https://arxiv.org/abs/2311.03687): Large Language Models (LLMs) have seen great advance in both academia and industry, and their popularity results in numerous open-source frameworks and techniques in accelerating LLM pre-training, fin...
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#8-bit_Float_Support_in_H100_and_RTX_40_series_GPUs): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.


### ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/) (16 messages🔥): 
        
- **Mixtral Training Partially Complete**: `@sebastian.bodza` has stopped training Mixtral at 86% completion, citing the data looks fine except for a minor issue with an exclamation mark. They explained that the process was time-intensive (55 hours) but could be improved with rolling batches, although it would require code adjustments. The results can be found on the Hugging Face repository [here](https://huggingface.co/datasets/SebastianBodza/wikipedia-22-12-de-dpr).

- **Prompt Design May Influence Specificity**: `@sebastian.bodza` noted that unspecific questions could arise from prompts that do not specify a "closed" question format, to which `@bjoernp` and `@philipmay` added their observations on the same issue, suggesting that post-processing or prompt adjustments could help filter or produce more specific questions.

- **Proposal for "Raw Query" Form in Model Input**: `@bjoernp` suggested including a "raw query" form in model input, providing examples such as "Geburtsdatum von Abraham Lincoln." `@philipmay` agreed, mentioning that for effective RAG systems this should be covered by the BM25 component, and also suggested the use of BERTopic for the extraction of keyword queries.

- **Jina AI Announces New Bilingual Embeddings**: `@thewindmom` shared [Jina AI's announcement](https://jina.ai/news/ich-bin-ein-berliner-german-english-bilingual-embeddings-with-8k-token-length/) of a new bilingual German/English embedding model with an 8k token length, and its plans to make it available on AWS Sagemaker and HuggingFace. They noted its performance similar to multilingual e5 base and a novel German benchmark suite based on the MTEB presented on [GitHub](https://github.com/jina-ai/mteb-de).

- **Skepticism Over Embedding Models with Extended Context Lengths**: `@philipmay` shared a LinkedIn post regarding M2-BERT with a 32K context length, accompanied by comments from Nils Reimers warning of poor performance for models with context sizes larger than 300 tokens. `@hammadkhan` expressed trust in Reimers' opinion on embeddings, while `@sebastian.bodza` mentioned that top models often use 1024 embedding dimensions.

**Links mentioned**:

- [Ich bin ein Berliner: German-English Bilingual Embeddings with 8K Token Length](https://jina.ai/news/ich-bin-ein-berliner-german-english-bilingual-embeddings-with-8k-token-length/): Jina AI introduces a German/English bilingual embedding model, featuring an extensive 8,192-token length, specifically designed to support German businesses thriving in the U.S. market.
- [SebastianBodza/wikipedia-22-12-de-dpr · Datasets at Hugging Face](https://huggingface.co/datasets/SebastianBodza/wikipedia-22-12-de-dpr)
- [GitHub - jina-ai/mteb-de: MTEB: Massive Text Embedding Benchmark](https://github.com/jina-ai/mteb-de): MTEB: Massive Text Embedding Benchmark. Contribute to jina-ai/mteb-de development by creating an account on GitHub.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Niche AI Models Nail NSFW Prose**: User `@slono` emphasized the effectiveness of specialized models such as *mlewd/noromaid* over standard ChatGPT for NSFW storytelling, and expressed interest in adapting these models for programming tasks due to their superior performance.

- **Geppetto Project on Backburner**: An API tool called [geppetto](https://github.com/wesen/geppetto/blob/task/add-event-ui-connection/pkg/steps/ai/ollama/chat.go), designed for interfacing with **ollama**, was mentioned by `@slono`, but its readiness was implied to be on hold due to other priorities.

- **ChatGPT Gets a Guardian**: `@swyxio` shared a [Reddit post](https://www.reddit.com/r/ChatGPT/comments/196k679/chatgpt_has_a_new_guardian_tool/) about OpenAI's introduction of a new ChatGPT Guardian tool, developed in partnership with NASS, which redirects procedural election-related inquiries to CanIVote.org.

- **Putting FrankenMoE on Ice**: A [tweet by @main_horse](https://fxtwitter.com/main_horse/status/1746779017674702853?s=46&t=90xQ8sGy63D2OtiaoGJuww) generated discussion about halting development on mergekit MoEs for six months, with contrasting views on the benefit of esoteric prompting approaches.

- **Synthetic Datasets Step into the Limelight**: A new synthetic image-to-code dataset called WebSight, created with models from Mistral and Deepseek, was spotlighted by `@swyxio` in a [tweet by @LeoTronchon](https://fxtwitter.com/LeoTronchon/status/1746952870824394953), who discussed the possibility of adapting the firellava model to utilize it.

- **Pivoting to Luma Calendar for Paper Club**: `@swyxio` mentioned a transition from **Luma multisession to Luma calendar** that required members of the paper club to reconfirm attendance, potentially increasing the number of participants in the upcoming session.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (21 messages🔥): 
        
- **Discovering the Power of Niche Models**: User `@slono` shared their excitement after exploring local models for NSFW story writing, praising the mlewd/noromaid variants for being far superior to standard ChatGPT responses. They are particularly eager to use these models for programming, citing the potential improvement over instruct-based interactions.
  
- **Coding with Ollama**: `@slono` mentioned working on an API tool, [geppetto](https://github.com/wesen/geppetto/blob/task/add-event-ui-connection/pkg/steps/ai/ollama/chat.go), to interact with **ollama**, indicating that it isn’t ready yet as they have other priorities to complete first.
  
- **Guardian Tool for Responsible ChatGPT Use**: `@swyxio` linked to a new Guardian tool in ChatGPT, sharing a [Reddit post](https://www.reddit.com/r/ChatGPT/comments/196k679/chatgpt_has_a_new_guardian_tool/) and elaborating on OpenAI's collaboration with NASS to direct procedural election-related questions to CanIVote.org.

- **Pause on FrankenMoE**: The thread discussed a Google Doc about *mergekit MoEs* that's stirring debate, with `@swyxio` referencing a [tweet by @main_horse](https://fxtwitter.com/main_horse/status/1746779017674702853?s=46&t=90xQ8sGy63D2OtiaoGJuww) suggesting a six-month pause on frankenMoEs and `@slono` noting the potential effectiveness of esoteric prompting ideas.

- **An Exploration of Synthetic Multimodal Datasets**: `@swyxio` highlighted a synthetic image-to-code dataset, WebSight, linked in a [tweet by @LeoTronchon](https://fxtwitter.com/LeoTronchon/status/1746952870824394953), created using models from Mistral and Deepseek and expressed interest in a finetune for firellava model to use this dataset.

**Links mentioned**:

- [Tweet from main (@main_horse)](https://fxtwitter.com/main_horse/status/1746779017674702853?s=46&t=90xQ8sGy63D2OtiaoGJuww): i am calling for a 6-month pause on all frankenMoEs until someone explains why this should work at all, ever  ↘️ Quoting Teknium (e/λ) (@Teknium1)   .@DudeMan6790 in @NousResearch discord shared a doc...
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPT/comments/196k679/chatgpt_has_a_new_guardian_tool/)
- [Tweet from Leo Tronchon (@LeoTronchon)](https://fxtwitter.com/LeoTronchon/status/1746952870824394953): 2024 is the year of multimodal, but also of synthetic data! 👨‍🔬  GPT4-V is pretty good at image to code, but most open-source VLMs struggle. Since there were no scaled Image2Code datasets, we decide...
- [geppetto/pkg/steps/ai/ollama/chat.go at task/add-event-ui-connection · wesen/geppetto](https://github.com/wesen/geppetto/blob/task/add-event-ui-connection/pkg/steps/ai/ollama/chat.go): golang GPT3 tooling. Contribute to wesen/geppetto development by creating an account on GitHub.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages): 
        
- **Platform Shift May Increase Paper Club Attendance**: `@swyxio` mentioned that due to a shift from **Luma multisession to Luma calendar**, all paper club members had to reconfirm their attendance. This change might lead to an abnormally large turnout at this week's paper club.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Langchain Embedding Innovations**: `@meeffe` showcased using OpenAI embeddings in Langchain, sparking discussions on the advanced utilization of embeddings with the `from langchain_openai.embeddings import OpenAIEmbeddings` snippet.
- **Strategizing Memory in Langchain**: `@roi_fosca` explored integrating memory in Langchain, touching upon the use of LCEL expressions and `RedisChatMessageHistory`, and noted concerns about token limits.
- **Frontend Scaling from Streamlit to Production**: `@rjuro` sought advice on transitioning to production-ready frontend solutions for a FAQ chatbot, indicating a move beyond Streamlit for projects using Chroma, Gemini, and Langserve frameworks.
- **Spatial Computing Collaboration Celebrated**: `@abdullahi__` shared insights about spatial computing's role in enabling collaborative environments through a [LinkedIn post](https://www.linkedin.com/posts/abdullahi-fahm_one-mit-lab-has-already-pioneered-collaborative-activity-7152637643439181824-wKA0?utm_source=share&utm_medium=member_desktop), sparking interest in its multifaceted applications.
- **Dynamic LLM Configuration via FastAPI and Pydantic**: `@pramodhgopalan_80290` discussed the configuration of LLMs on the fly using FastAPI and pydantic, querying about the use of `with_config()` in `langserve.APIHandler` for dynamic per user LLM initialization.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (15 messages🔥): 
        
- **Embeddings Import Shared**: `@meeffe` highlighted a code snippet regarding embeddings in Langchain using `from langchain_openai.embeddings import OpenAIEmbeddings`. This snippet implies active development or usage of OpenAI's embedding features within Langchain.
- **Exploring Memory Strategies in Langchain**: `@roi_fosca` shared insights on incorporating memory in Langchain using LCEL expressions and `RedisChatMessageHistory`. They mentioned a potential concern about token limits when loading history into the context.
- **Seeking Frontend Wisdom for Chatbots**: `@rjuro` asked for advice on moving from Streamlit to production-ready frontend solutions for a FAQ chatbot integrated with Chroma, Gemini, and Langserve.
- **Showcasing Advanced Retrieval-Augmented Generation (RAG) Techniques**: `@rahuldey8431` discussed experimenting with RAG solutions and shared a demo link of a code base expert system. They also expressed an interest in collaborating with others on RAG techniques.
- **Inquiry About Multilingual Support in Langchain**: `@huzhenghui` queried about the environmental support for LCEL, questioning whether it is exclusively for Langchain Python or available in other languages too.

**Links mentioned**:

[Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/d1204aa3d0c84555b01db15277fb5695?sid=e12a7e1b-9be2-4dda-97b1-9d20fb700ec7): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **Configuring LLMs on the Fly**: `@pramodhgopalan_80290` shared their current setup using **FastAPI** and **pydantic** for configuring different language model providers such as Azure and Cohere, and inquired how to initialize the correct model using `langserve.APIHandler`. They are seeking advice on whether to use `with_config()` or require a different code structure to configure the Language Learning Model (LLM) dynamically per user.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (3 messages): 
        
- **Exploring Spatial Computing's Potential**: `@abdullahi__` shared a [LinkedIn post](https://www.linkedin.com/posts/abdullahi-fahm_one-mit-lab-has-already-pioneered-collaborative-activity-7152637643439181824-wKA0?utm_source=share&utm_medium=member_desktop) highlighting how spatial computing can create collaborative environments and foster new opportunities.
- **Unveiling Gemini AI App on Google Play**: `@vansh12344` announced the release of the **Gemini AI** app which combines chatting with AI and image-to-text processing, highlighting features like on-device chat history and code outputs in markdown. The app is available on the [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.geminiai&referrer=lcdc).
- **Code Base Assistant Chat App Demo**: `@rahuldey8431` shared a demo of a chat-based code assistant that can understand and explain complex code bases and technical documentation. The tool and demo can be found at this [Loom video](https://www.loom.com/share/d1204aa3d0c84555b01db15277fb5695?sid=e12a7e1b-9be2-4dda-97b1-9d20fb700ec7) and [Netlify app link](https://sage-platypus-36a0c2.netlify.app/), respectively. `@rahuldey8431` also invites DMs to discuss advanced RAG techniques.

**Links mentioned**:

- [Gemini AI - Apps on Google Play](https://play.google.com/store/apps/details?id=com.projecthit.geminiai&referrer=lcdc)
- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/d1204aa3d0c84555b01db15277fb5695?sid=e12a7e1b-9be2-4dda-97b1-9d20fb700ec7): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
- [React App](https://sage-platypus-36a0c2.netlify.app/)


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Nous-Hermes-2 Takes the Lead**: `@teknium` launched **Nous-Hermes-2 Mixtral 8x7B**, boasting higher performance than MistralAI's Mixtral Instruct. The model has both SFT+DPO and SFT-Only variants, hosted on Hugging Face, links for which include [Nous-Hermes-2 DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO), [Nous-Hermes 2 SFT](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT), and [DPO Adapter](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-adapter).

- **Training with Axolotl**: The new **Nous-Hermes-2** utilized the **Axolotl** training framework, confirming training at full precision without changing gating or auxiliary loss functionality.

- **Sticking To Conventions**: `@teknium` responded to `@baptistelqt`, stating no modifications were made to expert layers' initialization or gating in the creation of Nous-Hermes-2; they adhered to standard procedures of the Hugging Face trainer.

- **Expert Specialization Explorations Envisioned**: Following `@baptistelqt`'s interest in visualizing expert specialization in Nous-Hermes-2, `@teknium` admitted interest but cited a lack of capability in generating such graphs, similar to those in the Mixtral paper.

- **Off-Topic Multimedia Share**: User `pradeep1148` shared a YouTube link with no accompanying information, [YouTube Video](https://www.youtube.com/watch?v=KGqWqgloSfY).

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (16 messages🔥): 
        
- **Nous-Hermes-2 Sets a New Benchmark**: `@teknium` announced the release of **Nous-Hermes-2 Mixtral 8x7B**, an open-source language model, in both SFT+DPO and SFT-Only variants. It claims to outperform MistralAI's Mixtral Instruct model in popular benchmarks and is available on Hugging Face ([Nous-Hermes-2 DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO), [Nous-Hermes 2 SFT](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT), [DPO Adapter](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-adapter)).

- **Axolotl as the Training Framework**: `@teknium` confirmed that the **Axolotl** training framework was used for developing Nous-Hermes-2, and that the model was trained at full precision without any modifications to gating mechanisms or auxiliary loss.

- **Keeping It Standard**: In reply to `@baptistelqt`'s query about any modifications in the expert layers' initialization or the gating mechanism, `@teknium` clarified that the process involved standard training procedures as managed by the default Hugging Face trainer.

- **Curiosity About Expert Specialization**: `@baptistelqt` expressed interest in analyzing the expert specialization of Nous-Hermes-2 Mixtral 8x7B with visualizations akin to those in the Mixtral paper. `@teknium` showed interest in this as well but mentioned a lack of knowledge on creating such graphs.

**Links mentioned**:

[Tweet from Teknium (e/λ) (@Teknium1)](https://fxtwitter.com/Teknium1/status/1746990384738357731): It&#39;s finally time! Our Mixtral 8x7B model is up and available now!  Nous-Hermes-2 Mixtral 8x7B comes in two variants, an SFT+DPO and SFT-Only, so you can try and see which works best for you!  It&...


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=KGqWqgloSfY


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Anthropic Advocates for Open AI Models**: A discussion highlighted a [paper by Anthropic](https://arxiv.org/pdf/2401.05566.pdf) on the risks of malicious fine-tuning, underscoring the importance of transparency in training datasets and model framework for safe AI development.

- **Quality Red Team Research Recognized**: Praise was given for the high quality of Anthropic's red team paper, favorably contrasting it with another work, the nightshade paper, and setting a standard for what good red team research should entail.

- **Concerns and Confusions Over AI Open-Sourcing and Regulation**: The guild debated the implications of open-source large language model (LLM) usage and possible legal restrictions, sharing a [linked article](https://1a3orn.com/sub/machine-learning-bans.html) discussing AI safety group positions and misunderstandings that could inform potential regulation.

- **Lull in Literature**: One member, mkaic, expressed disappointment over the lack of updates from *hf papers*, indicating a slower day in the world of AI research publications.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (3 messages): 
        
- **Anthropic's paper on malicious fine-tuning**: `@twoabove` discussed a paper by Anthropic that suggests the only "safe" models may be those with a completely open training framework and datasets, referencing a [malicious fine-tune study](https://arxiv.org/pdf/2401.05566.pdf).

- **Praise for red team paper quality**: `@astropulse` expressed approval for Anthropic's red team paper, implicitly criticizing another paper, nightshade, by stating this is what a good red team paper should look like.

- **Debate over open-source LLMs and proposed regulations**: `@progamergov` shared a [link](https://1a3orn.com/sub/machine-learning-bans.html) to a discussion about misunderstandings related to AI safety groups, their stance on the use of open-source LLMs, and the impact of potential legislative bans on such models.

**Links mentioned**:

[Many AI Safety Orgs Have Tried to Criminalize Currently-Existing Open-Source AI](https://1a3orn.com/sub/machine-learning-bans.html)


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (1 messages): 
        
mkaic: hf papers no update today, sadge


        

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Microservice Matchmaking with Emojis**: `@dbreunig` mentioned *standing up a microservice* that is specialized in matching text to **single emojis**.
- **Emoji-Suggest in Action**: `@dbreunig` shared a link to [emoji-suggest.fly.dev](https://emoji-suggest.fly.dev/Preparing%20for%20a%20Long%20Bike%20Ride) demonstrating the utility in context with the phrase "Preparing for a Long Bike Ride".
- **Brief but Positive Feedback**: `@mroswell` responded with a simple "Nice.", suggesting approval of the shared microservice.
        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

Only 1 channel had activity, so no need to summarize...

teknium: https://fxtwitter.com/Teknium1/status/1746990384738357731
        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Serverless Tracker Shart Shared**: `@stevekamman` updated the channel with his "serverless tracker" NPM shart, providing a [comparison link](https://npmtrends.com/@aws-lambda-powertools/commons-vs-@cloudflare/kv-asset-handler-vs-aws-lambda-vs-miniflare-vs-netlify-vs-vercel-vs-wrangler) among various serverless providers and tools.
- **Bytes newsletter promotion**: `@stevekamman` also promoted [Bytes](https://bytes.dev), a JavaScript newsletter with over 100,000 developer subscribers, suggesting it as a fun and informative read for developers.

**Links mentioned**:

[@aws-lambda-powertools/commons vs @cloudflare/kv-asset-handler vs aws-lambda vs miniflare vs netlify vs vercel vs wrangler | npm trends](https://npmtrends.com/@aws-lambda-powertools/commons-vs-@cloudflare/kv-asset-handler-vs-aws-lambda-vs-miniflare-vs-netlify-vs-vercel-vs-wrangler): Comparing trends for @aws-lambda-powertools/commons 1.17.0 which has 188,823 weekly downloads and unknown number of GitHub stars vs. @cloudflare/kv-asset-handler 0.3.0 which has 664,546 weekly downloa...

        
---
id: 156435c9-3f17-4b0d-bacb-39074b5f3d98
title: '12/25/2023: Nous Hermes 2 Yi 34B for Christmas'
date: '2023-12-26T07:45:27.644769Z'
original_slug: ainews-12252023-nous-hermes-2-yi-34b-for-christmas
description: >-
  **Teknium** released **Nous Hermes 2** on **Yi 34B**, positioning it as a top
  open model compared to **Mixtral**, **DeepSeek**, and **Qwen**. **Apple**
  introduced **Ferret**, a new open-source multimodal LLM. Discussions in the
  **Nous Research AI Discord** focused heavily on **AI model optimization** and
  **quantization techniques** such as **AWQ**, **GPTQ**, and **AutoAWQ**, with
  insights on throughput metrics and proprietary optimization methods like
  tensor parallelization. Additional highlights include the addition of the
  **NucleusX Model** to **Huggingface transformers**, a 30B model achieving **80
  MMLU**, and the **YAYI 2** language model by **Wenge Technology** trained on
  **2.65 trillion tokens**. The community also discussed hosting inference
  servers and updating model cards for better deployment.
companies:
  - teknim
  - apple
  - huggingface
  - wenge-technology
  - nous-research-ai
  - deepseek
  - mistral-ai
  - qwen
models:
  - nous-hermes-2
  - mixtral
  - yayi-2
  - nucleusx
  - ferret
  - 30b-model
topics:
  - quantization
  - model-optimization
  - throughput-metrics
  - tensor-parallelization
  - multimodality
  - model-card
  - inference
  - language-model-pretraining
  - batching
  - model-benchmarking
people:
  - teknium
  - carsonpoole
  - casper_ai
  - pradeep1148
  - osanseviero
  - metaldragon01
---


<!-- buttondown-editor-mode: plaintext -->Today Teknium released Nous Hermes 2 on Yi, making it the top open model compared to Mixtral, DeepSeek, Qwen, and others:

 ![image.png](https://assets.buttondown.email/images/eba7e4a8-d0b8-44a5-b766-e9699ec8157e.png?w=960&fit=max) 

Apple also introduced [Ferret](https://appleinsider.com/articles/23/12/24/apples-ferret-is-a-new-open-source-machine-learning-model?utm_source=ainews&utm_medium=email), a multimodal LLM.

Also, here's the year in memes.

https://www.youtube.com/watch?v=m3kA3eJpnqo

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Extensive discussion on **technical aspects of AI model optimization**, specifically focusing on quantization methods including AWQ, GPTQ and AutoAWQ. Dialogues pivoted around perceived inefficiencies of public quantization techniques, proprietary optimization methods, and model throughput metrics. The conversation involved users `@teknium`, `@carsonpoole`, and `@casper_ai` in the #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) channel.
- Sharing of different **valuable links** in the #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) channel: addition of NucleusX Model to `transformers`, a research paper on Huggingface, a 30B model with 80 MMLU, and the YAYI 2 language model by Wenge Technology.
- Announcement of the release of **Nous Hermes 2**, an advanced model transcending previous Hermes models, trained over Yi 34B and downloadable from HuggingFace, as shared by `@teknium` in the #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) channel.
- Multifaceted discussions held in the #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) channel: wide-ranging conversations on ML models, text-to-speech and text-to-music datasets, the impact of AI in the movie industry, the launch of Nous Hermes 2 and its quantification process.
- Conversations in the #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) channel pertained to hosting service recommendations for inference servers and ways to run Hermes models on a Mac.
- A dialogue about finding inference code examples for a specific model and the need to update the respective **model card**, as seen in the #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) channel.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (45 messages🔥): 
        
- **Quantization Benefits and Limitations**: There was a long technical discussion between `@teknium`, `@carsonpoole`, and `@casper_ai` regarding the effects and efficiency of quantization methods in AI model optimization. `@teknium` argued that **public quantization techniques such as AWQ and GPTQ are not efficient** compared to undisclosed techniques used internally by some organizations.
- **Throughput Metrics**: The discussion also delved into metrics for model throughput across various batch sizes, with further exploration of the tradeoffs between **batching vs sequential generation**.
- **Proprietary Optimization Methods**: `@teknium` suggested that some organizations may be using proprietary parallel decoding methods that offer significant speed improvements. `@carsonpoole` believed it was more likely that organizations were using **tensor parallelization across multiple GPUs for speed up**.
- **AutoAWQ Performance**: `@casper_ai` weighed in, stating that **AutoAWQ outperforms vLLM** in their benchmarks up to a batch size of 8. However, FP16 eventually delivers higher throughput with enough concurrent generations.
- **Job Inquiry**: A user, `@pradeep1148` expressed interest in working for Nous Research in a brief message.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (4 messages): 
        
- **NucleusX Model Addition to Transformers**: User `@osanseviero` pointed out that [NucleusX Model is being added to `transformers`](https://github.com/huggingface/transformers/pull/27259).
- **Reference to a Research Paper**: User `@metaldragon01` shared a [link to a research paper](https://huggingface.co/papers/2312.14862) on Huggingface.
- **30B Model with 80 MMLU**: User `@metaldragon01` mentioned a **30B model with 80 MMLU** but didn't specify additional details.
- **YAYI 2 Language Model**: User `@metaldragon01` shared a [link to the YAYI 2 language model](https://huggingface.co/wenge-research/yayi2-30b) developed by Wenge Technology. This model uses **2.65 trillion Tokens** of high-quality, multi-language corpus for pre-training.


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (2 messages): 
        
- **Announcement of Nous Hermes 2**: User `@teknium` officially announced the release of **Nous Hermes 2**. This model builds on the Open Hermes 2.5 dataset and surpasses all previous Open Hermes and Nous Hermes models in benchmark scores. The model was trained over **Yi 34B** and can be downloaded from HuggingFace [here](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B).


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (214 messages🔥🔥): 
        
- **Discussion on Different ML Models**: `@gabriel_syme .beowulfbr` questioned how to identify the weights that should be reduced in ML models while discussing a research paper. In another thread, `@Otto von Bismarck` and `@alpindale` discuss the performance of AI model goliath and possible testing models. Later `@.beowulfbr` shared his personal experience with Reddit's toxic environment, while revealing the criticism his AI model, codeninja, received on the platform.
- **Text-to-Speech and Text-to-Music Datasets**: `@qwerty_qwer` announced that he has access to massive data sets for text-to-speech and text-to-music models, to which `@nruaif` responded by suggesting diffusion-based text2sound models and provided a link to a [GitHub repository](https://github.com/daniilrobnikov/vits2) of VITS2, an efficient text-to-speech model. 
- **AI and Movie Industry Discussion**: `@mihai4256` shared a [twitter post](https://twitter.com/JWach26/status/1739260697895117280?t=1mBMGpiRi1EERgJEZ9tUmg&s=19) discussing the potential of AI in democratizing the movie industry. He later also inquired about the professional whereabouts of Jeremy Howard, founder of Fast.ai who recently launched a new venture, Answer.ai.
- **Nous Hermes AI Model**: `@teknium` excitedly announced the release of the new Nous Hermes 2 AI model and shared the [model chat link](https://chat.openai.com/g/g-MGIdYisxl-small-answer) as a Christmas present to the community. This led to several appreciative reactions and excited queries from other users on the platform. 
- **Discussion on Model Quantification and Performance**: `@teknium` and `@n8programs` had a detailed discussion about how to perform quantization for the new model, including the hardware requirements and other relevant aspects. They discussed various benchmarks, and `@n8programs` went on to carry out quantization successfully and shared the [quantized model on GitHub](https://huggingface.co/N8Programs/Nous-Hermes-2-Yi-34B-GGUF/tree/main).


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (5 messages): 
        
- **Hosting Services for Inference Servers**: User `@kenakafrosty` asked for recommendations on hosting services for inference servers, expressing a preference for a solution that allows for serverless and pay-only-for-compute operation, but doesn't incur long start-up times.
- **Running Hermes Models on Mac**: User `@ac_sd` inquired if Hermes models can be run directly on a mac, and also asked for clarification on a specific file format. Responding, user `@n8programs` confirmed that Hermes models can indeed be run on a mac.


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (5 messages): 
        
- **Inference Code for Model**: User `@vic49.` expressed difficulty in finding inference code examples for a specific model online, noting the absence of such information on the **model card**. `@qnguyen3` responded to the concern, assuring that they would update the model card later in the day with the required information.


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Release and discussion over new AI models: **Apple's Ferret**, an open-source multimodal Learning Language Model, was introduced and discussed for its unique image querying feature; **Dolphin 2.6 Phi-2**, another new model with coding focus, was also introduced and its obedience and dependencies were discussed. Information and links were provided for further conversation. 

- AI and tech discussions: A heated conversation on loading large models led to suggestions of altering layer numbers; chat history maintenance, AI-generated text changes and real-time code autocompletion were also talked about. Issues with special characters in windows usernames were briefly discussed and solutions provided.

   - In a related context, examination and comparison of **AMD Radeon RX 6800 XT**'s performance was discussed through published performance tables; benefits of a **3060 ti** over a **2080 Super** in a multi-GPU setup emerged, an upgrade to **64GB RAM** for larger models was shared, and optimization setups for the **Mixtral model** were discussed through a shared guide. The idea of fitting multiple GPUs through PCI risers was proposed, with potential issues acknowledged.

- LM studio in the business field and app development: An LM Studio Android app project from `@thelefthandofurza` was announced; business usage of LM Studio and testing procedures were explored as `@docorange88` inquired about it.

- Extension releases: A pre-release version of the [AI navigator extension](https://chromewebstore.google.com/detail/contextual/clignepnaepogpgndkbdcnfppjblogak) that supports multiple AI technologies was announced.

- Community engagement: Christmas and holiday greetings were exchanged among the users; an issue concerning a model loading error was addressed and discussions about experimental builds for long conversations and their limitations were explored. A specific instance of system troubleshooting via re-installation was also addressed.

- Lastly, miscellaneous discussions surrounding model configurations and upcoming updates were conducted. This includes GPU usage for loading models, bundling of OpenChat presets, and persistent errors with Mixtral models on Linux platforms.

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (17 messages🔥): 
        
- **Apple's New Open Source ML Model**: User `@pierrunoyt` shared a [link](https://appleinsider.com/articles/23/12/24/apples-ferret-is-a-new-open-source-machine-learning-model) about **Ferret**, a new open-source multimodal Learning Language Model released by researchers from Apple and Cornell University in October. The model can use regions of images for queries.
- **Holiday Greetings**: Users `@heyitsyorkie`, `@authentictimers`, `@thelefthandofurza`, `@izukumidoriya2211`, and others exchanged Christmas and holiday greetings.
- **LM Studio Android App Project**: `@thelefthandofurza` announced that they are working on an LM Studio Android app to work with the inference server, which was mostly developed by ChatGPT. Plans to share the code on GitHub and leave it open for communal improvements were stated.
- **Business Usage of LM Studio**: `@docorange88` inquired about testing and using LM Studio for business purposes, preferring a direct private message for communication.
- **AI Music Generation Discussion**: `@american_pride` initiated a conversation about AI music generation models, specifically mentioning Suno. `@musenik` contributed to the discussion, voicing a preference for LM Studio due to Suno's requirements for online generation and account creation. The question of whether there are any music generation models that run on LM Studio remains open.


### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (36 messages🔥): 
        
- **Loading Large Models**: `@bearmcx98` encountered an issue when trying to load a large model with 40 layers on a machine with 8gb VRAM and 32gb RAM. `@fabguy` suggested reducing the number of layers to 10 as the current configuration might be too much for the machine, particularly with models above Q3 or Q4.

- **Message History with AI**: `@oaaliyev` asked about using Python or another language to chat with AI via API, saving the message history, and maintaining the chat history after reloading the script. `@fabguy` explained that this needs to be programmed manually by storing the history in a file as it is currently not supported on the server side.

- **Changing the AI's Generated Text**: `@fabguy` explained a way to change the AI's generated text, suggesting that users can simply modify the words the AI uses instead of trying to argue or persuade the AI. This will make the AI treat the change as its own idea.

- **Real-time Code Autocompletion**: `@funapple` asked if there is a way to use active models in LMS for real-time code autocompletion. `@heyitsyorkie` recommended exploring the continue extension in VS Code, which does something similar through local models.

- **Challenges with Special Characters in Windows Usernames**: `@proutes` faced an issue possibly linked to having a special character (é) in the windows username. `@yagilb` confirmed that it was due to the illegal character and suggested changing the path by clicking the gear icon next to “Chats” top left.


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (1 messages): 
        
- **Discussion on Dolphin 2.6 Phi-2**: User `@clickclack777` introduced **Dolphin 2.6 Phi-2**, a new, uncensored chat model with a coding-focus [link](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2).
- The user commented that **Dolphin 2.6 Phi-2** is highly obedient but not DPO tuned, therefore it might require system prompt encouragement.
- The model, developed by Eric Hartford and Fernando Fernandes, is sponsored by [convai](https://www.convai.com/). 
- **Dolphin 2.6 Phi-2** is based on [Phi-2](https://huggingface.co/microsoft/phi-2) and follows Microsoft's microsoft-research-license, which prohibits commercial use. 
- To converse about the model, they shared a Discord [link](https://discord.gg/SmbBewAM).


### ▷ #[🛠-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/) (6 messages): 
        
- **GPU Usage in Loading Models**: User `@mynd` suggested that one should be able to fully load the model on a **3090 GPU** by trying `-1` for the number of layers, implying that this action would transfer all of the layers to the GPU.
- `@pefortin` responded by advising to monitor the vram usage, which is expected to increase. If it doesn't, there could be an issue with the installation/config.
- **Bundling of OpenChat Preset**: `@sublimatorniq` expressed a desire for the OpenChat preset to be bundled as other chat templates for convenience.
- `@ptable` responded by asking why doesn't the user simply save it as a config. But `@sublimatorniq` countered that having it bundled like other presets would align with the goal of convenience.


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (3 messages): 
        
- **AI Navigator Extension Christmas Release**: `@sublimatorniq` announced a pre-release version of the [AI navigator extension](https://chromewebstore.google.com/detail/contextual/clignepnaepogpgndkbdcnfppjblogak) that supports `ollama`, `lm-studio`, `mistral`, `gemini` and more.
- **Functionality Questions**: User `@vic49.` asked about the extension's functionality, which wasn't clear from the webpage.
- **Explanation of the Extension**: In response, `@sublimatorniq` detailed that the extension counts tokens, queries the context, hyperlinks responses, and scrolls to the source.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (114 messages🔥🔥): 
        
- **Graphics Card Performance Testing**: `@fabguy` and `pcrafter` discussed the performance of **AMD Radeon RX 6800 XT** in terms of tokens per second. The exact numbers, tested on a system with i5-12600k CPU and 32GB of RAM, are available [in the posted table](https://raw.githubusercontent.com/discord-conversation/hardware-discussion-comment-summary/main/table/performance-table.md). `pcrafter` found the 0.11 tokens/s amusing, as they could predict the next word in the generated text.
- **GPU Choice for Multi-GPU Setup**: `@pefortin`, `@heyitsyorkie`, and `rugg0064` had a discussion on a better choice between **NVIDIA 2080 Super and 3060 ti** for a multi-GPU setup, the consensus leaned towards the **3060 ti** due to its better compatibility with a 3090 in the same system.
- **Extended RAM Usage**: `@dedr1ck` shared their upgrade to **64GB RAM** to fit larger models. They noted that while the new RAM is slower (7200MHz vs. 600MHz), it does provide the necessary capacity to run larger models like Mixtral which was previously crashing their system. `@heyitsyorkie` commented on this, stating that the performance benefit mostly comes from extra VRAM and system RAM mainly speeds up time to first token.
- **Optimizing Mixtral Model Setup**: `@heyitsyorkie` shared a [guide](https://rentry.org/HowtoMixtral) detailing steps on setting up **Mixtral models** - from which Kobold version and model quants to select, to troubleshooting common issues. Quality might degrade under 4-bit and ensuring a minimum of 20GB VRAM/RAM for better speeds was suggested.  
- **Considerations for Multi-GPU Setup**: `@pefortin` raised the idea of fitting multiple GPUs on a standard motherboard using PCI risers, although they acknowledged potential performance issues because of the risers' limited transfer speed. They shared plans to conduct experiments with different card mixes on Linux and report back with results. `totallybored` and `rugg0064` discussed the possible bottlenecks presented by such a setup, especially with the added complexities of inferencing across different GPUs.


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (9 messages🔥): 
        
- **Issues with mixtral models**: User `@doderlein` reported an error loading mixtral models in the Linux version, receiving an error message regarding the tensor 'blk.0.ffn_gate.weight'. User `@fabguy` responded that this issue is currently unsupported in the Linux version, but should be fixed in the next release.
- **Impressions on mixtral models**: `@fabguy` shared that the mixtral models are less impressive than the hype suggested and advised `@doderlein` to try out other models such as open Hermes.
- **Unavailable Experimental Build for Long Conversations**: User `@yagilb` suggested an experimental build that addresses issues with very long conversations deviating off course. However, this build is currently only available for Mac.
- **Persistent Model Loading Error on Linux**: User `@eason0731` also encountered the same error on Ubuntu 22.04.3 LTS while trying to load a model locally. The user inquired about the release of a Linux version that would address this persistent issue, referencing a previous chat that promised a fix in the next release after 0.2.8. `@yagilb` directed them to recent discussions in another channel for more information on the subject.


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (1 messages): 
        
@heliosprime_3194 glufy: I uninstalled and installed it back and now it is working


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- In-depth discussions around the implementation, performance and fine-tuning of the **Mistral model series**, emphasising their potential utility in various applications. Key techniques and deployment issues include the functionality of **.safetensors** files, **Mistral 8x7b deployment on Edge Coral AI**, the handling of **GGUF file formats** in model deployment, and the specificities of the **HuggingFace version of the instruct-v0.2** with mlx.
- User `@qwerty_qwer` made available a large **text-to-music dataset** potentially beneficial for training certain models.
- Explored the importance of developing efficient **audio generation models**, with `@flame1536` underscoring the necessity of small, fast audio generating models similar to **ElevenLabs**.
- Proposal by `@tafferboy` for implementing an **AI Summary feature on Discord** to help members keep up with discussions.
- Discussion of the potential of models in **tool usage and coding**, particularly through a technique described as "negotiation" by `@poltronsuperstar`. A related discussion centred around the possibilities of **model interaction and independent improvement** viewed as a key to achieving **artificial general intelligence (AGI)**. This includes an emphasis on model reliability as a current issue facing AI application development.
- Clarifying the differentiation between the **paid API of Mistral-AI** and free versions available on platforms like **Perplexity.AI**. Consideration of whether existing **large language models (LLMs)** could be enhanced by online weight updates and the introduction of rich human feedback.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (43 messages🔥): 
        
- **Opening and Understanding .safetensors Files**: User `@hasanurrahevy` inquired whether it was possible to open and understand .safetensors files. `@blueridanus` provided a Python code snippet to open these files and explained that these files hold the weights of a model. The weights are symbolized by the mentioned file names like "model.embed_tokens.weight" (which is of dimensions torch.Size([32000, 4096])). 

- **Deployment Inquiry for mistral 8x7b on Edge Coral AI**: `@queuelabs` raised a question about optimizing and deploying a mistral 8x7b model on edge coral ai, specifically regarding how to optimize model for edge inference through flash storage using an SD card.

- **Contextual - AI Assisted Navigation and Querying**: `@sublimatorniq` shared the news of an XMAS pre-release of an AI navigator extension called Contextual that supports mistral/gemini/ollama/lm-studio/etc. Included in the announcement was a [link](https://chromewebstore.google.com/detail/clignepnaepogpgndkbdcnfppjblogak) to the extension on the Chrome Web Store.

- **Availability of Large Text-to-Music Dataset**: `@qwerty_qwer` offered to share their large dataset consisting of 1 million songs from Spotify and approximately 20k hours of audio files with captions. This data might be useful if someone wanted to train text-to-music or text-to-speech models. 

- **Need for Fast, Small Audio Generation Models**: `@flame1536` pointed out the lack of work done on fast, small audio generation models and emphasized how important they could be to unlocking new applications. They suggested the need for a small ~7B model capable of local operation, which would offer quality similar to ElevenLabs. 

- **Suggestion for AI Summary Feature on Discord**: `@tafferboy` suggested that the server moderators should enable the AI summary feature on Discord. This would allow users to conveniently review conversations that took place while they were away.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (2 messages): 
        
- **Mistral Open Source version**: `'@ken70wtf` and `@tom_lrd` discussed about the benefits of open source and open weight models. They mentioned that it would theoretically allow anyone to modify models to remove or minimise any alignments they don’t like. They also discussed the current state of **Mistral 8x7b** by dolphin 2.5 and its performance with Chinese language.
- **Instruct-v0.2 with mlx**: `'@unskilless` shared their experience and queries about using the HuggingFace version of **instruct-v0.2** with mlx. Notably, they noticed a difference in the **Mistral 7B** models in terms of the missing output layer in the HuggingFace version as compared to the .pth ones.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (15 messages🔥): 
        
- **Using Mistral AI locally**: User `@sakst` asked for guidance on how to use **Mistral AI** locally after downloading the files posted on 8th December. `@sublimatorniq` recommended using platforms like [Ollama AI](https://ollama.ai/) and [LM Studio AI](https://lmstudio.ai/) for this purpose. 

- **Using LM Studio for Mistral AI on Windows**: `@sakst` attempted using **LM Studio** for running **Mistral 0.2** on Windows, but faced difficulties uploading the downloaded files due to large file size (around 90 GB). 

- **Understanding GGUF format for AI models**: `@dutchellie` clarified that the files to be used with **LM Studio** should be in **GGUF format**, which is a quantization format that reduces the size of the models. These **GGUF format** files are not available on Twitter but can be downloaded from **Huggingface**. 

- **Sources for GGUF files**: `@Dutchellie` named a Huggingface user, **TheBloke**, who posts quantized models in several formats, including **GGUF**. 

- **Confusion Resolved**: With the advice from `@dutchellie`, `@sakst` expressed an understanding of the necessary steps and expressed gratitude towards the community for the assistance.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
- **Mistral Model Details**: `@casper_ai` asked `<@707162732578734181>` about any plans to publish a detailed paper or blog on the **Mistral model series**. Specifically, they're interested in learning about the architecture optimization for training and hyperparameters used. They mentioned that while MegaBlocks are available, it's unclear which parts were used for Mistral.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (1 messages): 
        
.tanuj.: Burned through a lot of tokens, but the agent framework is looking promising!


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (77 messages🔥🔥): 
        
- **Using AI Models for Tool Usage and Coding**: `@poltronsuperstar` and `@victronwolfson` discussed the potential of AI models in tool usage and coding practice. `@poltronsuperstar` suggested a method of "fake few shot learning" with large context tokens and multiple iterations until the desired output is achieved. They refer to this approach as "negotiation" and shared that they successfully imitated the behavior of Discord's official bot using GPT-3.5 in this manner.

- **Models Interacting with Themselves and AGI**: `@blueridanus` and `@poltronsuperstar` had a back-and-forth on the potential of models interacting with themselves, with `@poltronsuperstar` expressing confidence that this approach is the path to AGI (artificial general intelligence). There was some contention on whether AGI can be achieved solely through advancements in language modeling, or if more components are required. 

- **Intelligent Self-Improvement of Models**: In an extended discussion on AGI, `@poltronsuperstar` put forth the idea that self-improvement in a model is key to achieving AGI. They proposed a scenario where the complexity of a codebase that can generate code equals or surpasses the complexity of its own codebase - signaling an ability for self-enhancement and possibly AGI. 

- **Ideas to Improve Current LLMs**: `@blueridanus` suggested exploring approaches to update current language large models (LLMs), incorporating online weight updates and ability to make gradient steps based on reasoning capabilities and rich human feedback, in order to enhance their learning capabilities. They acknowledge the issue of catastrophic forgetting as a hurdle in the way. 

- **Reliability as the Biggest Issue**: In response to `@tonyaichamp`'s query about the biggest difficulty in building LLM apps currently, `@sublimatorniq` stated that reliability is a significant challenge.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (17 messages🔥): 
        
- **Comparison of Mistral-AI API with Free Alternatives**: `@ved_ikke` questioned the additional benefits of the paid API of **Mistral-AI** compared to the free version available on other platforms such as Perplexity.AI. `@blueridanus` clarified that some platforms offer evalutation versions but if one doesn't have any use for the API itself, they can use other platforms which host free versions for evaluation.
- **Accessing Mistral-AI through Other Platforms**: `@ved_ikke` referenced [Perplexity AI Labs](https://labs.perplexity.ai) as a platform where **Mistral AI** can be accessed for free. `@blueridanus` pointed out that this is an instance of a playground hosted for users to evaluate their offering.
- **Metric for Mistral embed**: `@ak35` asked for clarification on what the metric is for Mistral embed, mentioning it has 1024 dimensions but not specifying if the metric is cosine, dotproduct, or euclidean.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Several users expressed interest in learning AI & Prompt Engineering despite lacking a tech background, with suggestions to refer to the [Prompt Engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) on the OpenAI website.
- OpenAI Discord members exchanged Christmas greetings, while discussing technical comparisons between Co-pilot and ChatGPT, noting noticeable improvements in OpenAI's system performance.
- Practical applications of GPT-4 were discussed, particularly regarding its usage for summarizing school lecture transcriptions due to the extended information context window.
- A collection of resources and tools for data extraction and analysis from Excel spreadsheets were recommended by multiple users.
- User experience issues regarding browser compatibility, interface accessibility, API quotas, and refund policies were discussed in the openai-questions channel, along with several troubleshooting suggestions.
- In the gpt-4-discussions channel, discussions centered on modifying the current prompt/message cap in OpenAI projects, adjusting prompting habits according to OpenAI's guide, and the potential of GPT in captioning multi-speaker audio.
- OpenAI's [Prompt Engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) was linked and discussed in the prompt-engineering channel, alongside conversations about common challenges in prompt engineering such as model evasiveness and hallucinations. An inquiry about model predictions of non-positive hypothetical character reactions sparked discussion and linkage to OpenAI's updated usage policies and successful interaction examples.
- Similar discussions occurred in the api-discussions channel, focusing on the Prompt Engineering guide, the prevalent challenges in prompt engineering, and queries on predictive behavior based on character profiles as well as user interactions with the chatbot model.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (7 messages): 
        
- **Learning AI & Prompt Engineering**: User `@saketrathore_52744` expressed interest in learning AI & Prompt Engineering despite not having a tech background. `@thunder9289` reassured the user that a tech background was not necessary and suggested referring to the **Prompt Engineering guide** on the OpenAI website. `@definitely_not_y` also mentioned an OpenAI course by Andrew NG.
- `@thunder9289` provided the link to the **Prompt Engineering guide** ([here](https://platform.openai.com/docs/guides/prompt-engineering)) in response to `@saketrathore_52744`'s request.
- **Use of Reference Images in Stable Diffusion**: `@rchap92` asked if stable diffusion can use reference images in a "get as close as possible" approach.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (48 messages🔥): 
        
- **Christmas Celebrations**: The OpenAI Discord chatroom members including `@ta_noshii`, `@peter20225953`, `@intermatrixnaut`, and `@loschess` amongst others, exchanged *Merry Christmas* greetings, sharing various Christmas-themed emojis and stickers.  

- **Comparing Co-pilot and ChatGPT**: A comparison between **Co-pilot** and **ChatGPT** was discussed by users `@pruo` and `【ｐｅｎｕｌｔｉｍａｔｅ】`. They noted that while both are about the same for chat, Co-pilot has a music plugin that ChatGPT may lack.  

- **OpenAI Speed Performance**: Users discussed a noticeable speed improvement in OpenAI's system performance. `@lugui` noticed that the stream speed is way improved, and other users such as `@pruo` and `【ｐｅｎｕｌｔｉｍａｔｅ】` concurred, speculating it could be due to reduced demand because of Christmas or the addition of more GPUs.

- **Use of GPT-4 for Notes**: User `@pencil9195` inquired if GPT-4 plus is worth using for summarizing school lecture transcriptions. `@jaicraft` responded that GPT-4 can use more information for its responses and hence might be better at summarizing, particularly due to its 32k context window.

- **Tools for Excel Data Analysis**: Users `@brianjwash`, `@lumirix`, `@lugui` and `@michael_6138_97508` discussed utilities for detailed extraction of information from Excel data. Suggestions ranged from utilizing Advanced Data Analysis on CSV data to running your own model, possibly with embedding/vectorization techniques.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (13 messages🔥): 
        
- **Browser Compatibility and Privacy Issues**: `Rock` expressed frustration about the interface being unfriendly to privacy browsers, questioning the testing processes prior to updates.
- **Interface and Accessibility**: `skousenx` had difficulties finding Dall-E 3 on ChatGPT Plus and experiencing an different interface than expected. Postulated that the issues might be related to their location (Peru), signing up with a Google account, or due to new plus users having limited tools.
- **Troubleshooting Suggestions**: In response to `skousenx`'s issue, `froggy_chacko` suggested clearing cache/cookies and trying a VPN. They also recommended reaching out to support. 
- **Refund Policy Inquiry**: `skousenx` asking about the possibility of a refund so they can attempt signing up again with a new account.
- **API Quota Issue**: `arthurananda`, a Plus subscriber, reported a rate limit error message accusing them of exceeding their current quota, despite not having used the API previously. They inquired how to resolve this issue.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (11 messages🔥): 
        
- **Removing the 30 prompt cap**: User `@yungamain` asked if there is a way to **remove the 30 prompt cap**. `@satanhashtag` suggested buying another account or switching for the API service.
- **Increasing Messages Limit**: `@Rock` advised `@yungamain` to hit "learn more" and make a case every time the cap is hit for why they should have more messages.
- **GPT Translation Issues**: `@joker002` reported an issue where the bot would only translate 10 of the 20 rows requested. `@Rock` suggested that the issue might be due to recent modifications in how the output works and advised `@joker002` to modify their prompting habits according to OpenAI's guide.
- **Accessing the Prompt Guide**: `@joker002` sought help in finding the Prompt Guide. `@lumirix` shared the link to the guide: [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- **GPT Captioning Multiple Voices**: `@arnoldtri` asked if there's a GPT that can caption different voices of Podcasts / multiple speakers. No answer was provided in the observed messages.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (10 messages🔥): 
        
- **OpenAI's Prompt Engineeering Guide**: User `@exhort_one` introduced a [link](https://platform.openai.com/docs/guides/prompt-engineering) to OpenAI's guide on prompt-engineering. They wondered why it had not been discussed in the channel before.
- **Difficulties in Prompt Engineering**: `@tonyaichamp` asked about common challenges faced in prompt engineering for Language Learning Model (LLM) applications. User `@beanz_and_rice` responded, citing **model evasiveness** and **hallucinations** as the main problems.
- **Model Speculation Behavior**: `@rchap92` raised a question about whether the chatbot model would ever predict non-positive reactions from characters in hypothetical situations. 
- **OpenAI Usage Policies**: In response, `@eskcanta` recommended reviewing OpenAI's updated [usage policies](https://openai.com/policies/usage-policies), advising to discuss with the model about desired outputs while taking these policies into account.
- **Example Case with OpenAI chatbot model**: `@eskcanta` shared a [link](https://chat.openai.com/share/0daf8924-eab4-4f0b-a0db-493bda704e48) to an example of interacting with the OpenAI chatbot, demonstrating how the model can handle conflicts and nuances in conversation.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (10 messages🔥): 
        
- **Prompt Engineering Guide**: `@exhort_one` shared a [link](https://platform.openai.com/docs/guides/prompt-engineering) to OpenAI's guide on prompt engineering, mentioning that the document isn't broadly discussed in the chat. `@bambooshoots` added that the guide seems to be a recent addition from the last month or two.
- **Challenges in Prompt Engineering**: In response to `@tonyaichamp`'s query about difficulties faced in prompt engineering for language model apps, `@beanz_and_rice` mentioned **"Model Evasiveness and Hallucinations"** as significant challenges.
- **Speculations Based on Character Profiles**: `@rchap92` asked if the chatbot could predict character reactions based on their profiles instead of opting for the 'best case scenario'. `@eskcanta` responded with a recommendation to refer to the updated [usage policies](https://openai.com/policies/usage-policies) of OpenAI as it could guide user interactions with the chatbot. They also shared an [example](https://chat.openai.com/share/0daf8924-eab4-4f0b-a0db-493bda704e48) of an educated discussion with the model.
- **Bot Responses to Speculative Questions**: `@rchap92` further mentioned that the bot tends to revert to a positive outcome most of the time when asked speculative questions, which `@pratham_shetty` found amusing.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Discussion around using **Huggingface**: Includes the possibility of a session on Huggingface for engineering students, Apple's open-source multimodal LLM, interest in blockchain partnerships and smart contracts with Huggingface, possible Huggingface service interruption, query about using Dolphin 2.5 mixtral 8x7b on Huggingface, and a question about the gradio_client functionality.
- Learning experiences and inquiries shared in the **Today-I'm-Learning** channel: Highlighting findings on MobileSAM, query about open-source projects, reminder about channel etiquette, guidance sought for beginners, and expressions of appreciation for shared information.
- Exciting find about an [**NLP course from HuggingFace**](https://huggingface.co/learn/nlp-course/) shared by `jaminchen` in the Cool-Finds channel.
- In **I-Made-This** channel, sharing of achievements and seeking solutions: `@andysingal` shared his fine-tuned DistilBERT model with achieved accuracy, and there was a discussion about how to download images with suggestions from `@yjg30737`. Also, a conversation about the differences between Runpod and Kaggle notebooks for downloading generated images.
- **Computer-Vision** channel highlighted model fine-tuning advice (**BLIP-2** model), recommended resources for understanding computer vision models ([Piotr's YouTube channel](https://youtube.com/@Roboflow?si=WPxPnS2KZAiXzK31)), and the showcasing of notable Spaces on the HuggingFace hub ([EfficientSAM Space](https://huggingface.co/spaces/SkalskiP/EfficientSAM)).
- Conversations in the **NLP** channel covered recommendations for image-to-text tasks, characterization of QA and Seq2seq models, discussions on the level of abstraction in models, and a question about methods for text entailment using T5 models. Notably, two models for image-to-text tasks were recommended: [**Donut**](https://huggingface.co/naver-clova-ix/donut-base) and [**Nougat**](https://huggingface.co/facebook/nougat-base).

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (23 messages🔥): 
        
- **Huggingface session**: `@ishavverma` expressed interest in finding someone to deliver a session on Huggingface to engineering students, with the goal of familiarizing them with the platform.
- **Apple's open-source multimodal LLM**: `@dame.outlaw` shared a [VentureBeat Article](https://venturebeat.com/ai/apple-quietly-released-an-open-source-multimodal-llm-in-october/) about Apple quietly releasing an open-source multimodal LLM.
- **Chain Partnerships and Huggingface**: A series of conversations were held between `@cakiki`, `@earduman2`, and `@robolicious` discussing the possibility of **blockchain partnerships** and whether Huggingface has any **smart contracts** (which it doesn't). 
- **Potential Huggingface service interruption**: `@casanovasan` wondered whether the **download service** was down, as a vae pack installation had suddenly stopped.
- **Huggingface Dolphin Integration Query**: `@notsaiff` inquired about the steps to use **Dolphin 2.5 mixtral 8x7b** on Huggingface, noting its free AI hosting.
- **Gradio_Client Functionality**: `@_gilfoyle_` asked if it was possible to change the text/state of textboxes via the **gradio_client**.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (10 messages🔥): 
        
- **MobileSAM**: `@merve3234` shared learning about **MobileSAM**, providing a link to the findings [here](https://x.com/mervenoyann/status/1738959605542076863?s=20).
- **Question about Open-source**: `@neuralink` questioned whether `@merve3234` has open-sourced a project, to which the response was it is a work in progress.
- **Channel Etiquette**: `@cakiki` reminded users to keep on topic and not cross-post in the channel.
- **New Member Inquiry**: `@alluring_chipmunk_62732_31615` inquired on how to start as an absolute beginner in the channel.
- **Appreciation**: `@osanseviero` and `@llmsherpa` expressed appreciation for shared info and notes from various users.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (1 messages): 
        
jaminchen: nlp course from HuggingFace 🙂 https://huggingface.co/learn/nlp-course/


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 messages): 
        
- **Rating Classification Model**: User `@andysingal` shared a link to his **fine-tuned version** of the DistilBERT model. His model achieved a **loss of 0.9611 and an accuracy of 0.7011** on the evaluation set. No other data on the model's description or intended uses was given. The model can be found [here](https://huggingface.co/Andyrasika/bert_clf_results).
- **Downloading Images Discussion**: `@yjg30737` provided a guide on **how to download images** from a created notebook. The user referred `@andysingal` to the Overview and Parameters & Variables sections of his notebook, which can be accessed [here](https://www.kaggle.com/code/yoonjunggyu/stable-diffusion-generated-image-downloader#Overview).
- **Runpod vs Kaggle Notebook**: `@andysingal` expressed interest in understanding the differences between **Runpod and Kaggle notebooks** for downloading generated images. `@yjg30737` suggested downloading and running the source code from the Kaggle notebook on the user's platform to observe the results. The user noted potential modifications required for Kaggle-based code sections. 
- **Variables and Parameters**: `@yjg30737` clarified that the **Variables and Functions can be copied and pasted** from the source. They also detailed the process of downloading from Kaggle as compressing image files into a zip folder and downloading via a specific class provided by Kaggle.
- **Appreciation for Shared Information**: `@andysingal` thanked `@yjg30737` for sharing useful information on handling Variables and Parameters when creating styled images. The user agreed to try the shared download script.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (5 messages): 
        
- **Fine-tuning BLIP-2 Model**: `@srikanth_78440` asked for advice on fine-tuning an image-captioning model, which was answered by `@nielsr_`. He provided a link to a [HuggingFace notebook](https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb) with detailed guidance. He added that both `pixel_values` and `input_ids` need to be prepared and the model labels need to be a copy of the `input_ids`, with padding tokens replaced by `-100`.
- **Piotr's YouTube Channel for Computer Vision**: `@nielsr_` recommended Piotr's [YouTube channel](https://youtube.com/@Roboflow?si=WPxPnS2KZAiXzK31) for understanding computer vision models better for `@blackbox3993`. The channel features various applications of computer vision models.
- **Spaces on the Hub**: `@nielsr_` highlighted some cool Spaces on the HuggingFace hub, providing a link to one such [Space](https://huggingface.co/spaces/SkalskiP/EfficientSAM) created by Piotr showcasing a comparison between Segment Anything Model (SAM) and EfficientSAM.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (5 messages): 
        
- **Image-to-text Models**: `@edude11` recommended two models for image-to-text tasks: Donut and Nougat. Donut for generic tasks, available at [`huggingface.co/naver-clova-ix/donut-base`](https://huggingface.co/naver-clova-ix/donut-base), and Nougat for academic documents, available at [`huggingface.co/facebook/nougat-base`](https://huggingface.co/facebook/nougat-base).
- **QA and Seq2seq Models**: `@opencuiguy` explained to `@merve3234` that QA models are encoder only models that extract answers from a given context. In contrast, seq2seq models are encoder-decoder models that generate the answer. Information on these models could be found at the following pages: [Question Answering](https://huggingface.co/tasks/question-answering) and [Text Generation](https://huggingface.co/tasks/text-generation) on HuggingFace.
- **Level of Abstraction in Models**: Responding to `@opencuiguy`, `@merve3234` confirmed that seq2seq models are the lower-level abstraction, and question-answering is higher level because seq2seq models can be used to solve question answering.
- `@opencuiguy` asked `@merve3234` about suitable methods for text entailment using a T5 model and requested a code that could act as a study reference.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Discussion on retrieval-augmented generation (RAG) implementing tools, as **RAG Local Tool Inquiry** raised by `@nanobitz` with request for any known good tool equipped with user-interface for storage and retrieval of past GPT interactions.
- Recommendation and exchange of academic resources and practical tools:
    - `@caseus_` suggesting a [research paper](https://arxiv.org/pdf/2305.19268.pdf) considering it a **"must-read"** for those engaged in working with **quantized models**.
    - `@noobmaster29` introducing [promptbench](https://github.com/microsoft/promptbench), a **unified evaluation framework for large language models**.
    - `@nanobitz` sharing findings from an [Unsloth's blog post](https://unsloth.ai/blog/mistral-benchmark) about how to finetune Mistral 14x faster with features including sliding window attention.
- Detailed analysis by `@nafnlaus00` on Mixtral's inference code and query around number of experts per token per layer as found in the [GitHub commit](https://github.com/vllm-project/vllm/commit/b5f882cc98e2c9c6dde7357dbac2ec0c2c57d8cd).
- Speculation and advice on fine-tuning parameters, epochs and embeddings:
    - `@noobmaster29` questioning if alpha is an adjustable parameter in post fine-tuning of a model considering it to be causing high loss.
    - `@noobmaster29` also inquiring about the appropriate number of epochs for **larger datasets**, suggesting a limit of **3 epochs** for a dataset of **500 million tokens**.
    - Queries by `@dreamgen` around fine-tuning specifics of special token embeddings during LoRA and FFT.
    - Desire by `@dreamgen` to **freeze all embeddings except those for new tokens** during fine-tuning.
- Query about finetuning deepseek-coder 6.7B model by `@shrex8791`, with a specific problem being described where the model keeps parallelizing, using all memory and seeking advice to make it run unparalelled.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (11 messages🔥): 
        
- **RAG Local Tool Inquiry**: `@nanobitz` asked if anyone knows of a **good RAG local tool with UI** to store and retrieve past GPT conversations.
- **Quantized Models Inference**: `@caseus_` suggested a [paper](https://arxiv.org/pdf/2305.19268.pdf) about inference by downstream with quantized models. They deemed it a **"must-read"** for people working with quantized models.
- **Mixtral's Inference Deep Dive**: `@nafnlaus00` shared their investigation into Mixtral's inference code, [a link to the GitHub commit](https://github.com/vllm-project/vllm/commit/b5f882cc98e2c9c6dde7357dbac2ec0c2c57d8cd), and raised questions about the number of experts per token per layer in Mixtral.
- **Promptbench Evaluation Framework**: `@noobmaster29` shared a link to [promptbench](https://github.com/microsoft/promptbench) on GitHub, which is a **unified evaluation framework for large language models**, and asked if anyone had tried it.
- **Model Fine-tuning Parameter**: `@noobmaster29` asked if alpha is a parameter that can be modified after fine-tuning a model, expressing surprise about what they perceived as a high loss.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (3 messages): 
        
- **Comparison of Sequence Lengths**: `@dreamgen` suggested mentioning the sequence length for a better comparison.

- **Unsloth's Mistral Benchmark Findings**: `@nanobitz` shared a link to Unsloth's blog post by Daniel Han discussing how to finetune Mistral 14x faster. The article revealed that the QLoRA support for Mistral 7B, CodeLlama 34B, and other models based on the Llama architecture has been released. It includes features like sliding window attention, preliminary Windows and DPO support, and 59 shared notebooks. [Unsloth's Blog](https://unsloth.ai/blog/mistral-benchmark)

- **Benchmark Findings Review**: `@nanobitz` acknowledged that `@casper` has previously examined the findings from Unsloth's post.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (9 messages🔥): 
        
- **Dataset Size and Epochs**: `@noobmaster29` inquired whether 3 epochs would be too much for **larger datasets**. When asked to define "large", noobmaster29 clarified a dataset size of about **500 million tokens**.
- **Fine-tuning Special Tokens with LoRA**: `@dreamgen` questioned the specifics of fine-tuning special token embeddings using LoRA and FFT. They asked whether **only the added token embeddings** would be fine-tuned or all, and if this process is configurable. `@caseus_` advised including the "lm head and embed token" layers for LoRA.
- **Frozen vs Unfrozen Embeddings**: Subsequently, `@dreamgen` expressed the desire to be able to **freeze all embeddings with the exception of those for new tokens**.
- **Finetuning Deepseek-coder 6.7B**: `@shrex8791` queried about a challenge they were facing whilst finetuning the **Deepseek-coder 6.7B** model. According to shrex8791, the model kept parallelizing and as a result, used up all memory. They sought advice on how to make it unparallel.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Switching Models based on User Queries**: User `@shellord` is working on project that needs to switch between a *General Question and Answering* model and a *Function Calling* model based on user query.
- **LangChain Related Queries**: User `@emreweb3` asked if **LangChain** includes smart contracts, while user `@a404.eth` shed light on issues surrounding streaming for custom agents incorporating complete RAG chains. A potential event in Denver was also discussed, highlighting current limitations due to the team's security requirements.
- **Challenges in LLM App Development**: Conversation initiated by user `@tonyaichamp` about the major issues experienced while developing LLM apps.
- **FastAPI Dependencies in langserve[client]**: User `@sidlly.eth` raised concerns over why FastAPI needs to be included in the package when adding **langserve[client]**, emphasizing the perception that a client-side SDK shouldn't require FastAPI.
- **GenAI Stack Use and Language Processing Innovations**: Dialogue between user `@tachi3` and `@shamspias`, centering around the utilization of the GenAI Stack. Meanwhile, `@andysingal` shared a [medium blog post](https://medium.com/ai-advances/revolutionizing-language-processing-with-langchain-and-mixtral-8x7b-b955ec2fb5df) encapsulating a new approach to revolutionizing language processing using Llama-cpp and StructuredOutputParser.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (10 messages🔥): 
        
- **Selecting Models for Different User Queries**: `@shellord` sought advice on a project that requires using different models based on the user query. The goal is to switch between a General Question and Answering model and a Function Calling model depending on the type of user query.
- **Discussion on LangChain and Smart Contracts**: `@emreweb3` queried whether **LangChain** incorporates smart contracts. The user `@lhc1921` was tagged but a response isn't present in the provided chat log.
- **Challenges in Developing LLM Apps**: `@tonyaichamp` elicited opinions on the greatest challenges or nuisances in developing LLM apps.
- **Enabling Streaming in Custom Agents**: `@mohammed.shokr` explored how to enable streaming for a custom agent incorporating a complete RAG chain. User `@a404.eth` responded, asking for the code since streaming with agents could be tricky compared to **LECL**.
- **Discussion on LangChain Hack Night Locations**: A conversation took place about potential locations for a LangChain Hack Night. User `@glenn_sjobs` responded to `@shiftybit` explaining the high costs for Hawaii but promised possible future events within the contiguous US. User `@a404.eth` offered to host such an event in Denver, but a meet-up with the LangChain team wasn't possible due to security requirements.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **Langserve Client and FastAPI Dependency**: User `@sidlly.eth` expressed a concern about the need to include **FastAPI** in their package when adding **langserve[client]**. They believe there's no reason for a client-side SDK to require FastAPI.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (3 messages): 
        
- **Discussion on GenAI Stack**: User `@tachi3` asked `@shamspias` if they had tried the GenAI stack. In response, `@shamspias` clarified that they have only gone through the readme and description so far, and **haven't actually tried it out yet**.
- **Blog Post** on "Revolutionizing Language Processing with Langchain and Mixtral-8x7B": `@andysingal` shared a [Medium Blog post](https://medium.com/ai-advances/revolutionizing-language-processing-with-langchain-and-mixtral-8x7b-b955ec2fb5df) discussing a Llama-cpp and StructuredOutputParser Approach to revolutionising language processing. The article is authored by Ankush K Singal and published under AI Advances.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Discussion on **xDAN-AI's Model Performance** and its claim of being the top performer on MT-bench with strong abilities in Humanalities, Coding and Writing with 7b, including user `@cryptossssun`'s enthusiastic endorsement and posted [link](https://huggingface.co/xDAN-AI/xDAN-L1-Chat-RL-v1) to the model's Discord, Twitter, and Huggingface platforms.
- Expression of **skepticism surrounding xDAN-AI's model**, with `@.pathos` and `@technotech` raising doubt about the 7B model's performance and its assertion of being 'close to GPT-4'.
- User feedback regarding the UX and quality of **AI tools** with `@rtyax` comparing the Copilot for IDEs and Continue tools, finding the former to be superior due to its high-quality UX and response quality while finding the latter less useful due to its lack of auto-completion features. 
- Recommendation by `@bjoernp` for `@rtyax` to try out **ClipboardConquerer**, an AI tool, with `@rtyax` expressing interest and agreeing to share their experience with the tool in the future.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (4 messages): 
        
- **xDAN-AI's Model Performance**: User `@cryptossssun` shared the [link](https://huggingface.co/xDAN-AI/xDAN-L1-Chat-RL-v1) of xDAN-AI's new model claiming it as the **Top 1 Performer on MT-bench** and made a bold claim about it being the first top model performing well in Humanalities, Coding and Writing with 7b. The post also contains links to the model's Discord, Twitter, and Huggingface platforms.
- **Users' Skepticism**: `@.pathos` and `@technotech` expressed their skepticism about the 7B model's performance claiming that it is 'close to GPT-4', questioning its credibility.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (3 messages): 
        
- **Discussion on AI tools UX and quality**: User `@rtyax` shared his experience with different AI tools. Found **Copilot for IDEs** to be the best in terms of user experience and response quality. Also, discussed *Continue* a **Copilot-alternative that integrates with any local or remote LLM**, however, found it a lot less useful since it doesn't offer auto-completion, only chatting/refactoring. 
- **Suggestion to try ClipboardConquerer**: User `@bjoernp` suggested trying **ClipboardConquerer**. `@rtyax` expressed interest in trying the tool and mentioned sharing their experience afterward.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Conversation revolves around holiday greetings with `@venadore` from `#general-chat` wishing everyone a Merry Christmas, complemented by a [YouTube Video: A.I. Rewind 2023 (but in memes)](https://youtu.be/m3kA3eJpnqo), and `cryptossssun` from `#oo` sharing a similar sentiment.
- A [Twitter Post](https://fxtwitter.com/teknium1/status/1739444560848453866) was shared by `@teknium` in `#general-chat` without any additional context provided.
- `@undi` and `@fredipy` both in `#general-chat`, expressed anticipation and congratulations for an unspecified 'release', stirred much interest but lacked more detailed information.
- A [Twitter Post](https://twitter.com/shootime007/status/1739312828111360339) linked by `cryptossssun` in `#oo`, but without any further context discussed.

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (5 messages): 
        
- `@venadore` shared a [YouTube Video: A.I. Rewind 2023 (but in memes)](https://youtu.be/m3kA3eJpnqo) recapping the year of AI in memes. They also wished everyone a Merry Christmas.
- `@teknium` shared a [Twitter Post](https://fxtwitter.com/teknium1/status/1739444560848453866).
- `@undi` congratulated on the release and mentioned seeing it on another website, but didn't specify what the release was.
- `@fredipy` responded to `@teknium` and showed excitement about trying 'it' out, although it's unclear what 'it' refers to.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 messages): 
        
cryptossssun: Hi Merry Christmas !
https://twitter.com/shootime007/status/1739312828111360339


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **AGI Space Appreciation**: User `@0xevil` expressed enthusiasm for the **basement AGI space**.
- **Tweet Link**: User `@teknium` shared a [link to a Tweet](https://fxtwitter.com/teknium1/status/1739444560848453866?).
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Control over Context Length in Threads**: User `@joshcho_` mentioned their frustration over the lack of control over context length for threads. They stated, *"...i have to delete then copy everything over (or is there another way)"*, hinting at possible interest in more efficient ways to manage thread content.
        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

Only 1 channel had activity, so no need to summarize...

- Users `@._z` and `@vince_uc` exchanged **Christmas** greetings on the channel.
       
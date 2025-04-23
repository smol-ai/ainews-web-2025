---
id: b4652fae-87f4-454e-8635-09c342357284
title: '[AINews] 12/10/2023: not much happened today'
date: '2023-12-10T23:49:57.169413Z'
status: sent
type: archival
source: api
metadata: {}
original_slug: ainews-12102023-not-much-happened-today
---

<!-- buttondown-editor-mode: plaintext -->famous last words but it's a quiet day and everyone is heading out to neurips (so are we). Andrej called out sources of alpha [here](https://twitter.com/karpathy/status/1733968385472704548) and we're considering adding them, please shoutout what reddits/discords/anime pfp anons we should add.

swyx

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Members expressed interest in attending **NeurIPS** and meeting up, with suggestions for future AI events in Australia. `@richardblythman` urged those interested in an open-source, decentralized AI project to reach out to them. Users shared their projects, like `@cyborg_1552`'s [photo GPT AI tool](https://www.photogptai.com/) and `@pradeep1148`'s [introduction of StableLM Zephyr 3B](https://www.youtube.com/watch?v=YWYNLaWDoNQ).
- User `@gabriel_syme` triggered interest around **Mixtral** by sharing a [GitHub link](https://github.com/open-compass/MixtralKit). Performance comparisons between **Mixtral and GPT-3.5** heated discussions. `@mihai4256` unveiled their fine-tuned model, Pallas-0.2, available on [Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.2). A [Youtube video](https://youtu.be/y9k-U9AuDeM? si=2X5j64_cdsdKwWEw) discussing open-source LLMs usage sparked brief reactions. 
- Both `OpenHermes-2.5-neural-chat-v3-3-Slerp` and **Mixtral** were the topic of hype for their performances, with debating dictates on the latter's GPU requirement. Tools such as `Tensorboard`, `Wandb`, `evalplus`, `llamahub` were stated beneficial for fine-tuning and evaluating models. User experiences on model hosting platforms like **Ollama** and **LM Studio** were exchanged with contrasting opinions favoring both.
- A robust conversation on **MoE** led by `@gabriel_syme` clarified why Mixtral, a model based on **Mistral**, is set apart from previous implementations. Discussions on fine-tuning LLMs suggested limited data requirements. The potential of **Mixtral** being competitive with **GPT-4** after finetuning was proposed. `@wlrd` explained how open-source LLMs could be implemented, leading to the **OpenHermes 2.5 - Mistral 7B** model. Speculations on **GPT-3.5** suggested it's a 20B model and forecasted its soon open-source release. Inference optimization possibilities for ChatGPT touched on strategic batching, potential caching, and user base size.
- The **memes** channel saw an array of emojis and memes shared by members for fun and communication. Specific interests in speakers like **Yann** and **Karpathy** were expressed. User `@teknium` amusingly delineated a character as being heavily concerned about x risk.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (19 messages🔥): 
        
- **NeurIPS Meetup**: 
    - `@blue_matcha` asked if anyone was attending NeurIPS in the hopes of meeting up, and `@teknium` said they might be available on a Thursday and Friday. 
    - `@gabriel_syme` expressed disappointment in NeurIPS consistently being in the US, later revealing they are based in Australia. `@gabriel_syme` also proposed hosting an event in Australia the following year.

- **Open Source and Decentralized AI Co-founder Search**: 
    - `@richardblythman` is in search of a co-founder for a project in the open-source and decentralized AI space and asked anyone interested to DM them.

- **Interest in Australian AI Conferences**:
    - `@deki04` pointed out that there would be considerable interest in Australian-based AI conferences, recounting a well-attended in-person fastAI course held in Brisbane led by Jeremy Howard.

- **Photo GPT AI Development**: 
    - `@cyborg_1552` mentioned the development of a [tool](https://www.photogptai.com/) using Stable Diffusion, offering to write a blog post if people are interested. They also provided a link to their [github](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) for those wishing to explore further.

- **Introduction of StableLM Zephyr 3B**:
    - `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=YWYNLaWDoNQ) introducing StableLM Zephyr 3B, a large language model.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (1 messages): 
        
nonameusr: i think he used markdown


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (24 messages🔥): 
        
- **Discussion about Mixtral and its Architecture**: `@gabriel_syme` shared a [GitHub link](https://github.com/open-compass/MixtralKit) to MixtralKit – a toolkit for the `mixtral-8x7b-32kseqlen` model. `@cyborgdream` posted a [twitter link](https://twitter.com/abacaj/status/1733660077154816013), sharing that Mixtral outperforms GPT-3.5 in benchmarks even before fine-tuning. The subsequent discussion involved `@nonameusr`, `@euclaise`, and `@chhillee` debating the benefits and uniqueness of Mixtral's Transformer-based architecture.

- **Release of New Fine-Tuned Model**: `@mihai4256` announced the release of their fine-tuned model, Pallas-0.2, hosted on [Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.2). This model, a fine-tune of `Tess-34B-v1.4`, is designed for reasoning tasks and performs well with long system prompts.

- **Video about Open Source LLMs usage**: `@teknium` shared a [Youtube video](https://youtu.be/y9k-U9AuDeM?si=2X5j64_cdsdKwWEw) answering the question "Should You Use Open Source Large Language Models?" `@n8programs` and `@nonameusr` gave one-word responses to the question, with conflicting opinions.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (639 messages🔥🔥🔥): 
        
- **Fine-tuning and Performance Discussions**: Users discuss the fine-tuning and performance of several models, including **Hermes 2.5**, **Mistral**, and **GPTs Agent**. For instance, `@nonameusr` suggests that `OpenHermes-2.5-neural-chat-v3-3-Slerp` (also nicknamed "Slurpy") outperforms the original `Hermes` in some regards but notes inconsistencies. Several users also discuss the performance of `Mixtral` (or `Mixtral MoE`), discussing topics like its GPU requirements and its behavior when quantized.

- **Model Hosting and Management Platforms**: Multiple users compare their experiences using **Ollama** and **LM Studio** for hosting and managing AI models. While some users express a preference for Ollama, others point out that LM Studio may be more customizable and better support a wider range of models.

- **Compute and Training Resources**: Users like `@vatsadev` and `@gabriel_syme` discuss their computing resources, with the discussion also touching on the potential of university resources. 

- **Useful Tools**: Discussion also touched on various tools like `Tensorboard`, `Wandb`, `evalplus`, and `llamahub`, which can be useful for fine-tuning, testing, and evaluating models.

- **New Models and Techniques**: The channel saw mentions of new models and techniques, like 'slerp' (in the context of `OpenHermes-2.5-neural-chat-v3-3-Slerp`). Some users also speculate about the `Mixtral` and `StripedHyena` models and the potential for further improvements to them via fine-tuning or merging strategies. Finally, `@ldj` suggests that `Mixtral`'s method of choosing "experts" during its computation could influence its performance.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (123 messages🔥🔥): 
        
- **Mixture of Experts (MoE) Discussion**: Users `@akhxl`, `@cyborgdream`, and `@gabriel_syme` engaged in a conversation about MoE, with `@akhxl` initially expressing confusion about the sudden hype over a technique that's been around for some time. `@gabriel_syme` provided an explanation, stating that previous implementations didn't yield useful models and that Mixtral, based on **Mistral**, has shown practical utility.
- **Finetuning Large Language Models (LLMs)**: In a dialogue involving `@akhxl` and `@gabriel_syme`, clarifications about the amount of data needed for finetuning were offered. `@gabriel_syme` noted that recent advancements didn't require substantial data to finetune a good model due to the quality of base models and expansive pretraining data availability. A discourse on the potential of Mixtral to perform comparably to **GPT-4** after finetuning ensued with `@cyborgdream` predicting such an outcome.
- **Open Source LLMs Usage**:`@.plot` and `@wlrd` held a conversation regarding the acquisition and implementation of open-source LLMs. `@wlrd` pointed out that the models' weights are open-sourced and can be fetched from *Hugging Face* and gave an example link to the **OpenHermes 2.5 - Mistral 7B** model.
- **GPT-3.5 Turbo Discussion**: A nuanced discussion over the **GPT-3.5 Turbo** specifications occurred, primarily involving `@cyborgdream`,`@agcobra1`, and `@n8programs`. The discourse ranged from its performance compared to both smaller and larger models, with `@cyborgdream` suggesting the model is possibly a 20B model, basing on the leaked **G3PO** information and predicting its open-source release soon.
- **Inference Optimization for ChatGPT**: User `@zohad_sikder` initiated a conversation regarding potential optimizations for faster inference in ChatGPT. Speculations from `@teknium`, `@bjoernp`, `@eas2535` and `@skadeskoten` ranged from the unlikely use of quantization to strategic batching and potential caching for frequently asked questions. The fast response time of ChatGPT was discussed, with `@zohad_sikder` hypothesizing a robust caching mechanism due to the substantial user base.


### ▷ #[memes](https://discord.com/channels/1053877538025386074/1166105758635655270/) (10 messages🔥): 
        
- **Meme Sharing and Reactions**: Users in this channel, namely `@teknium` and `@Error.PDF`, frequently share emojis and meme reactions. Notable mentions include the **"Y not both"** and **<:pepeshy:1151280286345207819>** emojis.
- **Desire for Certain Speakers**: `@teknium` expressed a desire for individuals such as **Yann and Karpathy** to speak, leading to responses and discussions among the users.
- **Character Evaluation**: `@teknium` expressed their opinion on an unidentified individual, characterizing them as **"crazy psycho about x risk"**.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- An ongoing discussion centered around the topic of **AI bias, morality, and fair use** in the context of copyrighted content and AI. Conversations delved into issues such as biases in large language models (LLMs) and the philosophy of truth, alongside speculations surrounding Google's new AI, Gemini, and alternative AI technology options like Mistral Instruct and gpt4all.
- Members engaged in various **technical discussions regarding GPT-4**, touching upon 'Dynamic Limits', waitlist duration, prefix prompt exploration, ChatGPT's performance and access issues, and differences in features across various devices. Speculations were made about the development of GPT-5 and the opening of GPT Store in the new year.
- Issues with and improvements for **GPT usage** have been a pressing topic, with dissatisfaction expressed over the dialogue summarization by GPT, missing features in GPT Builder, and the absence of a feature allowing Inline editing or trim for AI responses. A parallel conversation took place regarding the acquisition of developer access for ChatGPT plugins, clarification of OpenAI's Terms of Service, and the need for comprehensive guides on custom GPTs.
- Conversations about **game development using GPT** and chatbot performance indicated a healthy interest in the potential applications of AI technology. Issues with captcha during API key generation, searching specific conversations, and perceived changes in GPT output fueled the debate on current limitations and areas for improvement in the AI system.
- A notable topic in the guild was **prompt engineering**, digging deep into the usage of emotional language and the implementation of personalities in PPM. The community also dived into issues concerning text chunking, embeddings, and creation of detailed prompts. The sharing of a series of detailed prompt guidelines and command protocols for GPT-4, dalle, and browser tools reflected collaborative efforts to enhance utilisation of the AI model.


**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (123 messages🔥🔥): 
        
- **Discussion on AI bias and morality**: Users `@whynot66k20ni`, `@light.grey.labs`, `@solbus`, `@lhc1921` engaged in a deep conversation regarding the inherent nature of biases in large language models (LLMs), the philosophy of truth, and the potential self-awareness of AIs. 
- **ChatGPT's AI ethics and 'fair use'**: `@.dooz`, `@lhc1921`, `@light.grey.labs` discussed the 'fair use' in the context of copyrighted content and AI. `.dooz` suggested that transformative use of copyrighted content could be constituted as fair use.
- **Discussion about OpenAI's GPT Store release**: `@lumirix` shared an excerpt from an email received by GPT creators promising the release of the GPT Store early next year and promising other great updates to ChatGPT.
- **Alternatives to OpenAI ChatGPT**: `@mysticmarks1` recommended Mistral Instruct and gpt4all as alternatives or additions to OpenAI's ChatGPT for `@sneakobrah` who was seeking alternative chat AIs.
- **Discussion on Google's AI Gemini**: `@prajwal_345` shared a [link](https://analyticsindiamag.com/google-fools-everyone-with-gemini/) about Google's Gemini AI suggesting that it was announced under pressure, and it outperformed OpenAI's GPT-4 on various benchmarks.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (112 messages🔥🔥): 
        
- **GPT-4 Dynamic Limits and Waitlist Discussion**: `@dr.youvi.avant` asked about the new GPT-4 'Dynamic-Limits'. `@stefatorus` mentioned that unlocking older GPT versions is possible but can be expensive, with his usage amounting to approximately 200 EUR per month. `@killer.5643` inquired about the GPT-4 waitlist duration, with `@7877` mentioning the upcoming GPT Store, and `@jonathan_91672` sharing that he waited about a month for his invitation. 

- **GPT-4 Prefix Prompt Exploration**: `@israel_a4` shared a YouTube tip from Wes Roth which allows users to see GPT-4's Prefix or Secret Prompt by using a certain code. When asked about a potential patch to prevent this, `@elektronisade` stated that no such plans were in place due to the inherent functioning of the models. 

- **ChatGPT Performance and Access Issues**: Several users reported issues with ChatGPT, with `@mrcrack_` mentioning consistent network errors and ADA's ineffective image reading. `@zz99mz` mentioned the issue of the domain not loading at all. `@pruo` indicated trouble with their custom instructions, and `@mrcrack_` also voiced dissatisfaction with the dynamic limits. 

- **Features in Different Devices**: `@gd2x` inquired about the speech feature absence in the Android version of ChatGPT, which `@elektronisade` attributed to the use of an adblocker. A discrepancy between features available in Android and iOS versions was also discussed.

- **GPT-3 Extensions and GPT Store Speculations**: `@youraveragedev` speculated about GPT-5's development, but `@clockrelativity2003` denied its current training. A discussion about GPT Store's opening in the new year was held by `@lugui`.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (158 messages🔥🔥): 
        
- **Issues and Improvements in GPT**: User `@stealth2077` expressed concerns about GPT ending dialogues with a concluding summary paragraph, even after providing explicit instructions not to do so. `@stealth2077` has also proposed a feature for inline editing or trim for AI responses for easier control over generated conversations, a topic joined by `@ath0rus`. `@stealth2077` voiced dissatisfaction over the reduction of GPT usage from 50 to 40 and the removal of additional 10 usages reserved for custom GPT testing.
- **GPT Builder Limitations**: `@amanshrestha` experienced issues in GPT Builder, which seemed to stem from the Python environment. `@stealth2077` also expressed frustration over the restrictions in changing custom instructions mid-chat, and he highlighted the need for a better functionality to edit a chat's context.
- **ChatGPT Plugins**: `@keebs1995` inquired about gaining developer access for ChatGPT plugins for building a calculator app for their industry. `@elektronisade` informed that plugins are being phased out and suggested using custom GPTs instead.
- **Terms of Service (ToS) Clarifications**: User `@eric.turnr` sought elaboration on the OpenAI ToS section mentioning "Automatically or Programmatically extract data or Output (defined below)." `@lumirix` clarified that "Output" is defined in the Content section of the ToS.
- **Performance Issues & Enhancements**: A few users, including `@Shunrai` and `@lucianah`, reported lagging and network error issues with GPT. `@Rock` asked for comprehensive guides about the workings of custom GPTs, and `@strange073` sought clarification on how to access the GPT-4 API with a single dollar donation.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (25 messages🔥): 
        
- **Use of GPT for Game Development**: `@cerebrocortex` shared their experience working on a Civilization-like game, expressing surprise at how well ChatGPT manages tasks like inventory management. They requested people's feedback on their game.
- **ChatGPT Plus Invites**: `@pietman` and `@mlgpro0225` mentioned people receiving invites to join ChatGPT Plus, indicating that the waitlist might be moving forward.
- **Debugging GPT builder**: `@cerebrocortex` asked about updating instructions for a custom GPT and `@Capcon` suggested saving changes to the draft and using the "update" button to publish changes.
- **Searching Specific Conversations in ChatGPT**: `@q16.kr` asked if it is possible to search a specific conversation made with ChatGPT and `@pietman` replied it's not currently available.
- **ChatGPT API Key Generation Issue**: `@realspacekangaroo` reported an issue with captcha while trying to generate a new API key, deeming it excessively difficult and leading to them being locked out from generating new API keys.
- **Change in GPT Output**: `@victronwolfson` noticed a change in the outputs of `gpt-4-1106-preview` over the last week.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (36 messages🔥): 
        
- **Using emotion in prompts**: `@eskcanta` discusses the use of emotional language in prompts and its impact on the ChatGPT during a conversation about a paper named "ai emotional prompt". They not that they could not find a specific prompt used in the paper for testing and cannot thereby reproduce the results.
- **Introducing personalities in PPM**: `@eligump` and `@mysticmarks1` engaged in a dialogue regarding the development of a PPM (persistent personality mode) with two personalities. `@mysticmarks1` shares a [link](https://chat.openai.com/share/ba013894-5bac-43f4-9d6e-3310c5d9e1bc) to illustrate how to implement behaviors like stutters and airheadedness in dialogues. 
- **Creating detailed prompts**: `@cybector` shares a draft of a detailed prompt for the python programming language and invites other users for feedback and suggestions to improve it. 
- **Issues with text chunking and embeddings**: `@merpnderp` requests for resources or discussions about strategies for text chunking and embeddings due to costs for density experiments. `@eskcanta` suggests experimenting with web interface ChatGPT to find potential cost-saving solutions. `@m0bsta` expresses difficulties in this approach due to the limit in messages.
- **Prompt and Guidelines for GPT-4**: `@cat.hemlock` shares a series of detailed prompt guidelines and command protocols for GPT-4, dalle, and browser tools in markdown form. This consisted of the base information, tools used, and various policies to guide the use of the AI model. She also goes on to show the JSON format of what a typical detailed prompt would look like.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (36 messages🔥): 
        
- `eskcanta` discussed [EmotionPrompt's use in language models](https://discord.com/channels/974519864045756446/1182753259081957407), questioning its implementation and effectiveness due to the lack of clear prompt examples in the referenced paper.
- `madame_architect` highlighted part of EmotionPrompt's implementation from the available documentation. They provided examples of emotional stimuli and mentioned that the base prompts & template to which these stimuli were added were also present in the companion documents.
- In a series of messages, `eligump` and `mysticmarks1` discussed the creation and manipulation of **Private Playground Models (PPMs)**, particularly how to incorporate roleplay and specific language styles.
- A user named `mattiacastioni` asked for help in a linked conversation thread. The nature of this request was not further discussed.
- `cybector` shared a template for engaging with ChatGPT surrounding Python programming language discussions, specifically instructing the model to source information from the official Python documentation.
- `merpnderp` asked for recommendations of resources related to strategies for text chunking and embeddings, aiming to decrease costs in production. `eskcanta` suggested discussing cost-saving strategies with ChatGPT.
- Lastly, `cat.hemlock` shared guidelines for using the **markdown, dalle, python, and browser tools** in OpenAI's ChatGPT, as well as an example of how to construct a "default prompt".


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Active discussion and developments around the **Mixtral** integration prompted by `@caseus_`, with a focus on sample packing, sharding, and addressing various technical issues. The creation of the `mixtral-multipack` branch highlighted alongside [relevant GitHub Links](https://github.com/OpenAccess-AI-Collective/axolotl/compare/main...mixtral_sharded).
- Release of a new dataset `Verified-Camel-zh` on Hugging Face by `@noobmaster29` with [direct access to the dataset](https://huggingface.co/datasets/noobmaster29/Verified-Camel-zh).
- A conversation identifying common issues in model error reporting and proposed solutions, such as changing `model_type` and disabling `is_mistral_derived_model`. 
- Sharing and exploration of various scientific paper processing libraries, such as the [allenai/papermage](https://github.com/allenai/papermage), [axa-group/Parsr](https://github.com/axa-group/Parsr), and the [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) libraries, for transforming PDFs, documents, and images into structured data.
- Dialogues on the RLHF channel about the upcoming Data Programming Override (DPO) strategy for data set creation; specifically, the need for two distinct DPO datasets to handle "unalignment" and provision "quality answers". 
- Miscellaneous conversations including a podcast with an axolotl representative, AI projects, tokens in coding, and a YouTube video titled *The Insane Biology of: The Axolotl*.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (25 messages🔥): 
        
- **Mixtral Integration and Development**: 
    - `@caseus_` shared updates on **Mixtral** integration with axolotl, including the addition of a `mixtral-multipack` branch and the merge of Mixtral MoE finetuning w multipack. 
    - To use the updated features, users must install the latest version of `transformers` from git main.
    - For further development, `@caseus_` shared a link to a work-in-progress branch by `@214834317774422028` ([GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/compare/main...mixtral_sharded)).

- **New Dataset Release**:
    - `@noobmaster29` announced a new dataset on Hugging Face called `Verified-Camel-zh` ([link to dataset](https://huggingface.co/datasets/noobmaster29/Verified-Camel-zh)).

- **Miscellaneous Discussions**: 
    - `@swyxio` highlighted a podcast featuring an axolotl representative, and shared several AI-related resource and project links.  
    - A conversation took place on the use and naming of tokens in coding, notably the use of the start and stop tokens.
    - `@noobmaster29` shared a YouTube video titled *The Insane Biology of: The Axolotl* ([link to video](https://youtu.be/bFkIG9S2Mmg?si=bwXWKBM8fI-sPT-R)).


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (170 messages🔥🔥): 
        
- **Mixtral Sample Packing**: `@caseus_` has been working on implementing sample packing for Mixtral and has created a `mixtral-multipack` branch. There were reports of initial high loss that decreases, indicating the potential effectiveness of this approach. `@faldore` has been using the `mixtral-multipack` branch and reported stable operation and decreasing loss rates. 

- **Fixes and Workarounds**: Certain errors were encountered by users, for which workarounds and fixes were suggested. Specifically, disabling `is_mistral_derived_model: true` and changing `model_type: AutoTokenizerForCausalLM` seemed to resolve some issues. There was also a suggestion from `@casper_ai` to remove deepspeed if using a single GPU.

- **VRAM requirements**: Concerns regarding VRAM usage were discussed, with `@caseus_` suggesting strategies to reduce VRAM usage, such as freezing early layers of the model. Running Mixtral on 2xA6000 and 4xA100 GPUs was mentioned, with ambitions to achieve full finetuning on 4 to 8xA6000s. `@casper_ai` created a branch with parts of sharding to optimize VRAM usage, but it is still a work in progress. 

- **Model Error Reporting**: `@ludis___` reported a `RuntimeError` when running Mixtral which read "output tensor must have the same type as input tensor". This was resolved by the removal of certain configuration parameters. 

- **LoRA and qLoRA usage**: There were successful runs of Mixtral using qLoRA on GPU configurations such as 4xA100 and A40. However, attempts to run with LoRA resulted in errors related to the `bnb` package. 

Links:

- [Github branch for mixtral-multipack](https://github.com/OpenAccess-AI-Collective/axolotl/tree/mixtral_mltipack)
- [Github issue for Mixtral optimization](https://github.com/OpenAccess-AI-Collective/axolotl/issues/930)
- [Github pull request for Mixtral memory saving](https://github.com/OpenAccess-AI-Collective/axolotl/pull/934)
- [Github branch for Mixtral sharding](https://github.com/OpenAccess-AI-Collective/axolotl/tree/mixtral_sharded)


### ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (3 messages): 
        
- **Potential Hiring Discussion**: `@faldore` expressed a sentiment that a certain situation could have been improved if they were hired. 
- **Elon Musk Employment Opinion**: In response, `@nruaif` suggested that working under Elon Musk might not be desirable.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (5 messages): 
        
- **Merging Qlora Chat Mixtral Issue**: `@matts9903` reported an error received while attempting to merge the `mixtral` model with the Axolotl tool. The issue is with a validation error for `repo id`: ```huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: './qlora-out'.```
    
- `@caseus_` suggested using an absolute path to the qlora-out directory but the suggestion didn’t resolve the issue. 

- `@caseus_` then shared a recent change to model merging [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/commit/1d21aa6b0ac0e1de832b5d57c82da34220346046) and requested a stack trace for further troubleshooting.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (4 messages): 
        
- **PaperMage Library**: `@noobmaster29` shared a link to GitHub for the [allenai/papermage](https://github.com/allenai/papermage) library, suggesting that it might be worth testing. This library supports NLP and CV research on scientific papers.
- **Parsr Library**: `@visuallyadequate` is currently experimenting with the [axa-group/Parsr](https://github.com/axa-group/Parsr) library, which transforms PDFs, documents, and images into enriched structured data.
- **Tika Library**: `@visuallyadequate` mentions having used the Tika library, describing it as having provided the best solution so far, but they have not yet tested PaperMage.
- **Unstructured Library**: `@joshuasundance` shared a link to the [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) GitHub library, which provides open-source libraries and APIs for building custom preprocessing pipelines.


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (5 messages): 
        
- **DPO Completion**: `@caseus_` mentioned needing to finish the DPO (Data Programming Override), after having been sidetracked by work on **Mixtral**.
- **Unalignment and Quality Answers DPO Dataset**: `@faldore` discussed the idea of needing two DPO datasets, one for **"unalignment"** and another for providing **"quality answers"**.
- **Rejected Field Inquiry and Comparison**: `@nruaif` suggested asking **Llama 2 7B chat** for the rejected field and additionally compared it with **GPT 4**, suggesting that in 90% of cases, the Llama 2 7B chat would yield worse answers.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Extensive discussion on **using local models with chat LLMs in LangChain**, featuring insights from `@_egeres` on the potential use of environment variables and subclassing `LLM` and ideas from `@lhc1921` surrounding the use of a backend like llama.cpp for handling constrained grammar.
- Queries raised by various members but remained unanswered, including:
  - `@analyticsrepo`'s question on **Gemini integration** from Google into LangChain.
  - `@_ashisharya`'s request for comprehensive resources on *agent coding and deployment.*
  - `@xstepz`'s guidance request on *limiting the usability of pandas functions in Kork package.*
  - `@yasuke007`'s *seeking advice on learning pathway for AI development* with a specific focus on the necessity of Python knowledge when using langchain with React.js.
  - `@rajib2189`'s inquiry about the potential *use cases for running language models locally.*
- Announcement by user `@reletreby` regarding the **Askly December Release**, now integrating **OpenAI ChatGPT 3.5** and **HuggingFaceH4/zephyr-7b-beta** from HuggingFace. New features include multi-file reasoning, summarization, web search, necessitating users to *delete and re-upload old files to enable the new functionalities.* Full details shared via [Askly's blog](https://www.askly.ai/blog/askly-december-release).

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (72 messages🔥🔥): 
        
- **Gemini from Google integration**: A user `@analyticsrepo` asked about the status of integrating Gemini from Google to LangChain, but no answer was provided.
- **LangChain with Local Models**:  `@_egeres` and `@lhc1921` discussed extensively the possibility of using local models with chat LLMs in LangChain. `@_egeres` mentioned the possibility of tweaking API endpoints via environment variables and sub-classing `LLM`. `@lhc1921` suggested the use of a backend like llama.cpp that is capable of taking constrained grammar.
- **Resources for Agent Coding and Deployment**: `@_ashisharya` asked for comprehensive resources on Agent coding and deployment, but didn't receive any response.
- **Kork Package with Pandas**: `@xstepz` sought guidance on how to limit the pandas functions accessible to their agent using the Kork package, but didn't receive any response.
- **Learning Pathway for AI Development**: `@yasuke007`, a new AI developer, asked for advice on whether Python would be necessary in their AI development journey using langchain with React.js, but received no response.
- **Use Cases for Running Language Models Locally**: `@rajib2189` asked about the possible use cases for running language models locally, like personal assistant or edge type of analytics, but received no response.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Askly December Release**: User `@reletreby` announced the latest version of **Askly** which is significantly upgraded with the integration of **OpenAI ChatGPT 3.5** and the open-source model **`HuggingFaceH4/zephyr-7b-beta`** from HuggingFace. The new features include multi-file reasoning, summarization, web search, and more. However, to access these features, users who had uploaded files on or before December 1st, 2023, need to delete their old files and reupload them. This is critical to activate the new functionalities. The complete details were shared on the [Askly's blog](https://www.askly.ai/blog/askly-december-release).


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Interaction between `@astra1337` and others after **demo presentations**, highlighting the interest shown by the audience for additional explanation. Additionally, `@astra1337` raised a query about the awareness of **Pygmalion AI** with respect to a **video game demo**.
- Query by `@mister_poodle` about the fine-tuning process of **Mistral-OpenOrca** for specific tasks, with a particular focus on enhancing performance for a **Named Entity Recognition (NER) task with JSON outputs**.
- Dialogue around diagramming tools, with *Whimsical* and *Excalidraw* being highlighted. 
    - *Whimsical* was introduced by `@teknium` and tested by `@gabriel_syme`, noting its tendency for collaborative features. 
    - *Excalidraw* was suggested by `@lightningralf` who provided the link [Excalidraw](https://excalidraw.com/) and noted the existence of an *Obsidian* plugin.

**Alignment Lab AI Channel Summaries**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (3 messages): 
        
- **Astra1337 Interaction with Others regarding Demos**: User `@astra1337` mentioned that people approached them for additional information after some **demo presentations**.
- **Discussion on Pygmalion AI**: `@astra1337` asked someone from a **video game demo** if they were aware of **Pygmalion AI**, a research group known for creating video game characters with memory.


### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (1 messages): 
        
- **Fine-tuning Mistral-OpenOrca**: `@mister_poodle` inquired about fine-tuning **Mistral-OpenOrca** for specific tasks using personal datasets, showcasing an intention to improve the model's performance on a **Named Entity Recognition (NER) task with JSON outputs**. No link or additional information was provided by `@mister_poodle` in this context.


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/) (8 messages🔥): 
        
- **Discussion on diagramming tools**: `@teknium` introduced *Whimsical* as a diagramming website. Upon trying it, `@gabriel_syme` thought that it had collaborative features since it prompted for creating a workspace.
- **Excalidraw recommendation**: `@lightningralf` recommended *Excalidraw* as another option, linking to the website, and additionally, mentioned a plugin for *Obsidian*. Here is his recommended link: [Excalidraw](https://excalidraw.com/).


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Using qlora with small batches and context window**: In a response to a query, `@eugeneyan` shared that a 24gb GPU should work for using qlora with a small batch size and decent context window (batch of 2, context window 512 - 1024).
- **Features query about HumanLoop**: `@jozexotic` expressed concerns about the slow development of new features in HumanLoop, specifically the lack of access to models outside of OpenAI and asked if anyone knew about these additions being on the near term agenda for the platform.
- **Frustrations with chatgpt+**: `@slono` expressed a considering to cancel their chatgpt+ subscription due to the slow progress and recurrent stream errors.
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

pradeep1148: https://www.youtube.com/watch?v=YWYNLaWDoNQ
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

.psychickoala: any of you seen best practices to force parallel function calling
        

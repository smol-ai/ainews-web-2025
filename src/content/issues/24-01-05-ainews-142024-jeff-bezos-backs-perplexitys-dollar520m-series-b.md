---
id: 10c69001-13c7-47bf-aa5a-e5c7636d97c1
title: '1/4/2024: Jeff Bezos backs Perplexity''s $520m Series B.'
date: '2024-01-05T08:29:59.746847Z'
original_slug: ainews-142024-jeff-bezos-backs-perplexitys-520m
description: >-
  **Perplexity** announced their **Series B** funding round with notable
  investor **Jeff Bezos**, who previously invested in **Google** 25 years ago.
  **Anthropic** is raising **$750 million**, projecting at least **$850 million
  in annualized revenue** next year and implementing "brutal" changes to their
  Terms of Service. Discussions in **Nous Research AI Discord** cover topics
  such as **document recall limits from gigabytes of data**, **RNN memory and
  compute trade-offs**, **synthetic datasets**, and benchmarking of models like
  **WizardCoder-33B-V1.1**, **MobileLLaMA-1.4B-Base**, **ShearedLLaMA**, and
  **TinyLLaMA**. Other highlights include **UnsLOTH** optimizations for
  multi-GPU systems, **AI rap voice models**, **context-extending code**, and
  architectural innovations like applying **Detectron/ViT backbones to LLMs**,
  **sliding window attention** in **Mistral**, and parallelizing **Mixtral
  8x7b** with **FSDP** and **HF Accelerate**.
companies:
  - perplexity
  - anthropic
  - google
  - nous-research
  - mistral-ai
  - hugging-face
models:
  - wizardcoder-33b-v1.1
  - mobilellama-1.4b-base
  - shearedllama
  - tinyllama
  - mixtral-8x7b
topics:
  - document-recall
  - rnn-memory
  - synthetic-data
  - benchmarking
  - multi-gpu-support
  - context-length
  - model-architecture
  - sliding-window-attention
  - model-parallelism
  - gpu-optimization
people:
  - jeff-bezos
---


<!-- buttondown-editor-mode: plaintext -->As widely rumored, Perplexity [announced their Series B](https://x.com/AravSrinivas/status/1743046115707285877?s=20). Most notable investor is Jeff Bezos, who also invested in Google 25 years ago.

 ![image.png](https://assets.buttondown.email/images/f69b655e-ef33-44a5-b52d-569c3c55b210.png?w=960&fit=max) 

Elsewhere, Anthropic's ongoing $750m fundraising prompts it to issue [very ambitious forecasts](https://www.theinformation.com/articles/anthropic-projects-at-least-850-million-in-annualized-revenue-rate-next-year) and institute "brutal" ToS changes:

 ![image.png](https://assets.buttondown.email/images/8b77db4e-0e64-4250-b245-ba63779207fa.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Querying Memory Matters**: In a discussion within the **#ctx-length-research** channel, `@benjoyo` wondered about the limits of **document recall from gigabytes of data**. On the topic of costing, `@teknium` noted that, though **costs do rise with context size**, the increase isn't exponential. `@euclaise` shined a light on **RNN memory and compute** subtleties, explaining that in **RNN mode**, memory usage is consistent but requires extensive compute, while non-RNN mode provides a sub-linear compute trade-off for greater memory use. 
- **Software Sovereignty and GPU Struggles**: The **#off-topic** channel saw a lively debate about **data sovereignty** ignited by `@benjoyo`'s question about potential pitfalls of European hosting. `@gabriel_syme` grappled with **graphics card issues on Windows 11**, while `@max_paperclips` hailed the dawn of **synthetic datasets**, much to the chagrin of already overworked GPUs. 
- **Language Models New and Newer**: The **#interesting-links** channel buzzed with talk of language models, from **WizardCoder-33B-V1.1's promising performance** to **Humaneval-Test-Set-Alpha's theoretical prowess under extreme conditions**. `@metaldragon01` highlighted an [article](https://arxiv.org/abs/2401.02412) on a novel model expansion technique called CALM, causing `@gabriel_syme` to speculate on its potential combination with LORAs. In response to `@euclaise` championing the performance of **MobileLLaMA-1.4B-Base**, `@.benxh` asserted the superior benchmarking results of ShearedLLaMA. Meanwhile, evaluation hangs in the balance for **TinyLLaMA**. 
- **Model Possibilities and Practicalities**: In the **#general** channel, `@gabriel_syme` sought clarity on applying **UnsLOTH** optimizations to multi-GPU systems, prompting `@teknium` to emphasize the proprietary nature of UnsLOTH. Mention of a potential 200k context Nous Hermes model stirred intrigue, but it was an **AI rap voice model** that truly had tongues wagging. Queries about Mac-compatible **AI trainers** were fielded, and excitement persisted for **context-extending code** updates.
- **LLM Architectures – Adapt, Implement, Explore**: The **#ask-about-llms** channel saw `@__nord` propose **applying a Detectron/ViT backbone to LLMs** for operation on a private dataset. On high-throughput inference engines, `@uchihakrishnaa` discovered **ctranslate2** as a potential pick to outperform TGI and vLLM. **Sliding window attention** got a nod as a technique worthy of implementation in Mistral, and the prospect of **adding bounding boxes to architecture** without fine tuning was greeted with interest. `@kyleboddy` voiced issues with parallelizing **Mixtral 8x7b** across GPUs, for which `@orangetin` suggested **FSDP** and **HF Accelerate** as potential solutions.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (15 messages🔥): 
        
- **Limits of Document Recall**: User `@benjoyo` speculated about the limitations of retrieving information from large amounts of data, suggesting that **gigabytes of documents** would not yield perfect recall.
- **Costs Increase with Context Size**: `@teknium` stated that while the cost of context does increase as size increases, it is not an exponential increase.
- **RNN Memory and Compute**: `@euclaise` informed that in **RNN mode**, memory usage is O(1), but it necessitates O(N) compute. Non-RNN mode allows for sub-linear compute, but requires more memory.
- **Recall Efficiency with Mamba**: `@ldj` shared that Mamba achieved over 95% associative recall on sequences of 1 million tokens, outperforming SOTA DNA models trained using Hyena.
- **Context Management and Retrieval as Research Directions**: `@maxwellandrews` suggested that better retrieval (knowledge, not text chunks) and better model context management are both valid and independently useful research directions. This was supported by `@gabriel_syme`, who opined that there's more discovery potential in retrieval vs context, given the existing extensive work on the latter.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (13 messages🔥): 
        
- **Discussion on Data Sovereignty**: In response to `@benjoyo`'s query about potential problems of a platform being hosted in Europe, `@gabriel_syme` raised the issue of **data sovereignty**. 

- **Windows 11 Graphics Card Problems**: `@gabriel_syme` reported that **Windows 11** caused issues with his graphics card, nuking the drivers and rendering them non-functional. 

- **Synthetic Datasets Enthusiasm vs GPU Consumption**: `@max_paperclips` cheered for **synthetic datasets**, to which `@carsonpoole` humorously noted the increased workload for GPUs ("GPUs about to go brrr").

- **Question About Funding Synthetic Datasets**: `@gabriel_syme` sought clarity on whether work might fund synthetic datasets; `@teknium` clarified that it's allowed, and `@gabriel_syme` indicated having noted the terms. 

- **Twitter Ads and Bots**: `@euclaise` raised a point about the influx of ads from **Chai on Twitter**. The discussion broadened with `@metaldragon01` and `@gabriel_syme` remarking on the presence of pervasive porn ads and bots on the platform.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (38 messages🔥): 
        
- **New Language Model WizardCoder-33B-V1.1 Outperforms GPT3.5-Turbo**: `@metaldragon01` shared a [Twitter update](https://fxtwitter.com/WizardLM_AI/status/1742906065359167730) from `@WizardLM_AI` on the release of WizardCoder-33B-V1.1, a new language model that excels in a variety of benchmarks. However, `@giftedgummybee` expressed concern over the lack of available dataset and code for reproducing the data.

- **Humaneval-Test-Set-Alpha Shows Promise**: `@n8programs` humorously claimed that their model "Humaneval-Test-Set-Alpha" can achieve a 100% success rate on the humaneval with less than a megabyte of data.

- **Discussion on Model Composition and Expansion**: `@metaldragon01` shared an [article](https://arxiv.org/abs/2401.02412) introducing CALM (Composition to Augment Language Models), a new way to merge models and give them new capabilities. `@gabriel_syme` speculated on the use of cross attention between an anchor model and LORAs (Locally Optimized Robust Anchors).

- **MobileLLaMA-1.4B-Base Shows Promising Performance**: `@euclaise` shared a link to [MobileLLaMA-1.4B-Base](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base) on HuggingFace, a downscaled LLaMA model that delivers comparable benchmark performance to other recent open-source models. However, `@.benxh` argued that ShearedLLaMA performs better based on benchmarks.

- **TinyLLaMA Awaits Evaluation**: Following a miscommunication, `@qnguyen3` clarified that TinyLLaMA has been completed and is waiting for `@387972437901312000` to run evaluation.

**Links mentioned**:

- [mtgv/MobileLLaMA-1.4B-Base · Hugging Face](https://huggingface.co/mtgv/MobileLLaMA-1.4B-Base)
- [LLM Augmented LLMs: Expanding Capabilities through Composition](https://arxiv.org/abs/2401.02412): Foundational models with billions of parameters wh...
- [Paper page - LLaMA Pro: Progressive LLaMA with Block Expansion](https://huggingface.co/papers/2401.02415)
- [Tweet from WizardLM (@WizardLM_AI)](https://fxtwitter.com/WizardLM_AI/status/1742906065359167730): 🔥 Excited to release WizardCoder-33B-V1.1, the SO...


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (120 messages🔥🔥): 
        
- **UnsLOTH Queries Clarified**: `@gabriel_syme` inquired about the nature of **UnsLOTH**, a LORA trainer with a custom kernel. `@teknium` clarified that it's a proprietary product and the code for a multi-GPU system would not be provided freely. `@beowulfbr` linked an issue from axolotl's Github repository where a user attempted to apply UnsLOTH's optimizations.
- **200k Nous Model Speculations**: While discussing large context models, `@nonameusr` casually asked about the potential of a 200k Nous Hermes model. `@ldj` mentioned that such a model already exists in the form of Nous Capybara. It was then clarified that the query was specific to a Hermes variation of the 200k context model. 
- **Peek into AI Rap Models**: A discussion was initiated by `@euclaise` on the potential existence of an AI rap voice model. Various responses pointed towards services like UberDuck and MAYK, including creating vocals using RVC.
- **AI Trainers on Mac Questions**: `@agcobra1` asked about the potential to train models on a Mac. Although specifics were not provided, `@n8programs` suggested using LORA MLX or Transformers with a MLX backend.
- **Context Extending Code Updates**: There was a conversation about a new code claiming to perform a magical, no-training required, context extension (`@spaceman777`). `@ldj` indicated that users were working on adding Mamba support to the Llama.cpp project, and potentially Mamba would make handling larger context models more practical.

**Links mentioned**:

- [Popular &quot;AI Hub&quot; Discord Taken Down Following Copyright Complaints * TorrentFreak](https://torrentfreak.com/popular-ai-hub-discord-taken-down-following-copyright-complaints-231005/)
- [Uberduck | Make Music with AI Vocals](https://www.uberduck.ai/): Generate high-quality voices by synthesizing your ...
- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/Teknium1/status/1680633119664410624): @yacineMTB @abacaj I&#39;ve told you about a syste...
- [Apply unsloth optimizations · Issue #908 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/908): ⚠️ Please check that this feature request hasn&#39...
- [Your Virtual Music Studio – mayk.it](https://www.mayk.it/): We’re a Virtual Music Studio for next-gen music cr...


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (24 messages🔥): 
        
- **Modifying LLM Architecture**:
    - User `@__nord` raised the question about modifying **LLM's** architectures and training. They are planning to implement a certain paper and if it is successful,.. they will consider adding a Detectron/ViT backbone to it for a private dataset they work with.
- **Inference Engine Alternatives**:
    - `@uchihakrishnaa` expressed a need for inference engines that provide greater throughput than **TGI** and **vLLM** for a fine-tuned **vicuna model**. `@night_w0lf` suggested looking into **ctranslate2** as a potential solution.
- **Discussion on Sliding Window Attention**:
    - User `@lwasinam` inquired about implementing *sliding window attention*, a technique employed by **Mistral**.
- **Incorporating Bounding Boxes into architecture**: 
    - In a discussion featuring `@__nord` and `@max_paperclips`, the potential for modifying attention without fine tuning was discussed. It was suggested that a new pair of keys and values could be projected with the same weights as the original ones.
- **Parallelizing Mixtral Over Multiple GPUs**: 
    - `@kyleboddy` expressed difficulties with parallelizing **Mixtral 8x7b** over multiple GPUs for training and inference. `@orangetin` recommended **FSDP** for training and **HF Accelerate** for inference.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Meshing Multiple ChatGPT Subscriptions**: Users can club multiple subscriptions of ChatGPT together under the Enterprise plan. This was confirmed by `@lugui` in response to `@shalevbartal_38833`'s query. [Check the subscription details here.](https://chat.openai.com/#pricing)
- **Clever ASCII Alteration Proposed for Generative Attribution**: `@iruint` proposed an intriguing solution for text copying issues - alter ASCII character encoding to create detectable corruptions in copied text. `@.dooz` and `@lugui` expressed both interest and uncertainty in this idea.
- **Mobile ChatGPT Now Enjoys PC Compatibility**: According to `@satanhashtag` and `@dystopia78`, users can use Mobile ChatGPT+ subscriptions on their PCs. However, do note, you'll need to purchase from mobile if using PayPal.
- **OpenAI's GPT Store Launch Is on the Horizon**: Exciting times as `@uchenkenta` announced the imminent launch of OpenAI's GPT Store, a platform allowing developers to distribute their custom AI applications. To join, developers need to have a ChatGPT Plus or Enterprise subscription.
- **Free Users and Their GPT Store Access Left in Question**: Concerning GPT Store's launch, `@misangenius` wondered if free users would have access to the custom GPTs. `@muyfashionista` speculated that the cost of OpenAI's APIs might be passed onto consumers through the apps.
- **Debut of DadBot, a Chatmate You Longed for**: `@tdgrpodcast` introduced 'DadBot', inviting users for a chat. [You too can meet DadBot here.](https://chat.openai.com/g/g-OhGvGkxM9-dadbot)
- **Developers' GPT Prompt Limit Hack**: Hit Custom GPT's prompt limit? No problem - you can continue the chat by copying the context into GPT-4, as suggested by `@darthgustav.`.
- **GPT Store Launch Raises Security Concerns**: As GPT Store launch nears, `@r3plac3d` voiced potential security threats like cloning, and called for more robust security measures than those currently recommended by OpenAI.
- **Image Variation in Custom GPT-3 Miffs Users**: `@jungle_jo` observed that their custom GPT-3 producing similar images repeatedly devaluing user's experience. `@eskcanta` suggested adding three user-derived keywords to prompts for more varied outputs.
- **Budding Business Opportunity in Prompt Engineering**: `@iiimandalorianiii` express success in selling a set of AI chatbot prompts to a corporate client for $1500.
- **Need for AI that Follows Instructions Better**: `@zdev26` reported issues with their tailored `GPT 3.5 Turbo` ignoring additional user instructions, asking OpenAI for a solution.
- **Ever Thought About Replacing Words with `ChatGPT`?**: `@dnp_` required assistance for replacing niche-specific words with placeholders in a set text, but dealing with negative words seemed challenging.


**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (20 messages🔥): 
        
- **Multiple ChatGPT Subscriptions Inquiry**: User `@shalevbartal_38833` inquired about purchasing multiple subscriptions of ChatGPT. `@lugui` provided guidance that purchasing multiple subscriptions simultaneously is possible only with an Enterprise plan and also shared the [subscription link](https://chat.openai.com/#pricing).

- **A Solution for Generative Attribution Problem?**: A user `@iruint` made an intriguing suggestion to solve the issue of generative attribution. They proposed altering the ASCII character set, using robot characters instead of standard 'a' and 'i', and applying these changes across non-latin characters. The idea is to make it possible to detect a violation when the character encoding in the copied text is corrupted. `@.dooz` and `@lugui` joined in the discussion, posing questions about the practicality and reliability of such a solution.

- **ChatGPT Subscription Across Devices**: User `@knownx.` asked if a ChatGPT+ subscription purchased on mobile could be used on a PC. They received responses from `@satanhashtag` and `@dystopia78` confirming that this indeed is possible, however, payments would need to be made from the mobile platform if using PayPal.

- **Announcement of GPT Store Launch**: `@uchenkenta` shared [news](https://rebruit.com/openai-set-to-launch-gpt-store-next-week-a-platform-for-custom-ai-apps/) about the upcoming launch of the GPT Store by OpenAI. The GPT Store will be a platform for developers to distribute custom applications built on OpenAI's AI models. Developers will need to follow OpenAI's updated usage policies and brand guidelines and have a ChatGPT Plus or Enterprise subscription. 

- **Access to GPT Store for Free Users**: In relation to the GPT Store launch announcement, `@misangenius` questioned whether free users would have access to custom GPTs from the store. `@muyfashionista` suggested that the cost of OpenAI's APIs might be embedded into the apps and passed onto customers and referenced a discussion about potential [monetization](https://community.openai.com/t/the-future-of-gpts-their-marketplace-monetization-joint-discussion-to-improve-planning/504689).

**Links mentioned**:

[OpenAI Set to Launch GPT Store Next Week: A Platform for Custom AI Apps](https://rebruit.com/openai-set-to-launch-gpt-store-next-week-a-platform-for-custom-ai-apps/): OpenAI is set to introduce the GPT Store, a platfo...


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (69 messages🔥🔥): 
        
- **DadBot Debuts**: `@tdgrpodcast` invited users to test a new AI called **DadBot** they created, providing a link to it (https://chat.openai.com/g/g-OhGvGkxM9-dadbot).
- **GPT Store Launch Announcement**: Users `@pierrunoyt` and `@nickthepaladin` discussed the impending launch of the **GPT Store**, expressing some concerns regarding membership requirements.
- **Helpful Hints for Using Custom GPTs**: `@darthgustav.` provided fellow developers with a workaround for continuing conversations when the prompt limit in a **Custom GPT** has been reached: just copy the context into GPT-4 and continue the chat.
- **Security Concerns with GPT Store Launch**: `@r3plac3d` raised strong concerns about security issues in light of the upcoming **GPT Store** launch. They indicated that previous recommendations from OpenAi, like disabling the code interpreter, are insufficient protection against cloning and other potential threats.
- **Profile Picture Upload Issues**: `@francisrafal` shared difficulties uploading a profile picture to a GPT. It was discovered that using Chrome as a browser resolved the issue, indicating a potential issue with the Brave browser.
- **Limits on Custom GPTs Questioned**: `@holden3967`, `@thepitviper` and others raised concerns about the limitations placed on **Custom GPTs**, such as the 25 prompts/3hours limit. Questions were asked about known limit loopholes, the need for OpenAI plus accounts, and the expectation that paying customers should get higher limits.
- **Adjusting Builder Profile Info**: `@r3plac3d` and `@scargia` discussed adjusting builder profile information on the OpenAI platform, where `@scargia` stated that users can edit their name via the account profile link (https://platform.openai.com/account/profile).


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (38 messages🔥): 
        
- **Struggling with Image Variation**: User `@jungle_jo` noticed that their custom GPT-3 model, which generates images based on a set of guidelines, was creating images with **similar patterns** in 5 out of 10 cases.
- **Lack of randomness**: Upon discussing with `@eskcanta`, they suggested that the AI, being stateless, might be picking the same choices each time. AI's struggle with generating randomness was highlighted.
- **Improving image variation**: `@eskcanta` suggested to rewrite the user's prompt by adding three keywords and then taking inputs from these. This leads to 'true randomness' and more variation in the generated images.
- **Sales in Prompt Engineering**: `@iiimandalorianiii` shared their successful experience of selling a set of prompts for AI chatbots to a corporate client for $1500.
- **Request for Texting Story Prompt**: `@user691378` asked for help to create a prompt that can generate texting stories for social media.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (38 messages🔥): 
        
- **Overcoming `GPT 3.5 Turbo` Ignoring Added User Instructions**: `@zdev26` reports experiencing an issue with their tuned `GPT 3.5 Turbo` chatbot ignoring additional user instructions for outputs. They've noticed that the model frequently ignores guiding prompts such as 'User has provided instructions for the next message: "Ask for her phone number in a funny way"'.
- **Manipulating `ChatGPT` for Word ~Replacement~**: `@dnp_` asks for assistance on how to get `ChatGPT` to copy a set text, but replace certain niche-specific words with a placeholder like "fillintheblank". There seem to be challenges with negative words in the text.
- **Generating Unique AI Images**: `@jungle_jo` reports a problem with their custom `GPT` model which generates images based on a set of guidelines. They have noticed that the model tends to generate similar images repeatedly, lacking in variation. They would like advice on how to improve this.
- **Solutions to Image Variation Problem**: `@eskcanta` provides thoughtful advice for `@jungle_jo`'s problem, suggesting methods to stimulate random image generation. They recommend a rewrite of current instructions, adding three keywords derived from user prompts.
- **Prompt Engineering as a Business Opportunity**: `@iiimandalorianiii` shares an interesting story of selling a set of prompts to a corporation for $1500. They proposed the idea of prompt engineering to the corporation and wrote a set of prompts for their processes.


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Child Risks in LAION 5B**: '@chad_in_the_house' discussed **child pornography appearing in LAION 5B** from findings by Stanford researchers, forcing the removal of certain databases. 
- **Debunking the Myth of OpenClip Going Explicit**: '@thejonasbrothers' and '@pseudoterminalx' concluded, after analysis, that **OpenCLIP** *couldn't generate explicit or illegal images due to data limitations.*
- **The Ups and Downs of Dataset Rips**: '@thejonasbrothers' and '@pseudoterminalx' shared experiences in feeding high-quality ripped data from films and anime into models, detailing hallucination issues and possible solutions. 
- **Terms of Play with Claude**: '@itali4no' referenced a [tweet](https://vxtwitter.com/llmsherpa/status/1742750406898339946?s=46) indicating **Anthropic's terms of use for Claude had changed**, now restricting its use for research, red-teaming, or in competing models' development.
- **aMUSEd to Boost Text-to-Image Processes**: '@thejonasbrothers' highlighted new lightweight **Masked Image Model (MIM) called aMUSEd** via an [arXiv paper](https://arxiv.org/abs/2401.01808) that enhances style learning from single images & speeds up large-scale text-to-image processes.
- **Sculpt Your 2D Images with 3D**: '@thejonasbrothers' shared an [arXiv paper](https://arxiv.org/abs/2401.01702) on **'Image Sculpting'** - a tool that edits 2D images with 3D geometry tools for increased precision.
- **Meet Unicron, the Self-healer for Large-Scale Language Model Training**: '@vrus0188' introduced **Unicron** via a [Reddit post](https://www.reddit.com/r/machinelearningnews/comments/18ye1km/alibaba_researchers_unveil_unicron_an_ai_system/) - an AI by Alibaba researchers designed for efficient self-healing in large-scale language model training.
- **Why Stable Diffusion Struggles with Age Concepts**: '@phryq' and '@JH' discussed the inherent limitations of **Stable Diffusion Models** in representing age-based features and the challenge of sampling out-of-distribution (OOD). Pretraining could offer a solution, although it necessitates comprehensive testing.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (132 messages🔥🔥): 
        
- **Controversy over Illegal Content in LAION 5B**: `@chad_in_the_house` discussed the finding of child pornography within LAION 5B by Stanford researchers, which led to certain databases being taken offline. 

- **Discussions on Potential Generation of Inappropriate Imagery**: Conversation took place between `@thejonasbrothers` and `@pseudoterminalx` regarding the OpenCLIP model's capability (or lack thereof) to generate explicit or illegal imagery, with both agreeing it's unlikely due to limitations in data.

- **Training Models on Dataset Rips**: A detailed conversation occurred between `@thejonasbrothers` and `@pseudoterminalx` on their experiences feeding models high-quality rip data from movies and anime, the challenges they experienced with hallucinations, and the possible solutions they attempted.

- **Consequences of Terms of Service Violation for the use of Claude**: A tweet cited by `@itali4no` mentioned that Anthropic had updated its terms of use for Claude, restricting it from being used for research, red-teaming, or in the development of competing models.

- **Appreciation for MistralAI's Policy on Using Outputs**: `@SegmentationFault` praised MistralAI for its more lenient stance on using the outputs of their models for training other models.


**Links mentioned**:

[Tweet from The LLM Sherpa (free/acc) (@LLMSherpa)](https://vxtwitter.com/llmsherpa/status/1742750406898339946?s=46): Anthropic updated terms.  So, to use Claude, you h...


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (6 messages): 
        
- **Lightweight image model aMUSEd to boost text-to-image generation**: `thejonasbrothers` shared a link to an [arXiv paper](https://arxiv.org/abs/2401.01808) about a new, lightweight **Masked Image Model (MIM)** called **aMUSEd**. This model, developed by Patrick von Platen, aims to speed up text-to-image generation processes and enhance learning additional styles from a single image. This method could revolutionize large-scale text-to-image operations.
- **Image Sculpting: A new way to edit 2D images with 3D tools**: `thejonasbrothers` posted another [arXiv paper](https://arxiv.org/abs/2401.01702) presenting a tool called **Image Sculpting**, which allows for editing 2D images with 3D geometry tools. This novel approach could increase the precision of image editing and enhance the potential of generative models.
- **Unicron: Alibaba's self-healing AI system for language model training**: `vrus0188` shares a [Reddit post](https://www.reddit.com/r/machinelearningnews/comments/18ye1km/alibaba_researchers_unveil_unicron_an_ai_system/) about **Unicron**, an AI system developed by Alibaba researchers. This system is uniquely designed for efficient self-healing in large-scale language model training. 
- **Conditional Velocity Score Estimation Paper from WACV2024**: `wangshuai` posted a PDF link to the best paper from WACV2024 titled "**Conditional Velocity Score Estimation for Image Restoration**". [PDF Link](https://openaccess.thecvf.com/content/WACV2024/papers/Shi_Conditional_Velocity_Score_Estimation_for_Image_Restoration_WACV_2024_paper.pdf)

**Links mentioned**:

- [Image Sculpting: Precise Object Editing with 3D Geometry Control](https://arxiv.org/abs/2401.01702): We present Image Sculpting, a new framework for ed...
- [aMUSEd: An Open MUSE Reproduction](https://arxiv.org/abs/2401.01808): We present aMUSEd, an open-source, lightweight mas...
- [Reddit - Dive into anything](https://www.reddit.com/r/machinelearningnews/comments/18ye1km/alibaba_researchers_unveil_unicron_an_ai_system/)


### ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/) (3 messages): 
        
- **Conceptual Understanding in Stable Diffusion Models**: User `@phryq` questioned the conceptual capability of Stable Diffusion in learning to represent age-based features in images. They queried if the model could interpolate ages it was not directly trained on.
- **Sampling Out-of-Distribution Limits**: `@JH` clarified that, due to the age being modeled by input tokens and not actual numbers, models are typically challenged in interpolating between different ages (30 and 40, for example). This challenge is recognized as sampling out-of-distribution (OOD).
- **Possible Exceptions with Pretrained Models**: `@JH` further added that pretrained models, like a clip encoder used with a text-to-image model, might have already learned to recognize tokens for ages. Therefore, even if it's OOD for your training data, the age might just be in distribution for the pretrained model.
- **Emphasis on Testing and Verifying Model Capability**: For guaranteeing the learned concepts, `@JH` recommended the development of comprehensive tests to periodically check the model's abilities, gauge the extent of training required, and evaluate the potential need for data augmentation.
- **Deeper Insight into Concept formation in Stable Diffusion**: `@phryq` further emphasized on the difference between LLM (that can develop a concept) and Stable Diffusion, suggesting that Stable Diffusion models might be limited to understanding exactly what they've been trained on, without forming deeper "conceptual understandings."


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Data Management Gets Cozier with Hugging Face and DVC**: `@Andyrasika` wrote a blogpost about managing data using Hugging Face and Data Version Control (DVC) [Blogpost Link](https://huggingface.co/blog/Andyrasika/hf-dvc).
- **Bilingual Content Creation Sees a New Star**:  `@manishiitg` showcased their model trained on Hindi and English data, ideal for content writing, classification, and coherent content creation. This model isn’t a fit for coding/maths tasks [Model Link](https://huggingface.co/manishiitg/open-aditi-hi-v1).
- **Whispers of ASR Endpoint Inference Issues**: `@blahblah6407` experienced an issue while creating and testing an endpoint inference for ASR with a finetuned Whisper.
- **Building Budget PCs for Deep Learning**: `@sedthh` sought advice on a budget PC build for deep learning, posing a choice between a 3090 or 4080, or other alternatives in the same price range.
- **Discovering Stable Diffusion XL**: `@8bit888` shared a blog post on Stable Diffusion XL: a guide for image generation without limitations [Link](https://open.substack.com/pub/dedai/p/sdxl?r=15zr&utm_campaign=post&utm_medium=web&showWelcome=true).
- **Bank Statement Image Annotation Automatization**: `@detraxsenpai` sought advice on automating the annotation of bank statement images and making unusual amendments.
- **Chatbot on the Horizon**: `@jryarianto` sought guidance for creating a chatbot able to provide real-time answers from a database while maintaining access controls. `@absolutt1` suggested that the Retrieval Augmented Generation (RAG) system would be a good fit.
- **Gradio 4.13 Lands with a Bang**: `@abidlabs` announced the Gradio 4.13 release, detailing the new features. He also shared the [full changelog](https://www.gradio.app/changelog).
- **Gemini Pro Vision Opens for Testing**: `@aiman1993` shared a Hugging Face Spaces link to the Gemini Pro Vision Streamlit Application, welcoming others to experiment [Link](https://huggingface.co/spaces/disham993/gemini-pro-vision-streamlit-application).
- **Python Packages Unleashed**: `@lawls.net` highlighted the complexity of managing Python packages and encouraged the practice of setting up separate virtual environments for each project.
- **Mimic Mammalian Learning In AI**: `@amylizzle` shared an interesting paper on a new error propagation method that simulates mammalian learning [Link](https://www.nature.com/articles/s41593-023-01514-1).
- **AnimateDiff Takes the Stage**: `@hina_tech` released the [AnimateDiff prompt travel GUI](https://github.com/JojoYay/animatediff-cli-prompt-travel), now available on Gradio.
- **AI Comes to Flashcards for Easy Recall**: `@venkycs` shared a link to AI Hub, a platform that offers AI concept flashcards to help learners [Link](https://ai-hub.app.link/0FyWQhZ25Fb).

**HuggingFace Discord Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 messages): 
        
- **Data Management with Hugging Face and DVC**: `@Andyrasika` wrote a blogpost about streamlining data management with Hugging Face and Data Version Control (DVC). The post explores how DVC and Hugging Face's ecosystem interact to transform how data is managed within projects. [Blogpost Link](https://huggingface.co/blog/Andyrasika/hf-dvc)
- **Piano MIDI Generation with Transformers**: `@afmck` has published a blogpost titled TchAIkovsky, focusing on piano MIDI generation using Transformers. [Blogpost Link](https://huggingface.co/blog/afmck/tchaikovsky)
- **Hindi-English Model for Content Creation and Classifications**: `@manishiitg` has trained a model on Hindi and English data, optimized primarily for content writing, role playing, classification, and generating coherent content. The model is not optimized for coding/maths tasks. [Model Link](https://huggingface.co/manishiitg/open-aditi-hi-v1)
- **Multiple Choice Question Handling with Transformers and PyTorch**: `@Andyrasika` discusses how to leverage Transformers and PyTorch for handling Multiple Choice Questions in a blogpost. [Blogpost Link](https://huggingface.co/blog/Andyrasika/mcq-pytorch-transformers)
- **AI Chatbot for Code Generation and Plot Editing**: `@sophiamyang` showcases an AI chatbot that makes use of Panel and Mixtral 8x7b to run code and edit matplotlib plots. [Blogpost Link](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat)
- **Preventing Data Contamination in LLMs**: In a blogpost by `@rishiraj`, he talks about managing evaluation data contamination during model merging and introduces tools to streamline processes and maintain data integrity. [Blogpost Link](https://huggingface.co/blog/rishiraj/merge-models-without-contamination)
- **Understanding Counting in Probability**: `@ariG23498` writes an instructional blogpost on the importance of counting for understanding probability and its use cases. [Blogpost Link](https://huggingface.co/blog/ariG23498/count-n-objects)
- **Shoe Image Classification Dataset**: `@Andryasika` has created a dataset of 15,000 images of shoes, sandals, and boots, ideal for multiclass classification with deep neural networks. [Dataset Link](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)

**Links mentioned**:

- [Streamlining Data Management with Hugging Face and DVC: A Seamless Integration](https://huggingface.co/blog/Andyrasika/hf-dvc)
- [TchAIkovsky – Piano MIDI Generation with Transformers](https://huggingface.co/blog/afmck/tchaikovsky)
- [manishiitg/open-aditi-hi-v1 · Hugging Face](https://huggingface.co/manishiitg/open-aditi-hi-v1)
- [Leveraging Transformers and PyTorch for Multiple Choice Question Tasks](https://huggingface.co/blog/Andyrasika/mcq-pytorch-transformers)
- [Build an AI Chatbot to Run Code and Tweak plots](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat)
- [Celebrity Look A Like - a Hugging Face Space by tonyassi](https://huggingface.co/spaces/tonyassi/celebrity-look-a-like)
- [Combating Evaluation Data Contamination in LLMs: Strategies for High-Quality Finetuning and Model Merging](https://huggingface.co/blog/rishiraj/merge-models-without-contamination)
- [Counting &#39;n&#39; objects](https://huggingface.co/blog/ariG23498/count-n-objects)
- [Andyrasika/ShoeSandalBootimages · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)


### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (85 messages🔥🔥): 
        
- **Urgent Help With Gradio Requested**: In general queries, `@nigg.pablo` requested urgent help with gradio.
- **Confusion on Huggingface API Usage and Payments**: `@.o.sarge.o.` expressed confusion about their Huggingface API usage and payment expectations, noting a lack of charges despite extensive usage with the "openai/whisper-large-v3" model.
- **Unexpected Issue with ASR Endpoint Inference**: User `@blahblah6407` experienced an issue during the creation and testing of an endpoint inference for ASR with a finetuned Whisper, reporting a specific error related to an unexpected 'ignore_warning' argument and seeking assistance in resolving it.
- **Operating Llama 2 on an M1 Macbook Pro**: User `@sia4030` sought guidance in getting Hugging Face models working with Llama 2 on an M1 Macbook Pro. They were assisted by `@lee0099`, who guided them through resolving issues with the `python convert` command and the lack of a `config.json` file.
- **Difficulty Signing up to Hugging Face**: User `@illumes` reported difficulty signing up to Hugging Face, expressing that process seemed to halt after the CAPTCHA stage. The issue was subsequently addressed by `@sakalys`, who suggested the issue may be due to the use of an ad blocker, or a browser like Brave. They then advised switching to another browser to avoid the CAPTCHA issue.
- **PC Build Advice for Deep Learning**: User `@sedthh` sought advice in building a budget PC for Deep Learning, specifically requesting recommendations between a 3090 or 4080, or other suitable alternatives within the same price range.
- **Inquiring About Interview Footage**: `@vishyouluck` expressed an interest in viewing an interview involving `@504681610373758977` (Sayak Paul), with `@qwerty_qwer` providing the [link](https://www.youtube.com/watch?v=IlIhykPDesE) to the podcast.
- **Potential Strength of AI Model Discussed**: `@not_lain` and `@vipitis` highlighted the release of WizardCoder-33B-V1.1, mentioning its strong start.
- **Problem with Open LLM Evals Submission**: `@kquant` expressed difficulty in uploading models to the open LLM evaluation queue, their model submission appeared to have failed with no clear reason for the failure given.

**Links mentioned**:

- [meta-llama/Llama-2-13b-chat-hf at main](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main)
- [ArtificialThinker Demo on GPU - a Hugging Face Space by lmdemo](https://huggingface.co/spaces/lmdemo/artificialthinker-demo-gpu)
- [HuggingChat](https://huggingface.co/chat)
- [Kquant03/EarthRender-32x7B-bf16_eval_request_False_bfloat16_Original.json · open-llm-leaderboard/requests at main](https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/main/Kquant03/EarthRender-32x7B-bf16_eval_request_False_bfloat16_Original.json)
- [Tweet from WizardLM (@WizardLM_AI)](https://fxtwitter.com/WizardLM_AI/status/1742906065359167730): 🔥 Excited to release WizardCoder-33B-V1.1, the SO...
- [When running as python module - meta-llama/Llama-2-7b-hf does not appear to have a file named config.json  · Issue #26432 · huggingface/transformers](https://github.com/huggingface/transformers/issues/26432): System Info Python 3.10 Transformer 4.32.0.dev0 Tr...
- [GitHub - julien-c/arxiv-to-hf: Chrome extension to add a link from each Arxiv page to the corresponding HF Paper page](https://github.com/julien-c/arxiv-to-hf): Chrome extension to add a link from each Arxiv pag...
- [Llama 2 for Mac M1](https://medium.com/@auslei/llama-2-for-mac-m1-ed67bbd9a0c2): Getting Llama 2 working on Mac M1 with llama.cpp a...
- [How to install Llama2 on a Mac M1 &amp; M2(Mac-Silicon)?](https://medium.com/@movahedi/how-to-install-llama2-on-a-mac-m1-m2-mac-silicon-ab5760bc6ca): An important point to consider regarding Llama2 an...
- [meta-llama/Llama-2-70b-chat-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Accelerating Generative AI Part III: Diffusion, Fast](https://pytorch.org/blog/accelerating-generative-ai-3/): This post is the third part of a multi-series blog...
- [Release v0.25.0: aMUSEd, faster SDXL, interruptable pipelines · huggingface/diffusers](https://github.com/huggingface/diffusers/releases/tag/v0.25.0): aMUSEd  aMUSEd is a lightweight text to image mode...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 messages): 
        
- **Python Package Management Challenges**: User `@lawls.net` highlighted the complexity and the importance of handling Python packages and virtual environments properly. They stressed the importance of using the correct versions of packages and setting up a separate virtual environment for each project.
- **Stable Diffusion XL for Image-Generation**: `@8bit888` shared a blog post titled [Stable Diffusion XL: A tutorial for designing without limitation](https://open.substack.com/pub/dedai/p/sdxl?r=15zr&utm_campaign=post&utm_medium=web&showWelcome=true), It offers a guide to installing and using Stable Diffusion XL, an open-source image generating tool.
- **Journey into Python Packaging and PyPI Publishing**: `@vipitis` mentioned that they are trying to learn about **Python packaging**, specifically optional extras and also **publishing to PyPI**.

**Links mentioned**:

[Install Stable Diffusion XL on MacOS](https://open.substack.com/pub/dedai/p/sdxl?r=15zr&utm_campaign=post&utm_medium=web&showWelcome=true): DALL-E &amp; Midjourney are great but free is bett...


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (4 messages): 
        
- **Mimicking Mammalian Learning Through Error Propagation**: `@amylizzle` shared an interesting [paper](https://www.nature.com/articles/s41593-023-01514-1) presenting **a new error propagation method that mimics mammalian learning**.

- **Quadrilingual `@sia4030`** highlighted their diverse language skills, speaking **English, Swedish, Persian fluently and French at beginner level**.

- **Flashcards for Mastering AI via AIHub**: `@venkycs` shared a [link](https://ai-hub.app.link/0FyWQhZ25Fb) to **AI Hub**, a platform that offers **flashcards to easily master AI concepts** like transfer learning.

- **@dttch Open to New Connections**: `@dttch` expressed openness to forming **new connections** within the servers, illustrating the welcoming and collaborative atmosphere of the community.

**Links mentioned**:

- [undefined](https://ai-hub.app.link/0FyWQhZ25Fb)
- [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation - Nature Neuroscience](https://www.nature.com/articles/s41593-023-01514-1): This paper introduces &#8216;prospective configura...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (8 messages🔥): 
        
- **AnimateDiff Prompt Travel GUI Now on Gradio**: `@hina_tech` shared their new creation, the [AnimateDiff prompt travel GUI](https://github.com/JojoYay/animatediff-cli-prompt-travel), that is now available on Gradio.
- **DreamDrop V1 by ehristoforu**: `@ehristoforu` introduced [DreamDrop V1](https://huggingface.co/openskyml/dreamdrop-v1), a modern model trained on Deliberate V5 with MJLora. They provided optimal settings, negative prompts, and additions for optimal use. Various versions of the model can be found in the [Files tab](https://huggingface.co/openskyml/dreamdrop-v1/tree/main).
- **Discussion on Tokenizing vs. Embedding**: `@lordgrim0033` sought clarity on the difference between tokenizing and embedding. `@torres8552` clarified that **tokenizing breaks down text into words or sub-words and assigns an ID to each token**, while **embedding converts these tokens into high-dimensional vectors**.
- **Don't Cross-Post Reminder**: `@cakiki` gently reminded `@venkycs` **not to cross-post.**

**Links mentioned**:

[GitHub - JojoYay/animatediff-cli-prompt-travel: animatediff prompt travel](https://github.com/JojoYay/animatediff-cli-prompt-travel): animatediff prompt travel. Contribute to JojoYay/a...


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (2 messages): 
        
- **Casual Affirmative Responses**: 
    - Both `@chad_in_the_house` and `@lunarflu` shared positive and affirmatory responses to a previous message or statement. The details of the original message or statement were not provided in the given message history.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **Annotations for bank statement images**: `@detraxsenpai` is seeking advice on automating the annotation of bank statement table images. They express interest in using an existing model that has about 70-80% accuracy and manually tweaking it for full accuracy.
- **Possibility of GPU acceleration**: `@pragma9538` suggested checking if torch.cuda is being leveraged on NVIDIA GPU for improved processing efficiency.
- **Open Collaboration Offer**: `@pragma9538` has expressed openness for collaboration in this area. Those interested can reach out via direct message.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (12 messages🔥): 
        
- **Chatbot Overlord**: `@jryarianto` is seeking guidance for building a chatbot capable of accessing a database and providing real-time answers while maintaining strict access controls. `@absolutt1` suggests the use of the **Retrieval Augmented Generation (RAG)** system for this task, which is suited for assistants.
- **Searching for Learning Paths**: `@notooth` is looking for tutorials to train **Phi-2** or **LLAMA-2** models. Our man of mystery `@babylonkiwu` is also in the hunt and wonders about the feasibility of training **Phi-2.7** and **Mistral 7B** on **free colab** using Qlora.
- **RAG How-to Guide**: To help jryarianto, `@absolutt1` mentions the widespread availability of tutorials on building chatbots using the RAG method, capable of providing contextually relevant responses.
- **Gemini Pro Vision Deploys**: `@aiman1993` shares a Hugging Face Spaces link to the **[Gemini Pro Vision Streamlit Application](https://huggingface.co/spaces/disham993/gemini-pro-vision-streamlit-application)**, inviting others to experiment and test it.
- **RAG Disciple**: `@jryarianto` acknowledges a lack of familiarity with the RAG system but expresses interest in exploring `@absolutt1`'s recommendation further.

**Links mentioned**:

[Gemini Pro Vision Streamlit Application - a Hugging Face Space by disham993](https://huggingface.co/spaces/disham993/gemini-pro-vision-streamlit-application)


### ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/) (1 messages): 
        
- **Gradio 4.13 Launch Announced**: `@abidlabs` announced the launch of **Gradio 4.13** with a post listing the new features and fixes including Button components fix, `.select()` event fix for image selection, Chatbot and Model3D component fixes, security enhancements for the FileExplorer component, and ensured compatibility with Python 3.12. He also shared the [full changelog](https://www.gradio.app/changelog).
- **Lighting up Lite**: Thanks to [@whitphx](https://github.com/whitphx), AnnotatedImage support on Wasm has been introduced in Lite.
- **Developers, Join the SharedWorker mode**: Another thanks to [@whitphx](https://github.com/whitphx) for adding a development instruction for lite in SharedWorker mode.
- **Functional Test Fixes**: `@aliabid94` deserves a shout-out for fixing functional tests.

**Links mentioned**:

[Gradio Changelog](https://www.gradio.app/changelog): Gradio Changelog and Release Notes


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity's Series B Celebrations**: Perplexity successfully raised **$73.6M in Series B Funding** led by big names like IVP, NVIDIA, Jeff Bezos and more, aiming to create the world's fastest and most accurate answers platform. Perplexity's current phenomenol performance includes 10M monthly users and over a million mobile app installations. [@AravSrinivas's Tweet](https://fxtwitter.com/AravSrinivas/status/1742918329797574709?s=20) spread the news further and attracted congratulations from various users in the Discord guild. 

- **Queries on Perplexity's Internet Interactive Capabilities**: In the experimental mode, Perplexity AI does have internet access but it cannot cite references yet. Users were curious about this aspect and are looking forward to new developments, which supposedly include citation features. [@nqiwbh07r44p's query](https://discord.com/channels/1047197230748151888/1047649527299055688/) sparked this discussion.

- **Incorporating PPLX API** : Beyond the Perplexity platform, the PPLX API can help developers deploy larg°e language models into their software. The Perplexity API, introduced in this [blog post](https://blog.perplexity.ai/blog/introducing-pplx-api), provides a fast inference and an easy-to-deploy system. A future feature request includes the return of raw snippets and reference links as well as setting the number of snippets in Perplexity API's online models. 

- **Exploring Perplexity's Online Models**: Perplexity's online models, featured in this [blog post](https://blog.perplexity.ai/blog/introducing-pplx-online-llms), provides factual and up-to-date responses in their search. However, users in the Discord guild also noticed Perplexity's absence of Mixtral for online responses.

- **Space Ed for Tech Enthusiasts**: A [blog post](https://takeitpersonelly.com/2023/04/06/8-skills-you-should-master-if-you-want-to-work-in-space-tech/) highlights the top 8 skills needed for anyone keen on joining the space tech industry. With innovative companies like SpaceX and Blue Origin leading the way, this is an area of interest for technology enthusiasts.

**Perplexity AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/) (1 messages): 
        
- **Perplexity Raises $73.6 million in Series B Funding**: User `@enigmagi` announces that Perplexity raised **$73.6 million in Series B Funding** led by IVP with participation from NVIDIA, NEA, Bessemer, Elad Gil, Jeff Bezos, among others. The company dreams of building the world's most accurate and fastest platform for answers. 
    - A blog post was shared where readers can look up more details: [blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round](https://blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round)
- **Perplexity's Success and Future Plans**: They reported achieving **10M monthly users** and serving more than half a billion queries in 2023. A *quote taken directly from the announcement post*: *"Our ambition is to serve the entire planet’s unbounded curiosity, and we’re just getting started."*
- **Milestone in Mobile**: The company also revealed that over a million users have installed their mobile apps, both on iOS and Android. 
- **Challenging Google's Dominance**: Perplexity, according to a linked WSJ article, is backed by Jeff Bezos and some venture capitalists who believe that AI will upend the way people find information online, thus challenging the dominance of Google in web search: [www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger](https://www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger-using-ai-to-try-to-upend-internet-search-0859bda6)
- **Impressive Startup Growth**: Despite being started less than two years ago and having fewer than 40 employees, Perplexity's product is used by roughly 10 million people monthly.


**Links mentioned**:

- [Perplexity Raises Series B Funding Round ](https://blog.perplexity.ai/blog/perplexity-raises-series-b-funding-round): Announcing Perplexity's Series B Funding Round
- [WSJ News Exclusive | Jeff Bezos Bets on a Google Challenger Using AI to Try to Upend Internet Search](https://www.wsj.com/tech/ai/jeff-bezos-bets-on-a-google-challenger-using-ai-to-try-to-upend-internet-search-0859bda6): Perplexity, with a fraction of Google’s users, rai...


### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (96 messages🔥🔥): 
        
- **Experimental Perplexity AI Internet Usage**: User `@nqiwbh07r44p` inquired if the experimental mode of Perplexity AI has access to the internet and how to make it show its references. `@giddz` and `@icelavaman` confirmed that the experimental model has internet access but cannot currently cite references, a feature that is supposedly on the roadmap.
- **Becoming a Perplexity AI Tester**: User `@nqiwbh07r44p` asked about becoming a tester for Perplexity AI. `@giddz` provided a Discord link and `@icelavaman` mentioned that the program is currently closed.
- **Perplexity vs Other AI Models**: User `@marcopaone` asked for opinions on which AI model to choose between Gemi Pro, GPT-4, and Claude 2.1. `@icelavaman` recommended GPT-4. 
- **Perplexity AI App and VPN**: User `@mares1317` noted that with VPN, the web version of Perplexity only requires to check the box of Cloudflare and the app has no issues. User `@icelavaman` pointed out that this might not be the case for everyone and it depends on the VPN provider/hotspot, yet `@mares1317` insisted that no problems occur with a good VPN provider like Proton.
- **Perplexity AI Funding News**: Users `@blackwhitegrey`, `@giddz`, `@billbuttliquor`, `@serpentineuk`, `@theoutbacklp`, `@keef_kahn`, and `@theoutbacklp` congratulated Perplexity team on their new funding round announcement, which valued Perplexity at $520 million. The news includes notable investors such as Jeff Bezos and Nvidia.

**Links mentioned**:

- [AI-powered search engine Perplexity AI, now valued at $520M, raises $73.6M | TechCrunch](https://techcrunch.com/2024/01/04/ai-powered-search-engine-perplexity-ai-now-valued-at-520m-raises-70m/): Perplexity AI, a search engine heavily leveraging ...
- [Perplexity Careers](https://blog.perplexity.ai/careers): Join our team in shaping the future of search and ...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (4 messages): 
        
- **Space Tech Skills to Master**: `@madhusudhan7` shared a [blog post](https://takeitpersonelly.com/2023/04/06/8-skills-you-should-master-if-you-want-to-work-in-space-tech/) about the **8 skills one needs to master** to work in the growing space tech industry. The post mentions that the space tech industry is innovating and companies like SpaceX and Blue Origin are making space travel more accessible.

- **Arav Srinivas Funding Announcement**: `@icelavaman` shared a [tweet](https://fxtwitter.com/AravSrinivas/status/1742918329797574709?s=20) from `@AravSrinivas` announcing that they have raised **$73.6M at a $520M valuation**. The funding round was led by IVP, with participation from NVIDIA, Jeff Bezos, `@tobi`, Databricks, `@naval`, `@rauchg`, `@balajis`, among others.

- **Search Queries on Perplexity AI**: Both `@_joewei` and `@maxymus85` shared [queries](https://www.perplexity.ai/search/nzapzj6zTcqlwR9QwlycOw?s=c#5b90b3bf-a208-48af-819d-14c830975540) from the [Perplexity AI database](https://www.perplexity.ai/search/Interestingly-way-before-LaXrL30YTHGct3tlZPtKeA?s=c), though the content or purpose of these search queries are not specified in the messages.

**Links mentioned**:

- [8 Skills You Should Master If You Want to Work in Space Tech - Take It Personel-ly](https://takeitpersonelly.com/2023/04/06/8-skills-you-should-master-if-you-want-to-work-in-space-tech/): If you're interested in pursuing a career in the s...
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://fxtwitter.com/AravSrinivas/status/1742918329797574709?s=20): Excited to announce we&#39;ve raised 73.6M$ at 520...


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (9 messages🔥): 
        
- **Expanding Perplexity API features**: `@maxwellandrews` requests the option to return both raw snippets and reference links for online models via the Perplexity API, and suggests providing the ability to set the number of snippets to return. They also suggest the possibility of a separate search API and questions the lack of Mixtral for online responses. 
- **PPLX API vs Perplexity**:
    - `@blackwhitegrey` asks about the advantages of using PPLX API over Perplexity, and `@icelavaman` responds that the API allows developers to integrate LLM's into their products.
    - `@icelavaman` then clarifies that online LLM's can search in response to `@blackwhitegrey`'s query about the API's capabilities.
- **Perplexity API Introduction**: A [link](https://blog.perplexity.ai/blog/introducing-pplx-api) to the introduction of pplx-api on the Perplexity blog is shared by `@icelavaman`, which provides information about the api's features like ease of use, fast inference, and reliable infrastructure.
- **Perplexity online models**: `@icelavaman` shares another [link](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) from the Perplexity blog about the introduction of perplexity's online models and their unique advantage of providing factual and up-to-date responses.
- **Adding Perplexity Models to Typingmind**: `@blackwhitegrey` inquires about how to add Perplexity models to Typingmind.

**Links mentioned**:

- [Introducing pplx-api ](https://blog.perplexity.ai/blog/introducing-pplx-api): Perplexity Lab's fast and efficient API for open-s...
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Anthrowpic's Terms Ruffle Feathers**: `@stellaathena` criticized **Anthrowpic's Terms of Service** for being overly strict, potentially hindering research and competition, based on a [tweet from `@LLMSherpa`](https://vxtwitter.com/llmsherpa/status/1742750406898339946?s=46). Members speculate on the possible motivations behind these terms.
- **Possible Legal Implications of Anthrowpic's Terms**: `@fern.bear` voiced concern about the potential legal issues associated with Anthrowpic's terms, noting it might pose complications for businesses who use their services.
- **Twitter Analysis Project Takes Flight**: `@sirmalamute` proposed an open-source Twitter analysis project using **NLP** and showed openness to collaborations, feature-feedback, and project explorations.
- **Early Termination in Neural Networks Turns Heads**: Discussion initiated by `@gabriel_syme` about early exit strategy in deep neural networks drew attention and suggestions like looking into Adaptive Computation Time and expert-specific stopping in Mixture of Experts (MoEs).
- **A Potential Roadblock in the HF Street**: `@micpie` noticed a potential issue in `lm_eval/api/task.py` due to a change in Hugging Face's dataset structure, discussed by `@hailey_schoelkopf` and `@micpie`.
- **Buzz Around Document Packing in GPT-Neox**: `@tastybucketofrice` and `@ad8e` delved into the efficiency of document packing in GPT-Neox, with ad8e hinting at a more efficient packing scheme mentioned in a [professional paper](https://arxiv.org/abs/2107.02027) and expressing willingness to contribute to the project.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (44 messages🔥): 
        
- **Anthrowpic's Terms of Service Deemed Harsh**: `@stellaathena`, citing a [tweet](https://vxtwitter.com/llmsherpa/status/1742750406898339946?s=46) from `@LLMSherpa`, criticized **Anthrowpic's Terms of Service** for being strict and possibly stifling research and competition. The terms of use apparently demand users to halt activities when a breach is claimed. Some users speculated this either as a defensive move or a sign of the company's struggle.
- **Clarity Over Terms, Research Restrictions**: `@thatspysaspy` sought clarification about the "no research" claim in Anthrowpic's terms. `@stellaathena` and `@mrgonao` pointed to the term banning development or training of models, which could be interpreted as prohibiting substantial areas of AI research.
- **Terms Deemed Too Restrictive, Possibly Problematic**: `@fern.bear` voiced concern about the possible legal implications of waiving opposition to injunctive relief in the terms, noting it could pose complications for businesses interested in using Anthropic's services.
- **Proposed Open Source Twitter Analysis Project**: `@sirmalamute` proposed an open-source Twitter analysis project using **NLP** with features like sentiment analysis, polarity check, political affiliation determination, and so on. The user is open to feedback, exploring possible features and collaboration on the project. The initial aim is to create a toolkit rather than a traditional research paper or database.
- **Challenges and Considerations for Twitter Analysis Project**: `@ad8e` queried `@sirmalamute` on the project's specifics, including handling necessary labels for analysis. `@sirmalamute` mentioned potential sources or methods for labels and discussed the possible need to navigate Twitter's policy when considering distribution of tweets for analysis.

**Links mentioned**:

[Tweet from The LLM Sherpa (free/acc) (@LLMSherpa)](https://vxtwitter.com/llmsherpa/status/1742750406898339946?s=46): Anthropic updated terms.  So, to use Claude, you h...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (20 messages🔥): 
        
- **Early Exit in Neural Networks**: User `@gabriel_syme` initiated a discussion on early exit strategy in deep neural networks, particularly in regards to terminating and outputting at an intermediate layer instead of the last one. `@thooton` suggested looking into Adaptive Computation Time, which `@gabriel_syme` appreciated. The topic then evolved into discussing stopping at specific experts/layers in Mixture of Experts (MoEs).
- **Papers on Adaptive Computation and Transformers**: `@thisisniq` and `@zphang` shared links to [papers](https://arxiv.org/pdf/1603.08983.pdf) and [more recent ones](https://arxiv.org/abs/1807.03819, https://arxiv.org/abs/2207.07061) on adaptive computation time and transformer models respectively.
- **Multilingual Model for Long-sequence Classification**: User `@_michaelsh` sought recommendations for a multilingual model designed for long-sequence classification tasks. `@stellaathena` recommended mT5 or BLOOM.
- **Point2CAD Project Shared**: User `@digthatdata` shared a link to a GitHub repository related to the [Point2CAD project](https://github.com/YujiaLiu76/point2cad).
- **Opinion on CALM and AI Research Communication**: `@gabriel_syme` expressed their thoughts on the Composition to Augment Language Models (CALM) approach and its comparison to LoRA. They also criticized the communication style of a tweet they found, arguing it didn't offer practical solutions.

**Links mentioned**:

- [LLM Augmented LLMs: Expanding Capabilities through Composition](https://arxiv.org/abs/2401.02412): Foundational models with billions of parameters wh...
- [Universal Transformers](https://arxiv.org/abs/1807.03819): Recurrent neural networks (RNNs) sequentially proc...
- [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061): Recent advances in Transformer-based large languag...
- [GitHub - YujiaLiu76/point2cad: Code for &quot;Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds&quot;](https://github.com/YujiaLiu76/point2cad): Code for &quot;Point2CAD: Reverse Engineering CAD ...


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (5 messages): 
        
- **HF Datasets changes affect lm-thunderdome**: `@micpie` noticed HF datasets updates might have caused issues with previous commits and required commenting out code in `lm_eval/api/task.py`. Issue was brought by the removal of `name=self.DATASET_NAME` in line 732.
- **Potential issue due to HF's transition**: `@hailey_schoelkopf` noted that the dataset issue might be related to HF moving away from dataset loading scripts and using `trust_remote_code=True`.
- **Pull request merged**: `@hailey_schoelkopf` announced that a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1243) in the `lm-evaluation-harness` repo which removes the `self.dataset_path post_init process` has been merged.
- **Toxigen evaluation questioned**: `@johnnysands` raised a question regarding the evaluation method used for the Toxigen dataset, asking whether turning it into a binary classification task is standard practice as mentioned in the Toxigen paper or unique to lm-eval-harness.

**Links mentioned**:

[Remove self.dataset_path post_init process by lintangsutawika · Pull Request #1243 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1243)


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (15 messages🔥): 
        
- **Access to VM and updates on Documentation/Cleanup**: `@tastybucketofrice` offered VM access to a user and inquired about documentation and cleanup progress.

- **Efficient Timing in Torch CUDA**: `@tastybucketofrice` recommended the use of `torch.cuda.Event` timers for accurate timing in CUDA-based code, specifically within a distributed setting, citing examples from [EleutherAI's cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication).

- **Tracking Upstream on GPT-NeoX**: `@tastybucketofrice` explained why GPT-NeoX tracks upstream, focusing on usability, support for multiple systems, readability, and interpretability. Cherry-picking of optimizations is done as they mature upstream, and they are open to new upstream-tracking workflows.

- **Efficiency of Document Packing in GPT-NeoX**: `@ad8e` and `@tastybucketofrice` discussed document packing in GPT-NeoX, with `@ad8e` pointing out the potential inefficiencies of the current system. They also referenced a packing scheme from a [professional paper](https://arxiv.org/abs/2107.02027). `@tastybucketofrice` offered to add this to the development roadmap upon receiving solid evidence of its effectiveness.

- **Potential Code Contributions for Document Packing**: `@ad8e` indicated that they might be able to submit a PR for improving document packing in the future. `@tastybucketofrice` and `@hailey_schoelkopf` referenced previous work done on sequence packing without attending to one another and potential codes for efficient packing. `@hailey_schoelkopf` suggested reaching out to the lead author of a [published paper](https://arxiv.org/abs/2310.10638) for code related to this matter.

**Links mentioned**:

- [In-Context Pretraining: Language Modeling Beyond Document Boundaries](https://arxiv.org/abs/2310.10638): Large language models (LMs) are currently trained ...
- [Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance](https://arxiv.org/abs/2107.02027): Effective training of today&#39;s large language m...
- [cookbook/benchmarks/communication at main · EleutherAI/cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication): Deep learning for dummies. All the practical detai...
- [gpt-neox/tools/datasets/preprocess_data_with_mask.py at e5a7ea71e96eeada636c9612036dc85e886d973d · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/e5a7ea71e96eeada636c9612036dc85e886d973d/tools/datasets/preprocess_data_with_mask.py#L361): An implementation of model parallel autoregressive...
- [GitHub - EleutherAI/gpt-neox at multitask](https://github.com/EleutherAI/gpt-neox/tree/multitask): An implementation of model parallel autoregressive...
- [GitHub - EleutherAI/gpt-neox at FIM](https://github.com/EleutherAI/gpt-neox/tree/FIM): An implementation of model parallel autoregressive...


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mixtral steps into the Spotlight**: @i_am_dom announced that **Mixtral** is now [available on Replicate](https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1) with new updates. User also delved into nuances of syntax for incorporating system prompts in Mistral models and Mixtral's sensitivity to prompt formatting based on their experience.
- **Chatting Effectively with Mistral**: @gbourdin and @lovis07 expressed a need for guidelines on effectively using Mistral as a chatbot. @i_am_dom responded with an applicable Python [script](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py).
- **RAM Impact on Mixtral's Performance**: @arianagrandefan4024 queried if DDR5 RAM would boost Mixtral's performance. @someone13574 clarified that it would help if running Mixtral on a CPU, otherwise, it wouldn't present a significant advantage.
- **Instructing the Mixtral with System Prompts**: @nickbro0355 was interested in how to make Mixtral work with a system prompt. @bdambrosio suggested using the full llama-2 template or inserting a system message within `<<SYS>>[message]<</SYS>>`.
- **Running Models on Local Devices & Apple Silicon**: Suggestions around running models locally and using Apple's M2 Neural Engine for accelerating Mistral models were discussed, with @bdambrosio mentioning the potential of OrangePi 5B and @jdo300 curious about leveraging the new MLX framework and the M2's neural engine.
- **Opportunity for Engineer Fine-Tuners**: User @arkalius75 appealed for a skilled engineer fine-tuner for a mission—compensation included. Interested parties can **reach out via direct message**.
- **Debating AGI and AI Perceptions**: Users discussed whether GPT-4 can be considered a weak AGI, how AI achievements become a moving target over time, the inadequacy of our understanding of intelligence and how substantial is GPT-4's intelligence as compared to ordinary animals.
- **La Platform Impressions & Questions**: User @acast_37857 shared their positive impressions of "la platform" and raised questions about implementing "mistral-embed". Related, @sublimatorniq spotted a potential issue with abrupt stops related to `\[.*?\]` patterns in the prompt data and Mixtral's stop parameters, humorously deciding to start including seeds in all their requests.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (43 messages🔥): 
        
- **Mixtral Now Available on Replicate**: `@i_am_dom` shared a link to the dedicated [Mixtral](https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1) page on Replicate, revealing new updates <i>("I meant this btw. Not censored anymore suddenly.")</i>.

- **Formatting System Prompts in Mistral Models**: An in-depth discussion was observed regarding the correct syntax for incorporating system prompts in Mistral models. `@i_am_dom` shared a detailed example from [Huggingface Spaces](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py) and clarified that the `<<SYS>>` format is used with LLaMA, not Mistral.

- **Mixtral's Sensitivity to Prompt Formatting**: `@i_am_dom` highlighted that from his experience, Mistral's output quality is heavily influenced by the format of the input prompt, based on a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/) discussing the same.

- **Effective Use of Mistral as a Chatbot**: `@gbourdin` and `@lovis07` expressed a need for clear documentation or guidelines on how to effectively use Mistral as a chatbot with history and rag. `@i_am_dom` responded with a link to a [Python script](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py) as a practical solution.

- **RAM Specifications Affecting Mixtral Performance**: `@arianagrandefan4024` queried if moving from DDR4 to DDR5 RAM would improve Mixtral's performance. `@someone13574` clarified that it would help if running Mixtral on a CPU, otherwise, it wouldn't present a significant advantage.

**Links mentioned**:

- [Tiktokenizer](https://tiktokenizer.vercel.app/)
- [app.py · openskyml/mixtral-46.7b-chat at main](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py)
- [mistralai/Mixtral-8x7B-Instruct-v0.1 - API Reference - DeepInfra](https://deepinfra.com/mistralai/Mixtral-8x7B-Instruct-v0.1/api): Mixtral mixture of expert model from Mistral AI. T...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/)
- [text-generation-webui/instruction-templates/Mistral.yaml at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml): A Gradio web UI for Large Language Models. Support...
- [FastChat-release/fastchat/conversation.py at 2855bf974f0973f85adb2bb7a9d075255b353ecf · mistralai/FastChat-release](https://github.com/mistralai/FastChat-release/blob/2855bf974f0973f85adb2bb7a9d075255b353ecf/fastchat/conversation.py#L846): An open platform for training, serving, and evalua...
- [mistralai/mixtral-8x7b-instruct-v0.1 – Run with an API on Replicate](https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1)


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (5 messages): 
        
- **Does Mistral Medium support function calls?**: `@rantash68` raised a query about the capability of **Mistral Medium** to have "function call" feature akin to **GPT-4**. But, they couldn't find any definitive answer to this question.
- **Mistral's language prowess questioned**: `@acast_37857` sparked a discussion about the language capability of **Mistral Tiny**. Despite the bot being marketed as English-speaking, the user noticed it responding accurately in French. The tongue-in-cheek response by `.superintendent` points out that Mistral is a French company.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (10 messages🔥): 
        
- **Inquiring about System Prompts with Mixtral**: User `@nickbro0355` was interested in how to make mixtral instruct use a system prompt. `@bdambrosio` suggested using the full llama-2 template or trying to include a system message within `<<SYS>>[message]<</SYS>>`.
- **Running Models on Local Devices:** `@choudhary_sahab101` asked for suggestions regarding the most optimal way to run the model on a local device. `@bdambrosio` mentioned the potential of OrangePi 5B, a local 6TOPS unit that supports most Pytorch including transformers.
- **Utilizing Apple's M2 Neural Engine for Models:** `@jdo300` is curious about what API back-end would be the best to run Mistral models on Apple silicon. Specifically seeking to leverage the new MLX framework and the M2's neural engine to accelerate model inference.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **Calling all engineer fine-tuners for a paid mission**: `@arkalius75` is on the search for a skilled engineer fine-tuner for a mission—compensation included. Interested parties are encouraged to **send a direct message** to `@arkalius75`.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (15 messages🔥): 
        
- **Is GPT-4 a Weak AGI?**: In a discussion about GPT-4 and AGI, `@poltronsuperstar` questioned whether **GPT-4**, which is above average human in many tasks, could be termed as a weak *AGI*. `@blueridanus` responded that in a way, yes, GPT-4 is a weak AGI. 
- **Changing Perception of AI Over Time**: `@poltronsuperstar` also raised a concern about how AI achievements become a moving target over time. For instance, if GPT was shown to someone from 2004, they would probably consider it an AGI. 
- **The Unknowns of Intelligence**: `@blueridanus` voiced that we don't understand intelligence enough to define foolproof criteria for it. It's the lack of understanding that might be preventing us from creating an AGI. 
- **Comparative Intelligence of GPT-4 and Animals**: At the same time, `@blueridanus` also stated that in many meaningful ways, GPT-4 is *dumber than very ordinary animals are*.
- **Defining AGI – A Moving Goalpost?**: `@duck` suggested to consider AGI as a *range* rather than a single point. The idea of an AI being similar to an infant progressing to become an adult was floated as a possible interpretation for AGI.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (4 messages): 
        
- **Introduction to "la platform" and Questions on "mistral-embed"**: User `@acast_37857` shared their positive first impressions of "la platform" and asked for guidance on using "mistral-embed", unclear from which platform to implement it.
- **Abrupt Stops Issue Detected**: User `@sublimatorniq` noticed a correlation between abrupt stops and the presence of `\[.*?\]` patterns in the prompt data. Upon their removal, the abrupt stops seemed to cease, suggesting that the issue could stem from these patterns' interactions with mixtral's stop parameters.
- **Potential Issue with Mixtral Stop Parameters**: `@sublimatorniq` noted that Mixtral for ollama is configured with `PARAMETER stop "[INST]"` and `PARAMETER stop "[/INST]"`, which might be causing confusions leading to unexpected stops.
- **Sending Seeds with Requests**: After the observed issue, `@sublimatorniq` humorously indicated their decision to start including seeds in all their requests.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Confusion Angles fire up a debate**: In a detailed image posted by `@yamashi`, users were thrown into a discussion on which angle marked - A, B, C, or D - is the correct one.
- **Closer Gap vs Lower Loss**: `@nafnlaus00` sparked a conversation saying that *"it's better to aim for minimizing the distance between eval_loss and train_loss vs only caring about getting as low of an eval_loss as possible"*, especially when handling unclean datasets. 
- **Early Beta of RL Training announced**: `@caseus_` revealed that RL (Reinforcement Learning) training is merged and now supports DPO and IPO. Though in its beta stages, the community's suggestions and pull requests are much appreciated to refine the development.
- **"Fine-Tuning Mixtral" Talk arises**: Potential issues with fine-tuning Mixtral were brought to light by `@dangfutures`. However, the query remained unanswered.
- **Curriculum Learning Idea pops up in Axolotl-dev**: `@suikamelon` showed interest in implementing curriculum learning, a process where the model is trained starting with "easy" samples. [@Caseus_](https://github.com/OpenAccess-AI-Collective/axolotl/blob/59b2d302c8780ed83e6a0201b741574ee51a1a5e/src/axolotl/utils/data.py#L324) suggested considering yaml for disabling shuffle. 
- **Multigpu Config Hitch**: `@qeternity` had issues with axolotl's deepspeed stages and asked for multi-gpu config examples. The user was then recommended to refer to the nccl.md docs by `@caseus_`.
- **Application of User Feedback**: `@athenawisdoms` asked how user feedback on responses could be beneficial to model improvement, to which `@noobmaster29` floated the idea of setting up a reinforcement pipeline.
- **The Synthetic dataset size conundrum**: `@antti_45069` raised a query regarding the size of a synthetic dataset in comparison to typical code datasets which usually are of the range **100k+**.
- **Introduction of Search RAG API**: The availability of the [**Search RAG API**](https://www.sciphi.ai/) was announced by `@emrgnt_cmplxty`. This tool appears promising for challenges relating to synthetic data and grounding responses.
- **Contextual Connotations and Misspellings**: A potential misspelling in the phrase "and be vary of the context" was caught by `@nafnlaus00`, but `@le_mess` clarified that it doesn't affect overall performance.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (18 messages🔥): 
        
- **Uncertain Angles sowing confusion**: User `@yamashi` posted an image with multiple angles marked and caused a small confusion among users, asking for the correct angle – A, B, C, or D? 
- **A call for more general solutions rather than aiming for the lowest eval_loss**: `@nafnlaus00` expressed the perspective that it's better to aim for *[minimizing the distance between eval_loss and train_loss vs only caring about getting as low of an eval_loss as possible]*, particularly in situations where the dataset isn't clean.
- **Announcement of Reinforcement Learning (RL) training with chatml support**: `@caseus_` shared that RL training is now merged and supports DPO and IPO, but is currently in beta and needs further polishing. Open to suggestions/Pull Requests from the community, Caseus_ confirmed it is in its early stages and welcomes any external assistance.
- **Non-English Model Tuning**: User `@noobmaster29` provided a helpful link (https://arxiv.org/pdf/2401.01854.pdf) for non-English model tuners.
- **Potential issues with fine-tuning mixtral**: `@dangfutures` queried if there were still issues fine-tuning mixtral. The question remained unanswered within the provided chat history.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (7 messages): 
        
- **Disabling Shuffling in Axolotl**: `@suikamelon` questioned how to disable the shuffling feature in [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/59b2d302c8780ed83e6a0201b741574ee51a1a5e/src/axolotl/utils/data.py#L324) and later confirmed it seems to work after making modifications. `@caseus_` suggested considering a yaml setting for it.
- **Concept of Curriculum Learning**: `@suikamelon` expressed interest in experimenting with the concept of "curriculum learning" by training the model starting with "easy" samples.
- **Sample Packing Randomization**: `@caseus_` mentioned that randomization occurs when using sample packing, but `@suikamelon` confirmed having disabled it.
- **Suggestion for Windowed Random Sampler**: `@caseus_` proposed the idea of incorporating a windowed random sampler in future development.

**Links mentioned**:

[axolotl/src/axolotl/utils/data.py at 59b2d302c8780ed83e6a0201b741574ee51a1a5e · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/59b2d302c8780ed83e6a0201b741574ee51a1a5e/src/axolotl/utils/data.py#L324): Go ahead and axolotl questions. Contribute to Open...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (24 messages🔥): 
        
- **Mistral Fine-Tune Query**: `@henriklied` asked if 8-bit lora and 8192 sequence length are feasible for fine-tuning mistral on 3x4090's with a dataset containing 430,791,018 tokens. `@nanobitz` responded that if it fits, it'll just take some time.
- **VRAM Queries for qlora Finetunes**: `@leoandlibe` inquired about the amount of VRAM required for 13B/34B/70B qlora finetunes. To which, `@nanobitz` replied that to load 13B, it takes roughly 13GB VRAM for qlora + optimizer + batch size.
- **Script Automation**: `@athenawisdoms` questioned about automatically running a second command/script after the axolotl command has finished or crashed. `@leoandlibe` suggested using python subprocess run to listen and trigger the required actions.
- **Axolotl's Multiturn Chat Modelling**: `@evil_malloc` questioned if axolotl is suited for multi-turn chat models and how the model is trained. `@nanobitz` explained that the model is trained on all assistant messages.
- **Multigpu Config Example**: `@qeternity` asked examples on multigpu config as they encountered issues with axolotl's deepspeed stages during dataset prep. `@caseus_` recommended them to check out the nccl.md docs.
- **User Feedback's Usage**: `@athenawisdoms` queried about how the user feedback on generated responses can be employed to improve the model. In response, `@noobmaster29` suggested possibly setting up a reinforcement pipeline.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (2 messages): 
        
- **Synthetic dataset size debate**: `@antti_45069` inquired about an undisclosed language learning model and remarked that a **synthetic dataset** of 1780 rows is quite small compared to other code datasets that usually range in the **100k+** range. 
- **Introduction of Search RAG API**: `@emrgnt_cmplxty` announced that the [**Search RAG API**](https://www.sciphi.ai/), suitable for synthetic data and grounding responses, is now available for users to experiment with.


### ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (2 messages): 
        
- **Watch out for Context**: `@nafnlaus00` pointed out a potential misspelling in the phrase "and be vary of the context".
- **Misspelling doesn't affect performance**: `@le_mess` confirmed the misspelling, but noted that it **doesn't hurt performance**.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Call for Better Conversational Retrieval**: `@irfansyah5572` expressed that their **ConversationalRetrievalChain** setup was returning all found source documents, not just the relevant ones. `@seththunder` suggested a solution involving a `similarity_score_threshold`.
- **Langchain Out, Custom Utils In**: `@atefyamin` and `@evolutionstepper` voiced strong criticisms of **Langchain** and hinted at developing their own utilities.
- **On the Hunt for Image Resizers**: `@rajib2189` initiated a hunt for preferred **image resizing packages** within the tech community.
- **RAG Meets Tabular Data**: `@michael_71751` asked for guidance on using a **RAG with tabular data input and output**, along with transforming tabular data in a linguistic context.
- **Markdown Love in LCEL**: `@cryptossssun` inquired about **selecting and loading markdown files** from a local directory. `@seththunder` proposed using `DirectoryLoader`.
- **LLMChain vs ConversationChain Showdown**: `@nav1106` sought clarity on the **differences between LLMChain and ConversationChain**, with `@seththunder` suggesting **ConversationChain** is the better bet for simple conversational contexts.
- **Pipeline Dreams with MultiChains**: `@seefrog` was curious if it was feasible to **connect multi-chains using a pipeline operator**, confirmed by `@seth_thunder`.
- **GCP Adventures in JVMs**: `@johnda98` delved into questions around running a **JVM within a Python standard app engine on GCP** and the possibilities of initiating a JRE or GCP JVM within a containerized langserve.
- **A New Challenger, Search RAG API**: `@emrgnt_cmplxty` highlighted the launch of the **Search RAG API**, potentially a game-changer for synthetic data and grounding responses. Give it a whirl [here](https://www.sciphi.ai/).
- **Tutorial Transmissions on Video Interactivity and Advanced AI Retrieval**: `@a404.eth` shared a how-to on building a tool to chat with videos for optimizing video metadata. Additionally, `@lhc1921` flagged this resource for mastering advanced retrieval in the realm of AI: [Advanced Retrieval for AI](https://learn.deeplearning.ai/advanced-retrieval-for-ai).

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (28 messages🔥): 
        
- **ConversationalRetrievalChain relevance issue**:
    - `@irfansyah5572` pointed out that their ConversationalRetrievalChain setup was returning **all found source documents**, not just the ones relevant to the query asked. `@seththunder` suggested using a line of code with a `similarity_score_threshold` to tackle this issue.
- **Conversational miscommunications? Ditch Langchain!**
    - `@atefyamin` and `@evolutionstepper` voiced strong concerns regarding the limitations of Langchain. Mentioning everything from parser issues to the trouble of synchronous tasks, they expressed a clear **preference for building their own utilities** instead.
- **Image Resizing**: 
    - User `@rajib2189` opened a discussion asking about **which packages are usually used for image resizing**.
- **RAG with tabular data**: 
    - `@michael_71751` asked for advice on using a **RAG with tabular data input and output** and also sought assistance on transforming tabular data using a linguistic context.
- **Loading markdown files with LCEL**: 
    - `@cryptossssun` queried about the potential way to **select and load markdown files** from a local directory using a custom function. They planned to use these files as the context for retrieval.
    - `@seththunder` suggested using `DirectoryLoader` and specifying **just the markdown files** in the directory glob.
- **LLMChain vs ConversationChain**: 
    - `@nav1106` queried about the **differences between LLMChain and ConversationChain** and when to use one over the other.
    - `@seththunder` responded saying both were very similar but **ConversationChain is preferred** when you want to have a simple conversation before asking your question.
- **MutliChains Connection with Pipeline Operator**: 
    -  `@seefrog` asked if it was possible to **connect multi-chains using a pipeline operator**
    - `@seth_thunder` confirmed that this can be done using a SequentialChain.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (3 messages): 
        
- **Running JVM within a Python Standard App Engine on GCP**: User `@johnda98` asked if anyone has experience running a JVM within a Python standard app engine on GCP for a layer2 project that involves billing AItoken counts in crypto tokens/in-protocol currency using a crypto SDK in Java via py4j.
- **Deploying Langserve on GCP**: `@johnda98` has successfully deployed langserve on GCP and is looking to integrate it with a JVM.
- **Running JRE/GCP JVM within containerized Langserve**: `@johnda98` brought up a query whether it's possible to initiate a JRE or GCP JVM within a containerized langserve deployed on GCP via cloud run.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Search RAG API now available for trials**: `@emrgnt_cmplxty` announced that the Search RAG API is now up and working. They highlighted its potential usefulness for **synthetic data and grounding responses**. They provided a link for anyone interested to give it a try: [Sciphi.ai](https://www.sciphi.ai/).


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (2 messages): 
        
- **Build a 'Chat with Your Videos' Tool**: User `@a404.eth` posted a [YouTube video tutorial](https://youtu.be/DTjj50mfEP8) illustrating how to create a simple LCEL chat that can transcribe and interact with video content. This tool aims to assist content creators by generating improved titles, descriptions, and keywords for their videos.
- **Advanced Retrieval for AI**: User `@lhc1921` shared a [link](https://learn.deeplearning.ai/advanced-retrieval-for-ai) for learning advanced retrieval methods for artificial intelligence.

**Links mentioned**:

- [DLAI - Learning Platform Beta](https://learn.deeplearning.ai/advanced-retrieval-for-ai)
- [Building an OpenAI Custom RAG with LangChain: The Ultimate Tutorial to Chat with your Videos!](https://youtu.be/DTjj50mfEP8): I hate writing video descriptions and titles so I ...


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **GPT Store Enters the Market**: `@swyxio` shared a tweet from `@eerac` about the upcoming launch of GPT store, a hub for AI-related projects in 2024, advising to port existing applications like weather, podcast, mindfulness, to-do list, word games, etc., to CustomGPT. [Tweet Link](https://fxtwitter.com/eerac/status/1742969953924346272?s=46&t=90xQ8sGy63D2OtiaoGJuww).
- **ChatGPT Introduces Reply-to-Part Feature?**: In a discussion triggered by a screenshot, `@coffeebean6887`, `@fanahova`, and `@dimfeld` debated the novelty and the possible ChatGPT exclusivity of a feature to reply to parts of messages.
- **The Challenge of Related-Article Features**: `@swizec` requested strategies to evaluate features offering related articles, encouraging a collective brainstorming.
- **DevOps GPT Deemed A Flop**: `@btdubbins` expressed dissatisfaction with "DevOps" GPT, highlighting an array of errors and labeling it one of the worst implementations seen.
- **Akuity Tops Kubernetes LLm Projects**: In response to `@austintackaberry's` inquiry about significant Kubernetes LLm products/projects, `@fanahova` recommended Akuity, a product revaled to be the driving force behind Argo. [Akuity Link](https://akuity.io).
- **Anticipating TLDRAW Episode**: In the ai-event-announcements channel, `@swyxio` teased the upcoming episode with **TLDRAW** and invited feedback. The [preview link](https://www.latent.space/p/3a8b36dc-3c36-434b-ae12-9ee5659a5997) was provided.
- **Chat Transcription vs User Comfort**: A discussion initiated by `@picocreator` in the llm-paper-club channel raised concerns around the potential infringement on user comfort by AI-based chat transcriptions, emphasizing the importance of user trust.
- **Reproducing Papers as a Group Activity**: `@ivanleomk` pitched the idea of reproducing and training models based on discussed papers, suggesting a 1-1.5 months timeline for interested contributors.
- **Acknowledgment of Successful Claim**: `@swyxio` thanked a user (`<@206404469263433728>`) for a successful, albeit unspecified, claim in the llm-paper-club channel.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (23 messages🔥): 
        
- **GPT Store Announcement**: `@swyxio` shared a tweet from `@eerac` announcing the impending launch of GPT store, an ideal starting place for AI-related ambitions in 2024. The suggestion was to port existing favourite applications like weather, podcast, mindfulness, to-do list, word games, etc., to CustomGPT. [[Tweet Link]](https://fxtwitter.com/eerac/status/1742969953924346272?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- **New ChatGPT Feature?**: A screenshot shared prompted a conversation about the ability to reply to parts of messages on ChatGPT. Both `@coffeebean6887` and `@fanahova` had not seen this feature previously, whereas `@dimfeld` thought it was new and exclusive to ChatGPT.
- **Talking Embeddings and Related Articles**: `@swizec` asked the community for effective ways to evaluate related-articles features.
- **Distress Over "DevOps" GPT**: `@btdubbins` observed that their "DevOps" GPT generated an array of errors, making it one of the worst implementations they've seen.
- **Request for Kubernetes LLm Product/Project**: `@austintackaberry` inquired about notable kubernetes llm products/projects, to which `@fanahova` recommended Akuity, a product reported to be behind Argo. [[Akuity Link]](https://akuity.io)

**Links mentioned**:

- [draw fast • tldraw](https://drawfast.tldraw.com/): Draw a picture (fast) with tldraw
- [lens by tldraw](https://lens.tldraw.com/): An infinitely scrolling drawing and hallucinating ...
- [Tweet from Eric Rachlin (@eerac)](https://fxtwitter.com/eerac/status/1742969953924346272?s=46&t=90xQ8sGy63D2OtiaoGJuww): If your goals for 2024 involve building something ...
- [Deploy to Kubernetes with Argo CD as a managed service](https://akuity.io): Akuity SaaS platform for Argo CD. Argo enterprise ...
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/18yf5dk/first_verification_post_got_taken_down_here_is/)


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **Upcoming TLDRAW Episode Preview**: `@swyxio` shared a [preview link](https://www.latent.space/p/3a8b36dc-3c36-434b-ae12-9ee5659a5997) to the upcoming episode with **TLDRAW** and welcomed comments.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (6 messages): 
        
- **Recording Chats: An Invasive Affair?**: `@picocreator` posed a query on whether chat transcriptions via AI could infringe upon user's comfort. A concern was raised that this could potentially deter members from asking questions in a session, emphasizing the need for trust and user comfort.
- **An Invitation to Reproduce Papers**: `@ivanleomk` proposed an idea of reproducing and training models based on discussed papers. Suggestion was for interested parties to take on this project over the course of 1-1.5 months.
- **Successful Claim Acknowledged**: `@swyxio` thanks a user (`<@206404469263433728>`) who claimed an unknown item or task, though the exact details are left unsaid.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Performance Rivals between Stage and Turbo**: `@maxidl` noted that both **StageTurbo and Turbo** had comparable performance in ranking a document set, with only a 1% deviation in the nDCG score favoring StageTurbo. 
- **Plug for German Embedding Models**: Highlighting a need for localized text embeddings, `@sebastian.bodza` shared a [research article](https://arxiv.org/abs/2401.00368) and expressed the desire to develop similar models for the German language.
- **Rise of Retrieval Tasks for AI**: `@sebastian.bodza` shared the first part of a dataset on Huggingface, highlighting a move towards training AI models to complete retrieval tasks with specific context-driven guidelines.
- **Underscoring Need for More Real German Questions**: Amidst his work on German wikipedia data, `@philipmay` emphasized the need for more real questions in training data. He also shared a positive experience with training a German DPR model on a translated and curated SQUAD dataset, noting the seamless blend of English instructions with German text through **GPT-3.5-turbo**.
- **BERT Models and Token Lengths Debated**: `@philipmay` and `@_jp1_` conversed about the optimal token length for BERT models, hovering around a range of 252-2048 tokens depending on general or specific use cases.
- **Classic Info Retrieval vs. Embedding Models Discourse Sparked**: `@thewindmom` questioned the use of classic information retrieval systems over embedding models, pointing the conversation towards CoBERT and the challenges of dense embeddings as per a [Hacker News post](https://news.ycombinator.com/item?id=38869223) and a [Twitter thread](https://twitter.com/bclavie/status/1742963012619608368).
- **Introduction of German MTEB Benchmark**: `@rasdani` announced a German-oriented contribution to the MTEB benchmark with a retrieval benchmark based on the GermanQuAD, now hosted on [DiscoResearch's GitHub](https://github.com/DiscoResearch/mteb).
- **Call for Collaboration on Open GitHub Issue**: `@rasdani` shared an open [GitHub issue](https://github.com/DiscoResearch/mteb/issues/1) on the DiscoResearch mteb, inviting ideas and contributions.
- **Video Meetings Proposed for Collaboration**: Espousing a collaborative spirit, `@philipmay` suggested video meetings to discuss different approaches and potential synergies.

**DiscoResearch Channel Summaries**

### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (17 messages🔥): 
        
- **Performance Comparison of StageTurbo and Turbo**: `@maxidl` noted that in testing both to rank a set of documents, **StageTurbo and Turbo** performed very close, with Turbo only 1% less in nDCG. 
- **Discussion on German Embedding Models**: `@sebastian.bodza` shared a [research article](https://arxiv.org/abs/2401.00368) on high-quality text embeddings and expressed the need to build similar models for german language and experiments with Textexercises.
- **Exploring Retrieval Tasks for AI**: `@sebastian.bodza` shared the first part of a dataset on Huggingface, and the Proof of Context implementation to train a model to complete retrieval tasks following specific guidelines.
- **Interest in German Dataset Development**: `@philipmay` affirmed working on the generation of questions from German wikipedia data, and highlighted the need for more real questions in the training data. He also shared his recent positive experience with **GPT-3.5-turbo** in mixing English instructions with German text, and a successful training of a German DPR model on a translated and curated SQUAD dataset.
- **Creation of Dedicated Channel for AI Models Discussion**: `@philipmay` and `@bjoernp` agreed on the creation of a separate channel for discussions related to embeddings and DPR models, leading `@bjoernp` to officially create a [new channel](https://discord.com/channels/1178995845727785010/1192471915504341062) for the said purpose.

**Links mentioned**:

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368): In this paper, we introduce a novel and simple met...
- [SebastianBodza/RAG_Aufgaben · Datasets at Hugging Face](https://huggingface.co/datasets/SebastianBodza/RAG_Aufgaben)


### ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/) (12 messages🔥): 
        
- **Discussing Maximum Context Length for BERT Models**: `@philipmay` asked for opinions on the maximum context length (in tokens) for BERT models used in DPR training, referencing their previously trained model with a max token length of 252. `@_jp1_` suggested a minimum of 512 for general purpose embedding models, stating that 1 or 2k might be better. For specific RAG use cases, they suggested 2xx might be sufficient. [Reference](https://cdn.discordapp.com/attachments/897965898816577616/898337004896350258/dpr.png)
- **Question about CoBERT and Classic IR**: `@thewindmom` inquired if anyone had looked at CoBERT or worked with classic IR instead of embedding models. They also shared a link to a [Hacker News post](https://news.ycombinator.com/item?id=38869223) discussing the challenges of using dense embeddings and a [Twitter thread](https://twitter.com/bclavie/status/1742963012619608368) that they intended to explore further.
- **Introduction of First German MTEB Benchmark Contribution**: `@rasdani` announced the hosting of a [fork of MTEB](https://github.com/DiscoResearch/mteb) for German benchmarks on the DiscoResearch Github organisation, including a retrieval benchmark based on the GermanQuAD in the `germanquad-retrieval` branch. They also shared the testset results on MRR@10.
- **Open GitHub Issue on DiscoResearch mteb**: `@rasdani` shared an open [GitHub issue](https://github.com/DiscoResearch/mteb/issues/1) on the DiscoResearch mteb, inviting interested individuals to take it up.
- **Proposing Video Meeting for Discussion**: `@philipmay` suggested a video meeting to clarify the different approaches being followed and discuss potential collaboration points. The invitation was extended to several channel members.

**Links mentioned**:

- [Show HN: RAGatouille, a simple lib to use&amp;train top retrieval models in RAG apps | Hacker News](https://news.ycombinator.com/item?id=38869223)
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368): In this paper, we introduce a novel and simple met...
- [Adding German to MTEB · Issue #183 · embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb/issues/183): Hi everybody, I think it would be great to add Ger...
- [Custom class for e5 embedding model family · Issue #1 · DiscoResearch/mteb](https://github.com/DiscoResearch/mteb/issues/1): The e5 embedding family requires &quot;query: &quo...
- [unilm/e5/utils.py at master · microsoft/unilm](https://github.com/microsoft/unilm/blob/master/e5/utils.py#L98C1-L99C1): Large-scale Self-supervised Pre-training Across Ta...
- [GitHub - DiscoResearch/mteb: A fork containing German benchmark contributions for MTEB, the Massive Text Embedding Benchmark. All contributions will be upstreamed to MTEB.](https://github.com/DiscoResearch/mteb): A fork containing German benchmark contributions f...


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **SciPhi.AI Reveals Knowledge Engine for LLM Agents**: `@emrgnt_cmplxty` shares a link to the [Sciphi.AI](https://www.sciphi.ai) website, a project that proposes to enable LLM agents with seamless access to humanity's key knowledge through web-scale search and data synthesis.
- **You.com API Discussion**: `@robotums` bumped [You.com's API](https://api.you.com/) in the chat, wondering how it compares to the Metaphor API. In response, `@jeffreyw128` explained that while You.com largely wraps the Brave API, Metaphor uses an *embedding-first search* approach, with an emphasis on search customization and computational resources.
- **Upcoming Powerful Snippets from Metaphor**: `@jeffreyw128` teased an upcoming feature from Metaphor, promising impressive snippets that offer high customizability, including control of *sentence count*, *snippets per page*, and *specific queries pertaining to the snippet*.
- **Improved Response noted on Lebesgue Integration Query**: `@nosa_.` stated a significantly improved response on a Lebesgue integration question they queried previously, to which `@emrgnt_cmplxty` cheerfully acknowledged.

**Links mentioned**:

- [YOU API | Innovative AI API | API Calls for All Companies](https://api.you.com/)
- [Home - SciPhi](https://www.sciphi.ai)

        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **A Curiosity Check**: `@dook4` inquired about the group's activity, to which `@far_el` responded encouragingly, hinting at upcoming activities in the next week.
- **A Show of Excitement**: `@s3nh1123` expressed excitement with an emoji.
- **Question Regarding Data Release**: `@maxmatical` asked about `wizardlm/wizardcoder` and their data release, posing the concern of irreproducible research.
        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **New Kid on the Block, Search RAG API**:@emrgnt_cmplxty announced that the new **Search RAG API** is live. It holds compelling implications for synthetic data and grounding responses. Get more information [here](https://www.sciphi.ai/).
- **Benchmarks Beating Models**: @maxmatical raised an important point about AI model training: models may exhibit poor performance as base models because **they're tuned to outdo benchmarks**. Whether deliberate or accidental, this training approach can affect a model's robustness across applications.

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (1 messages): 
        
- **New Search RAG API Now Available**: User `@emrgnt_cmplxty` announces that the **Search RAG API** is now available for use, highlighting its benefits for synthetic data and grounding responses. Details can be found at this [link](https://www.sciphi.ai/).


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (1 messages): 
        
- **Models Trained to Beat Benchmarks**: User `@maxmatical` suggested that models may not perform effectively as base models because **they are primarily trained to beat benchmarks**, either intentionally or unintentionally.


        

---
The Datasette/LLM (@SimonW) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
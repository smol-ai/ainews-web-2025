---
id: 2bb42a5c-ee60-4fe5-8167-fb6b53efaad4
title: 12/8/2023 - Mamba v Mistral v Hyena
date: '2023-12-08T22:40:04.800968Z'
original_slug: ainews-1282023-mamba-v-mistral-v-hyena
description: >-
  Recent AI news highlights new models from Mistral, Mamba, and StripedHyena,
  discussions on Anthropic's Claude 2.1, Google Gemini, and various AI model
  deployment and evaluation topics.
companies:
  - mistral
  - mamba
  - stripedhyena
  - anthropic
  - google
models:
  - mistral-8x7b-moE
  - mamba-3b
  - stripedhyena-7b
  - claude-2.1
  - gemini
  - dialogrpt-human-vs-machine
  - cybertron-7b-v2-gguf
  - stripedhyena-nous-7b
  - falcon-180b
topics:
  - large-language-models
  - model-evaluation
  - model-deployment
  - prompt-engineering
  - alignment
  - open-source
  - cpu-performance
  - transformer-batching
  - image-training
---


<!-- buttondown-editor-mode: plaintext -->Happy Friday. 3 new models are the talk of the town today:

- Mistral's new [8x7B MoE model](https://news.ycombinator.com/item?id=38570537) (aka "Mixtral") - a classical attention model, done well. Andrej's recap [here](https://twitter.com/karpathy/status/1733181701361451130).
- [Mamba models](https://twitter.com/tri_dao/status/1731728602230890895), a range  of models up to 3B by (former guest) [Tri Dao of Together](https://www.latent.space/p/flashattention)
- [StripedHyena 7B](https://fxtwitter.com/omarsar0/status/1733223272412594501) - a descendant of the subquadratic attention replacement [Hyena](https://arxiv.org/abs/2302.10866) out of Stanford's Hazy Research lab released earlier this year, that is finally competitive with Llama-2, Yi, and Mistral 7B.

This is all very substantial and shows what happens when you ship model weights instead of heavily edited marketing videos.

[TOC] 

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- In-depth discussion around AI models such as **Anthropic's Claude 2.1**, including the exploration of its prompting technique, evaluation of AI models as a "needle in haystack" scenario and the workaround for alignment approach needed for Claude 2.1. Plans for a meeting between users `@maxwellandrews` and `@raddka` were also discussed. 

- Dialogue about the applications and hacking of **Dreambooth** for image training, along with a request for assistance in this process. Announcement of **Gemini**, a new AI from Google believed to be potentially superior to **GPT-4** was shared.

- Sharing of AI-related tweets, resources, and models such as the `DialogRPT-human-vs-machine` model on Hugging Face, which predicts if a response seems more likely to be from a human or a machine. A [Colab Notebook Demo](https://colab.research.google.com/drive/1cAtfkbhqsRsT59y3imjR1APw3MHDMkuV?usp=sharing) was provided for practical engagement with the model.

- Numerous factors raised during conversation about AI model building and maintenance, including data extraction, model relases especially the **"cybertron 7b v2 GGUF"** and **Mistral 8x7B MoE** models, chatbot model **StripedHyena-Nous-7B**, open-source plan for the **StripedHyena's training code** and other topics such as GPU requirements for running the **Mistral 8x7B MoE** model, evaluation of memory requirements and debugging of model inference. 

- Various issues and possibilities regarding the deployment of **Large Language Models (LLMs)**. Points included deployment of LLM models on CPUs, setup of such projects, potential of LLM for exam marking and grading, and also hosting of embedding models. Reference to resources like [Falcon 180B initial CPU performance numbers](https://www.reddit.com/r/LocalLLaMA/comments/16bynin/falcon_180b_initial_cpu_performance_numbers/) and [Transformer Batching](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) were provided.

- User engagement in the memes channel, with `@nonameusr` sharing a [Twitter link](https://vxtwitter.com/vega_holdings/status/1727368097869488292) and expressing a humorous confusion towards the content of the linked post.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (8 messages🔥): 
        
- Discussion about **Anthropic's Claude 2.1**: `@if_a` shares a link to the [Anthropic Claude 2.1's prompting technique](https://www.anthropic.com/index/claude-2-1-prompting) study, highlighting how Claude 2.1 well it recalls information across its 200,000 token context window. However, `@if_a` also considers that testing the long context capability will require more work on **prompt engineering**.
- Evaluating AI Models: `@gabriel_syme` comments on the evaluation of AI models, specifically criticizing such evaluations as a "needle in haystack" scenario which may not provide accurate performance measures. 
- Alignment Methods: `@intervitens` finds it interesting that the team at Anthropic had to use a workaround to bypass their own alignment approach for Claude 2.1. `@raddka` sees it as an inevitable development given the stringent compliance restrictions that may have hampered the development of the model.
- Upcoming Meeting: `@maxwellandrews` and `@raddka` plan a meeting for the following week, with the tentative meeting time proposed to be between 12-5 PM. Further details are to be finalized later.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (8 messages🔥): 
        
- **Dreambooth Discussion**: `@gezegen` and `@wlrd` discussed about the usage of **Dreambooth** in image training and the possible reason for the time it takes to get back results.
- **Hacking into Dreambooth**: `@yorth_night` shared their experience of hacking into **Dreambooth** for the past week to conduct long prompt training, noting some complex aspects of the code.
- **Request for Assistance**: `@gezegen` suggested `@yorth_night` to share their process of hacking into **Dreambooth**, offering potential help.
- **Gemini Announcement**: `@pradeep1148` shared a [YouTube video](https://youtu.be/mAGLD5598cs) announcing **Gemini**, a new AI from Google that's potentially better than **GPT-4**, with features for multimodal reasoning across text, images, video, audio, and code.
- **General Interaction**: `@gabriel_syme` greeted the channel with a good morning message.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (1 messages): 
        
nonameusr: https://huggingface.co/allenai/tulu-2-dpo-70b


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (5 messages): 
        
- **Sharing AI-related tweets**: `@yorth_night` and `@nonameusr` shared [tweets](https://fxtwitter.com/apples_jimmy/status/1732553640215495109) and [tweets](https://vxtwitter.com/ChatGPTapp/status/1732979491071549792) on AI responsible tech and CAIS theory.
  
- **DialogRPT-human-vs-machine**: `@euclaise` posted a link to the `DialogRPT-human-vs-machine` model on Hugging Face, which aims to predict if a response is more likely to come from a human or a machine. They included a [Colab Notebook Demo](https://colab.research.google.com/drive/1cAtfkbhqsRsT59y3imjR1APw3MHDMkuV?usp=sharing) for users to interact with the model.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (375 messages🔥🔥): 
        
- Discussion amongst developers on developing a similar **pipeline for extracting data** from **GitHub repositories** and **.tex files**. User `@zakkor` mentioned improving the output with **additional LLM (Language Learning Model) filtering pass**. User `@wlrd` showed interest in collaborating on this effort and brought up a related ongoing discussion thread. [Link to discussion thread](https://discord.com/channels/1053877538025386074/1178740658136162395)

- User `@nonameusr` announced the release of the **"cybertron 7b v2 GGUF"** model and that it performs close to the **"yi-34b"** model. Testing of the model was also discussed.

- Notable discussion on the new model **Mistral 8x7B MoE**, with users having queries about its memory requirement, how to run inference, and discussion on pc hardware required for running this model.

- Chatbot model **StripedHyena-Nous-7B** was announced by the user `@weyaxi` and caused a discussion among users, leading to `@theemozilla` explaining the model's architecture, development, and shared plans for the model's future iterations.

- Users discussed posting **Models from Nous Research** on **Huggingface**, with `@nonameusr` discussing **"una-xaberius-34b-v1beta"** as the potential best model and referring to **"xaberius"** explicitly.

- User `@theemozilla` mentioned plans to **open-source their custom training code** upon approval from Together Research. The training code is related to the **StripedHyena-Nous-7B (SH-N 7B)** model.

- Further conversation about the **MoE (Mixture-of-Experts) distribution** and memory requirements of running the **Mistral 8x7B MoE** model, concluding with `@bjoernp` suggesting that 2 A100s in 80GB seem to be enough. 

- Users talked about the **decoding methods** with `@gabriel_syme` referring to a Twitter link throwing light on recent advances and new decoding methods. A discussion ensued about the compatibility and the need for these methods. 

- Gradio image inference **errors with OpenHermes were discussed**. After running into server errors, `@qnguyen3` suggested using the code in Github and editing the `llava/model/multimodal_encoder/builder.py` file to include 'ikala'.

- User `@euclaise` shared his experience with **Echo-3B**, a model he created for RP/creative tasks. Some technical issues causing the model to output nonsense were mentioned.

- **Gemini API** for generating datasets was brought up by `@jaredquek`, followed by a discussion about the potential repercussions of using it due to Google and OpenAI account termination policies.

- User `@raddka` commented on the newly released Mistral 8x7B MoE model, speculating it as a candidate for developing lightweight models to improve Language Learning Model (LLM) shortcoming and suggesting the creation of one specific for coding to evaluate/improve its answers.

- Issues with `openhermes vision` server startup were discussed and resolved with the help of `@qnguyen3`.

- Members briefly mentioned the plan for **training Mixtral MoE on dialogue datasets** and compared the model's benchmark performance. They also speculated on how the benchmarking could potentially be biased since the posted values were too extreme to be deemed legitimate.

- Users discussed possible_gpu setups for the **Mistral 8x7B MoE model**, including debating over collecting multiple Nvidia RTX 3090 cards, the necessity of PCIe lanes, and how to optimally cool the GPUs.

- **StripedHyena-Nous-7B's evaluation** utility was highly praised by users, citing its reduced memory footprint and better performance. Canvassing for its integration with various APIs and deployment platforms were also discussed.

- **Continuous Debugging and Discussions** around running `Mixtral model's` inference and the issues surrounding it. The need for its codebase and users' efforts to run the inference in various ways were also highlighted.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (80 messages🔥🔥): 
        
- **LLMs (large language models) and CPUs**: `@.beowulfbr` initiated a conversation about deploying a 7B or 70B LLM model purely on CPUs. `@raddka` mentioned it would be incredibly slow. `@coffeebean6887` and `@decruz` reiterated that commercially viable LLMs would require a GPU and also discussed the advantages of batched infra on GPUs. They referred to [Falcon 180B initial CPU performance numbers](https://www.reddit.com/r/LocalLLaMA/comments/16bynin/falcon_180b_initial_cpu_performance_numbers/) and [Transformer Batching](https://le.qun.ch/en/blog/2023/05/13/transformer-batching/) for relevant discussions.
- **Project Setup for LLMs**: `@akhxl` discussed troubleshooting issues with `Litellm` and `Ollama` getting a 404 error. This was resolved by ignoring the `api_base` line. 
- **LLMs Learning Capability**: A substantial debate was initiated by `@akhxl` around if and how an LLM could understand new information and topics it had not previously encountered in its training data. `@adamsvoboda` explained this could be achieved through RAG by incorporating the external information into the prompt as context. However, the user still had further queries about the LLM's interpretation and contextual understanding, which were not fully addressed in the given conversation. 
- **LLMs in Exam Marking**: User `@.___init___` suggested the potential use of LLMs in exam marking and grading, while `@coffeebean6887` indicated that the math part might be complicated.
- **Hosting Embedding Models**: `@coco.py` asked about the existence of tools to host embedding models similarly to how vLLM hosts language models. `@decruz` recommended open source solutions like `gte-small`, which could be hosted on an edge function. However, no specific information about hosting embedding models in an OpenAI API-compatible manner was provided.


### ▷ #[memes](https://discord.com/channels/1053877538025386074/1166105758635655270/) (2 messages): 
        
- A user `@nonameusr` shared a [Twitter link](https://vxtwitter.com/vega_holdings/status/1727368097869488292) from Vega Holdings on the channel. Further, the user expressed confusion or surprise towards the content of the linked post with the comment: "*wtf is this 😭*".


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Cross-channel discussion highlighted the anticipation, skepticism, and discussions surrounding Google's **Gemini AI model** and future versions of **OpenAI's GPT model**. Users voiced their opinion about Google's launch video for Gemini, and speculation is made about the groundwork OpenAI has already laid for versions up to GPT-7 (quotes by `@DawidM` and `@aiden4729`).
- Users in various channels reported **server issues**, **performance problems** and **communication challenges** with OpenAI products like **ChatGPT** and **GPT-4**. This included messages disappearing, decreased performance, especially with strange conversation names, long wait times in getting support replies, payment system issues, and non-responsiveness of chatbots (quotes by `@_steeven_`, `@maybedara`, `@dabuscusman`, `@mrcrack_`, `@seliathas`, `@bubbarob19`, `@ryanxcharles`, `@spider7124`, `@exobyt`, and `@solbus`).
- The community also discussed **AI's potential impact on scientific research** and debated its capability to outcompete human researchers, especially in terms of mathematical discovery (conversation between `@axiomaticself` and `@feltsteam0`).
- Discussion of **VPN usage** with OpenAI technologies emerged across the channels, with users stating their VPNs weren't being detected or blocked and further clarifying that multiple users from the same IP could be filtered out (discussion by `@youraveragedev`, `@satanhashtag`, `@lugui`, and `@olympusdev`).
- Numerous **technical questions, suggestions, and clarifications** were made across the channels, spanning topics such as GPT token limit, interactions among custom GPTs, the use of a VPN to access certain AI models, prompting ChatGPT for specific response formats, API usage for web search capabilities, the performance of GPT4 with extended token length, and extracting a DALLE prompt from an image among others.
- Users shared and critiqued resources, notably `@zelia13` prompted for feedback on an AI Knowledge Management [Video](https://www.youtube.com/watch?v=au1Yznvx2Io) and `@iyioio` shared a link to [OpenAI's function-calling document](https://platform.openai.com/docs/assistants/tools/function-calling). Explorative prompts and their outputs were also shared through links.
- Lastly, reminders about the OpenAI general conduct emerged, detailing incidences such as violation of posting rules and potential consequences.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (92 messages🔥🔥): 
        
- **Gemini vs GPT-4**: Users expressed mixed feelings about the anticipated launch of Google's Gemini AI model, with some excited about its potential while others expressed skepticism due to Google's [alleged misrepresentation in the demo video](https://www.techcrunch.com/the-gemini-video). `@DawidM` claims that the video was post edited to present the product in an overly positive manner, calling it a "shady marketing tactic".

- **OpenAI Model Updates**: There's speculation about the release of future versions of OpenAI's GPT model, with user `@aiden4729` pointing out that OpenAI has already laid the groundwork for versions up to GPT-7 according to trademark filings. `@thepitviper` predicts the announcement of GPT-4.5 in competition with Google's Gemini.

- **Concerns over AI in Scientific Research**: User `@axiomaticself` expressed concerns about the potential for AI to outclass human researchers in the field of mathematics. `@feltsteam0` provided reassurance, stating that we are still years away from AI fully automating scientific discovery and that likely the future lies in augmented human-AI research teams.

- **Feedback Request on AI Knowledge Management Video**: User `@zelia13` shared a [video](https://www.youtube.com/watch?v=au1Yznvx2Io) on knowledge management using AI for project managers and requested feedback on it. 

- **Challenges with AI/Chatbot Access**: Some users discussed difficulties in accessing certain AI and chatbot models due to regional restrictions. User `@offline` suggested using a VPN as a workaround.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (246 messages🔥🔥): 
        
- **Server Issues with ChatGPT**: Multiple users including `@_steeven_`, `@maybedara`, and `@dabuscusman` reported experiencing server issues with ChatGPT. User `@_steeven_` mentioned messages disappearing upon hitting enter, and several users assumed server overload might have caused it. However, some users like `@pruo` and `@dabuscusman` later confirmed that they had successful access to ChatGPT.
- **ChatGPT Performance**: `@mrcrack_` expressed dissatisfaction with the current state of ChatGPT, mentioning a decrease in performance over the previous few months, specifically citing strange conversation names such as "Ebola virus and flu".
- **VPN Usage**: `@youraveragedev` asked whether OpenAI now permits VPN usage, as their VPNs weren't being detected or blocked. This was confirmed by `@satanhashtag` and clarified by `@lugui` and `@olympusdev` who suggested that VPNs have always been allowed, but multiple users from the same IP could be filtered out.
- **OpenAI Support**: User `@seliathas` expressed frustration with OpenAI's support system, stating their communication wasn't answered for a prolonged period. `@elektronisade` explained that depending on the topic, replies may take a substantial amount of time.
- **Product Availability and Usage**: `@Foufou` flagged a translation error on the website, citing a miscommunication of chat restrictions in French. User `@theyonatan` asked for suggestions on making ChatGPT repaste complete code. Meanwhile, `@goni0755` inquired about the reliability of the DALLE image generation API compared to others. Also, `@zyrqlo` and `@thepitviper` discussed about using "Bard" with "Gemini Pro".


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (69 messages🔥🔥): 
        
- **API usage and Web Search Capabilities**: User `@shikhar1209` inquired about the possibility of using web search with GPT-4 through the API, though no clear answer was provided.
- **Issues with GPT-4 Performance and Functionality**: Several users including `@spider7124`, `@exobyt`, and `@solbus` reported experiencing errors and non-responsiveness with chatgpt. Additionally, user `@yoruiopz` expressed dissatisfaction with GPT-4's declining performance and recurring browsing time-outs.
- **Fine-Tuning GPT Token Limit**: `@pearlyeti` raised a query regarding the token limit per example for fine-tuning, specifically if it remained at 4096 for GPT-4, but no clear response was provided. 
- **ChatGPT Misunderstandings and Miscommunications**: Several users including `@treytay` and `@maguiresfuture` expressed concerns about misleading information or promises being given by ChatGPT during certain interactions.
- **Billing Issues and Subscription Problems**: Users `@bubbarob19` and `@ryanxcharles` reported experiencing continuous issues with the payment system on the platform, hindering access to services.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (31 messages🔥): 
        
- **Issues with Custom GPT Disappearance**:
    - Users `@_interstitialism_`, `@a2jhagrhbm92awno`, and `@budadude` reported the disappearance of their custom GPTs. The issue seemed to resolve on its own as `@a2jhagrhbm92awno` later reported that their GPTs came back.
- **Utilizing Images with GPT**:
     -  `@thermaltf` asked if there exists a GPT that can generate a Dalle prompt from an uploaded image, but received no direct response. 
     - `@demo_02817` needed help with image editing using OpenAI and shared code snippets, but no solution was provided in this chat history.
- **Interaction Among Custom GPTs**:
     - `@moraj123` inquired about the possibility of creating interactions between custom GPTs, to which `@rjkmelb` responded it was not directly possible.
- **Clearing Conversation History**:
     - `@sjjj.` asked if there was a way to delete some chat conversations. `@pietman` revealed that there was no such feature at the moment and shared information about a helpful Chrome extension for this purpose.
- **Larger Token Length and Model Performance**:
     - `@the_unbeatable_gusty_chatterbox` raised a question about the performance of GPT4 with 128K token length, commenting on the lower quality outputs with longer inputs. No answer was given in the discussion history.
- **Setting Up Public GPTs**:
    - `@serenejay` asked if creating public GPTs is only possible for those who host them. `@solbus` clarified that it can be enabled by verifying a website or adding a billing name on the user's Builder profile. There is no official central repository for GPTs at the moment, but one is expected next year.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (11 messages🔥): 
        
- **Prompting Assistant's API for a Specific Response Format**: User `.jkyle` queries about prompting the assistant's api to provide a response formatted as a json for parsing and output. User `iyioio` suggests using functions and provides [OpenAI's function-calling document](https://platform.openai.com/docs/assistants/tools/function-calling) as a resource. `.jkyle` intends to use the model output as the actual desired output.

- **Discussion on Functions feature in the API**: Among `.jkyle` and `iyioio`, there's a discussion of understanding the functions feature. The dialogue involves how the assistant defines an input specification for an external function, ensuring output aligns with the format, and how the execution process occurs.

- **Interesting Prompts**: Users `mindfultatiana` and `eskcanta` share and explore interesting prompts, including explaining concepts 'as if to an AI', 'as if I were a fish' or 'as if I were a a library book of poetry'. They experiment with these prompts on the model, sharing the resulting output via links to OpenAI chat shares.

- **Directing AI's Behavior**: User `clumsylulz` experiments with directing the AI's behavior by asking it to respond as if it were a human with schizoaffective disorder.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (11 messages🔥): 
        
- **API Prompting for JSON Outputs**: User `@jkyle` asks for advice on prompting the assistant's API to always provide a JSON response. User `@iyioio` suggests using the function feature of the API and describes how it works. `@jkyle` then interprets this functionality to mean that it provides an output object that can be fed to an external function, then returned to close the loop. `@iyioio` confirms that this is almost right, highlighting that the function call received from the assistant is just another message in the thread with special formatting. 
- **Instructive Prompts**: User `@mindfultatiana` shares an effective prompt: `"Can you explain to me like I'm 5 years old..."`.
- **Interactive Prompts**: User `@eskcanta` demonstrates the use of prompts to instruct the model to provide explanations in various perspectives such as 'as if to an AI', 'as if I were a fish' and 'as if I were a a library book of poetry', providing corresponding chat links for each.
- **Conditioned Prompts**: User `@clumsylulz` proposes a conditional response scenario, instructing the model to act as if it were a human with schizoaffective disorder. 
- Links:
    - [Function Calling](https://platform.openai.com/docs/assistants/tools/function-calling)
    - [Explanation 'as if to an AI'](https://chat.openai.com/share/52861eab-31b9-4a27-81b9-ec22a9435cdf)
    - [Explanation 'as if I were a fish'](https://chat.openai.com/share/4b7b2d7f-5414-4490-bb27-7557edf486e6)
    - [Explanation 'as if I were a a library book of poetry'](https://chat.openai.com/share/5948aac4-bba7-4172-9fb6-5a1d6abadcea)


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **AI Models Training**: Various AI models were discussed through users `@noobmaster29`, `@le_mess`, `@nanobitz`, `@metaldragon01`, `@casper_ai`, and `@yamashi`. Mamba's sluggish performance compared to Mistral was highlighted and an evaluation problem with Mamba was identified. Interest was expressed in the announcement of a new model from Mistral. Users also compared the performance of **Llama models** and **GPT-3.5**.
- **AI in Different Languages**: The need for effective AI models for different languages, including French and Dutch, was pointed out.
- **Unified Memory and Mac's M Processor Advantage**: The advantages of using Mac's M processor and a unified memory system for AI inference and possibly training were discussed. A recent release by Apple demonstrating how to train models on their machines was mentioned.
- **Challenges in Model Deployment and Adjustment**: A dialogue took place around the issues faced in deploying and adjusting AI models, particularly those from Mistral. It was hinted that users wouldn't be able to finetune the new 70B model. 
- **Mistral's New Model Release**: The release of the new **Mistral model** was discussed extensively, with speculations that the model could feature a Mixture of Experts (MoE). Transition steps to use the model, including converting the model to PyTorch and adding megablocks as a requirement, were mentioned.
- **Tokens' Impact on Model Training**: A discussion unfolded about whether a response with 120 tokens would have more influence compared to a response with 60 tokens during model training. It was suggested that tokens trained is probably a better metric for measuring the influence on the model.
- **DPO Fine-tuning Support**: Questions were raised regarding the possibility of axolotl offering support for DPO fine-tuning. Evidence that DPO is possible was offered with a [link to DPOpenHermes-7B on Hugging Face](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B/blob/main/configs/dpo.yml), along with detailed YAML configuration code.
- **Segmenting in User Input**: Interest was shown in segmenting in the user input and the possibility of using the same process for **segmenting retrieved docs** was confirmed.
- **Multi-GPU Training Issue**: Problems were reported with **training on runpod** using 8 A40 GPUs. The issue seems to be with finetuning where it hangs after the first eval pass showing 100% GPU utilization on all 8 GPUs. A mismatch between collective operations in PyTorch distributed was identified as a possible reason.
- **Relevant URLs**: Various links were shared to aid discussions and provide resources:
   - [Pearl GitHub](https://github.com/facebookresearch/pearl)
   - [Huggingface Dataset](https://huggingface.co/Q-bert/Optimus-7B)
   - [YouTube Interview - Mistral AI](https://www.youtube.com/watch?v=auQBhg692Js)
   - [OpenAI backend UI Chat interface on GitHub](https://github.com/huggingface/chat-ui)
   - [Twitter post by Mistral AI](https://twitter.com/MistralAI/status/1733150512395038967)
   - [Mistral Model on GitHub](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen/tree/main) 
   - [Implementation of Mixtral in Llama](https://github.com/dzhulgakov/llama-mistral/tree/main)
   - [Content Detailing New Model](https://github.com/mistralai/megablocks-public/tree/main/megablocks/layers)
   - [Detailed Paper on New Model](https://arxiv.org/abs/2309.02411)
   - [DPOpenHermes-7B on Hugging Face](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B/blob/main/configs/dpo.yml)
- **Reminder on Guild Conduct**: Members were reminded about guild conduct, with a note on repetitive advertising.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (179 messages🔥🔥): 
        
- **Discussion on Training Various AI Models**: Over the course of this conversation, several users, including `@noobmaster29`, `@le_mess`, `@nanobitz`, `@metaldragon01`, `@casper_ai`, and `@yamashi`, discussed the training of various artificial intelligence models such as **Mamba** and **Mistral**. It was pointed out by `@caseus_` that Mamba seems to be slow compared to Mistral, and `@nanobitz` mentioned an evaluation issue with it. There was also a discussion on comparing the performance of **Llama models** and **GPT-3.5**. The participants expressed interest in the announcement of a new model from Mistral. 
- **Application of AI in Different Languages**: Various users discussed the application of AI models in different languages. `@yamashi` pointed out the need for an effective model to work in French and Dutch.
- **Advantages of Unified Memory and Mac's M Processor for Training**: Conversation between `@yamashi` and `@noobmaster29` highlighted the benefits of using Mac's M processor and unified memory system for AI inference and possibly even training. `@yamashi` mentioned a recent release by Apple demonstrating how to train models on their machines.
- **Challenges with Model Deployment and Adjustment**: Several users, including `@faldore`, `@yamashi` and `@noobmaster29`, discussed the challenges they face in deploying and adjusting AI models, particularly those from Mistral. `@faldore` mentioned a conversation with a Mistral developer who indicated that users wouldn't be able to finetune the new 70B model.
- **Relevant URLs**:
    - GitHub repository for a Production-ready Reinforcement Learning AI Agent Library by Meta [https://github.com/facebookresearch/pearl](https://github.com/facebookresearch/pearl)
    - Dataset on Huggingface that could potentially affect model rankings [https://huggingface.co/Q-bert/Optimus-7B](https://huggingface.co/Q-bert/Optimus-7B)
    - YouTube interview with Arthur Mensch about Mistral AI [https://www.youtube.com/watch?v=auQBhg692Js](https://www.youtube.com/watch?v=auQBhg692Js)
    - Open-source UI chat interface to use with OpenAI backend [https://github.com/huggingface/chat-ui](https://github.com/huggingface/chat-ui)
    - Twitter post by Mistral AI announcing the release of a new model [https://twitter.com/MistralAI/status/1733150512395038967](https://twitter.com/MistralAI/status/1733150512395038967)
    - GitHub repository to the Mistral Model [https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen/tree/main](https://huggingface.co/someone13574/mixtral-8x7b-32kseqlen/tree/main)


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (56 messages🔥🔥): 
        
- **New Mistral Model Release**: The team discussed the release of a new **Mistral model** which was shared by `@casper_ai` [^Twitter link^](https://twitter.com/MistralAI/status/1733150512395038967). Members speculated that the model, named **8x7b**, could feature Mixture of Experts (MoE).

- **Model Specifications and Download**: `@yamashi` shared the model's specifications denoting it had 32k sequence length and appeared to feature MoE. The download speed of the model varied with `@casper_ai` initially reporting a slow download speed of 239 kb/s while `@yamashi` managed to download at a speed of 11MB/s.

- **Conversion and Use of the Model**: Multiple members, including `@caseus_`, discussed the possible steps needed to use the new model, including the conversion of the model to PyTorch and adding megablocks as a requirement. `@bjoernp` was in the process of working on a conversion script and invited anyone interested to help out by joining a call via his [Discord](https://discord.gg/kgXkgcdy).

- **Implementing Mixtral in Llama**: `@casper_ai` shared a [GitHub link](https://github.com/dzhulgakov/llama-mistral/tree/main) to a forked Lama repo where someone had implemented Mixtral into it, potentially helpful for the conversion process.

- **Content Detailing the New Model**: Various links to content detailing the new model were shared including a [GitHub link](https://github.com/mistralai/megablocks-public/tree/main/megablocks/layers) by `@_jp1_` and an [arXiv link](https://arxiv.org/abs/2309.02411) shared by `@c.gato`.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (12 messages🔥): 
        
- **Clarification on train_loss**: `@nanobitz` explained that the single dot labeled as 'train/loss' represents the final train_loss, and it's different from 'train/train_loss'. The latter is presented as a graph during training.
- **Impact of tokens on model training**: `@c.gato` had a discussion about whether a response with 120 tokens would have more influence compared to a response with 60 tokens during model training. `@le_mess` advised that tokens trained is probably a better metric for measuring the influence on the model.
- **Constant improvement**: `@c.gato` also expressed an understanding of the need for continuous refinement to keep up with improvements in their model.


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (3 messages): 
        
- **Possibility of DPO Fine-tuning Support**: User `@josh.sematic.dev` asked about the potential for axolotl to provide support for DPO fine-tuning. 
- **Zhengler]: No Current Implementation**: User `@le_mess` stated that the topic has been discussed many times but indicated there hasn't been an implementation yet.
- **Evidence of DPO Capability**: `@noobmaster29` shared that it's understood DPO is already possible and supported this with a [link to DPOpenHermes-7B on Hugging Face](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B/blob/main/configs/dpo.yml), along with a detailed YAML configuration code.


### ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (2 messages): 
        
- **Discussion on Segmenting**: User `@gabriel_syme` showed interest in user input segmentation and speculated if the same process could be utilized for **segmenting retrieved docs**. In response, `@le_mess` affirmed this possibility.


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (10 messages🔥): 
        
- **Multi-GPU Training Issue**: User `@vastolorde95` reported issues with **training on runpod** using 8 A40 GPUs. The issue appears to be with finetuning where it hangs after the first eval pass showing 100% GPU utilization on all 8 GPUs.
- **Error Analysis**: A possible NCCL error was suspected by `@vastolorde95`. After activating detailed debug information with environment variables `TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P`, a mismatch between collective operations in PyTorch distributed was identified: `RuntimeError: Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(OpType=ALLGATHER, TensorShape=[4], TensorDtypes=Float, TensorDeviceTypes=TensorOptions(dtype=float (default), device=cuda, layout=Strided (default), requires_gr ad=false (default), pinned_memory=false (default), memory_format=(nullopt))), but Rank 2 is running collective: CollectiveFingerPrint(OpType=ALLGATHER)`.
- **Successful Single GPU Training**: `@vastolorde95` noted that the same setup works with a single H100 GPU, though it was too slow, indicating that the issue was mainly with the multi-GPU setup.
- **Checkpointing Lag**: `@casper_ai` suggested that it could also be related to checkpointing being slow on 8 GPUs, possibly due to the machine having a slow disk speed.


### ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/) (1 messages): 
        
le_mess: Bro stop advertising this. I've said it before 😅


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Jetbrains AI Assistant** appreciation by `@slono` highlighting its consistent docstring style saving significant time.
- Anticipation for **Duet AI** from Google gotten from `@swyxio`'s discussions about an upcoming technology release, referring to the release of Mistral 8x7B.
- Importance of **AI compression** and introduction to **Mixture of Experts (MoE)** paradigm, instigated by links and a YouTube video shared by `@coffeebean6887` on compressing the 1.6 trillion parameter SwitchTransformer-c2048 model.
- Mention of the promising **AI startup scene in France** by `@slono`.
- Debate comparing **Cursor vs VS Code** with varying opinions: tollbooth presenting an OpenAI API ([@btdubbins](https://discord.com/users/325599707743510528)), beneficial AI search functionality ([@guardiang](https://discord.com/users/765625752364941313)), VSCode made for enterprise users considering data security ([@mitch3x3](https://discord.com/users/172398079356026881)), and the effectiveness of such tools attributed to integration with codebases and the surrounding UI ([@slono](https://discord.com/users/765625752364941313)).
- Announcement of a **new podcast episode** shared by `@swyxio`, sharing a [Twitter link](https://fxtwitter.com/latentspacepod/status/1733160841997070683) on the `ai-event-announcement` channel.
- `@swyxio`'s share on **Latent Space University Day 4** content about *Image Generation* via the [link](https://buttondown.email/ai4eng/archive/ai-for-engineers-beta-day-4-image-generation/).
- Detailed discussion on the **impact of formatting on model training** with a specific mention of the space sign impacting model's output, brought up by `@eugeneyan`'s anecdote.
- `@slono`'s elaboration on the **role of whitespace in code and its influence on tokenization and learning**. It also noted the significant differences in token counts due to varying use of whitespace.
- `@slono`'s query and `@eugeneyan`'s clarification regarding the usage of `[INST]` context
- `@__chef__`'s humorous remark about the complexity involved in training large models, speculating "How many parameters does it take to learn a space? Over 7 billion".

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (45 messages🔥): 
        
- **Jetbrains AI Assistant**: User `@slono` mentioned they were appreciating the Jetbrains AI chat assistant and highlighted that it had been fine-tuned to a consistent docstring style thus saving a significant amount of time. 
- **AI Upcoming Technology Release and Current States**:  `@swyxio` discussed the release of Mistral 8x7B, giving an indirect hint about the release of **Duet AI** from Google, and referring to a tweet as an interesting artifact of modern data. 
- **AI Compression and New Research Papers**: User `@coffeebean6887` shared few links related to compressing the 1.6 trillion parameter SwitchTransformer-c2048 model, along with a YouTube video about the **Mixture of Experts (MoE)** paradigm.
- **AI Startup Scene in France**: User `@slono` mentioned considering moving back to France due to potentially interesting opportunities in the AI startup scene there.
- **Cursor vs VS Code Discussion**: There was an ongoing debate on the benefits and drawbacks of Cursor versus VS Code. `@btdubbins` showed some skepticism towards cursor, feeling it was simply a tollbooth presenting an OpenAI API. Still, `@guardiang` found advantage in its AI search functionality and `@mitch3x3` reinforced that MS makes VSCode for enterprise users, taking into account data security. `@slono` believed that the usefulness of such tools was more related to integration with codebases and the UI surrounding that.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (2 messages): 
        
- **New Podcast Episode**: `@swyxio` announced the release of a new episode by sharing a [Twitter link](https://fxtwitter.com/latentspacepod/status/1733160841997070683).
- **Latent Space University Day 4**: `@swyxio` shared the fourth day's content about **Image Generation** from Latent Space University. The provided [link](https://buttondown.email/ai4eng/archive/ai-for-engineers-beta-day-4-image-generation/) covers how to leverage DALL-E API for image generation.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (8 messages🔥): 
        
- **Impact of Formatting on Model Training**: `@eugeneyan` described a scenario where fine-tuning a 7B model to generate responses in a specific JSON format yielded unexpected results, with model outputs varying significantly depending on whether the starting training data had a space following the starting syntax (`<s> [INST]` versus `<s>[INST]`).
- **Discussion on the Role of Whitespace in Code**: `@slono` indicated that whitespace in code (such as present in languages where whitespace is significant) can have a strong impact equivalent to key symbols such as `{`, potentially influencing tokenization and learning.
- **Token Counts and Whitespace**: `@slono` also noted that varying the use of whitespace can lead to significant differences in token counts.
- **Confusion about "[INST]" Context**: `@slono` asked for clarification on the `[INST]` context used by `@eugeneyan`, who explained that it is part of Mistral's prompt format.
- **Learning Parameter Space**: `@__chef__` humorously pondered, "How many parameters does it take to learn a space? Over 7 billion", hinting at the complexity and scale involved in training large models.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- There were discussions regarding **technical difficulties** with LangChain, including concerns about embedding models for JS/TS, issues with viewing old documentation, compatibility inquiries about Llama2 model with SQLDatabaseToolkit, web scraping and source retrieval questions, and installation problems with the langchain python package on Python versions 3.10 and below. A user also brought up worries about consistent document retrieval from Mivus standalone. Another user experienced cache issues with `InMemoryCache()` or `SQLAlchemyCache` in the Conversational Retrieval Chain. A solution for converting a Document to JSON was sought due to a serialization issue.
- A lively **exchange over database preferences** was observed, with users expressing support for LanceDB. 
- In the share-your-work channel, one user, `@marvinbraga`, demonstrated his **[Voice Virtual Assistant on WhatsApp](https://www.youtube.com/watch?v=1k4ADq0quBI)**, as well as promoting a discount for his book and suggesting his [GitHub repo](https://github.com/marvinbraga/marvin).
- Another user, `@bigansh`, announced the launch of **version 2 for myGPTBrain**, with a plethora of new features, an updated landing page, document parsers and the introduction of a subscription model. This was accompanied by a user guide on [Loom](https://www.loom.com/share/dcf1977fffbf413186d6b199151924f4?sid=15c40271-7ccd-4731-ac9f-78cac326c650) and a [public launch blog post](https://open.substack.com/pub/mygptbrain/p/launching-mygptbrain-20?r=2ir30o&utm_campaign=post&utm_medium=web). Feedback on potential new features and design suggestions were requested.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (40 messages🔥): 
        
- **Support for Embedding Models**: User `@jungle_jo` was initially unsure about JS/TS support for embedding models. However, the user later updated their query saying that they've found langchain to have a lot of support for this.
- **Viewing Old Documentation Versions**: User `@ellen0609` inquired about how to view previous versions of langchain's documentation. `@.trouble_` offered to help the user via Direct Message.
- **Compatibility of Llama2 model with SQLDatabaseToolkit**: User `@daksana_40937` asked about the compatibility of Llama2 model with SQLDatabaseToolkit.
- **Web Scraping and Source Retrieval**: User `@abed7053` asked if there's a way to perform web scraping with TypeScript/JavaScript in langchain. This user also couldn't find how to retrieve source context in API response.
- **Langchain Installation Issues on Python v3.10 and Below**: User `@infinityexists.` was having issues with the langchain python package not installing beyond version 0.0.27 on Python version 3.8.0. `@quantumqueenxox` suggested upgrading Python to 3.10 and above.
- **Consistent Document Retrieval Concerns**: User `@ranjith8249` was having issues about retrieving consistent documents from Mivus standalone using langchain conversational chain.
- **Cache Issue in Conversational Retrieval Chain**: User `@seththunder` encountered an issue with `InMemoryCache()` or `SQLAlchemyCache`, stating that neither worked in storing the provided answers in the cache while using Conversational Retrieval Chain in LangChain.
- **Declared Support for LanceDB**: User `@hucki_rawen.io` and `@timcarambat` discussed their preference for LanceDB, a solution they found preferable for their needs.
- **Converting Document to JSON**: User `@b0otable` asked for help converting a Document to JSON, citing an issue with the error message "Object of type Document is not JSON serializable".


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **Creation of Voice Virtual Assistant**: `@marvinbraga` shared a [YouTube video](https://www.youtube.com/watch?v=1k4ADq0quBI) explaining how he created a voice virtual assistant that interacts via WhatsApp. The video covers topics such as OpenAI integration, building an API that stores conversations by ID, integration with WhatsApp via Facebook API, and audio processing with Pygame.
    - Additionally, Marvin shared a discount coupon for his book 'Python, ChatGPT e Django REST', and suggested visiting his [GitHub repository](https://github.com/marvinbraga/marvin) which contains the project's source code.
- **Update Launch of myGPTBrain**: `@bigansh` announced the launch of version 2 for myGPTBrain, including new features, updated landing page, document parsers, and an introduction of user subscriptions. A product update guide is available on [Loom](https://www.loom.com/share/dcf1977fffbf413186d6b199151924f4?sid=15c40271-7ccd-4731-ac9f-78cac326c650), and a [public launch blog post](https://open.substack.com/pub/mygptbrain/p/launching-mygptbrain-20?r=2ir30o&utm_campaign=post&utm_medium=web). The user also requested feedback on potential new features and design suggestions.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Discussion on **ChatGPT's Performance** led by users `@jeffreyw128`, `@res6969`, and `@pantsforbirds`. Three possibilities were proposed: significant "lobotomization", an update to prioritize using fewer tokens, or infrastructure issues leading to a loss of parameters. 
- Inquiry by `@robhaisfield` in the `#finetuning` channel about procuring a JSON for fine-tuning the 3.5-turbo model.
- Conversations around **UNA's ability to align the Mixture of Experts (MoE)** at any level within a neural network. The standout model **Xaberius 34B v1 "BETA"** was specifically mentioned. Further discussion on future focus on **Mixtral** was raised by `@the.palmik`. `@the.palmik` also inquired if anyone had successfully run **Mixtral**, followed by `@robhaisfield` asking about the requirements to get Mixtral running. [Hacker News Post](https://news.ycombinator.com/item?id=38570537)
- User activity notifications and engagements in the `#irl` channel; `@thisisnotawill` announced a temporary leave, `@psychickoala` checked for active users, `@frandecam` confirmed user presence but announced their departure, and `@res6969` expressed positive retrospection about their time spent.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (8 messages🔥): 
        
- **ChatGPT Performance Discussion**: Users `@jeffreyw128`, `@res6969` and `@pantsforbirds` express concerns about the perceived performance drop in ChatGPT. `@res6969` speculated that **ChatGPT has been significantly "lobotomized"**. `@pantsforbirds` proposed that the system might have been updated to **prioritize using fewer tokens** or that there could be infrastructure issues resulting in loss of parameters.


### ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/) (1 messages): 
        
robhaisfield: Anyone have a JSON I can use to fine-tune 3.5-turbo so I can just see how it works?


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (3 messages): 
        
- **UNA aligning the MoE**: `@the.palmik` mentioned that UNA can align the Mixture of Experts (MoE) at almost any level within a neural network. They specifically mentioned **Xaberius 34B v1 "BETA"** as a noteworthy example. They also expressed a future focus on **Mixtral**. [Related Post on Hacker News](https://news.ycombinator.com/item?id=38570537)
- **Inquiry regarding Mixtral Implementation**: `@the.palmik` asked if anyone had successfully run **Mixtral**. This was followed up by `@robhaisfield` asking about the requirements to get Mixtral running.


### ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/) (4 messages): 
        
- User `@thisisnotawill` announced they would be away for a bit.
- User `@psychickoala` later asked if people were still present in the chat.
- `@frandecam` responded to specify that people were indeed still active but they would be leaving.
- User `@res6969` expressed their enjoyment for their time on the chat retrospectively.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- Discussion on the **GPU requirements for Bf16** was triggered by `@ufghfigchv`, noting that it requires the utilization of newer GPUs such as a6000 or a100 for optimal performance.
- `@teknium` shared a [Twitter link](https://fxtwitter.com/teknium1/status/1733233296962953567?s=46) in the general channel without providing further context.
- An exploration into the **Megablocks Research Paper** shared by `@.mrfoo`, with additional insight being provided into different GitHub repositories related to Megablocks: [MistralAI's version](https://github.com/mistralai/megablocks-public) which includes custom code, and the [official version by Stanford Futuredata](https://github.com/stanford-futuredata/megablocks) which was stated to have more recent updates.
- The announcement of **Mistral's new 8x7B Model** by `@moonlightgarden` in the moe-main channel. 
- A shared [link towards Mistral AI's status](https://x.com/mistralai/status/1733150512395038967?s=46&t=HxvRqfgufhVJ4z1puB-WHg) by `@huevosabio` which, unfortunately, led to an error page.
- Pradeep1148 shared a [YouTube video](https://youtu.be/mAGLD5598cs) in the off-topic channel without any additional context.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (2 messages): 
        
- **GPU Requirements for Bf16**: `@ufghfigchv` mentioned that Bf16 is faster, but it requires the use of newer GPUs like a6000 or a100.
- **Twitter Link Shared by teknium**: `@teknium` shared a [Twitter link](https://fxtwitter.com/teknium1/status/1733233296962953567?s=46) with no further context provided.


### ▷ #[papers](https://discord.com/channels/1131084849432768614/1131305311714672710/) (3 messages): 
        
- **Megablocks Research Paper**: `@.mrfoo` shared a research paper titled **Megablocks** which can be found at the [following link](https://arxiv.org/pdf/2211.15841.pdf).
- **Megablocks GitHub Repo by MistralAI**: The [repository](https://github.com/mistralai/megablocks-public) by MistralAI for Megablocks was shared by `@.mrfoo`.
- **Official Megablocks GitHub Repo**: `@stereoplegic` noted that the official Megablocks repository (owned by Stanford Futuredata) was updated more recently. The repo can be found at [this link](https://github.com/stanford-futuredata/megablocks).
- **Custom Code on Mistral's Version of Megablocks**: `@.mrfoo` noted that Mistral's Megablocks repository contains custom code focusing on their new MOE. He also mentioned there is a separate branch for this as well.


### ▷ #[moe-main](https://discord.com/channels/1131084849432768614/1139310171076706464/) (2 messages): 
        
- **Mistral's new 8x7B Model**: User `@moonlightgarden` informed the channel that **Mistral** has released a new 8x7B model.
- **Mistral AI Status Link**: User `@huevosabio` shared a [link to Mistral AI's status](https://x.com/mistralai/status/1733150512395038967?s=46&t=HxvRqfgufhVJ4z1puB-WHg), but the page showed an error message: "*Something went wrong, but don’t fret — let’s give it another shot.*".


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
pradeep1148: https://youtu.be/mAGLD5598cs


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

- An upcoming event titled **Novus #14** was shared by `@jaskirat` with a [link for details and registration](https://lu.ma/novus14), prompting `@Raigon` to inquire about the possibility of event recording. 
- There was a discussion on segmentation models selection with `@mattrixoperations` recommending models like `FastSAM`, `MobileSAM`, `SAM`, and `Yolo-seg`, specially endorsing the `YOLOV8-seg` model. He particularly cautioned against using `SAM` for microscopy tasks, urging for the use of a smaller model and its fine-tuning for tagged data, with the [source](https://docs.ultralytics.com/tasks/segment/#models) mentioned.
- `@erisianrite` announced their plan to use `YOLO` as a baseline for performance comparison with their own model, also showing interest in studying segmentation models and expressing gratitude for `@mattrixoperations`' advice.

**MLOps @Chipro Channel Summaries**

### ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/) (2 messages): 
        
- `@jaskirat` posted a [link](https://lu.ma/novus14) to an event titled **Novus #14**. The post included a [cover image](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/1e/b4acf364-1850-4e27-a7c8-2e3a7d99a152).
- Afterward, `@Raigon` asked if this event was recorded, but the response wasn't provided in this dataset.


### ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/) (2 messages): 
        
- **Segmentation Models Suggestion**: User `@mattrixoperations` shared some insights on segmentation model choices, suggesting `FastSAM`, `MobileSAM`, `SAM`, and `Yolo-seg`. He particularly recommended the `YOLOV8-seg` model but advised against using `SAM` for microscopy tasks. He advised for the use of a smaller model and fine-tuning it on some tagged data ([source](https://docs.ultralytics.com/tasks/segment/#models)).
- **Use of YOLO for Baseline Comparison**: User `@erisianrite` mentioned their plan to utilize `YOLO` as a baseline to compare performance with the model they develop. They also expressed their intent to study segmentation models, expressing gratitude for `@mattrixoperations`' suggestions.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- The **OpenML Guide** was introduced to the community by `@severus_27`. This guide provides an array of *open-source*, *free resources* related to AI, covering various topics like computer vision, NLP, deep learning, AI in healthcare, robotics, and the mathematical principles underpinning AI. 
- The OpenML Guide is accessible via its [website](https://www.openmlguide.org/), and its [GitHub repository](https://github.com/severus27/OpenML-Guide) is available for contributions or support via a GitHub star.

**Alignment Lab AI Channel Summaries**

### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (1 messages): 
        
- **OpenML Guide Introduction**: User `@severus_27` introduced the OpenML Guide, mentioning that it offers a wealth of free resources such as books, courses, papers, guides, articles, tutorials, notebooks and many more for learning topics related to AI like computer vision, NLP, deep learning, AI in healthcare, robotics and the mathematics behind AI's core principles. Additionally, all the resources are open source and freely accessible.
- **OpenML Guide Website and Github Repo**: The OpenML Guide can be accessed at their [website](https://www.openmlguide.org/) and also has a [Github repository](https://github.com/severus27/OpenML-Guide) where the users can contribute or show their support by giving it a star.


### ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/) (1 messages): 
        
- **Introduction to OpenML Guide**: `@severus_27` introduced the **OpenML Guide**, an open-source, free resource offering a considerable array of AI-related content such as books, courses, papers, guides, articles, tutorials, notebooks, and more. The guide caters to a variety of AI interests including computer vision, NLP, deep learning, AI in healthcare, robotics, and the mathematics behind AI's core principles. The [OpenML Guide website](https://www.openmlguide.org/) and the project's [GitHub](https://github.com/severus27/OpenML-Guide) repository were shared.


        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The AI Engineer Foundation Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it
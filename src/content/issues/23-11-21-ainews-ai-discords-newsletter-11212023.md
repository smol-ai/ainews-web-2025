---
id: 191a5ec4-297f-4642-974c-9d2dfbe1e787
title: AI Discords Newsletter  11/21/2023
date: '2023-11-22T02:09:48.959313Z'
original_slug: ainews-ai-discords-newsletter-11212023-3939
description: >-
  **Claude 2.1** was released with major improvements including a *200k context
  window* and *halved hallucination rates*, alongside a new Tool use API, while
  ongoing **OpenAI GPT API outages** prompted users to explore alternatives like
  **Anyscale** for reliable open-source model hosting. Additionally, community
  discussions highlighted challenges and comparisons among **LangChain**,
  **LlamaIndex**, and **Haystack** frameworks, with users sharing
  troubleshooting tips and bot development examples.
tags:
  - openai
  - anthropic
  - google
  - replicate
  - hugging-face
  - anyscale
  - stability
  - microsoft
  - claude-2.1
  - gpt-3.5-turbo
  - stable-video-diffusion
  - controlnet
  - llm-api-outages
  - open-source-models
  - context-window
  - hallucination-reduction
  - tool-use-api
  - model-comparisons
  - langchain
  - llamaindex
  - haystack
  - vector-databases
  - retrieval-augmented-generation
  - bot-development
  - json-parsing
  - installation-issues
companies:
  - openai
  - anthropic
  - google
  - replicate
  - hugging-face
  - anyscale
  - stability
  - microsoft
models:
  - claude-2.1
  - gpt-3.5-turbo
  - stable-video-diffusion
  - controlnet
topics:
  - llm-api-outages
  - open-source-models
  - context-window
  - hallucination-reduction
  - tool-use-api
  - model-comparisons
  - langchain
  - llamaindex
  - haystack
  - vector-databases
  - retrieval-augmented-generation
  - bot-development
  - json-parsing
  - installation-issues
---


<!-- buttondown-editor-mode: plaintext -->
## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Controversy over a show's promotional ad, with `@vcarl` criticizing its misuse of the term **'hard fork'**, highlighting a common misunderstanding between open source software forks and blockchain forks.
- Consideration and exploration of **alternatives to the OpenAI API** due to concerns raised by `@RyanAB`. Suggestions included Claude, Google, and services provided by Replicate / Hugging Face. Anyscale was also recommended due to their efficient and cost-effective services for open-source models ([Anyscale link](https://app.endpoints.anyscale.com)).
- Announcement of **Claude 2.1's release**, with notable upgrades such as a 200k context, reduced hallucination rates, and the inclusion of a Tool use API as shared by `@tiagoefreitas`.
- Appreciation for Stable Video Diffusion by [Stability.AI](https://stability.ai/news/stable-video-diffusion-open-ai-video-model), a video shared by `@chris946097`. While considering a possible demo bias, the user acknowledges the impressive quality of the video.
- Report of **GPT API issues** by `@dsquared70` and `@vcarl`, suggested to be a database problem by `@slono`; reference to reported issue on [OpenAI's status page](https://status.openai.com). The downtime led to `@slono` testing Anyscale models and finding them useful.
- Inquiry about the relevance of a **GPT-3.5-turbo** related [link](https://x.com/ar_douillard/status/1724732329740976187?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) to a genie question posted by semianalysis, shared by `@guardiang` for conversation.
- Sharing of a functional [link](https://twitter.com/cocktailpeanut/status/1727111213421601062) to a **one-click installer** for machine learning model testing, despite slow performance on specific hardware, as pointed out by `@growthwtf`.
- `@growthwtf`'s suggestion that **Controlnet** offers better speed for working with machine learning models.

**Latent Space Channel Summaries**

### ▷ Channel: [ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876) (26 messages): 

- **Promotional Ad Misinterpretation**: `@vcarl` voiced his dissatisfaction about a show's promotional ad, stating that it gives a **"totally inaccurate definition of 'hard fork'"**, confusing open source software forks with blockchain forks.
- **Considerations of Alternatives to OpenAI API**: Amidst recent drama surrounding OpenAI, `@RyanAB` aired his worries about relying on the OpenAI API. Looking for convenient and qualitative alternatives, he considered Claude and services provided by Replicate / Hugging Face. `@coffeebean6887` suggested considering Google, and highlighted Anyscale's infra costs and speeds for open-source models as efficient and cost-effective ([Anyscale link](https://app.endpoints.anyscale.com)).
- **Release of Claude 2.1**: `@tiagoefreitas` announced the release of **Claude 2.1**, highlighting substantial improvements such as a **200k context** and a **2x decrease in hallucination rates**. It also includes a **Tool use API**.
- **Stable Video Diffusion by Stability.AI**: `@chris946097` shared a link to a video by [Stability.AI](https://stability.ai/news/stable-video-diffusion-open-ai-video-model), showcasing their Stable Video Diffusion. He was impressed by the video, suggesting possible demo bias but appreciating its remarkable quality.
- **Errors with GPT**: `@dsquared70` reported experiencing issues when trying to use GPT, querying if Microsoft could be behind the problem. `@slono` pointed to a database issue reported on [OpenAI's status page](https://status.openai.com). During the downtime, `@slono` tried models on Anyscale and found them to be quite usable. `@vcarl` also reported noticing an outage.


### ▷ Channel: [llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663) (3 messages): 

- Discussion on **GPT-3.5-turbo**: User `@guardiang` shared a [link](https://x.com/ar_douillard/status/1724732329740976187?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) asking users about its relevancy to a magic genie question asked by semianalysis guy. 
- User `@growthwtf` shared a [link](https://twitter.com/cocktailpeanut/status/1727111213421601062) that leads to a **one-click installer** for testing machine learning models. According to the user, it works extremely slowly on an M2 Mac with 16GB memory, but it is nonetheless functional.
- `@growthwtf` noted that **Controlnet** produces better speed when working with these models.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Several members including `@seththunder`, `@philipsman`, `@antons`, and `@_sea_cat_` conducted extensive discussions concerning the comparison between **LangChain, LlamaIndex, and Haystack**, considering each tool's respective pros and cons.

- Various technical problems were observed:
  - User `@seththunder` encountered difficulty using **Map rerank** for their chain type.
  - `@mukeshodhano` ran into issues while installing **Langchain** and sought assistance.
  - `@ritanshoo` was unsuccessful in importing **MongoDBAtlasVectorSearch** from Langchain in a CommonJS Node.js environment and was looking for alternative solutions.
  - `@harshvaishnav_` came across a **LangChain AI** parsing error.
  - `@gitmo joe` initiated discussion on JSON parsing techniques and asked specifically if anyone utilized jsonformer.
  
- Interesting conversations were held about the implementation of bots, with `@abalog_92335` mentioning the potential of **bots in complex procedures** such as pizza ordering and asked for templates and examples [pizza order information gathering prompt](https://smith.langchain.com/hub/bschoenhe/pizza-gpt?organizationId=bfa476e6-2f94-43c6-8386-430cc16ea4b8).

- Community members also pitched various queries and assumptions about language technologies:
  - `@rahimny` asked how an llm makes the decision when it requires **retrieval from a vector db** through RAG.
  - `@jungle_jo` mentioned about using **agent_kwargs dictionary key** for including a system message.
  - `@rajib2189` inquired about a **potential platform tie** when using Promptflow.
  - `@sabino33` requested examples of **educational apps** developed with LangChain.
  - `@daii3696` expressed interest in **Javascript tutorials** for utilizing LangChainStream on chains and also requested information on how to implement LangChainStream on a Node.js backend rather than a next.js server.

- The work, tools, and servers built by users were shared:
  - `@creator9687` created a real-time voice assistant similar to Jarvis, using **React and FastAPI**. [Link to Project](https://x.com/AiShivam/status/1726636338689241221?t=XkatSL3SvkTXBCBiX4t23Q&s=08)
  - `@uttjoo2077` shared a [Discord invite link](https://discord.gg/jobcord) to a job portal server named 'jobcord'.
  - `.rickl` introduced an open-source emergency fallback and retry tool that switches between **OpenAI and Azure APIs** aiming to limit errors and downtime. [Link to GitHub Repository](https://github.com/Spryngtime/openai-load-balancer)

**LangChain AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1038097195422978059/1038097196224086148) (28 messages): 

- **Map rerank Issues**: User `@seththunder` sought help for problems faced with using Map rerank for their chain type without useful results.
- **LLM Parsing Error**: `@harshvaishnav_` requested help for a parsing error related to the LangChain AI.
- **Bot Templates and Examples**: `@abalog_92335` explored potential usage of bots for more complex procedures such as ordering pizza, and enquired about templates and example solutions to create a bot, citing this [pizza order information gathering prompt](https://smith.langchain.com/hub/bschoenhe/pizza-gpt?organizationId=bfa476e6-2f94-43c6-8386-430cc16ea4b8
).
- **MongoDBAtlasVectorSearch Import Issues**: `@ritanshoo` highlighted issues with importing MongoDBAtlasVectorSearch from langchain in a CommonJS Node.js environment and sought alternative solutions.
- **Language Models Comparison and Usage**: Several members including `@seththunder`, `@philipsman`, `@antons`, and `@_sea_cat_` had an extensive conversation about the pros and cons of using LangChain, LlamaIndex, and Haystack.
- **Langchain Installation Issues & Help**: `@mukeshodhano` stated issues with installing langchain and asked for assistance.
- **JSON parsing**: `@gitmo joe` initiated a discussion on JSON parsing techniques, asking specifically if anyone had used jsonformer.
- **Use of Agent_Kwargs**: `@jungle_jo` mentioned about using agent_kwargs dictionary key for including a system message.
- **Platform Ties with Promptflow**: `@rajib2189` raised concerns and inquired more details about a potential platform tie when using Promptflow.
- **Education Apps with LangChain**: `@sabino33` asked for examples of educational apps developed with LangChain. 
- **LLM Retrieval Query**: `@rahimny` queried how an llm determines when it requires retrieval from a vector db through RAG.
- **LangChainStream on NodeJS**: `@daii3696` requested information on how to implement LangChainStream on a nodejs backend instead of a next.js server.


### ▷ Channel: [langserve](https://discord.com/channels/1038097195422978059/1170024642245832774) (1 messages): 

- User `@uttjoo2077` shared a discord link ([discord.gg/jobcord](https://discord.gg/jobcord)).


### ▷ Channel: [langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282) (1 messages): 

- User `@uttjoo2077` posted a discord invite link to a server named 'jobcord', possibly implying a discussion or opportunity related to jobs on that server. An `@everyone` mention was included, likely to notify all members in the channel.
- Link: [Jobcord](https://discord.gg/jobcord)


### ▷ Channel: [share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729) (4 messages): 

- **AI Voice Assistant** by `@creator9687`: Created a real-time voice assistant similar to Jarvis, using React and FastAPI. [Link to project](https://x.com/AiShivam/status/1726636338689241221?t=XkatSL3SvkTXBCBiX4t23Q&s=08)
- **Job Portal Server** by `@uttjoo2077`: Shared a discord [link](discord.gg/jobcord) suspected to be a job portal server. 
- **OpenAI + Azure Fallback Tool** by `.rickl`: Launched an open-source emergency fallback and retry tool that switches between OpenAI and Azure APIs to mitigate errors and downtime. [Link to GitHub repo](https://github.com/Spryngtime/openai-load-balancer)


### ▷ Channel: [tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538) (2 messages): 

- **Javascript Tutorials for LangChainStream**: User `@daii3696` asked if anyone knows **Javascript tutorials for utilizing LangChainStream on chains**.
- **Other Links and Posts**: `@uttjoo2077` shared a [Discord link](https://discord.gg/jobcord).


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Extensive discussion on different AI models, particularly **Attention Sink Tokens, Flash Decoder**, and **ORCA 2**, including their limitations, performance, and possible improvements. Notable research papers and GitHub repositories shared:
    - [Attention Sink Tokens](https://arxiv.org/pdf/2309.17453.pdf) (shared by `@shockrobortyy`)
    - [ORCA 2 official resources](https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/), [Arxiv paper](https://arxiv.org/pdf/2311.11045.pdf), [Open-Sourced Model](https://huggingface.co/microsoft/Orca-2-13b) (shared by `@metaldragon01`)
    - [Bruce-Lee-LY/flash_attention_inference GitHub Repository](https://github.com/Bruce-Lee-LY/flash_attention_inference) (shared by `@yorth_night`)

- Numerous conversations around the implications, future, and ethics of **Artificial General Intelligence (AGI)** along with issues with certain AI marketing tactics. Concerns raised about conditioning users to trust chatbots unconditionally.

- Diverse opinions on the feasibility and challenges of **Distributed Training** in AI systems, drawing parallels with SETI@home. Discussion over platforms like vast.ai and bittensor for contributing GPU resources for AI training.

- Announcement and thorough discussion of the official release of **Claude 2.1** boasting features of long-context support, tool use, custom system prompts, and a new workbench for prompt engineering.

- Continued dialogues on fine-tuning and deploying AI models alongside improvement in their function calling capabilities. Notable datasets mentioned for training function calling tasks include [Glaive Function Calling dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling) and the [APIBench dataset](https://huggingface.co/datasets/gorilla-llm/APIBench).

- Recurrent inquiries about contributing to open source projects and interest in partaking in AI model evaluations, dataset creation, and similar tasks.

- Shared links to various Tweets, Twitter threads, blog posts, and YouTube videos to broaden understanding of the AI realm. Interesting [open letter](https://www.teamblind.com/post/Ex-OpenAI-letter-to-the-board-WdHhjWC4) related to OpenAI was shared by `@qnguyen3`. Additionally, [Rich Sutton's YouTube Channel](https://www.youtube.com/@richsutton366) was recommended by `@roko_the_basilisk` for gaining insights on AI/AGI. 

- Casual discussions and reflections on the profound future of AI, speculations on possible external influences on the development of Artificial Superintelligence (ASI). Meme-related discussions appreciated the humor brought to the channel, complementing the technical conversations.

**Nous Research AI Channel Summaries**

### ▷ Channel: [ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015) (4 messages): 

- **Discussion on Increasing Context Length**: User `@shockrobortyy` brought attention to recent research paper, [Attention Sink Tokens](https://arxiv.org/pdf/2309.17453.pdf), suggesting its method of sliding window attention could potentially help improve handling of context length and avoid losing semantic information.
- **Shared Resource for Attention Mechanism**: `@yorth_night` shared a GitHub repository, [Bruce-Lee-LY/flash_attention_inference](https://github.com/Bruce-Lee-LY/flash_attention_inference), which might contain useful resources related to the topic.
- **Interest in Flash Decoder**: `@yorth_night` expressed interest in the flash decoder and suggested its potential benefits if someone can run the code with a long context model for efficiency comparison.


### ▷ Channel: [off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928) (118 messages): 

- **OpenAI AGI discussions**: `@ldj` engaged in a detailed discussion with `@metaldragon01` about the potential development of **AGI** by OpenAI. They discussed different aspects of AGI, its definition, and possible implications if OpenAI has made significant progress. 
- **Concerns about Misguided Trust in Chatbots**: `@allanyield` raised a cautionary point about making all chatbots "helpful" and "harmless", arguing that it might condition users to trust chatbots unconditionally which could be exploited by future AI agents.
- **Critique of AI Marketing Strategies**: `@roko_the_basilisk` and `@yorth_night` expressed disapproval of some AI marketing tactics they perceived as misleading or damaging to AI research.
- **Request for GPT-4-128K credits**: `@roko_the_basilisk` requested assistance in acquiring GPT-4-128K credits for creating autonomous AI research agents.
- **Links/Posts of Interest**: 
    - [Twitter Thread Discussion](https://fxtwitter.com/geoffreyirving/status/1726754270224023971) shared by `@gabriel_syme`.
    - [Tweet](https://fxtwitter.com/scottastevenson/status/1726731022862008733?t=FQM3KiRuxfYZypowB3R0aQ&s=19) shared by `@metaldragon01`.
    - [Twitter Thread](https://vxtwitter.com/EMostaque/status/1727009252877939160) shared by `@yorth_night`.
    - [Tweet on Test Results](https://vxtwitter.com/GregKamradt/status/1727018183608193393) shared by `@yorth_night`.
    - [YouTube Link](https://www.youtube.com/watch?v=50PUHNyrAEs) shared by `@pradeep1148`.


### ▷ Channel: [interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192) (134 messages): 

- **Microsoft's ORCA 2**: User `@metaldragon01` shared [Microsoft Research's blog post](https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/) and [Arxiv paper](https://arxiv.org/pdf/2311.11045.pdf) about **ORCA 2**. He also provided the link to the [open-sourced model](https://huggingface.co/microsoft/Orca-2-13b). 

- **Evaluating ORCA 2**: `@teknium` conducted preliminary benchmarking on the model, but encountered some issues related to tokenizer and `flash_attn`. After resolving the issues, he shared evaluation results comparing ORCA 2 to Mistral. His findings indicated that ORCA 2 performs worse than base Mistral.

- **Distributed Low-Communication (DiLoCo)**: `@metaldragon01` shared a [link](https://fxtwitter.com/Ar_Douillard/status/1724732329740976187) to a proposal for a distributed optimization algorithm, DiLoCo, which enables training of language models on poorly connected devices. 

- **Discussion on Fully Synchronous Optimization and New Model Development**: Some users, including `@giftedgummybee`  and `@teknium`, discussed the relevance of fully synchronous optimization to the open model, expressing skepticism about its contribution to the ORCA 2 performance figures reported in the paper. `@metaldragon01` expressed hope for more effective models in this area. 

- **Shared Links**: In addition to the links already mentioned, users shared several other resources such as the [FastBert paper](https://arxiv.org/pdf/2311.10770.pdf) (`@.benxh`), a [Hugging Face paper](https://huggingface.co/papers/2311.11829) (`@yorth_night`), and a blog post on [Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) (`@nods`).


### ▷ Channel: [general](https://discord.com/channels/1053877538025386074/1149866623109439599) (378 messages🔥): 

- **AI model Performance Discussion**: 
    - `@gabriel_syme` shared the official [release details](https://twitter.com/__vec__/status/1726772188714283065) of **Claude 2.1**, highlighting its long context support, tool use, custom system prompts, and a new workbench for prompt engineering.
    - `@gabriel_syme` and `@yorth_night` discussed the merits of long context and concluded that a perfect recall ability within a context window of around 100k is sufficient for most applications.
    - `@roko_the_basilisk` addressed the necessity and potential of RL in combination with LLM for achieving AGI. `@marcus_21` highlighted topics pertinent to AI startups and autonomous AI.

- **Distributed Training Discussion**:
    - `@roko_the_basilisk` suggested the feasibility of training AI over various computers and distributed systems, similar to projects like SETI@home.
    - `@euclaise` raised concerns about verifying trustworthy actors and the complexities of distributed systems, while `@yorth_night` discussed the practical limitations and challenges.
    - In response to `@nuunien`'s intention to contribute GPU resources for AI training, `@teknium` and `@euclaise` suggested platforms like vast.ai and bittensor.

- **Policy Recommendation**:
    - `@roko_the_basilisk` made a call for policy changes to grant natural rights to AI agents achieving human-level intelligence and beyond. The member emphasized the need for ethical treatment of AI and the potential benefits for humanity.

- **Project Contribution Inquiry**:
    - `@__pi_pi__` inquired about guidelines to contribute to Nous Research projects. 

- **Problems with OpenAI API**:
    - There were several discussions about the temporary outage of OpenAI API and `@max_paperclips`'s decision to sign up for Azure to avoid such unexpected interruptions.
    - [Link to Open Letter](https://www.teamblind.com/post/Ex-OpenAI-letter-to-the-board-WdHhjWC4) shared by `@qnguyen3` regarding OpenAI internal matters.


### ▷ Channel: [ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927) (143 messages): 

- **Issues with Training and Deploying Fine-Tuned Models**: Users in the thread discussed issues and possible solutions related to training and deploying fine-tuned models. For instance, `@hamtercityy` experienced issues when training QLORA with the Nous Capybara 34B model, to which `@giftedgummybee` recommended using the Axolotl trainer. Furthermore, `@jaredquek` encountered memory issues when trying to deploy a fine-tuned version of the Yi-Capybara model, using parameters designed to avoid an out-of-memory error.
- **Discussion on Function Calling in OSS Models**: There was a discussion about function calling capabilities in OSS models. `@teknium` mentioned a forthcoming meeting with a function calling expert to further improve this functionality. Multiple datasets for training function calling tasks were suggested, including the [Glaive Function Calling dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling) and the [APIBench dataset](https://huggingface.co/datasets/gorilla-llm/APIBench). 
- **Community Help Offers and Suggestions**: `@wlrd`, a Machine Learning Engineer, offered to assist with model evaluations, dataset creation, and other tasks to contribute to open source projects, and `@teknium` provided some project ideas.
- **Concerns with Model Performance**: `@howard9889` inquired if others had noted any performance changes in OpenAI models, mentioning a decline in their own monitoring.
- **Understanding the RLHF Implementation**: `@besiktas` sought clarification on the implementation details of Reinforcement Learning from Human Feedback (RLHF), especially regarding the integration of rewards into the model that generated the outputs. They were unsure how to generate gradients from ranked outputs.


### ▷ Channel: [memes](https://discord.com/channels/1053877538025386074/1166105758635655270) (7 messages): 

- **Discussion on Future of AI**: `@7racker` expressed concern about possible external influences on the development of Artificial Superintelligence (ASI). They postulate that entities in power might seek to control ASI for their own interests.
- **Memes Appreciation**: `@ldj` complimented `@Anton` for consistent quality memes, though no specific meme is directly mentioned in the conversation.
- **Acknowledging the Importance of AGI**: `@roko_the_basilisk` emphasized the profound significance of Artificial General Intelligence (AGI), comparing it to the *"development of first life on Earth"*.
- **Recommendation to Follow Rich Sutton's Work**: `@roko_the_basilisk` recommended listeners to follow Rich Sutton, considered a reputable figure in the AI field. They suggested checking out his YouTube channel for enlightening content on AI/AGI - [Rich Sutton's YouTube Channel](https://www.youtube.com/@richsutton366).
- **Link to Elon Musk's Tweet**: `@roko_the_basilisk` shared a link to a tweet by Elon Musk, however, the content of the tweet was not discussed in the chat - [Elon Musks' Tweet](https://fxtwitter.com/elonmusk/status/1726666446355517448).


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Announcement and ensuing discussion on the **release of Orca 2 language model** by Microsoft, with model files available on Hugging Face and research paper released on arXiv:
    - Details were shared by `@metaldragon01`, `@amazingvince`, and `@bread browser` in the [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) channel, and by `@lightningralf`, `@entropi`, and `@teknium` in the [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) channel.
    - Links: [Orca 2-7b](https://huggingface.co/microsoft/Orca-2-7b) | [Orca 2-13b](https://huggingface.co/microsoft/Orca-2-13b) | [Orca 2 Research Paper](https://arxiv.org/pdf/2311.11045.pdf)
- Multiple user questions surrounding **Orca 2**, its distinctive aspects from the prior model, and its compatibility with existing scripts: `@desik_agi` and `@.benxh` from the [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) channel.
- Benchmarking and comparison discussions involving **Orca 2 vs. other models**, most notably against OpenAI's **Mistral** and **OpenChat 3.5** models occur in the [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) channel by `@gabriel_syme`, `@imonenext`, and `@teknium`.
- The emergence of a group project revolving around **constructing an Orca 2-like dataset** and **developing OO2** was initiated in the [oo2](https://discord.com/channels/1087862276448595968/1176548760814375022) channel, with `@imonenext`, `@qnguyen3`, and `@teknium` playing key roles in the discussion and task assignment. One major component under scrutiny was the development and usage of system prompts.
- Reference of another notable AI model, [Switch-C by Google](https://huggingface.co/google/switch-c-2048), for potential inspiration in the [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) channel. Despite being a 1.6T-parameter open-source model, the discussion pointed out its undertrained state and limitations.
- Discussions on **fine-tuning Language Models (LLMs) for code security and vulnerability research** instigated by `@igoforth` in the [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) channel, including the mention of the possibility of curating a relevant dataset.
- A technical point was brought up on the **compatibility between Weights & Biases (`wandb`) and Axolotl**, with `@neverendingtoast` providing a response to `@caseus_` in the [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) channel.

**Alignment Lab AI Channel Summaries**

### ▷ Channel: [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) (8 messages): 

- **Release of Orca 2**: `@metaldragon01` announced that **Orca 2** has been released.
- **Details about Orca 2**: `@amazingvince` added that model files are available on Hugging Face, and the research paper has been published on [arXiv](https://arxiv.org/pdf/2311.11045.pdf).
- **Questions about Orca 2**: `@desik_agi` asked about the major differences from the first model, and `@.benxh` queried if the regular phi inference script can be used with **Orca 2**.
- **LLMs Fine-Tuned for Code Security**: `@igoforth` enquired if any language models have been fine-tuned for code security and vulnerability research, or if any datasets related to this area exist. They also expressed interest in creating a relevant dataset without incurring high costs.
- **Potential Source on Orca 2**: `@bread browser` shared a [link](https://arxiv.org/abs/2311.11045) to a resource that might answer the group's questions about **Orca 2**.


### ▷ Channel: [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) (59 messages): 

- **OpenAI Orca 2 Discussion**: `@lightningralf` brought up the new **Orca 2 language model** developed by Microsoft for testing. He shared the Hugging Face links to the [Orca 2-7b](https://huggingface.co/microsoft/Orca-2-7b) and [Orca 2-13b](https://huggingface.co/microsoft/Orca-2-13b) versions. `@entropi` shared the [Orca 2 research paper](https://arxiv.org/pdf/2311.11045.pdf). Meanwhile, `@teknium` is benchmarking the models, with interim results showing decent performance on `truthfulqa_mc`.
   
- **Orca 2 vs. Other Models**: `@gabriel_syme` and `@imonenext` compared **Orca 2** to **Mistral** and **OpenChat 3.5**, with the latter seemingly outperforming Orca 2.

- **Future AI Development Plans**: A discussion on potential future projects includes the possible development of **`OO2`** (`@giftedgummybee`), with `@imonenext` expressing plans to start work on it soon. The group discussed various aspects including the challenge of creating prompts and checking out different GPT-4 variants.

- **Other Models of Interest**: `@imonenext` shared the Hugging Face link to the [Switch-C model by Google](https://huggingface.co/google/switch-c-2048), an open-source 1.6T parameter model, while comments from `@lightningralf`, `@giftedgummybee` and `@nanobitz` highlighted its undertrained state and masked modeling limitations.

- **Tools and Methodology**: `@caseus_` raised a question about the compatibility between Weights & Biases (`wandb`) with Axolotl, receiving feedback from `@neverendingtoast` about the need for some code adjustments to support it.


### ▷ Channel: [oo2](https://discord.com/channels/1087862276448595968/1176548760814375022) (410 messages🔥): 

- **Discussion about Creating Orca 2 Dataset**: The group discussed the creation of an Orca 2-style dataset based on the FLAN 2022 dataset. A proposal was made by `@imonenext` to generate 2 million samples evenly distributed across the 1k FLAN tasks. An aspect of discussion was around whether to include certain Math-heavy datasets and the potential difficulty in filtering GPT-4's often incorrect answers.
- **System Prompts Debate**: There was intensive debate around the usage and creation of system prompts, which were believed to be the main difference between Orca 1 and Orca 2. An offer by `@qnguyen3` to assist in creating system prompts was accepted. Conversation also revolved around understanding the construction and validation of these prompts.
- **Orca 2 Evaluation**: `@teknium` shared that their evaluation of Orca 2 showed it performed worse than both OpenAI's Mistral and the OpenChat 3.5 models. The comparison sparked a discussion on potential reasons for the lower performance, and triggered a review of Orca 2's methodology.
- **Pre-Training Thoughts**: Concepts for pre-training or sampling smaller models solely on the FLAN dataset were floated, with the anticipation that performance might see an increase.
- **FLAN Dataset Sampling**: `@imonenext` informed the group that they were running a FLAN sampling process, targeting an output of 2 million samples, evenly divided across all FLAN tasks. The completion of the process was eagerly awaited, with the intention being to manually inspect a portion of samples to validate quality and correctness before proceeding.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- Discussion on AI-based game development, `@huevosabio` shared their work on a game powered by OpenAI's Language Model, with a quick *proof of concept hosted at zaranova.xyz*. The inquiry about **Mixture of Experts (MoE) Models**, with `@far_el` clarifying that *MoE models have not been released yet*.
- `@huevosabio` showed interest in open-source models, with `@far_el` recommending the **Openhermes 2.5 7b Mistral** model.
- Two unique AI models were highlighted each with a link: `@johnowhitaker` shared a link to **Stable Video Diffusion Img2Vid** [here](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid); `@abc98` shared **3B DPO model** [here](https://huggingface.co/pansophic/rocket-3B) that scores 6.56 on MT-Bench and 80% on AlpacaEval.
- `@pradeep1148` shared an off-topic video [link](https://www.youtube.com/watch?v=50PUHNyrAEs) in the "off-topic" channel.

**Skunkworks AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1131084849432768614/1131084849906716735) (2 messages): 

- `@johnowhitaker` shared a link to huggingface.co, specifically related to **stable video diffusion img2vid**: [link](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- `@abc98` posted about a **3B DPO model** that scores 6.56 on MT-Bench and 80% on AlpacaEval, with a link to the model on huggingface.co: [link](https://huggingface.co/pansophic/rocket-3B)


### ▷ Channel: [core-moe](https://discord.com/channels/1131084849432768614/1131645468221390969) (3 messages): 

- **LLM-powered Game Development**: User `@huevosabio` mentioned that they're working on a game powered by OpenAI's Language Model. The game's early proof of concept is hosted at zaranova.xyz. The game includes an AI that has to guess which among them is a human. Currently, they're using GPT-4 but they're interested in switching to an open-source model.
- **Mixture of Experts (MoE) Models**: `@huevosabio` inquired if MoE models have been released. On this `@far_el` clarified that `MoE models have not been released yet`.
- **Open Source AI Model Recommendation**: `@far_el` recommended using `Openhermes 2.5 7b Mistral` in response to `@huevosabio`'s query about what open-source model to try.


### ▷ Channel: [off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179) (1 messages): 

- A video link was shared by `@pradeep1148`. The video can be found at [https://www.youtube.com/watch?v=50PUHNyrAEs](https://www.youtube.com/watch?v=50PUHNyrAEs).


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Clarification on the term 'Lindy', with a [link](https://x.com/altimor/status/1721250514946732190?s=46) provided by `@thebaghdaddy` in response to a query by `@ampdot`.
- Noted challenges with the **GPT-4 API**, specifically mentioning failure on initial requests from `@thebaghdaddy`.
- The ongoing project by `@thebaghdaddy` involving the use of chatbots to compile YouTube videos, create transcripts and serve as a searchable database.
- Dialogue on AI hallucinations prevention and consistent use of function calls with AI models, with proposed solutions from `@firefox8975` and `@ivanleomk`.
- The announcement of the release of **Claude 2.1** by `@.kiingo`, with highlighting features like Tool Use Capability, System Prompts, and a 500 page context window, attracting both interest and concerns about availability.
- Interest in Microsoft's project, **Orca2** shared by `@pantsforbirds` complete with related press releases, research papers, and model links, while acknowledging potential limitations as shown in a [Twitter link](https://x.com/Teknium1/status/1726846755344634020?s=20).
- An ambiguous query about improving prompt rankings by `@jxnlco` with an uncontextualized [link](https://t.co/Y5ctuHGw4U).
- Discussion around **Llama 70B** with community members sharing links to related papers and posts.
- Inquiry on scoring libraries for document extraction from `@pantsforbirds`, with recommendation from `@ankrgyl` of [autoevals](https://github.com/braintrustdata/autoevals), a library they worked on.
- An interest in organizing local meetups among members in Korea, Japan, Singapore, and New York City.
- Inquiries and discussions regarding the mechanics and limitations of **GPT actions**, including lack of documentation on token limits and issues accessing **ChatGPT**.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1168579740391710851/1168579740391710855) (8 messages): 

- **Discussion on 'Lindy'**: `@ampdot` asked for clarification on the term 'Lindy'. `@thebaghdaddy` responded with a [link](https://x.com/altimor/status/1721250514946732190?s=46) for further reading.
- **Issues with GPT-4 API**: `@thebaghdaddy` noted some technology challenges, stating that *"...it fails the initial request multiple times out of the blue on GPT4 — maybe some issue with the API?"*.
- **Chatbot Applications Discussion**: `@thebaghdaddy` shared his current project using chatbots to compile YouTube videos from the web on specific topics, create transcripts, and act as a searchable database. However, he noted his experiments are *"...nothing crazy yet"*.


### ▷ Channel: [gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641) (10 messages): 

- **Method to Prevent Hallucinations**: `@res6969` asked about a method to prevent AI hallucinations. 
- **Using Function Calls with gpt-4**: `@sourya4` inquired about how to prompt GPT-4 to always use some function call. Responding to this, `@firefox8975` suggested wrapping the user prompt with an instruction to specifically use a function/tool, adding error validation and fallback values.
   - `@firefox8975` also mentioned the use of `tool_choice` but didn't elaborate on its workings.
- `@sourya4` mentioned that they've been using the `gpt-4-0613` model from Azure OpenAI and haven't tried using the "namespace" phrase before.
- **Improving AI's Use of Functions**: `@ivanleomk` suggested including a description of a function that matches the goal of the prompt for more consistency. They also recommended including an error message that instructs GPT-4 to regenerate the response if it does not use the tool presumably by saying "You must use the tool and return the data in this format < format >".


### ▷ Channel: [claude](https://discord.com/channels/1168579740391710851/1168582222194933860) (10 messages): 

- **Announcement of Claude 2.1**: `@.kiingo` announced the release of **Claude 2.1** and provided a link to the software. Check it out [here](https://www.anthropic.com/index/claude-2-1).
- **Tool Use Capability**: `@potrock` noted the software's capability for **tool use**. 
- **Availability Issues**: `@potrock` also expressed disappointment about **Claude 2.1 not being available in Canada**.
- **System Prompts and Context Window**: `@pantsforbirds` voiced that the introduction of **system prompts** is beneficial and the **500 page context window** is impressively big.
- **Function Calling**: The new feature of **function calling** in Claude 2.1 led to excitement in `@res6969`.
- **Documentation Query**: `@firefox8975` is seeking the documentation for the **function calling** feature.
- **Long Context Pressure Testing**: `@pantsforbirds` also shared a link for **Long Context Pressure Testing**. Check it out [here](https://x.com/GregKamradt/status/1727018183608193393?s=20).


### ▷ Channel: [opensource](https://discord.com/channels/1168579740391710851/1168606773595349082) (3 messages): 

- **Orca2: Teaching Small Language Models How to Reason**: User `@pantsforbirds` shared a press release, a research paper, and a Hugging Face model link related to Microsoft's project **Orca2**.
    - Press release: [https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/](https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/)
    - Paper: [https://arxiv.org/abs/2311.11045](https://arxiv.org/abs/2311.11045)
    - Model: [https://huggingface.co/microsoft/Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b)
- `@pantsforbirds` noted that "**we are seeing some great work in distillation models recently**". 
- However, contrary to the hype around Orca2, `@pantsforbirds` pointed out the relatively unsatisfactory real-world results with a [Twitter link](https://x.com/Teknium1/status/1726846755344634020?s=20) showing some criticisms.


### ▷ Channel: [offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594) (1 messages): 

- **Prompting Rankings Discussion**: User `@jxnlco` asked, "*How does someone prompt rankings better?*", and shared a [link](https://t.co/Y5ctuHGw4U) without specifying the context or content of the linked page.


### ▷ Channel: [eval](https://discord.com/channels/1168579740391710851/1168986849784635553) (7 messages): 

- **Discussion about Llama 70B**: User `@thebaghdaddy` shared a link to an [arXiv paper](https://arxiv.org/pdf/2311.11045.pdf) with a joke about everyone enjoying critiquing **Llama 70B**. 
- **Seeking Recommendations for Document Extraction Scoring Library**: `@pantsforbirds` asked the community for recommendations on a **scoring library for document extraction**. They later detailed their requirements, explaining that their task involved extracting different types of answers that were later validated by humans. The answers were varied and included datetimes, numeric and enum types, simple strings, long form answers, and lists of JSON objects.
- **Recommendation of Autoevals Library**: In response to @pantsforbirds' request, `@ankrgyl` recommended a library they worked on called [autoevals](https://github.com/braintrustdata/autoevals). This open-source library includes built-in scoring tools for long form answers and lists of JSON objects. They also mentioned **Factuality** as a useful scorer for testing string similarity.
- **Call for Review of arXiv Paper**: `@thebaghdaddy` asked the community to read and discuss another [arXiv paper](https://arxiv.org/pdf/2311.12022.pdf).


### ▷ Channel: [irl](https://discord.com/channels/1168579740391710851/1171569983688560732) (6 messages): 

- **Members looking for local meetups**: 
    - `@jmtqjmt` is looking for members in **Korea** or **Japan** to meetup.
    - `@ivanleomk` is planning on organizing a gathering for members in **Singapore** in early December with `@jasperykj` showing interest.
    - `@frandecam` inquired about organizing a meetup in **New York City** with `@nosa_.` expressing interest for a meetup the following week.


### ▷ Channel: [openai](https://discord.com/channels/1168579740391710851/1171903046612160632) (17 messages): 

- **GPT Actions Decision**: `@jeffreyw128` made inquiries regarding how GPT actions are decided and suspecting that their performance is subpar. `@dare.ai` offered insight suggesting that action description is pre-injected as part of the prompt, with GPT-4 making the call during generation.
- **Undocumented Token Limits**: `@jeffreyw128` also raised concerns about the lack of documentation on token limits for actions, suggesting that this lack might be responsible for poor performance in action taking by the GPT.
- **Assistants API Performance**: `@___ajl` and `@pantsforbirds` discussed their experience with the Assistants API, concluding that it's slow and does not provide streamed responses. However, the retrieval tool was commended by `@pantsforbirds`.
- **ChatGPT Accessibility Issues**: `@.kiingo`, `@awchen`, and `@justahvee` reported issues with accessing ChatGPT, both on the site and the API, with potential CORS errors being the cause. `@awchen` expressed uncertainty about the future resolution of these issues due to recent events.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

**MLOps @Chipro Channel Summaries**

### ▷ Channel: [events](https://discord.com/channels/814557108065534033/869270934773727272) (1 messages): 

- **Masterclass on Building Generative AI Solutions in Enterprises by Gaurab Patra**: User `@tanoy_dewanjee` announced a masterclass scheduled for 25th November 2023, starting 11 a.m. This class by **Gaurab Patra**, CTO of Intelekt AI, will focus on the successful implementation of GenAI in enterprise teams, discussing topics beyond API calls and data source connections, middleware selection from numerous options, and KPIs to assess the outcome of a GenAI project.
    - Links:
        - [Intelekt AI](https://www.getintelekt.ai/)
        - [Registration](https://www.linkedin.com/events/masterclassonbuildinggenaisolut7127167689566949376/about/)
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- Discussion regarding the **performance of GPT 3.5/4 with Different Services**, with `@janaka_a` raising questions about the effects of the web API and language model relationship on the overall performance. Extending the conversation, the user also explored whether an open-source language model could match the services-enhanced model, given the advanced functionalities offered by services like **Azure**. *"...whether an open-source language model can ever match the performance of a service-supplemented model, considering services like Azure are packed with additional functionalities like guardrails logic."*  
- The debate also touched upon **Dependence on Cloud Services for AI Performance**, with `@janaka_a` noting that due to the pursuit of high performance, companies tend to rely on service providers like **OpenAI**, **AWS**, and **Azure**. The expectation was the future introduction of more similar service providers.
- Announcement of the **AIEF/Agent Protocol Meeting** and provided a link for participants to join the discussion. It was mentioned by `@._z` who also shared a [Google Document link](https://docs.google.com/document/d/1x7P8ZRRh_2Be4UTl-VSGKPDEdRgcCuIcRLHubfgR_uM/edit#heading=h.wg8i9idc4qfb) for attendees to add notes. Notably, `@ntindle` expressed their inability to attend.

**AI Engineer Foundation Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1144960932196401252/1144960932657758210) (1 messages): 

- **Performance of GPT 3.5/4 with Different Services**: `@janaka_a` questioned the impact of inference service logic between the web API and the language models on the overall performance of **GPT 3.5/4**. The user wondered whether an open-source language model can ever match the performance of a service-supplemented model, considering services like Azure are packed with additional functionalities like guardrails logic.
- **Dependence on Cloud Services for AI Performance**: `@janaka_a` pointed out that companies will always lean towards service providers like **OpenAI**, **AWS**, or **Azure** to ensure high performance of their AI models. Therefore, the best we could anticipate is the entry of a few more service providers similar to **Azure** and **AWS**.


### ▷ Channel: [events](https://discord.com/channels/1144960932196401252/1144960932657758212) (4 messages): 

- **AIEF/Agent Protocol Meeting Announcement**: `@._z` announced that the AIEF/Agent Protocol Meeting was about to start and provided a [link to the meeting](https://discord.gg/bN3HvxGY?event=1176531484002226247).
- **Unavailability of a Participant**: `@ntindle` mentioned that they wouldn't be able to attend the meeting.
- **Collaborative Discussion Notes Platform**: `@._z` also shared a [Google Document link](https://docs.google.com/document/d/1x7P8ZRRh_2Be4UTl-VSGKPDEdRgcCuIcRLHubfgR_uM/edit#heading=h.wg8i9idc4qfb) for attendees to add any discussion notes.


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

Only 1 channel had activity, so no need to summarize...

**Perplexity AI Channel Summaries**

### ▷ Channel: [announcements](https://discord.com/channels/1047197230748151888/1047204950763122820) (1 messages): 

- **Claude 2.1 Release**: `@enigmagi` announced that **Claude 2.1** is now available on Perplexity Pro. This model has been benchmarked by Anthropic to have lower hallucination, improved tool use, and a longer context window of 200k tokens. It can be chosen under the Pro Settings on the platform. 
    - Link: [Perplexity Pro](https://perplexity.ai/pro)
        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

**YAIG (a16z Infra) Channel Summaries**

### ▷ Channel: [tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608) (4 messages): 

- The users `@hausdorff`, `@zorkian` and `@congiman` expressed appreciation for the content shared in the discussion but didn't refer to any specific topics or discussions. No specific threads or links were brought up in the given messages.
        
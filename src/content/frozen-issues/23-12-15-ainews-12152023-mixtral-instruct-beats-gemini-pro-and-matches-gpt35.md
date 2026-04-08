---
id: aaf39f9e-09fa-4e00-b45d-9330235210ce
title: '12/15/2023: Mixtral-Instruct beats Gemini Pro (and matches GPT3.5)'
date: '2023-12-15T22:33:20.436628Z'
type: archival
original_slug: ainews-12152023-mixtral-instruct-beats-gemini-pro
description: >-
  Thanks to a **karpathy** shoutout, **lmsys** now has enough data to rank
  **mixtral** and **gemini pro**. The discussion highlights the impressive
  performance of these state-of-the-art open-source models that can run on
  laptops. In the **openai** Discord, users compared AI tools like
  **perplexity** and **chatgpt's browsing tool**, favoring Perplexity for its
  superior data gathering, pricing, and usage limits. Interest was shown in AI's
  ability to convert large code files with **deepseek coder** recommended.
  Debates on privacy implications for AI advancement and challenges of running
  LLMs on local and cloud GPUs were prominent. Users reported issues with
  **chatgpt** including performance problems, loss of access to custom GPTs, and
  unauthorized access. Discussions also covered prompt engineering for large
  context windows and speculations about **gpt-4.5** and **gpt-4** future
  developments.
companies:
  - lmsys
  - openai
  - deepseek
  - cloudflare
  - huggingface
models:
  - mixtral
  - gemini-pro
  - gpt-3.5
  - gpt-4.5
  - gpt-4
  - chatgpt
topics:
  - performance
  - context-window
  - prompt-engineering
  - privacy
  - local-gpu
  - cloud-gpu
  - code-generation
  - model-comparison
  - model-usage
  - api-errors
people:
  - karpathy
---


<!-- buttondown-editor-mode: plaintext -->Thanks to a [Karpathy](https://twitter.com/karpathy/status/1734687074350166089) shoutout, Lmsys  [now has enough data to rank Mixtral and Gemini Pro](https://twitter.com/lmsysorg/status/1735729398672716114/photo/1):

![image.png](https://assets.buttondown.email/images/18b77832-b18d-4497-9a49-dcda81bfa548.png?w=960&fit=max) 

Very impressive for a SOTA open source model you can run on your laptop. Discords are also positively reviewing Mistral-medium and confused about the Le Platforme API.

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Intense discussions about **various AI models and tools** were held, with users comparing the likes of **Perplexity and ChatGPT's browsing tool**; `@chief_executive` favored Perplexity due to its superior data gathering capabilities compared to ChatGPT, its pricing, and usage limits. Tools like **Gemini Pro** and **Mistral** were also mentioned for their distinct capabilities.
- Users expressed interest in AI's ability to **convert large code files to another programming language**, with **DeepSeek Coder** being recommended for such tasks. Problems related to context window size with large models were discussed.
- An lively debate on the potential implications of the **eradication of privacy for AI advancement** was led by `@【ｐｅｎｕｌｔｉｍａｔｅ】` and `@qpdv`.
- There were several points about handling **LLMs on local and Cloud GPUs** - focusing on cost-effectiveness, complications, and choices for suitable GPUs were detailed by users like `@afterst0rm` and `@lugui`.
- Users also reported multiple issues regarding OpenAI services, including **performance issues** with ChatGPT, **loss of access** to custom GPTs, **unauthorized access**, and **problems with publishing GPT** as public reported by several participants. The new **archive button** feature in ChatGPT was a topic of discussion too.
- Future developments in AI were piqued, including discussions and speculations about **GPT 4.5** and GPT 4, and the **performance of various AI models** in different tasks.
- Queries regarding **ChatGPT Discord bot** and **increased errors with the Assistant API** were addressed in the OpenAI Questions channel.
- There were also inquiries and advice exchanged about **crafting prompts for LLM with large context window** and handling large contexts in [prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) and [api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) channels.
  
*Note: All usernames and direct quotes are mentioned in italics.*

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (154 messages🔥🔥): 
        
- **Performance of Various AI Models and Tools**: 
    - `@chief_executive` compared the information gathering capabilities of **Perplexity** and **ChatGPT's browsing tool**, stating that *Perplexity is superior and doesn't take as much time as ChatGPT*. 
    - `@chief_executive` and `@pietman` also discussed about **Perplexity's pricing and usage limits**. `@chief_executive` mentioned Perplexity has more than 300+ GPT-4 and Copilot uses a day, it's worth the cost due to its browsing capabilities.
- **Converting Programming Languages via AI**: 
    - `@Joakim` asked about AI models that can convert large code files to another programming language. 
    - `@rjkmelb` recommended trying out **DeepSeek Coder** for such tasks while `@afterst0rm` pointed out issues related to context window size in large models and suggested breaking down larger tasks for better results.
- **Discussion on Privacy and AI**: 
    - `@【ｐｅｎｕｌｔｉｍａｔｅ】` and `@qpdv` had a discussion speculating total removal of privacy for AI progress. Discussions pivoted towards potential *implications of interconnectivity and privacy*.
- **Utilization of AI models**: 
    - `@chief_executive` shared their experience with **Gemini Pro** mentioning its video analysis capabilities and Multi-Model Language Model (LLM) potential. 
    - `@afterst0rm` shared that their work uses **Mistral for classification (~80% right)** and **GPT-3.5 for generative tasks**. They also mentioned services like **Cloudflare** and **Poe** for supporting Mistral which offer lower latency.
- **Running LLMs on Local GPUs and Cloud GPUs**:
    - `@afterst0rm` and `@lugui` discussed the complications and costs of running LLMs locally on GPUs in the current market and how lower-cost alternatives like HuggingFace or Cloudflare can be effective. `@lugui` mentioned the **free tier option on Cloudflare** as a beneficial feature.
    - `@millymox` sought advice on selecting a cloud GPU for their use case and compared pricing and specifications of **RTX 3090 with 24GB VRAM** and **A6000 with 48GB VRAM**. Various users suggested getting whatever has more VRAM for the best price, and that having more memory could be more beneficial than more tensor cores.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (546 messages🔥🔥🔥): 
        
- **ChatGPT New Features**: `@klenak`, `@thecroug` and `@dr.quantum` discussed a newly added archive button next to the three-dot menu in the ChatGPT history interface. The new button archives chat and removes it from the history. Users indicated difficulty in finding an option to access the archived chats.
- **False Information Regarding GPT 4.5**: `@DawidM` and `@satanhashtag` had a debate over the credibility of a leak about GPT 4.5. They noted that the rumor originated from Reddit and agreed that future announcements should be considered reliable only if they come directly from OpenAI or verified sources.
- **AI Performance in Varying Tasks**: In the discussion regarding AI capabilities, `@cynicalcola` queried the best AI model between Claude Pro or ChatGPT Plus with GPT 4 Advanced Data Analysis for summarizing and interpreting PDFs. `@rjkmelb` and `@australiaball` supported ChatGPT stating that Claude seems to lose context in complicated/long documents while ChatGPT is able to focus on details and discuss anything the user asks it to, but the choice depends on the speed of handling and the complexity of data under consideration.
- **ChatGPT Performance Issues**: Multiple users, including `@jcrabtree410` and `@.australiaball`, reported experiencing issues with browsing speed and lag in the ChatGPT service. They indicated these issues have persisted for several days. It was suggested that there might be a server issue causing this.
- **GPT Comparison**: During a discussion on the performance of different versions of the GPT model, `@arigatos` and `@barret` agreed that the python language tends to work best with the GPT model due to its linguistic similarities to English, which makes the translations more accurate and reliable.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (102 messages🔥🔥): 
        
- **ChatGPT Discord bot on server**: A user (`@jah777`) inquired if there is a ChatGPT bot for Discord that they could use for their server. It was addressed by `@satanhashtag` stating that it is possible with the [API](https://github.com/openai/gpt-discord-bot) but it's not free.
  
- **Increased errors with Assistant API**: `@sweatprints` reported seeing failed requests with the assistant API over the past few hours, suggesting an increase in errors.

- **Inability to access chatGPT**: Many users such as `@sharo98`, `@s1ynergy`, and `@crafty.chaos` have reported experiencing problems accessing their ChatGPT on their computers. Various possible solutions have been suggested by `@solbus`(one being trying different browsers and also checking network settings), but no definitive solution seems to have been found yet.

- **Loss of access to custom GPTs**: `@milwaukeeres` raised concern about losing access to a custom GPT they had spent time building. `@fulgrim.alpha` also inquired about sharing access to a custom GPT without a Plus subscription, but `@solbus` explained that this is currently not possible.

- **Unauthorized access to account**: `@babbalanja` reported a breach of their account, with new unauthorized chats conducted. They were advised by `@solbus` to change their account password and reach out for assistance on help.openai.com.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (48 messages🔥): 
        
- **Problems with Publishing GPT as Public**: Users including `@franck.bl`, `@optimalcreativity`, and `.parkerrex` have reported issues with the option to publish a GPT as public being greyed out or disabled. Despite having previously published GPTs, they are now unable to do so and suspect a glitch. `@elektronisade` also confirmed that the confirm button seems to switch to disabled with delay or on click. `@optimalcreativity` reported seeing errors in the browser console.
- **Archive Button Concerns**: `@imdaniel__`, `@rjkmelb`, and `@gare_62933` shared their experiences with the 'Archive' button. Conversation threads were disappearing from their menu, they were able to recover only by using browser history.
- **Chat GPT lagging issue**: User `@.australiaball` experienced issues with Chat GPT lag. The chat displayed, but the user could only scroll the side bar or had trouble interacting with the interface. `.australiaball` asked users to react with a thumbs up if they faced same issue.
- **Interest in GPT-4 Development**: `@lyraaaa_` joked about wanting to use GPT-4 as a junior developer if there were 8xA100 instances available.
- **Work Performance of Different AI**: `@drcapyahhbara` shared their opinion that while Gemini appears to be overhyped and underperformed, GPT-4 and Bing AI seem to be performing well, with Bing AI showing marked improvement.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (9 messages🔥): 
        
- **LLM System Prompt techniques for RAG systems**: User `@jungle_jo` is inquiring about crafting prompts for LLM with a context window that receives lots of information. The user provided a sample style for the prompt.
- **Handling Large Contexts in ChatGPT**: `@eskcanta` provided their experience with secretkeeper, which involves pulling data from uploaded knowledge files up to 300k characters. They note that although ChatGPT handles this scale of data, it might not work effectively with larger, more complex inputs.
- **Marking Up Reference Texts**: `@knb8761` asked whether triple-quotes are still necessary for marking up reference texts, citing the [OpenAI Platform documentation](https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions). The current practice is to just insert an empty line and paste the reference text.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (9 messages🔥): 
        
- **LLM System Prompt techniques for RAG systems**: `@jungle_jo` sought advice on techniques to use for an **LLM system prompt in RAG systems**, elaborating that the issue is how the LLM will deal with a lot of information in its context window to best answer a user query. They provided a prompt style example to demonstrate their point.
- **Uploaded Knowledge Files**: `@eskcanta` shared their experience with pulling data from uploaded knowledge files. They mentioned that they have **used up to 300k characters** of plain text but haven't gone above that yet.
- **Usage of Triple Quotes for Marking Up Reference Texts**: `@knb8761` asked if triple quotes are still needed for marking up reference texts, referencing the [OpenAI documentation](https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions) which recommends using them. They noted that they usually paste the reference text after an empty line following their question.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Extensive discussion on AI model performance, including specific mentions of **SOLAR-10.7B**, **GPT-3.5**, **Tulu**, **Qwen 72B**, **Hermes 2.5**, **PHI-2b**, and **OpenChat 3.5**. Notably, `@nemoia` argued that SOLAR-10.7B outperforms all the 13B models while `@agcobra1` doubted it surpassing larger models like Qwen 72B. In addition, `@n8programs` reported slower than expected run times for PHI-2b.

- The Discord community dialogued on the best models for **embeddings and vector storage**. Suggestions included **fast embed from Quadrant** and **gte-small**. `@lightningralf` shared cautionary [tweet](https://twitter.com/somewheresy/status/1735725994600738983) about synthetic data generation from an unmodified embedding of scientific papers on ArXiv using ada-002.

- Conversation on desired features and performance for model blending and merging sparked interest in tools like [mergekit](https://github.com/cg123/mergekit). Participants compared different infrastructures for running models, such as **MLX**, **Ollama**, and **LM Studio**. 

- GPU requirements for AI training were debated, with ancillary discussion on industry advancements like the upcoming **nVidia H200**. There were questions about the requirements for **renting GPUs**, with suggestions for services that accept cryptocurrencies.

- Dialogue about potential **benchmark contamination** specifically in relation to models trained on the metamath dataset. `@nonameusr` expressed concern, and `@tokenbender` linked to a similar discussion on [Huggingface](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/265#657b6debf81f6b44b8966230).

- Detailed guidance provided on the process of building **tailored evaluations** for AI models. `@giftedgummybee` offered a six-step strategy which included components such as identifying the evaluation scope and compiling results data.

- A new model, ["Metis-0.1"](https://huggingface.co/Mihaiii/Metis-0.1), was announced by user `@mihai4256` as a result of reasoning and text comprehension. Highlighting a strong performance on GSM8K and trained on a private dataset.

- Shared YouTube video links for ["Viper - Crack 4 Tha First Time"](https://www.youtube.com/watch?v=eRNcm7FQln4) and ["Stanford CS25: V3 I Recipe for Training Helpful Chatbots"](https://www.youtube.com/watch?v=mcep6W8oB1I), as well as discussions about the strength of open-ended search with external verifiers, the **Phi-2** update, and a comparison of **FunSearch** with **openELM**.

- A GitHub discussion on **state-space model architecture**, Mamba, was initiated by `@vincentweisser`. Related [GitHub link](https://github.com/havenhq/mamba-chat) and [Arxiv paper](https://arxiv.org/pdf/2312.00752.pdf) about the same were shared.

- A discussion regarding the significance of a **benchmark performance**, with a score of **74**, considered substantial by `@gabriel_syme` assuming that the community could fine-tune for improved results.

- A question about the experience in replacing one of the experts with another pre-trained one in Mixtral model was raised by `@dragan.jovanovich`. A similar thread by `@crainmaker` was posed about the parameter differences required for **character-level transformers** to achieve comparable performance to **BPE-based transformers**.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (1 messages): 
        
nonameusr: this makes a lot of sense


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (8 messages🔥): 
        
- **Discussion on Embeddings and Vector Storage**: `@adjectiveallison` initiated a discussion on the best models for embeddings and vector storage, and specifically asked about the performance of **Jina 8k**.
- **Fast Embed from Quadrant**: `@lightningralf` suggested using fast embed from quadrant if that solution is being used. He urged caution in adopting a particular model due to the possibility of embedding the entire **Arxiv database**. 
- **Reference to a Twitter Post**: `@lightningralf` shared [a link to a tweet](https://twitter.com/somewheresy/status/1735725994600738983) that refers to the generation of synthetic data from an unmodified embedding of every scientific paper on **ArXiv**. This process apparently uses **ada-002**.
- **Preference for gte-small**: `@natefyi_30842` shared that they prefer using **gte-small** for their work, arguing that larger embeddings like **Jina** are better suited to projects involving large volumes of data, such as books.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (2 messages): 
        
- **Benchmark Performance Discussion:** User `@artificialguybr` mentioned that another user had already conducted a benchmark test, stating there was **not a big gain**. In response, `@gabriel_syme` suggested that a score of **74** could be considered substantial, and assumed that the community could fine-tune for even more significant results.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (11 messages🔥): 
        
- **Open-ended Search with External Verifiers**: User `@gabriel_syme` commented on a discussion about the strength of open-ended search with external verifiers.
- **Phi-2 Inquiry**: User `@huunguyen` asked for opinions or updates on **Phi-2**.
- **FunSearch and ELM**: `@gabriel_syme` compared FunSearch to openELM in-context and mentioned he will look into the islands method. He compared it to Quality-Diversity algorithms but noted they're not the same.
- **Shared YouTube Links**: `@nonameusr` shared a music video titled ["Viper - Crack 4 Tha First Time"](https://www.youtube.com/watch?v=eRNcm7FQln4) and `@atgctg` shared ["Stanford CS25: V3 I Recipe for Training Helpful Chatbots"](https://www.youtube.com/watch?v=mcep6W8oB1I) from YouTube.
- **Metis-0.1 Model Announcement**: `@mihai4256` introduced a 7b fine-tuned model named ["Metis-0.1"](https://huggingface.co/Mihaiii/Metis-0.1) for reasoning and text comprehension, stating that it should score high on GSM8K and recommending few-shot prompting while using it. He emphasized that the model is trained on a private dataset and not the MetaMath dataset.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (189 messages🔥🔥): 
        
- **Concerns about Benchmark Contamination**: Several users discussed the problem of models trained on benchmark datasets, specifically metamath. `@nonameusr` expressed concern that any models using metamath could be contaminated, and `@tokenbender` linked to a discussion on [Huggingface](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/265#657b6debf81f6b44b8966230) on the same issue. There were calls for people to avoid using metamath in their blends.
- **Performances of Different Models**: The performance and utility of different AI models were discussed, including Hermes 2.5, PHI-2b, and OpenChat 3.5. `@tsunemoto` offered to provide mistral medium prompt replies on request. `@n8programs` reported that PHI-2b runs slower than expected at 35 tokens/second (mixed-precision) using mlx on an m3 max.
- **Interest in Merging AI Models**: Various users expressed interest in merging different models to enhance performance. The toolkit [mergekit](https://github.com/cg123/mergekit) by `@datarevised` was suggested for this purpose, and there was discussion on merging Open Hermes 2.5 Mistral due to its good foundational properties.
- **Use of Different Inference Engines**: There was discussion about the use of different infrastructures for running models, such as MLX, Ollama, and LM Studio. `@n8programs` reported poor results from mlx, while `@coffeebean6887` indicated that ollama would likely be faster, even at higher precision than the default 4 bit.
- **Neural Architecture Search**: `@euclaise` raised the idea of using Neural Architecture Search instead of merging parameters, on which there was brief discussion on potential application to subset of the search space concerning different model merges.
- **GPU requirements for AI**: Debate was sparked about the demand for GPU resources in AI training and if centralized or decentralized compute would be more efficient. The discussion also drifted to industry advancements and potential upcoming GPU resources, notably the nVidia H200.
- **Potential Issues with AI Services**: User `@realsedlyf` asked about the requirements for renting GPUs, specifically if a credit card was required, to which `@qasb` and `@thepok` suggested services accepting cryptocurrencies.
- **Neural Model Blending**: User `@dragan.jovanovich` brought up a question about experience in replacing one of the experts with another pre-trained one in Mixtral model.
- **Model Performance Comparison**: A [twitter post](https://fxtwitter.com/lmsysorg/status/1735729398672716114/photo/2) from LMSysOrg comparing the performance of different models was shared by user `@artificialguybr`. Mixtral-Medium and Mistral-Medium were suggested for addition to the comparison by `@zakkor` and `@adjectiveallison` respectively.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (45 messages🔥): 
        
- **Discussion about Model Performance**: The users debated over the performance of various models. `@nemoia` mentioned that SOLAR-10.7B outperforms all the 13B models, and `@n8programs` shared anecdotal results from experimenting with Tulu, describing it as almost equal to GPT-3.5. However, `@agcobra1` expressed doubt about SOLAR-10.7B surpassing larger models such as Qwen 72B.
  
- **Evaluation Strategy for Models**: A discussion was sparked about building tailored evaluations. `@natefyi_30842` shared that he needs to evaluate multiple use-cases, and `@giftedgummybee` provided a detailed 6-step guide, which includes identifying the evaluation scope, curating a list of ground truth examples, verifying that they aren't in common datasets, building a structure to test the model, and compiling the results as data.

- **Llava API Issue**: `@papr_airplane` reported encountering a ValueError when working with llava's API regarding mismatching numbers of `<image>` tokens and actual images.

- **Character-Level Transformer vs BPE-Based Transformer**: `@crainmaker` asked about the parameter differences required for character-level transformers to achieve comparable performance to BPE-based transformers.

- **State-Space Model Architectures**: `@vincentweisser` brought up discussion about the state-space model architecture called Mamba. Shared relevant [GitHub link](https://github.com/havenhq/mamba-chat) and [Arxiv paper](https://arxiv.org/pdf/2312.00752.pdf).


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mixtral Configuration and Use**: Discussion on issues with Mixtral's configuration, with `@dizee` resolving issues by adjusting `config.json` files. Further conversation about the non-usage of the sliding window in Mixtral models, instruction of setting the context length at 32768, given by `@tlacroix_`.
- **Chatbots and Documents Classification with Mistral**: Users `@.akatony` and `@naveenbachkethi_57471` initiated a discussion on building a chatbot and using Mistral for document classification. Various web UI recommendations have been provided for use with Mistral API, including the [OpenAgents](https://github.com/xlang-ai/OpenAgents) project suggested by `@cyborgdream`.
- **The Performance and Optimisation of Models**: A series of messages, mainly by `@ex3ndr` and `@choltha`, voiced frustration towards the model's inability to accurately answer rule-based questions and offered potential solutions for optimizing the MoE architecture.
- **SillyTavern Roleplay and API Modification**: Conversation initiated by `@aikitoria` advising `@xilex. ` on how to modify the OpenAI API URL in the backend of SillyTavern for roleplay/story specifications.
- **Library Updates, Input Data, and Fine-Tuning**: Noticeable library updates by `@cpxjj` announcing the new version of LLaMA2-Accessory and technical discussions on fine-tuning parameters and correct data input format. Discrepancies were noted by `@geedspeed` between example resources and original instructions. Suggestions were made for learning rates on Mixtral fine-tuning.
- **Showcase of New Developments**: Users shared new projects, including the open-sourced library for the integration of Mistral with .NET by `@thesealman` and the latest version of LLaMA2-Accessory from `@cpxjj`.
- **Discussion and Requests on Features of la-plateforme**: Users inquired about and discussed features of the Mistral platform including API rate limits, open-sourcing the Mistral-Medium model, potential feature requests regarding function calling and VPC Deployment capabilities, comparisons and feedback on different Mistral versions, and Stripe verification issues.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (90 messages🔥🔥): 
        
- **Mixtral Configuration Issues**: User `@dizee` discussed some issues with getting Mixtral running with llama.cpp, pinpointing issues with config.json files. They were able to get the Mixtral branch working, but noted that the main llma.cpp repo was causing the error.

- **Document Classification with Mistral**: User `@naveenbachkethi_57471` inquired about the possibility of using Mistral for document classification similarly to Amazon Bedrock. User `@potatooff` suggested the use of few shot prompts for this purpose.

- **Sliding Window in Mixtral Models**: User `@tlacroix_` clarified that for Mixtral models, the sliding window should not be used. The context length for these models remains at 32768. 

- **Chatbot Integration Discussion**: User `@.akatony` initiated a discussion about building a chatbot, with `@.dontriskit` suggesting a combination of Rocketchat with live chat plugin and N8N with LLM.

- **Mixtral Context and Configuration Questions**: In a conversation about Mixtral's context and configuration, `@tlacroix_` explained that the context length for Mixtral is 32768, referencing the sliding_window setting.

- **Available Chatbot UI for Mistral API**: `@cyborgdream` asked for recommendations on good web UI to use with Mistral API that support certain features. `@lee0099` suggested Chatbot UI and `@cyborgdream` found [OpenAgents](https://github.com/xlang-ai/OpenAgents) that suits the outlined requirements.

- **Introduction of Streamlit Chatapp**: `@jamsyns` shared a [Streamlit chat app](https://github.com/vindiw/mistral-streamlit-chat) which interacts with Mistral's API.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (9 messages🔥): 
        
- **Performance of Models Running on Default Parameters**: `@svilupp` was considering testing models by adjusting parameters, but was uncertain if it would be a fair comparison since typically models are tested with their default settings.
- **Models' Ability to Answer Rule-Based Questions**: `@ex3ndr` expressed frustration that the current model can't correctly answer questions about the rules of FreeCell, a card game.
- **Potential Modifications to MoE Architecture**: `@choltha` proposed an idea for optimising the MoE architecture by adding a "forward-pass-termination" expert that would force the model to skip a step if it is only tasked with "easy" next tokens.
- **Locally Running Mistral-Embed**: `@talon1337` asked for information about Mistral-Embed, expressing interest in knowing the recommended distance function and how it can be run locally.
- **Details on Mixtral Model Training Datasets**: Both `@pdehaye` and `@petet6971` had queries about Mixtral dataset specifics - `@pdehaye` was interested in what the 8 experts in Mixtral are expert at and `@petet6971` sought information or a license related to the datasets used to train Mixtral-8x7B and Mixtral-8x7B-Instruct.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (5 messages): 
        
- **Roleplay/Story Utilization of SillyTavern**: `@aikitoria` suggested `@xilex.` to use **SillyTavern** for roleplay/story specifications, mentioning that while it is not officially supportive for **Mistral API**, a workaround can be achieved by swapping the OpenAI API URL in the backend.
- **Swapping API URL in SillyTavern**: `@aikitoria` provided guidance on where to modify the API URL in SillyTavern's backend - in the `server.js` file.


### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (2 messages): 
        
- **Discussion on Jinja Templates in Different Projects**: User `@saullu` questioned why `@titaux12` thought that a Jinja template had to be fixed. `@saullu` clarified that no one is saying it's wrong, but highlighted that they only guarantee the reference. They also pointed out that using the chat template creates bugs in `vllm` and provided a link to [the issue](https://github.com/vllm-project/vllm/issues/2012).
- **Utility of Jinja Templates for Supporting Multiple LLMs**: `@titaux12` responded by expressing the usefulness of a working Jinja template for projects that enable multiple LLMs to run seamlessly. They also mentioned working on supporting multiple models in `privateGPT`.
- **Recommendations for Complying with Model Outputs**: `@titaux12` stated that they would do their best to comply with the output their model wants, emphasizing the importance of the token representation (`1` (BOS)) in the first position, with no duplication or encoding as another token.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (13 messages🔥): 
        
- **Fine-tuning on Apple Metal using mlx / mlx-examples**: `@cogbuji` asked if the Mistral model from the [mlx-examples](https://github.com/ml-explore/mlx-examples) is suitable for fine-tuning for Q/A instructions on Apple Metal. They also asked if there's a separate 'instruction' model that should be used for that purpose.
- **Recommended Learning Rates For Tuning Mixtral**: Both `@ludis___` and `@remek1972` suggested that learning rates around 1e-4 and 0.0002 respectively have yielded good results when tuning Mixtral.
- **New Version of LLaMA2-Accessory and Finetuning Parameters**: `@cpxjj` announced the latest version of LLaMA2-Accessory which supports inference and instruction fine-tuning for mixtral-8x7b, and shared the [detailed documentation](https://llama2-accessory.readthedocs.io/en/latest/projects/mixtral-8x7b.html) for it. They also provided links to [full finetuning settings](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/dialog_ultrachat200kWizardcode_mistral.sh) and [peft settings](https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/accessory/exps/finetune/sg/dialog_ultrachat200kWizardcode_mistralPeft.sh). 
- **Confusion about Formatting Input Data**: `@geedspeed` expressed confusion about how to format the data input for fine-tuning of mistral-7b-instruct and cited an [example colab notebook](https://colab.research.google.com/drive/1JtrVh--bcPR-CR8QNOyXd3Z5eZt0WgOw?usp=sharing) from AI makerspace. They noticed a discrepancy between the example notebook and the instructions from Mistral about using [INST] tokens.
- **API for Fine-tuning**: `@robhaisfield` suggested the implementation of an API for fine-tuning, specifically mentioning the need to fine-tune "medium". `@jamiecropley` echoed this sentiment, noting it isn't clear whether the current API supports fine-tuning.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (6 messages): 
        
- **Mistral with .NET**: User `@thesealman` has completed an open-sourced library for using **Mistral with .NET**. The user would appreciate any feedback and has released a NuGet package. The GitHub repo is available at [https://github.com/tghamm/Mistral.SDK](https://github.com/tghamm/Mistral.SDK).
- **Understanding Mistral for Language Learning App**: New community member `@webcreationpastor` is seeking resources to understand AI, specifically **Mistral**, for application in a language learning app. 
- **Blogposts for Understanding Mistral**: In response to `@webcreationpastor`'s query, `@daain` has shared a series of blogs for understanding Mistral. The available posts include [Understanding embeddings and how to use them for semantic search](https://www.danieldemmel.me/blog/understanding-embeddings-and-how-to-use-them-for-semantic-search) and [What are large language models and how to run open ones on your device](https://www.danieldemmel.me/blog/what-are-large-language-models-and-how-to-run-open-ones-on-your-device).
- **LLaMA2-Accessory Support for mixtral-8x7b**: User `@cpxjj` has announced their latest version of **LLaMA2-Accessory**, which now supports both inference and instruction finetuning on the **mixtral-8x7b** model. Detailed documentation is available at [https://llama2-accessory.readthedocs.io/en/latest/projects/mixtral-8x7b.html](https://llama2-accessory.readthedocs.io/en/latest/projects/mixtral-8x7b.html).
- **Support for Mistral AI API in Microsoft's Autogen**: `@tonic_1` shared a link to a [GitHub issue](https://github.com/microsoft/autogen/issues/991) discussing adding support for the **Mistral AI API** (and Mixtral2) in Microsoft's Autogen.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (1 messages): 
        
balala: Привет


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (68 messages🔥🔥): 
        
- **Rate Limits and API access**: User `@robhaisfield` asked about the API rate limits, with `@tlacroix_` providing an answer of 2M tokens per minute, 200M per month. The conversation also suggested the possibility of the JavaScript client being made open-source to accept PRs for better compatibility.
  
- **Open-source Mistral Medium discussions**: A discussion sparked by `@netbsd` raised the question about open-sourcing the Mistral-Medium model. The community shared mixed thoughts about the feasibility and the impact of doing so.
  
- **Potential Mistral Platform Feature Requests**: Users showed interest in function calling with `@lukasgutwinski`, VPC Deployment capabilities with `@pierre_ru`, and Mistral Large with `@casper_ai` all voicing their interest in these features. `@tlacroix_` confirmed that VPC Deployment capabilities is being worked on.
  
- **Mistral Medium Feedback and Comparisons**: Various users like `@rad3vv`, `@lukasgutwinski`, and `@flyinparkinglot` shared positive feedback about the performance of Mistral Medium. `@tarruda` also noted that he found the performance of Mistral Tiny superior to a quantized version of Mistral on Huggingface.
  
- **Stripe Verification Issue**: User `@phantine` faced issues with phone number verification via Stripe while subscribing, but the issue was resolved through using the email verification option.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Detailed discussion on fine-tuning **Mistral**, touching on handling out-of-memory errors, GPU VRAM requirements for qLoRA Mixtral training, and model distribution across multiple GPUs using **deepspeed** or **Data parallel**. Mentioned a [Twitter post](https://twitter.com/SebastienBubeck/status/1735688006332485834) by **Sebastien Bubeck** on the positive results from fine-tuning **phi-2** on math exercises.
- In Axolotl development, understanding and handling of GPU memory errors was explored, with a resourceful [PyTorch blog post](https://pytorch.org/blog/understanding-gpu-memory-1/?utm_content=275432243&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024) shared. There were also attempts to resolve import errors related to DeepSpeed PyTorch extension, subsequent TypeError encounter, and a possible [github PR hotfix](https://github.com/OpenAccess-AI-Collective/axolotl/pull/951) for the issue.
- Persistent issues regarding **ChatML template** and **Axolotl's** inference capabilities were discussed in general help, particularly focusing on the inability of the models to properly learn.
- In datasets, a [ToolAlpaca GitHub repository](https://github.com/tangqiaoyu/ToolAlpaca) generalizing tool learning for language models was shared.
- Notable problems regarding inference on **Runpod** came to light, particularly involving long inference times and lack of completion in comparison to **HuggingFace platform**. Also, queries on how to stop **Runpod** pods automatically after training instead of default restart behavior.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (16 messages🔥): 
        
- **Mistral Fine-Tuning with Sample Packing Issues**: User `@dirisala` mentioned that they encountered out-of-memory errors when trying to fine-tune **Mistral** (not the latest version) with sample packing enabled. The errors were resolved by disabling sample packing.
- **VRAM Requirement for qLoRA Mixtral Training**: `@jaredquek` asked about the VRAM requirement for training qLoRA Mixtral. `@faldore` responded by stating that the training used **4 A100s**.
- **Model Distribution Across Multiple GPUs**: User `@kaltcit` inquired if axolotl supports distributing the model across several GPUs for training. `@nruaif` confirmed that it's possible either by using **deepspeed** or normal **Data parallel**. They also added that if the model doesn't fit on a single GPU, **deepspeed 3** can be used.
- **Tweet On Phi-2 Fine-tuning**: `@noobmaster29` shared a [Twitter link](https://twitter.com/SebastienBubeck/status/1735688006332485834) from **Sebastien Bubeck** discussing the promising results from fine-tuning **phi-2** on 1M math exercises similar to CodeExercises and testing on a recent French math exam.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (20 messages🔥): 
        
- **Volunteer for Template Work for Training**: User `.tostino` expressed a willingness to participate in getting the chat templates operative for training and stated that the main task is for training data to run through the template and properly tokenize/mask some parts of the training data.

- **GPU Memory Issue**: `@tmm1` shared a [PyTorch blog post](https://pytorch.org/blog/understanding-gpu-memory-1/?utm_content=275432243&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024) on handling and understanding GPU memory errors which concern the error message:  `torch.cuda.OutOfMemoryError: CUDA out of memory`.

- **Attempt to Resolve ImportError**: User `@hamelh` is trying to assist `@teknium` in resolving an ImportError, specifically pertaining to the shared object file: `/home/nous/.cache/torch_extensions/py310_cu117/fused_adam/fused_adam.so`. There is speculation that it could be due to a bad installation or something to do with flash attention within the DeepSpeed PyTorch extension.

- **TypeError Encounter following PyTorch Fix**: Following the fix of the PyTorch issue, `@hamelh` encountered a TypeError related to the LlamaSdpaAttention.forward() function. A YAML file and full traceback were provided to assist troubleshooting.

- **Potential Hotfix**: `@caseus_` mentioned that a PR in Axolotl's repository might fix the TypeError encountered by `@hamelh`. The PR is meant to address issues caused by the latest release of Transformers, which changed the use of SDPA for attention when flash attention isn't used, thereby breaking all the monkey patches. You can find the PR [here](https://github.com/OpenAccess-AI-Collective/axolotl/pull/951).


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (84 messages🔥🔥): 
        
- **Model Training and Templating Issues**: 
    - `@noobmaster29` and `@self.1` discuss issues with the ChatML template and Axolotl's inference capabilities. The primary concern seems to be the failure of models to properly learn the `


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (2 messages): 
        
- **ToolAlpaca Repository**: User `@visuallyadequate` shared a [GitHub link](https://github.com/tangqiaoyu/ToolAlpaca) to **ToolAlpaca**, a project that generalized tool learning for language models with 3000 simulated cases.


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (2 messages): 
        
- **Issues with Inference on Runpod**: `@mustapha7150` is running a [huggingface space](https://huggingface.co/spaces/diffusers/stable-diffusion-xl-inpainting) on runpod (1x A100 80G) and experiencing issues with extremely long inference times of almost one hour which never complete, despite it taking only a few seconds on HuggingFace's platform. They've also tried using an A40 with no notable difference.
- **Query on Automatically Stopping Pods Post-Training**: `@_jp1_` asked for advice on how to automatically stop pods after training has finished, instead of having the default behavior of restarting. They were unable to find a way to change this beyond manually stopping it via API.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- A vibrant dialogue took place on the classification of **GPU performance levels**, instigated by `@nerdimo`'s question about the ranking of an RTX 4080. `@osanseviero` suggested that any set-up without 10,000+ GPUs could be considered "[GPU poor](https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini)".
- Several users experienced and offered solutions for **technical issues**. For instance, `@vishyouluck` encountered a problem using the `autotrain_advanced` function, with `@abhi1thakur` recommending them to post their issue on GitHub for more thorough assistance.
- An in-depth conversation revolved around **model hosting and inference speed**. `@sagar___` sought out tips for hosting a TensorRT-LLM model, and they, along with `@acidgrim`, discussed the challenges of memory use during inference and the concept of employing multiple servers for larger models. 
- Multiple users expressed curiosity and shared their discoveries about recurrent neural networks, sparked by discussions on **GRU vs. RNN, LSTM preference, and the TD Lambda algorithm**. In this context, `@nerdimo` noted they were using the **Andrew Mg ML** spec as their learning resource.
- Users shared and reviewed their creations, such as `@not_lain`'s RAG based space for **pdf searching**, `@aabbhishekk0804`'s Hugging Face space for PdfChat using the **zephyr 7b model**, and `@vipitis`'s development of a new metric for fine-tuned models.
- The **reading group** planned to review a paper on diffusion models titled "[On the Importance of the Final Time Step in Training Diffusion Models](https://arxiv.org/abs/2305.08891)". `@chad_in_the_house` also offered insights into the problems with common diffusion noise schedules and sampler implementations.
- The guild announced the deployment of **Diffusers benchmarks** for tracking the common pipeline performance in `diffusers`, with automatic reporting managed by a `benchmark.yml` [file](https://github.com/huggingface/diffusers/blob/main/.github/workflows/benchmark.yml) in the GitHub repository.
- Various users raised queries, shared resources and guidance about training **diffusion models with paired data, Segformer pretrained weights discrepancy, context length of Llama-2 model**, the procedure for **fine-tuning language models**, and the **size and download time for Mixtral-8x7B model**.
- Among cool finds shared, noteworthy were the [Reddit discussion](https://www.reddit.com/r/flipperzero/comments/15me1ew/ai_generated_code/) on AI generated scripts for **Flipper Zero**, and a [HuggingFace paper](https://huggingface.co/papers/2312.08723) on an end-to-end music generator trained using deep learning techniques.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (56 messages🔥🔥): 
        
- **GPU Classes and Performance**: `@nerdimo` and `@osanseviero` had a discussion about GPU classes and performance. `@nerdimo` asked if an RTX 4080 would be considered mid-class or poor-class GPU, to which `@osanseviero` responded by mentioning an [article](https://www.semianalysis.com/p/google-gemini-eats-the-world-gemini) that humorously claims everyone without 10k+ GPUs is GPU poor.
- **Issues with autotrain_advanced**: `@vishyouluck` had issues when using `autotrain_advanced` for finetuning a model. `@abhi1thakur` responded that more information was needed to help and suggested posting the issue on GitHub. 
- **Survey on LLM Tool Usage**: `@jay_wooow` shared a [survey](https://tally.so/r/mO7q0p), which aims to understand the motivations and challenges in building with LLMs (Large Language Models).
- **Segformer Performance Concerns**: `@shamik6766` expressed concerns that the pretrained Segformer model on Hugging Face does not match the official mIOU (mean Intersection over Union) values mentioned in the paper.
- **Model Hosting and Inference Speed**: `@sagar___` asked for suggestions on how to host a TensorRT-LLM model permanently, observing that memory usage goes up during inference. `@acidgrim` also discussed running stablediffusion and Llama on mpirun, considering the use of multiple servers for large models.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 messages): 
        
- **GRU vs. RNN**: User `@nerdimo` mentioned that their intuition was that the addition of gates made the GRU more complex than a standard RNN. They also indicated that they are learning from the **Andrew Mg ML** spec, and they plan to supplement their learning through side projects.
- **LSTM Preference**: `@nerdimo` expressed a preference for LSTM over other options, citing that it provides more filtering and parameters.
- **TD Lambda Algorithm**: User `@d97tum` shared that they are trying to code the **TD Lambda algorithm** from scratch.
- **GRU vs. LSTM**: `@merve3234` clarified that they were comparing GRU to LSTM, not RNN, which aligns with `@nerdimo`'s earlier comment.
- **Model Addition to stabilityAI**: `@nixon_88316` queried if anyone was available to add any model to StabilityAI.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (4 messages): 
        
- **AI generated code for Flipper Zero**: User `@_user_b` shared a [link](https://www.reddit.com/r/flipperzero/comments/15me1ew/ai_generated_code/) to a Reddit discussion about the possibility of generating scripts for **Flipper Zero**, a fully open-source customizable portable multi-tool for pentesters and geeks.
- **TensorRT-LLM model hosting**: `@sagar___` asked for suggestions on how to keep a model loaded continuously using TensorRT-LLM. 
- **Music Streaming with AI**: `@not_lain` shared a [HuggingFace paper](https://huggingface.co/papers/2312.08723) about an end-to-end music generator trained using deep learning techniques, capable of responsive listening and creating music, ideal for projects like radio or Discord bots. The user was impressed by the model's consistent performance and the fact that it can stream music without waiting for the AI to finish processing.
- **Feedback on Musical AI**: The user `@osanseviero` responded positively to the shared HuggingFace paper commenting it was "very cool".


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (3 messages): 
        
- **RAG Based Searching Space**: `@not_lain` has completed building a RAG based space designed for pdf searching using the `facebook/dpr-ctx_encoder-single-nq-base` model. The model and its related code can be accessed via [this link](https://huggingface.co/spaces/not-lain/RAG).
- **PdfChat Space**: `@aabbhishekk0804` announced the creation of a space on Hugging Face for PdfChat. This utilized the **zephyr 7b model** as an LLM and can be viewed [here](https://huggingface.co/spaces/Aabbhishekk/ChatPdf).
- **Metric Development**: `@vipitis` shared preliminary results for the metric they are developing. This includes a problem with post-processing that needs to be addressed, as identified with fine tuned models.


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (4 messages): 
        
- **Next Paper for Reading Group**: `@chad_in_the_house` suggested the next paper for the reading group to be "[On the Importance of the Final Time Step in Training Diffusion Models](https://arxiv.org/abs/2305.08891)" due to its importance & influence in the field. Moreover, he mentioned the significant work `@636706883859906562` has done in relation to this, sharing a link to the [Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-text-to-image.ipynb) of his implementation.
- **Discussion on Diffusion Models**: `@chad_in_the_house` highlighted the issues of common diffusion noise schedules not enforcing the last timestep to have zero signal-to-noise ratio (SNR) and some sampler implementations not starting from the last timestep, indicating that these designs are flawed and cause discrepancies between training and inference.
- **Problem Implementing the Paper**: `@chad_in_the_house` found it interesting that stability failed to implement this exact research paper, inviting further discussion on this topic.
- **Epsilon Loss Problems**: `@pseudoterminalx` pointed out that the implementation fell short due to issue with epsilon loss.


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **Introduction of Diffusers Benchmarks**: User `@sayakpaul` announced the introduction of **Diffusers benchmarks** to track the performance of most commonly used pipelines in `diffusers`. The [`benchmarks are accessible here`](https://huggingface.co/datasets/diffusers/benchmarks).
- The **automatic reporting workflow** is managed by a file named `benchmark.yml`, residing in the `.github/workflows` directory of the `diffusers` GitHub repository. The [`workflow file can be found here`](https://github.com/huggingface/diffusers/blob/main/.github/workflows/benchmark.yml).
- The **benchmarks** include several configurations, such as `StableDiffusionXLAdapterPipeline`, `StableDiffusionAdapterPipeline`, and `StableDiffusionXLControlNetPipeline` using TencentARC models like `t2i-adapter-canny-sdxl-1.0` and `t2iadapter_canny_sd14v1`.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Discussion on Diffusion Model Training**: User `@guinan_16875` inquired about the possibility of training diffusion models with paired data, offering an example of a child's photo and a corresponding photo of the father. They questioned if the practice of training diffusion models with images of the same style could be seen as a form of style transfer.
- **Diffusion Models & Paired Data Training**: In response to `@guinan_16875`'s query, `@asrielhan` suggested looking into **Instructpix2pix** and **InstructDiffusion**, stating these may involve a similar process to the proposed method.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **Segformer Pretrained Weights Discrepancy**: `@shamik6766` brought up an issue regarding the mismatch of mIOU values of Segformer b4 or b5 models on cityscape or ADE20k datasets, between what is referenced in the paper and what's available in Hugging Face. The user asked for assistance in obtaining the correct pretrained weights. They also mentioned using `#nvidia`.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (8 messages🔥): 
        
- **Context Length of Models**: User `@ppros666` asked about the context length of **Llama 2**, particularly whether it's 4000 or 4096. `@Cubie | Tom` clarified that it's **4096**, and this information can be found in the [model configuration](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json#L12) on HuggingFace. Additionally, `@Cubie | Tom` shared that in Python, this value can be accessed via `model.config.max_position_embeddings`.

- **Fine-tuning Language Models**: `@ppros666` also expressed their intention to fine-tune a Language Model for the first time. They found a tutorial titled ["Llama-2 4bit fine-tune with dolly-15k on Colab"](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing), and asked if it is a good resource to follow given the fast-paced nature of the field.

- **Model Download Size and Time**: `@vikas8715` asked about the download size and time for **Mixtral-8x7B** model. @vipitis reported that the size is approximately **56GB**. `@nerdimo` advised that a powerful GPU with high VRAM is recommended for handling such large models.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Discussion on Training Data for Diffusion Models**: User `@guinan_16875` started a discussion on the training data used for existing diffusion models. They noted that current models use images of the same style, making the task essentially one of style transfer. The user proposed an alternative **paired data training method**, using an example of a photo of a child and the corresponding photo of the father in one training session. The aim is to use the trained model to generate a predicted photo of the father from the child's photo. 
- In response, `@asrielhan` mentioned **Instructpix2pix and InstructDiffusion** as possible models that can perform similar tasks.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Gemini-Pro Streaming Query**: `@fullstackeric` inquired about streaming capabilities in Gemini-Pro, receiving a reference to Google's conversational AI Nodejs tutorial, and explored the potential for compatibility with the Langchain library.
- *Community Interaction - User Surveys and Announcements*: `@jay_wooow` shared a developer-focused open-source survey via [tally.so link](https://tally.so/r/mO7q0p), while `@ofermend` announced an early-preview sign up for an Optical Character Recognition (OCR) feature in Vectara through a [google form](https://docs.google.com/forms/d/e/1FAIpQLSdGELyboIuytmLPqZNXwS5ur7gXTx28IWWONeqlOV-LSSxwaA/viewform).
- **Discussion on Handling JSON Input in Prompts**: `@infinityexists.` questioned the feasibility of incorporating JSON in PromptTemplate, which, as per `@seththunder`, can be implemented using a Pydantic Output Parser.
- **Experiences with AI Workflow Tools, Particularly Aiflows and LangChain**: `@tohrnii` elicited feedback from users on their experiences with [Aiflows](https://github.com/epfl-dlab/aiflows), comparing it to LangChain.
- **Technical issue Reports**: User `@ackeem_wm` reported a "422 Unprocessable Entity" error during usage of Langserve with Pydantic v1. Concurrently, `@arborealdaniel_81024` flagged a broken link in the Langchain release notes.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (42 messages🔥): 
        
- **Gemini-Pro Streaming Inquiry**: `@fullstackeric` asked if gemini-pro has streaming which is confirmed by `@holymode` with a code snippet for it in python and later in Javascript. Also, Google's conversational AI node.js tutorial is suggested for reference. However, `@fullstackeric` is looking for a solution through langchain library which doesn't appear to be supported just yet, according to `@holymode`.
- **Survey Notification**: User `@jay_wooow` shared a [tally.so link](https://tally.so/r/mO7q0p) encouraging the community to participate in an open-source survey to help understand developers' motivations and challenges while working with Lower Level Models (LLMs). The findings will also be published in an open-sourced format.
- **Prompt Accepting JSON Input Query**: `@infinityexists.` asked if we can pass a JSON in PromptTemplate, with `@seththunder` suggesting that this can be achieved using a Pydantic Output Parser.
- **AI Workflow Tools Discussion**: `@tohrnii` asked about community experiences with aiflows (a toolkit for collaborative AI workflows), comparing it to LangChain. The tool can be found in this [Github link](https://github.com/epfl-dlab/aiflows).
- **Vectara's OCR Feature Announcement**: `@ofermend` announced an optical character recognition (OCR) feature for Vectara that would allow users to extract text from images. Those interested in being part of the early preview were directed to sign up via a provided [google form](https://docs.google.com/forms/d/e/1FAIpQLSdGELyboIuytmLPqZNXwS5ur7gXTx28IWWONeqlOV-LSSxwaA/viewform).


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **POST /openai-functions-agent/invoke HTTP/1.1" 422 Unprocessable Entity Error**: User `@ackeem_wm` reported receiving a "422 Unprocessable Entity" error when using Langserve with Pydantic v1, even though the playground works fine with his requests.


### ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 
        
- **Broken Link in Release Notes**: User `@arborealdaniel_81024` reported a broken link in the release notes which was supposed to lead to a template in question. The URL specified was `https://github.com/langchain-ai/langchain/tree/master/templates/rag-chroma-dense-retrieval`. The user has flagged this for attention so it can be fixed.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Engaged discussion on **evaluation models and benchmarks** between users `@bjoernp` and `_jp1_`, exploring the *probability of rating accuracy* and potential methods of creating effective benchmarks, including using a test set of the eval model's training set.
- Conversations surrounding technical aspects of **Mixtral implementations** with users `@huunguyen`, `@tcapelle`, `@goldkoron`, `@dyngnosis`, `@bjoernp` and `@someone13574`. Topics ranged from *training of linear experts* and *router retraining* to execution of Mixtral on specific GPUs and *backpropagation techniques through a topk gate*.
- Extensive discussions on **Llama-cpp-python issues**, debugging strategies and other potential inference methods amongst `@rtyax`, `@bjoernp` and `@.calytrix`. With reported silent failure issues with Llama-cpp-python, alternative models like VLLM were suggested, leading to the idea of a unified API that includes Ooba. Additionally, different efficient methods for downloading models were shared by `@rtyax` and `@bjoernp`.

**DiscoResearch Channel Summaries**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (6 messages): 
        
- **Evaluation Models and Benchmarks Discussion**: User `@bjoernp` and `_jp1_` engaged in a discussion about the use of evaluation models and benchmarks. `@bjoernp` explained that the evaluation model is used to rate other models, demonstrating a positive correlation between actual benchmark score and the rating model score, indicating *correct rating with high probability*. On the other hand, `_jp1_` argued that the evaluation model's capability should be almost equivalent to measuring MMLU of the evaluation model itself.
- **Test Set as Benchmark**: `_jp1_` suggested that a test set of the eval model training set could be used as the eval model benchmark, while `@bjoernp` agreed that a held-out test set could indeed be an effective benchmark, especially if the dataset creator is trusted.
- **Invitation to Collaborate**: `_jp1_` extended an invitation for others to work on developing a better English model based on their existing German data eval model.


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (8 messages🔥): 
        
- **Expert Copies in Training**: User `@huunguyen` questioned whether the experts are copies of an original linear and then continued train, or if they are completely *new linears trained end2end*.
- **Running Mixtral on A100 with 40GB**: `@tcapelle` enquired if anyone managed to run **Mixtral** on an A100 GPU with 40GB of memory. In response, `@goldkoron` mentioned they run it on a combination of 3090 and 4060ti 16GB GPUs which theoretically equate to 40GB.
- **Router Retraining for New Experts**: `@dyngnosis` theorized that *the router portion might need to be retrained* to properly route tokens to the new expert.
- **Backpropagation Through Topk Gate**: `@bjoernp` expressed uncertainty about *how backpropagation works through the topk gate*, suggesting further study on router outputs could provide insight.
- **Training With or Without Topk**: `@someone13574` speculated whether during training topk is used, or whether a small amount of "leak" is allowed through topk.


### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (20 messages🔥): 
        
- **LLama-cpp-python issues**: `@rtyax` reported that Llama-cpp-python is failing silently during generate / chat_completion, even with the low level api example code. This behavior was observed after rebuilding llama-cpp-python for mixtral and it's not clear if the problem is localized. Despite these issues, Oobabooga was reported to work correctly with the same Llama-cpp-python wheel.
- **Debugging suggestions**: `@bjoernp` suggested debugging easier models first, like Mistral-7b. He also pointed out the possibility of tweaking thread settings in `fasteval/evaluation/constants.py` to potentially help with the issue.
- **Alternative models and inference methods**: `@bjoernp` offered VLLM as an alternative model since it supports `min_p` like Llama-cpp-python. However, `@rtyax` showed interest in continuing to explore Llama-cpp-python further. Both `@.calytrix` and `@bjoernp` advocated for a unified API approach, with `.calytrix` specifically mentioning the value of having Ooba as an inference engine for increased flexibility.
- **Model downloading techniques**: Both `@rtyax` and `@bjoernp` shared their preferred methods for downloading models. `@rtyax` uses `download_snapshot` from Hugging Face, while `@bjoernp` uses `huggingface-cli download` with the `HF_HUB_ENABLE_HF_TRANSFER=1` option for optimized download speeds. `@.calytrix` acknowledged the effectiveness of the latter, with recent improvements making it more reliable.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussions around **AI model hosting and pricing**; a comparison of **Anyscale's performance** across different servers was conducted by *'@dimfeld'* who reported inferior performance on Mistral 8x7b compared to Fireworks and DeepInfra.
- **Anyscale's performance enhancement** efforts were subsequently highlighted, showcasing their acknowledgment of these performance issues and subsequent work towards resolving them as shared by *'@coffeebean6887'* through a [Twitter post](https://twitter.com/robertnishihara/status/1735529685290000396).
- An update depicting **Anyscale's improvements in benchmarking**, cited a 4-5x lower end-to-end latency achievement.
- A lesser known aspect of the AI industry surfaced; the intense **price competition in AI model hosting/serving**.
- The resourcefulness of the community came to light with *'@guardiang'* sharing a [link](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs) to the **OpenAI API documentation** for the benefit of guild members.
- A brief mention of the term '**Qtransformers**' by *'@swyxio'* in the #llm-paper-club channel without any accompanying context or discussion.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (13 messages🔥): 
        
- **Performance of Anyscale on Mistral 8x7b**: `@dimfeld` reported that **Anyscale** was significantly slower when tried on Mistral 8x7b compared to Fireworks and DeepInfra.
- **Performance Enhancement Efforts by Anyscale**: In response to the above, `@coffeebean6887` shared a [Twitter post](https://twitter.com/robertnishihara/status/1735529685290000396) indicating that Anyscale is aware of the performance issue and they are actively working on improvements.
- **Benchmarking Improvements**: In a follow-up, `@coffeebean6887` noted that a PR indicates that Anyscale has achieved 4-5x lower end-to-end latency in benchmarks.
- **Price Competition in Hosting/Serving**: `@coffeebean6887` and `@dimfeld` discussed the noticeable price competition around hosting and service providing for AI models.
- **Link to OpenAI API Doc**: `@guardiang` posted a [link](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs) to the new Twitter/OpenAIDevs account on the OpenAI platform.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages): 
        
swyxio: Qtransformers


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Discussion regarding the usage and performance of the **GPT-4 Turbo API** in production environments. User `@evan_04487` sought clarification on operating the API despite its preview status, and also inquired about any initial intermittent issues which `@res6969` assured were non-existent in their experience.
- Inquiry on **resources for MLops** about scaling synthetic tabular data generation, particularly *books and blogs* by user `@ayenem`.
- `@thebaghdaddy` queried on the effectiveness of **MedPalm2** for kNN purposes compared to their currently used model and **GPT4**, and planned to run comparative trials soon.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (3 messages): 
        
- **Using GPT-4 Turbo API in Production**: User `@evan_04487` raised a question about the utilization of the **GPT-4 Turbo API** in production environments despite its preview status. It was mentioned that while the limit for transactions per minute (TPM) was raised to **600k**, doubling the capacity of GPT-4, and requests per minute (RPM) stayed the same as GPT-4's at **10,000**, it still remained technically a preview.
- `@res6969` responded positively, stating that they actively used it, and it performed well. `@evan_04487` further questioned about the initial intermittent issues with the system, and whether those were resolved.


### ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/) (1 messages): 
        
- **MLops Resources for Scaling Synthetic Tabular Data Generation**: User `@ayenem` asked if anyone knew of **books or blogs on the subject of MLops** specifically discussing scaling synthetic tabular data generation for production.


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/) (1 messages): 
        
- **Usage and Comparison of Models**: `@thebaghdaddy` is currently using an unspecified model and finds it satisfactory. They haven't used **MedPalm2** so they are unable to compare the two. They are also pondering whether **MedPalm2** could provide **CoT** more effectively for kNN purposes than **GPT4**, and plan to test this soon.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **LLM Developers Survey:** `@jay_wooow` is conducting an [open-source survey](https://tally.so/r/mO7q0p) aimed at understanding the motivations, challenges, and tool preferences of those building with LLMs (Language Models). He wants to gather data that could help the wider developer community. The survey results, including the raw data, will be published when the target number of participants is met. The objective is to influence product development and to provide useful data for other AI developers.
        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **DC Optimization for AI workloads**: A user discussed that most of the existing **Data Centers (DCs)** are not well optimized for **AI workloads** as the economical factors differ.
- **Inference at Edge and Training**: The user found the concept of performing **inference at the edge** and **training where power is cheaper** to be an interesting approach.
        

---
The Alignment Lab AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The MLOps @Chipro Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The AI Engineer Foundation Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
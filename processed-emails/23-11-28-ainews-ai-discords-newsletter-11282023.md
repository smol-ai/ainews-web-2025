---
id: f022eda6-77d2-40a2-ae32-41a9fa6fa3fe
title: AI Discords Newsletter  11/28/2023
date: '2023-11-28T22:14:48.819294Z'
original_slug: ainews-ai-discords-newsletter-11282023-6795
description: >-
  Discussion on AI context management, mathematical modeling, upcoming AI
  engineering book from OReilly, paper discussions on prompt improvements,
  retrieval-augmented generation, lookahead decoding, and LangChain templates.
  Also covers OpenAI GPT-4 performance issues, user experiences, and interest in
  GPT-5.
companies:
  - openai
  - oreilly
models:
  - gpt-4
topics:
  - context-management
  - multi-agent-systems
  - ai-engineering
  - prompt-improvement
  - retrieval-augmented-generation
  - rag
  - lookahead-decoding
  - langchain
  - gpt-4
  - gpt-5
---


<!-- buttondown-editor-mode: plaintext -->
## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion about seeking **recommended references** but without any specified context.
- User `@swyxio` hinted at something referred to as "**CodeForward**" without any further context.
- Exploration for resources on the topic of **context management in AI**, particularly with a focus on mathematical modelling or theory in multi-agent shared memory systems, as introduced by `@slono`.
- Anticipation for an upcoming **AI Engineering book from OReilly**, suggested as a potentially useful resource by `@swyxio` and `@henriqueln7`.
- In the paper club, `@eugeneyan` requested `@yikesawjeez` to select this week's papers for pre-reading. The proposed topics were papers on prompt improvements ([arxiv.org/abs/2311.09277](https://arxiv.org/abs/2311.09277)) and ([arxiv.org/abs/2311.05997](https://arxiv.org/abs/2311.05997)), which later changed to a RAG (Retrieval-Augmented Generation) context management paper ([arxiv.org/abs/2311.09210](https://arxiv.org/abs/2311.09210)) after a reminder about needing **long-context methods**.
- Expressions of interest in discussing **Lookahead Decoding** in the future by `@slono` and a positive response to the current topic selection by `@swizec`.
- `@semra5446` suggested an additional future paper for discussion: [arxiv.org/pdf/2311.01906.pdf](https://arxiv.org/pdf/2311.01906.pdf).
- Creation of a LangChain template by Harrison was mentioned for the paper **Skeleton-of-Thought**, shared by `@yikesawjeez`.

**Latent Space Channel Summaries**

### ▷ Channel: [ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876) (8 messages): 

- **Recommended References**: User `@coffeebean6887` initiated a discussion about recommended references, although no specific context was given.
- **CodeForward**: User `@swyxio` mentioned something called "CodeForward", but didn't specify what it is or its relevance.
- **Context Management in AI**: User `@slono` is looking for resources related to the mathematical modelling or theory behind context management in AI, specifically in multi-agent shared memory systems.
- **OReilly's AI Engineering Book**: Both `@swyxio` and `@henriqueln7` hinted at an upcoming book on AI Engineering from OReilly. Their excitement may be due to expected high-quality content or relevance to the discussion topics.


### ▷ Channel: [llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663) (11 messages): 

- **Paper Selection for Discussion**:  `@eugeneyan` asked `@yikesawjeez` for the paper selection for the week to allow members time to pre-read. `@yikesawjeez` initially proposed papers on prompt improvements ([arxiv.org/abs/2311.09277](https://arxiv.org/abs/2311.09277)) or ([arxiv.org/abs/2311.05997](https://arxiv.org/abs/2311.05997)). After `@eugeneyan`'s reminder about studying long-context methods, `@yikesawjeez` proposed a RAG (Retrieval-Augmented Generation) context management paper instead ([arxiv.org/abs/2311.09210](https://arxiv.org/abs/2311.09210)).
- **Interest in Future Discussion Topics**: `@slono` expressed interest in discussing Lookahead Decoding ([lmsys.org/blog/2023-11-21-lookahead-decoding/](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)) in the future. `@swizec` expressed enthusiasm for the topic selection in the paper club this week.
- **Additional Paper Suggestion**: `@semra5446` suggested another interesting future paper for discussion: [arxiv.org/pdf/2311.01906.pdf](https://arxiv.org/pdf/2311.01906.pdf).
- **LangChain Template**:  `@yikesawjeez` also mentioned a LangChain template made by Harrison for the paper **Skeleton-of-Thought**.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Users across channels reported instances of performance issues, erratic behaviors, and possible limitation of **GPT-4**, triggering speculation about its functionality, early stage glitches, and server capacities. User `ahlatt` shared a specific instance with GPT-4 prematurely stopping a `TradingEnv` Python class generation.
- **AI and Language** bias and limitations discussion initiated by user `posina.venkata.rayudu`, referencing a *Nature Human Behaviour* article. `.dooz` emphasized that language limits in LLMs arise from technical aspects rather than a plan to enforce hegemony.
- **Use of ChatGPT** for varied applications, from scenario modeling and exercise design to producing content for Dungeons and Dragons games and academic papers.
- Several users experienced problems with **OpenAI's service login and performance**, leading to speculations ranging from high server load to scheduled maintenance and updates.
- Several users were found expressing **interest in GPT-4 and GPT-5**, sparking light-hearted speculations about a possible GPT-5 during server downtimes.
- **GPT-4 Store delay** was discussed by various users including `.xiaoayi`, `dkracingfan`, `kyleschullerdev_51255`, and `lodosd` with concerns about the ambiguous timeline of the launch.
- **Possibility of Cooperation** between OpenAI's large models and Boston Dynamics' robots was speculated by `@davidvon`, with extended discussions on the level of consciousness and self-awareness in AI and AGI, specifically GPT models.
- Users highlighted **issues with OpenAI services**, such as issues with image generation tools, the inability to find plugin options, lost conversation history, and specific DALL.E experiments with errors. 
- Discussions revolved around **effective prompt engineering** and plugin usage, with instances such as user `komal0887` inquiring how to produce non-evaluative sentences using the "gpt-3.5-turbo-instruct" model and `leoalvarenga` seeking plugin recommendations.
- `NastyTim` expressed significant dissatisfaction with GPT-4's service and customer support, with other users suggesting alternative options and advising contacting OpenAI's customer support.
- @xsrusapa and @solbus held a brief discussion regarding advice on integrating **Slack with custom GPTs** in order to ask questions and receive answers back via Slack.

**OpenAI Channel Summaries**

### ▷ Channel: [ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273) (101 messages): 

- **GPT-4 Performance Discussion**: Users `@3hx`, `@cr7462`, `@xyza1594`, `@JohnPringle`, and `@dogdroid` expressed concerns about apparent performance issues and possible "nerfing" of GPT-4 generation. They discussed instances where the model required prompt refinement, stopped code generation prematurely, and other erratic behaviors. `@ahlatt` shared a specific example where GPT-4 prematurely stopped generating a `TradingEnv` Python class.
- **AI and Language Discussion**: User `@posina.venkata.rayudu` brought up a recent article from Nature Human Behaviour about AI and language problems, indicating Chatgpt handles approximately 100 languages but expressing frustration over article's publishing tactics. Another user, `.dooz`, responded that language bias in LLMs is due to technical limitations rather than a secret plot to hegemonize.
- **Discussion on Maxi's Introduction**: User `@maxiisbored` introduced themselves and interacted with `@posina.venkata.rayudu` and `@jclove_`.
- **Customer Dissatisfaction**: `@NastyTim` expressed significant dissatisfaction with the GPT-4 service and frustrations about perceived lack of customer support from OpenAI. Responders `.dooz`, `@kesku`, and `@satanhashtag` recommended other options (`GPT-4 API`, `BingChat`, prompt engineering techniques), clarified the beta status of the product, and suggested to contact `support@openai.com`.
- **AI and AI Data Analyst Products**: User `@tonyfancher` inquired about AI data analyst products. User `@fran9000` shared a link to Sean Carroll's podcast discussing the current state of LLMs and their future. `@jaimd` reported that Bing chat appeared to be filtering conversations about intelligent AIs.


### ▷ Channel: [openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304) (459 messages🔥): 

- **Use of ChatGPT for Diverse Applications**: Users discussed the diverse ways they use ChatGPT, from @aloy2867 looking for therm scenarios and exercises, to @satanhashtag using it for their DnD game, to @xyza1594 suggesting it could be used as a tool for writing papers.
- **ChatGPT performance and login issues**: Beginning with @cjdaigle2 experiencing login issues, many users reported ChatGPT being slow, down, or throwing errors. Some speculated that the problem was due to high server load or momentary downtime for maintenance and updates. 
- **Issues with data and features**: Users like @winksaville, @pruo, and @thefoodmaster raised issues, such as problems with image generation tools, ChatGPT skipping some answers, the inability to find plugin options, and lost conversation history. 
- **Interest and speculation about upgraded versions**: Users like @m_12091 and @chadgpt showed interest in upgraded versions like GPT-4 and GPT-5. Some users, like @chotes and @deltamza, humorously speculated about the possible release of GPT-5 during the downtime. 
- **Warning about spamming and content misuse**: @kesku warned users about misrepresenting AI generated output as human generated in line with OpenAI's terms of service. Users @zinthaniel and @aryann warned about the possible temporary unavailability of some features and services due to high demand and the inadvisability of abusing refresh functions.


### ▷ Channel: [openai-questions](https://discord.com/channels/974519864045756446/974519864045756454) (206 messages🔥): 

- **Issue with Subscription and Plan Access**: User `@alexgogan` mentioned about losing his Plus membership and being moved automatically to the free plan. 
- **Concerns About GPT Role Performing**: User `@crerigan` had a question on whether creating multiple GPTs with single jobs is better than having a single GPT to do related jobs. `@solbus` suggested trying both approaches and observing the outcomes.
- **Issues with Creating OpenAI Accounts**: User `@phamngocanh9x` reported inability to create an OpenAI account from Vietnam. 
- **Web Scraping Suggestions Requested**: User `@vipraz` asked for ways to scrape the web for articles, and included other related details such as article title, description and author's information.
- **Integrated Custom GPTs Inquiry**: User `@xyrusapa` asked for advice on using Slack to ask questions to a custom GPT and receive answers back into Slack.
- **Difficulty with Model**: User `@dotails` struggled with an issue `!Unterminated string in JSON at position 548 (line 1 column 549)`.


### ▷ Channel: [gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244) (83 messages): 

- **Training GPTs**: Users such as `@haribala`, `@DawidM`, `@pietman`, and `@jaynammodi` discussed the limitations of training GPTs. It appears that the models cannot be "trained" in a traditional sense - rather, they rely on connecting to an external API, feeding instructions, and a knowledge base of files to customize responses.
  
- **Launch of OpenAI GPT Store**: Users `@.xiaoayi`, `@dkracingfan`, `@kyleschullerdev_51255`, and `@lodosd` expressed concerns about the delayed launch of the official OpenAI GPT store. As of the end of November, no official news or updates relating to the launch timeline had been released.

- **API Integration and Enterprise Workflow**: `@phild66740`, `@solbus`, and `@kyleschullerdev_51255` discussed possibilities to integrate GPTs into an enterprise workflow, particularly re-purposing custom GPTs for API use. 

- **Chat Functionality Issues**: Towards the end of the discussion, users like `@halalarax`, `@rangerslayer97`, `@hipo_.`, and others reported temporary outages and issues with ChatGPT including unexpected responses.

- **Integrating plugins with Assistant API**: Lastly, user `@zaprime.` inquired about using plugins with the assistant API, but no response was given at the time of transcript cutoff.


### ▷ Channel: [prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970) (30 messages): 

- **Plugins Recommendations**: There is a request by `@leoalvarenga` asking for plugin recommendations, but no recommendations were provided.
  
- **Prompt Engineering**: `@komal0887` asked for assistance in writing a prompt that would generate articles without evaluative sentences using **gpt-3.5-turbo-instruct** model.
  
- **Discussion on OpenAI and Robotics**: A conversation initiated by `@davidvon` speculated on the possibility of OpenAI's large models cooperating with robots, potentially giving them physical bodies.
  
- **Debate on AGI and Self-Awareness**: There was a heated discussion led primarily by `@jaynammodi`, `@.pythagoras`, and `@syndicate47` debating the extent of consciousness and self-awareness in AI and AGI, specifically GPT models.
  
- **Issues with DALL.E**: `@kh.drogon` lamented difficulties in getting DALL.E to correctly spell words on an image.
  
- **Extracting Information from Transcripts**: `@greenysmac` asked for guidance on refining a prompt to correctly identify the speaker count from a transcript. Suggestions from `@notsoluckyducky` and `@solbus` proposed the use of advanced data analysis.


### ▷ Channel: [api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970) (30 messages): 

- **Plugin Recommendations**: `@leoalvarenga` asked for plugin recommendations. No further details or responses were given. 
- **Generating Non-Evaluative Articles**: `@komal0887` sought help with writing a prompt to generate an article without evaluative sentences using the "gpt-3.5-turbo-instruct" model.
- **Cooperation with Boston Dynamics' Robots**: In a series of comments, `@davidvon` speculated on the potential cooperation between OpenAI's large models and Boston Dynamics' robots to allow AI to have a physical body.
- **DALL.E's Spelling Issues**: `@kh.drogon` shared an issue with DALL.E misspelling words when creating logos. The user sought advice, but no responses were present.
- **OpenAI Model's Consciousness Debate**: A debate occurred between `@davidvon`, `@jaynammodi` and `@.pythagoras` regarding whether OpenAI's large models have free will and self-awareness, revolving around a GPT model-operated robot. `@eskcanta` and `@syndicate47` offered critical perspectives on these assertions.
- **Counting Speakers in a Transcript**: `@greenysmac` asked for help in refining a prompt that counts speakers in a transcript. `@notsoluckyducky` recommended using advanced data analysis to parse the names, remove duplicates, and return the list length. An additional validation was provided by `@solbus`.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Discussions about using and configuring **LangChain** for various functionalities and fine-tuning procedures:
    * Concern about the user interface of LangChain agents shared by `@moooot`, showing interest in more intuitive UIs for displaying information.
    * Request for instructions on connecting a LangChain agent to Zapier, posed by `@jonanz`.
    * `@steve675` questioned the possibility of creating a Q&A assistant for tabular data in LangChain.
    * Technical issues faced by `@menny9762` with passing a prompt into a ConversationalRetrievalQAChain and setting the `{lang}` variable, with `@seththunder` providing input.
    * `@l0st__` sought assistance on fine-tuning a LangChain agent that access SQL databases, and discussions about "stuck" loops and delayed responses.
    * `@sampson7786` asked how to request an invite code for LangSmith, with suggestions given by `@seththunder`.
- Tutorials, resources, and work sharing related to LangChain:
    * A new tutorial shared by `@kulaone` about deploying LangChain on Cloud Functions using Vertex AI models, available on [YouTube](https://www.youtube.com/watch?v=q_vrmT8MyEs).
    * `@joshuasundance` promoting the latest version of BERTopic that now supports LCEL runnables, with link provided to [GitHub Release](https://github.com/MaartenGr/BERTopic/releases/tag/v0.16.0).
    * `@andysingal` discussing the role of LangChain+LlamaIndex in extracting tables and images using Multimodal LLM, citing a [Medium article](https://medium.com/ai-advances/utilizing-multimodal-llm-for-extracting-tables-and-images-f9fd86e3d002) as reference.
    * Additional request from `@l0st__` for guidance on fine-tuning a LangChain agent, providing an example conversation flow.

**LangChain AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1038097195422978059/1038097196224086148) (52 messages): 

- **Request for Langsmith Invite code**: @sampson7786 asked about how to request for an invite code for Langsmith. `@seththunder` suggested that it's done from the LangChain website.
- **Langchain Conversational Retrieval QA Chain**: @menny9762 discussed issues with passing a prompt into a `ConversationalRetrievalQAChain` and figuring out how to set the variable `{lang}`. `@seththunder` suggested that anything in `{}` needs to be set as input variables.
- **SQLDatabase Agent**: @l0st__ shared the need for assistance on fine-tuning a LangChain agent that has access to a SQL Database to act as a sales assistant, mentioning about issues of getting "stuck" in loops and delayed responses. `@gloria.mart.lu` recommended SQLDatabase agent & toolkit for usage.
- **User Interface for LangChain**: @moooot expressed concern on why the user interface for agents is essentially a chatbot and inquired about more intuitive UIs for displaying the information fetched by agents.
- **Integration of LangChain with Zapier**: @jonanz asked for guidance on how to connect a LangChain agent to Zapier, referencing the deprecated link on the LangChain website. A solution was not provided in the chat.


### ▷ Channel: [langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282) (2 messages): 

- **Creating Q&A Assistant for Tabular Data**: User `@steve675` inquires if **LangChain** allows for creating a Q&A assistant specifically tailored to tabular data. His need is for a chatbot which can interact with multiple tables, given that each table may have varying columns.


### ▷ Channel: [share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729) (5 messages): 

- **BERTopic Runnable Support**: `@joshuasundance` shared the [latest version of BERTopic](https://github.com/MaartenGr/BERTopic/releases/tag/v0.16.0) which now supports LCEL runnables, highlighting its usefulness and expressing excitement for its potential applications. 
- **LangChain Agent Fine-Tuning Request**: `@l0st__` sought assistance for fine-tuning a LangChain agent that can act as a sales assistant and access SQL databases, providing a detailed use-case as an example.
- **Multimodal LLM for Data Extraction**: `@andysingal` shared a [Medium article](https://medium.com/ai-advances/utilizing-multimodal-llm-for-extracting-tables-and-images-f9fd86e3d002) discussing the role of LangChain+LlamaIndex in extracting tables and images using Multimodal LLM.


### ▷ Channel: [tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538) (2 messages): 

- **LangChain Deployment Tutorial**: User `@kulaone` shared a new tutorial on **how to deploy Langchain on Cloud Functions using Vertex AI models for scalability**. The tutorial can be watched on [YouTube](https://www.youtube.com/watch?v=q_vrmT8MyEs).
- **Fine-Tuning LangChain Agent**: User `@l0st__` expressed a need for assistance in **fine tuning a LangChain agent** that has access to a SQL Database and would act as a sales assistant. He provided an example conversation flow for clarification.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Discussion surrounding the upload and processing of the [MJ_dataset](https://huggingface.co/datasets/tsunemoto/MJ_dataset/tree/main) on Hugging Face by `@tsunemoto` who ran into upload speed limitations. This led to a conversation on rate limiting and possible ways around it with the new Hugging Face account.
- Random Twitter and YouTube links shared by `@.andrewnc` and `@pradeep1148` without any contextual details.
- Conversations on *tinygrad* development and the [YouTube stream](https://www.youtube.com/watch?v=2QO3vzwHXhg&ab_channel=georgehotzarchive) of George Hotz building upon **openhermes**, shared by `@yorth_night`.
- Introduction of the new compute provider, [Voltage Park](https://voltagepark.com/), by `@coffeebean6887`, with its connections to **Imbue**, **Character**, **Atomic**, and an [unannounced $500m funding round](https://www.crunchbase.com/organization/voltage-park).
- Variety of Twitter posts, and research links shared by `@papr_airplane`, `@if_a`, and `@oozyi`.
- Extensive conversations on performance and fine-tuning of chatbot models in the context of **FFT, QLORA, DeepSpeed, Axolotl, FSDP** and the difficulties of training large models. 
- User experiences of **GPT-4** highlighting its lackluster performance and the need for more user input.
- Updates on recent model tunes and testing, specifically the "koishi-120b-qlora" by `@ludis___`. 
- Technical challenge and subsequent discussion on running inference for **StyleTTS2** shared by `@nemoia` and `@lhl`.
- A focus on the limitations of model benchmarking and potential manipulation of results.
- Discussions on building high-PC specs for LLMs with focuses on CPU, RAM, and PCIe lane considerations.
- Queries on open-source models optimized for planning and function call abilities, however, no specific model was recommended. 
- Comparisons of sentence similarities recommended by `@philpax` using sentence embedding and cosine similarity operation, with a [link](https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities) shared to Sentence-BERT's documentation.
- In-depth discussion on the performance of various models on different GPUs and approaches for pruning and quanting including **ExlLlama** and **GGUF**, with an emphasis on ExlLlama working better with 2 GPUs for large models.
- Shared link to a quanted model of Hermes 2 with exllama-2: [https://huggingface.co/bartowski/OpenHermes-2.5-Mistral-7B-exl2](https://huggingface.co/bartowski/OpenHermes-2.5-Mistral-7B-exl2)

**Nous Research AI Channel Summaries**

### ▷ Channel: [off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928) (7 messages): 

- **Sharing of MJ_dataset**: `@tsunemoto` shared a link to his [MJ_dataset](https://huggingface.co/datasets/tsunemoto/MJ_dataset/tree/main) on Hugging Face, which contains a million rows from a midjourney dataset. It took him approximately 3 days to process and upload due to restrictions on upload speeds.
- **Discussion on upload speed limitations**: `@tsunemoto` and `@crainmaker` discussed the issues of rate limiting on Hugging Face. `@tsunemoto` mentioned that despite trying to use the Rust-based library for transferring data to Hugging Face by setting the hf transfer environment variable, he still experienced limitations possible due to his new Hugging Face account.
- **Links shared**: 
    - `@.andrewnc` shared a [Twitter link](https://twitter.com/var_epsilon/status/1729303212346294411?t=RKkiAExOfEMmOPBhxS1xPA&s=19) without any additional context.
    - `@pradeep1148` shared a [YouTube link](https://www.youtube.com/watch?v=oqMWrDjbkFI) also without any additional context.


### ▷ Channel: [interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192) (10 messages): 

- **Tinygrad Progress**: `@yorth_night` commented that **tinygrad is looking more and more sexy** with `@vatsadev` agreeing, **getting better every day**.
- **George Hotz Stream**: `@yorth_night` shared a [YouTube link](https://www.youtube.com/watch?v=2QO3vzwHXhg&ab_channel=georgehotzarchive) of George Hotz working with **openhermes** as a base for "**Q***".
- **New Compute Provider Available**: `@coffeebean6887` introduced a new compute provider, [Voltage Park](https://voltagepark.com/), stating it received a large part of its funding from **Imbue** and also mentioning contracts with **Character** and **Atomic**. Additionally, Voltage Park is reported to provide Burst and On-Demand **h100s**. 
- **Voltage Park Funding**: `@coffeebean6887` also shared that the new compute provider, Voltage Park, had an [unannounced $500m funding round](https://www.crunchbase.com/organization/voltage-park). 
- **Shared Research and Social Media Links**: Links were shared by `@papr_airplane` and `@if_a` ([Twitter posts](https://twitter.com/billyuchenlin/status/1729194671178760551?t=biqmZkaJsPVGNQ62gEMKQw&s=19), [another Twitter post](https://twitter-og.com/ldjconfirmed/status/1729264801224782057?t=kxjsKiqLObpDxWmdZPzrWQ&s=19)) and `@oozyi` ([Research paper](https://arxiv.org/abs/2311.16079)).


### ▷ Channel: [general](https://discord.com/channels/1053877538025386074/1149866623109439599) (232 messages🔥): 

- **Performance and Finetuning of Chatbot Models**: Users explored model performance and finetuning methods, with discussions revolving around **FFT, QLORA, DeepSpeed, Axolotl, FSDP** and regarding difficulties of training large models and running out of GPU memory. `@teknium` shared one of his wandb runs for a QLORA over Hermes 2 experiment with the Trismegistus dataset [[Link]](https://wandb.ai/teknium1/occult-expert-mistral-7b/runs/coccult-expert-mistral-6). 

- **User Opinions on GPT-4**: The community expressed some disappointment with GPT-4's behaviour, particularly noting it leaves more work for the user and appears to be providing shorter, incomplete generation results.

- **Model Recommending and Testing**: `@ludis___` announced completion of a 120B tune on instruct data named "koishi-120b-qlora", and the community showed interest in model Deepseek, particularly the 7B version. `@vatsadev` remarked on OpenHermes becoming a strong brand in local models, recognized by half of Twitter as the go-to replacement for ChatGPT. 

- **StyleTTS2 Troubleshooting**: `@nemoia` sought advice on running inference for StyleTTS2. `@lhl` shared his notes on StyleTTS2 setup for reference [[Link]](https://llm-tracker.info/books/howto-guides/page/styletts-2-setup-guide).

- **Model Benchmarks**: The community discussed the use and potential shortfalls of benchmarking as a measure of model performance, raising issues such as overfitting to benchmarks and manipulation of benchmark results.


### ▷ Channel: [ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927) (66 messages): 

- **PC Specs for LLMs**: `@orabazes` initiated a discussion about building a high spec PC for LLMs and asked whether using a Threadripper CPU is necessary. Various users weighed in, with `@night_w0lf` suggesting that any latest generation CPU should suffice and `@.andrewnc` recommending speccing out a Lambda Labs workstation. The importance of having enough system RAM and the right number of cores were also emphasized.
  
- **LLM Planning and Function Calling**: `@htahska` asked for recommendations on open source models that are optimized for planning and function calling abilities. No model was specifically recommended in the available excerpts. 

- **Graphics Cards and PCIe Lanes**: `@philpax` provided key information about the limitations of x670 having only 24 PCIe lanes and thus impacting the number of GPUs one can fit.

- **Autonomous Browser-Based LLM Project**: `@htahska` asked for suggestions about LLMs, focusing on their planning and function calling abilities. No specific suggestions were made in the responses.

- **Comparing Sentence Similarities**: `@philpax` recommended `@asada.shinon`, who wanted to calculate the distance between sentences, to use sentence embedding and perform a cosine similarity operation. A link to Sentence-BERT's documentation was shared: [https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities](https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities).

- **GPU Performance and Quanting Tools**: There was an extensive discussion about the performance of various models on different GPUs, notably with `@teknium` and `@coffeebean6887`. Different approaches for pruning and quanting, including ExlLlama and GGUF, were discussed; it was noted that ExlLlama was working better with 2 GPUs for large models. 

- **Link Shared**: Font to a model of Hermes 2 quanted with exllama-2: [https://huggingface.co/bartowski/OpenHermes-2.5-Mistral-7B-exl2](https://huggingface.co/bartowski/OpenHermes-2.5-Mistral-7B-exl2)
  
NB: LLMs = Large Language Models, ExlLlama and GGUF = Quanting tools, GPUs = Graphics Processing Units, PCIe = Peripheral Component Interconnect express.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- A message from user `@bryhen22` on #looking-for-collabs seeking collaboration without providing further context.
- User `@magusartstudios` and `@ldj` discussed a specific [YouTube video](https://www.youtube.com/watch?v=rMBLZtPmlsQ) shared in #general-chat. The content of the video was described as being **cursed**.
- The same [YouTube video link](https://www.youtube.com/watch?v=rMBLZtPmlsQ) was shared by user `@magusartstudios` in #fasteval-dev without additional commentary or context.

**Alignment Lab AI Channel Summaries**

### ▷ Channel: [looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261) (1 messages): 

- User `@bryhen22` tried to send an unidentified recipient a message. Further context or detail is not provided in the message.


### ▷ Channel: [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) (2 messages): 

- User `@magusartstudios` shared a [YouTube link](https://www.youtube.com/watch?v=rMBLZtPmlsQ).
- User `@ldj` described the content as **cursed**.


### ▷ Channel: [fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363) (1 messages): 

- User `@magusartstudios` shared a YouTube link: [https://www.youtube.com/watch?v=rMBLZtPmlsQ](https://www.youtube.com/watch?v=rMBLZtPmlsQ)


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

**Skunkworks AI Channel Summaries**

### ▷ Channel: [off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179) (1 messages): 

- `@pradeep1148` shared a [YouTube video link](https://www.youtube.com/watch?v=oqMWrDjbkFI).
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Active discussions on **GPT-4's code generation capability and inference speeds**. Concerns include incomplete function implementations and deteriorating speeds (particularly during peak hours), observed by `@potrock`, `@pantsforbirds` and `@res6969`. A hypothesis by `@evanwechsler` and `@nosa_.` posits potential cost and latency optimizations compromising model response. API performance above the cursor still holds as per `@res6969`. There was shared agreement on the increased necessity for **prompt engineering** to maximize performance, with `@thebaghdaddy` suggesting a [resource](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) by Lilian Weng on the topic.

- In the **Open Source Models** section, `@thisisnotawill` sought recommendations for storytelling models, with `@pantsforbirds` suggesting **Mistral** or its variants, also sharing their success with a custom retrieval model for their RAG pipeline. Links to fine-tuned versions of Mistral 7B like [Dolphin-2.1](https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF) and new models like [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) were shared, along with a [Twitter thread](https://twitter.com/bindureddy/status/1729253715549602071) offering additional context.

- Tangential observations on the **credibility of AI rumors** and **image generation patterns** under scope by `@pantsforbirds` and `@justahvee`. Notably, the 'more fantasy' specification leading to a psychedelic space vibe in images was sought to be analyzed.

- Users `@thisisnotawill`, `@justahvee`, and `@pantsforbirds` were discussing multiple aspects of an **AI Dungeon Master** in the collaboration channel. Challenges identified include the difficulty of maintaining a cohesive storyline over hundreds of messages and balancing player freedom with plot coherence. Simplifying the problem-solving approach and adjusting current methods were proposed as potential solutions.

- `@jeffreyw128` discussed the role of traditional search providers and the significance of vector and keyword search for speed in their project "metaphor".

- Creation of a **prompt channel** was proposed by `@pantsforbirds` in the feedback-meta channel.

- Detailed discussion on the **cost structure** of running a PDF-to-RAG/LLM pipeline was led by `@res6969`, who revealed splits between Azure OCR and GPT-4 and a drift towards a growing share of OCR costs. Transition from AWS Textract to Azure OCR for table extraction capability was also shared. `@pantsforbirds` revealed challenges with Nougat with a switch to AWS Textract mulled, further sharing a [link](https://aws.amazon.com/blogs/machine-learning/amazon-textracts-new-layout-feature-introduces-efficiencies-in-general-purpose-and-generative-ai-document-processing-tasks/) discussing Amazon Textract's new layout feature.

- A user inquired about others located in Vermont.

- Issues about OpenAI's speed and inability to login were raised by `@robhaisfield` and `@iyevenko`. Desire for a **70b model from Mistral** was expressed to reduce dependency on OpenAI. An incident disrupting services at OpenAI was shared by `@pantsforbirds`, linking to the [overall status](https://status.openai.com/) and the [specific incident](https://status.openai.com/incidents/q58417g6n5r7).

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ Channel: [gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641) (28 messages): 

- **Code Quality in GPT-4**: `@potrock` raised concerns about code generation quality in **GPT-4**, stating it has become "hilariously bad" and often leaves placeholder comments or refuses to implement functions completely. This was echoed by `@pantsforbirds` who shared the frustration of GPT-4 only completing part tasks.

- **Inference Speed Issues**: Both `@pantsforbirds` and `@res6969` noted worsening inference speeds, particularly during peak business hours. 

- **Possible Optimization for Cost & Latency**: Users like `@evanwechsler` and `@nosa_.` speculated that OpenAI may be optimizing GPT-4 for cost and latency, possibly compromising on response length or quality.

- **Consistent API Performance**: According to `@res6969`, API performance, particularly for applications above the cursor, has so far remained unaffected.

- **Prompt Engineering**: `@pantsforbirds` and `@thebaghdaddy` agreed that with the latest models, fine-tuning the prompt has more of an impact on performance than typical engineering. `@thebaghdaddy` shared a link to a [post on prompt engineering by Lilian Weng](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) for further reading.


### ▷ Channel: [opensource](https://discord.com/channels/1168579740391710851/1168606773595349082) (7 messages): 

- **Open Source Models for Creative Writing/Storytelling**: `@thisisnotawill` inquired about open-source model recommendations for creative writing and storytelling. `@pantsforbirds` suggested **Mistral** or its fine-tuned variants, specifically citing their success with a custom retrieval model for their RAG pipeline using Mistral 7b, fine-tuned on **SageMaker**.
- **Fine-tuning Models**: `@pantsforbirds` shared their experience of fine-tuning **Mistral 7B** themselves using SageMaker and suggested that there are open-source tunes of Mistral 7b available which can serve as a good starting point for broader use-cases.
- **Resource Links**: `@pantsforbirds` shared a link to a fine-tuned version of **Mistral 7B**, namely [Dolphin-2.1](https://huggingface.co/TheBloke/dolphin-2.1-mistral-7B-GGUF), on Hugging Face, stating that it performed well in their experience.
- **Additional Model Recommendation**: `@pantsforbirds` also mentioned a new model, [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha), on Hugging Face. They also shared a [Twitter thread](https://twitter.com/bindureddy/status/1729253715549602071), providing additional context, but clarified that they had not personally tested the performance of this model.


### ▷ Channel: [offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594) (4 messages): 

- **Credibility of AI Rumors**: `@pantsforbirds` and `@justahvee` discussed the validity of some rumors about AI. While `@pantsforbirds` declared an overall skepticism, `@justahvee` found an email allegation more plausible than a 'random post', but still questioned the feasibility of skipping towards a high-level test like exploiting AES. 
- **Image Generation Patterns**: `@pantsforbirds` brought up a pattern noticed in image generation where continual requests for enhancing a certain aspect (e.g., 'more fantasy') leads to a result resembling a psychedelic space image, and was seeking input on the phenomenon.


### ▷ Channel: [collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018) (11 messages): 

- **AI Dungeon Master Building Challenges**: User `@thisisnotawill` shared their experience with developing an AI Dungeon Master, revealing the difficulty of maintaining a cohesive storyline over hundreds of messages. Despite using hierarchical summarization and RAG, the AI seems to falter in plot continuity. Concerns include the AI repeatedly referring back to past events or veering into an unexpected direction. 
- **Balancing Player Freedom and Plot Continuity**: `@thisisnotawill` expressed the need for an AI solution that can maintain the narrative's coherence while still allowing for player freedom and a dynamic plotline. The challenge is managing the fine balance between AI-guided plot arcs and player choices.
- **Simplifying Problem Solving Approach**: `@justahvee` suggested narrowing the focus to solve the most crucial problems first. They advised isolating a single story arc and successfully execute it end-to-end before delving into dynamic plots. The emphasis is on creating a system that works well with a particular set of user needs before expanding to a wider range.
- **Real-Time AI-assisted DMing**: User `@pantsforbirds` revealed their work on creating a real-time AI-assisted Dungeons and Dragons tool. The model in development aims to parse player conversations and furnish possible details on player queries. 
- **Collaboration and Sharing**: Both `@thisisnotawill` and `@pantsforbirds` expressed interest in each other's projects and the potential for collaboration or learning from each other's experiences.


### ▷ Channel: [speed](https://discord.com/channels/1168579740391710851/1168986766607384638) (2 messages): 

- **Vector and Keyword search**: `@jeffreyw128` discussed the use of traditional search providers for keyword search in their project named "metaphor", highlighting its speed. They mentioned that they also have a neural, vector-based search system, and their project handles the return of HTML contents.


### ▷ Channel: [feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549) (1 messages): 

- **Proposed Prompt Channel**: `@pantsforbirds` suggested the creation of a **prompt channel** for discussion and sharing of prompts, indicating the potential benefits and utility of having such a centralized space.


### ▷ Channel: [cost](https://discord.com/channels/1168579740391710851/1169026016887459961) (15 messages): 

- **Cost Structure of Running a PDF-to-RAG/LLM Pipeline**: User `@res6969` shared an insight that while operating a PDF-to-RAG/LLM pipeline, OCR through Azure accounts for 52% and GPT-4's new pricing stands for 48% of total costs per document. It was also mentioned that the rate of OCR costs will increase over time while GPT-4's share will decrease.
- **Azure OCR vs Open Source Models**: When asked about the accuracy of Azure OCR, `@res6969` stated that it was essentially 100% accurate with the exception of occasional I's and L's. Open source models were said to be nearly as accurate but as expensive to run.
- **Migration from AWS Textract to Azure OCR**: `@res6969` shared that they moved from AWS Textract to Azure OCR due to the equivalent cost. Azure OCr was preferred as it provided higher rate limits on the beginning plan and better support for table extraction from documents.
- **Problems with Nougat**: User `@pantsforbirds` raised concerns about facing production issues with Nougat, stating that it was particularly challenging to debug. `@pantsforbirds` expressed interest in testing AWS Textract due to its multicolumn support.
- **Experience with AWS Textract**: `@pantsforbirds` shared a link to an AWS blogpost titled ["Amazon Textract's new layout feature introduces efficiencies in general-purpose and generative AI document processing tasks"](https://aws.amazon.com/blogs/machine-learning/amazon-textracts-new-layout-feature-introduces-efficiencies-in-general-purpose-and-generative-ai-document-processing-tasks/).


### ▷ Channel: [irl](https://discord.com/channels/1168579740391710851/1171569983688560732) (1 messages): 

- An user going by the handle `@thebaghdaddy` asked if anyone in the channel is located in **Vermont**.


### ▷ Channel: [openai](https://discord.com/channels/1168579740391710851/1171903046612160632) (8 messages): 

- **OpenAI Platform Issues**: User `@robhaisfield` raised an issue about OpenAI being slow. `@iyevenko` confirmed the problem and mentioned inability to log into the platform.
- **Wish for Mistral 70b Model**: `@robhaisfield` expressed a desire for a **70b model from Mistral** to reduce dependence on OpenAI.
- **OpenAI Incident Information**: `@pantsforbirds` informed about an incident at OpenAI, linking this issue to some db problems on OpenAI's end. They provided links to the [overall status](https://status.openai.com/) and the [specific incident](https://status.openai.com/incidents/q58417g6n5r7) for further details.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

**MLOps @Chipro Channel Summaries**

### ▷ Channel: [general-ml](https://discord.com/channels/814557108065534033/828325357102432327) (3 messages): 

- **Cross-posting Content**: User `@wangx123` shared a [YouTube video link](https://www.youtube.com/watch?v=a8Ar4q1sGNo&t=41s) across multiple channels. This led to a brief discussion among other users including `@c.s.ale` and `@huikang` about the feasibility and need of a *Discord rule to automatically remove post if posted onto multiple channels in a short period of time*.
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- Discussed the importance of proper **commit practices** and code quality in the `#general` channel. `@kasparpeterson` expressed concern over direct commits to master, CI failures, and `prettier` and `eslint` errors. Suggestions to improve these practices included adhering to Git PR flow, enforcing guidelines, and adding Github Actions for public repos to ensure quality.
    - "*We had some straight commits to master, and some CI failures...*" - `@kasparpeterson`
- In the `#general` channel, `@._z` acknowledged these issues, suggesting that they were a result of the current setup and invited a revision of the contributing guidelines.
- Meeting participations and proposals were addressed by `@pwuts`, showing willingness to answer queries regarding a previously proposed idea and intent to participate in upcoming meetings.
- In the `#events` channel, posts from `@hackgoofer` and `@._z` included notifications and reminders of the AIEF Weekly Meeting and the AIEF/Agent Protocol meeting.
    - [Event link](https://discord.gg/kTjfCBdA?event=1178825560793821235)
- The topic of including **authentication in the core protocol** emerged as a discussion point in the `#events` channel, with `@kasparpeterson` proposing its inclusion and sharing a [detailed comment](https://github.com/AI-Engineer-Foundation/agent-protocol/issues/39#issuecomment-1824366809) on GitHub. Members `'@hackgoofer'`, `'@ntindle'`, and `'@juanreds'` displayed unified approval for integrating authentication into the core protocol.
    - "*I recommend that authentication should be part of the core protocol...*" - `@kasparpeterson`

**AI Engineer Foundation Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1144960932196401252/1144960932657758210) (10 messages): 

- **Discussion on Commit Practices**: In the conversation between `@kasparpeterson` and `@._z`, the proper commit practices and use of precommit hooks were analyzed. `@kasparpeterson` noted there have been straight commits to master and CI failures, along with `prettier` and `eslint` errors, suggesting that the team should follow git PR flow. `@._z` explained these errors might have occurred due to their current setup and suggested revising the contributing guidelines.
- **Efforts Toward Code Quality**: `@kasparpeterson` further proposed enforcement of existing guidelines rather than revising them. He suggested adding Github Actions for public repos to ensure quality, including a separate CI job for every push to validate the schema, using a simple example server and testing command. The objective would be to provide a firm safety to the core repository.
- **Planning for Future Meetings**: `@pwuts` expressed his willingness to answer any queries about an idea he proposed, which was discussed in a previous team meeting. He also plans to attend the coming meeting.


### ▷ Channel: [events](https://discord.com/channels/1144960932196401252/1144960932657758212) (10 messages): 

- **Tomorrow's AIEF Weekly Meeting Announcement**: User `@hackgoofer` shared a Discord invite link for an upcoming AIEF Weekly Meeting [event](https://discord.gg/kTjfCBdA?event=1178825560793821235).
- **Reminder of AIEF/Agent Protocol Meeting**: User `@._z` sent out a reminder about the AIEF/Agent Protocol meeting happening four hours after the message was posted.
- **Discussions on Including Authentication in the Core Protocol**: `@kasparpeterson` recommended that authentication should be part of the core protocol and shared a link to his detailed explanation on GitHub. He indicated that it is optional to pass tokens in the Authorization header field. He also suggested to approach the implementation lean and *skip the plugins* and finish the info endpoint as the config options are optional anyway. Their [comment can be found here](https://github.com/AI-Engineer-Foundation/agent-protocol/issues/39#issuecomment-1824366809).
- **User Responses to the Topic of Authentication**: Other users `'@hackgoofer'`, `'@ntindle'` and `'@juanreds'` shared their views on the topic. There was a general consensus in favor of including authentication in the core protocol, with `'@ntindle'` and `'@juanreds'` specifically stating they agree it should be part of the core protocol.
- **AIEF/Agent Protocol Meeting Starting Reminder**: User `@._z` posted a reminder message that the meeting was starting and prompted attendees to check the meeting notes and add any topics for discussion.


        

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

**YAIG (a16z Infra) Channel Summaries**

### ▷ Channel: [ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565) (2 messages): 

- **AI Research Video Discussion**: `@nickw80` recommended a [YouTube video](https://www.youtube.com/watch?v=zjkBMFhNj_g&t=2s) about the latest AI research. He highlighted that despite the title, the video covers recent studies and is worth watching.
        
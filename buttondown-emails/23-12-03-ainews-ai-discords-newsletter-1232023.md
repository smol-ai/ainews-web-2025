---
id: a307471c-a827-46ea-b698-b95f82067daa
title: '[AINews] AI Discords Newsletter  12/3/2023'
date: '2023-12-03T19:48:19.378522Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-ai-discords-newsletter-1232023
---

<!-- buttondown-editor-mode: plaintext -->[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- *'Open-ended' discussion and proposals* on ways to Query and Update Knowledge Graphs, Graph based RAG setups, Synthetic Data Pipelines, and Temporal Representation in Knowledge Graphs, with a few users like `@raddka`, `@spaceman777`, and `@maxwellandrews` sharing their ideas, along with some relevant resources like a [neo4j advanced RAG template](https://github.com/langchain-ai/langchain/tree/master/templates/neo4j-advanced-rag).
- *Conversations revolving around AI Models* where users engaged in tests with local language models, efficiency and performance comparisons, model merging discussions, and anticipating future developments. Key topics included the OpenEmpathic project, the 'I will tip you $200 trick' for interacting with AI models, and the newly released Hermes Vision model. Several useful links to model notebooks, and project descriptions were shared during these conversations.
- *AI-related suggestions, complaints, and resources* were exchanged, with the notable OpenML Guide shared by `@lily_33846` and highlighting the vague use of the term 'breakthrough' in AI discussions by `@asada.shinon`.
- *Announcement of Nous Hermes 2.5 Vision* by `@teknium`, emphasizing on new features such as **Function Calling on Visual Information** and **SigLIP Integration**. The release can be downloaded from [Hugging Face repository](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision).
- *Insights on smaller datasets, quantization, and various AI models* were shared in the channel '#ask-about-llms', with active engagement from users like `@spaceman777`, `@crainmaker`, `@n8programs`, and `@casper_ai`. Topics included project involving multiple LoRAs, AutoAWQ quantization, comparisons of Starling-7B, Notus, and OpenHermes, and a discussion on Replit's performance on HumanEval.
- *Guild-specific inquiries and offers*, like how to use the hydra bot, and `@main.ai`'s offer of free GPU resources captured in the channel '#bots' and '#off-topic'.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015) (13 messages): 
        
- **Querying and Updating Knowledge Graphs**: User `@raddka` proposed a possibility of querying the graph with function calling in order to update the graph without changing the model. Additionally, if direct querying could be challenging, an implementation of a semantic search on the graph was suggested which would allow the model to perform function calls without needing to have a precise query. 
- **Graph Based RAG Setup** : `@spaceman777` discussed the potential of a hierarchical graph based RAG setup citing a [neo4j advanced RAG template](https://github.com/langchain-ai/langchain/tree/master/templates/neo4j-advanced-rag). This setup offers the opportunity to explore varying dimensions like named entities/topics/etc allowing to navigate through various levels of granularity in knowledge extraction, aiding in developing a more integrated knowledge graph at a deeper level. 
- **Synthetic Data Pipeline for KG**: `@maxwellandrews` discussed potential solutions for knowledge graph construction by proposing a synthetic data pipeline. This pipeline would extract entities through foundation models and add these to an "edit and append" Knowledge Graph that would accumulate a complete KG data from the entire document. A diverse dimensionality during the extraction process was suggested for embedding both intrinsic and extrinsic dimensions. 
- **Temporal Representation in KG**: In response to `@natefyi_30842`'s question about representing time dimension in a KG, different techniques were offered by `@maxwellandrews` and `@raddka`. This included using an additional set of edges, incorporating IDs, or querying an SQL database after the KG with relevant data combined for handling the time dimension. `@maxwellandrews` also shared seven methods of representing time in a knowledge graph as suggested by GPT4. These methods range from attaching timestamps, using temporal edges and versioning to creating time nodes, temporal properties and temporal ontologies.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928) (20 messages): 
        
- **Test with Local Language Models (LLM)**: `@airpods69` proposed a test where a large set of questions was to be run through a locally set up Large Language Model (LLM). Different users provided various chatbot responses to this experiment with models including `OpenHermes-2.5` and `internlm-chat-20b, Q5_K_M`. The questions and the responses can be found on pastebin links shared by users like `@intervitens` and `@tsunemoto`. [https://pastebin.com/qsbLC333](https://pastebin.com/qsbLC333), [https://pastebin.com/R3ZHX54E](https://pastebin.com/R3ZHX54E), [https://pastebin.com/vK33qbPZ](https://pastebin.com/vK33qbPZ).

- **'Tip Trick' in Interacting with AI Models**: `@fullstack6209` admitted to using an interaction strategy - the 'I will tip you $200 trick' - when communicating with AI models. While there was humor in the ensuing discussion, with `@intervitens` suggesting using a similar tactic with doggy treats, the underlying point was the potential effectiveness of this interaction 'trick'.

- **Use of Rust in ML Tasks**: `@.beowulfbr` expressed interest in how the Rust programming language could enhance Machine Learning (ML) tasks, sparking a brief discussion. `@lightningralf` shared links to Rust-related Discord communities for those interested: [https://discord.gg/rust-lang](https://discord.gg/rust-lang), [https://discord.gg/MZ9bUAyDKZ](https://discord.gg/MZ9bUAyDKZ), [https://discord.gg/ccuSf9NeN5](https://discord.gg/ccuSf9NeN5).

- **Offer to Utilize Free GPUs**: `@main.ai` informed that they have free GPU resources available for use and requested anyone interested to direct message them. The usage possibility of these GPUs was tied to the disclaimer of possible fire risk. This managed to spark interest among users like `@rey1337` and `@euclaise`.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192) (4 messages): 
        
- `@yorth_night` shared a [Twitter link](https://vxtwitter.com/ArmenAgha/status/1731076069170835720) without providing context.
- `@asada.shinon` expressed irritation about the vague use of the term 'breakthrough' in AI discussions, urging for more detailed explanations on how these breakthroughs advance AI.
- `@lily_33846` shared a resource, the [OpenML Guide](https://github.com/severus27/OpenML-Guide), describing it as embracing open source and free resources, and offering a numerous learning materials and updates in the AI field.


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272) (1 messages): 
        
teknium: @everyone 
Announcing Nous Hermes 2.5 Vision! 

NousResearch's latest release builds on Hermes 2.5 model, adding powerful new vision capabilities thanks to <@282315082749444097>!

Download: https://huggingface.co/NousResearch/Nous-Hermes-2-Vision

**Prompt the LLM with an Image!**
**Function Calling on Visual Information!**
**SigLIP Integration!**

This is Nous' latest version of a multimodal model with powerful capabilities, and further iterations will come in the future. 

Learn how to inference the model with instructions in the model card and stay tuned for GGUF quantization, with eventual support in @LMStudio and other inference engines!


### ▷ #[bots](https://discord.com/channels/1053877538025386074/1149866614590816256) (2 messages): 
        
- **Hydra Bot Usage Inquiry**: `@abdksyed` asked how to use the hydra bot. `@hydra` responded, indicating that they would provide assistance.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599) (424 messages🔥): 
        
- **Performance Comparison and Tests**: Users `@n8programs`, `@teknium`, `@coffeebean6887`, `@elbios`, `@n8programs`, and `@tsunemoto` engaged in discussion about various AI models' performances, including **Hermes 2.5**, **Capybara**, **Mistral**, **UltraChat**, **Goliath**, and **DeepSeek**. They discussed inference speeds, model size trade-offs, and compatibility with systems like the M3 Max or AWS.
- **Model Merging Discussion**: A lively conversation took place about the process and feasibility of merging different models, such as combining UltraChat and base Mistral to create a new **Mistral-Yarn** variant. User`@weyaxi` shared their notebook on how to implement such merges.
- **OpenEmpathic Project Discussion**: `@spirit_from_germany` called for assistance in expanding the categories of the Open Empathic project, sharing a [YouTube video](https://youtu.be/GZqYr8_Q7DE) tutorial and the [project link](https://dct.openempathic.ai/).
- **Discussion on AI Model Performance and Open Source Tools**: Various users discussed their experiences with different AI models, the efficiency of open-source platforms such as Ooba and LMStudio, and future developments in AI, such as the anticipated release of a token by Nous Research.
- **Release and Feedback on Hermes Vision Model**: `@teknium` announced the release of the Hermes Vision model and users have been sharing their experiences and providing feedback on its performance. Some pointed out issues with hallucination and inaccuracies, suggesting room for improvement.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927) (107 messages): 
        
- **Project Involving Multiple LoRAs and Fine Tuning Targets**: `@spaceman777` suggested checking out a project involving the use of multiple LoRAs (Learning Rate Multipliers) applied as gradients over the transformer layers. They provided a link to the project [here](https://huggingface.co/Gryphe/MythoMix-L2-13b).

- **Comparisons of Smallest Datasets Yielding Coherent Results:** A discussion took place regarding the smallest datasets that can yield coherent results. Both `@crainmaker` and `@n8programs` agreed that TinyStories is still leading in this aspect.

- **AutoAWQ Quantization Discussion:** A series of interactions occurred revolving around AutoAWQ quantization and how to optimize it to prevent out of memory issues. The users `@jdnuva`, `@teknium`, and `@casper_ai` discussed using a device map to help with memory management during quantization, with `@casper_ai` (the author of AutoAWQ) providing specific advice on the topic.

- **Comparison of Language Models Starling-7B, Notus, and OpenHermes:** `@lightningralf` inquired about the comparison between the language models Starling-7B, Notus, and OpenHermes, providing the respective links for each ([Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) and [Notus-7B-v1](https://huggingface.co/argilla/notus-7b-v1)).

- **Discussion on Replit's Performance on HumanEval**: `@Sid` posed a question on why Replit models show strong performance on HumanEval despite using relatively fewer tokens and epochs. They noted that these models don't appear to use synthetic data.


### ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831) (1 messages): 
        
lily_33846: OpenML Guide: Embracing Open Source and Free Resources, Offering a Wealth of Books, Courses, Papers, Guides, Articles, Tutorials, Notebooks, AI Field Advancements, and Beyond.

https://github.com/severus27/OpenML-Guide


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Notable debate on the comparative reasoning abilities of **GPT-3** and **GPT-4** in SQL contexts, triggered by a question from `@plakis`.
- Various technical inquiries:
    - User `@quantumqueenxox` encountering challenges in implementing the system prompt in a **Conversational Retrieval Chain**. 
    - `@_thebeginner` expressing setup issues with **Github Codespaces** for Langchain.
    - Question from `@sowmyan_90011` on using an async call inside a custom `@tool` in Langchain.
    - Query by `@menny9762` on optimization strategies when using a **Parent Document Retriever** inside a **ConversationalRetrievalQAChain**.
- Substantial interest in personalized user experiences with AI tools:
    - `@wei5519` expressing interest in creating a **personal stylist** using **LangChain + ChatGPT**.
    - `@seththunder` exploring strategies to set up an individual chat history for each user instead of a universal history.
    - `@jan_naj` seeking advice on mimicking **OpenAI Assistant** features using a Langchain agent.
- User `@discossi` shared a valuable collection of **machine learning resources** and updated their chatbot project to improve context search accuracy using **Textacy** for metadata key extraction. Access the GitHub repository [here](https://github.com/ossirytk/llm_resources/).
- Conversation on the applicability of AI tools for image manipulation, with a query from `@user2m` regarding an AI tool for **image masking** on clothing models, and `@seththunder` informing that LangChain doesn't currently support images.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148) (26 messages): 
        
- **GPT-3 vs GPT-4 Reasoning Ability**: User `@plakis` has observed that in the context of being a SQL agent, **GPT-4** demonstrates superior reasoning ability compared to **GPT-3**. They asked if there's a way to improve GPT-3's performance in this aspect or if a specific fine-tune model of GPT-3 could help achieve this.
- **System Prompt in Conversational Retrieval Chain**: User `@quantumqueenxox` asked for advice on how to use the system prompt inside a **Conversational Retrieval Chain**, providing code and describing a validation error received. 
- **Offline Version of Langsmith**: User `@tech36814_71262` asked if there exists an offline version or alternative of **Langsmith**, indicating the need for an offline debug tool for Langchain.
- **Setup Issue with Github Codespaces for Langchain**: User `@_thebeginner` encountered an error while setting up a project via **Github codespaces** for Langchain. They reported an error message, with ChatGPT suggesting the error was due to a missing '../core' directory during the Docker image build process.
- **Personal Stylist with LangChain + ChatGPT**: User `@wei5519` expressed interest in building a personal styler using **LangChain + ChatGPT**, asking for any recommended resources or tutorials.
- **Async Call in Custom Tool**: User `@sowmyan_90011` had a question about how to use async call within a custom tool using the @tool decorator in Langchain.
- **Using Parent Document Retriever**: User `@menny9762` inquired about using a **Parent Document Retriever inside a ConversationalRetrievalQAChain** and discussed the optimization of document retrieval.
- **Individual Chatbot Memory for Users**: User `@seththunder` asked if it was possible to set up individual memory/chat history for each user interacting with their chatbot, instead of having a universal chat history.
- **Replicating OpenAI Assistant with Langchain**: User `@jan_naj` asked for advice on replicating the **OpenAI Assistant's features** (specialized agents with thread messages) using a **Langchain agent**.
- **AI Tool for Image Masking**: User `@user2m` asked if there exists an **AI tool to seamlessly overlay a high-quality image of new t-shirts on the clothing models** they have photographed. `@seththunder` responded that LangChain does not support images as of yet.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729) (2 messages): 
        
- **Machine Learning Resources**: User `@discossi` shared a [GitHub repository](https://github.com/ossirytk/llm_resources/) containing **links to several useful beginner resources for machine learning and Language Model's (llm's)**, which includes free courses, YouTube channels, books, research papers, and developer tools.

- **Chatbot Text Document Parsing**: `@discossi` also discussed about a recent update in their [chatbot project](https://github.com/ossirytk/llama-cpp-chat-memory); where they've implemented **text document parsing with Textacy to extract entities like people, organisations, and locations**. These entities can be used as metadata search keys for vector store memory, which improves context search accuracy and reduces the need for manually typing metadata keys for documents.


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538) (1 messages): 
        
wei5519: Hi , I would like to build a personal stylist using langchain + Chatgpt  for practice. Are there any resources or tutorials for similar projects that you'd recommend? How can I approach this problem? Thank-you


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Members expressed interest in exploring polysemanticity in Large Language Models (LLMs), inviting conversations and possible collaborations.
- Discussion on the drawbacks of training tokenizers with over 1 million tokens: "*lmao 1mil+ token tokenizers. I'm good i would rather not train that.*"
- Detailed conversation on the potential monetization of **Axolotl** and **OpenChat Trainer**, focusing on hosted training and enterprise support plans, while highlighting controversies around inserting feature paywalls.
- Professional profile of a **Full Stack Developer** with extensive experience in financial sector was shared seeking **remote work** opportunities, showcasing expertise with React.js, Angular, Vue.js, Python, FastAPI, Django, and microservices, as well as projects involving ChatGPT and OpenAI.
- Extensive technical discussion on **JIT compilation**, reflecting a strong preference for JAX over PyTorch: "*best JIT in the world*" and resolving issues with PyTorch's JIT that "*break 50% of the time*". Also, shared expectations towards the new **Orca-2 dataset** release.

**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804) (1 messages): 
        
mayhem1349.: Hey everyone - has anyone worked on polysemanticity on LLMs before? Would love to have a discussion


### ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261) (1 messages): 
        
mayhem1349.: Hey everyone - has anyone worked on polysemanticity on LLMs before? Would love to have a discussion


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) (1 messages): 
        
ufghfigchv: lmao 1mil+ token tokenizers. I'm good i would rather not train that.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553) (16 messages): 
        
- **Monetizing Axolotl and OpenChat Trainer**: `@imonenext` suggested the idea of possibly monetizing aspects of Axolotl and OpenChat Trainer. 
- **Possible Opportunities for Revenue**: `@caseus_` proposed that potential opportunities could arise from providing hosted training and offering enterprise support plans.
- **Paywall Controversy**: The idea of inserting paywalls for features was debated. `@caseus_` expressed hesitation due to potential backlash from the open-source software community. `@imonenext` suggested that features not likely to be used by the open-source software community could be put behind a paywall, but this was strongly disagreed with by `@giftedgummybee`. 
- **Opposition to Paywalls and Closed Source**: `@giftedgummybee` expressed strong opposition to both paywalls for features and closing the source, citing the need to avoid antagonizing users and referring to the backlash that occurred during the Reddit API situation.
- **Suggested Approach**: Ultimately, `@giftedgummybee` suggested that more success would be found in leveraging the platform and providing benefits rather than directly selling it for profit.


### ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027) (1 messages): 
        
frankyan.ai: Hi, I hope this proposal finds you well.

I have over 25 years of work experience in full stack development and software engineering. I have worked as a programmer and a chief engineer for various projects in the financial sector. I have learned and updated my skills by following the latest trends in technology.

Some of my accomplishments include:
• Creating websites and single page applications and dashboard with React.js, Next.js, Angular, Vue, Three.js, WebGL, jQuery, HTML5, CSS3, Bootstrap, Tailwind CSS, Material UI, Storybook UI Components from scratch
• Designing and implementing microservices using different languages and frameworks such as Python, FastAPI, Flask, Django, Node.js and Express.js, and using Database with MySQL, Postgres and MongoDB
• Using open-source components and automation tools to build with AWS's built-in components
• Experiencing the evolution of system architecture from hardware servers to virtual machines, to containerization and Cloud
• Developing core banking systems and integrated solutions for financial equipment
• Building CI/CD pipelines for startup teams

I practice DevOps skills daily by creating a project that uses Terraform and Ansible scripts to request and release various cloud resources, such as hosts, networks, domains, etc. It also sets up CI/CD pipelines and various infrastructures for logging, monitoring, and tracing.

I have worked on several projects using ChatGPT and other technologies. Some of my accomplishments include:
• Implementing AI-enhanced resume matching based on company recruitment requirements using FastAPI + ReactJS + OpenAI
• Building a simple system that allows my friends in China to access ChatGPT using Express + TypeScript + ChatGPT and deploying it on Alibaba Cloud using Docker
• Researching FastChat to support LangChain integration and installing LangChain on AWS

Please let me know if you are looking for a fully remote developer on your side.
I can get started working immediately.

Regards!


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022) (4 messages): 
        
- **Orca-2 Dataset Release Discussion**: User `@caseus_` expressed a hope for the release of the **Orca-2 dataset** without answers, citing a [Twitter post](https://fxtwitter.com/winglian/status/1731169945461932203).
- **JAX vs PyTorch JIT Compilation**: `@imonenext` expressed a strong preference for **JAX**, calling it the "*best JIT in the world*", and criticized PyTorch's JIT capabilities.
- **Problems with PyTorch JIT**: `@euclaise` agreed with `@imonenext`, stating that PyTorch's JIT tends to "*break 50% of the time*".
- **Advantages of JAX JIT**: `@euclaise` further praised JAX's JIT, noting its ability to implement things like **custom types** (e.g., complex32) or algorithms like **prefix-sum scan** without having to resort to CUDA code.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- Highlighting an unusual tweet by the `ChatGPTapp` account, `@bk_42` prompted a discussion. `@guardiang` took it as their view of the next year or two and speculated on GitHub's release, including the addition by MSFT.
- `@slono` responded to a trend he observed on Hacker News around the perception of LLMs being useless, attributing some of this to generational differences and changes in education. `@swyxio` recommends a [LLM course reading list](https://github.com/mlabonne/llm-course), which `@slono` recognized as substantial.
- `@optimus9973` shared Simon Willison's take on a related matter via this [Fediverse link](https://fedi.simonwillison.net/@simon/111160011218088991). He expressed a desire to lead by example rather than convincing LLM skeptics, referencing Simon Willison's efforts.
- `@danimp` shared his approach of educating close friends and family about AI by translating technical terms to layman's language, aiming to mitigate potential misunderstandings and highlighting its practical applications.
- There was a brief discussion regarding Q's issues, kicked off by `@aravindputrevu`. A debated topic was about operational security and data privacy in the context of programming and handling AI, with insights from `@slono` and `@tiagoefreitas`.
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- `@pantsforbirds` compared the speed and output quality of a newly set-up programming branch with Nougat, expressing satisfaction with initial results and mentioning the need for a more formal benchmarking. 
- A lively discussion about an interactive hacking game named [**LLM Prompting Hack Game**](https://gandalf.lakera.ai/), shared by `@thebaghdaddy`. Comments from `@joshcho_` and `@nosa_.` highlighted the enjoyment and difficulty of the game. 
- New AI model [**Nous Hermes 2 Vision**](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision) was introduced by `@potrock`, and `@pantsforbirds` expressed their interest in testing the model in the future.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855) (1 messages): 
        
jeffreyw128: @jeffreyw128#0623 what’s funny is during


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082) (2 messages): 
        
- **Nous Hermes 2 Vision**:
    - User `@potrock` shared a link to the [**Nous Hermes 2 Vision**](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision) model hosted on Hugging Face.
    - `@pantsforbirds` expressed interest in testing this model, stating that they have saved it for future examination.


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638) (1 messages): 
        
pantsforbirds: I set it up on a new branch and it’s absolutely smoking Nougat in terms of speed. Results look pretty good too. Need to set up a real benchmark, but so far I’m very impressed. 

The worst part of Nougat is that when it hallucinates, it goes wild. It inserted a wild rant about silicon substrates in the middle of my legal document parsing 💀


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677) (5 messages): 
        
- **LLM Prompting Hack Game**: `@thebaghdaddy` shared a link to a game (https://gandalf.lakera.ai/) where participants attempt to prompt hack an LLM with instructions not to reveal a password. A second LLM starts screening the outputs from level 3, which adds to the game's difficulty. `@joshcho_` and `@nosa_.` commented on the challenge, with `@nosa_.` indicating they found the game fun.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Model Version Log Inquiry**: `@elbios` asked if there was a 'version log' or changelog of Samanthas. `@faldore` responded that there's no dedicated log, **the updates are shared through their blog and model cards**.

        

---

## [Ontocord (MDEL discord)](https://discord.com/channels/1147858054231105577) Discord Summary

Only 1 channel had activity, so no need to summarize...

huunguyen: Let's use this standard <@718586490866630740> <@802204699235188767>
        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

Only 1 channel had activity, so no need to summarize...

steven_ickman: I’m sorry I’ve been out of town the last couple of weeks and my last day at Microsoft is Monday. I’m trying to wrap up all my work with Microsoft and should be able to engage with this group front desk Tuesday on.
        

---
The Skunkworks AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
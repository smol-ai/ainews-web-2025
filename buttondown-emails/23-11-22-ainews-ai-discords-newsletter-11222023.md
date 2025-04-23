---
id: d316fcf5-a656-422c-b716-8dff01195fa4
title: '[AINews] AI Discords Newsletter  11/22/2023'
date: '2023-11-23T00:10:39.341305Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-ai-discords-newsletter-11222023
---

<!-- buttondown-editor-mode: plaintext -->
## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion regarding **GPT Errors and Outages**: Users reported experiencing difficulties using GPT due to a reported outage. An OpenAI status link indicating a problem with "database replicas" was shared. [Status OpenAI Link](https://status.openai.com)
- Exploration of **Alternative Models**: In response to the GPT outage, users used different models, including ones from anyscale. One user was experimenting with laundering messy notes through claude/chatgpt. Results were shared which are found here: [Prefetching Data on Hover with RTK Query](https://write.as/mnmlmnl/prefetching-data-on-hover-with-rtk-query).
- Updates on **OpenAI Leadership**: Mention of Sam Altman's return was given, along with concerns around potential trust issues in the market due to the event.
- Clarification around **ChatGPT's 128k Context Window**: Questions were posed about its availability and if it still loses the same amount of context as prior to its last update.
- Detailed explanation of **"Test-time Search" Discussion**: The concept of "test-time search", especially in the context of using LLMs for code generation, was clarified. Relevant resources on the topic was shared: [Beam Search Link](https://d2l.ai/chapter_recurrent-modern/beam-search.html) and [ChatGPT Link](https://chat.openai.com/share/c63e6e80-dcd8-4ccf-8e42-2e7e23d6b006).
- Announcement of a **Local Consistency Models Discussion** session which included `<@296887155819675650>` and a link to join the session was shared [this link](https://lu.ma/llm-paper-club).
- Further discussion on **Latent Consistency Models** in another chat session featuring Elon Musk.
- Sharing of Stella Biderman's tweet outlining **LLM Training**: "how to train LLMs in 2023" [Link](https://x.com/BlancheMinerva/status/1721380386515669209?s=20).
- Recommendation for a survey paper providing an understanding on **Long Context Methods** [Link](https://arxiv.org/abs/2311.12351).
- Sharing and experience of **One-click Installer**, and note on its performance on different hardware [Link](https://twitter.com/cocktailpeanut/status/1727111213421601062).
- Mention of **Controlnet's Speed** for improving the performance of the one-click installer.

**Latent Space Channel Summaries**

### ▷ Channel: [ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876) (12 messages): 

- **GPT Errors and Outages**: User `@dsquared70` reported experiencing errors when trying to use GPT, while `@vcarl` also noticed an outage. `@slono` shared the OpenAI status link which indicated an issue with "database replicas" [Status OpenAI Link](https://status.openai.com).
  
- **Alternative Models**: Amidst the outage, `@slono` experimented with models on anyscale and found them to be usable. He also discussed laundering his messy notes through claude/chatgpt, sharing the link to his findings on the experiment [Prefetching Data on Hover with RTK Query](https://write.as/mnmlmnl/prefetching-data-on-hover-with-rtk-query).
  
- **Reinstatement of Sam Altman**: `@thatferit` mentioned the news of Sam Altman's return, but suggested that trust issues might still persist in the market due to the event.

- **ChatGPT's 128k Context Window**: `@kb.v01` asked about the availability of the 128k context window in ChatGPT, stating it seems to lose the same amount of context as before the last announcement.

- **"Test-time Search" Discussion**: `@cakecrusher` asked for clarification on what "test-time search" means in the context of using LLMs for code generation. `@slono` responded with an explanation and shared a resource that provides more information on the topic [Beam Search Link](https://d2l.ai/chapter_recurrent-modern/beam-search.html). He also provided a chatgpt link that helped him find the resource [ChatGPT Link](https://chat.openai.com/share/c63e6e80-dcd8-4ccf-8e42-2e7e23d6b006).


### ▷ Channel: [ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499) (1 messages): 

- **Local Consistency Models Discussion**: A session discussing **Local Consistency Models** was announced by `@swyxio` to start in 12 minutes. The session included `<@296887155819675650>` and was accessible via [this link](https://lu.ma/llm-paper-club).


### ▷ Channel: [llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663) (12 messages): 

- Discussion on **Latent Consistency Models**: `@eugeneyan` posted an invitation to join a talk featuring Elon Musk on **Latent Consistency Models**.
- Comment on **LLM Training**: `@swyxio` shared a tweet from Stella Biderman outlining "how to train LLMs in 2023" [Link](https://x.com/BlancheMinerva/status/1721380386515669209?s=20).
- Reference to **Long Context Methods Paper**: `@swyxio` recommended a survey paper that explores various long context methods [Link](https://arxiv.org/abs/2311.12351).
- Experiment with **One-click Installer**: `@growthwtf` shared a working one-click installer for training models, noting that it runs extremely slowly on an M2 Mac with 16GB memory [Link](https://twitter.com/cocktailpeanut/status/1727111213421601062). 
- Mention of **Controlnet's Speed**: `@growthwtf` mentioned that Controlnet offers better speed for the one-click installer.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Announcement of the **Template Hub Launch** for LangChain. Users are encouraged to view, explore, and like templates, and provide feedback on desired future templates. [Template Hub](https://templates.langchain.com/) provided as a resource.
- Various technical queries and assistance requests in **LangChain integration**, including LangChainStream implementation, LLMs assistance, temperature settings effects, integrating tools into agents, and GoogleDriveLoader credentials. Noteworthy assistance was provided by `@1072591948499664996` in response to `@snake_ai`'s query on integrating openai functions with open source llama models or Neo4j DB QA chain with llamacpp LLM.
- A notable issue regarding a **FastAPI application integration with LangChain** was reported by `@FRODO` on GitHub. The mentioned issue can be seen [here](https://github.com/langchain-ai/langchain/issues/13750).
- Ongoing discussion on **Langserve's functionality**, particularly the implementation of tracing Langserve endpoints, accessing tracer programmatically, and token usage tracking within a client app.
- A brief discussion on **key value database** and its interoperability with Azure Cache for Redis with a link provided by `@rpall_67097`. The referenced link can be viewed [here](https://www.mongodb.com/databases/key-value-database#:~:text=Key%20value%20databases%2C%20also%20known,associated%20value%20with%20each%20key.).
- Sharing of personal work, such as the launch of an **open source library for error mitigation in OpenAI's APIs** with fallback to Azure, a LinkedIn post on Streamlit app development, and the early version of **Sunholo LLMOps** tool for LLM creation on Google Cloud Platform. Shared resources include:
    - OpenAI Load Balancer Tool [link](https://github.com/Spryngtime/openai-load-balancer) by `@rickl`.
    - LinkedIn post on Streamlit app development [link](https://www.linkedin.com/posts/akash-desai-1b482b196_app-streamlit-22-november-2023-activity-7133088412391063552-mXWc) by '@akash AI'.
    - Early version Sunholo LLMOps [link](https://github.com/sunholo-data/sunholo-py) by `@markedmo`.
- Shared link to the Discord server, **Jobcord**, by `@h3l1221` in various channels. [Jobcord link](https://discord.gg/jobcord).
   

**LangChain AI Channel Summaries**

### ▷ Channel: [announcements](https://discord.com/channels/1038097195422978059/1058033358799655042) (1 messages): 

- **Template Hub Launch**: `@hwchase17` announced the release of a new hub for viewing, exploring, and liking templates on LangChain. The intent is to make this the easiest way to start building with gen ai.
- **Liking and Requesting Templates**: Users are encouraged to like templates they've found useful and can now request a template, giving feedback on what templates they would like to see in the future.
-  Link:
    - [Template Hub](https://templates.langchain.com/) for viewing, exploring, and liking LangChain templates.


### ▷ Channel: [general](https://discord.com/channels/1038097195422978059/1038097196224086148) (39 messages): 

- **LangChain SDK Integration**: `@daii3696` asked about how to implement `LangChainStream` from `vercel ai sdk` on a nodejs backend. No users provided an answer.
- **LLM Assistance**: `@snake_ai` experienced an error in implementing `create_structured_output_chain` with open source LLMs like llama2 and sought help from user `@1072591948499664996`. They also inquired about the possibility of using openai functions with open source llama models or Neo4j DB QA chain with llamacpp LLM.
- **LangChain Querying and Temperature Settings**: `@mukeshodhano` and `@seththunder` discussed LangChain installation versions and how temperature settings affect the output.
- **Adding Tools to Agents**: `@hamza_sarwar_` sought advice on how to integrate an SQL tool into an agent containing both RAG and Search tools.
- **Issues, Feedback and Suggestions**: 
    - `@optimizeprime_` needed advice on implementing a RAG system on a hierarchical dataset. He requested insights on effective strategies, best practices and potential challenges.
    - `@gaston411` needed help with GoogleDriveLoader credentials for LangChain and asked for suggestions, examples or tutorials.
    - `@tale.sh` reported a spam issue and requested action be taken.
    - `@FRODO` needed assistance with a problem he encountered when integrating a FastAPI application with LangChain's chains. The issue is detailed [here](https://github.com/langchain-ai/langchain/issues/13750).


### ▷ Channel: [langserve](https://discord.com/channels/1038097195422978059/1170024642245832774) (3 messages): 

- **Tracing Langserve Endpoints**: User `@mbaburic_24680` discussed the implementation of tracing Langserve endpoints from a client app through `RemoteRunnable`. They stated that Langsmith is tracing client interactions and FastAPI endpoints in separate traces. They are seeking to track and measure tokens spent by the FastAPI endpoint from their client app.
- **Accessing Tracer programmatically**: `@mbaburic_24680` wishes to access the tracer on the endpoints side programmatically and retrieve token usage and cost data from the client app. They had reviewed the cookbook and examples on the website but couldn't find a suitable solution.
- **Single Root Trace**: If it is not possible to have a single root trace, `@mbaburic_24680` is inquiring whether it would be feasible to trace from the client app the endpoint accessed via a RemoteRunnable or have Langsmith tracing on the server side (Langserve).
- **Token Tracing**: User `@mbaburic_24680` is seeking guidance on how to track tokens spent programmatically within the Langserve and the calling app context as their endpoints are automatically created via `add_routes`.
- **Discord Server Promotion**: `@h3l1221` shared a link to a Discord server, [discord.gg/jobcord](https://discord.gg/jobcord), which was not directly related to the ongoing discussion.


### ▷ Channel: [langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282) (4 messages): 

- **Query about an Unknown Subject**: User `@lordreyan` asked what was going on in the channel, but did not specify further details, making the context of his queries unknown.
- **Discord Link Shared**: `@h3l1221` shared a discord invite link, directing to **jobcord**, and mentioned **everyone** in the channel.
- **Key value Database Query Response**: In response to a query from user `@1121736064751636510`, `@rpall_67097` shared a link from mongodb.com that explains what a *[key value database](https://www.mongodb.com/databases/key-value-database#:~:text=Key%20value%20databases%2C%20also%20known,associated%20value%20with%20each%20key.)* is. 
- **Azure Cache for Redis Query**: `@rpall_67097` asked user `@1033432389516546158` about the interoperability of the subject with Azure Cache for Redis, seeking any related examples.


### ▷ Channel: [share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729) (4 messages): 

- **Emergency OpenAI + Azure fallback and retry tool Release**: `@rickl` announced the launch of an open source library that helps mitigate errors & downtime from OpenAI's APIs by falling back to Azure (and vice versa). The tool was shared on GitHub and can be found here: [github.com/Spryngtime/openai-load-balancer](https://github.com/Spryngtime/openai-load-balancer).
- **LinkedIn Post by Akash AI**: User `@akash AI` shared a LinkedIn post related to app development using Streamlit, with the post available here: [linkedin.com/posts/akash-desai-1b482b196_app-streamlit-22-november-2023-activity-7133088412391063552-mXWc](https://www.linkedin.com/posts/akash-desai-1b482b196_app-streamlit-22-november-2023-activity-7133088412391063552-mXWc)
- **Release of Sunholo LLMOps** : `@markedmo` released the early version of Sunholo LLMOps on pip. This tool aids in enabling Langchain and other LLM creation on Google Cloud Platform and can be found here: [github.com/sunholo-data/sunholo-py](https://github.com/sunholo-data/sunholo-py).
- **Invitation to Jobcord Discord server**: User `@h3l1221` shared an invitation to a Discord server named Jobcord. This is the link to the mentioned server: [discord.gg/jobcord](https://discord.gg/jobcord).


### ▷ Channel: [tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538) (1 messages): 

- A link to the **Jobcord Discord server** was shared by `@h3l1221`. The link: [discord.gg/jobcord](discord.gg/jobcord)


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Extensive discussions about OpenAI's APIs, most notably between Azure OpenAI API vs OAI API. The requirement of a custom domain email address for setting up Azure OpenAI API was mentioned by user `@roko_the_basilisk`, who also reported Azure OpenAI as 40-50% faster. Another user `@max_paperclips` mentioned Azure compute to be more stable.
- Noted potential outage of OAI API discussed by users `@teknium` and `@max_paperclips`.
- Users `@teknium` and `@gabriel_syme` had an enlightening discussion on implementing **Quality Diversity Artificial Intelligence (QDAIF)** for improving diversity within synthetic data generation, especially beneficial for language model training.
- A question raised by `@nonameusr` on why the model Nous-Capybara-34B had not made it to the language model leaderboard yet.
- Multiple AI-related discussions and links were shared by various users, which were about XwinCoder-34B as potentially good at coding, and AI applications within Twitter. Several important links were also shared. [Twitter Discussion Link 1](https://fxtwitter.com/blader/status/1727105236811366669?s=46), [Twitter Discussion Link 2](https://twitter.com/__vec__/status/1726772188714283065), [Twitter Discussion Link 3](https://fxtwitter.com/levelsio/status/1727141971603370134)
- In-depth discussions on the utilization of YaRN + LoRA. A user named `@pritishmishra3` confirmed their use [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) as the base model which they fine-tuned with LoRA.
- Confirmation about preservation of long-context properties with LoRA.
- Shared [GitHub link](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) for script to perform stress tests on yarn models by user `@yorth_night`.
- Discussion about the drawbacks of using GPT-4 for bias evaluation, with an understanding that it may lead to bias towards its own answers. Users `@casper_ai` and `@ldj` participated.
- User `@elbios` expressed concerns over OpenAI's free voice chat feature in the ChatGPT app, stating that it offered better voices which affected their startup.
- A range of relevant interesting links were shared such as OpenAI Discussion Forum on Democratic Inputs, an ArXiv paper titled "Modeling Regret Minimization in Games with ML", a GitHub project that converts screenshots into HTML/CSS, and a link related to GPT-4v.
- Questions and answers on why LoRa works better on Transformer layers compared to fine-tuning the final layers. Users `@robtoth`, `@max_paperclips` and `@.andrewnc` contributed.
- Clarification about 'llamafied' model, referring to a model that properly loads under LlamaForCausal in the Hugging Face transformers library.
- Advice shared on fine-tuning OpenHermes 2.5 using a dataset for Elixir code completion. Notably, adding new tokens to the tokenizer and retraining the embed layers were suggested.
- Strategies discussed for incorporating Japanese language into large language models (LLMs).
- Surprising Tweet from MC Hammer about large language models shared in Memes channel. [Link Here](https://twitter.com/MCHammer/status/1727248949210456230#m)

**Nous Research AI Channel Summaries**

### ▷ Channel: [ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015) (15 messages): 

- **Using YaRN + LoRA**: `@dreamgen` asked if anyone had used YaRN + LoRA, to which `@pritishmishra3` replied that they had used [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) as the base model and fine-tuned it using LoRA. In response to another query by `@dreamgen`, it was clarified that nothing special was done to preserve long-context properties as they are preserved with LoRA.
- **Script for Stress Test**: `@yorth_night` shared a [GitHub link](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) to a script for performing stress tests on the yarn model.
- **GPT-4 Evaluation Bias**: There was a discussion about the bias of GPT-4 as an evaluative measure. `@casper_ai` highlighted that GPT-4 is biased towards its own answers, and thereby using it for evaluation could result in misjudgment. They argued that it's a bad practice to use GPT-4 for evaluation. Confirming this viewpoint, `@ldj` expressed disappointment as they previously thought that it was an objective, algorithmically-based test, not AI-based.


### ▷ Channel: [off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928) (12 messages): 

- **Changing Companies Discussion**: User `@roko_the_basilisk` discussed the interpretation of "changing ships" as possibly meaning changing companies or onboarding at a new place. 
- **New Information Revealed**: User `@ifd` alerted that some new information or gossip just got released, though the specifics were not mentioned. 
- **Stress Test Inquiry**: User `@ryan_ryan_ryan_ryan` asked if there is a stress test for Yarn models. `@teknium` suggested the password test, referencing its discussion in the yarn paper.
- **Food Advice**: `@tsunemoto` humorously compares food options to AI systems stating that OpenHermes is preferred over GPT-3.5 Turbo.
- User `@crainmaker` mentioned another user with the handle `<@105478490198929408>`, but the context or purpose of this action was not provided.


### ▷ Channel: [interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192) (7 messages): 

- **Democratic Inputs Forum**: User `@roko_the_basilisk` shared a link to the OpenAI Discussion Forum on Democratic Inputs. [Link Here](https://democratic-inputs.forum.openai.com/login)
- **ArXiv Paper**: `@metaldragon01` provided a link to an ArXiv paper titled "Modeling Regret Minimization in Games with ML". [Link Here](https://arxiv.org/abs/2311.10770)
- **Screenshot-to-Code Repository**: `@yorth_night` posted a GitHub link for a project that converts screenshots into HTML/CSS. [Link Here](https://github.com/abi/screenshot-to-code)
- **GPT-4v**: `@metaldragon01` shared a GitHub Pages link related to GPT-4v, hinting that the dataset is included. [Link Here](https://sharegpt4v.github.io/)
- **AWS Party Rock**: `@firefox8975` posted a link to an intriguingly named AWS project, "Party Rock". [Link Here](https://partyrock.aws/)
- **Greg Fodor's Twitter Post**: `@manveerxyz` shared a link to a tweet by Greg Fodor discussing AI implications. [Link Here](https://twitter.com/gfodor/status/1727425728961298740?s=46)


### ▷ Channel: [general](https://discord.com/channels/1053877538025386074/1149866623109439599) (169 messages🔥): 

- **Azure OpenAI API vs OAI API**: Various users discussed issues and uncertainties concerning OpenAI's APIs. `@roko_the_basilisk` mentioned the requirement of a custom domain email address to set up Azure OpenAI API and it reported it to be 40-50% faster. `@max_paperclips` believed Azure compute to be more stable.
- **Potential outage of OAI API**: `@teknium` and `@max_paperclips` discussed a potential outage of the OAI API. `@max_paperclips` mentioned setting up Azure as a replacement in case of instability.
- **Effect of OpenAI's voice chat feature on user's startup**: `@elbios` expressed frustration over OpenAI's free voice chat feature in the ChatGPT app, which allegedly provided better voices that affected their startup building a similar app.
- **Ideas on Synthetic Data and Quality Diversity Artificial Intelligence (QDAIF) for Language Models**: `@teknium` and `@gabriel_syme` discussed the implementation of Quality Diversity Artificial Intelligence (QDAIF) for enhancing diversity in synthetic data generation, especially in the context of language model training.
- **Nous-Capybara-34B not on Model Leaderboard Yet**: User `@nonameusr` questioned why Nous-Capybara-34B was not yet on the language model leaderboard.
- **Discussions and Links about AI Advancements and Concerns**: Various users shared links and engaged in discussions about AI advancements, including `@teknium` sharing information on XwinCoder-34B, a model potentially good at coding, and `@yorth_night`, who shared a [Twitter link](https://vxtwitter.com/apples_jimmy/status/1727431072735227949) about AI applications within Twitter.
  
Links:
- [Twitter Discussion Link 1](https://fxtwitter.com/blader/status/1727105236811366669?s=46)
- [Twitter Discussion Link 2](https://twitter.com/__vec__/status/1726772188714283065)
- [Twitter Discussion Link 3](https://fxtwitter.com/levelsio/status/1727141971603370134)
- [Twitter Discussion Link 4](https://twitter.com/tszzl/status/1727113637775790127?t=Tlmg03AOEa6BqEct1fxr2w&s=19)
- [Twitter Discussion Link 5](https://fxtwitter.com/vanstriendaniel/status/1726970033497399440)
- [Twitter Discussion Link 6](https://vxtwitter.com/apples_jimmy/status/1727431072735227949)
- [Twitter Discussion Link 7](https://twitter.com/natolambert/status/1727474191925182849)
- [Hugging Face Model](https://huggingface.co/Xwin-LM/XwinCoder-34B)
- [Open Letter to the OpenAI Board](https://www.teamblind.com/post/Ex-OpenAI-letter-to-the-board-WdHhjWC4)
- [Autogen Promptbreeder](https://github.com/uukuguy/multi_loras)
- [Modal Labs for GPU Rental](https://modal.com)


### ▷ Channel: [ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927) (162 messages🔥): 

- **LoRA on Transformer Layers vs Fine Tuning Last Few Layers**: User `@robtoth` initiated a discussion asking why LoRa works better on Transformer layers' output compared to fine-tuning the last few layers. User `@max_paperclips` suggested that **LoRa targets specific features** that the final layer may be weak at, hence its strong performance. User `.andrewnc` added that **task specific knowledge** in LLMs seems to be held throughout the network rather than just in the final layers.
- **Llamafied Model**: User `@quilalove` asked about the meaning of a 'llamafied' model. User `@teknium` clarified that it refers to a model that **loads properly under LlamaForCausal** in the Hugging Face transformers library.
- **Adding New Tokens and Fine-Tuning OpenHermes 2.5**: User `@jonas69301` asked for advice on how to fine-tune OpenHermes 2.5 using a dataset for Elixir code completion. User `@yorth_night` suggested **adding the new tokens to the tokenizer** and retraining the embed layers. Later in the discussion, `@teknium` recommended **using axolotl** as the training tool.
- **LLM Usage for Small Data**: User `@kenanui` asked for the best approach to make an LLM remember a small JSON data. User `.wooser` recommended checking if **Langchain meets the requirements**.
- **Japanese Language in LLMs**: In a separate discussion, `.wooser` and `@yorth_night` discussed the challenges and potential strategies of incorporating **Japanese language** into LLMs. Suggestions included finetuning on top of existing models and continued pretraining.


### ▷ Channel: [memes](https://discord.com/channels/1053877538025386074/1166105758635655270) (2 messages): 

- `@.wooser` shared a **Twitter post from MC Hammer** discussing large language models. The specific tweet can be found [here](https://twitter.com/MCHammer/status/1727248949210456230#m)
- In another message, `@.wooser` expressed surprise at MC Hammer's interest in **large language models**.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Dialogue about starting new projects targeting text model training and improving raw decoders inspired by audio and vision. The project, termed [The-Daily-Train](https://github.com/Algomancer/The-Daily-Train), is spearheaded by `@algomancer` with a call for collaboration and sponsorship.
- `@Jethro Skull` shared an ongoing project regarding the creation of a Microsoft STOP template for Open Interpreter, suggesting further improvements and customization. A related YouTube video was shared [here](https://www.youtube.com/watch?v=kSX9aLPwuwQ).
- A [Twitter post](https://twitter.com/openai/status/1727206187077370115?s=46&t=HUarsCF30BFrz3xURMUMPQ) by OpenAI was shared by `@ufghfigchv`, followed by a request by `@nanobitz` for detailed documentation or a written list related to the post's content.
- Discussions around Pull Requests, their approval process, and automapping to point to local files took place. Users also sought help regarding running a branch in Google Colab and observed issues with tokenization, linked to this [colab](https://colab.research.google.com/drive/1o_fKb-P_2u-QwggQzzQPNaOj6-PkjVTb#scrollTo=3SGgTfikxC-z).
- Concern regarding minimum hardware requirements for running Openchat locally were raised by `@alangr6`, mentioning a GTX 1660 Ti as a possible option.
- An extensive discussion was carried out around FLAN and prompt templating, with `@ufghfigchv` sharing that GPT-4 often performed better without templates. A link to SlimOrca QA pairs embedding was shared [here](https://atlas.nomic.ai/map/c2ebe418-d321-4799-9a51-808f69a80713/2b108714-0517-4e1b-a195-3421b54867ae).
- Proposals to improve maths performance, constructing new prompts, and using ShareGPT for generating instructions were pitched. User `@.benxh` suggested creating a tool like Wolfram Alpha, while `@imonenext` proposed forking FLAN, leading to the [FLAN](https://github.com/OpenOrca/FLAN_OO2) fork, and shared the idea to use ShareGPT to improve LLM's performance.
- A process of manual labeling and categorizing of FLAN v2 tasks was proposed and executed by `@propback`. A corresponding [Google Sheet](https://docs.google.com/spreadsheets/d/1q35zllRHzldFiXa50kJXSRF-0dWp9o0EszBpRQh9dwQ/edit?usp=sharing) was created as part of the initiative.

**Alignment Lab AI Channel Summaries**

### ▷ Channel: [general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841) (3 messages): 

- **Research Project on Text Models**: `@algomancer` is planning to start a research project on training text models on single h100s within a 24-hour timeframe. The goal is to explore new architectural ideas, inspired by audio and vision, to improve raw decoders in text models. Collaboration interest and sponsorship through Github was indicated. `@algomancer` later updated with a link to the project [The-Daily-Train](https://github.com/Algomancer/The-Daily-Train).
- **Microsoft STOP Template for Open Interpreter**: `@Jethro Skull` is working on creating a template for the Microsoft STOP paper and is interested in integrating it into Open Interpreter. They shared a YouTube link [here](https://www.youtube.com/watch?v=kSX9aLPwuwQ) related to the project. Despite it requiring further improvements, the author encourages others to customize and fully automate it for self-improvement in source code. The note warns about containment as the code develops intelligence rapidly.


### ▷ Channel: [oo](https://discord.com/channels/1087862276448595968/1118217717984530553) (2 messages): 

- **OpenAI Twitter Post**: `@ufghfigchv` shared a [Twitter post](https://twitter.com/openai/status/1727206187077370115?s=46&t=HUarsCF30BFrz3xURMUMPQ) from OpenAI.
- **Documentation Request**: User `@nanobitz` expressed interest in a detailed documentation or written list related to the content of the shared Twitter post.


### ▷ Channel: [open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182) (26 messages): 

- **Process to get PRs approved**: User `@jay9265` raised a question about the process for getting Pull Requests approved and shared a link to a PR for the `oo-phi-1_5` model found on [Hugging Face](https://huggingface.co/Open-Orca/oo-phi-1_5/discussions/5).
- **Adjusting Auto Map to Point to Local Files**: `@nanobitz` suggested altering the automap to point to local files, specifying details in a json file.
- **Working with Branches in Colab**: `@jay9265` sought assistance on running a branch in Google Colab for testing; `@nanobitz` proposed cloning the repo to a folder, then pointing to that folder with any tool in use.
- **Issues with Tokenizer**: Post-merge, `@jay9265` noticed differences in the `bos` token and had problems with tokenization. This was brought up in relation to this [colab](https://colab.research.google.com/drive/1o_fKb-P_2u-QwggQzzQPNaOj6-PkjVTb#scrollTo=3SGgTfikxC-z).
- **Merge of Changes**: `@nanobitz` successfully merged the changes after some tests, noting they consulted someone before the merge.


### ▷ Channel: [qa](https://discord.com/channels/1087862276448595968/1147528698669584424) (1 messages): 

- **Minimum Hardware Requirements for running Openchat locally**: User `@alangr6` inquired about the **minimum hardware requirements** to run Openchat locally, mentioning the possibility of using a GTX 1660 Ti.


### ▷ Channel: [oo2](https://discord.com/channels/1087862276448595968/1176548760814375022) (88 messages): 

- **Discussion on FLAN and Templating**: `@ufghfigchv` initiated a discussion about the need to remove some of the prompt templating in FLAN. They observed that models like GPT-4 responded better when the templating was removed, and shared an [Embedding of SlimOrca QA pairs](https://atlas.nomic.ai/map/c2ebe418-d321-4799-9a51-808f69a80713/2b108714-0517-4e1b-a195-3421b54867ae).
- **Plan to Improve Math Performance**: `@.benxh` proposed an approach to improve maths performance, by constructing a tool akin to Wolfram Alpha to generate always correct maths solutions, before teaching each step of the process to the LLM.
- **Forking FLAN and Constructing New Prompts**: `@imonenext` offered to create a fork of FLAN to construct new prompts and improve the LLM's performance. This culminated in the creation of [FLAN](https://github.com/OpenOrca/FLAN_OO2), with `@imonenext` inviting participants to join the OpenOrca Github organisation to collaborate.
- **Proposal to Use ShareGPT for Instruction Generation**: `@imonenext` proposed using few-shot ICL with ShareGPT to generate new instructions based on real-world user instructions. A subsequent proposal was made to modify the system's prompt, invoking a Python interpreter like a function, allowing the model to generate Python code to solve math problems.
- **Categorization of FLAN v2 Tasks**: `@propback` suggested manually labeling and categorizing FLAN v2 tasks, akin to the process described in the Orca 2 paper. This process included creating categories and sub-categories, which was done using a [Google Sheet](https://docs.google.com/spreadsheets/d/1q35zllRHzldFiXa50kJXSRF-0dWp9o0EszBpRQh9dwQ/edit?usp=sharing).


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

**Skunkworks AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1131084849432768614/1131084849906716735) (2 messages): 

- **3B DPO Model Performance**: `@abc98` mentioned a **3B DPO model**, which scores **6.56 on MT-Bench and 80% on AlpacaEval**, and shared the link to the Hugging Face page for the model: [https://huggingface.co/pansophic/rocket-3B](https://huggingface.co/pansophic/rocket-3B)
- `@rifur` acknowledged the information about the DPO model promptly, but didn't provide any additional details.
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Conversation about implementing conditionals in **GPT-4** for intelligent data retrieval, with a proposed method involving the use of specific prompts for web queries and decision-making fields *[general]*.
- Feedback on **GPT-4** and **Claude 2** shared, highlighting their effectiveness and similarity to custom GPTs. Issues with random API failures were also pointed out *[general]*.
- Announcement of future testing with Lindy for handling more complex tasks, promising to share updates on the results *[general]*.
- Discussion on using `system` messages for instructions and `user` messages for incoming inputs in **GPT-4**, suspecting it to be a source of their issues *[gpt4]*.
- Inquiry about the best methods to incorporate tables into **GPT-4** context, with the discussion considering HTML, markdown, and screenshots. Using raw CSV data as a communication method was suggested *[gpt4]*.
- Comparison of Claude AI with GPT-4-128k, with general consensus on impressive results from GPT-4-128k *[claude]*.
- Discussion on the need for AI models to use the entire context, and experiences with document extraction using GPT models while expressing the need for further experiments with claude2.1 *[claude]*.
- Inquiry about tools for processing semi-structured spreadsheets not solely column-centric, but request required further clarification *[resources]*.
- Discussion launch about desirable movie editing AI solutions *[offtopic]*.
- Consensus on the removal of an inactive channel from the Discord guild *[feedback-meta]*.
- Discussion on the events leading to Sam Altman's departure from OpenAI, referencing a Reuters article. Speculation about an AI breakthrough called **Q*** possibly being a mix of Q-Learning and A*. [Reuters Article](https://www.reuters.com/technology/sam-altmans-ouster-openai-was-precipitated-by-letter-board-about-ai-breakthrough-2023-11-22/) *[openai]*.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ Channel: [general](https://discord.com/channels/1168579740391710851/1168579740391710855) (10 messages): 

- **Implementing Conditionals with GPT-4**: User `@alyosha11` asked about implementing conditionals in **GPT-4** to use Google search when the information required is beyond its training cut-off, while preferring to use training data wherever possible.
- **Possible Approach to Query Implementation**: `@jeffreyw128` suggested a method involving a prompt that clearly specifies when to perform a web-query. They proposed using a boolean field for a web-query and another for an answer, prompting so that only one is populated, possibly with another field for decision-making.
- **Testing Metaphor**: `@justahvee` enquired about the progress of testing with Metaphor. 
- **Feedback on GPT-4 and Claude 2 Use**: `@thebaghdaddy` expressed satisfaction with the performance of **GPT-4** and **Claude 2**, describing the models as similar to custom GPTs working together. However, they also noted some API issues causing random function failures.
- **Future Testing with Lindy**: `@thebaghdaddy` mentioned plans to conduct more complex tasks with Lindy later in the week and promised to provide updates on the performance.


### ▷ Channel: [gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641) (11 messages): 

- **system vs user messages**: `@sourya4` discussed their use of `system` messages for all instructions and `user` messages only for incoming messages from users, suspecting this might be a source of their issues.
- **Tables in GPT-4 Context**: `@iyevenko` asked about the best way to incorporate tables in context for GPT-4, with options considered ranging from HTML, markdown, to screenshots.
- **Advice on Incorporating Tables**: `@potrock` recommended using raw CSV data for communicating tables over API, and alternatively suggested using markdown.
- A **link** was shared by `@potrock`, but without additional context provided. [Link here](https://x.com/stevemoraco/status/1727370446788530236?s=46)


### ▷ Channel: [claude](https://discord.com/channels/1168579740391710851/1168582222194933860) (8 messages): 

- **Comparison with gpt-4-128k**: `@potrock` posted a comparison of Claude AI with GPT-4-128k, culminating in an exclamation of "Wild". 
- **Impressive Results with GPT-4-128k**: `@pantsforbirds` has also expressed their positive views based on the good results they are getting from GPT-4-128k.
- **Discussion on Context Use**: `@.psychickoala` asked about the usage of the entire context by AI models.
- **Document Extraction with GPT Models**: `@pantsforbirds` shared that they "frequently run into 32-50k when doing document extraction" but also stated the need to conduct some experiments with claude2.1.
-  Links:
    - [Tweet comparing Claude AI and GPT-4-128K](https://x.com/stevemoraco/status/1727370446788530236?s=46) shared by `@potrock`.


### ▷ Channel: [resources](https://discord.com/channels/1168579740391710851/1168760058822283276) (2 messages): 

- **Tools for Semi-Structured Spreadsheets**: User `@emma9_` inquired about **tools specifically designed to process semi-structured spreadsheets** that aren't solely column-centric. 
- `@thebaghdaddy` requested for more clarification, demonstrating a lack of understanding about the question or details provided.


### ▷ Channel: [offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594) (1 messages): 

- **Movie Editing AI Inquiry**: User `@jmtqjmt` launched a discussion about good movie editing AI solutions.


### ▷ Channel: [feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549) (5 messages): 

- **Discussion on Removing Inactive Channels**: Both `@jmtqjmt` and `@res6969` agreed to **remove channel <#1174092982350270464> due to inactivity**. The action was taken by `@res6969`.


### ▷ Channel: [irl](https://discord.com/channels/1168579740391710851/1171569983688560732) (1 messages): 

- User `@ivanleomk` indicated that he sent a private message but did not disclose any further information or topic of discussion in the public channel.


### ▷ Channel: [openai](https://discord.com/channels/1168579740391710851/1171903046612160632) (3 messages): 

- **Sam Altman's Ouster from OpenAI**: User `@kiingo` shared a [Reuters article](https://www.reuters.com/technology/sam-altmans-ouster-openai-was-precipitated-by-letter-board-about-ai-breakthrough-2023-11-22/) about the circumstances leading to Sam Altman's departure from OpenAI. The discussion centered around an AI breakthrough called Q* and its role in the board's actions.
- **Q-Learning + A***: User `@nosa_` speculated that Q* could be a combination of Q-Learning and A* methodologies.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

**MLOps @Chipro Channel Summaries**

### ▷ Channel: [general-ml](https://discord.com/channels/814557108065534033/828325357102432327) (1 messages): 

- **ML Job Roles and Tools**: User `@sulphatet` sought clarification on where fields like decision science and decision intelligence fit within the overall scope of ML job roles, as presented in an ML interview article on a certain user's website. They also pondered on the importance of tools like Power BI and Tableau in such roles.
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The AI Engineer Foundation Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
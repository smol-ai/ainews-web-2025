---
id: a1ee8502-437a-478e-8745-89dfdc26e01e
title: Is Google's Gemini... legit?
date: '2023-12-06T22:22:18.156000Z'
original_slug: ainews-ai-discords-1262023-9118
description: >-
  **Google's Gemini** AI model generated significant discussion for its
  marketing claims and performance expectations, with skepticism about its
  32-shot chain of thought evaluation and a planned reveal on *December 13th*.
  Meanwhile, **OpenAI's GPT-4** remains a key comparison point, with users
  reporting technical issues and debating subscription waitlists and future
  enhancements.
tags:
  - google
  - openai
  - gemini
  - gemini-pro
  - gemini-ultra
  - gpt-4
  - gpt-3.5
  - claude-2.1
  - palm-2
  - model-evaluation
  - chatbot-errors
  - subscription-issues
  - prompt-engineering
  - context-window
  - multimodal-capabilities
  - ai-competition
  - knowledge-file-management
  - text-to-speech
  - ai-marketing-claims
companies:
  - google
  - openai
models:
  - gemini
  - gemini-pro
  - gemini-ultra
  - gpt-4
  - gpt-3.5
  - claude-2.1
  - palm-2
topics:
  - model-evaluation
  - chatbot-errors
  - subscription-issues
  - prompt-engineering
  - context-window
  - multimodal-capabilities
  - ai-competition
  - knowledge-file-management
  - text-to-speech
  - ai-marketing-claims
---


<!-- buttondown-editor-mode: plaintext -->Hi alpha testers!

That's right, there's now a custom intro for these newsletters. We're very flattered that hundreds of you have somehow found this crappy MVP and so I decided to put in a little last-mile human touch commentary. 

The big news of the day is of course Google Gemini. Multiple discords talking about it - the marketing is great, but people are rightly skeptical. Chief among them is that the central MMLU claim is based on 32-shot chain of thought: ![image.png](https://assets.buttondown.email/images/1974364b-279d-49d2-b9ef-304fb9c2d5ae.png?w=960&fit=max) 


We will know more on Dec 13th.

in other news:

- our bot now attempts to read links dropped by users. So for example, if you see the Latent Space Paper Club, we drop in arxiv links, and the summarizer knows the title and abstract of those papers. same for twitter/youtube/hn etc.
- discord links now head straight to the first message captured, thanks to Sam Ching for feedback.

til tomorrow,

swyx

---


[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Active discussion around **Google's Gemini and OpenAI's GPT-4** models, with emphasis on their capabilities, performance expectations, and comparisons, as spotted in conversations between users like `@solbus`, `@bambooshoots`, `@anurag9249`, `@DawidM`, `@alexblue66`, `@dezuzel`, and others. Context window size of Gemini and the Bard platform's image sourcing capability were featured in the discussions.

- There were several issues reported regarding **OpenAI services**, including chatbot errors, access problems, and file management. Notable issues were with the GPT premium subscription not being recognized, knowledge files disappearing on GPT, and invalid response errors when processing specific requests. Usernames linked with these reports include `@merlinkyle`, `@signate`, `@chepe87`,`@creator320`, `@da.a`, `@coalescamarenus` among others.

- The perceived **superiority of Gemini Pro over GPT-3.5** stirred up market competition discussions. Users showed dissatisfaction with indefinite wait periods for ChatGPT+ subscriptions and deducted usage on failed prompts. Several functional improvements for upcoming GPT versions were suggested, focused on speech and multimodal capabilities, speed, and context handling. 

- Different **prompt engineering techniques** were proposed to enhance AI generated outputs. In-depth discussions were on potential root causes of shortened AI responses. `@iyioio` introduced a new prompt language for GPT models hosted on NPM titled "convo-lang". Inquiries on erratic JSON response behavior and unspecified AI instruction handling were additionally raised and debated. 

- Users shared resources and experiences around **AI model evaluation**. `@tony_07612` provided [link](https://simplyput.ai/evaluating-ai-data-assistants/) to a comparative evaluation of GPT4, Claude 2.1, and PaLM2 while discussing improvements and functionality issues with GPT.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (113 messages): 
        
- **Google's Gemini vs OpenAI's GPT-4**: There was active discussion about Google's newly announced AI model, Gemini, and its comparison with OpenAI's GPT-4. Users `@solbus`, `@bambooshoots`, `@anurag9249`, `@DawidM`, `@dezuzel`, and others explored the topic, highlighting the "Gemini Pro" that is currently live on Google's Bard platform and the forthcoming "Gemini Ultra." 
- **Gemini Performance Expectation**: `@alexblue66` and `@bambooshoots` exchanged views on expectations for Google's Gemini model performance. However, users also emphasized that it's hard to verify claims on unreleased models.
- **Bard and Gemini**: Users `@anurag9249` and `@offline` touches on Bard's capabilities, discussing if Bard can use Gemini for image generation, and clarifying that it only finds existing images from the web.
- **Issues with OpenAI Login**: User `@merlinkyle` sought for assistance with login issues into OpenAI, which `@satanhashtag` helped to troubleshoot.
- **Reporting OpenAI Chatbot Errors**: `@solbus` provided information for `@pipounix.` on how to document errors encountered with the GPT chatbot, sharing specific channels for such reports.
- **Gemini's Context Window**: The context window size of Google's Gemini was another point of concern, with `@anurag9249` and `@kotykd` discussing that Gemini has a 32k context window.
- **AI Data Assistant Evaluation**: User `@tony_07612` shared a [link](https://simplyput.ai/evaluating-ai-data-assistants/) to his comparative evaluation of GPT4, Claude 2.1, and PaLM2 as AI data assistants.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (307 messages🔥): 
        
- **GPT-4 vs Gemini**: Users discussed their experiences with both the OpenAI GPT-4 model and Google's Bard tool powered by the newly-announced Gemini model. Some users expressed that they found Gemini Pro to be superior to GPT-3.5 and close to GPT-4 in its current state. A report was mentioned which indicated Gemini Ultra (not yet accessible for usage) outperformed GPT-4 by 1-2% in several categories.
- **ChatGPT Subscription Waitlist**: Users showed increasing frustration regarding the indefinite wait to subscribe to ChatGPT+. There is ongoing discussion but `@solbus` confirmed that there have been no official announcements regarding when subscriptions will be reopened.
- **Technical Issues with ChatGPT**: Several users reported experiencing technical issues with ChatGPT, including messages about exceeding the prompt limit and receiving errors during conversation. `@picturesonpictures` expressed concern about these ongoing issues, and the apparent deduction of failed prompts from their usage count, which they deemed unacceptable.
- **ChatGPT's Future Enhancements**: Users speculated about improvements that could be implemented in future versions of ChatGPT, including multimodal capabilities in a single model, improved speed, and better context handling. Some users also expressed a desire for GPT-4 to be open to more direct user access, separate from a subscription model.
- **Competition and Market Outlook**: The recent improvements in Google's Bard tool sparked a discussion about the competitive landscape for AI dialogue models. Some users pointed out that competition is ultimately beneficial for the consumers and expected to see more innovations and service improvements as a result. There were also speculations about possible fee structures for Google's upcoming Bard Advanced featuring Gemini Ultra.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (56 messages): 
        
- **Premium GPT Subscription Issues**: There were several complaints about GPT services including `@signate`, `@eejai42`, `@joeycan2`, `@beeman.dev`, and `@yoruiopz`. Discussions revolved around GPT premium subscription not being recognized despite successful payments, network errors, service interruptions, and fatal server errors particularly when attaching specific forms of documents to GPT conversations.

- **Deletion of "Knowledge" Files**: Users like `@chepe87`, `@alienpotus`, and `@tarn0ld` reported their knowledge files disappearing on GPT. Despite uploading their files successfully, these files often disappeared when revisited or after publishing. `@eskcanta` suggested it could be a new bug, advising users to report it to OpenAI support and hoping for a fix.

- **User Verification and Re-access to Services**: `@creator320` and `@nellly1` discussed issues regarding user verification and account suspension, and whether they could still use OpenAI services with their old accounts.

- **Text to Speech Function Inquiry**: Users (`@da.a`, `@ex_hort.`, `@satanhashtag`, `@nachos.ai`) conversed about the existence of a text to speech function. The conclusion was that it exists on mobile and users on other platforms might need plugins.

- **Custom GPT File Limit**: `@coalescamarenus` and `@solbus` acknowledged an issue with GPT not recognizing uploaded files, which may be due to the present cap of 10 files per GPT. The missing file issue might separately be a persistent bug in GPT's file management.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (69 messages): 
        
- **GPT Knowledge File Issues**: Several users, including `@heuehdhdb`, `@chepe87`, `@a2jhagrhbm92awno`, and `@weiwenn` reported issues with uploaded **knowledge files disappearing** after trying to edit their GPTs. `@pietman` noticed that **GPT also ignores his knowledge files** and doesn't perform web browsing or run Python as instructed.
- **GPT Functionality Issues**: User `@borisproductions` expressed dissatisfaction with the **limitations on GPT**, particularly when attempting to get help for an exam question, which was flagged as a violation of OpenAI's terms of service.
- **Vision Use Case Query**: User `@zainsheikh` sought advice on a use case, asking how to **identify and separate common images** using **GPT-4 Vision**. They were redirected to a different channel by `@satanhashtag`.
- **GPT Status**: Users reported **various functionalities of GPT not working properly** despite the official OpenAI status page indicating no current issues. `@pietman` noted the discrepancy, while `@satanhashtag` reported no problems on his end.
- **File Translation**: User `@mysteries_praisethefool` enquired about **GPT's ability to translate files**. No response was provided during the discussion.
- **Return File in GPT Action Response**: User `@jdo300` inquired about the possibility of **returning a file (like a CSV) in the response of an action** for a GPT. No solution was proposed during the discussion.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (31 messages): 
        
- **Tips to Improve Chatbot Responses**: `@27.3degwest` shared ideas about improving a chatbot's output length and relevance. They suggested incorporating brainstorming steps in the prompts and reducing the 'temperature' parameter to cut down on irrelevant outputs.

- **Adding Instructions Shortens Chatbot Responses**: `@tonyaichamp` reported a specific behavior where including additional instructions about the response format seemed to shorten the response length of the AI. They linked to a detailed OpenAI forum discussion about this observation [here](https://community.openai.com/t/json-responses-in-gpt-3-5-turbo-1106-much-shorter-than-without-json/543904/23).

- **List of Reasoning Type Prompts**: `@alienanthony` inquired about other reasoning-type prompt formats apart from "StepByStep", and `@tonyaichamp` suggested the LearnPrompting website as a resource.

- **Developing a New Prompting Language for GPT Models**: `@iyioio` mentioned that they were developing a new prompting language applicable to GPT models, including GPT-3.5, GPT-4, and GPT-vision. They shared that the project "convo-lang" could be found on NPM.

- **Getting Correct JSON Responses from OpenAI API**: `@quantumqueenxox` raised a concern about receiving unstable JSON responses from the OpenAI API. `@ex_hort.` commented that the issue could be due to unclear or incomprehensible instructions.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (31 messages): 
        
- **Model Behavior with Additional Output Instructions**: `@tonyaichamp` observed that adding detailed output formatting instructions seems to result in the AI generating significantly shorter responses, a behavior which was also discussed on the [OpenAI forum](https://community.openai.com/t/json-responses-in-gpt-3-5-turbo-1106-much-shorter-than-without-json/543904/23). 
- **Different Prompt Generation Types**: `@alienanthony` inquired about different prompt generation types for reasoning questions and `@tonyaichamp` referred them to the LearnPrompting website. 
- **New Prompting Language for GPT Models**: Devised by `@iyioio`, a new language for prompting GPT models has been developed and shared on the NPM package manager under the name "convo-lang", with feedback requests open to the community. 
- **Regarding JSON Output Stability**: `@quantumqueenxox` voiced difficulty with getting consistent JSON formatted output from the AI. However, there was no definitive solution shared in the discussion.
- **Improved Prompt Instructions**: `@ex_hort.` offered a method to effectively include the prompt in a conversation without enclosing it in separate brackets.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Detailed discussion around the context and utility of both **Small and Large Language Models**, with conversations addressing size determinants, and performance intricacies; *"...small language models have fewer parameters and are easier to fine-tune for specific tasks"* as mentioned by `@thebaghdaddy`. A valuable educational resource was shared by the same user: an [Introduction to Large Language Models video](https://m.youtube.com/watch?v=zjkBMFhNj_g).
- Excitement and anticipation towards Google's **Gemini AI** Announcement, with a link to the official Gemini AI blog post [here](https://blog.google/technology/ai/google-gemini-ai/). Notably, `@pantsforbirds` expressed hopes about the appropriate balance in Gemini Ultra's safety checks.  
- Important updates and concerns related to GPT-4 Turbo discussed, such as increased rate limits, decreasing latency as noticed by `@res6969`, and reported issues with Chat-GPT latency and the Vision API.
- Sharing of an open-source **Shisa 7B** multilingual model by `@lhl`, meant to perform well in Japanese and English. The model and associated documentation can be found [here](https://huggingface.co/augmxnt/shisa-7b-v1).
- Evaluation practices discussed in the context of long-form question answering by AI systems and the modeling of the fitness function.
- Forward planning for an upcoming event, with participation confirmations, a proposal for involvement from the **Google Gemini** team, and suggestions for hosting *"mini demo days of prompting madness"* during the event.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (15 messages): 
        
- **Understanding Small Language Models**: In response to `@dongdong0755`'s question, `@thebaghdaddy` explained that **small language models** have fewer parameters and are easier to fine-tune for specific tasks. They also mentioned that there have been instances where small models outperformed large ones with specific adjustments and fine-tuning. However, they noted that in their experience, GPT-4 generally outperforms smaller models.
- **Introduction to LLMs Link shared**: `@thebaghdaddy` shared an [Introduction to Large Language Models video](https://m.youtube.com/watch?v=zjkBMFhNj_g) by a scientist at OpenAI.
- **Categorizing Size of Language Models**: `@dongdong0755` and `@thebaghdaddy` discussed whether a language model with 7b parameters (as with Llama2) could be considered small. They concluded that the categorization can depend on the specific task, although generally, models with less than 10b parameters are considered small.
- **Google Gemini AI Announcement**: `@pantsforbirds` shared a [link to Google's Gemini AI announcement](https://blog.google/technology/ai/google-gemini-ai/) and expressed excitement for the API release on the 13th. Other members, including `@wenquai`, `@adeelzaman`, and `@res6969`, also expressed interest and anticipation for the release.
- **Safety Measures for models**: `@pantsforbirds` quoted from the Gemini announcement that significant trust and safety checks were being conducted on **Gemini Ultra** before its release. They expressed the hope that the model would not be overly restricted, as that could make it difficult to use.


### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (10 messages): 
        
- **GPT-4 Turbo Rate Limits Increase**: User `@res6969` noted a significant increase in **GPT-4 Turbo rate limits**, sharing they just got access to 600k TPM on GPT-4 Turbo.
- **Decreasing Latency in GPT-4**: The same user, `@res6969`, also reported a **decrease in latency**.
- **Issues with Chat-GPT Latency**: User `@pantsforbirds` reported experiencing fluctuations and generally **poor latency in the Chat-GPT**. However, they noted a lack of solid benchmark for the same observation in the API.
- **Vision API Issues**: Users `@blakeandersonw` and `@daymanfan` pointed out that the **Vision API was not functioning** for them.


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 
        
- **Multilingual Model - Shisa 7B**: User `@lhl` has publicly released a Japanese and English (JA/EN) bilingual model known as **Shisa 7B**. The aim of the model is to achieve high performance in Japanese language tasks while still retaining substantial capabilities in English. It uses synthetic data and models like [Mistral 7B](https://huggingface.co/augmxnt/shisa-7b-v1). All datasets, methodologies, and code used have been made public for reproducibility. The model can be found on [Hugging Face](https://huggingface.co/augmxnt/shisa-7b-v1).


### ▷ #[eval](https://discord.com/channels/1168579740391710851/1168986849784635553/) (4 messages): 
        
- **Improvements with Context Window**: `@res6969` mentioned they have seen significant improvements by using a context window in their application.
- **Evaluating Generated vs Human Content**: `@pantsforbirds` raised a question about best practices for evaluating the output of a long-form question answered by AI, as compared to human-created content.
- **Fitness Function Modeling**: `@pantsforbirds` also expressed curiosity about modeling the fitness function in AI systems.


### ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/) (6 messages): 
        
- **Attending Event**: User `@jeffreyw128` confirmed his attendance to the event.
- **Number of Attendees**: User `@res6969` shared that the **current number of people attending** the event is 35.
- **Outreach to Google Gemini**: `@res6969` pondered whether they could involve staff from **Google Gemini** in the event.
- **Proposed IRL Event Activity**: User `@frandecam` suggested hosting *"mini demo days of prompting madness"* during IRL events.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion around various Large Language Models (LLMs) such as the open-source **Magicoder** (outperforms ChatGPT on HumanEval+ as per [paper](https://arxiv.org/abs/2312.02120)) introduced by `@swyxio` and Google DeepMind's **Gemini** mentioned by `@aravindputrevu` and `@guardiang`. 
- `@kbal11` highlighted how certain phrases in prose give away the underlying LLM with reference to a [blog post](https://blog.j11y.io/2023-11-22_multifaceted/).
- Exploration of Open Source Software for **text to video** and **image to video** conversion initiated by `@__chef__`.
- Sharing of technology such as [MLX](https://github.com/ml-explore/mlx) (an array framework for Apple silicon) and [AxLearn](https://github.com/apple/axlearn) by `@kevmodrome`.
- Innovative application of GPT for suggesting meal choices based on dietary restrictions by `@philltornroth`.
- Quotes and engagement in the LLM Paper Club sessions hosted by `Kevin Ball`, `Eugene Yan` & `swyx` with explicit reference to the **Emergence paper** discussion ([link](https://arxiv.org/abs/2206.07682)) and the discussion and critique on **Q-Transformer** ([link](https://qtransformer.github.io/)) for using transformers in reinforcement learning.
- Addressed queries and concerns amongst participants regarding meeting access and translation quality in the LLM Paper Club. 


**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (15 messages): 
        
- **Introduction of Magicoder**: `@swyxio` shared a [paper](https://arxiv.org/abs/2312.02120) describing **Magicoder**, a fully open-source Large Language Model (LLM) for code. Key points include: Magicoder models are trained on 75K synthetic instruction data using OSS-Instruct, a novel approach of using open-source code snippets. MagicoderS-CL-7B even surpasses the prominent ChatGPT on HumanEval+ (66.5 vs. 65.9 in pass@1).
- **LLM Prose Style Observation**: `@kbal11` posted a [blog post](https://blog.j11y.io/2023-11-22_multifaceted/) describing how certain phrases reveal the LLM behind the generated prose, giving it a unique 'vibe'.
- **Text to Video Discussion**: `@__chef__` initiated a discussion about Open Source Software models/frameworks for converting text to video, and asked about similar models for converting images to videos.
- **Mention of MLX and AxLearn**: `@kevmodrome` shared links to repositories, [MLX](https://github.com/ml-explore/mlx), an array framework for Apple silicon, and [AxLearn](https://github.com/apple/axlearn).
- **GPT Suggested Meal Choices**: `@philltornroth`shared an interesting application of GPT, using it to suggest meal choices based on food preferences and dietary restrictions by inputting a photo of the menu.
- **Introducing Gemini**: `@aravindputrevu` and `@guardiang` mentioned and discussed **Gemini**, a technology developed by Google DeepMind. The performance of Gemini Ultra was compared to GPT-4V in various reasoning and understanding tasks.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **Emergence Paper Discussion Announcement**: `@swyxio` announced the start of a discussion session starting in 5 minutes on the **Emergence paper**. The paper can be found at [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682) with authors including *Jason Wei*, *Yi Tay*, *Rishi Bommasani*, among others.
- **LLM Paper Club Session**: The discussion is a part of the weekly LLM Paper Club sessions hosted by `Kevin Ball`, `Eugene Yan` & `swyx`. These sessions aim to review LLM papers, especially the foundational ones, breaking them down and discussing their contents. The registration for this multi-session event can be done at [https://lu.ma/llm-paper-club](https://lu.ma/llm-paper-club).


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (11 messages): 
        
- **Q-Transformer Discussion**: `@cakecrusher` shared a link to a paper titled ["Scalable Offline Reinforcement Learning via Autoregressive Q-Functions"](https://qtransformer.github.io/) or **Q-Transformer**. The paper discusses the use of transformers as the q function in reinforcement learning.
- **Meeting Access Issues**: `@iamkrish10` had trouble finding the link to join the meeting. `@swizec` and `@coffeebean6887` assisted him, suggesting it was available in Discord and might need to be found on webui.
- **Translation Quality Concerns**: `@hackgoofer` expressed dissatisfaction with the seamless translation feature, calling the output and input disappointing.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Conversations are revolving around **LangChain Loader** with `@uhaseeb` inquiring `@fatema_08922` about her use of *UnstructuredHTML Loader* for a project.
- An active technical dialogue was seen, discussing **LangChain functions;** from *setting the Instruct Prompt with LangChain* to *merging documents in LangChain*, and visualization methods and tools.
- There was keen interest and issues related to **LangChain Integration with Azure Open AI**, as outlined by `@manojmd.` while detailing his experiences with enterprise data and accuracy issues. 
- Various recommendations and conversations around *Document Storage* were made, notably by `@.dame99` and `@veryboldbagel` discussing the merits of storing variable subject documents in different collections or the same.
- Users shared their work which ranged from a **state-of-the-art chatbot API service** by `@.broodstar`, a **DIY assistant GPT** by `@appstormer_25583`, to `@m1337d` and `@synacktra` respectively sharing information on the embedding of *Lang chain into llm* and a new version of *Hypertion*. Links being shared for better access and understanding:
    - [@appstormer_25583's DIY assistant](https://beta.appstorm.ai/share?url=18c2600b)
    - [@m1337d's tweet about Lang chain and llm integration](https://twitter.com/introsp3ctor/status/1732417785454850060?t=h1XGyGeiCTMbR-6DK3uMRg&s=19)
    - [Hypertion's GitHub repository](https://github.com/synacktraa/hypertion)
    - [Hypertion's Pypi project](https://pypi.org/project/hypertion)

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (19 messages): 
        
- **Use of LangChain Loader for a Project**: User `@uhaseeb` asked `@fatema_08922` about which LangChain loader she was using for her project. She confirmed using the `UnstructuredHTML Loader`.

- **Setting the Instruct Prompt with LangChain**: `@menny9762` queried about setting the instruct prompt with LangChain on using ollama for running a local LLM model. He then suggested a solution himself, which involves using the `chain.call` method and the `handleLLMNewToken` callback.

- **Merging Documents in LangChain**: A discussion led by `@b0otable` revolved around merging documents in LangChain. Initially asking whether there are in-built tools to merge documents after they've been returned as 'docs', the user later clarified wanting to merge documents after already having the 'docs'.

- **LangChain Integration with Azure Open AI**: `@manojmd.` talked about integrating LangChain with Azure Open AI for enterprise data. He also shared issues confronted during the process, including the lack of accuracy in results and requirement of source files of uploaded documents.

- **Tutorial for LangChain Data Ingestion**: `@aaravvv__` asked for a tutorial or document to assist in data ingestion to pinecone instance using LangChain.

- **Alternative AI Engineer Discords**: `@rez0` expressed interest in finding out other AI engineer Discords that community members recommend and use frequently.

- **AI Tools for Document Consultation**: `@bru.leo` sparked a conversation on the best non-coding platforms for consulting information from documents. `@m0xt` came up with a response recommending Jerry's talk on handling embedded tables in PDFs.

- **Visualizing LangChain Processes**: User `@m0xt` asked about the methods/tools, apart from Langsmith, that can be used to visualize chains, steps, variables, schemas, etc. in LangChain.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (2 messages): 
        
- **Document Storage Conversation**: User `@.dame99` mentioned that they have different documents with different subjects and they store them in separate collections. In response, `@veryboldbagel` suggested considering storing the documents in the same collection and using a filter, questioning whether the documents have different schemas or completely different subject areas, and if it would be useful to query across multiple document types at once.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (5 messages): 
        
- **Chatbot API Service**: User `@.broodstar` offered access to a **state-of-the-art chatbot API service** he is developing. The service claims to design conversations that evoke emotions, unfold narratives, and intellectually stimulate users for better user engagement. He offered the service for free to a few interested individuals.
- **DIY Assistant GPT**: `@appstormer_25583` shared a [link](https://beta.appstorm.ai/share?url=18c2600b) to a **DIY assistant GPT** that can analyze home repair images and provide tips.
- **Lang chain and llm**: `@m1337d` shared a [tweet](https://twitter.com/introsp3ctor/status/1732417785454850060?t=h1XGyGeiCTMbR-6DK3uMRg&s=19) indicating that Lang chain can now be embedded into the llm (Llama). This allows the Lang chain code to run directly inside as a callback to the llm.
- **New Version of Hypertion**: `@synacktra` announced a new version of Hypertion with Pydantic model support. He shared the links to the [GitHub repository](https://github.com/synacktraa/hypertion) and the [Pypi project](https://pypi.org/project/hypertion). Hypertion is a Python library for streamlined and efficient LLM function calling.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Discussion on **best practices for logging and use of production to evaluate**, with focus on chatbots and code search. User `@red_shoes_3` seeks advice on how to collect production data to use as training data across different applications.
- Reference to [a tweet](https://twitter.com/SchmidhuberAI/status/1732430359969571014?t=TjxdpzJfRxhUfDwQJZUC4A&s=19) from **Jürgen Schmidhuber** addressing his contributions to deep learning architectures geared towards planning, shared by user `@lightningralf`.
- Community interaction with user `@ajink024` introducing themselves, expressing their participation in the **Open Source Meetup AI Data Source**, along with user `@kainan_e` advocating for a meetup on the process of deploying **AI applications from experiment to large-scale production**.
- Technical discussion centered around **performance benchmarks** with lm-eval harness 4.0 and lower-than-expected [ARC](https://github.com/openai/lm-eval) scores, as well as concerns about **Winogrande + GSM8k** leaderboard results raised by user `@nanobitz`.
- Issues and insights shared by users `@imonenext` and `@benjamin_w` on interfacing **Torch.jit with Flash Attention** and the associated challenges, pointing out an issue with `max_seqlen` type with `torch.jit.trace`, limitations with **PyTorch SDPA's** implementation, and a potential 5% performance improvement with **LayerNorm via Torch.jit**. Relevant links include [this issue](https://github.com/Dao-AILab/flash-attention/issues/701) and a [blog post](https://benjaminwarner.dev/2023/08/16/flash-attention-compile) (note this link is broken).

**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (2 messages): 
        
- **Production Ops for LLMs**: `@red_shoes_3` enquired about best practices for logging and using production to evaluate, specifically in the context of chatbots, code search, etc. They seek insight on how to collect production data that can later be used as training data in different application scenarios.
- **Deep Learning and Planning**: `@lightningralf` shared a [Tweet from Jürgen Schmidhuber](https://twitter.com/SchmidhuberAI/status/1732430359969571014?t=TjxdpzJfRxhUfDwQJZUC4A&s=19), where Schmidhuber presents his contributions to deep learning architectures capable of planning in response to claims from Yann LeCun. Schmidhuber cites numerous publications of his work dating back to 1990 and concludes with the hope that he will someday provide a model of his own.


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (3 messages): 
        
- **Introduction and Community Engagement**: User `@ajink024` introduced themselves as Asante and mentioned their participation in the **Open Source Meetup AI Data Source**.
- **AI Applications Production Meetup Plug**: `@kainan_e` encouraged others to attend a meetup currently being held at 180 Townsend, covering the process of taking AI applications from experiment to large-scale production. The meetup was set to feature a walk-through of a reference architecture aimed at simplifying this process. Detailed information was shared with [this link](https://www.pinecone.io/community/events/pinecone-meetup-sf-taking-ai-apps-to-production/).
- **Late Registration**: `@kainan_e` mentioned that late registration can still be accommodated at the door.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (8 messages): 
        
- **Performance Benchmarks with lm-eval harness 4.0**: User `@nanobitz` sought input after noting lower-than-expected [ARC](https://github.com/openai/lm-eval) scores using the new 4.0 release of lm-eval harness. The results seemed especially lower compared to the ones on HF OpenLLM leaderboard with the 7b chat.
- **Concerns over Winogrande + GSM8k Leaderboard Results**: `@nanobitz` also posted a query about the leaderboard results for **Winogrande + GSM8k** which seemed off.
- **Interfacing Torch.jit with Flash Attention**: `@imonenext` shared a problem they encountered when trying to use Torch.jit with Flash Attention, leading to this [issue](https://github.com/Dao-AILab/flash-attention/issues/701) regarding the incompatibility of `max_seqlen` type with `torch.jit.trace`.
- **Flash Attention 2** Performance with PyTorch SDPA: `@benjamin_w` mentioned that Flash Attention 2 has some limitations when interfaced with PyTorch's SDPA implementation, leading to minor performance losses due to graph breaks on every external CUDA call. Despite this, they shared a link to a [blog post](https://benjaminwarner.dev/2023/08/16/flash-attention-compile) (link broken) discussing scenarios where Flash Attention 2 outperforms PyTorch's SDPA implementation.
- **LayerNorm Speed Improvement with Torch.jit**: `@imonenext` gave the input that using jit with LayerNorm can boost performance by up to 5%.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

- User `@erisianrite` is currently **exploring Weakly Supervised Semantic Segmentation techniques, Segment Anything**, and **traditional CNN architectures** for a microscopy task. Recommendations were requested to understand the current trends in these topics.

**MLOps @Chipro Channel Summaries**

### ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/) (1 messages): 
        
twenzy03: Hi


### ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/) (1 messages): 
        
- **Exploring Weakly Supervised Semantic Segmentation Techniques**: `@erisianrite` is working on a project evaluating Segment Anything, weakly supervised semantic segmentation (WSSS) techniques, and traditional CNN architectures for a microscopy task. They are looking for recommendations to get up to speed with the current state of the art in WSSS techniques.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

rfhuang: rumor mill is talking of gpt-5 to have video understanding
        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Free Scrimba Subscription Request**: User `@vince_uc` expressed gratitude to `<@500607885714128897>` for a free scrimba subscription, mentioning that they had used a free version on YouTube and wished for the same for this one as well.
        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Discussion on AWS Aurora Limitless**: User `@rafaelnadal` brought up a discussion about **AWS Aurora Limitless**, AWS's new competitor to Yugabyte/Cockroach. The user wondered why AWS launched its product later than its competitors (Yugabyte and Cockroach started in 2015/2016) and if this indicates a large market for distributed/Active-Active, ACID databases.
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
---
id: f57c70e2-cab0-4c52-b940-e3d98d2def04
title: '12/22/2023: Anyscale''s Benchmark Criticisms'
date: '2023-12-23T01:16:52.465251Z'
original_slug: ainews-12222023-anyscales-benchmark-criticisms
description: >-
  **Anyscale** launched their **LLMPerf leaderboard** to benchmark LLM inference
  performance but faced criticism regarding metrics transparency like cost per
  token and throughput, and the validity of comparing public LLM endpoints. In
  **OpenAI Discord** discussions, users reported usability issues with **Bard**
  and preferred **MSFT Copilot** for storytelling, noting fewer hallucinations.
  The community debated the value of upgrading from **GPT-3.5** to **GPT-4**,
  with many finding paid AI models worthwhile for coding productivity. Bugs and
  performance issues with **OpenAI API** were highlighted, including slow
  responses and message limits. Future AI developments like **GPT-6** and
  concerns about OpenAI's transparency and profitability were also discussed.
  Prompt engineering for image generation was another key topic, emphasizing
  clear positive prompts and the desire for negative prompts.
companies:
  - anyscale
  - openai
  - microsoft
models:
  - gpt-4
  - gpt-3.5
  - bard
  - msft-copilot
topics:
  - benchmarking
  - performance
  - prompt-engineering
  - api
  - bug-tracking
  - model-comparison
  - productivity
  - coding
  - storytelling
  - hallucinations
  - future-ai
people: []
---


<!-- buttondown-editor-mode: plaintext -->Following up from [their initial work](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference) in November, [Anyscale announced their LLMPerf leaderboard](https://twitter.com/anyscalecompute/status/1737883193720922413), where they of course come out looking great, but it has come under intense scrutiny:

 ![image.png](https://assets.buttondown.email/images/2861e289-511f-4ccb-990d-dd2edca2846a.png?w=960&fit=max) 

Criticisms:

- Need to expose [cost per token, throughput, and windowed versions not just burst](https://twitter.com/soumithchintala/status/1738241213327692174)
- [Don't compare public LLM endpoints](https://twitter.com/dzhulgakov/status/1737917306565697990?s=46) because batching/load/timing really matter
- [straight up contested by Vipul of Together](https://twitter.com/vipulved/status/1738075362448527805?s=46)
- in ML history [it hasn't worked to overoptimize](https://x.com/jiayq/status/1738014510336909397?s=20) for one thing. 


[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Performance and Usability of AI Models**: Discussion under [#ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) regarding user experience with different AI models. Issues reported with **Bard** and a preference expressed for **MSFT Copilot**, particularly in storytelling tasks. *"`@muyfashionista` shared their testing of multiple AI models and mentioned issues with **Bard**"*. Also, the distinction between Bing Chat and Microsoft Copilot was clarified.
- **GPT-4 and Its Expected Value**: Questions raised about the worth of upgrading to **GPT-4** and if the pricing for AI models was justified. According to discussions, many in the community found the productivity boost from models like ChatGPT and Copilot worth the cost for coding work.
- **AI-related Challenges and Bugs**: Across various channels, a range of issues and potential bugs were discussed. From api-discussions, performance issues with GPT and OpenAI API were noted. User `@bmaxhacks` noted API responses taking more than 2.5 seconds. Problems with models not responding well to custom formats, GPTs disappearing, and a potential bug allowing 80 messages per hour on GPT-4 were all pointed out. Exchanges explored AI model performance changes with different programming languages.
- **Future AI Outlook and Discussion**: On #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/), users speculated about future AI developments, like possible **GPT-6** release and potential impacts on the economy. Also, voiced concerns about OpenAI's transparency, profitability, and service investments.
- **Prompt Engineering for Image Creation**: Conversation in #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) around guiding AI's visual outputs and controlling what appears in generated images. User `@eskcanta` shared advice on maximizing output quality, emphasizing the value of clear, detailed prompts and avoiding negative instructions. User `@suptrox` desired negative prompts to guide the AI model.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (50 messages🔥): 
        
- **Performance and Usability of AI Models**: `@muyfashionista` shared their testing of multiple AI models and mentioned issues with **Bard**, including **making mistakes** and either going on indefinitely or not returning responses. They found **MSFT Copilot** a better choice for storytelling and found fewer hallucinations compared to Bard. 
- **Difference Between Microsoft Bing and Microsoft Copilot**: In light of `@eljajasoriginal`'s query, `@muyfashionista` clarified that **Microsoft has rebranded Bing Chat to Microsoft Copilot**. They also pointed out a difference between the MSFT Copilot for Microsoft 365 version and the one found in Bing Chat.
- **Upgrade from GPT 3.5 to GPT 4**: In response to `@6nueve`'s query, `@elektronisade` and `@lugui` suggested that **upgrading to GPT4 could be useful**, especially for tasks like coding. They cautioned, however, that the productivity of the AI model also heavily depends on the language used, the project type, and the quality of the prompt.
- **Value of Paid AI Models for Work**: `@6nueve` wondered about the worth of paying for AI models, which led to a discussion with `@lugui`, `@afterst0rm`, and others. The general view was that **the productivity boost from using AI models like ChatGPT and Copilot justifies the cost**, provided the user does coding for work. They found it enhanced their productivity and made work more enjoyable.
- **AI Model Performance with Different Programming Languages**: `@afterst0rm` and `@lugui` discussed that **the AI models' performance changes with programming languages**. They found models to be better at mainstream languages like Python, Java, and JS, while newer languages like Rust gave less satisfactory results. The AI models could even generate boilerplate code in Python but struggled with newer syntax and patterns.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (165 messages🔥🔥): 
        
- **Discussion on GPT-V Performance**: User `@pudochu` expressed considerable frustration with **GPT-V**, particularly its speed and vision capabilities, asserting that it is not fit for purpose in its current state. User `@captainsonic`, however, suggested that successful use of GPT-V may require more prompt engineering and customization for specific use cases. 
- **GPT-4-Turbo Usage**: There were several discussions around GPT-4-Turbo's performance and cost effectiveness. Users such as `@beowulfbr` and `@feltsteam0` analyzed the cost per token of GPT-4-Turbo and its alternatives, noting that the price varies depending on use case and the volume of tokens processed. 
- **Concerns about OpenAI Transparency and Profit Utilization**: `@mischievouscow` and `@feltsteam0` had a back-and-forth conversation about how OpenAI might be using its profits. There was speculation about investing in NVIDIA GPUs and an assertion that the costs of training and inference would take up a significant portion of revenues.
- **Speculation on Future AI Development**: The conversation includes speculations about when **GPT-6** will be available, with comedic references to the awaited release of GTA 6. There were also discussions on the potential economic outputs resulting from AI advancements, with `@samaltman` suggesting that AI might increase economic output by 20-30x.
- **Platform and Feature Issues**: Several users reported experiencing issues with the OpenAI platform. `@kingofcringe` reported missing the stop generation button while `@cozy_artist` had issues generating a file to download. `@jaicraft`, however, discovered a feature where text from a response can be selected and replied to.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (57 messages🔥🔥): 
        
- **Issues with GPT and Custom Formats**: User `@ritalin60mg` raised a concern about GPT-3.5 not responding well when asked to reply in a custom format compared to GPT-4. They are seeking a standard way to instruct it that minimizes character usage.
- **Managing Threads and Deletion**: `@yummisery` was unclear about the deletion of threads and whether they were automatically deleted after inactivity. Responding to this, `@brokearsebillionaire` clarified that while runs time out, threads do not and it's the user's responsibility to clean them up.
- **ChatGPT Behavior and Troubleshooting**: `@princesslunarcat` reported issues with ChatGPT, shifting from GPT 4 to GPT 3.5 after one message. `@solbus` offered various troubleshooting suggestions, but the issue persisted. The problem was then reported to the OpenAI support.
- **Performance Issues with OpenAI API**: `@bmaxhacks` raised a concern about the OpenAI API taking more than 2.5 seconds for most responses, despite a small amount of data being sent.
- **Incorporation of External APIs and Iteration Issue**: `@yummisery` asked whether assistants' code interpreter tool could call external API endpoints. They further inquired how to handle iteration involving multiple API calls in the context of a function.
- **Repetition Control for User Inputs**: `@solononforever` sought advice on pseudocode for tracking conversation history to prevent user repetition using prompts or langchain, to which `@snowmobile` responded with a python code suggestion. 
- **Function Calling in LangChain**: `@bluehipp0` asked why there are no open source LLMs that provide function calling and sought examples of using LangChain to simulate function calling.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (50 messages🔥): 
        
- **GPTs Disappearing Glitch**: There were numerous reports of GPTs and chats disappearing, often returning later. `@csharp69`, `@redash999`, and others experienced this issue. Specifically, `@redash999` mentioned a recently created GPT called "Water Engineer", which did not reappear.

- **Confusion over GPT4.5 Turbo Pricing**: `@killymbapps` expressed confusion over how pricing works for GPT4.5 Turbo for GPT assistants, noting a lack of clear information.

- **Stock Data Analysis GPT**: `@emperor_azir` shared a [link](https://chat.openai.com/g/g-jC8FvZ9SW-stock-data-analysis-live-financial-data) to a GPT that provides live financial data and technical analysis.

- **Potential Bug**: `@raidedcluster` discovered a potential bug that allows them to get 80 messages per hour with GPT-4, instead of the limit.

- **Potential Artistic Applications for GPTs**: `@fortun8te`, `@jamiecropley`, and others discussed the potential for GPTs to analyze and understand art styles based on image inputs. They expressed hope for future improvements in this area. `@elektronisade` stated current GPTs cannot grasp styles beyond rough categories.

- **Restrictive GPT Names**: `@wndapnda02` reported an issue where GPTs with verified domain names were restricted for public sharing, suggesting possible trademark violations. 

- **Problem with Knowledge Base Analysis**: `@bacaxnot` complained about custom GPTs now preferentially using Code Interpreter over Retrieval for analyzing their knowledge base, leading to slower and less accurate responses.

- **ChatGpt Rumor**: `@chemo2493` stated that GPT 4.5 Turbo was debunked as a rumor.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (7 messages): 
        
- **Creating Specific AI Visualizations**: User `@suptrox` expressed a wish to prompt a DALL-E style model to visual an expansive garden and **nothing else**. They noted having trouble with unwanted elements appearing in the image and expressed a desire for *'negative prompts'* to guide the model.
- **Guiding AI Visual Output**: Responding to `@suptrox`, `@eskcanta` shared an approach to guiding AI visual outputs. They emphasized focusing on positive descriptions of what should be in the image and avoiding negative instructions. They also reiterated the importance of precise and detailed prompts.
- **Image Creation Example**: `@eskcanta` provided a concrete example, using a prompt to create four images of a detailed, unspoiled, and endless garden. It garnered positive reactions from other users.
- **Feedback on Approach**: `@catking335` reacted positively to the images produced, asking `@eskcanta` about the creation process. `@eskcanta` reiterated the value of clear, detailed, and positive prompts in guiding ChatGPT-4.
- **Appreciation of Technique**: `@bambooshoots` also appreciated the technique demonstrated by `@eskcanta`, noting it sparked some ideas. They expressed excited anticipation for exploring it further.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (7 messages): 
        
- **Image Generation with DALLE**: User `@suptrox` wanted to generate an image that only showcases a garden landscape, and wondered if DALLE supports negative prompts to suppress unwanted elements. 
- `@eskcanta` responded by advising that AI generally performs better with positive rather than negative instructions. They demonstrated this by generating a series of four images representing an "endless natural, unspoiled, pristine garden" using the ChatGPT-4 model. Their advice for getting the desired output was to accurately describe what one wants, review the output, and iteratively refine the instructions based on the observed discrepancies.
- Several users, such as `@catking335` and `@bambooshoots`, expressed admiration for the generated image. `@catking335` asked about the creation process, to which `@eskcanta` detailed the prompt they used and recommended techniques to enhance the model's output, such as having a more detailed conversation with the model to fine-tune the desired output.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- In-depth discussion on AI **model competition**, with focus shifting towards small multimodal models, small language models, distributed learning models and cheap fine-tuning models, as suggested by `@red_code`.
- Extensive exchange regarding **Open Source Front End** for Mistral AI akin to OpenAI's playground, the use of **litellm with various models**, how to measure **token count in Mistral output**, and training Mistral with unique datasets. A link to litellm [config.yaml](https://github.com/brokearsebillionaire/llmcord/commit/76114cc67cd70c35caba7081c0f2fa61bf0213e8) and vLLM issue on [Autogen](https://github.com/microsoft/autogen/issues/1037) was shared. User `@rrross` also followed up with a [blog post](https://bit.ly/count-mistral-tokens) on how to count Mistral tokens. 
- Detailed subject matter on model capabilities, comparison and use-cases. This includes selection of a **suitable model for English to Norwegian translations**, application of **Mistral in the realm of coding and retrieving Linux commands**, and the **performance** of a 7B model with a 3090 GPU.
- Showcased the versatility of Mistral AI with examples including hosting **Mistral Models on own Servers**, a bot compatible with Mistral API - [**llmcord**](https://github.com/jakobdylanc/llmcord), a new third-party package using Mistral.AI's Web API [here](https://raku.land/zef:antononcube/WWW::MistralAI), and a [Twitter post](https://x.com/gneubig/status/1738338237498921251?s=46) being shared.
- Exchanged miscellaneous discussions on the usage of **r/MistralAI subreddit**, possibility of using **Mistral on Android**, running **AI models on phones**, with a [GitHub link](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui) being shared about getting models to run on iPhones.
- Delved into issues faced by users on the **Mistral platform**, including the erratic responses from `mistral-small`, payment issues for users in **India**, query on the existence of a **bug bounty program**, lack of clarity on the **Embeddings endpoint request limit** and how to obtain an **API access invite**.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (70 messages🔥🔥): 
        
- **Future Competition of Models**: User `@red_code` predicts that soon the competition would focus on small multimodal models or small language models, distributed learning models, and cheap fine-tuning models.
- **Open Source Front End**: User `@dv8s` inquired about an open-source or publicly available front end for Mistral AI, akin to OpenAI's playground.
- **Use of Litellm with Models**: A discussion between `@cyb3rward0g` and `@brokearsebillionaire` regarding the use of litellm with various models, specifically how it interacts with instruct fine-tuned Mistral models. They shared links to litellm [config.yaml](https://github.com/brokearsebillionaire/llmcord/commit/76114cc67cd70c35caba7081c0f2fa61bf0213e8) and vLLM issue on [Autogen](https://github.com/microsoft/autogen/issues/1037).
- **Token Count in Mistral Output**: `@rrross` queried how to measure the token count in a Mistral output. `@sublimatorniq` suggested using a library similar to OpenAI's. `@rrross` followed up with a [blog post](https://bit.ly/count-mistral-tokens) on how it can be done. `@daain` also suggested an online tokenizer tool.
- **Training Mistral with Unique Datasets**: `@novemberfalls` mentioned an interest in training the Mistral model with a unique dataset. `@antononcube` asked for clarification on the dataset structure and intended use, with potential recommendations depending on the dataset attributes.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (78 messages🔥🔥): 
        
- **GPU Support for Larger Models**: `@novemberfalls` asked if a 3090 GPU would support the 7B model, to which `@mrobino` confirmed that it would, given Quantized Yes.
- **Queries on Model Capabilities**: `@christiansieb` asked for more examples in the documentation as they've faced issues with context relevance in both **Mistral** and **Mixtral** in comparison to **ChatGPT**.
- **Discussion on Translation Models**: `@ved_ikke` asked for model recommendations for English to Norwegian translation work. Various models like **gpt3.5**, **Helsinki NLP**, **Mixtral**, **Yi**, and **GPT4** were discussed by users, including their experiences and preferences. `@laoji_lcg` expressed satisfaction with **GPT3.5** for translating English into Chinese.
- **Mistral for Programming**: Several users, including `@ved_ikke` and `@laoji_lcg`, had a discussion on **Mixtral**'s utility in the realm of coding. They agreed it was excellent in understanding and debugging code, but `@laoji_lcg` believed its output tended to be complex compared to streamlined code from **GPT4**.
- **Chatbot for Linux Command Retrieval**: `@giblaz` asked for model recommendations for retrieving Linux commands. `@dutchellie` suggested trying **Mixtral**.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (106 messages🔥🔥): 
        
- **Discussion about the performance of different models on different hardware**: `@dutchellie` and `@sublimatorniq` discussed the performance of the `llama.cpp` backend, which has been built into the *exllamav2* model, and noted that it gives considerable gains in performance. Dutchellie shared their experiences of using exllamav2 on an AMD GPU, and its improvement from 6t/s with llama.cpp to 50t/s ([source](https://github.com/turboderp/exllamav2)).
- **Difficulties with Mac and AMD hardware**: They noted the absence of Mac support in popular projects and mentioned that users with these systems are always chasing after the tech, though they agreed that Macs perform significantly well for large models.
- **Discussion around `exllama` and `ollama`**: The discussion skewed towards the comparison between `exllama` and `ollama`, with dutchellie sharing that `exllama` appeared to be outperforming `ollama` in their personal tests. `@sublimatorniq` raised a known issue with "`mixtral` on mac" that has been reported for long delays and slowness ([source](https://github.com/jmorganca/ollama/issues/1556)).
- **LLM models and fine-tuning**: There was also a mention of the recent release of *Dolphin 2.6 Mixtral* by Eric Hartford, and a jesting mention of the addition of 'samantha-based empathy data' in the new release.
- **Download speed issues with Huggingface**: Lastly, they expressed frustration with the download speeds from Huggingface, with sublimatorniq noting particular difficulty connecting to Huggingface from Cambodia.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (9 messages🔥): 
        
- **Hosting Mistral Models on own Servers**: `@ds_gamer` reported that they have hosted Mistral models on their own servers including other OS AI models and offer an OpenAI-compatible API access for free. However, they mentioned that except for Mixtral model which is currently a paid-only service, due to high computation costs associated with it.

- **Introduction to Open Source Bot llmcord**: `@jakobdylanc` shared an open-source bot named [**llmcord**](https://github.com/jakobdylanc/llmcord) which is compatible with the Mistral API and allows multiplayer chat with Large Language Models (LLMs).

- **Use of gpt-4-vision-preview in llmcord**: `@jakobdylanc` confirmed to `@antononcube` that gpt-4-vision-preview can be used with the bot llmcord. This bot supports both OpenAI and Mistral API.

- **Third Party Package using Mistral AI's Web API**: `@antononcube` shared a link to a new third-party package using Mistral.AI's Web API which can be accessed [here.](https://raku.land/zef:antononcube/WWW::MistralAI)

- **Tweet Link**: `@mrobino` shared a [link](https://x.com/gneubig/status/1738338237498921251?s=46) to a tweet without any additional context.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (12 messages🔥): 
        
- **Usage of r/MistralAI subreddit**: User `@anonboxis` questioned whether anyone in the channel uses the **r/MistralAI subreddit**.
- **Use of Mistral on Android**: User `@scottsilverstein` asked if it's possible to use **Mistral on an Android phone**. `@akshay_1` responded that it cannot run locally yet.
- **Running AI Models on Phones**: `@sublimatorniq` shared a [GitHub link](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui) about getting some models to run on iPhones. Further, `@rtyax` mentioned that **Mistral can run on a phone with 12GB memory**, citing that **koboldcpp and llama.cpp work with Termux**. However, they also noted that **it'd be slow on a phone with 8GB memory**.
- **Autopoietic Cognitive Architecture Paper**: User `@poltronsuperstar` shared that they drafted a paper on **autopoietic cognitive architecture**, with an amusing note on the acronym nearing "CACA", meaning poop in French. They noted it as an attempt at **AGI**.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (42 messages🔥): 
        
- **Issues with Mistral Endpoint Responses**: User `@sublimatorniq` raised an issue about receiving chopped and overly short responses, sometimes stopping unexpectedly, from the `mistral-small` model, even after setting `max_tokens: 1024`. The `finish_reason` reported is 'stop'. This issue occurs regardless of context length and seems to arise frequently when the model is early into its response. `@lerela` acknowledged the issue and promised an investigation.
- **Payment Issues from India**: `@tafferboy` is experiencing payment method failures for Mistral from India due to restrictions against recurring charges. A manual payment or wallet top-up feature was suggested as potential workarounds. `@lerela` acknowledged the issue and stated they are looking into options.
- **Bug Bounty Program Query**: `@poltronsuperstar` asked about the existence of a bug bounty program and whether finding a vulnerability on the platform would aid in securing a developer role. There was no explicit response to this query.
- **Embeddings Endpoint Request Limit**: `@lautaro.p` asked about the request limit for the Embeddings endpoint but didn't receive a response.
- **API Access**: `@aurelien6964` inquired how one could get an invite for the API but did not receive a response.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- `@nonameusr` introduced **LUNA-SOLARkrautLM-Instruct** on Hugging Face: [Hugging Face Link](https://huggingface.co/fblgit/LUNA-SOLARkrautLM-Instruct).
- `.beowulfbr` shared an academic paper on code translation with LLMs: [arXiv Paper](https://arxiv.org/abs/2308.03109).
- Training Mixtral and other model variants was a focus of conversation, with `@wesh3104` and `@ldj` discussing the fine-tuning of **OpenHermes-Mixtral-8x7B**.
- `@fullstack6209` initiated the talk about the best performing models in the 7b-13b, 4-8bit range. `@teknium` suggested either the Hermes or Openchat models.

Interesting links were shared and discussed within the guild: 

- `@asada.shinon` mentioned concerns about Opera, citing user data selling and other dubious practices, sharing a detailed report: [Rentry Report](https://rentry.org/operagx).
- `@emrgnt_cmplxty` shared a Twitter link of a project named AgentSearch, aiming to enhance knowledge accessibility for LLM agents: [Project on Twitter](https://twitter.com/ocolegro/status/1737899295573991452).
- Talk on the cost implications of utilizing various models, with `.beowulfbr` discussing the expense of using `gpt-4-turbo` for handling 1,000,000 tokens.
- Feedback from `@night_w0lf` on Gemini Pro/Bard's performance in Python coding, acknowledging Gemini Pro's superior performance over GPT4 and mentioning an upcoming version of ChatbotUI.
- Discussion on the creative nomenclature of some models, specifically, Meow and Sauerkraut SOLAR, initiated by `@fullstack6209`, `@gabriel_syme`, and `@beowulfbr`.


**Nous Research AI Channel Summaries**

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (9 messages🔥): 
        
- `@nonameusr` shared a tweet from Charlie B. Holtz: [Twitter Link](https://vxtwitter.com/charliebholtz/status/1737667912784134344)
- `@asada.shinon` raised concerns on Opera (GX/One/Crypto Browser) due to its practices, citing that the company is known for selling user-data, censorship, backdoored software, and involvement in predatory loans and anti-competitive behavior. Contact details along with a link for more details were shared: [@0VERIMAGINE Twitter](https://twitter.com/0VERIMAGINE) and [Rentry Report](https://rentry.org/operagx)
- `@metaldragon01` shared a tweet from _akhaliq: [Twitter Link](https://fxtwitter.com/_akhaliq/status/1738050817100325354)
- `@.beowulfbr` discussed the cost difference between `gpt-4-turbo` and other models such as CursorAI and ChatGPT. Notably, they found `gpt-4-turbo` more expensive for handling 1,000,000 tokens.
- `@nonameusr` introduced **LUNA-SOLARkrautLM-Instruct** – a UNA-Sauerkraut variant of the powerful Upstage, shared through a Hugging Face link: [Hugging Face Link](https://huggingface.co/fblgit/LUNA-SOLARkrautLM-Instruct)
- `@night_w0lf` provided feedback on Gemini Pro/Bard's performance in coding, particularly Python. They expressed satisfaction with Gemini Pro's performance over GPT4, and shared news about a forthcoming version of ChatbotUI supporting many API providers and local models via Ollama.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (133 messages🔥🔥): 
        
- **Discussion on Training Mixtral and Other Models**: There were substantial discussions about the training of Mixtral and different models. `@wesh3104` mentioned their preference for Medium as an alternative to the paid use of their LLM (programming assistant). 
- **OpenHermes-Mixtral-8x7B Fine-tune**: `@wesh3104` and `@ldj` discussed the OpenHermes-Mixtral-8x7B fine-tune posted on Hugging Face. `@ldj` clarified that you could train on a single node.
- **Litmus Tests for LLMs**: `@n8programs` detailed a litmus test for sentence generation, which is apparently challenging for all but a few high-level language models. They had a discussion with `@nonameusr` on this topic.
- **Code Translation with LLMs**: `.beowulfbr` shared an [interesting paper](https://arxiv.org/abs/2308.03109) about how large language models (LLMs) perform when translating code from one programming language to another.
- **AgentSearch, an Open-Core Effort**: `@emrgnt_cmplxty` shared a link to a [project on Twitter](https://twitter.com/ocolegro/status/1737899295573991452) called AgentSearch which works towards making humanity's knowledge accessible for LLM agents. Multiple users provided feedback on it, discussing its merits and areas for improvement.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (6 messages): 
        
- **Discussion about Best Models in the 7b-13b Range**: User `@fullstack6209` inquired about the best performing models in the 7b-13b, 4-8bit range. `@teknium` suggested either the **Hermes** or **Openchat** models, or any of their merges.
- **Nomenclature of Models**: Users `@fullstack6209`, `@gabriel_syme`, and `@beowulfbr` engaged in a discussion about the eccentric names of models, particularly **Meow** and **Sauerkraut SOLAR**.
- **Validation of Openchat's Performance**: User `@beowulfbr` confirmed the good performance of the **Openchat** model.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **HuggingFace model issues and applications**: Community addressed varying drops in connections when using HuggingFace, potential usage of models like `T5ForConditionalGeneration` and `Seq2SeqTrainer` for various tasks, utilizing HuggingFace resources for learning, and issues encountered when uploading models via Git Bash. Users shared advice on running LLM on lower-spec PCs, generating multiple-choice questions using AutoTrain with Seq2Seq, and the cost comparison between GPT-4 and fine-tuning an open-source model. A query was also raised about the housing of data on HuggingFace. 
- **Tools and resources**: Links were shared to coding chatbots like [Bing Chat](https://github.com/janhq/jan), the [Huggingface NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1), recent projects [Tweak Mpl Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat) and [Cheshire-Cat](https://github.com/cheshire-cat-ai), live competitive programming contest [Fall Challenge 2023](https://www.codingame.com/contests/fall-challenge-2023), and an [explanation of Virtual Sensors](https://www.iec.ch/blog/untapped-potential-virtual-sensors) by IEC. 
- **User-created projects**: Includes a model for TOEIC reading part 5 generation, a **ppo** agent that plays **Huggy** trained using the Unity ML-Agents Library, a rapid model named **Mixtral-8x7B-Instruct-Demo**, a blog post on generating music as MIDI, a tutorial on evaluation metrics for regression models, and a presentation on Ensemble Learning recorded on [YouTube](https://youtu.be/RCtnCMVYsKw). A new test model was also mentioned but without additional details.
- **Discussion on computer vision and NLP applications**: Debates included the possible use of diffusion models for separating superimposed images and models for converting images into 3D objects. A query was made about combining prompt tuning and LORA in one shot. The use of LayoutLM with the RVL-CDIP dataset and the challenges of applying LLAMA-2 to a height comparison task were discussed. Users also sought help for integrating GPT models with Gradio UI in Google Colab.


**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (37 messages🔥): 
        
- **Issues with HuggingFace Models and Dropbox**: `@gokiburigeemu` queried about if HuggingFace uses Dropbox for housing data and if there is a limit to connections. They mentioned an issue of too many connections with HuggingFace models.
- **Re-generation Commands in mistral orca 7b for Q/A generation**: `@enka55` is seeking a way to continue generating Q/A after the model stops, despite setting the `max_new_tokens` to 16K. Outlining the similar case with ooba textgen, they stated it generates a few Q/A and then stops unless generating is manually continued.
- **Modifying T5ForConditionalGeneration and Seq2SeqTrainer for Training**: `@ianbestfellow` is looking for guidance on how to modify the forward part of `T5ForConditionalGeneration` for training and Seq2SeqTrainer. 
- **Quantizing and Upload Model to Hugging Face Issues**: `@sumit.sumit` is encountering an AttributeError when trying to quantize a model in Collab and upload it to Hugging Face, both with the `facebook/opt-350m` model and `vilsonrodrigues/falcon-7b-instruct-sharded` model.
- **Running LLM models on Low-End PCs**: `@melvjn_03` pondered if LLM models can run on low-end PCs, to which `@lawls.net` clarified they can, especially with quantized GGUF files. They also mentioned that it's possible to run the model equivalent of ChatGPT 3.5-turbo on a local gaming PC.
- **Creating Multiple Choice Questions with AutoTrain and Seq2Seq**: `@robolicious` is considering using AutoTrain with Seq2Seq for a specific use-case geared towards creating multiple choice questions based on some examples and a specific level of difficulty. `@doctorpangloss` suggested trying 30 shot generation before opting for fine tuning or training in gpt4.
- **Bing Chat and Jan as Coding Chatbots**: `@melvjn_03` expressed interest in trying out coding chatbots, to which `@lawls.net` suggested Bing Chat and shared a [link to Jan](https://github.com/janhq/jan), an open-source alternative to ChatGPT that runs offline.
- **GPT-4 vs Fine Tuning Cost Comparison**: `@robolicious` requested a source that compares the cost of using GPT-4 vs fine tuning an open-source model.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (9 messages🔥): 
        
- **Huggingface NLP Course**: `@llmsherpa` shared a link to the Huggingface NLP Course [here](https://huggingface.co/learn/nlp-course/chapter1/1).
- **Issue with Amazon-Reviews-Multi Dataset**: `@vatsal2161` pointed out an issue with the *amazon-reviews-multi* dataset from the Huggingface NLP course, stating that the dataset is now defunct.
- **Learning Methods for New NLP Topics**: `@regi6135` seeks guidance on how to learn about recently emerging NLP topics, requesting for pointers or links for a better understanding.
- **Progress Update by Neuralink**: `@neuralink` shared an update of their learning progress, mentioning their work on **DoReMi**, **end-to-end FP8** training in 3D Parallleism, and other relevant subjects.
- **Issues Uploading to Huggingface via Git Bash**: `@deadsg` reported a problem uploading to Huggingface from Git Bash and requested help.
- **Discussion on Note-Taking Format**: `@onceabeginner` mentioned being inspired by `@neuralink`'s note-taking format and discussed wanting a deeper understanding of specific topics.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (6 messages): 
        
- **Panel Tweak Chat**: User `@andysingal` shared a link to a project called [Tweak Mpl Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat), also mentioning that it has been duplicated from [ahuang11/panel-fleet](/spaces/ahuang11/panel-fleet).
- **Modern NLP Repos**: `@andysingal` also mentioned about adding [Panel Tweak Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat) project to a list of modern NLP repositories on [Github](https://github.com/andysingal/modern_nlp_2/blob/main/awesome-repos.md).
- **Cheshire-Cat**: `@nickprock` shared a project called [Cheshire-Cat](https://github.com/cheshire-cat-ai), a framework to develop AI assistants which supports Hugging Face models.
- **Fall Challenge 2023**: `@lustforserotonin` posted a link to a live competition on CodinGame named [Fall Challenge 2023](https://www.codingame.com/contests/fall-challenge-2023). 
- **Virtual Sensors**: `@grojas123` introduced to the concept of Virtual Sensors, a technology based on machine learning. Shared a link from [IEC](https://www.iec.ch/blog/untapped-potential-virtual-sensors) which provides a detailed explanation about it.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (6 messages): 
        
- **PPO Agent Playing Huggy**: `@cloudhu` shared their model of a **ppo** agent that was trained to play **Huggy**. The agent was trained using the [Unity ML-Agents Library](https://github.com/Unity-Technologies/ml-agents).
- **Mixtral-8x7B-Instruct-Demo at Lightning Speed**: `@myg5702` shared a [Hugging Face space](https://huggingface.co/spaces/FumesAI/Mixtral-8x7B-Instruct-Demo-at-Lightning-Speed), featuring a model named **Mixtral-8x7B-Instruct-Demo** that is running at a lightning speed.
- **Generating Music as MIDI**: `@alexsushidog` posted a [blog](https://afmck.in/posts/2023-12-22-tchaikovsky/) about generating music as MIDI by training a transformer from scratch in **JAX**.
- **Evaluation Metrics for Regression Models**: `@torres8552` created a Kaggle notebook that explains the math behind commonly used evaluation metrics in regression tasks. The notebook provides insight into how to interpret these metrics and how to create custom functions to compute them using Python. The [notebook is available on Kaggle](https://www.kaggle.com/code/lusfernandotorres/evaluation-metrics-for-regression-models).
- **Ensemble Learning Presentation**: `@johko990` delivered a presentation at a Python Pizza conference on Ensemble Learning, which is available [via YouTube](https://youtu.be/RCtnCMVYsKw). The talk reinterprets the fairy tale of Snow White in the context of Ensemble Learning.
- **New Test Model**: `@bread browser` mentioned creating a new test model, but didn't provide any further details or links.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Superimposition of Images**: `@cachalote123` asked whether it's possible to use **diffusion models** or any other technology to separate two superimposed images from old films.

- **Conversion of Images to 3D Objects**: `@drishya1` was seeking advice on which model would be the best choice for converting images to 3D objects after successfully using the **control net** to convert scribbles to images.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **Integrating GPT Models with Gradio UI in Google Colab**: User `@designfailure` is seeking help for compiling two GPT use cases using Gradio UI in a Google Colab environment. The desired functionalities are:
  1. **GPT-4 Vision or LlaVA to capture images** in response to a prompt query.
  2. A **GPT chatbot that parses and retrieves image captions**, using them to complete chat responses.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (17 messages🔥): 
        
- **Running Prompt Tuning and LORA**: `@opencuiguy` asked if it's possible to combine prompt tuning and LORA in one shot and desired to address the issue where prompt tuning does not expose the new token for flexibility.
- **Fine-Tuning Sent-Transformer**: `@nickprock` inquired about the number of examples required to fine-tune a sentence transformer using msmarco with tripletloss, as he could only find information related to TSDAE requiring between 60-100k sentences.
- **Modifying T5ForConditionalGeneration & Seq2SeqTrainer**: `@ianbestfellow` sought guidance on how to modify the forward part of T5ForConditionalGeneration and the Trainer for his research.
- **LayoutLM and RVL-CDIP**: `@gh6608` asked if anyone has had success using LayoutLM with the RVL-CDIP dataset.
- **Transliteration Model Training**: There was a discussion between `@asterix3651` and `@vipitis` about creating a transliteration model; `@asterix3651` needs a model for converting words from one language to their romanized form even for out-of-vocabulary words.
- **LLAMA-2 Issue**: `@notooth` described how the LLAMA-2 model was having difficulty with a height comparison task and was seeking guidance on how to improve its performance.
- **Loss Measurement in Text Generation**: `@exponentialxp` was curious about the parts of the text for which the loss is measured during text generation - whether it's confined to the Assistant/Response or includes all parts.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Separation of Superimposed Images**: `@cachalote123` asked about the possibility of using **diffusion models** or any other technology to separate superimposed images originating from old films.
- **Converting Images to 3D Objects using Diffusers**: `@drishya1` is trying to convert images into 3D objects using diffusers. They mentioned having used **Control Net** to convert scribbles into images and are now seeking advice for the best model to convert these images into 3D.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Discourses on how to create embeddings with a regular LLM, with `@marijnfs` asking about the possibility of averaging the activations of a higher-up layer or using the last token's activation vector.
- Queries and thoughts on **GPT4** fine-tuning, presented by `@dangfutures`.
- An exploration of open source alternatives to **Weights & Biases** for generating loss/evaluation graphs, with `@rtyax` seeking suggestions and `@noobmaster29` proposing **Tensorboard**.
- Encompassing interchanges about the merge of a pull request [#787](https://github.com/OpenAccess-AI-Collective/axolotl/pull/787) after rebase, discussions about changes in permissions, and the news of a merged PR [#2232](https://github.com/huggingface/accelerate/pull/2232) from `@nanobitz`, `@caseus_`, and `@dreamgen`.
- Brief dialogue on the features and benefits of torch SDP attention as an alternative to flash attention by `@caseus_`, sharing the [torch documentation link](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for additional information, courtesy of `@nanobitz`.
- Conception of **Mistral** class code's sliding window feature mentioned by `@nanobitz`.
- Suggestions from `@le_mess` and `@caseus_` on utilizing Axolotl for fine-tuning models and starting with **tinyllama** example.
- Insights into datasets compatible with Axolotl by `@dreamgen`, `@touristc`, `@le_mess`, and `@visuallyadequate`, discussing resources like ShareGPT, OpenAssistant/oasst1, OpenOrca, and datasets found in Hugging Face, such as Guanaco-Unchained-Refined and Wizard-Vicuna-Refined.
- Discourse on using toxic-dpo and other fine-tuning datasets, the anticipation for **Dolphin 3.0**, paucity of open source options for RAG datasets, and the relevance of [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) from `@dreamgen`, `@faldore`, and `@nruaif`.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (6 messages): 
        
- **Creating embeddings with a regular LLM**: `@marijnfs` inquired about the possibility and methods of creating embeddings with a regular LLM. They asked if it's an option to take the activations of a higher-up layer and average them, or take the last token's activation vector.
- **Fine-tuning GPT4**: User `@dangfutures` asked if anyone has had any success fine-tuning **GPT4**.
- **Open Source Alternatives to W&B**: User `@rtyax` sought suggestions for open source alternatives to **Weights & Biases** for generating loss/evaluation graphs. `@noobmaster29` suggested **Tensorboard** as one such open source option.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (22 messages🔥): 
        
- **Discussion on Merging PR**: User `@nanobitz` discusssed about merging a pull request [#787](https://github.com/OpenAccess-AI-Collective/axolotl/pull/787) after rebase. User `@caseus_` confirmed that it should be fine and made necessary changes in permissions to allow the merge.
- **Permissions Adjustment**: There was further discussion about adjustments for permissions led by `@caseus_` in response to an accidental push to the main, trying to limit future issues.
- **Merge of huggingface/accelerate PR**: In another conversation, user `@caseus_` shared the news of a merged PR [#2232](https://github.com/huggingface/accelerate/pull/2232) related to FP8 support integration by huggingface/accelerate.
- **torch SDP Attention**: User `@nanobitz` and `@caseus_` engaged in a brief discussion about the benefits and functionality of torch SDP attention, with a link shared to the [torch documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for further details. It was pointed out as an alternative to flash attention by `@caseus_`.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (10 messages🔥): 
        
- **Mistral Class Code**: `@nanobitz` mentioned that the **Mistral** class code has a sliding window feature.
- **Token Initialization**: `@dreamgen` expressed a desire for the ability to initialize the embedding of a **new token** from some existing token id.
- **Fine Tuning Queries**: `@thetonsil.` asked two questions about fine-tuning models:
    - They enquired whether `fine tuning can be done using only CPU resources`. `@le_mess` responded, stating that while it is theoretically possible, it would likely be very time-consuming and suggested renting a cloud GPU instead.
    - They also sought guidance on using **Axolotl**. As per the advice of `@le_mess`, they were directed to the examples in the examples folder, specifically the **mistral** and **llama** models.
- **Running the Examples**: `@le_mess` also provided instructions on how to run these examples using the `accelerate launch -m axolotl.cli.traion <example_path>` command after setting up the environment.
- **Starting with TinyLlama**: `@caseus_` suggested trying out a **tinyllama** example as a potential starting point.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (8 messages🔥): 
        
- **ShareGPT and OASST Datasets**: In a discussion on datasets compatible with Axolotl, `@dreamgen` mentioned ShareGPT as an original resource with some paper analyzing and breaking down chats based on user intent, while `@touristc` faced difficulties working with popular datasets like RyokoAI/ShareGPT52K and OpenAssistant/oasst1. 
- **OpenOrca Dataset**: `@le_mess` suggested **OpenOrca** as a functional dataset with Axolotl. Confirming the correct dataset type, `@touristc` asked if it should be set as alpaca_w_system.load_open_orca. 
- **Multiple-Round Conversational Datasets**: `@touristc` expressed interest in identifying multiple-round conversational datasets that work well with Axolotl, pointing out the limitation of the OpenOrca dataset being one round QA. `@nanobitz` mentioned that the current discussion could be deemed as a single round. 
- **Guanaco-Unchained-Refined and Wizard-Vicuna-Refined Datasets**: `@visuallyadequate` shared the [links to two datasets](https://huggingface.co/datasets/) on Hugging Face named Guanaco-Unchained-Refined and Wizard-Vicuna-Refined and highlighted their main focus was giving lists and code blocks consistent formatting.


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (13 messages🔥): 
        
- **Using Toxic-DPO Dataset in Fine-Tuning**: User `@dreamgen` mentioned that they have been using the toxic-dpo dataset in recent fine-tuning exercises, but raised a concern about mixing signals from unalignment and quality data. They suggested filtering big DPO datasets like Intel/orca_dpo_pairs for safety rejects in the "chosen" answer.
- **Waiting for Dolphin 3.0**: User `@faldore` indicated that they are waiting for **Dolphin 3.0** before further fine-tuning their system, with the aim of having a solid system prompt, instruct, conversation, RAG, and Agent dataset in place. However, they noted that Dolphin is already quite uncensored.
- **Lack of Open RAG Datasets**: In response to `@dreamgen`'s inquiry about RAG datasets, `@faldore` mentioned that there are not many great open source options and development in this area is needed.
- **Use of Chatbot Arena Conversations Dataset**: `@dreamgen` raised a question about why many models are not using the [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) for DPO/RL. In response, `@nruaif` explained that the dataset might be too large, yet they have observed people using smaller subsets.
- **Publication of Dataset Subsets**: Upon `@dreamgen`'s query about whether the subsets of the Chatbot Arena Conversations dataset are published, `@nruaif` did not provide a concrete answer. `@dreamgen` speculated that a subset where gpt-4 wins could be relatively safe.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Active discussions on various AI models and solutions, with particular focus on the **NexusRaven model**; skeptics and advocates alike shared their perspectives, and the relevant [Hugging Face link](https://huggingface.co/Nexusflow/NexusRaven-V2-13B) was provided. 
- Announcement by `@glenn_sjobs` about an upcoming **AI Hackathon** involving the use of LangChain in the developer environment; the official [link](https://www.defense.gov/News/Releases/Release/Article/3610215/chief-digital-and-artificial-intelligence-office-to-host-hackathon-in-hawaii/) was shared for additional information.
- Opinions and queries about the functionalities of various platforms were discussed including **S3 Storage**, **Streamlit**, **ContextualCompressor**, and **VSCode Auto Import**; a considerable repository of Python course material was also referenced in the [Google Drive link](https://drive.google.com/drive/folders/1CgN7DE3pNRNh_4BA_zrrMLqWz6KquwuD).
- Introduction of **Cumuli**, a new Chrome extension for AWS optimized with **GPT-4 Turbo with vision**; the [GitHub link](https://github.com/petrgazarov/cumuli-aws-console-chat) for the extension was provided.
- Sharing of a new **book on LangChain**, discussing how to effectively use the LangChain framework and work around its inherent weaknesses; the authors shared links for purchasing on [Amazon](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463).
- Users expressed questions related to controlling the length of chat history while using session-level memory, with the functionalities of `RunnableWithMessageHistory`, `charact_prompt | llm` and `RedisChatMessageHistory(session_id, url=REDIS_URL)` being discussed.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (27 messages🔥): 
        
- **S3 Storage Advice**: `@lhc1921` suggested considering S3 cloud or on-premises storage.
- **Python Problems Resource**: `@b1llygamidesuyo` shared a [link to a Google Drive folder](https://drive.google.com/drive/folders/1CgN7DE3pNRNh_4BA_zrrMLqWz6KquwuD) containing approximately 5 terabytes of Python course material.
- **Streamlit Button Inquiry**: `@infinityexists.` inquired about how to add a button (having an image or PNG) in Streamlit.
- **ContextualCompressor Prompt**: `@legendary_pony_33278` asked for help about the prompt for using ContextualCompressor in a non-English language.
- **Discussion on NexusRaven**: Users `@_egeres` and `@lhc1921` discussed the [NexusRaven model](https://huggingface.co/Nexusflow/NexusRaven-V2-13B). `_egeres` found out that this model can replicate the behavior of OpenAI's function calling API, while `@lhc1921` expressed skepticism over the claim that it surpasses GPT-4.
- **Hackathon Announcement**: `@glenn_sjobs`, a senior software/AI engineer for the US Secretary of Defense, informed the community about an upcoming AI Hackathon that would include LangChain in the developer environment. He shared the [official link](https://www.defense.gov/News/Releases/Release/Article/3610215/chief-digital-and-artificial-intelligence-office-to-host-hackathon-in-hawaii/) for more information and the application process.
- **VSCode Auto Import Issue**: `@solononforever` asked for assistance with VSCode's auto import feature, which wasn't working properly for them.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **Session-level Memory Control**: `@cryptossssun` inquired about controlling the length of chat history while using session-level memory in the `RunnableWithMessageHistory` function, particularly when using `charact_prompt | llm` and `RedisChatMessageHistory(session_id, url=REDIS_URL)`.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **New Book on LangChain**: User `@tusharg` shared a link to a new [book on LangChain](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463) available on Amazon. The book provides insights on how to leverage LLMs' capabilities and explores their fundamentals, ethical dimensions, and application challenges.
- **Cumuli, a new Chrome extension for AWS**: `@petrgazarov` introduced Cumuli, a Chrome extension that adds a LLM chat panel to all AWS pages. It allows users to add screenshots of the console to their queries for context-aware responses. The extension uses **GPT-4 Turbo with vision**. The extension can be accessed on [GitHub](https://github.com/petrgazarov/cumuli-aws-console-chat).


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **Latest Book on LangChain**: `@tusharg` shared a link to a new [book on Amazon](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463) which covers **LangChain framework** to develop applications like agents and personal assistants, web searches integration, and code execution. The book also focuses on *leveraging LLMs' capabilities and working around their inherent weaknesses*.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Progress in chunking documents for embedding to generate a multi-billion synthetic token dataset; demonstrated approach employs recursive segmentation into 512 length pieces with the title appended back before embedding, detailed by `@emrgnt_cmplxty`.
- Discussion on applying hierarchical search to the chunked documents, using the natural hierarchy of the web; an initial vector search indexes the leading chunk and title for each webpage, followed by a fetch of the full chunks for the top N matches, which are then re-ranked using the most similar chunk from each. The method amounts to an approximate 30 times reduction in the embeddings needed for the initial search stage. 
- Mention of document summary preparation by `@gabriel_syme`, corroborating the preparatory processes shared by `@emrgnt_cmplxty`.
- Introduction to the [Mergekit](https://github.com/cg123/mergekit) toolkit on GitHub for merging pretrained large language models, illustrated with a user model case: [MetaMath-neural-chat-7b-v3-2-Slerp](https://huggingface.co/Weyaxi/MetaMath-neural-chat-7b-v3-2-Slerp).
- Inquiry from `@cryptossssun` on the development of long context field abilities, target context recognition and extraction in AI models. 
- Query on the fine-tuning process in comparison with a 7-billion parameter model, leading to a remark by `@caseus_` on the model "qlora" under discussion, denoting its current state as less than satisfactory.

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (9 messages🔥): 
        
- **Chunking documents for embedding**: `@emrgnt_cmplxty` discussed their approach to preparing documents for embedding, mentioning that they are chunking the documents recursively into 512 length chunks, and then appending the title back on before embedding.
- **Generating a multi-billion synthetic token dataset**: `@emrgnt_cmplxty` stated the approach of preparing the documents for embedding will help generate a **multi-billion synthetic token dataset** without worries about quality. 
- **Hierarchical Search**: `@emrgnt_cmplxty` clarified that by hierarchical search, they mean they use the natural hierarchy of the web to perform the search. They index the leading chunk + title for each webpage and use that as the first vector search. They then fetch the full chunks for the top N matches and use the most similar chunk from each webpage to re-rank, which results in approximately a factor of 30 reduction in the embeddings needed to search over at the first stage. 
- **Preparation for document summaries**: `@gabriel_syme` mentioned that they are also preparing for a similar process with document summaries.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (3 messages): 
        
- **Mergekit Toolkit for Merging Language Models**: `@lightningralf` shared a link to the [Mergekit](https://github.com/cg123/mergekit) toolkit on GitHub, which offers tools for merging pretrained large language models. 
- **Use of Mergekit in MetaMath-neural-chat Model**: Further elaborating on the usability of Mergekit, `@lightningralf` provided an example of [MetaMath-neural-chat-7b-v3-2-Slerp](https://huggingface.co/Weyaxi/MetaMath-neural-chat-7b-v3-2-Slerp), a model by `@Weyaxi`, which used Mergekit for model merging.
- **Inquiry about Long Context Field**: `@cryptossssun` raised a query on whether anyone is focusing on aspects such as the long context ability and target context recognition and extraction.


### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (2 messages): 
        
- **Fine-tuning Inquiry**: `@emrgnt_cmplxty` asked about the fine-tuning process and the comparison of a model to the 7-billion parameter model.
- **Model Quality**: In response, `@caseus_` clarified that the model being discussed is a **qlora** and commented that it's not that good.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **QA Agents**: User `binary6818` shared information about a project called [camelQA](https://camelqa.com/) which offers AI services for testing mobile apps, and asked if anyone knows similar projects.
- **Converting Scanned Books to Chat**: User `.kareemk` asked for recommendations on open-source options for an OCR PDF to RAG pipeline to create a "chat with my scanned books". User `coffeebean6887` suggested that most OCR libraries should work fine with printed and scanned text but more complex documents like hand-written notes may require advanced OCR models.
- **Anyscale LLM Performance Benchmark Criticism**: User `guardiang` mentioned that Anyscale is receiving criticism for the LLM performance benchmark they published yesterday, sharing [this link to the topic](https://x.com/soumithchintala/status/1738241213327692174?s=20).
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- User `@tomsegura` in the general channel expressed optimism about the Skunkworks AI community and its potential for significant advancements, stating "*I do truly believe that someone like you guys are the future and are going to be what brings us real advancements that real people (us) can use.*"
- A discussion initiated by `@cyborgdream` in the datasets channel about the potential applications of a State Of The Art synthetic-generated dataset, comprising of 1-2 billion tokens, intended to benefit the Open source AI community.
- The off-topic channel featured a conversation about a [tweet](https://twitter.com/8teAPi/status/1737237462672707872) linked by `@shreepandey’. The tweet highlighted a significant decrease in latency within a real-time voice chat tutor developed by [Smarterchild.chat](http://Smarterchild.chat). This sparked interest about the methods employed to achieve this improved performance.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (2 messages): 
        
- User `@tomsegura` expressed support and belief in the potential of the Skunkworks AI community, stating "*I do truly believe that someone like you guys are the future and are going to be what brings us real advancements that real people (us) can use.*"


### ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/) (2 messages): 
        
- **Potential Synthetic Generated Dataset**: User `@cyborgdream` enquired about the desirable area of interest for a **State Of The Art** *synthetic dataset* of about 1-2B tokens that could be beneficial for the Open-source software **AI community**.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (2 messages): 
        
- **Decrease in Latency of Speech Feature**: `@shreepandey` shared a link to a [tweet](https://twitter.com/8teAPi/status/1737237462672707872) from `@8teAPi`. The tweet discusses the real-time voice chat tutor developed by [Smarterchild.chat](http://Smarterchild.chat) that has significantly reduced latency. `@shreepandey` and `@skaios` were interested in knowing how this reduction in latency was achieved.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Users inquired about the performance of **qLoRA**, with `@tcapelle` asking how it's determined that qLoRA doesn't perform well.
- `@tcapelle` further suggested **model fine-tuning** via freezing the top layers of the model, rather than using LoRA or qLoRA.
- A new guild member, `@philipmay` was introduced and expressed gratitude for the invitation. 
- A productive dialog was triggered by `@jiha`, who shared a [YouTube video](https://youtu.be/DngRcgYjDfU?si=x708Sq49YqkLr_75) titled "OpenDalle - Like running Dall-e Local", showcasing **OpenDalle’s capabilities**.
- `@datarevised` was recognized for creating the OpenDalle video shared in the chat and `@jiha` appreciated the quality of the video, expressing that OpenDalle seems cool based on the video content.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (2 messages): 
        
- **Query on QLoRA Performance**: `@tcapelle` asked for information about how it's determined that **qLoRA** doesn't perform well.
- **Suggestion for Model Fine-tuning**: `@tcapelle` proposed that rather than using **LoRA** or **qLoRA**, it might be beneficial to try **freezing the top layers** of the model.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (4 messages): 
        
- **Introduction of new member**: `@philipmay` joined the chat and expressed gratitude for the invitation.  
- **Discussion on OpenDalle Video**: `@jiha` shared a [YouTube link](https://youtu.be/DngRcgYjDfU?si=x708Sq49YqkLr_75) to a video titled "OpenDalle - Like running Dall-e Local", which appears to showcase the capabilities of OpenDalle.
- **Creator of the OpenDalle Video**: `@datarevised` confirmed they created the OpenDalle video shared in the chat. 
- **Appreciation for OpenDalle Video**: `@jiha` complimented the quality of the video and expressed that OpenDalle seems cool based on the video content.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- User `@emfastic` has developed an internal tool employing decorator syntax and expressed future intentions for it to go open-source.
- Minimal usage of embeddings by `@emfastic`, else there would be a consideration of using **llamaindex**.
- An inquiry from swyxio surrounding the slowness issues facing some users, asking "how slow are you seeing it".

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (2 messages): 
        
- User `@emfastic` mentioned that they crafted an internal tool using decorator syntax and are planning to open source it soon.
- `@emfastic` also stated that they are not making extensive use of embeddings, else they would consider using **llamaindex**.


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages): 
        
swyxio: how slow are you seeing it


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Issue with XGBoost on Apple M Pro Series**: User `@leg4l` reported an issue with running [XGBoost](https://xgboost.readthedocs.io/) on a Macbook Pro with Apple M Pro Series, where the application utilized the efficiency cores but exerted minimal pressure on the performance core. A possible GitHub discussion related to this problem was shared: [Incomplete CPU utilization with M1 Pro, M1 Max, and M1 Ultra architectures - Issue #8320](https://github.com/dmlc/xgboost/issues/8320)
        
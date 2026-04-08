---
id: a7cce46d-05bb-4fe2-ae06-fce26608d9ee
title: '12/12/2023: Towards LangChain 0.1'
date: '2023-12-13T03:45:12.627715Z'
type: archival
original_slug: ainews-12122023-towards-langchain-01
description: >-
  The **Langchain rearchitecture** has been completed, splitting the repo for
  better maintainability and scalability, while remaining backwards compatible.
  **Mistral** launched a new Discord community, and **Anthropic** is rumored to
  be raising another **$3 billion**. On the **OpenAI Discord**, discussions
  covered **information leakage** in AI training, **mixture of experts (MoE)
  models** like **mixtral 8x7b**, advanced **prompt engineering techniques**,
  and issues with **ChatGPT** performance and API access. Users also explored AI
  applications in **logo generation**, **education**, and **gaming**, and shared
  solutions for **Oauth2 authentication** problems. A new small language model
  named **Phi-2** was mentioned from **Microsoft**.
companies:
  - langchain
  - mistral-ai
  - anthropic
  - openai
  - microsoft
models:
  - mixtral-8x7b
  - phi-2
  - gpt-3
  - chatgpt
  - gpt-4
topics:
  - mixture-of-experts
  - information-leakage
  - prompt-engineering
  - oauth2
  - logo-generation
  - education-ai
  - gaming-ai
  - api-access
  - model-maintainability
  - scalability
people: []
---


<!-- buttondown-editor-mode: plaintext -->The [big Langchain rearchitecture](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/?utm_source=ainews&utm_medium=email) seems to be complete: 

![https://blog.langchain.dev/content/images/size/w1248/format/webp/2023/12/Transformation---shortened.png](https://blog.langchain.dev/content/images/size/w1248/format/webp/2023/12/Transformation---shortened.png)

This splits up the langchain repo to be more maintainable and scalable, an inevitable part of every integration-heavy open source framework:

![https://blog.langchain.dev/content/images/size/w1600/2023/12/LangChain-Stack---split---V3.png](https://blog.langchain.dev/content/images/size/w1600/2023/12/LangChain-Stack---split---V3.png)

It's all backwards compatible so no big rush.

In other news, [Mistral has a new Discord](https://discord.com/invite/mistralai) (we'll be adding to our tracker) and **Anthropic** is rumored to be raising another $3b.

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Discussion on Information Leakage in AI Training and Mixture of Experts Models**: `@moonkingyt` queried about information leakage during AI conversations and expressed curiosity about a model called "mixtral 8x7b". It was clarified that this model is a type of Mixture of Experts (MoE) model.
- **Advanced Use Cases of AI**: Users explored various potential AI use cases, such as modifying a company's logo for specific events, explaining complex concepts in large classes at universities, and potential applications for local language models.
- **Technical Challenges with OpenAI**: Multiple users shared issues they have experienced with ChatGPT, such as server unresponsiveness, slow response times, and difficulty with image uploads. Users also discussed the feasibility of training an AI model using a Discord server and HTML text files, and how to go about doing it.
- **Concerns about ChatGPT's Performance and Behavior**: User frustrations included perceived degradation in response quality/nuance, poor memory retention, and poor customer support. Some users reported that GPT is not answering questions as before and instead provides references to search answers on the internet.
- **Discussion on Accessing OpenAI GPT-4 API**: Users discussed the access and cost of GPT-4's API, the usage limit of the ChatGPT Plus, how to access it, and the process of reaching human support. An issue about incorrect highlighting in diff markdowns was found and reported.
- **Prompt Engineering Techniques**: Users discussed various advanced prompt engineering tactics, with an emphasis on clear prompt setting for obtaining desired outputs. Techniques such as outlining for longer written outputs, iterative proofreading, and implementation of a dynamic model system were proposed.
- **Oauth2 Issues and Solutions**: Oauth2 authentication issues were solved in collaboration among users. Discussion regarding minor bugs and the process of reporting them also took place.
- **Using AI for Games**: Users shared their attempts at getting AI to play games like hangman. A potential feature of toggling the Python display in the interface for such engagements was proposed.
- **Request for Detailed 2D Illustration**: A user requested detailed 2D illustration, and was guided to use DALL·E 3 on Bing's image creator or through ChatGPT Plus! An issue of not being able to edit custom GPT's was also reported.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (42 messages🔥): 
        
- **Information Leakage in AI Training**: `@moonkingyt` queried about cases where chatbots leak information during conversations. `@xiaoqianwx` responded affirmatively without further elaborating the context.
- **Discussion on Mixture of Experts Models**: `@moonkingyt` expressed curiosity about a term called "mixtral 8x7b". `@xiaoqianwx` clarified that it's a type of Mixture of Experts (MoE) model.
- **Logo Generation with AI**: `@julien1310` sought guidance on whether an AI can modify a company's existing logo for a special event like a 20th anniversary. `@elektronisade` suggested using specific features like inpainting from StyleGAN or its derivatives.
- **Usage of AI in Educational Institutions**: `@offline` mentioned that some universities don't permit AI assistance, hinting at the potential of AI like GPT-3 to explain complex concepts, often in a more detailed manner than instructors in large classes.
- **Question about Local Language Models**: `@Shunrai` asked the group for recommendations on good local language models. It spurred subsequent discussions, but no specific models were recommended.
- **Microsoft's New Language Model**: `@【ｐｅｎｕｌｔｉｍａｔｅ】` shared news about a small language model named Phi-2 being released by Microsoft. Further details or a link to more information were not provided.
- **AGI (Artificial General Intelligence) Debates**: A side discussion was sparked by `@chief_executive` questioning the term AGI if it keeps getting "nerfed" every step. Meanwhile, `@【ｐｅｎｕｌｔｉｍａｔｅ】` criticized constant shifting of AGI's goalpost.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (337 messages🔥🔥): 
        
- **Issues with OpenAI ChatGPT**: Users `@ruhi9194`, `@stupididiotguylol`, `@rjkmelb`, `@yyc_`, `@eskcanta`, `@toutclaquer`, `@gingerai`, `@ivy7k`, `@millicom`, `@ayymoss`, `@knightalfa`, `@loschess`, `@primordialblanc` reported multiple technical issues with ChatGPT including server unresponsiveness, slow response times, difficulty with image uploads, and error messages such as "Hmm...something seems to have gone wrong."
- **Discussion on Using Custom GPT Files**: User `@maxdipper` asked about the possibility of training a model off a Discord server and whether HTML text files can be utilized for feeding data to the model.
- **Confusion Over Image Analysis in ChatGPT**: User `@toutclaquer` shared an issue where image analysis in ChatGPT was not working. Error messages were displayed when attempting to analyze images.
- **Concerns about ChatGPT's Behavior and Performance**: Users `@prncsgateau`, `@the_time_being`, `@.nasalspray`, `@one_too_many_griz`, `@fefe_95868` shared their frustrations with ChatGPT's change in behavior, perceived degradation in response quality/nuance, poor memory retention, and poor customer support.
- **Questions about Accessing GPT-4 API**: User `@moonkingyt` inquired about the access and cost of GPT-4's API. `@elektronisade` provided a link to OpenAI's documentation outlining the access conditions and `@offline` further clarified the requirement of being a Pay-As-You-Go customer with a minimum spend of $1. `@DawidM` raised concerns about degradation of service for premium users.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (170 messages🔥🔥): 
        
- **ChatGPT Troubleshooting and Questions**: Many users such as `@mustard1978`, `@yyc_`, `@blinko0440`, `@jonsnov`, among others, reported various issues with the platform, with common challenges being inconsistency of responses, error messages, issues in uploading images, and problems when the system has reached a usage limit. `@solbus` provided extensive troubleshooting advice, suggesting solutions such as checking VPN status, trying different browsers, and clearing browser cache and cookies.
- **Access Termination Issues**: User `@mouse9005` shared an issue around account termination and appealed for help. `@rjkmelb` advised them to contact OpenAI support via their [help center](https://help.openai.com).
- **ChatGPT Plus Subscription and Usage Limit**: A discussion occurred centered on the usage limit of the ChatGPT Plus, involving `@lunaticspirit`, `@rjkmelb`, and `@DkBoss`. The limit for ChatGPT Plus is 40 messages per 3 hours. A reference for the discussion was provided in the form of this [link](https://community.openai.com/t/chatgpt-plus-and-usage-limits/544425#:~:text=ChatGPT%20Plus%20has%20a%2040,seem%20to%20be%20very%20short.).
- **Accessing Tools and Plugins in the New ChatGPT Plus UI**: `@_.gojo` inquired about how to access the new GPT-4 model and plugins in the playground with the ChatGPT Plus subscription. `@solbus` provides a [link](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4) to how one can access GPT-4 API and also explains how to enable plugins from the account settings.
- **Payment Issues**: Users like `@Ugibugi`, `@michaelyungkk`, and `@tardis77b` have encountered challenges with the payment process for subscribing to the service. `@solbus` and `@eskcanta` recommended contacting OpenAI help center for assistance.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (74 messages🔥🔥): 
        
- **Oauth2 Issues and Solutions**: Users `@jdo300` and `@draennir` had discussions about issues they were facing with Oauth2 authentication for their API servers. They worked collaboratively to solve their issues and `@draennir` eventually solved his issue by adding `application/x-www-form-urlencoded` among the parsers in Express.
- **Dynamic model suggestion for GPT**: User `@offline` suggested the implementation of a dynamic model system that would decide which GPT model to use based on the requirements of the user's request. The idea is further discussed and supported by `@solbus`.
- **Minor bugs & Bug reporting**: An issue of incorrect highlighting in diff markdowns was found and reported by user `@offline`. They discussed about the bug reporting process with `@solbus`.
- **GPT Behaviour Issues**: User `@maypride` reported an issue with GPT not answering questions like before and instead it was providing references to search answers on the internet. `@cris_cezar` reported an inability to edit one of their custom GPT's, even after trying multiple browsers and clearing cache.
- **User Interface & Support**: `@cris_cezar` expressed confusion about the Discord interface, comparing it to an aviation cockpit. The user was guided to the OpenAI staff support via the help.openai.com chat. `@solbus` and `@eskcanta` discussed the process of reaching human support through the help.openai.com chat interface.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (32 messages🔥): 
        
- **ChatGPT and Writing Specific Word Count**: `@davidmajoulian` mentioned that ChatGPT underproduces the number of words when asked to write an article of a specific length, to which `@rjkmelb` responded that ChatGPT doesn't know how to count words and it's not realistic to expect it to produce an article of a specific length. The two users also discussed the possibility of using tools that add such functionality to OpenAI models.

- **Exploration of Advanced Prompt Engineering**: `@yautja_cetanu` initiated a discussion on advanced prompt engineering and the challenge in finding before-and-after examples of prompt adjustment that significantly improved the model's performance. Several users affirmed that given the advancements in models like ChatGPT, many traditional prompt engineering techniques seem less important, with the quality of prompt output largely leaning on the clarity and specificity of the instructions given to the AI.

- **Prompt Engineering Techniques**: `@eskcanta` and `@cat.hemlock` emphasized on the importance of clear, specific, error-free prompt setting for obtaining desired results from the AI, with the latter suggesting the use of outlining for longer written outputs and iterative proofreading.

- **Using AI for Games**: `@eskcanta` shared a failed attempt at getting the AI to play hangman and `@thepitviper` responded with an example where they managed to get the AI to generate a working Python game for hangman. There was also a suggestion for the ability to toggle Python display on and off for this kind of engagement.

- **Detailed 2D Illustration Request**: `@yolimorr` requested for a detailed 2D illustration, and `@solbus` mentioned that image generation is not supported on the server and directed the user to use DALL·E 3 on Bing's image creator or with ChatGPT Plus!.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (32 messages🔥): 
        
- **Prompt Length with ChatGPT:** `@davidmajoulian` asked about the issue of ChatGPT producing less content than requested in his prompts. He stated that when he requested a 1,500-word article, it returned only roughly 800 words. `@rjkmelb` clarified that **ChatGPT doesn't know how to count words** and suggested using the tool to assist in constructing an article rather than expecting a set word length. `@rjkmelb` also hinted at **building other tools using the OpenAI GPT-4 API** for advanced features. 

- **Advanced Prompt Engineering**: `@yautja_cetanu` initiated a discussion on advanced prompt engineering, expressing difficulty in providing pre and post examples for his meetup talk because of **how good chatGPT already is**. `@tryharder0569` suggested focusing more on how specific prompts enhance an application's functionality. `@madame_architect` recommended considering step-back prompting or thinking about output quality in terms of higher value or more helpfulness instead of a working vs non-working dichotomy. 

- **Prompt Output Quality Control:** `@cat.hemlock` emphasized the importance of being specific and deliberate in instructing the AI to avoid generating 'AI slop' (garbage output). They suggested **using ChatGPT to generate an article outline** first and refining it before expanding it into full content, followed by a final proofread.

- **ChatGPT Capability Limitations**: Certain limitations of ChatGPT were also mentioned. `@eskcanta` provided an example of the model failing to correctly play a game of Hangman even with a known word, known order of letters, in-conversation correction, and step by step examples to follow. `@thepitviper` demonstrated the use of Python with ChatGPT to run a slow, yet accurate Hangman game.
 
- **Python Tool Display Toggle**: A potential feature of toggling the Python display in the interface was discussed between `@thepitviper` and `@eskcanta` which could improve the user experience, though this would need to satisfactorily address situations where the system is performing an action to prevent undesired outcomes.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Summary of extensive discussions on the **SOLAR 10.7B** AI model across multiple channels involving benchmark results, performance comparison, and validity of its claims on outperforming other models. The model showcased significant gains in the AGIEval, with interesting skepticism and eventual recognition of its performance.
    - "*from first look, it's a flop*" - `@teknium`
    - "*TLDR: Good but dont deserve so much att*" - `@artificialguybr`
- Interaction on technical challenges in various areas: Base AI model described as an "untamed beast", the query on GPU performance leading to a discussion on cooling methods for Lambda's Vector One workstations, and queries on optimizing the inference speed with HuggingFace transformers.
- Dialogue on OpenHermes 2.5's ability for function calling and the complexities involved with shared experiences and resources. Microsoft's Phi-2 was mentioned for reportedly outperforming Mistral 7b and Llama 7b on benchmarks.
- Ongoing discourse on various **Mistral models**, the potential of fine-tuning models on Apple Silicon. Resources and discussion threads shared for further exploration.
- Community collaboration and assistance sought in areas like AI repositories needing frontend interface development, advice on choosing an orchestration library for developing an LLM-based application, with recommendations against LangChain favoring Llama-Index or custom platforms.
- The release announcement of the **UNA-Neural-Chat-v3-3-Phase 1** model, described as outperforming the original in initial tests.
- Miscellaneous points: Humorous observations on working at Microsoft, involvement in gene editing technologies, grammar and spelling problem-solving using bots, sharing content of interest across different channels in the form of tweets and video links.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (1 messages): 
        
teknium: https://fxtwitter.com/abacaj/status/1734353703031673200


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (18 messages🔥): 
        
- **Concerns about Base Model**: `@nagaraj_arvind` described the base AI model as an "untamed beast," expressing some challenges or difficulties with it.
- **AI Repos Assistance**: `@httpslinus` sought out any AI-related repos that need help with frontend interface building or extension, intending to assist with their current skill set.
- **Workplace Woes**: `@airpods69` shared their experience of leaving a workplace due to dissatisfaction and stress caused by the practices of a senior developer.
- **DeciLM-7B Discussion**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=wjBsjcgsjGg) about DeciLM-7B, a 7 billion-parameter language model. However, `@teknium` expressed disappointment, stating that DeciLM-7B's performance is driven by its gsm8k score and is worse at mmlu than Mistral.
- **Vector One Workstations Analysis**: `@erichallahan` shared a [Twitter post](https://fxtwitter.com/EricHallahan/status/1734627674079871050) providing analysis on Lambda's Vector One workstations. This led to a discussion on cooling methods, with `@fullstack6209` expressing a desire for air-cooled systems, and `@erichallahan` agreeing with this preference.
- **GPU Utilization Query**: `@everyoneisgross` asked about the GPU performance when running an LLM on GPU VRAM, using python to max out CPU, and multi-screening YouTube videos at the same time, alluding to possible visual glitches or artifacts.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (15 messages🔥): 
        
- **SOLAR 10.7B Benchmarks Shared**: `@teknium` posted benchmark scores for the **SOLAR 10.7B** AI model on various tasks, including: `truthfulqa_mc`: Value: 0.3182 (mc1), 0.4565 (mc2) and `arc_challenge`: Value: 0.5247 (acc), 0.5708 (acc_norm)
- **Comparative Evaluation with Other Models**: `@teknium` compared the performance of **SOLAR 10.7B** with that of **Mistral 7b**. SOLAR 10.7B seemed to achieve a notable gain in AGIEval (39% vs 30.65%) but demonstrated similarities in other evaluations (72% vs 71.16% in gpt4all, 45% vs 42.5% in truthfulqa).
- **Remarks on SOLAR 10.7B's Performance**: Despite the initial perception that SOLAR 10.7B's performance was not impressive (`"from first look, it's a flop"` and `"good but probably not as good as yi like it claims"`), the comparative results led `@teknium` to reassess the model's performance, significantly in AGIEval.
- **Final Thoughts on SOLAR 10.7B**: `@artificialguybr` summarised the discussion by saying that while SOLAR 10.7B had decent performance, it did not seem to warrant significant attention (`"TLDR: Good but dont deserve so much att"`).


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (42 messages🔥): 
        
- **SOLAR-10.7B Model Discussion**: `@metaldragon01` linked to [a Hugging Face post](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) that introduced the first 10.7 billion parameter model, SOLAR-10.7B. 
- **Claims of Outperforming Other Models**: The SOLAR team claims their model outperforms others, including Mixtral. Users `@n8programs` and `@teknium` expressed skepticism about the claims.
- **Evaluating the SOLAR-10.7B Model**: `@teknium` began benchmarking the new SOLAR model and planned to compare the results with Mixtral's scores.
- **SOLAR-10.7B Characterization**: In discussion, users concluded that SOLAR is likely pre-trained but may not be a base model, based on its similarity to a past iteration of SOLAR, which was not a base model. 
- **Tweet on Mixtral and 3090 Setup**: `@fullstack6209` asked about the optimal setup for Mixtral and a 3090, to which `@lightningralf` responded with a [Twitter link](https://fxtwitter.com/llm360/status/1734227314773495816) possibly containing the answer.


### ▷ #[bots](https://discord.com/channels/1053877538025386074/1149866614590816256/) (48 messages🔥): 
        
- **Gene Editing Technologies**: User `@nagaraj_arvind` asked about advances in gene editing technologies since 2015. The bot replied but the details were not elaborated in the messages.
- **Backward Sentences Completion**: `@crainmaker` seeked the bot to complete a string of backward sentences. 
- **Questions in Portuguese**: `@f3l1p3_lv` asked a series of questions in Portuguese, most of which are related to basic math problems, day of the week calculations, analogy formation, and appropriate pronoun selection for sentences. The bot `@gpt4` and `@compbot` helped to solve most of these questions.
- **Grammar and Spelling Problems**: `@f3l1p3_lv` also posted multiple spelling and grammar problems for the bot to solve, such as filling the blanks with specific letters or groups of letters, selecting pronouns for sentences, and choosing correctly written sentences. The bot `@gpt4` provided the solutions and explanations.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (337 messages🔥🔥): 
        
- Discussion on **Function Calling Capabilities**: User `@realsedlyf` inquired about OpenHermes 2.5's ability to do function calling, other users, including `@.beowulfbr` and `@tobowers`, shared their experiences with function calling with various models. `@.beowulfbr` shared a link to a [HuggingFace Model](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling) with function calling capabilities, whereas `@tobowers` shared his implementation of function calling using the SocialAGI code from [GitHub](https://github.com/opensouls/SocialAGI/blob/main/core/src/next/languageModels/FunctionlessLLM.ts).

- Announcement of **UNA-Neural-Chat-v3-3-Phase 1 Release**: User `@fblgit` announced the release of UNA-Neural-Chat-v3-3-Phase 1, which in initial tests outperforms the original model.

- Discussion on **Optimization of Inference Speed w/ hf transformers**: `@lhl` shared his experiences on optimizing the inference speed with HuggingFace transformers, detailing how he achieved better results.

- Discussion on **Usefulness and Performance of Small Scale Models**: `@a.asif` asked about the best performers in the realm of small scale models that could be run on laptops. Plugyy raised a question about the possibility of running the Mixtral MoE model on limited hardware with the help of quantization and llama.cpp.

- Links shared:
  1. A [tweet](https://fxtwitter.com/arthurmensch/status/1734470462451732839?t=NCKjPpiTOtOdfxDJ92H6IA&s=09) by `@nruaif` discussing a new development in AI modeling techniques.
  2. A [tweet](https://fxtwitter.com/NexusflowX/status/1732041385455624256?s=20) shared by `@tokenbender` discussing Real SoTA work in function calling.
  3. A [tweet](https://fxtwitter.com/Weyaxi/status/1734625859334537657?t=HERP4u20doaLI8kS8X0gOA&s=19) by `@weyaxi` discussing Marcoroni models.
  4. [Microsoft's Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) was mentioned in the discussion for reportedly outperforming Mistral 7b and Llama 7b on benchmarks. 
  5. User `@benxh` has a dataset to upload which would lead to >100B high quality tokens if properly processed.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (63 messages🔥🔥): 
        
- **Mistral Model Discussion**: Users `@agcobra1`, `@teknium`, `@gabriel_syme`, and `@giftedgummybee` discussed about various Mistral models, including base and instruct versions.
- **Library for LLM-based Application**: User `@coco.py` asked for advice on choosing an orchestration library between LangChain, Haystack, and Llama-Index for developing a LLM-based application. Users `@.beowulfbr` and `@decruz` expressed their problems with LangChain and recommended trying LLama-Index or building their own platforms.
- **Fine-tuning Discussion**: Users `@n8programs`, `@teknium`, `@youngphlo`, and `@eas2535` discussed the potential of fine-tuning models on Apple Silicon and brought up some resources including Github discussion threads [https://github.com/ggerganov/llama.cpp/issues/466](https://github.com/ggerganov/llama.cpp/issues/466) and Reddit posts [Fine tuning on Apple Silicon](https://www.reddit.com/r/LocalLLaMA/comments/152oudd/fine_tuning_on_apple_silicon/) and [Fine tuning with GGML and Quantization](https://www.reddit.com/r/LocalLLaMA/comments/15y9m64/fine_tuningggml_quantiziation_on_apple_silicon/).

- **Benchmarking Models**: `@brace1` asked for recommendations for benchmarking open-source models for a specific text extraction task. `@night_w0lf` suggested models like Mamba, Mistral0.2, Mixtral, OpenHermes etc, and also directed to a [leaderboard on Hugging Face](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). `@n8programs` cautioned about the limitations of benchmarks and recommended checking the models' ELO on [another leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).
- **Parameter Impact on Speed**: User `@n8programs` raised a query on why "stable diffusion", despite having around 3 billion parameters, was significantly slower than a 3 billion parameter LLM.


### ▷ #[memes](https://discord.com/channels/1053877538025386074/1166105758635655270/) (1 messages): 
        
- **Discussion on Microsoft Depiction**: User `@fullstack6209` shared a humorous observation about working at Microsoft, also questioning the absence of a certain group in their representation.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Release of Mixtral and Model optimization**: Mistral released **Mixtral 8x7B**, a high-performance SMoE model with open weights. The model was praised for its efficiency and speed, with `@aortega_cyberg` emphasizing that it's 4 times faster and performs better for coding compared to `Goliath`. Users also noted improvements in `Hermes 2.5` over `Hermes 2`. The availability and usage of `Mistral-Medium` model were also brought up. Read the full release details and pricing in [Mistral's blog post](https://together.ai/blog/mixtral).

- **Quantization, Software Integration and Challenges**: Discussions about quantization methods including `"INT3 quantization"` and `"AWQ"` were highlighted by `@vince62s` and `@casper_ai`. A need for exllamav2 support for contexts over 4k was also mentioned, and software limitations in LM Studio and exllama were brought up by `@aiui`.

- **Model Efficiency and Function-Calling**: The model's efficiency is theorized to originate from the "orthogonality" of its experts, as per `@fernando.fernandes`. A minimum-p routing system for expert evaluations was suggested which could enable scalable speed optimization. Strategies to mitigate similar problems and enrich Mixtral training data, such as including function-calling examples, were addressed with relevant resources shared, among them the Glaive Function-Calling v2 dataset on [HuggingFace](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2).

- **Benchmarks, Scoring Models, and Emotional Intelligence**: A new benchmark **EQ-Bench**, focusing on the emotional intelligence of LLMs, was introduced by `@.calytrix` who claimed its strong correlations with other benchmarks. Concerns about possible bias due to GPT-4 usage and potential improvements were discussed. GSM8K was recommended as a SOTA math benchmark. Score variations between mixtral instruct and other models on EQ-Bench stirred up conversations about potential causes.

- **Resources and New Development**: An explainer from Hugging Face on the Mixture of Experts (MoEs) model was shared [here](https://huggingface.co/blog/moe). A new model "Yi_34B_Chat_2bit" by Minami-su on HuggingFace was introduced [here](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit). Corrected pricing page link was shared [here](https://docs.mistral.ai/platform/pricing) and Mistral's Discord server link is [here](https://discord.gg/mistralai).

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (170 messages🔥🔥): 
        
- **Mixtral Model Performance and Optimization**: Users expressed a positive reception of the Mixtral model's performance and efficiency. `@Makya` stated that `"Hermes 2.5"` performs better than `"Hermes 2"` after adding [code instruction examples](https://link.to.examples). Regarding the `Mixtral-Instruct` model, `@aortega_cyberg` highlighted that although `Goliath` is better for creative writing, Mixtral performs better for coding and is 4x faster. `@goldkron` pointed out repetition loops in Poe's Mixtral implementation. 

- **Quantization and Software Integration**: Participants discussed different quantization methods, including 1-bit and 3-bit options. `@vince62s` discussed the need for `"INT3 quantization"` with `"AWQ"`. `@casper_ai` responded by stating that Mixtral is not yet ready for `"AWQ"` but anticipated that it should offer faster performance than 12 tokens per second once it is. In the meantime, `@vince62s` suggested that finetuning with six active experts might allow the Mixtral model to run on a single 24GB card. Furthermore, users discussed the need for exllamav2 support for contexts over 4k, with `@aiui` pointing out current limitations in LM Studio and exllama. 

- **Importance of Experts in Model Efficiency**: `@fernando.fernandes` theorized that the efficiency of Mixtral potentially results from the "orthogonality" of its experts, leading to maximal knowledge compression. `@someone13574` proposed a minimum-p routing system where experts are evaluated based on their softmax score compared to the top expert. This would potentially enable scalable speed optimization by controlling the minimum-p level. 

- **Using Function Call Examples in Training**: `@fernando.fernandes` suggested enriching the Mixtral training data with function-calling examples for diversity and to mitigate similar problems. He shared the Glaive Function-Calling v2 dataset on [HuggingFace](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) as a potential resource. Users also shared other potential datasets for this purpose.

- **Availability of the Mistral-Medium Model**: Users mentioned that the `Mistral-Medium` model is now available through the Mistral API. They speculated that this version could be around 70 billion parameters, though the model's exact size is as yet unconfirmed.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (9 messages🔥): 
        
- **Mixtral Release and Pricing**: `@bjoernp` shared the official announcement about the release of **Mixtral 8x7B**, a high-quality sparse mixture of experts model (SMoE) with open weights, by Mistral. He noted that it offers the **fastest performance at the lowest price**— up to 100 tokens/s for $0.0006/1K tokens on the Together Inference Platform. Its optimized versions are available for inferences on Together's platform. Full details in [Mistral's blog post](https://together.ai/blog/mixtral).
- **Mixture of Experts Explainer**: `@nembal` shared a link to an explainer from HuggingFace about the Mixture of Experts (MoEs) model. The post dives into the building blocks of MoEs, training methods, and trade-offs to consider for inferences. Check out the full explainer [here](https://huggingface.co/blog/moe).
- **Pricing Page Issue**: `@peterg0093` reported that the Mistral's pricing page link responded with a 404 error. However, `@_jp1_` provided the correct [link](https://docs.mistral.ai/platform/pricing) to Mistral's pricing information.
- **Mistral Discord Link**: `@le_mess` asked for the Mistral's Discord server link, which `@nembal` successfully provided. Join their server [here](https://discord.gg/mistralai).
- **New Model by Minami-su**: `@2012456373` shared a new model by Minami-su on HuggingFace named "Yi_34B_Chat_2bit". The model is optimized to run on an 11GB memory GPU with a weights-only quantization method called QuIP# to achieve near fp16 performance using only 2 bits per weight. Detailed information can be found [here](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit).


### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (15 messages🔥): 
        
- **EQ-Bench for Emotional Intelligence**: `@.calytrix` introduced his paper and benchmark on emotional intelligence for LLMs, **EQ-Bench**, claiming that it has strong correlations with other benchmarks (r=0.97 correlation with MMLU, for example). Mentioning that EQ-Bench seems to be less sensitive to fine-tuning for scoring well on other benchmarks. Notably, the **parameter size** seems to count for more as nuanced understanding and interpretations required for emotional intelligence seem more strongly emergent at larger param sizes. He also shared the [Paper Link](https://arxiv.org/abs/2312.06281) and the [Github Repository](https://github.com/EQ-bench/EQ-Bench).
- **Potential Improvements and Limitations to EQ-Bench**: `@onuralp.` suggested potential improvements including incorporating item response theory and reporting per-subject MMLU score correlations. The benefit of exploring model responses for well-structured scenarios involving agreeableness and negotiation was also suggested. A concern about potential bias due to choosing GPT-4 as the generator was brought up. 
- **EQ-Bench creator's response**: `@.calytrix` shared that the decision to use GPT-4 was based on resource constraints but confirmed that it did not generate the questions and answers. Defending the EQ-Bench, he argued that it genuinely measures some deep aspects of cognitive capabilities and offers useful discriminatory power.
- **Math Benchmark queries**: In response to a request from `@mr.userbox020` for references about SOTA math benchmarks, `@.calytrix` recommended GSM8K for its focus on reasoning over raw calculation. He also shared a [Paper Link](https://arxiv.org/pdf/2211.09066.pdf) exploring the use of the inference space as a scratchpad for working out problems using left-to-right heuristics.
- **Score Disparities between models on EQ-Bench**: Users `@bjoernp`, `@rtyax`, and `@.calytrix` engaged in a discussion about the lower scores mixtral instruct gets on EQ-Bench compared to its scores on other benchmarks. They showed curiosity about the disparity, suggesting possible reasons including a quick and dirty fine tune and inherent limitations of using 7b models as the base for MoE.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Engaged discussion around **Mixtral**, dealing with TypeError during *loading mixtral via vllm*, training on single A100 using *qlora*, LM Studio's support, and code updates including *support for Mixtral*. A notable mention of a *wandb project* showing possible training set up was shared by `@le_mess` [here](https://wandb.ai/mhenrichsen/mixtral?workspace=user-mhenrichsen). 

- Technical dialogues surround **Axolotl**, including activating *"output_router_logits"* setting in the model configuration for auxiliary balancing loss, and issues with loading multiple datasets with hyphenated names.

- Inquiry about current quantization technics led to the mention of **Hugging Face**'s APIs, GenUtils and **autogptq** for choice-based inference.

- Heartened interaction regarding training models, particularly around *VRAM minimum requirements* for training and fine-tuning AI models, issues and solutions related to stuck *Mixtral training*, discovering and sharing of *wandb project* results. Users shared their experiences while setting and solving problems in **Axolotl** setup and debugging with Docker, and discussed issues, views and workarounds for Axolotl training issues related to huggingface/transformers. A question related to loss spikes in pretraining remains open, without a common consensus.

- Dataset-specific chat involved the potentials and cons of training AI on YouTube debate transcripts, *interest in the local search repository*, personal recommendation and experiences with tools for PDF conversion to markdown, specifically mentioning [Marker](https://github.com/VikParuchuri/marker). The discourse broadened to alternative solutions for document processing, but budget constraints negated their use for some users.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (48 messages🔥): 
        
- **Issues loading mixtral via vllm**: `@papr_airplane` reported TypeError when trying to load mixtral via vllm using "pt" format. `@caseus_` announced the official mixtral modeling code release now supports mixtral.
- **Impact of Rope on Training**: User `@kaltcit` sparked a discussion on the effect of enabling rope on VRAM usage and training sequences. `@nanobitz` clarified that rope itself doesn't affect VRAM, but increasing the sequence length will.
- **Loading mixtral gguf onto LM studio**: `@papr_airplane` asked if anyone has loaded mixtral gguf onto LM studio, but `@nruaif` replied that LM studio doesn't support it yet, while `@faldore` claimed to achieve it using ollama. 
- **Training Mixtral on a single a100 using qlora**: `@le_mess` shared that training Mixtral using qlora on a single a100 seems possible, [linking to a wandb project](https://wandb.ai/mhenrichsen/mixtral?workspace=user-mhenrichsen).
- **Quality of Life Improvement in Docker Images**: `@caseus_` announced a new update adding `evals_per_epoch` and `saves_per_epoch` parameters that would be available in docker images soon. The update is designed to support quality of life improvement by eliminating the need to calculate total steps or back-calculate from the number of total epochs.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (11 messages🔥): 
        
- **Updated Modeling Code**: `@bjoernp` mentioned that the updated modeling code for **Mixtral** has been pushed.
- **Auxiliary Balancing Loss for the Router**: `@bjoernp` inquired about the activation of the auxiliary balancing loss for the router. According to `@caseus_`, enabling this feature is done through the configuration. `@caseus_` provided a sample code snippet, which indicates that it should be possible to enable this feature in **Axolotl** by adding `"output_router_logits: true"` under `model_config`.
- **Loading Multiple Datasets**: `@le_mess` reported an issue with loading multiple datasets that have `-` in their names, and asked if anyone else had encountered this problem. According to `@nanobitz` and `@noobmaster29`, the loading should work fine if using quotes or if they have not updated to the latest mixtral update for **Axolotl**.


### ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (3 messages): 
        
- **Discussion on Quantization Techniques**: User `@jovial_lynx_74856` asked about the current technology being used for quantization and inference, suggesting **Hugging Face**'s APIs and GenUtils. In response, `@nanobitz` mentioned **autogptq** for quantization, along with advising the choice-based inference.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (70 messages🔥🔥): 
        
- **Training Models with Different VRAMs**: Users discussed about the minimum requirements of VRAM for training AI models. `@nruaif` mentioned that the minimum is 1 A6000, which equates to 48 GB of VRAM. `@gameveloster` asked if 24 GB or 48 GB VRAM is enough to fine-tune a 34b model, to which `@le_mess` confirmed that 48 GB should be enough, but there could be problems with 24 GB.

- **Mixtral Training**: `@le_mess` shared issues about stuck Mixtral training with no drop in loss. Other users suggested enabling `output_router_logits`, increasing learning rate by a factor of 10, and provided references to similar training issues. `@le_mess` tracked his experiment using Weights & Biases. [Link to result 1](https://wandb.ai/mhenrichsen/mixtral/runs/gitdchmv?workspace=user-mhenrichsen), [Link to result 2](https://wandb.ai/mhenrichsen/mixtral/runs/sagr5cca). `@caseus_` suggested reference training with higher learning rate [Link](https://wandb.ai/oaaic/slimorca-mixtral)

- **Axolotl Setup and Debugging with Docker**: `@jovial_lynx_74856` posted about issues encountered while setting up Axolotl with CUDA version 12.0 on an 80 GB A100 server and running the example command. `@le_mess` suggested running the setup in a Docker environment on Ubuntu and CentOS.

- **Axolotl Training Issues**: `@jovial_lynx_74856` reported errors encountered during Axolotl training. `@caseus_` offered some insights and shared a link for reference on relevant changes introduced by huggingface/transformers [Link](https://github.com/huggingface/transformers/pull/26572#discussion_r1400917933). The issue seems to down to falling back to SDPA (Scaled Dot Product Attention) instead of the eager version and `@jovial_lynx_74856` found a workaround by setting `sample_packing` to `false` in the YML file.

- **Loss Spike in Pretraining**: `@jinwon_k` asked about loss spikes in pretraining. The users did not provide solutions or pointers. This question remains open.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (20 messages🔥): 
        
- **Training AI on Debate YouTube Transcripts**: `@papr_airplane` suggested training AI on YouTube transcripts of debate competitions to improve reasoning skills. However, `@faldore` countered that it risks teaching the AI to always argue with the user.
- **Access to Local Search Repository**: `@emrgnt_cmplxty` offered to give early access to a new [repository](https://github.com/SciPhi-AI/local-search) after `@papr_airplane` expressed interest.
- **Discussion on PDF to Markdown Conversion Tools**: `@visuallyadequate` shared personal experiences with multiple tools and libraries for PDF to markdown conversion. They shared the link to [Marker](https://github.com/VikParuchuri/marker) as a strong candidate, although noted its limitations with tables and tables of contents. They concluded some PDFs fare better than others while being converted.
- **Alternative Solutions for Document Processing**: Other users suggested alternatives for document processing. `@lightningralf` mentioned using [Llama Index Hub](https://llamahub.ai/l/smart_pdf_loader?from=all), while `@joshuasundance` mentioned using Azure Document Intelligence with Langchain. However, these alternatives weren't very suitable for `@visuallyadequate` given budget constraints.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Technical Outage**: A reported issue with all models not working, acknowledged by user `@harrison_2k`. 
- Query about starting study in NLP with a suggested exploration into transformer models as an initial step.
- Question about fine-tuning BERT Model on a Turkish Q/A dataset for better performance and accuracy.
- Conversation on creating an **EA for XM's standard account** for FX Automated trading and speculations about it.
- Users participated in a discussion on a **Neural Network Distillation paper**, with a link to the paper provided in both the *general* and *reading group* channels, supporting the exchange of ideas and further understandings. 
- Discussion around the **runtime error in the Nougat Transformers space** with a suggestion to try running it in a different space provided by user `@osanseviero`.
- Introduction of a new model **"Deepsex-34b"** by user `@databoosee_55130` and several shared links associated with the model.
- Shared interest in the value of an **Object, Property, Value list (OPV)** in the digital field.
- Concern about the layout of the required **dataset for a research paper** in the field.
- Sharing of an [article](https://www.newscientist.com/article/2408015-supercomputer-that-simulates-entire-human-brain-will-switch-on-in-2024/) on a **neuromorphic supercomputer that simulates the human brain**.
- Promotion of [Learning Machine Learning for Mathematicians](https://arxiv.org/abs/2312.04556) paper for more mathematicians to apply machine learning techniques.
- Presentation of a variety of educational videos and articles, including a [Retrieval Augmented Generation (RAG) video](https://www.youtube.com/watch?v=rhZgXNdhWDY), [XTTS2 Installation guide video](https://youtu.be/pNTTTwap12Y), and a [breakthrough in language models - MOE-8x7b](https://www.marktechpost.com/2023/12/12/mistral-ai-unveils-breakthrough-in-language-models-with-moe-8x7b-release/).
- Sharing of newly created models on HuggingFace, such as the *SetFit Polarity Model with sentence-transformers/paraphrase-mpnet-base-v2* and the *Go Bruins V2 Model*, with requests for guides on how to create similar models like the ABSA Model.
- Discussions primarily held on Wednesdays in the Reading Group channel in addition to the introduction to the paper *Distilling the Knowledge in a Neural Network*.
- Inquiry on **creating separate bounding boxes for each news article in a newspaper**.
- Discussion around the best **Encoder/Decoder for a RAG system**, with BERT being the current model in use, and a high interest in knowing more about model rankings for multilingual or code-specific use cases. The *Palm* model is frequently recommended.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (21 messages🔥): 
        
- **Fault in Models**: `@noone346` reported that all models appear to be down, a concern that was acknowledged by `@harrison_2k`.
- **Initiating NLP Study**: `@bonashio` sought advice on starting with NLP, already having a basic understanding of Deep Learning. `@jskafer` suggested an exploration into transformer models first.
- **Fine-tuning BERT Model on Turkish Q-A**: `@emirdagli.` asked for suggestions on fine-tuning a changeable BERT model on Turkish Q-A with a dataset of ~800 questions and ~60-70 answers.
- **Inquiry about MT4 and FX Automated Trading**: `@ronny28717797` mentioned their intent to create an EA for performing on XM's standard account, detailing specific performance expectations and preferences.
- **Discussion on Distillation Paper**: `@murtazanazir` initiated a discussion on a distillation paper, providing a [link](https://arxiv.org/abs/1503.02531) to the paper on arXiv.
- **Nougat Transformers Space Runtime Error**: `@pier1337` reported a runtime error on `@697163495170375891`'s Nougat Transformers space, asking if it can be run on a 16GB RAM CPU only. `@osanseviero` provided a link to try it in a different [space](https://huggingface.co/spaces/hf-vision/nougat-transformers).
- **Introduction of Deepsex-34b Model**: `@databoosee_55130` introduced a new model, "Deepsex-34b" and shared multiple links related to it.
- **Object, Property, Value (OPV) List**: `@noonething` sparked a discussion by questioning the value of an Object, Property, Value list in the field.
- **Dataset Layout Query**: `@largedick` asked about the layout of a dataset required for a paper.
- **Neuromorphic Supercomputer Article Share**: `@bread browser` shared a [New Scientist article](https://www.newscientist.com/article/2408015-supercomputer-that-simulates-entire-human-brain-will-switch-on-in-2024/) about a supercomputer that simulates the entire human brain.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (1 messages): 
        
- **Machine Learning for Mathematicians**: `@alexmath` shared a link to an [arXiv paper](https://arxiv.org/abs/2312.04556) encouraging more mathematicians to learn to apply machine learning techniques.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **Retrieval Augmented Generation (RAG) Video**: User `@abelcorreadias` shared a [video](https://www.youtube.com/watch?v=rhZgXNdhWDY) from Umar Jamil that explains the entire Retrieval Augmented Generation pipeline.
- **XTTS Installation Guide**: `@devspot` posted a link to a [YouTube video](https://youtu.be/pNTTTwap12Y) that explains how to install XTTS2, a popular Text-To-Speech AI model, locally using Python.
- **Breakthrough in Language Models - MOE-8x7b**: `@tokey72420` shared a [link](https://www.marktechpost.com/2023/12/12/mistral-ai-unveils-breakthrough-in-language-models-with-moe-8x7b-release/) to an article on MarkTechPost about Mistral AI's new breakthrough in language models with its MOE-8x7b release.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 messages): 
        
- **SetFit Polarity Model Creation**: `@andysingal` shared the link to his newly created model on HuggingFace, the [SetFit Polarity Model with sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/Andyrasika/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity), and encouraged others to try Aspect-Based Sentiment Analysis (ABSA) with it.
- **Guide Request**: `@fredipy` asked for a guide on how to create models similar to `@andysingal`'s ABSA Model.
- **Go Bruins V2 Model Showcase**: `@rwitz_` showcased his fine-tuned language model, [Go Bruins V2](https://huggingface.co/rwitz/go-bruins-v2), on the HuggingFace platform.


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (3 messages): 
        
- **Meeting Schedules**: `@chad_in_the_house` confirmed that meetings usually occur on **Wednesdays** and are often held in a dedicated thread.
- **Discussion on Distillation Paper**: `@murtazanazir` initiated a discussion on the paper [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), inviting others to join in for an elaborate understanding.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (2 messages): 
        
- **Bounding Box for Newspaper Segmentation**: User `@navinaananthan` asked for suggestions on **models or methodologies** to create a separate bounding box for each news article when uploading a newspaper. The user didn't get answers yet.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (5 messages): 
        
- **Best Encoder/Decoder for RAG System**: User `@woodenrobot` inquired about the most suitable encoder/decoder for a RAG system. They currently use **BERT**, but are open to other suggestions as they are unsure about BERT's ability to scale up as data grows over time.
- **Rankings for Multilingual/Code**: `@woodenrobot` expressed an interest in seeing model rankings specifically for multilingual/code use cases.
- **Palm as a Top Model Suggestion**: `@woodenrobot` noted, based on interactions with bard, that **Palm** appeared frequently as the recommended model.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- A [new architecture](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/) for **LangChain**,  namely `langchain-core` and `langchain-community`, was introduced by `@hwchase17`. This updated architecture will facilitate the development of context-aware reasoning applications using LLMs. Questions about this change are welcomed.
- Various technical inquiries and issues: 
    - New techniques for **tracking token consumption** per LLM call and **improving SQL query result summarization** were requested.
    - An issue with **LangSmith API** maintaining a log of LLM calls over the last 45 minutes was reported. 
    - Community members sought advice on **using LangChain for specific task completion, employing open-source alternatives for OpenAIEmbeddings**, and **enhancing chatbots to show similar records from a database**.
    - Discussion about using LangChain to improve Azure Search Service's multi-query retrieval and fusion RAG capabilities.
- In the **langserve** channel:
    - Users faced challenges with **callback managers in Llama.cpp** and the **integration of langserve, langchain, and Hugging Face pipelines**. 
    - A proposal was made to **create an issue about challenges with Langserve** and possible adjustments to use RunnableGenerators over RunnableLambdas. 
    - A request for **exposing the `agent_executor` in langserve**.
- A free 9-part **code-along series on building AI systems using OpenAI & LangChain** was shared. The first part specifically covers sentiment analysis with GPT and LangChain, MRKL prompts, and building a simple AI agent. The course can be found on this [link](https://www.datacamp.com/code-along/prompt-engineering-gpt-langchain) and a corresponding [YouTube Session](https://youtu.be/luRtEpFuwXA) is also available.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **New LangChain Architecture**:
    - `@hwchase17` shared a [blog post](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/) detailing the new `langchain-core` and `langchain-community`. The goal of this re-architecture is to make the development of context-aware reasoning applications with LLMs easier. This change is a response to LangChain's growth and community feedback, and implemented in a completely backwards-compatible way.
    - `@hwchase17` also offered to answer questions regarding this change.


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (12 messages🔥): 
        
- **Tracking Token Consumption Per LLM Call**: `@daii3696` inquired about a way to track token consumption for each LLM call in JavaScript. In response, `@pferioli` suggested using a callback and provided a code snippet, dealing specifically with the `handleLLMEnd` callback, which logs the token usage details.

- **Improve SQL Query Result Summarization**: `@jose_55264` asked about possible methods to expedite the summarization of results from SQL query execution. No specific solutions were provided in the given messages.

- **Using LangChain for Task Completion**: `@sampson7786` expressed a desire to utilize LangChain for completing a task with a specific procedure, seeking assistance on the platform for this issue.

- **LangSmith API Issues**: `@daii3696` raised concerns about an apparent issue with the LangSmith API as they were unable to trace their LLM calls for the past 45 minutes.

- **Open Source Alternatives for OpenAIEmbeddings in YouTube Assistant Project**: `@infinityexists.` is working on a YouTube assistant project and asked how HuggingFace can be used as an open-source alternative to OpenAIEmbeddings. They provided a link to the GitHub code used in their project.

- **Increasing Odds in Vector Search**: `@rez0` queried about the name of a function that splits retrieval into three queries in vector search to improve the chances of getting the wanted results.

- **Enhanced Search Capabilities in Chatbot**: `@hamza_sarwar_` is interested in enhancing their chatbot’s capabilities to display similar records from a database (e.g., vehicle info) when there are no precise matches to a user's query.

- **Azure Search Service Multi-query Retrieval/Fusion RAG with LangChain**: `@hyder7605` is working with Azure Search Service and wishes to integrate multi-query retrieval or fusion RAG abilities with Azure Cognitive Search using LangChain. They also aim to include advanced features like hybrid and semantic search in their query process but are unsure about defining filters and search parameters with LangChain.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (5 messages): 
        
- **Issues with Callback Managers in Llama.cpp**: User `@jbdzd` raised a challenge with callback managers in Llama.cpp during their use of langserve. They got a RuntimeWarning indicating a coroutine 'AsyncCallbackManagerForLLMRun.on_llm_new_token' was never awaited.
- **Struggles with Implementing Streaming in Langserve**: `@fortune1813` expressed trouble with the integration of langserve, langchain, and hugging face pipelines, especially with respect to streaming. They have investigated the notebooks but requested further clarification on proper streaming implementation.
- **Proposal to Create Issue in Langserve**: `@veryboldbagel` suggested that `@fortune1813` create an issue about their challenge in Langserve and share the full server script. They also brought to attention that RunnableGenerators should be used instead of RunnableLambdas for common operations, noting that it has been poorly documented.
- **Request for Agent Executor Exposure in Langserve**: `@robertoshimizu` queried about exposing `agent_executor` in langserve and shared a code snippet with an example. However, they are struggling as the input seems different when invoked in a Python script.


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **AI Code-Along Series**: User `@datarhys` shared a free 9-part code-along series on AI, with a focus on "Building AI Systems with OpenAI & LangChain", which was released by DataCamp. This series aims to guide learners from basic to more advanced topics in AI and LangChain.
  - The first code-along covers:
    - Performing sentiment analysis with GPT and LangChain
    - Learning about MRKL prompts used to help LLMs reason
    - Building a simple AI agent
- The instructor, Olivier Mertens, is praised for his ability to present complex topics in an accessible manner.
- To start this code-along, follow the link provided: [Code Along](https://www.datacamp.com/code-along/prompt-engineering-gpt-langchain)
- A YouTube session for the code-along is also available: [YouTube Session](https://youtu.be/luRtEpFuwXA)


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Conversation about a **Mixtral-based OpenOrca Test** initiated by `@lightningralf`, with the reference to a related [fxtwitter post](https://fxtwitter.com/mattshumer_/status/1733927635246305633?s=20) from the OpenOrca's development team. 
- Speculation on the **speed of the machine learning process**, proposed solution includes using server 72 8h100 to enhance performance.
- `@teknium's` declaration of **testing an unidentified model** and the need for further clarification of the said model.
- Inquiry from `@mister_poodle` on **ways to extend or fine-tune Mistral-OpenOrca for specific tasks**, namely boosting NER task performance using their datasets and generating JSON outputs.

**Alignment Lab AI Channel Summaries**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (5 messages): 
        
- **Discussion about a Mixtral-based OpenOrca Test**: `@lightningralf` asked `@387972437901312000` if they tested Mixtral based on OpenOrca, linking a [fxtwitter post](https://fxtwitter.com/mattshumer_/status/1733927635246305633?s=20)
- **Question about Process Speed**: `@nanobitz` expressed surprise about the speed of the process, with `@lightningralf` suggesting the use of server 72 8h100.
- **Unidentified Model Testing**: `@teknium` mentioned testing some model, but being uncertain about which one.


### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (1 messages): 
        
- **Extending/Fine-tuning Mistral-OpenOrca for Specific Tasks**: User `@mister_poodle` expressed interest in using their datasets to boost Mistral-OpenOrca's performance on an NER task with JSON outputs. They sought examples or suggestions for extending or fine-tuning Mistral-OpenOrca to achieve this goal.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion about fine-tuning open source models with recommendations for platforms such as [Replicate.ai](https://replicate.ai/) with the ability to run models with a single line of code shared by `@henriqueln7`.
- Introduction of an all-in-one LLMOps solution, [Agenta-AI](https://github.com/Agenta-AI/agenta), including prompt management, evaluation, human feedback, and deployment by `@swizec`.
- Recognition of the popularity of daily AI newsletters highlighted through a [Twitter post](https://fxtwitter.com/karpathy/status/1734659057938477174?s=46&t=90xQ8sGy63D2OtiaoGJuww) by Andrej Karpathy shared by `@swyxio`.
- Inquiry about the ideal size of language models for RAG applications by `@henriqueln7`, focusing the debate on balancing world knowledge and reasoning capacity.
- Sharing of a [YouTube video](https://www.youtube.com/watch?v=xNBiPd2H9J0) presenting insights into the future of AI, titled "AI and Everything Else - Benedict Evans | Slush 2023" by `@stealthgnome`.
- Request for access to Smol Newsletter Discord text and API made by `@yikesawjeez` with the goal of displaying content as embeds or creating a daily digest.
- Development progress update by `@yikesawjeez` on a SK plugin, planning to complete this before catching up on backlog items.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (11 messages🔥): 
        
- **Fine-tuning Open Source Models**: User `@henriqueln7` asked for recommendations on platforms to fine-tune open source models and presented https://replicate.ai/ as a suggestion. This platform allows one to run open-source models with one single line of code.

- **All-in-one LLMOps platform**: User `@swizec` shared a link to the GitHub repository for [Agenta-AI](https://github.com/Agenta-AI/agenta). This platform provides an all-in-one LLMOps solution, including prompt management, evaluation, human feedback, and deployment.

- **Daily AI newsletters**: `@swyxio` commented on the popularity of daily AI newsletters, putting forward a [Twitter post](https://fxtwitter.com/karpathy/status/1734659057938477174?s=46&t=90xQ8sGy63D2OtiaoGJuww) by Andrej Karpathy as an example.

- **Size of Language Models for RAG Applications**: `@henriqueln7` proposed a question about the ideal size of language models for RAG Applications. The question aimed to clarify whether a smaller or larger model would be better, considering the smaller model's limited world knowledge and the larger model's superior reasoning abilities.

- **AI Overview for 2023**: User `@stealthgnome` shared a [YouTube link](https://www.youtube.com/watch?v=xNBiPd2H9J0) to the 'AI and Everything Else - Benedict Evans | Slush 2023' video, offering insights into the future of AI.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (2 messages): 
        
- **Requesting Access to Smol Newsletter Discord Text and API**: User `@yikesawjeez` requests `@272654283919458306` for an API to access the raw text or .md files from the smol newsletter discord. An API would enable them to display the content as embeds using another user, Red, or create a daily digest.
- **Development on SK Plugin**: `@yikesawjeez` is currently working on a sk plugin and plans on finishing that before going through their backlog.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Conversation around **Anthropic Fundraising** with user `@res6969` mentioning a rumor about Anthropic raising an additional $3 billion, and jesting about **function calling issues**, suggesting that more funds might solve them.
- Discussion about **Fine-tuning for Email Parsing** initiated by `@robhaisfield`, regarding the number of examples needed for creating a JSONL file to parse email strings into structured objects.
- Share by `@robotums` of a Microsoft Research blog [article](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) about **Phi 2: The Surprising Power of Small Language Models**.
- User `@lhl` shared their experience with **Inference Engine Performance**, claiming a 50X performance boost after replacing some code with vLLM. They also compared transformers with other engines and shared a [GitHub repository](https://github.com/AUGMXNT/inference-benchmarks) containing detailed results.
- Dialogue on **prompting techniques** including the MedPrompt method and DeepMind's Uncertainty Routed CoT (Cooperative Output Transformations). This discussion also touched on OCR (Optical Character Recognition) usage and MS Research's achievements. All topics were introduced and discussed by users `@robotums` and `@dhruv1`.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/) (3 messages): 
        
- **Anthropic Fundraising**: User `@res6969` commented about a rumor that **Anthropic** is raising an additional $3 billion. 
- **Function Calling Issues**: `@res6969` made a jesting remark that maybe a few more billion dollars will get function calling to function properly.


### ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/) (2 messages): 
        
- **Fine-tuning for Email Parsing**: User `@robhaisfield` queried about the number of examples needed for creating a JSONL file to fine-tune a model to parse an email string into a structured object with nested replies. They asked if 30 examples would be sufficient.


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 
        
- **Phi 2: The Surprising Power of Small Language Models**: `@robotums` shared a [link](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) to a Microsoft Research blog article addressing the potential of smaller language models, accompanied by a list of contributors to the content.


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages): 
        
- **Inference Engine Performance**: User `@lhl` detailed their experience with different inference engines. They noted that after replacing some existing code with vLLM, they observed a **50X performance boost**. They also compared various inference options from transformers with other engines and shared their findings through a [GitHub repository](https://github.com/AUGMXNT/inference-benchmarks). The repo contains detailed results of the inferencing tests.


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/) (5 messages): 
        
- **Understanding When to Use OCR**: `@robotums` expressed curiosity about how to determine when OCR (Optical Character Recognition) is necessary for a page, asking if anyone knows how chatDOC accomplishes it.
- **Inquiring About the MedPrompt Technique**: `@dhruv1` asked if anyone has used the MedPrompt technique in their application.
- **Achievement by MS Research Using MedPrompt**: `@dhruv1` shared that [**MS Research**](https://link) has written a post about using the MedPrompt technique to surpass Gemini's performance on MMLU (Multiple-choice Machine Learning dataset from University of Minnesota).
- **DeepMind's Uncertainty Routed CoT Technique**: `@dhruv1` informed the channel that [**DeepMind**](https://link) has revealed a new technique called the uncertainty routed CoT (Cooperative Output Transformations) that outperforms GPT on MMLU.
- **Sharing the CoT Technique**: `@dhruv1` promised to share more about the CoT technique.


        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Event Timing Mix-up**: `@yikesawjeez` inquired about the event's time being marked at **9:30 PM**, which they had expected to be at **8:00 AM**. `@._z` clarified that the event is actually scheduled to be at **9:30 AM PST**. There was a minor confusion due to the time being marked incorrectly as PM instead of AM.
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- Mention of an individual's influence on **mistral licensing** in the #moe-main channel, with a comment from user ".mrfoo": *"Influencing mistral licensing I see. Nice!"*. 
- In the #off-topic channel, user "pradeep1148" shared a **YouTube link**: [https://www.youtube.com/watch?v=wjBsjcgsjGg](https://www.youtube.com/watch?v=wjBsjcgsjGg). The content of the video was not discussed.

**Skunkworks AI Channel Summaries**

### ▷ #[moe-main](https://discord.com/channels/1131084849432768614/1139310171076706464/) (1 messages): 
        
.mrfoo: <@1117586410774470818> : Influencing mistral licensing I see. Nice!


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=wjBsjcgsjGg


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **The Complete Data Product Lifecycle**: `@viv2668` shared links to two parts of an enhanced end-to-end guide on practical Data Products. 
   - [Part 1](https://moderndata101.substack.com/p/how-does-a-data-product-strategy) discussed conveying the value of the data product lifecycle to various stakeholders.
   - [Part 2](https://moderndata101.substack.com/p/the-complete-data-product-lifecycle) provides a condensed view of the lifecycle stages and discusses stakeholder involvement.
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
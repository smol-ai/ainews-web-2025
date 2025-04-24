---
id: 9aafab3a-27e5-44fa-bc6d-e939622430e1
title: '12/27/2023: NYT vs OpenAI'
date: '2023-12-29T10:14:01.623905Z'
original_slug: ainews-12272023-nyt-vs-openai
description: >-
  The news covers a detailed discussion on LM Studio's model performance,
  technical challenges, hardware optimization, community updates, and plugin
  releases, with focus on models like Phi2, Mistral 7b, openHermes 2.5, and
  hardware considerations including GPUs and Apple chips.
companies:
  - microsoft research
models:
  - phi2
  - mistral-7b
  - openhermes-2.5
topics:
  - large-language-models
  - model-performance
  - fine-tuning
  - hardware-optimization
  - gpu-acceleration
  - llm-api
  - vector-database
  - plugins
  - community-support
---


<!-- buttondown-editor-mode: plaintext -->Here is the [best thread](https://twitter.com/ceciliazin/status/1740109462319644905) on the NYT OpenAI lawsuit today:

 ![image.png](https://assets.buttondown.email/images/8500cd54-405d-49fb-bcf5-d827b22e17c0.png?w=960&fit=max) 

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Model Performance**: Extensive dialogue on model performance across various contexts, with specific focus on Phi2's U.S. history knowledge, comparison between OpenHermes 2.5 Mistral 7b and Phi2, and experience with LLama 2 chat 7b vs 13b. Highlighted the potential of fine-tuning existing models for improved performance.
- **Technical Challenges and Solutions**: Recurring conversation on technical challenges within LLM API and also software and configuration-related issues. Notable challenges include maintaining conversation history within LLM API, optimizing GPU layer settings to enhance inference speed, model configuration, handling of errors, and installation and debugging of the ChromaDB Plugin for LM Studio.
- **Model usage and choice**: Discussion around specific usage cases, like grading essays, family-friendly chatbot, etc., and choice of right model based on performance and model halluncination incidents. Attention to the role of 'assistant' in LM studios was also discussed.
- **Hardware Discussion**: In-depth discussion on hardware selection and optimization for better model performance. Topics ranging from configuration differences between DDR4 and DDR5, graphics card selection for Local Light Models and Stable Diffusion experiments, building AI-capable server racks with multiple GPUs, to the potential of Apple's new M1 and M3 chips and AMD's AI CPUs for running AI models.
- **Community and Plugin Update**: Demonstrated community’s collective knowledge sharing and problem-solving capacities with issues related to configuration, software errors and plugins. Notably, the [new release of the ChromaDB Plugin for LM Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2) was announced which allows image searching in the vector database.

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (284 messages🔥🔥): 
        
- **Performance of the Phi2 Model**: `@dharmabum125` and `@heyitsyorkie` discuss the performance, specifically in the context of U.S. history knowledge, of the Phi2 model developed by Microsoft Research (MR). `@dagbs` defends Phi2, citing a blog post explaining how the MR team intended Phi2 to excel mainly in coding, reasoning, and commonsense knowledge, not necessarily in history.
- **Comparison of Model Performance**: `@heyitsyorkie` compares the performance of OpenHermes 2.5 Mistral 7b and Phi2, arguing that the former can answer a specific history question correctly on the first try.
- **Improving the Performance of Mistral Hypermodel**: `@dagbs` suggests finetuning Mistral by feeding it with a historical dataset to improve its ability to answer history-related questions.
- **Questions about LM Studio**: Users `@code_3090` and `@queuelabs` ask questions about how to maintain conversation history with the LM Studio LLM API and what optimization techniques are employed by LM Studio, respectively. `@thelefthandofurza` provides a solution to `@code_3090`'s problem, explaining that previous messages need to be appended to the new message to retain conversation context.
- **Performance Concerns**: `@sheepdestroyer` reports consistently low inference speed with different models and settings. `@heyitsyorkie` advises trying smaller models, commenting that the reported speeds are consistent with running a 34b model on `@sheepdestroyer`'s specified hardware configuration.
- **Usage of Multiple GPUs**: Users `@rugg0064`, `@fabguy`, and `@psipiai` discuss how multiple GPUs can be used in running a model. They agree that distributing the computation of layers across multiple GPUs would not lead to faster times.
- **Running Multiple Models Simultaneously**: `@yoangab` figures out how to run multiple instances of LM Studio on a Mac to operate multiple servers concurrently. The command shared is `open -n LM\ Studio.app/`.
- **Inference Speed Adjustment**: A discussion sparked by `@yunta0`'s request for advice on tweaking GPU layers and CPU threads settings to optimize inference speed concludes with `@ptable` recommending adjusting the GPU layers to achieve approximately 80-90% VRAM usage.
- **New Release of the Chroma Database Plugin for LM Studio**: `@vic49.` announces a new version of the ChromaDB Plugin for LM Studio.


### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (43 messages🔥): 
        
- **Clearing Context within LM Studio Conversations**: `@the.one.n.only.prof` inquired about clearing context within an ongoing chat conversation in LM Studio. `@fabguy` highlighted that it's not possible within the same chat and the only way to clear context would be starting a new chat.
- **Multiple GPUs and GPU Preferences in LM Studio**: `@septagraam` sought clarification on how to allocate a specific model to run on a single GPU where the LM Studio system resides on a multi-GPU platform. He pointed out the "Open GPU Preferences JSON" option without knowing its syntax specifics. `@fabguy` advised customizing the tensor_split value in the preset JSON, suggesting a `100,0` allocation to run all layers on the first GPU.
- **Grading Essays with LLMs**: `@emerance`, an English teacher sought advice on utilizing language models in grading essays. Contributions from `@fabguy` and `@thelefthandofurza` advocated for clear objective grading guidelines and limiting context to evaluate each essay against the rubric. To enhance the consistency of grading, `@fabguy` proposed lowering the model temperature. Furthermore, `@yagilb` recommended breaking down the grading task into smaller, manageable portions for improved effectiveness.
- **Hugging Face Model Issues in LM Studio**: `@jvaleski` and `@madbits.` encountered difficulty when searching and loading Huggingface models in LM Studio respectively. `@heyitsyorkie` clarified that some model versions aren't supported in the Linux 0.2.8 build and provided the link to the 0.2.10 version, cautioning about potential stability errors.
- **Retrieving Embeddings via the LM Studio API**: `@johnv_jv` queried if the LM Studio API supports retrieving embeddings for a RAG implementation. `@vic49.` responded negatively, but highlighted the ability to incorporate text that originates as embeddings from a vector database.
- **Role of 'Assistant' in LM Studio**: `@sladix` asked how the 'assistant' role in LM Studio model is used. `@yagilb` and `@borick` suggested that the role is used for seeding the context with assistant messages and customizing the system behavior respectively.


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (41 messages🔥): 
        
- Discussion on **Hermes 2.5 vs Hermes 2 Performance**: Users `@clickclack777` and `@dagbs` discussed the performance and size of different model versions. `@clickclack777` mentioned that they found a mix of the default Phi-2 template in 0.2.10 with Dolphin prompt to be effective. `@dagbs` was looking forward to the release of mistral 7b, anticipating it to be considerably smaller in size (~8GB instead of ~25GB).
- Discussion on **Llama 2 chat 7b vs 13b**: `@brunodanuy` was inquiring about the best model for a family-friendly chatbot. `@heyitsyorkie` suggested the default llama 2 chat 7b, however, `@brunodanuy` had already been using the 13b version and was pleased with its performance. The conversation then shifted towards exploration of response generation speed, where they considered lower model sizes for a more efficient generation time.
- Discussion on **Phi 2 Model Performance**: `@brunodanuy` expressed interest in trying out the phi 2 model due to its smaller size compared to llama 13b. `@heyitsyorkie` warned that although it had a smaller size, it might not be as recommended for most tasks and could produce results that are "meh". `@brunodanuy` tried out the model and agreed that the responses were less satisfactory when compared to llama 13b.
- Discussion on **GPU Acceleration and Model Hallucination**: `@rarisma` asked whether Phi worked with GPU acceleration/layers and if GPU acceleration might be causing hallucination. `@heyitsyorkie` suggested that it might just be the model and that Phi-2 isn't very powerful and can easily get confused. However, according to `@heyitsyorkie`, lowering the number of layers to around 5 could potentially help tackle the hallucination problem.
- **About AWQ models**: `@xsnypsx` asked if AWQ models could be run either now or in the near future, to which `@heyitsyorkie` replied that AWQ support isn't currently available in lmstudio but it might come with time when the llama.cpp pr gets merged into the main branch.


### ▷ #[🛠-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/) (26 messages🔥): 
        
- **Configuration for model**: `@pdg` shared the configuration for a model, [OpenChat 3.5 1210 Starling SLERP - GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-starling-slerp-GGUF) which seems to be functioning well. 

- **Issues with Configuring Certain Models**: `@pdg` highlighted issues with some of the models particularly [Finance LLM - GGUF](https://huggingface.co/TheBloke/finance-LLM-GGUF) and [Solar 10.7B Instruct V1.0 Uncensored - GGUF](https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF). The main challenge was configuring the prompt format to get proper responses.

- **Models Error Handling**: `@heyitsyorkie` also reported experiencing similar errors with the already included llama-2-chat preset and `@pdg` confirmed encountering the same.

- **Hardware Optimization Issue**: `@badnature` raised an issue concerning low performance on a 7B Mistral model despite having 32 GB of VRAM available. `@fabguy` suggested checking for distribution across multiple cards and tweaking the tensor_split variable.

- **Optimizing Inference Speed**: `@badnature` later reported an improvement in the inference rate from 9tk/sec to 13 tk/sec after modifying load parameters. Discussed further ways to improve inference speed such as using lower quants, adjusting CPU threads, and updating drivers. `@fabguy` also mentioned the benefit of using a smaller context size.


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (53 messages🔥): 
        
- **ChromaDB-Plugin-for-LM-Studio V3.0.2 Release**: `@vic49.` shared the [news of a new ChromaDB Plugin release](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2) which allows image searching in the vector database in LM Studio. 
- **Installation and Debugging of Plugin**: User `@motocycle` downloaded and installed the plugin on their Mac, encountering several errors during setup. `@vic49.` guided them through debugging, and suggested adding the line `"import os"` to their `initialize.py` script to resolve one of the errors.
- **Torch Backend Error**: `@motocycle` encountered another issue related to the torch backend, with `@vic49.` suggested replacing the `get_best_device` function within `loader_vision_llava.py` to fix it. However, this led to another error regarding a 'boolean' object not being callable.
- **GPU and Settings Inquiry**: `@vic49.` highlighted that choosing 4-bit in settings may cause issues as it relies heavily on the 'bitsandbytes' library which could default to using the GPU, and suggested trying float16 in the "quant" option instead to make it run native Pytorch.
- **Thread Diverted to DMs** `@yagilb` requested `@vic49.` and `@motocycle` to shift their in-depth debugging to Direct Messages, which they agreed to.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (110 messages🔥🔥): 
        
- **Inference Speed and Memory**: User `@dagbs fabguy` discussed that part of the inference speed on GPU is the faster memory with a reference to the differences between DDR4 and DDR5. The comment sparked a discussion on the impact of different hardware configurations on speed and performance.
  
- **Hardware Choices for AI Use**: User `@Pierre-jean Lainé` asked for hardware choice recommendations given a certain budget constraint. Choices were compared based on performance for AI tasks, such as **Local Light Models** (LLMs) and **Stable Diffusion** experiments. Several users (`@heyitsyorkie`, `@rugg0064`) recommended the RTX 3060 12GB version for its superior performance in LLM usage.

- **Potential for Building AI-Capable Racks**: Users `@xenorhon` and `@heyitsyorkie` entertained the idea of building a server rack with multiple GPU cards for running AI models. The Nvidia Ada 6000 and RTX 3090 models were discussed as potential options. 

- **Discussion on Apple's M1 and M3 Chips for AI**: `@rugg0064` discussed the potential of apple's new M1 and M3 chips for running AI models noting that they are a cost-effective way to run models due to the chips' high memory sizes and high bandwidths. 

- **AMD's AI-Specialized CPUs**: `@totallybored` shared a link to [AMD's AI CPUs](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) and commented on waiting for the next generation of AI-specialized laptops.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Debate over the cost efficiency of OpenAI services, with users discussing the differences between GPT-4 and ChatGPT. There were also issues raised regarding message limits, accessibility of custom GPT chatbots, and the value of creating a vector memory database.
- Extensive discussion on AI-based text correction, AI creation with different programming languages, and utilizing AI for business planning and software development. User queries were addressed with various solutions like prompt engineering, utilizing predefined libraries, and focusing on AI assistant/bot distinction.
- Detailed discussions on evaluating AI models like GPT-4 and Novo ChatGPT, with users expressing both satisfaction and dissatisfaction with response quality. In addition to discussing features and limitations, users also discussed legal implications like New York Times (NYT) vs OpenAI lawsuit and potential legal issues in web scraping.
- Technical issues involving file uploading, 404 errors, API keys, and quota issues were reported and addressed. Furthermore, content flagging and policy enforcement were discussed in relation to the use of AI for therapy and potential content violations.
- Consideration of AI's role in assisting with writing tasks by generating more tokens and improving output quality. References to specific tools and resources like GitHub Copilot and [OnboardAI](https://app.getonboardai.com) were highlighted.
- Query on utilization of 'upload file' feature in ChatGPT and its potential as a knowledge base prompted responses explaining its functionality as a reference document. Furthermore, inconsistency in AI Assistant API information raised concerns.
- Discussions on the best ways to engage with ChatGPT to get more detailed and goal-specific responses were addressed, with suggestions on reframing queries and understanding the limitations of AI. The value and application of custom instructions and 'cookie cutter' prompts were thoroughly discussed.
- Users inquired about the feasibility of using fine-tuned Large Language Models (LLMs) for multi-step and chained tasks. The functional aspects and limitations of models like ChatGPT and Custom GPT were brought into focus.
- Overall, the discussions presented a broad spectrum of topics including platform capabilities, technical challenges, legal implications, user experiences, and strategies for maximizing value from AI technologies.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (55 messages🔥🔥): 
        
- **AI for Text Correction**: User `@silconumbathree` sought advice for a powerful AI capable of correcting a large volume of text (approximately 30,385 characters or 5,547 words) for spelling, grammar, and other mistakes. Several AI models were discussed, including **GoogleDeepMind** and **GPT-4**. In response, `@michael_6138_97508` advised that results could vary with each try since **ChatGPT** is nondeterministic. He further recommended trying out different prompt engineering techniques using a guide from [OpenAI Documentation](https://platform.openai.com/docs/guides/prompt-engineering/strategy-test-changes-systematically).

- **AI Creation with Java or Golang**: `@blazin77kong_89701` inquired about creating an AI using Java or Golang, specifically on a low-spec machine (Intel i3, 4GB RAM, 7-years-old). Multiple users, including `@solbus`, `@bambooshoots`, `@lugui`, and `@michael_6138_97508` conveyed the resource-intensive nature of self-hosted AI, suggesting utilization of existing AI tools or learning the basics of machine learning using Coursera, Udemy, or similar platforms. `@bambooshoots` recommended a text-generation project on GitHub, namely oobabooga/text-generation-webui, and information available on [Hugging Face](https://huggingface.co/).

- **Building AI with ChatGPT**: User `@mo___111` sought help developing a business plan using **ChatGPT**, specifically employing an SBA structure. `@solbus` suggested creating a custom GPT for this task with a focus on crafting an assistant instead of a bot through the **GPT builder** on chat.openai.com.

- **Software Development AI Assistants**: `@dydzio` asked about AI assistants that can be personalized by reading entire GitHub repositories. `@zeriouszhit` pointed towards GitHub Copilot as a solution; however, it was noted that the service is currently limited to VSCode. Another potential tool was brought forward by `@dydzio`, namely [OnboardAI](https://app.getonboardai.com), a service for beginners to learn from mistakes.

- **AI for Writing Tasks**: `@sinisterj12` expressed anticipation for language models like **ChatGPT/Gemeni** allowing for more tokens, hoping to write an entire biography through its aid. Meanwhile, `@afterst0rm` voiced satisfaction with the prompts and response quality of **GPT-4 Turbo**, sharing a link for a test performed on the model via [OpenAI Playground](https://platform.openai.com/playground/p/WLMBiIjhd4V7IYq1LdKFshFR?model=gpt-4-1106-preview&mode=chat).


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (332 messages🔥🔥): 
        
- **ChatGPT and OpenAI Pricing**: Users such as `@1aztralz` debated the cost efficiency of OpenAI services like GPT-4 compared to ChatGPT, and links to pricing information and token counting were provided by `@solbus`.
- **OpenAI Updates and Issues**: Users discussed the potential release of GPT-4.5 and questioned the current state of GPT-4, with `@millymox` stating that GPT-4 has become nearly unusable. `@pyhelix` asked about any updates regarding GPT-4.5.
- **File Uploading Errors**: `@euroeuroeuro` experienced issues with uploading files, regardless of the file type or browser used, prompting advice and discussion from other users including `@.pythagoras`.
- **NYT vs OpenAI Legal Discussion**: The New York Times (NYT) lawsuit against OpenAI was a recurrent topic in the chat, users like `@gamerg.` and `@lugui` providing their opinions on the possible financial incentives and damaging consequences for OpenAI and the AI industry as a whole.
- **Web scraping Discussion**: `@maxdipper` asked for help in web scraping for a Discord bot and was informed by `@kaveen` about the potential legal issues and difficulties such as anti-bot protections in trying to automate tasks with sites like Doordash.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (110 messages🔥🔥): 
        
- **Performance Discussion on ChatGPT-4**: The model's performance was evaluated by `@oscarrsm` and `@nlp.sensei` among others. `@oscarrsm` expressed dissatisfaction with the quality of scripts generated by ChatGPT-4, while `@nlp.sensei` sought guidance on how to access GPT-4 for comprehensive code generation and assessment. `@solbus` helped clarify that usage of the ChatGPT Plus API and GPT-4 involves separate billing and setup mechanisms.

- **AI with Java or Golang**: There was a query from `@blazin77kong_89701` about the feasibility of building AI with Java or Golang. 

- **404 Error Discussion**: A 404 error encountered by `@ariel129` during a NodeJS operation in a Docker environment prompted a back-and-forth with `@lugui`, leading to suggestions about checking environment variables and request URLs, bodies, and headers.

- **Model Evaluation and Preferences**: `@lugui` and `@elektronisade` had discussions with `@pudochu` regarding the strengths and weaknesses of various AI models, including those provided by OpenAI, the community (such as Hugginface), and Google. The conversation touched upon issues like language support, latency, and geographic availability.

- **Content Flagging and Policy Enforcements**: `@mutant_llama1` raised an issue about inability to assess potential content policy violations due to content flagging. A similar concern was voiced by `@mka79` whose therapeutic text session prompts were being flagged. `@lugui` pointed out that the use of AI for therapy might violate OpenAI's content policy.

- **API Key and Quota Issues**: `@shivasai2023` expressed difficulty in making API calls after purchasing a $20 plan for ChatGPT-4 and felt they had not received the expected credit. `@solbus` clarified the separation between ChatGPT Plus usage/billing and API usage/billing.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (12 messages🔥): 
        
- **Message Limits for GPT-4**: User `@greverden` inquired about the current hourly message limit for GPT-4. `@solbus` responded that the **current cap is 40 messages per 3 hours**. Custom GPTs have a lower cap of **25 messages per 3 hours**.
- **Accessibility of Custom GPT Chatbots for Plus Subscribers**: `@linkz.lynx` asked whether custom GPT chatbots are available to users with Plus Subscriptions globally, specifically in Poland. `@solbus` confirmed that **anyone with ChatGPT Plus can use and create custom GPTs**, directing users to the [editor](https://chat.openai.com/gpts/editor) and [discovery](https://chat.openai.com/gpts/discovery) pages on the OpenAI website.
- **Value of Vector Memory Database Versus GPT Plus**: `@smokzz` queried about the value of creating a vector memory database and using the OpenAI API compared to subscribing to GPT Plus.
- **Enquiry About Fine-Tuning Large Language Models (LLMs) for Multi-Step Tasks**: `@ruili09` posed a high-level query about the most effective way of using fine-tuned LLMs for compositional tasks. They asked whether it's better to **utilize one large model to execute all tasks simultaneously**, or to **use multiple fine-tuned models each tackling a separate task in a chained operation**.
- **Limitations and Usage of ‘Upload File’ Feature in Custom GPTs**: `@milodal` asked about the restrictions of the 'upload file' feature in custom GPTs and if these files could be used for cross-referencing details. `@solbus` provided an [FAQ link](https://help.openai.com/en/articles/8555545-file-uploads-with-gpts-and-advanced-data-analysis-in-chatgpt), explaining that **these files essentially work as reference documents** and the GPT can query or retrieve information from them.
- **Issues with AI Assistant API Information Consistency**: `@aviassaga` shared their experience with the assistant API occasionally providing recommendations outside the officially provided data, contradicting instructions given to the AI.
- **Custom GPT and Knowledge Base Functionality**: `@milodal` further probed into whether the custom GPT can automatically search the uploaded files to cross-reference user's query and utilize the content to provide new or additional info in response. They also asked if this could function as a private database that is continuously updated and provides accurate information.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (18 messages🔥): 
        
- **ChatGPT User Engagement Strategy**: User `@thebookoforlando777115` asked how to get more detailed output from **ChatGPT** instead of general responses by using the example of creating a short curriculum plan for a cyber security degree. `@eskcanta` suggested rephrasing the query to yield better results, including providing ChatGPT with specific goals or a series of goals, mentioning her background knowledge, and asking guidance about free or low-cost certification options. Also, it was mentioned that certain website information accessed by **ChatGPT** may not be completely accurate due to some sources having a robot.txt page that inhibits data crawling.
- **Engagement with the AI model**: `@eskcanta` further elaborated on the approach to engaging with the AI model, reminding that users should always verify the AI’s provided facts as it has a tendency to 'make things up' especially when repeatedly asked for specific information.
- **Using Custom Instructions**: When `@thebookoforlando777115` asked about ideal instructions for **ChatGPT**, `@eskcanta` advised experimenting and reminded that the effectiveness of the instructions depends upon the user's specific goals.
- **Role of "Cookie Cutter Prompts"**: In response to @beanz_and_rice's question about 'cookie cutter prompts', `@eskcanta` shared a few examples of her standard queries used with **ChatGPT** to better UNDERSTAND a word, concept or name or to interpret certain instructions. However, she showed skepticism on entirely relying on such prompts, and she encourages users to state specifically what they want from the model.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (18 messages🔥): 
        
- **ChatGPT's Usefulness and Limitations**: [@thebookoforlando777115](https://discord.com/channels/123456789101112131/123456789101112131) voiced their frustration over ChatGPT's general responses, while expressing a desire for a **tailored cybersecurity curriculum plan**. Responded `@eskcanta`, detailing how the AI is quite literal and may not make unique curriculums. They suggested reframing the query to ask about low-cost or free certification options and comparing different course series.
- **Google Access and Robot.txt**: `@eskcanta` explained that ChatGPT's access to some college curriculums could be blocked by the websites' robot.txt pages, implying some degree of manual research might be required.
- **Custom Instructions (CI)**: `@eskcanta` advised `@thebookoforlando777115` to experiment with Custom Instructions in ChatGPT, indicating that they are conversation-specific and don't impact ongoing exchanges. They added that "ideal or perfect" CIs depend on individuals' needs.
- **Cookie Cutter Prompts**: The term was used by `@thebookoforlando777115` to refer to seemingly universal, premade prompts. `@eskcanta` explains that cookie cutter prompts can be useful, offering a couple of prompts they use often. However, a suggested cookie cutter prompt by `@beanz_and_rice` was not found useful by `@eskcanta`, who characterized the prompt as unguided and hallucination-prone.
- **Conversation with ChatGPT about Cookie Cutter Prompt**: `@eskcanta` engaged in a conversation with an instance of ChatGPT using `@beanz_and_rice`'s cookie cutter prompt. The AI initially didn't provide a satisfactory response but eventually to "guessing" the user's intention, which `@eskcanta` found still unsatisfactory, highlighting the importance of clear and precise instructions for the AI.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Discussion on difficulties revolving around the use of Jax for dynamic slicing and alternative solutions such as padding tensors to make the code compatible with JIT.
- A shared INFO log detailing performance metrics of OpenChat instance, highlighting token throughput, GPU KV cache usage, and outstanding requests.
- Exploration and comparison of Jax and Pytorch programming for specific tasks e.g., physics programming.
- Inquiry into *sliding_window_attention_jax* function in pursuit of replicating recurrent neural networks functionalities.

- Announcement and sharing of new **Capybara dataset**, consisting over 10,000 multi-turn examples, further discussed in general channel. 
- Highlights on the [Youtube video](https://www.youtube.com/watch?v=A_RDhk9uEy4) demonstrating the implementation of Mixture Of Vectors for fine-tuning.
- Examination on the performance of OpenChat 3.5 awg 4bit on specific hardware setup.
- Queries and an ensuing discussion about choosing betting between various hardware crypto wallets: Ledger, BitBox02, Trezor, and Coldcard.
- Proposition to create a filtering good and bad training data using multimodal training model.
- Introduction of the idea for the creation of 'LibreAI', a non-profit organization aiming to fund AI research and development.

- Share of a [running LLM contamination detector](https://huggingface.co/spaces/Yeyito/llm_contamination_detector) created by Yeyito on Hugging Face.
- Discussion on the use of PowerInfer, a GPU-CPU hybrid interface that produces speed improvements; the provided [link to an article](https://pub.towardsai.net/powerinfer-11x-speed-up-llm-inference-on-a-local-gpu-ddb66c6cba80) furthers this discussion.
- Exploration on the idea of adding a bayesian probabilistic layer on top of PowerInfer for future improvements.

- Deep dive into the drawbacks of pushing **Mistral** beyond the 8k extension and discussion surrounding model merging and fine-tuned (FT) models with the highlight of the newly released open-source model, [CodeNinja](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B).
- Thorough explanation on curating training datasets from `@ldj` and `@solbus`, accompanied by the challenges faced during the process.
- Discussion about the performance of language models on managing long context lengths focusing on the importance of coherent and accurate conversation skills.
- Examination on the ongoing New York Times lawsuit against OpenAI and potential fallout including the issues of copyright and geopolitical implications.
- Conversation on the democratization of AGI (Artificial General Intelligence) development to prevent dominance by large entities with more resources and the importance of open-source solutions keeping pace with research progresses.

- Inquiry on creating a synthetic QA dataset using a T4 GPU with Turkish Wikipedia data due to RAM constraints and the need for alternative inference providers.
- Report on sources to obtain `books3.tar.gz` file, as all previously known links were down.
- Query on the correct chat format for using yarn-mistral-128k models.
- Discussion on alternatives like TogetherAI API, OpenRouter, and Google Gemini as cost-efficient options for AI inference, and also suggested using Ollama with Web-UI for displaying UI on a different server
- Inquiry about setting up Hermes 2 version on runpod along with required model loader parameters and UI template adjustments for a smooth setup.

- Announcement of new version [release for the vector database plugin](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.0) by `@vic49` for LM Studio, which now includes a feature to search images.
- Discussion on incorporating nous-hermes vision models into LM Studio plugin, with a gap highlighted in finding useful online code samples.
- Ability to contribute to the project shown by various members, with a known [update released](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2) and potential challenges discussed.
- Provided inference code insights for Obsidian, which would be similar to bakllava, with a commitment to keep the community updated.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (27 messages🔥): 
        
- **Difficulty with Jax and Dynamic Slicing**: User `@joey00072` shared code for the *sliding_window_attention_jax* function and aired frustration about the difficulty of implementing dynamic slicing in Jax. Other users offered suggestions on alternative venues to seek help, and `@rishiiyer` offered an immediate solution by suggesting to pad tensors instead of using dynamic slicing, which makes the code compatible with JIT.
- **Jax Versus PyTorch**: A brief discussion about Jax ensued, with `@rishiiyer` and `@joey00072` observing that Jax can be difficult for tasks that require dynamic functionality. Despite this, `@rishiiyer` mentioned a liking for Jax for physics programming.
- **OpenChat Performance Metric**: User `@fullstack6209` shared a series of INFO messages detailing performance metrics of an instance of OpenChat using an RTX 2080ti GPU with 11GB of VRAM. The logs detailed token throughput, GPU KV cache usage and outstanding requests.
- **Discussion on Sliding Window Attention Usage**: `@rishiiyer` inquired about `@joey00072`'s purpose for utilizing sliding window attention. `@joey00072` mentioned the goal of using sliding window attention with hidden states, inspired by recurrent neural networks functionalities.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (61 messages🔥🔥): 
        
- **Capybara Dataset Announcement**: `@ldj atgctg` posted about the new [Capybara dataset](https://huggingface.co/datasets/LDJnr/Capybara), which consists of over 10,000 multi-turn examples.
- **Efficient Fine-Tuning Using Mixture Of Vectors Implementation**: `@pradeep1148` shared a [Youtube video](https://www.youtube.com/watch?v=A_RDhk9uEy4) about the implementation of Mixture Of Vectors.
- **OpenChat 3.5 Performance**: `@fullstack6209` reported that OpenChat 3.5 awg 4bit runs well on a 2080ti with 11gb and can handle an 8192 maximum context at 40-70 tokens/sec.
- **Hardware Crypto Wallets**: `@john0galt` sought advice on choosing between various hardware crypto wallets: Ledger, BitBox02, Trezor, and Coldcard.
- **Improving Training Data**: `@skadeskoten` suggested a need for a model to filter good and bad training data, possibly using a multimodal model.
- **"LibreAI" Non-Profit Proposal**: `@narvikd` proposed creating a non-profit organization, potentially named 'LibreAI', to fund AI research and development.
- **Using OS Models for Profit**: `@gabriel_syme` emphasized the importance of focusing on solving significant problems instead of seeking VC funding.
- **Local LLMs Interactions**: `@jason.today` shared a link to a [GitHub repository](https://github.com/jasonjmcghee/rem) detailing an open-source approach for enhancing interactions with local LLMs.
- **EU Tipping Regulations**: `@fullstack6209` mentioned new European Union regulations on AI tipping.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (15 messages🔥): 
        
- **LLM Contamination Detector**: `@.beowulfbr` shared an [interesting link](https://huggingface.co/spaces/Yeyito/llm_contamination_detector) to a running LLM contamination detector created by Yeyito on Hugging Face. `@skadeskoten` also considered the tool to be interesting.
 
- **PowerInfer - GPU-CPU Hybrid Interface**: `@skadeskoten` shared a [link](https://pub.towardsai.net/powerinfer-11x-speed-up-llm-inference-on-a-local-gpu-ddb66c6cba80) to an article about PowerInfer, a GPU-CPU hybrid interface that results in significant speed increases. Some discussion followed with `@.beowulfbr` and `@georgejrjrjr` relating to the use of PowerInfer with other technologies.
 
- **Bayesian Probabilistic Layer**: `@skadeskoten` suggested to add some kind of a bayesian probabilistic layer on top of PowerInfer for improvements.
  
- **Quasi-Sparse Techniques**: `@georgejrjrjr` stated that these quasi-sparse techniques are only applicable to ReLU models and of limited use to SWiGLU models. In terms of inference acceleration results, georgejrjrjr mentioned the under-appreciated EAGLE. 

- **Mixtral and Sparsification**: `@georgejrjrjr` discussed about the potential prospects of  Mixtral, a quasi-sparse technique and wondering about post-conference updates over the sparsification code released by TD.

- **3D Gaussian Splatting**: `@yikesawjeez` shared a [link](https://efficientgaussian.github.io/) to an article about 3D Gaussian Splatting, a method which aids in real-time rendering and accelerated training, albeit with substantial memory resource requirement.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (377 messages🔥🔥): 
        
- **Model Performance in Specific Workloads and Model Merging Discussion**: Users engaged in a conversation about the performance of different models. `@Imonenext` mentioned the drawbacks of pushing **Mistral** beyond an 8k extension without continued pretraining. `@.beowulfbr` argued in favor of model merging and fine-tuned (FT) models despite some community backlash against these practices. They also highlighted their newly released open-source model, [CodeNinja](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B). Participants then discussed how merged models often game evaluation leaderboards, alienating some users.

- **Curation of Training Datasets**: Users expressed the challenges and methods of curating datasets for training language models. `@solbus` clarified that uploaded files serve as reference material rather than modifiers of the AI's base knowledge. In his detailed explanation, `@ldj` shared his process for generating and filtering datasets, highlighting how he created the [Capybara dataset](https://huggingface.co/datasets/LDJnr/Capybara). 

- **Handling Larger Context Lengths in Models**: Starting with `@spirobel`, a discussion about how well language models manage long context lengths arose. `@ldj` noted Capybara's impressive performance in long context tests and emphasized that the ability to manage long conversations coherently and accurately is a significant measure of a model's capabilities. 

- **Discussion on Open Source Movement and Copyright Issues**: Users debated the potential fallout from the ongoing New York Times lawsuit against OpenAI, expressing concerns about possible future restrictions on using copyrighted content for AI training. They also highlighted the potential geopolitical implications if certain regions, particularly China, are less restrictive.

- **Democratizing AGI Development**: A conversation about Artificial General Intelligence (AGI) unfolded, with `@gabriel_syme` voicing concerns about how larger entities with more resources could dominate AGI development, making it imperative that open source solutions keep pace. There was also discussion about the performance of commercial and open source models, with user `@teknium` noting that corporate attainment of AGI isn't inherently negative, provided they achieve it through research and not via regulatory advantages.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (9 messages🔥): 
        
- **Creating Synthetic QA Dataset with Turkish Wikipedia Data**: User `@umarigan` asked how they can create a synthetic QA dataset with Turkish Wikipedia data using a T4 GPU and without using large models like llama-70 due to RAM constraints. They suggested using models from inference providers, like llama-70 from anyscale, as it costs 1$ for a million tokens.
- **Accessing Books3**:`@fullstack6209` is looking for sources to obtain `books3.tar.gz` file, reporting that all links they had tried were down.
- **Correcting Chat Format for Yarn-Mistral-128k Models**:`@jpham` requested guidance on the correct chat format to use for yarn-mistral-128k models when there's a system prompt, as their attempts didn't yield satisfactory results.
- **Displaying UI for GPUs in remote servers**: `@narvikd` sought advice on what UI to use when their GPU is on a different server, noting that they typically use exl2 formats.
- **Alternatives for AI Inference Services**: `@night_w0lf` suggested to `@umarigan` and `@narvikd` alternatives like TogetherAI API, OpenRouter, and Google Gemini as cost-efficient options for AI inference, and also suggested using Ollama with Web-UI for displaying UI on a different server.
- **Setting Up Hermes 2 on Runpod**: `@rational_adherence` enquired about setting up the Hermes 2 version on runpod, specifically seeking guidance on selecting the model loader and adjusting parameters. The user mentioned working with a one-click UI template.


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (10 messages🔥): 
        
- **Vector Database Plugin for LM Studio Release**: `@vic49.` announced the release for the vector database plugin for LM Studio, which now includes a feature to search images. The [release can be found here](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.0).
- **Integration of Nous-Hermes Vision Models**: `@vic49.` expressed a desire to include some nous-hermes vision models in the plugin, but was unable to find any online sample codes for the same.
- **Contributions to the Project**: `@rishiiyer` offered to work on unfinished parts of the project, and the suggestion was received positively by `@teknium`.
- **Project Update**: `@vic49.` shared an update with the [link to the V3.0.2 release](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2) and mentioned some issues faced.
- **Inference Code for Obsidian**: `@qnguyen3` informed that the inference code for Obsidian would be similar to bakllava, with the exception of changing the EOS token to '###'. They expressed the intent to deliver an update as soon as possible.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral AI Discussion**: Various aspects of Mistral AI were discussed broadly, such as limitations in verbosity and tokenizer differences, as noted by `@.skyair`, `@flopsy1`, and `@meyelo`. Suggestions for efficient system prompts and integrating certain technologies with Mistral 7B were also shared by `@The Ledger Luminary` and `@.tanuj.`. [`GitHub link to tokenizer discussion`](https://github.com/imoneoi/mistral-tokenizer).
- **Deployment and Performance Insights**: Observations on running Mistral AI on certain hardware like the **A100 GPU** and a **Dell Precision 5500** laptop were mentioned by `@builderx` and `@rorolembrouille` respectively. Additionally, `@rorolembrouille` queried about AWS deployment costs and inference times, prompting a response of around **1 token/s** from `@frosty04212`.
- **Fine-Tuning and Model Use**: Community interest in finetuning the base model for specific applications was highlighted by `@lerela casper_ai`. `@pradeep1148` further shared a [YouTube video](https://www.youtube.com/watch?v=A_RDhk9uEy4) demonstrating the **Mixture of Vectors (MoV) approach** for efficient finetuning.
- **Model Comparison and Performance**: Users, particularly `@.gue22`, compared Mistral with GPT-4, expressing dissatisfaction with specific Mistral responses. In contrast, `@bam4d` clarified suitable model comparisons, suggesting `Mixtral` or `mistral-medium` as higher-performing options.
- **Progress towards AGI and Collaboration**: User `@poltronsuperstar` shared progress on using LLMs efficiently, planning a shift from GPT-4 to mistral-medium, and starting a project from scratch. The objective was to facilitate collaborative attempts towards AGI, inviting community participation.
- **Community Contributions and Requests**: In addition to PRs aimed at persisting stats and enhancing docstrings/types, issues with markdown document generation and formatting problems with Mistral models were raised in a conversation involving `@thesealman`, `@bam4d`, and `@poltronsuperstar`. A community logo initiative was proposed by `@m1337d` in the `#random` channel.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (20 messages🔥): 
        
- **Mistral's Verbosity**: A discussion was initiated by `@.skyair` and `@flopsy1` on the verbosity of Mistral's responses, even to yes/no questions. `@lee0099` suggested limiting tokens and using a stricter system prompt to mitigate this.
- **Tokenizer Differences**: `@meyelo` and `@sublimatorniq` talked about the use of a different tokenizer in Mistral, with a link to its GitHub repository found [here](https://github.com/imoneoi/mistral-tokenizer).
- **Prompt Instructions**: `@The Ledger Luminary` provided advice on preparing a more effective system prompt for Mistral. They suggested changing a paragraph of instructions into an itemized list of constraints.
- **Combining Technologies**: `@ton1785` asked for insights and experiences on integrating [GPT pilot](https://github.com/Pythagora-io/gpt-pilot) or [GPT Engineer](https://github.com/gpt-engineer-org/gpt-engineer) with Mistral 7B. `@.tanuj.` offered their observations, mentioning that their work was closer to a "prompt -> functioning project" pipeline.
- **Issues with Flagged Content**: `@mka79` raised an issue about their therapy session content being flagged by OpenAI's GPT-4 API. While the issue wasn't resolved, the user expressed interest in using Mistral Medium for generating longer outputs.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (9 messages🔥): 
        
- **Using GPU for Model**: `@serhii_kondratiuk` was trying to understand why his GPU wasn't being used during computation. `@iukea` suggested checking the model's load time and ensuring that the GPU was actually in use. `@serhii_kondratiuk` later found a solution by selecting GPU Offload and setting some layers for GPU utilization.
- **Clarification on RAM and VRAM**: `@iukea` asked `@serhii_kondratiuk` to verify if he meant RAM or VRAM when discussing memory load. `@serhii_kondratiuk` did not answer the question directly but mentioned a solution that involved adjusting the GPU settings. 
- **False Flags in Gemini**: `@mka79` raised an issue where Gemini was "consistently" false flagging. The error would appear when the response was 99% complete, which was found to be both amusing and annoying. The user did not provide a further context about the issue or a potential solution.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (6 messages): 
        
- **Mistral AI Performance on Different Systems**: User `@builderx` indicated that the performance of the **A100 GPU** was disappointingly slow.
- **Query about Running Mistral AI on a Specific Laptop**: `@rorolembrouille` asked whether they could run a Mistral AI open source LLM on their **Dell Precision 5500** laptop, which has a decent GPU. `@frosty04212` suggested they give it a try, noting that the optimal workflow would depend on their specific use case.
- **Inference Time for Mistral AI**: `@rorolembrouille` queried if the computation could take hours in the worst case, to which `@frosty04212` responded that unless using an inefficient system, the inference time is more likely to be around **1 token/s**.
- **Cost Analysis for Running Mistral AI on AWS**: `@rorolembrouille` inquired if anyone had calculated the cost per token for running the Mistral AI on AWS or a similar cloud platform. No specific answer was provided in the messages.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **Finetuning the Base Model**: `@lerela casper_ai` pointed out the **interest of the community in finetuning the base model**, especially when it comes to specific applications.
- **Efficient Fine-Tuning**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=A_RDhk9uEy4) providing an overview of the **Mixture of Vectors (MoV) approach** and the implementation of one of the proposed methods titled 'Efficient Fine-Tuning Using Mixture Of Vectors Implementation'.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (8 messages🔥): 
        
- **Mistral 7B and GPT-4 Comparison**: User `@.gue22` used the non-tuned Mistral base model to ask numerous tech-related questions. The user was dissatisfied with the answers, comparing the performance unfavorably to GPT-4 and stating "*OMG*". 
- **Model Selection for Asking Questions**: `@bam4d` suggested using the "instruct" model instead of the non-tuned base model when asking questions. They noted that the base model may not be as proficient in answering questions.
- **Concerns About Mistral Responses**: Using the [official LLM examples](https://github.com/ml-explore/mlx-examples) from Apple, `@.gue22` continued experimenting and found the responses from `mistral-7b-instruct-v0.1` to be unsatisfactory compared to GPT-4. The user expressed frustration over the hype surrounding novel models like Mistral, given what they perceive as a significant performance gap.
- **Comparison of Mistral with GPT-4**: `@bam4d` clarified that `Mistral-7B-Instruct-v0.1` is a much smaller model than GPT-4 and should be compared with 13B models instead. They recommended `Mixtral` or `mistral-medium` as potentially higher-performing models. 
- **Hardware Discussion and Future Expectations**: `@.gue22` mentioned their hardware setup, with a focus on increasing capacity to run more significant models. The user expressed hope for upcoming improvements in AI models and frustration with current progress in the field.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 messages): 
        
- **Using LLMs to Utilize Tools**: User `@poltronsuperstar` conducted experiments on Large Language Models (LLMs) and found that giving a chat-style command is the most reliable method to make LLMs use tools as shown in the provided example. Notably, `!i <agentName>` was mentioned as a method to open a conversation with another conversation, allowing for cognitive tree building.

- **Agent Efficiency**: `@poltronsuperstar` pointed out that despite reliability issues, if an agent's role is to solve unit tests, it will keep trying until successful. Though a human brain is still necessary to validate unit tests, there's a notable time gain--with the new process making `@poltronsuperstar` as productive as 15 past selves.

- **Shift in Tech Stack**: `@poltronsuperstar` shared an intention to shift from GPT-4 to mistral medium, and plans to start everything from scratch. An open-source project encompassing these attempts towards AGI (Artificial General Intelligence) is planned for next month. Offers for collaboration before the release were openly solicited. 

- **Codebase Concerns**: `@poltronsuperstar` mentioned that the decision to start from scratch instead of open sourcing immediately is due to a bloated codebase issue which can happen when a codebase builds itself until it becomes unwieldy.

- **Community Logo Initiative**: User `@m1337d` expressed an interest in starting work on ideas for a community logo.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (20 messages🔥): 
        
- **Making Pull Requests on Python Client's Repo & Test Refactoring**: `@poltronsuperstar` expressed a desire to make PRs on the python client's repo with the intent to persist stats and enhance docstrings/types. They also queried the potential rudeness of refactoring tests to which `@bam4d` responded affirmatively given the cover isn't broken and the changes are easily digestible. 
- **Generating Markdown Document & Formatting Problems with Mistral Models**: `@thesealman` reported issues with the generation of markdown documents when using streaming outputs on mistral-medium model. Some lines do not adhere to markdown formatting leading to lines being rendered on one line. A similar issue was observed with mistral-small ignoring markdown formatting.
- **Use of Mistral.ai for Code Debugging**: `@ved_ikke` sought advice on the best mistral.ai model suited for Python code debugging. `@The Ledger Luminary` advised testing different endpoints and assessing results in order to understand capabilities within contextual complexities, adding the potential use of huggingface for fine-tuned models.
- **Request for API Invitation & Clarification of Process**: `@rorolembrouille` requested an API invitation and was informed by `@lidonius` that such invitations are provided by mistralai, not other users/developers.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Topics around **Adding Tokens and Initial Loss**, touched upon by `@noobmaster29` and `@nanobitz`, sparking discussion regarding training models and handling high initial loss. 
- Addressing a **Dataset Cache Issue**, users `@mihai4256`, `@noobmaster29`, and `@nanobitz` attempted to troubleshoot dataset reloading and cleaning up cache, without resolution.
- Discussion into **Transformer Expert's TopK Parameter** led by `@nafnlaus00` and `@nanobitz`, offering insights into Torch's topk functionality and potential alterations.
- Several **Development Issues and Solutions** such as pre-commit action problems and Alternatives namely, the usage of [Pre-Commit.CI](https://pre-commit.ci), were delivered by `@tmm1` and `@le_mess`.
- Query regarding **DPO Readiness** for testing modals broadcasted by `@faldore`.
- Exploring **EOS Token Replacement** in scripts, a concern flagged by `@faldore`.
- Challenges encountered while **Generating Data with Local Model** shared by `@fred.bliss`, seeking advice on creating similar pairings to original material.
- `@natefyi_30842` initiated an inquiry about the comparison between **Internet and Books as Training Data Sources** for Language Learning Models (LLMs). 
- A helpful **Installation Command** for libopenmpi-dev and mpi4py offered by `@mr_morning`, which proved useful to `@q0d.`.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (24 messages🔥): 
        
- **Adding Tokens and Initial Loss**: `@noobmaster29` started training a model with an extra 15k tokens and noticed a high initial loss. `@nanobitz` suggested that this is normal, especially for completion datasets, and advised to let the training run for a bit to see if the loss decreases.
- **Dataset Cache Issue**: `@mihai4256` encountered a caching issue when trying to reload a dataset for the purpose of debugging. `@noobmaster29` suggested deleting the HuggingFace dataset folder to clear the cache, while `@nanobitz` pointed out that the latest version should not load the tokenized dataset from cache if the relevant key is set to `None`. The issue remained unresolved.
- **TopK Parameter of Transformer Experts**: `@nafnlaus00` provided clarification on the `torch.topk` function used in the Transformer Experts, explaining its indifference to the number of experts and the number of experts used at once. This led to a discussion with `@nanobitz` about the possibility and implications of increasing the `top_k` parameter.
- **Resources Shared**: `@noobmaster29` shared a link to a paper on [ArXiv](https://arxiv.org/pdf/2309.09530v1.pdf), though the context or relevance of the paper was not mentioned in the messages.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (6 messages): 
        
- **Issue with Pre-Commit Action**: `@tmm1` shared a link to a feature request on the GitHub repository for Pre-Commit Action, focusing on the issue of adding annotations from pre-commit errors to the specific lines in the actual PR diff ([GitHub issue #70](https://github.com/pre-commit/action/issues/70)).

- **Pre-Commit Linter Errors**: `@tmm1` suggested to surface the "dumb lint errors" from the logs for better accessibility and debugging.

- **Introduction of Pre-Commit.CI**: `@tmm1` introduced [Pre-Commit.CI](https://pre-commit.ci), which could be a potential way to enforce the discovery and automatic fixing of linter errors.

- **Pre-Commit.CI Approval**: `@le_mess` agreed with `@tmm1`'s suggestion, affirming that Pre-Commit.CI is an efficient tool that executes the linters.

- **DPO Readiness Query**: `@faldore` inquired about the readiness of DPO (Differential Privacy Optimizer) for testing, claiming that they have a model and dataset prepared for the purpose.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (26 messages🔥): 
        
- **Adding New Token as End of Sentence(EOS) Token**: User `@faldore` shared a script they used to modify the EOS token in a tokenizer. They replaced the `</s>` EOS token to a new token `


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **Generating Data with Local Model**: `@fred.bliss` experimented with generating decent-quality data using a local model and RAG methods. The effort did not succeed, and the user sought advice on creating at least *50 example rows of data* for each Q&A pair that are close to the original material.
- **Study on Base LLMs Training Using Internet vs. Books**: `@natefyi_30842` inquired if anyone has studied the difference between utilizing internet datasets vs. books to train base Language Learning Models (LLMs). They posited that although books might provide *higher quality yet less diverse data*, that might not necessarily be disadvantageous.


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (2 messages): 
        
- **Installation of libopenmpi-dev and mpi4py**: User `@mr_morning` suggested the following command for installation: `apt-get install libopenmpi-dev -y && pip install mpi4py`, which worked for him. User `@q0d.` found this interesting and appreciated `@mr_morning`'s help.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- A primary topic of interest was **AI model selection and development**:
    - `@woodenrobot` is grappling with the issue of choosing an appropriate AI model for creating "intelligent" NPCs for a MUD. Citing issues with Google Gemini and GPT4, the discussion turned towards the prospect of utilizing a local model (*general* channel).
    - With respect to Bert's forward pass performance on Candle and Pytorch, `@pigeonsai` asked whether slowness is expected on Candle compared to Pytorch (*general* channel).
- **Hardware and software inquiries** were prominent:
    - `@zorian_93363` asked if any CPUs have cores specifically optimized for essential AI mathematical operations, similar to CPUs with embedded graphics (*general* channel).
    - `@mr.aflaton` sought suggestions for creating a multilingual dialogue and voiceover system, even sharing some experience with Google’s translate API and querying about high-quality text-to-speech models (*general* channel).
- Contributors are pushing the limits with their **AI project explorations**:
    - Users `@neuralink` and `@merve3234` discussed an impressive implementation of end-to-end FP8 training in 3D parallelism, with hopes for an open-sourced repository soon (*today-im-learning* channel).
    - One of the community members, `@maybeispedro`, used pseudo labels to improve classification tasks in the domain of astronomy (*i-made-this* channel).
    - A key announcement was made by `@sayakpaul` regarding the release of **Diffusers 0.25.0**, which offers major updates such as a new model named aMUSEd, speed enhancements, and transition to PEFT for LoRA training (*core-announcements* channel).
- The guild was also a hub for **expert advice and resource sharing**:
    - `@asterix3651` requested for clarity on the restrictions around using Gemini for creating datasets and training models with them while referencing Google AI terms and conditions (*general* channel).
    - `@maybiespedro` shared a link to a paper that covers innovative work by researchers who are applying AI to astronomy, appreciated by `@osanseviero` (*cool-finds* channel).
    - `@hafiz031` sought advice on the most suited embedding model for business-related data and `@blakeskoepka` queried for the best platform to access AI research papers (*NLP* channel).
- **Technical challenges and solutions for the AI model usage** came to the fore:
    - `@el_musso` expressed intent to run the model at home on a local PC for use in parallel by his family and received a suggestion from `@merve3234` around local serving and ngrok tunnel (*diffusion-discussions* channel).
    - `@torqx` asked about Segmind Vega models support in a1111 as they experienced difficulties with loading a checkpoint from HuggingFace [Segmind-Vega repository](https://huggingface.co/segmind/Segmind-Vega/tree/main) (*diffusion-discussions* channel).
    - User `@shinomori_7` wanted to receive recommendations for a library that enables keyboard inputs for a game to be taken through hand gestures, but there was a lack of replies (*computer-vision* channel).

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (10 messages🔥): 
        
- **Finding a Chatbot Model for a MUD**: `@woodenrobot` asked for help in selecting a model for creating "intelligent" NPCs for a MUD. The user had tried Google Gemini and GPT4 but found issues with both and was now considering a local model for more control and cost savings.
- **Bert's Forward Pass Performance on Different Libraries**: `@pigeonsai` queried if it's anticipated for Bert's forward pass to be slower on Candle compared to Pytorch.
- **Hardware Capabilities for AI Operations**: `@zorian_93363` posed a question on whether any CPU manufacturer offers a CPU having cores devoted or optimized for specific mathematical operations essential for AI similar to CPUs with embedded graphics.
- **Software for Multilingual Dialogue and Voiceover**: `@mr.aflaton` asked for suggestions on how to create a system that can translate dialogues into multiple languages and convert these translations into voice using three distinctive voice types (man, woman, child). The user already had experience using Google's translate API for the translation part, but still required high-quality text-to-speech models compatible with Python. `@mr.aflaton` had tried pyttsx3 and GTTS libraries but found the first to produce low quality voiceovers and the latter to support only one voice type.
- **Usage Restrictions for Gemini in Model Training**: `@asterix3651` sought clarity on the restrictions around using Gemini for creating datasets and training models with them. The user referred to the Google AI terms and conditions stating that one can't use their services to develop models that compete with Gemini API or Google AI Studio, and asked about the most open source license that can be applied to datasets created via OpenAI or Gemini.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **3D Parallelism FP8 Training**: User `@neuralink` shared they have implemented 13% of end-to-end FP8 training in 3D parallelism from scratch, excluding the FP8 kernels. 
- **Anticipation for Open-Sourced Repo**: `@merve3234` voiced their excitement to see `@neuralink`'s open-sourced repository for the aforementioned project.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **Interesting Research in AI Applied to Astronomy**: `@maybeispedro` shared a link detailing the work of researchers applying AI to astronomy, including [Tuan Dung Nguyen](https://arxiv.org/search/astro-ph?searchtype=author&query=Nguyen,+T+D), [Yuan-Sen Ting](https://arxiv.org/search/astro-ph?searchtype=author&query=Ting,+Y), [Ioana Ciucă](https://arxiv.org/search/astro-ph?searchtype=author&query=Ciuc%C4%83,+I), [Charlie O'Neill](https://arxiv.org/search/astro-ph?searchtype=author&query=O'Neill,+C), [Ze-Chang Sun](https://arxiv.org/search/astro-ph?searchtype=author&query=Sun,+Z), and [Maja Jabłońska](https://arxiv.org/search/astro-ph?searchtype=author&query=Jab%C5%82o%C5%84ska,+M). Their work can be found in this [arXiv paper](https://arxiv.org/abs/2309.06126). 
- **Recognition of the Team's Efforts**: `@osanseviero` acknowledged the team's fascinating work and mentioned having met them in person with some very interesting plans ahead.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (1 messages): 
        
- **Use of Pseudo Labels in Astronomy**: User `@maybeispedro` shared their project which uses pseudo labels to improve classification tasks with tabular data. This method was specifically applied to Astronomy data. The project can be found on [Github](https://github.com/humphrey-and-the-machine/pseudo-labelling).


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **Diffusers 0.25.0 Announced**: User `@sayakpaul` announced the release of **Diffusers 0.25.0** with major updates including:
    - The introduction of a new model named aMUSEd, which interestingly is not based on diffusion.
    - Speed enhancements making SDXL (extends to other pipelines) **3x faster**.
    - Transitioning to PEFT for LoRA training.
- More information about the release can be found on [GitHub](https://github.com/huggingface/diffusers/releases/tag/v0.25.0).


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **Running the model at home on a local PC**:`@el_musso` has been considering running the model at home for use by his family at the same time. In response, `@merve3234` suggested **serving locally and using a ngrok tunnel** for the parallel use. `@el_musso` acknowledged the advice and committed to researching this topic.
- **Segmind Vega models support in a1111**: `@torqx` asked if **Segmind Vega models** are supported in a1111. They reported difficulty in loading a checkpoint obtained from the [`Segmind Vega page on HuggingFace`](https://huggingface.co/segmind/Segmind-Vega/tree/main). They did not receive any response or solution in this message thread.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (2 messages): 
        
- **Hand Gestures as Keyboard Inputs for Games**: User `@shinomori_7` asked for recommendations on a library that would help them take keyboard inputs for a game using hand gestures. They asked for help and are patiently waiting for replies.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (6 messages): 
        
- **Choosing an Embedding Model for Business Data**: `@hafiz031` requested advice on the most suitable embedding model for business and finance data.
- **Sources for AI Research Papers**: A question was raised by `@blakeskoepka` about the best platform to read research papers on AI. `@osanseviero` suggested the [Hugging Face papers page](https://hf.co/papers) which features manually curated papers on a daily basis. Further, VIPitis recommended using **ArXiv** with specific Keywords for more preprint publications.
- **Comparison of Encoder Models**: `@opencuiguy` responded to a comment from `@merve3234` stating a preference for encoder-only models for individual discrimination tasks, despite noting that they can be rigid.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **Running model on local PC for multiple users**: User `@el_musso` enquired about the possibility of running the model at home on a local PC to be used in parallel by multiple family members. `@merve3234` suggested serving locally and using an ngrok tunnel as a potential solution.
- **Segmind Vega models support in a1111**: `@torqx.` asked if segmind vega models are supported in a1111 as they have encountered an issue with the checkpoint they got from the HuggingFace [Segmind-Vega repository](https://huggingface.co/segmind/Segmind-Vega/tree/main).


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Request for AI/ML specialist in developing **Chat GPT Plugin** with backend already established made by `@heman35` in both [looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/) and [looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/) channels.
- `@venadore` experimented with and discussed **bias calls** in the general chat without providing further context or details.
- Discussion on the process of **fine-tuning MoE models** and determining the appropriate hyperparameters from user `@imonenext`. 
- An intense discourse related to **AI and Ethics**, criticizing a specific controversial AI research group for ethical shortcomings in their claims of novelty, regarded as being built on open-sourced tools as found on [twitter](https://fxtwitter.com/winglian/status/1740081525826167060). User reactions from `@caseus_`, `@undi` and `@giftedgummybee` were specifically highlighted.
- Transition to a lighter conversation initiated by `@gabriel_syme` joking about having a "free paper."

**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (1 messages): 
        
- **Building A Chat GPT Plugin**: User `@heman35` requested assistance with building a chat GPT plugin. They mentioned **having the backend ready** and specifically requested the help of an expert in **AI, ML, and NLP**. They encouraged interested individuals to Direct Message them.


### ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/) (1 messages): 
        
heman35: 👋 Hey , can you help me with Building A Chat GPT Plugin? We Have the Backend Ready.


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (2 messages): 
        
- **Discussion on bias calls**: `@venadore` mentioned that they were **experimenting with bias calls** and found a certain situation amusing, though they did not provide specific details.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (19 messages🔥): 
        
- **Fine-Tuning MoE Models**: User `@imonenext` asked the community about the process of fine-tuning **MoE models**, particularly how to choose the appropriate hyperparameters and whether the learning rate should be higher or lower as compared to dense models.
- **AI and Ethics Discussions**: An argument erupted regarding another AI research group's claim of novelty for a discovery that `@caseus_` and others believe was identified through open source tools. This led to a discussion of ethics and giving credit where it is due. Findings on twitter were shared [on this link](https://fxtwitter.com/winglian/status/1740081525826167060).
- **Recognition of Contributions**: Both `@undi` and `@giftedgummybee` criticized the controversial research group for allegedly not putting in effort on their own and not recognizing the contributions of those whose work they used, including open sourced discoveries and layer configurations from other developers. `@undi` made it clear that the technique that the research group claimed as their own was actually based on Charles' tools.
- **GPT-Agent Training**: The discussion further unraveled with `@giftedgummybee` suggesting that the controversial group was seeking publicity without contributing anything new to the community.
- **Free AI Research Paper**: In a lighter tone, `@gabriel_syme` humorously chimed in the conversation about getting a "free paper".


### ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/) (1 messages): 
        
heman35: 👋 Hey , can you help me with Building A Chat GPT Plugin? We Have the Backend Ready.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain**'s documentation update announced with an invitation for user feedback on the draft version. The update mainly pertains to Python and JavaScript libraries. The draft version link is provided [here](https://langchain-5o76madna-langchain.vercel.app/docs/get_started/introduction).
- Conversation around development and application of multipurpose chatbot, passing chat history using OllamaFunctions, generating FAQs from blog content, and employing Redis for chat storage in chatbot app design.
- Inquiry about the efficiency and strategic effectiveness of using single or multiple fine-tuned LLMs for complex tasks, with the GitHub Copilot task serving as a specific case study.
- Release and review of an open-source project, KwaiAgents, which claims to perform better than GPT-3.5 in specific AI agent tasks. The [Medium blog post](https://medium.com/@myscarletpan/can-7b-models-now-master-ai-agents-a-look-at-kwais-recent-llm-open-source-release-8b9e84647412) and [GitHub repository](https://github.com/KwaiKEG/KwaiAgents) were shared.
- Launch of an online LLM course aimed at beginners, featuring training on programming LLMs using Python and LangChain. The course link was provided [here](https://lnkd.in/geN5yrkk).
- Technical queries and discussions surrounding the application and functionality of langserve pertaining to input variable queries, issues with 'OpenAIAssistantFinish' Object, implementations of AgentExecutors, and potential problems with invokes in routes.
- Announcement of **ChromaDB Plugin** v3.0.1 release for LM Studio which contains revised scripts for a Pdf loader. The Github release link shared [here](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.1).
- Introduction of a GitHub project entailing a Menhir parser for Llama CPP developed in OCaml, with an open invitation for contributions to improve the grammars [link](https://github.com/meta-introspector/gbnf-nice-parser).
- An alert conducted by `@pradeepvj97` to `@sacha7031`  advising the latter to reset their API for security reasons. The specific context remains unclear.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **LangChain Documentation Update**: `@hwchase17` announced that LangChain's team is currently working on a documentation update, accepting feedback on the draft version shared [here](https://langchain-5o76madna-langchain.vercel.app/docs/get_started/introduction). The **LangChain** is a framework for developing applications powered by language models with context-aware reasoning. It comprises of several parts, including **LangChain Libraries** mainly in Python and JavaScript. However, some new functions used may not be merged in yet.
    - They expressed interest in hearing feedback concerning potential issues users might encounter in parts of the documentation other than the quick start and Modules.


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (11 messages🔥): 
        
- **Creating Multipurpose Chatbot with SQLDatabase Chain**: `@annamalai8892` asked for assistance on how to use SQLDatabase chain with other LLMChains in router chain for creating a multipurpose chatbot.
- **Passing Chat History in OllamaFunctions**: `@chronos.vitaqua` inquired about how to pass chat history to a chat bot using OllamaFunctions in a meaningful and efficient manner.
- **App for Generating FAQs from Blogs**: `@ashwinm` sought advice on building an app that generates FAQs from the content of a blog. In response, `@ashwinm evolutionstepper` suggested looking into the [KB_builder project](https://github.com/offskiies/KB_builder) on GitHub for guidance.
- **Fine-Tuning LLMs for a Compositional Task**: `@ruili09` sought opinions on whether to use a single extensively fine-tuned LLM for complex tasks or chain multiple fine-tuned LLMs for different steps of the task. They used the GitHub Copilot task as an example to illustrate their point.
- **KwaiAgents Open-Sourced Project**: `@myscarlet` shared a link to a [Medium blog post](https://medium.com/@myscarletpan/can-7b-models-now-master-ai-agents-a-look-at-kwais-recent-llm-open-source-release-8b9e84647412) and a [GitHub repository](https://github.com/KwaiKEG/KwaiAgents) for KwaiAgents, an open-source project by Kwai that claims to outperform GPT-3.5 in certain AI agent tasks.
- **Storing Chat Memory in Redis**: `@sampson7786` asked for advice on using Redis to store chat memory for distinct users while developing a chatbot app with LangChain.
- **Online LLM Course for Beginners**: `@altafr` announced the second batch of their course providing training in LLMs for beginners, including how to use Python and Langchain to program LLMs. The [course link](https://lnkd.in/geN5yrkk) was provided.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (6 messages): 
        
- **Querying Input Variables in Langserve**: `@cryptossssun` brought up a question about how to query the two input variables in the langserve service.
- **Issues with 'OpenAIAssistantFinish' Object**: `@stu.mach` shared an issue they encountered when trying to use an OpenAIAssistant with a custom executor with langserve, which led to an AttributeError related to 'get_input_schema'.  
- **AgentExecutors and Runnable**: `@a404.eth` suggested that AgentExecutors may not implement runnable, which could be necessary for the 'add_routes' function that `@stu.mach` was trying to use. 
- **Problems with Invokes in Routes**: `@a404.eth` noticed that `@stu.mach` was placing the response to an invoke in the routes, not a chain, and warned that this could lead to problems.


### ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 
        
@pradeepvj97 sacha7031: You should reset your API after sending this message


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (3 messages): 
        
- **Release of ChromaDB Plugin v3.0.1 for LM Studio**: `@vic49` announced the latest release of **ChromaDB Plugin** for LM Studio [V3.0.1 - SHOWTIME!](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.1), which includes revised scripts for a custom PDF loader named `pdf.py`.
- **GBNF Nice Parser for Llama CPP using Menhir**: `@m1337d` posted about his [GitHub project](https://github.com/meta-introspector/gbnf-nice-parser), a working version of a gbml menhir parser in OCaml for Llama CPP. He invited contributions for improving the grammars which can be used to constrain and customize the output from the Llama CPP.
- **KwaiAgents Release**: `@myscarlet` introduced the release of KwaiAgents, an auto-agent system with LLMs possessing agent capabilities. They also shared the corresponding links to the [GitHub repository](https://github.com/KwaiKEG/KwaiAgents) and a Medium post titled ["Can 7B Models Now Master AI Agents?"](https://link.medium.com/MBUxuLhZSFb). The system includes training data and benchmarks.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Announcement of new fine-tuned model, **CodeNinja**, from OpenChat by `@beowulfbr`. CodeNinja specializes in code assistance and is available on [Hugging Face](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B). Encouragement to test the model with the GGUF format model files available [here](https://huggingface.co/TheBloke/CodeNinja-1.0-OpenChat-7B-GGUF).
- Discussion on the lawsuit between New York Times and OpenAI, referring to a [Twitter thread](https://fxtwitter.com/ceciliazin/status/1740109462319644905?s=46&t=90xQ8sGy63D2OtiaoGJuww) shared by `@swyxio`.
- `@swyxio` notified about a forthcoming discussion on the **Beyond Human Data** paper conducted by `<@451508585147400209>`. The announcement included a link to join [the discussion](https://lu.ma/llm-paper-club) and mentioned the weekly frequency of these paper sessions.
- A [paper discussion](https://arxiv.org/abs/2312.06585) on scaling beyond human data coordinated by `@eugeneyan`. Other notable discussions included a problem reported by `@swyxio` concerning Discord's "RTC Disconnected" error and an upcoming discussion focus on [InsightPilot](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/) by `@187636841988620288`.
- User `@cakecrusher` raised a topic looking into when to fine-tune an embedding model versus a Retriever-Augmented Generation (RAG) model, highlighting potential differences in the requirements for each.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (7 messages): 
        
- **CodeNinja Release**: `@beowulfbr` announced the release of a new fine-tuned model from OpenChat, called **CodeNinja**, specialized in code assistance. The model, an enhanced version of openchat/openchat-3.5-1210, has been fine-tuned on over 400,000 coding instructions. Available on [Hugging Face](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B).
- **CodeNinja Feedback and Testing**: `@beowulfbr` mentioned receiving good feedback on Reddit and encouraged others to try out the **CodeNinja** model. The GGUF format model files are available [here](https://huggingface.co/TheBloke/CodeNinja-1.0-OpenChat-7B-GGUF).
- **OpenAI NYT Lawsuit Discussion**: `@swyxio` shared a [Twitter thread](https://fxtwitter.com/ceciliazin/status/1740109462319644905?s=46&t=90xQ8sGy63D2OtiaoGJuww) discussing the lawsuit between New York Times and OpenAI.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **Beyond Human Data Paper Discussion**: User `@swyxio` announced that `<@451508585147400209>` will conduct a discussion about the **Beyond Human Data** paper in 15 minutes. The discussion can be joined at [this link](https://lu.ma/llm-paper-club). This is a part of a weekly series where LLM papers are reviewed and discussed. Users interested can asked to be tagged in to `<@&1107197669547442196>` to get discord notifications.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (5 messages): 
        
- **Scaling Beyond Human Data Discussion**: `@eugeneyan` announced a [paper discussion](https://arxiv.org/abs/2312.06585) on scaling beyond human data, urging certain members to participate.
- **Discord Error Reporting**: `@swyxio` reported having an "RTC Disconnected" error on Discord, causing them to not be able to hear or speak.
- **InsightPilot Paper Discussion**: In the following week, `@eugeneyan` informed that `@187636841988620288` would lead the discussion on [InsightPilot](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/), an LLM-based automated data exploration system.
- **Anticipation for Future Discussions**: `@ivanleomk` expressed excitement for future paper discussion sessions in the upcoming year.
- **Query on Fine-tuning Models**: `@cakecrusher` initiated a discussion on when to fine-tune an embedding model versus a model for Retriever-Augmented Generation (RAG), noting the potential differences in requirements for each.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- User `@baptistelqt` made an inquiry about any existing workflow for **reading/reviewing academic papers** using **ChatGPT** in the #papers channel, though no responses or follow-up discussion ensued.
- In the #off-topic channel, user `@shreepandey` speculated on the potentials of Language Learning Models (LLMs). A hypothetical scenario was presented contemplating whether, given adequate conversational data from an alien species or animal sounds, LLMs could decipher some part of their language. Additionally, user `@pradeep1148` shared a [YouTube video link](https://www.youtube.com/watch?v=A_RDhk9uEy4) without providing context about the content of the video.
- In the #bakklava-1 channel, `@onuralp.spriobel` advised `@spirobel onuralp.` to investigate **cogVLM**, referencing discussions from the **nous research discord**.

**Skunkworks AI Channel Summaries**

### ▷ #[papers](https://discord.com/channels/1131084849432768614/1131305311714672710/) (1 messages): 
        
- **Paper Reading/Review with ChatGPT**: User `@baptistelqt` enquired if anyone had a workflow for **reading/reviewing academic papers** using **ChatGPT**. No follow-up discussions or responses were provided.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (2 messages): 
        
- **Speculating on Alien Languages and Animal Sounds**: User `@shreepandey` posed an intriguing question about whether, given a complete dataset of conversations from an alien species, we could discern some part of their language with language learning models (LLMs). This idea was extrapolated to the conversion of animal sounds into a human-understandable language. 
- **Video Link Shared**: User `@pradeep1148` shared a [YouTube video link](https://www.youtube.com/watch?v=A_RDhk9uEy4), context or content of the video wasn't described in the messages.


### ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (2 messages): 
        
- **Discussion on cogVLM**: `@onuralp.spriobel` recommended `@spirobel onuralp.` to look into **cogVLM** based on results discussed in the **nous research discord**.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Engagement around the **Mixtral Base Model Score** with user `@gheshue` indicating that .calytrix achieved a score of **50.27**, approximating the base model outcome.
- Discussion and inquiries surrounding the concept of training router layers during Mixtral fine-tuning, with user `@bjoernp` questioning the consensus. User `@sebastian.bodza` provided a [Twitter post](https://twitter.com/erhartford/status/1737350578135834812) by **Eric Hartford** as a reference, explaining the benefit of freezing the router layer for improved performance during training.
- Proposition by `@leecig` for the formation of a focus group dedicated to exploring the integration of various AI software technologies such as **MemGPT, AutoGen**, served via **Ollama** and potential interest in **GPTPilot**. User `@ismaelfaro` showed interest by responding with a thumbs-up emoji.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (3 messages): 
        
- **Mixtral Base Model Score**: `@gheshue` stated that .calytrix obtained a score of **50.27**, which is about the same as the base model.
- **Training Router Layers during Mixtral Finetuning**: User `@bjoernp` inquired if a consensus had been reached on whether to train router layers during Mixtral finetuning with Lora. 
- `@sebastian.bodza` responded, referencing a [Twitter post](https://twitter.com/erhartford/status/1737350578135834812) by **Eric Hartford**. In the post, **Eric Hartford** describes that freezing the router layer improves performance during the training process.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (2 messages): 
        
- **Focus Group on AI Software**: User `@leecig` proposed the idea of creating a focus group for combining various AI software technologies, namely **MemGPT, AutoGen**, and serving the models via **Ollama**. They also mentioned interest in **GPTPilot**. They invited users interested in this focus group to ping or DM.
- User `@ismaelfaro` expressed their interest by reacting with a thumbs-up emoji.


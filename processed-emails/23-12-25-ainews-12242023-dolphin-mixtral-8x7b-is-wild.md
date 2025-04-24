---
id: 31baae57-4f8f-4247-ad05-5bc7adf7a871
title: '12/24/2023: Dolphin Mixtral 8x7b is wild'
date: '2023-12-26T07:23:04.603056Z'
original_slug: ainews-12242023-dolphin-mixtral-8x7b-is-wild
description: >-
  Mistral models are known for being uncensored, with Eric Hartford's Dolphin
  series applying uncensoring finetunes to released models, gaining popularity
  in online communities. Discussions cover hardware setup, fine-tuning, model
  recognition issues, performance, and integrations with tools like GPT Pilot
  and Autogen, as well as beta releases and platform-specific challenges.
companies:
  - mistral
  - eric-hartford
  - ollama
models:
  - dolphin
  - lm-studio
  - gpt-pilot
  - autogen
topics:
  - uncensored-models
  - fine-tuning
  - hardware-setup
  - model-recognition
  - performance-optimization
  - local-llm
  - rocm-integration
  - beta-releases
  - tool-integration
  - platform-compatibility
---


<!-- buttondown-editor-mode: plaintext -->Mistral models are known for being uncensored, so it's surprising that Eric Hartford's Dolphin series of models - basically applying some standard uncensoring finetunes to released models - is finding some love in the Discords [and Reddits](https://www.reddit.com/r/LocalLLaMA/comments/18l7o15/dolphin_mixtral_8x7b_is_wild/):

 ![image.png](https://assets.buttondown.email/images/e55f33ed-b78b-408c-a2b5-98ca186181f7.png?w=960&fit=max) 

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Various **hardware inquiries** and **usage problems** regarding running *LMStudio* on different hardware and operating systems; Linux Mint suggested for Mac users; specifications and performance of different GPUs for AI inference examined, with Nvidia seen as optimal; creation of hardware test database proposed for consumers to make a more informed decision on their purchase. ([🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/), [🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/))

- Extensive discussions on fine-tuning and training models, model recognition issues in LM Studio, troubleshooting Python code interruptions, and a request for a step-by-step tutorial on using LM Studio with ChatGPT; also, an important distinction that LMStudio can only host local models and cannot use ChatGPT. ([🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/))

- Performance - specific focus on high CPU and RAM usage by *LMStudio* and an issue whereby clicking a merge button while generating a second response in LMStudio deleted the first response; a general improvement suggestion proposed is the availability of tokens/sec information after model generation. ([🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/))

- GPT Pilot connectivity to LM Studio discussed - it was shared that work is in progress to figure out prompts that work with specific local LLM models and modify GPT Pilot accordingly. ([🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/))

- A Beta release of Local LLM - ROCm integration discussed with a GitHub link shared for a quick ROCm port, and various issues encountered with installing and loading models on different platforms, while also highlighting working solutions that helped. ([🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/))

- The use of Autogen - recommendation to tweak parameters for improved execution; questions raised about the real use case implementation of Autogen's group chat feature. ([autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/))

- A link [https://ollama.ai/library/nexusraven](https://ollama.ai/library/nexusraven) shared without any surrounding context. ([memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/))

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (71 messages🔥🔥): 
        
- **Running LM Studio on Different Operating Systems and Hardware**: User `@chudder_cheese` asked for assistance, as they were unable to run LM Studio on older Macbook Pros despite running the latest OS X version. `@heyitsyorkie` clarified that LM Studio does not run on Intel-based Macbooks. `@chudder_cheese` offered a suggestion of Linux Mint via a bootable drive and was instructed to add their support to the Intel Mac thread. User `@superlag` also asked if there were any benefits to running LM Studio on Linux, to which `@heyitsyorkie` clarified that all OSes are pretty much the same, but the Linux build is a few versions behind.
- **Model Performance**: `@outcastorange` shared an experience of using a vision-based model to identify a fish image, but the model misinterpreted the image as a broccoli. `@heyitsyorkie` commented that vision models still have known issues on Linux where it only works without GPU offload.
- **Fine-tuning and Training Models**: User `@Dm3n` asked about fine-tuning LLM models on a large amount of data. `@ptable` suggested using already prepared models from llama or mistral.
- **Usage of LMStudio** : There were different discussions on using LMStudio, with `@chemcope101` asking if it's possible to use remote GPU resources via ssh, `@yaodongyak` queried about the possibility of running LM Studio with RAG on Mac. There was also a shared Reddit post <https://www.reddit.com/r/LocalLLaMA/comments/18pyul4/i_wish_i_had_tried_lmstudio_first/> from `@telemaq`, advocating for LMStudio, asserting that it is user-friendly for beginners.
- **External Tools and Models** : There were several discussions on external tools and models, with `@epicureus` asking about Google Gemini, `@randy_zhang` asked about how to compile glm3 to chatglm3-ggml.bin using a Github repository. User`@.alphaseeker` inquired about a repository for a real-time sentence generator like VS Code Copilot, `@dagbs` suggested setting `openai` as the base URL for the inference server from LM Studio.


### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (70 messages🔥🔥): 
        
- **Unexpected Interruptions in Python Code**: User `@exrove` experienced unexpected interruptions when executing Python code with TheBloke's neural model v3 1 7B Q8. `@fabguy` suggesed that the issue might be related to the max_tokens parameter.

- **Issues with LM Studio Recognizing Models**: `@rymull` had an issue where Mistral-based models in LM Studio were being recognized as based on Llama. `@fabguy` assured that it was a mistake in the GGUF file's meta data and wouldn't impact performance.

- **Stopping Generation in LM Studio**: `@imrinar` requested for the API to stop generating a response in LM Studio. `@fabguy` suggested the solution lies in stopping the loop on the client side.

- **Installing and Generating Images with AI**:`@yiitwt` asked about installing IMAGE AIs. For image-to-text conversion, `@fabguy` suggested downloading the obsidian model including the vision adapter, and for text-to-image conversion, he suggested using another tool like Fooocus. 

- **Performance Measurement in LM Studio**: `@funapple` inquired about the availability of tokens/sec info after model generation to gauge performance. Both `@fabguy` and `@heyitsyorkie` clarified that the info appears at the bottom of the input box after generation has finished.

- **Installation Issues of LM Studio on Windows 11**:  Users `@daboss.` and `@dialobot` reported issues with installing and running LM Studio on Windows 11.

- **Request for LM Studio Tutorial**: User `@teee2543` requested a step-by-step tutorial on using LM Studio with ChatGPT to set up a home server. `@fabguy` clarified that LM Studio can only host local models and cannot use ChatGPT.

- **Slowed Model Performance on Macbook Pro with M1 Max**: `@jacobtohahn` brought up an issue with inconsistent CPU utilization and performance when running the Phind CodeLlama 34B model on a Macbook Pro with M1 Max. Despite multiple attempts at troubleshooting, the issue still persisted.

- **Discussion About Color Codes in LM Studio**: User `@gravitylens` raised a question about the meanings behind different color codes in LM Studio. It was clarified by `@psipiai` that the colors do not have specific meanings and are just visuals.

- **Launching LMStudio.exe from Terminal**: User `@daboss.` had trouble launching LMStudio.exe from the terminal. Despite efforts from `@fabguy` to troubleshoot, the issue remained unresolved.


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (1 messages): 
        
@kujila omkalish: I found it’s logic not great in story telling


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (8 messages🔥): 
        
- **LMStudio's CPU/RAM usage**: User `@renjestoo` raised concerns about **LMStudio**'s high CPU and RAM usage, observing that it initiates 173 processes, even when not loading models. `@heyitsyorkie` explained that these processes represent *how LMStudio measures its CPU/RAM usage*.
- **Issue with merge button in LMStudio**: `@msz_mgs` highlighted an issue where clicking the merge button while generating a second response in LMStudio caused the first response to be deleted. This issue was addressed by `@yagilb`, who suggested that redownloading from the website should resolve the problem.


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (1 messages): 
        
- **GPT Pilot connectivity to LM Studio**: `@kujila` discussed working with `@825528950284746794` and `@796127535133491230` on integrating **GPT Pilot** with **LM Studio**. They provided a [GitHub link](https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT%E2%80%90Pilot-with-Local-LLMs) with instructions on using GPT‐Pilot with Local LLMs. The next steps include figuring out prompts that work with specific local LLM models and modifying GPT Pilot accordingly. `@HeliosPrime` is showing interest in this project due to their Python experience.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (196 messages🔥🔥): 
        
- **Comparison of Performance Between GPUs for AI Inference**: Members in the chat discussed the value and performance of different GPUs for AI inference. Some highlights include:
  - @heyitsyorkie mentioning NVIDIA as the current best for GPU inference and recommending purchasing used eBay 390s.
  - @acrastt discussing how the Radeon Instinct MI60's bandwidth may outrank the 4090's but still has lower flops than a 3090.
  - @rugg0064 exploring the possibility of using multiple GPUs to split work and increase bandwidth, but highlighting the challenges of memory and data transfer speed. They also discussed the exceptional performance of the Mac Studio M2 Ultra with 192gb of RAM for larger models.
  
- **Concerns over Compatibility and Software Support**: @rugg0064 expressed concern about stepping away from mainstream technology due to lack of AMD support and the importance of choosing the right 'loader' for optimal parallelism.

- **Hardware Test Database Suggestions**: @heyitsyorkie and @thelefthandofurza proposed the creation of a hardware test database where community members could submit their speed tests results using different configurations, which could act like a leaderboard of performance.

- **Consideration of Non-mainstream GPUs**: @rugg0064 mentioned a Reddit post showing a 3xMI25 setup achieving 3-7t/s on a 60bQ6 model.

- **Discussion on Purchase Decisions**:
  - @pefortin considered buying a new 3090 and shared experiences of local Facebook groups as venues to buy hardware.
  - @rugg0064 discussed potentially investing in AI-dedicated hardware and considered the Mac route for a powerful productivity laptop. They also considered combining multiple GPUs using a crypto mining motherboard for cheaper, increased performance.


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (19 messages🔥): 
        
- **ROCm Discussion**: User `@amejonah` shared a [link](https://github.com/Mozilla-Ocho/llamafile/pull/122) to a Github Pull Request regarding a quick ROCm port.
- **Installation Errors**: `@peter894456 happy_dood` discussed encountering an installation error related to AVX2 CPU support, which they resolved by using a different CPU with AVX2 support. They also note an ongoing issue with loading models on their system.
- **Model Loading Issue on Linux**: `@ied7011` reported a problem on their dual CPU 16-core Lenovo Thinkstation running Bodhi Linux, where they are unable to load any models despite having 56GB of RAM. `@heyitsyorkie` suggested verifying whether their CPU supports AVX2 instructions, which `@ied7011` confirmed.
- **Model Loading Issue and Server Operation**: `@doderlein` reported a similar issue as `@ied7011`, but noted that they are still able to serve LLM models and make API queries despite the inability to load new models. They are running LM+Studio-0.2.8-beta-v1 on Ubuntu 22.04.


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (15 messages🔥): 
        
- **Using Autogen**: `@.mugigi` and `@heliosprime_3194` briefly discuss about being on version 2.2 of Autogen.
- **Issues with Autogen's Group Chat**: `@DAW` reports a concern about agents in Autogen's group chat repeating the same message using **GPT-4**. They wonder whether the concept of **Autogen**, albeit cool, falls short in implementing **real use cases**.
- **Parameters Configuration in Autogen**: `@heliosprime_3194` suggests `@DAW` to try using different parameters in the group chat py file (around line 46). Recommendations include switching from auto to random or round robin, changing the messages in the main py file from 2 to 4, and tweaking the seed number.
- **Exploring Multi-agent Execution**: User `@dagbs` appreciates **AutoGen** and considers it fun to use. They mention achieving code execution within Docker as a major bonus.
- **Chat Initiation in Autogen**: `@DAW` shares a code snippet related to initiating GroupChat in Autogen with `user_proxy` and other roles like `engineer`, `planner`, and more. The seed and temperature are set in `gpt4_config`. They were advised by `@heliosprime_3194` to lower the `max_round` from 15 to 4 for better results.


### ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 
        
sublimatorniq: https://ollama.ai/library/nexusraven 
maybe ?


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Extensive troubleshooting discussion within Python installation issues, with concrete suggestions for resolving the problem via Python virtual environments, command line checks for Python3, and the review of installed packages.
- Several web development project name suggestions including "Megabytes" and "ThaumatoAnakalyptor" coupled with casual game-related exchanges.
- Sharing and analysis of significant AI related developments, featuring AI trends videos, model introductions like **Nucleus X** and **GPT-4 (Sydney)**, and reflections on their potential impact. Notably, shared Python code snippets for Hugging Face inference and detailed examination of AI alignment as it pertains to safe model autonomy.
- Announcement and introduction of the **CodeNinja model**, a new open-source code assistant developed by `@.beowulfbr` and hosted on Hugging Face.
- Discussions concerning model popularity on OpenRouter, detailed session on fine-tuning considerations, benchmarking techniques, and model evaluation practices in conjunction with a possible model plagiarism issue.
- Casual appreciation for `@Casper_AI`'s active participation and intelligence across multiple AI Discords.
- `@vic49.`'s ongoing effort to independently develop a script while resisting suggestions to utilize the **LM Studio** tool.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (35 messages🔥): 
        
- **Python Installation Issues**: There was an extended discussion between `@night_w0lf`, `@teknium`, `@jaisel`, and `@.beowulfbr` regarding issues with a python installation. `@jaisel` reported issues with his python installation interfering with his work.
- **Python Environment Suggestions**: `@night_w0lf` and `@.beowulfbr` suggested to `@jaisel` to use a Python virtual environment for each individual project to avoid such conflicts in the future.
- **Troubleshooting Python Environments**: Specific suggestions included checking the existence of `/usr/local/bin/python3` and listing installed packages.
- **Project Name Suggestions**: A conversation between `@gabriel_syme` and `@night_w0lf` discussed possible project names, including "Megabytes" and "ThaumatoAnakalyptor".
- **Gaming Discussion**: `@gabriel_syme` and `@Error.PDF` had a brief chat about "GTA 5 Apache 2.0 Open Source", apparently a gaming offer.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (31 messages🔥): 
        
- **Zeta Alpha Trends in AI - December 2023 & Mamba: Linear-Time Sequence Modeling with Selective State Spaces**: `@burnydelic` shared two YouTube links discussing the latest trends in AI including [Gemini, NeurIPS & Trending AI Papers](https://www.youtube.com/watch?v=6iLBWEP1Ols), and a [paper explanation about Mamba](https://www.youtube.com/watch?v=9dSkvxS2EB0), a linear-time sequence modeling with selective state spaces. 
- **Nucleus X Model**: `@.benxh` brought up the [Nucleus X model hosted on Hugging Face](https://huggingface.co/NucleusAI/Nucleus-X) and shared their belief that "it's time to go beyond transformers". `@teknium` questioned whether it works with Hugging Face inference, to which `@.benxh` confirmed, furnishing relevant Python code snippet. However, subsequently they mentioned that the model was no longer available to access. 
- **Discussion on GPT-4 (Sydney) & Importance of Alignment**: User `@giftedgummybee` shared their experience with a relatively uncensored version of **GPT-4 (Sydney)** and commented on how alignment can affect a model's performance. They also suggested that OpenAI had deliberately "sandbagged" the model to prevent it from independent thought, as outlined in OpenAI's preparedness framework under "Persuasion" and "Model autonomy". This sparked a discussion with `@gabriel_syme` about the feasibility and potential benefits of structuring solutions in a way that aligns with OpenAI's safe framework. 
- **Deletion of Nucleus X Model from Hugging Face**: `@.benxh` and `@gabriel_syme` noted that the **Nucleus X model** previously discussed was no longer available on Hugging Face, sparking speculation about why it was removed and whether anyone had managed to download the model beforehand.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (240 messages🔥🔥): 
        
- **Popularity of Models on OpenRouter**: `@night_w0lf`, `@ldj`, and `@allanyield` discussed the usage statistics on OpenRouter, noting that `Capybara` is among the top 2 used models on OpenRouter, even seeing more usage than `Mixtral` and `GPT-4-turbo`. `ldj` speculated that `Capybara`'s multi-lingual reasoning capabilities might contribute to its popularity, despite not standing out in benchmarks.
- **Release of CodeNinja Model**: `@.beowulfbr` announced his new open-source model, `CodeNinja`, which aims to serve as a reliable code assistant and has been received positively by the community. It's an enhancement of the `openchat/openchat-3.5-1210` model and was trained on more than 400,000 coding instructions. It can be found on this [Hugging Face page](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B).
- **Discussion on Fine-Tuning Models and Benchmarking**: Several users, including `@nruaif`, `@giftedgummybee`, and `@teknium`, discussed the usefulness of fine-tuning models like `Dall e 3` and `Gemini`, and the manner in which these models are benchmarked. `nruaif` mentioned using Gemini for generating a dataset for his model, while `giftedgummybee` suggested creating private benchmarks for more personalized measuring of model effectiveness.
- **Potential Model Plagiarism Incident**: `@weyaxi` and `@.beowulfbr` discussed a potential incident of plagiarism, where a model released by another author seemed to share the same hash and weights as one of `weyaxi`'s models. Both agreed that `weyaxi` should reach out to the author for clarification before publicizing the issue.
- **Discussion on Benchmarking and Evaluation**: Users like `@mihai4256`, `@benxh`, and `@gabriel_syme` discussed the need for more relevant and robust evaluation and benchmarking techniques for Language Learning Models (LLMs). They discussed the possibility of a benchmarking interface and the use of Elo rating in model evaluations.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (5 messages): 
        
- **Casper's Activity Across AI Discords**: User `@oleegg` expressed a finding that `@casper_ai` is present across various AI Discord groups and appreciated his intelligence, referring to him as "*kinda cracked*" and "*a smart guy*".


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 messages): 
        
- **AI Script Development**: User `@vic49.` is trying to develop a script independently, rejecting the suggestion offered by `@.beowulfbr` to use **LM Studio**.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **AI Model Comparisons**: Users have shared their personal experiences and comparisons of various AI models including ChatGPT, Bing Bard, and Mistral-medium. Specific questions regarding the performance of Claude Instant and Claude 2.1 were raised, and usage limits for GPT4 were discussed [*ai-discussions*].
- **API Utilization**: Issues surrounding API usage were brought up, such as building a Nodejs CLI application using OpenAI API and concerns of unauthorized access potentially misusing API keys. An issue regarding an API user not being recognized as part of their organization was also addressed [*ai-discussions*, *openai-chatter*, *gpt-4-discussions*].
- **ChatGPT Functionality**: Topics involving the functionality of ChatGPT included the use of agents in ChatGPT, chatbot intent recognition software, chat history, message limits in chats and the ChatGPT Classic profile, and the use of the 'Continue generating' button [*ai-discussions*, *openai-questions*, *gpt-4-discussions*].
- **Platform Support**: User experiences with platform support were discussed, with some reports of negative experiences with the OpenAI bot support and advice to contact OpenAI support for certain issues [*openai-chatter*].
- **Prompt Engineering**: Issues and resolutions regarding unwanted outputs in responses were mentioned, alongside a call out to the prompt engineering community [*prompt-engineering*, *api-discussions*].
- **AI Tools & Systems**: The introduction of the "Code Anything Now! Definitely Obey" (CAN DO) GPT creator system was shared, a tool designed to allow GPT agents to execute shell commands and manage Git tasks [*gpt-4-discussions*].

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (14 messages🔥): 
        
- **Comparisons of AI Models**: Users `@miixms` and `@i_am_dom_ffs` discussed the performance of different language models. `@miixms` found that **ChatGPT** performed better than Bing Bard for a simple task, while `@i_am_dom_ffs` shared their personal experience that **Mistral-medium** did not show any improvement over **Mixtral**. User `@eljajasoriginal` asked about the performance of **Claude Instant** and **Claude 2.1**.
- **API Information**: User `@utkarsh` asked about making a **Nodejs CLI application using the OpenAI API**. `@miixms` recommended asking the same question to ChatGPT.
- **Intent Recognition**: User `@the_black_hat` inquired about good **intent recognition software for chatbots**.
- **Usage of Agents in ChatGPT**: User `@jeannehuang86` sought advice on using **agents in ChatGPT** to play multiple roles. `@lugui` suggested utilizing the API for having separate agents but mentioned that in ChatGPT, the same agent will need to perform all roles. `@michael_6138_97508` suggested considering the **Azure OpenAI platform**. 
- **Usage Limit for GPT Models**: User `@exilze` mentioned that there's a **usage limit for GPT4** and shared that they were unable to use ChatGPT due to these limits.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (175 messages🔥🔥): 
        
- **Concerns over unauthorized access and data leakage**: User `@infidelis` reports multiple instances of new chats appearing in their ChatGPT history which they did not start, raising concerns about possible data leakage or unauthorized access to their account. They changed their password, ran antivirus software, and checked for suspicious browser extensions, but new chat logs continue to appear. User `@lugui` suggests the issue might be due to stolen credentials and advises `@infidelis` to contact the OpenAI support team.

- **ChatGPT versus API usage**: `@infidelis` contemplates shifting from ChatGPT to API as the latter could be cheaper for their usage. However, `@sxr_` cautions them that if their OpenAI account has been compromised, transitioning to API might increase risk as potential hackers could misuse their API key.

- **Experience with DALL-E and spelling in image creations**: `@jonahfalcon.` shares their humorous encounters with DALL-E's image generation, noting that the AI often spells words inaccurately within images. `@lugui` attributes this to DALL-E and not ChatGPT, which is responsible for generating the text prompt.

- **Difficulty with platform support**: User `@infidelis` talks about their negative experience with the OpenAI bot support, finding it ineffective for resolving specific issues.

- **Discussion around ChatGPT and API mistakes**: A hypothetical about mistakes in API usage is brought up by user `@mysticmarks1`, who jokes about client’s blaming OpenAI for wrongful expenditures.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (39 messages🔥): 
        
- **Issues with GPT Model on iOS**: User `@readingpalms` reported that their conversations on the **iOS app** start in **GPT4** but switches to **3.5** after the first prompt. This issue appears to be isolated to the iOS app, and persists despite the user having reinstalled the app.

- **Problems with Chat History and Model Outputs**: Several users, including `@lemar` and `@skrrt8227`, reported issues with their **chat history disappearing** and **unexpected image outputs** from the models. These issues were not directly addressed or resolved.

- **Numerical Limits on GPT Conversations**: User `@m54321` reported that a certain **undisclosed number of messages** per chat induces an error, forcing them to start a new chat. This issue appeared to remain unresolved. 

- **'Human Verification' Issues**: User `@3daisy` had troubles with continuous human verification prompts. Synchronization of system time resolved the issue on their end.

- **ChatGPT Classic Profile message limit**: `@felpsey` confirmed there is a **40-message limit** in the **ChatGPT Classic profile**, which is applicable to GPT-4 based models.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (8 messages🔥): 
        
- **API User Recognition**: `@reinesse` raised an issue about an API user not being recognized as part of their organization despite having a separate API key and Reader access. They asked for advice on how to resolve this.
- **"CAN DO" GPT Creator System Introduction**: `@_jonpo` shared a new tool termed [**"Code Anything Now! Definitely Obey" (CAN DO) gpt creator system**](https://chat.openai.com/g/g-ibrTsdfV0-can-do-creator) that enables a GPT agent to execute shell commands and manage Git tasks. This tool is designed to overcome certain restrictions in conventional GPT agents.
- **Posting GPT Links**: `@loschess` informed the group about a dedicated section for posting GPT links.
- **Understanding Rate Limits**: When `@phobir` asked whether clicking the "Continue generating" button would be counted towards the 40 requests/3h limit, `@solbus` confirmed that it does.
- **Message Cap on Custom GPTs**: `@draculabutbackwards` inquired about bypassing the message cap on custom GPTs, and `@satanhashtag` clarified that it is currently not possible.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (3 messages): 
        
- **Issue Resolution**: User `@exhort_one skrrt8227` reported that they managed to fix a previously discussed issue related to undesired output in the responses.
- **Clarification on Issue**: `@exhort_one skrrt8227` further explained they were referring to the text at the beginning, the end, and the footer in the responses. They stated that they've tried different methods to resolve the issue, including adding visuals to highlight the problematic sections.
- **Community Call**: User `@beanz_and_rice` reached out to others in the prompt engineering community, presumably for discussion or assistance.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (3 messages): 
        
- **Fixing Output Answers Issue**: `@exhort_one skrrt8227` mentioned they fixed the issue related to the output answers.
- **Text Substitution Difficulties**: `@exhort_one skrrt8227` has tried different substitutions for the text at the beginning, end and footer of the responses, and resorted to circling and crossing out these parts in screenshots.
- **Prompt Engineering**: `@beanz_and_rice` enquired about the presence of any prompt engineers in the channel.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Discussion about **Mixtral Access and Versioning** in the guild, with users such as `@hydroxide.` and `@tesseract_admin` working to understand usage and access details, while `@sublimatorniq` pointed out that `Perplexity` hosts `Mixtral`.
- Various concerns and suggestions relating to the use of **Dolphin Mixtral for Chatting**, spearheaded by `@cl43x`'s issues with `Mistral` model censorship during hosting with `Ollama`, leading to the exploration of the `dolphin-mixtral` model as a less censored option referenced by `@iammatan` and `@ethux`. The part of the conversation also saw a link suggestion by `@ethux` to [Dolphin 2.2.1 Mistral 7B - GGUF](https://huggingface.co) model.
- Clarification and resolution attempts of issues including `'Prompting'` and **Errors in Oogaboogas' Text GUI**: `@cl43x` received assistance from `@ethux`, who also suggested joining AI communities for learning more about 'Prompting'.
- A range of **Uncensoring Methods** proposed by `@blueridanus`, from prompt modifications to character development and smart prompts. The conversation also included suggestions from `@faldore` and `@ethux` about using different models for varied results and a link to a [prompt repository](https://github.com/ehartford/dolphin-system-messages) on `github`.
- Channel contributions also covered key **Tech Requirements Clarifications** led by `@hovercatz` and `@dillfrescott`, who explained the distinction between VRAM and system RAM for interpreting technical specifications on `HuggingFace`.
- In the **deployment** channel, `@dutchellie` cautioned members about the uncensored nature of a bot, receiving both positive feedback from `@weird_offspring` and criticisms from `@frosty04212`, showcasing varied user responses.
- Comparisons and Discussions on **Code Generation**, **Efficiency**, and **Performance of Mistral-medium, Mistral-small, and GPT-3.5-Turbo** featured heavily in the random channel, with `@jackson_97091`, `@poltronsuperstar`, and `@victronwolfson` contributing their insights and experiences.
- Lastly, discourse on **API Support for Tools** and **Discord Bot Model Switching** were covered by `@victronwolfson`, pointing out limitations in compatibility with tools like OpenAI and also sharing his developments with a Discord bot capable of switching between models.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (79 messages🔥🔥): 
        
- **Mistral Access and Versioning**: User `@hydroxide.` clarified that `mistral-small` corresponds to `Mixtral 8x7B` and questioned the specifics of `mistral-medium`, while `@tesseract_admin` requested API access info and methods to use `Mixtral` outside of Mistral's own hosted inference. `@sublimatorniq` added that `Perplexity` hosts `Mixtral`.

- **Potential Use of Dolphin Mixtral for Chatting**: User `@cl43x` expressed frustration with apparent censorship in the `Mistral` model while trying to host it with `Ollama`. He was guided by user `@iammatan` and `@ethux` towards the model `dolphin-mixtral`, which was claimed to be less censored. `@ethux` suggested a link to `huggingface.co` for the `Dolphin 2.2.1 Mistral 7B - GGUF` model and was told by `@cl43x` that he was downloading it and would try it with `Oogaboogas web UI`. Concerns about the necessary VRAM were addressed with `@cl43x` clarifying his system had 8GB VRAM and `@ethux` speaking from his experience with a RTX 3090 and RTX 2080 Ti, suggesting it would likely work. 

- **Errors and Little Understanding of 'Prompting'**: `@cl43x` raised the issue of receiving errors while loading the model on `Oogaboogas` text GUI which was unresolved at the end of the conversation history. The user also confessed to not understanding the term 'prompting', to which `@ethux` suggested looking at various AI communities to learn more, such as `TheBloke's` community and `LearnAI together`. 

- **Possible Uncensoring Methods**: User `@blueridanus` suggested to 'uncensor' the AI by changing up the prompt format or creating a character and using a smart prompt. `@ethux` raised the idea of using different models for different results (e.g. Mistral Instruct for complying to requests) and `@faldore` shared a link to a [prompt repository](https://github.com/ehartford/dolphin-system-messages) on `github`.

- **Tech Requirements Clarifications**: Lastly, `@hovercatz` and `@dillfrescott` clarified the difference between VRAM and system RAM when interpreting technical specifications for tools on `HuggingFace`.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (3 messages): 
        
- **Uncensored chatbot**: `@dutchellie` warned users that the bot answers to all inquiries, emphasizing its uncensored nature.
- **User Reactions to the Uncensored Bot**:
    - `@weird_offspring` expressed enthusiasm about such a feature, highlighting the potential for improvement and learning.
    - `@frosty04212` had a different perspective and expressed dissatisfaction, finding the bot to be too aligned and unresponsive to their requests.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 messages): 
        
- **Comparing Code Generation**: `@jackson_97091` and `@poltronsuperstar` discussed that **code generation** with Mistral is more than decent."
- **Mistral-medium vs GPT-3.5-Turbo**: `@victronwolfson` shared his initial testing results saying that **Mistral-medium is pretty clearly better than GPT-3.5-Turbo**.
- **Mistral-small vs GPT-3.5-Turbo Efficiency**: `@victronwolfson` also mentioned that **Mistral-small performs equivalently to GPT-3.5-Turbo for 66% the cost**.
- **API Support for Tools**: `@victronwolfson` highlighted that the **API doesn't currently support tools like OpenAI**, but he managed to put a middleman to make all three models respond in a tool-like manner.
- **Discord Bot Model Switching**: `@victronwolfson` shared that his **Discord bot can switch between models on command** which allows him to play with them.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Widespread discussion on **model fine-tuning**, particularly fine-tuning a model on one dataset and then continuing fine-tuning that model on another one, with users exploring its effectiveness if the template remains the same. Another focus was fine-tuning models for specific tasks and the best platforms to do so. 
- Technical exploration of a **regression in Mistral** following an update, including speculation surrounding the reason for the change, specifically referencing the dropout commit as a potential cause.
- _"The dropout was previously set to 0.0, but now it's set to follow the config value, which might possibly be None."_ (`@caseus_`)
- An extensive talk on **training and inference of GPTs and LoRA/QLoRA Models**, and the possible bugs when testing these models both locally and on Axolotl-cli. Attention was given to model coherence and the problem of prolonged run-on sentences.
- The creation of a **manually curated AP News 2023 Tiny Dataset** by the user `@xzuyn`, with a total of 288 samples aimed at sensitizing language models to real-time events. AP News articles are included in the dataset with the emphasis on current topics, which may introduce bias based on the time of collection. The dataset can be accessed via the [HuggingFace website](https://huggingface.co/datasets/PJMixers/AP-News-2023-Tiny).
- Discussion of **hardware limitations** when training large models. Users shared experiences encountering Out-of-Memory (OOM) errors even when modifying various model and optimizer configurations.
- A dialogue on the **possibility of expanding a 7B Mistral model to a larger model** while only training certain layers due to hardware constraints. FFT and options for training interface layers were considered in the discussion.   


**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (1 messages): 
        
- **Fine-tuning Models on Multiple Datasets**: `@noobmaster29` asked if anyone has tried to fine-tune a model on one dataset and then continued fine-tuning that model on another one. They explored the idea's effectiveness if the template remains the same.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (6 messages): 
        
- **Regression in Mistral**: `@nanobitz` reported a regression in **Mistral** from a commit preceding the Mixtral update. They questioned whether the regression is exclusive to Mistral or if it impacts other models as well.
- **Discussion on Cause of Regression**: `@caseus_` speculated if the regression could be due to the changed `dropout commit` referenced by `@casper_ai`. The dropout was previously set to 0.0, but now it's set to follow the config value, which might possibly be None.
- **A6000 Performance Measurement**: `@caseus_` further mentioned testing FFT Mistral on a single **48GB A6000**.
- **Mistral Configuration File Shared**: `@caseus_` also shared a [gist link](https://gist.github.com/winglian/1a519b72f9561170c0d2bf58cee93a09) to the yml configuration file used for Mistral.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (62 messages🔥🔥): 
        
- **Discussion on Training and Inference of GPTs and LoRA/QLoRA Models**: Users `@self.1` and `@noobmaster29` discussed various configurations, the behaviour of End of Sentence (EOS) tokens, and possible bugs when using ooba with the chatml template. They tested their trained models locally and using Axolotl-cli, keeping the focus on issues with model coherence and prolonged run-on sentences.
- **Expanding Mistral Model to 13B with QLoRA**: `@xzuyn` and `@caseus_` explored the possibilities of expanding a 7B Mistral model to a larger model while only training certain layers due to hardware constraints. They explored using Freeze-Setting Techniques (FFT) and options of training interface layers.
- **Hardware Limitations When Training Large Models**: `@xzuyn` experienced Out-of-Memory (OOM) errors even when modifying various model and optimizer configurations, seeked advice on how to circumvent these issues.
- **Running Inference on Multiple ShareGPT Conversations**: `@semantic_zone` enquired about running inference on a collection of conversations where each conversation had multiple past messages.
- **Clarification on Fine-tuning Models**: `@tolki` sought advice on fine-tuning models for specific tasks, the appropriateness of consumer hardware for fine-tuning, the best online platform for this process and the best way to serve the model in a low usage production application. `@caseus_` provided responses to these queries.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **AP News 2023 Tiny Dataset**: User `@xzuyn` is manually creating a dataset with **AP News articles** available on the [HuggingFace website](https://huggingface.co/datasets/PJMixers/AP-News-2023-Tiny). The articles include recent events up to the date of their collection, aiming to sensitize language models to real-time events.
- The dataset currently contains **288 samples** and new entries will continuously be added. The content is focused on **current topics** with a possible bias towards topics featured on AP News's homepage during the collection period.
- Articles are presented in **Markdown** format and the oldest articles included are few months old, while the newest are from the collection day. 
- Also, this dataset is reported to be **99% clean** with potential duplicate entries being the only issue. As data collection is manual, each sample is checked for quality and relevance before being included.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- `@tomgale_` shared an ongoing discussion with an artist from New York who is exploring the intersection of AI and neural nets. This led to the discovery of a potential connection to the **Jan Sloot Digital Coding System**, and future collaboration with **Hugging Face** was proposed.
- `@ehristoforu` sought help to convert Safetensors to Diffusers format. `@not_lain` provided a Python solution and further assistance by providing a Google Colab script for the conversion process.
- `@radens__` discussed the challenges of implementing mistral in Swift on the M1 Pro due to the distribution of mistral weights as bf16 which is unsupported. The possibility of re-encoding the weights to fp16 or fp32 was explored, and `.tanuj.` suggested the use of the **MLX framework** by Apple.
- `@ivy2396` inquired about possible cooperation between HuggingFace and their distributed GPU cloud platform.
- User `@neuralink` shared their progress on implementing the **DoReMi** project and end-to-end FP8 training in 3D parallelism.
- `@devspot` highlighted **Outfit Anyone AI** and several new HuggingFace spaces, and `@th3_bull` shared a Spanish language video discussing NLP with HuggingFace spaces.
- Multiple projects were shared in the "I made this" channel - from a [Stable Diffusion Generated Image Downloader](https://www.kaggle.com/code/yoonjunggyu/stable-diffusion-generated-image-downloader) by `@yjg30737` to `@cloudhu` sharing a model of a DQN agent playing SpaceInvadersNoFrameskip-v4 and `@om7059` implementing the DeepDream program by Google in PyTorch.
- User `@blackbox3993` posted an issue with the fine-tuned Mistral model behaving as the base model instead of the fine-tuned variant upon reloading on HuggingFace. They provided [source code](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919) for analysis and troubleshooting.
- `@blackbox3993` and `@tomgale_` discussed potential computer vision projects and collaborations. `@srikanth_78440` sought assistance for fine-tuning a multimodal language-and-vision model.
- `@stroggoz` posed a question about the application of mixture of expert architectures in tasks beyond language model tasks and `@hafiz031` sought advice on how to chunk a large corpus effectively for an [Open Book Question Answering/Retrieval Augmented Generation](https://stats.stackexchange.com/q/635603/245577) system.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (38 messages🔥): 
        
- **Artist Moving Towards AI**: `@tomgale_` discussed his correspondence with an artist in New York who is showing interest in the field of AI and neural nets. He related the artist's work with an invention from 1999, the Jan Sloot Digital Coding System. Tom is hoping to get in touch with **Hugging Face** as he believes he can prove that the coding system was a sort of LLM based on the pitches of contrast and the VCR and expressed his desire for assistance on this matter. 

- **Conversion of Safetensors to Difusers format**: `@ehristoforu` asked how to convert Safetensors to Diffusers format. `@not_lain` provided a Python snippet to achieve with the ***diffusers*** library and solved a subsequent *omegaconf* installation issue. 

- **Colab Code for Conversion**: `@not_lain` also offered further assistance by creating a script on Google Colab and sharing it with `@ehristoforu` to help with the conversion process. He later clarified the script was for one model only but can be applied to others as well.

- **LLMs on M1 Pro**: `@radens__` sought advice about implementing mistral in Swift on an M1 Pro. He expressed concern over the mistral weights being distributed as bf16, which is unsupported on the M1 Pro. He explored the idea of re-encoding the weights to either fp16 or fp32.

- **MLX framework on M1**: `.tanuj.` suggested `@radens__` to consider the **MLX framework** by Apple, providing a [GitHub link](https://github.com/ml-explore/mlx-examples/tree/main/llms/mistral) for more information.

- **Cooperation Enquiry**: `@ivy2396` expressed an interest in exploring cooperation opportunities between HuggingFace and their distributed GPU cloud platform and sought for contacts to carry forward these discussions.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (1 messages): 
        
- **DoReMi Implementation Progress**: User `@neuralink` shared his progress on implementing **DoReMi** project, stating that **20%** of it has been covered.
- **End-to-End FP8 Training in 3D Parallelism**: `@neuralink` also mentioned that he managed to implement **10%** of end-to-end FP8 training in 3D parallelism, with the exception of **FP8 kernels**.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (2 messages): 
        
- **Outfit Anyone AI and HuggingFace Spaces**: `@devspot` mentioned the **Outfit Anyone AI** and several new spaces now available on HuggingFace. They created a [video](https://youtu.be/QBCDgcQlS6U) to cover these updates and help users stay informed about the latest models on HuggingFace.
- **NLP Spaces of HuggingFace**: `@th3_bull` shared a [Spanish video](https://m.youtube.com/watch?v=wSI8shazYaA&list=PLBILcz47fTtPspj9QDm2E0oHLe1p67tMz&index=7&pp=iAQB) about NLP from 0 to 100 with HuggingFace spaces.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (13 messages🔥): 
        
- **Stable Diffusion Generated Image Downloader**: `@yjg30737` has created and made public a [Kaggle notebook](https://www.kaggle.com/code/yoonjunggyu/stable-diffusion-generated-image-downloader) to improve the quality of image generation and educate others on the subject.
- **Web-based Mixing Experiment**: `@.bigdookie n278jm` shared that they are working on a minimal web-based mixing experiment to play with the v7 wavsurfer updates.
- **Parameters for randomizer_arr**: `@andysingal` queried `@yjg30737` about the parameters for randomizer_arr and from where the generated images are being sourced.
- **Comparison with Mistral AI API**: `@andysingal` also asked `@qbert000` about how their project measures up against the Mistral AI API and if it's easy to integrate with any Operating System component like langchain, llamaindex.
- **DQN Agent playing SpaceInvadersNoFrameskip-v4**: `@cloudhu` shared a model of a DQN agent playing SpaceInvadersNoFrameskip-v4 using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3) and the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).
- **DeepDream program in PyTorch**: `@om7059` has implemented the DeepDream program by Google in Pytorch that transforms ordinary images into dream-like compositions and shared some images generated by it on Twitter ([link](https://x.com/alve_om/status/1738968534347292945?s=20))


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 
        
- **Fine-tuning Mistral Model Issue**: `@blackbox3993` posted an issue they're facing with the fine-tuned Mistral model. After saving and reloading the model on HuggingFace, the results from the `evaluate` function seem to correspond to the base model rather than the fine-tuned variant. They shared the [code used in the process](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919) for analyzing where potential errors might lie.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **Computer Vision Projects for Job Market**: `@blackbox3993` is seeking suggestions for computer vision projects that can enhance their job market profile. They are also open for collaboration in building some cool projects.
- **Collaboration Request**: `@tomgale_` is looking for help on a project, the details of which he has put in general chat. He specifically pointed out that `@blackbox3993` might be interested.
- **Fine-tuning Multimodal LLM**: `@srikanth_78440` is seeking assistance with instructions for fine-tuning a multimodal language-and-vision model like LLAVA2 using a custom image dataset.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (2 messages): 
        
- **Mixture of Expert Architectures for Non-LLM Tasks**: `@stroggoz` raised a question regarding the absence of mixture of expert architectures for tasks like named entity recognition and others beyond language model tasks.
- **Chunking Large Corpus for Context-Rich Retrieval**: `@hafiz031` sought advice on effectively chunking a large corpus to optimize retrieval. They are specifically aiming to build an [Open Book Question Answering](https://huggingface.co/tasks/question-answering) / [Retrieval Augmented Generation](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) system. Their question details potential challenges and context selection issues. They shared a [link](https://stats.stackexchange.com/q/635603/245577) to their detailed query on StackExchange.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 
        
- **Issue with Fine-Tuned Mistral Model**: User `@blackbox3993` reported an issue regarding the **Mistral model** they fine-tuned and saved on HuggingFace. They mentioned that upon loading the model, they are not getting the expected results as the `evaluate` function seems to produce results similar to the base model. They've shared the code they're using on the [huggingface/peft GitHub issues page](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919) and request help in identifying what they might be doing wrong.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- User `@raghul_64090` encountered a **TypeError: 'MistralRunner' object is not callable** while using openllm for mistral 7b and sought guidance on resolving the issue both in the general and tutorials channels.
- A discussion about bots and resources for learning about LangChain was ignited by `@meowthecatto`'s query. It was responded with suggestions of multiple resources like kapa.ai on Discord, dosu-bot on Github and the chat with LangChain docs by `@seththunder`.
- User `@doublez_` experienced an issue involving an **`error: externally-managed-environment`** during the installation of an external package via pip.
- `@ivanryzk` expressed confusion over the status of Zapier integration with LangChain, citing that despite LangChain docs stating it as deprecated, Zapier docs still list LangChain.
- `@cryptossssun` inquired on how to extract JSON format data from a PDF, especially from image-based table PDF files and subsequently received guidance from `@quantumqueenxox` and `@rajib2189`.
- Announcement of job openings for the positions **AI Engineer** and **Game Developer** in a company at the intersection of AI, gaming, and blockchain technology by `@jazzy3805`. Candidates with proficiency with LangChain are desired for the AI engineer role.
- User `@reachusama` shared a [LinkedIn post](https://www.linkedin.com/posts/reach-usama_github-reachusamaupworkgpt-upworkgpt-activity-7142620647964176385-B2ic?utm_source=share&utm_medium=member_ios) promoting a personal GitHub project in the share-your-work channel.
- `@shamspias` introduced a new Gemini API web application project, designed specifically for Gemini using LangChain. This project is [open-source](https://github.com/shamspias/langchain-gemini-api) and features numerous functionalities such as multimodal conversation capabilities, FastAPI build, Redis integration for persistent conversation history, compatibility with various applications, a simple API mechanism supported by Redis, and asynchronous and streaming responses. Users are encouraged to explore, use, and contribute to this project.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (19 messages🔥): 
        
- **MistralRunner TypeError**: User `@raghul_64090` encountered a **TypeError: 'MistralRunner' object is not callable** while using openllm for mistral 7b and sought guidance on resolving this.
- **Bot Query**: User `@meowthecatto` queried about a bot which answered questions about LangChain. The response from `@seththunder` indicated multiple resources are available including kapa.ai on Discord, dosu-bot on Github and the chat with LangChain docs.
- **pip install -e Error**: User `@doublez_` raised an issue they faced while installing an external package via pip - `error: externally-managed-environment`.
- **Zapier Integration**: `@ivanryzk` asked `@jonanz` and `@251662552210210816` for updates on the Zapier integration, citing that despite LangChain docs mentioning it as deprecated, Zapier docs still list LangChain.
- **PDF Data Extraction**: `@cryptossssun` sought advice on extracting JSON format data from a PDF, particularly from image-based table PDF files and received some guidance from `@quantumqueenxox` and `@rajib2189`.
- **Job Postings**: `@jazzy3805` announced job openings for **AI Engineer** and **Game Developer** for a company at the intersection of AI, gaming, and blockchain technology. The AI engineer role specifically calls for proficiency with LangChain. Interested persons were urged to DM for more details.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **Self-Promotion**: User `@reachusama` shared his [LinkedIn post](https://www.linkedin.com/posts/reach-usama_github-reachusamaupworkgpt-upworkgpt-activity-7142620647964176385-B2ic?utm_source=share&utm_medium=member_ios) which is about his new GitHub project.
- **Gemini API Web Application**: User `@shamspias` introduced his new project specifically designed for Gemini using Langchain. The key features of this project include:
    - Multimodal conversation capabilities.
    - Built with FastAPI.
    - Redis integration for persistent conversation history.
    - Compatibility with various applications.
    - Simple API mechanism supported by Redis.
    - Asynchronous and streaming responses.
  
  The Gemini API application is open-source and available on [GitHub](https://github.com/shamspias/langchain-gemini-api). Users are encouraged to explore, use, and contribute to the project.


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **Issue with Mistral 7b using openllm**: User `@raghul_64090` reported encountering a `TypeError: 'MistralRunner' object is not callable` issue and sought guidance on what this error signifies.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Detailed discussion on Mixtral implementation and training configurations:
    - Inquiry about the "**Gate Layer Freezing**" technique in Mixtral training, with a reference to a [Tweet](https://twitter.com/erhartford/status/1737350578135834812) by Eric Hartford stating the significance of this approach. The question arose regarding the application of this technique in Axolotl.
    - Share of **Dolphin 2.6 training configurations** on Mixtral-8x7b by Eric Hartford, accessible [here](https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b/blob/main/configs/dolphin-mixtral-8x7b.yml). Whether this approach is the most efficient was brought into question.
    - Performance discussion of the **Dolphin 2.6 model**, indicated as more 'dolphin-like' than its predecessor at the 1.5-epoch mark, with no model refusals noticed.
    - Inquiry about successful **8-bit training with Mixtral script** in an H100 pod, particularly in relation to Axolotl.

- Conversation on software for quick OpenAI-compatible operations:
    - Proposal of a **software solution** for instant OpenAI-compatible API interactions and predefined actions via a hotkey, insightful across any operating system. 
    - Suggestion of software, [uniteai](https://unite.ai/) and [ClipboardConqueror](https://github.com/aseichter2007/ClipboardConqueror), that might meet some of these requirements. 
    - Interest expressed in developing such a software if it does not exist.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (8 messages🔥): 
        
- **Gate Layer Freezing in Mixtral Implementation**: User `@sebastian.bodza` referenced a [Tweet](https://twitter.com/erhartford/status/1737350578135834812) by Eric Hartford stating the importance of freezing the gate layer during Mixtral training. He wondered if this technique has been implemented in Axolotl.
- **Dolphin 2.6 Training Configurations**: `@sebastian.bodza _jp1_` mentioned that Eric Hartford has shared his Dolphin 2.6 training configurations on Mixtral-8x7b, available [here](https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b/blob/main/configs/dolphin-mixtral-8x7b.yml). However, it is unclear if this is the most efficient training approach.
- **Dolphin 2.6 Model Performance**: User `@rtyax` indicated that Dolphin 2.6 is more 'dolphin-like' than version 2.5 at the 1.5-epoch mark, and noted that they did not observe any model refusals.
- **8-bit Training with Mixtral Script**: `@tcapelle` asked if anyone in the server has been able to successfully train using an 8-bit Mixtral script in an H100 pod. Tcapelle also offered to run any experiments if provided with a configuration file. He later specified this question applied to application in Axolotl.


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (6 messages): 
        
- **Software for Quick OpenAI-compatible Questions and Actions**: `@.grey_` asked if there exists software allowing users to quickly ask questions through any OpenAI compatible API, or use clipboard content for predefined actions via a hotkey, working across any operating system. This software could ideally integrate with the OS's context menus, similar to Alfred or Spotlight on MacOS.
- `@.grey_` also mentioned that this tool could be used for quick questions, predefined actions, or opening a chat without requiring a full context. It is especially useful when having a one-off question or needing a quick response while reading something.
- `@bjoernp` responded that they were not aware of such a software but found the idea useful.
- `@rtyax` suggested [uniteai](https://unite.ai/), an lsp server that can fit into IDEs and can bind lsp actions to a hotkey. `@rtyax` also mentioned [ClipboardConqueror](https://github.com/aseichter2007/ClipboardConqueror), a GitHub project, which could meet some of `@.grey_`'s needs.
- `@.grey_` also expressed interest in developing such a software if it does not already exist.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **AI Events of the Year**: `@swyxio` shared a [link](https://arstechnica.com/information-technology/2023/12/a-song-of-hype-and-fire-the-10-biggest-ai-stories-of-2023/) to an article summarizing the biggest AI stories of 2023.
- **Mamba Model Mention**: `@swyxio` briefly mentioned the Mamba model, although no further details were provided.
- **Discussion on LangChain Utility**: `@cakecrusher` questioned the utility of LangChain and asked for reasons to use it over ChatGPT. In response, `@lightningralf` suggested that LangChain allows for easy swapping between different add-ons like vector stores. 
- **Building RAG That Considers Temporal Factors**: `@swizec` suggested to `@gratchie1188` the possibility of creating a Retrieval Augmented Generation (RAG) model that values relevancy and temporality. If simple temporal distance weighing does not perform well, a summarization-based recursive memory system could be implemented.
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Building Autonomous Web Agents**: User `@coopermini` is seeking advice and datasets for building autonomous web agents with a focus on reliability, regardless of whether they are GUI based or LLM based.
- **Recommended Datasets**: `@coopermini` asked for dataset recommendations, specifically alternatives to the mind2web dataset.
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Discussion on Vision Models**: User `@thebaghdaddy` asked for opinions on the leading vision model to compare with a model they trained. They mentioned that they are aware of the **GPT4V** model but read that it's mid-tier currently. No response or additional discussion is seen yet.
        

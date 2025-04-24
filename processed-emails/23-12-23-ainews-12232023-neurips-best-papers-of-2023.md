---
id: 36ff4b14-66cf-4d3a-a67d-abedc139f795
title: '12/23/2023: NeurIPS Best Papers of 2023'
date: '2023-12-24T07:45:58.983278Z'
original_slug: ainews-12232023-neurips-best-papers-of-2023
description: >-
  The Latent Space Pod released a NeurIPS Best Papers recap, discussing AI
  performance optimization, malware security, video and music content, technical
  advancements in machine learning, LLM development, and architecture
  discussions including quantization and the Striped Hyena architecture.
companies:
  - neurips
  - huggingface
  - apple
models:
  - gpt-4
  - palm-2
  - hermes-2.5
  - mistral
topics:
  - ai-performance
  - malware-security
  - video-content
  - music-content
  - machine-learning
  - large-language-models
  - llm-interpretability
  - model-merging
  - quantization
  - attention-is-all-you-need
  - striped-hyena-architecture
---


<!-- buttondown-editor-mode: plaintext -->Shameless plug time - the Latent Space Pod shipped the [first NeurIPS Best Papers recap pod](https://twitter.com/latentspacepod/status/1738709627829883346)!

[![image.png](https://assets.buttondown.email/images/cc6e84b7-1b92-43da-9a96-94e656ea5aba.png?w=560&fit=max)](https://twitter.com/latentspacepod/status/1738709627829883346)

3hrs of the best papers of 2023. enjoy.



[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Discussions regarding **optimizing AI performance** with a focus on using shorter contexts for better results, potentially transitioning these insights to GPT-4 and publishing them as blogs.
- An active exchange about **malware security** in relation to HuggingFace, with a user sharing personal experience of a potential malware threat via email.
- Continued interest in **video and music content**, with users sharing YouTube links of various genres, including a discussion on the possible upscale of YouTube. Issues around clickbait in AI reporting also revealed, calling for more honest representations in AI-related media.
- In-depth conversation on **technical advancements in machine learning**, featuring research papers about DYAD, a novel alternative to linear layers, and Apple's newly launched ML Ferret. Users also navigated the process of accessing PALM2 through an API key, with plans to discuss Markovian-type planning for agent LLMs.
- Discussion on **Large Language Models** (LLM) centered around building specialized models, handling data scaling (with the sharing of relevant code), embedding and vector database management, exploring model merging strategies, and LLM interpretability resources. The community also shared various AI model performance results (e.g., Hermes 2.5, GPT-4, and Mistral).
- Active exploration of **Striped Hyena Architecture** and quantization in the ask-about-llms channel, discussing quantization challenges, RMSNorm issues, and potential fixes. Users also brought attention to an added red hint to "Attention is All You Need" and discussed issues with a NousResearch model.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (3 messages): 
        
- **Using Shorter Context for Better Results**: User `@cognitivetech` shared their view that **shorter context** yields better results when working with chatbot AI, rather than trying to use a long context and summarize a large amount at once. This observation holds true even when transitioning to **GPT-4**, according to them.
- **Request for Published Insights**: `@cognitivetech` expressed a wish for these insights to be published in a blog for easy reference in future discussions.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (32 messages🔥): 
        
- **Possible YouTube Upscale Discussion**: User `@fullstack6209` shared a [YouTube link](https://youtu.be/fbq-YlbUO_4?t=388) and expressed surprise about a possible upscale of all of YouTube. The subject matter of the video is Gabrielle Drake's portrayal in the SHADO UFO series.
- **Malware through HuggingFace Reference**: User `.beowulfbr` shared their personal experience of receiving an email from a supposed South Korean researcher, offering a $15 Amazon Gift Card in exchange for completing a study form. Beware of potential **malware**. The email was received due to the user's activity on HuggingFace.
- **Song Recommendation and Appreciation**: `@fullstack6209` shared another [YouTube music video link](https://www.youtube.com/watch?v=CqaAs_3azSs) to the song "Anvil" by Lorn. User `.beowulfbr` expressed admiration for the shared tune and requested `@fullstack6209` to share their playlist.
- **AI YouTube Channels Discussion**: User `@Error.PDF` expressed their disdain for YouTube channels reporting on AI but using misleading clickbait thumbnails, especially images of robots from the 'Black Mirror' series. `@n8programs` expressed a desire for an AI YouTube channel delivering non-clickbait actual news.
- **AI Explained - YouTube Channel Recommendation**: `@henriqueln7` shared the [YouTube link](https://m.youtube.com/@aiexplained-official) to the "AI Explained" channel, suggesting it as a good source for AI news, albeit slightly clickbait.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (9 messages🔥): 
        
- **Discussion on DYAD**: `@euclaise` shared a link to a research paper about DYAD, a layer designed to serve as a faster and more memory-efficient alternative to linear layers (nn.Linear() in Pytorch). This is used in common subcomponents like the ff module of Transformers. [Link to the research paper](https://arxiv.org/abs/2312.06881).
- **ML Ferret by Apple**: `@tofhunterrr` shared a link to Apple's Machine Learning repository called ML Ferret. The solution is described as an End-to-End Machine Learning Language Model that can accept any form of referring and ground anything in response. [Link to ML Ferret](https://github.com/apple/ml-ferret).
- **Access to PALM2 through the API key**: `@night_w0lf` and `@fullstack6209` had a discussion about accessing PALM2 through an API key provided at [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey).
- **Planning for Markovian-type planning for agent LLMs**: `@gabriel_syme` suggested setting up a meeting to discuss the Markovian-type planning for agent LLMs.
- **Reflection on the Power of Scale in Language Modelling**: `@gabriel_syme` shared a blog post discussing how the power of scale revealed in language modeling leads back to compositionality. [Link to the blogpost](https://windowsontheory.org/2023/12/22/emergent-abilities-and-grokking-fundamental-mirage-or-both/).


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (303 messages🔥🔥): 
        
- **Building Specialized Models and Data Scaling**: User `@nanowell` brought up the topic of building a set of specialized models that function differently but work together, to which `@n8programs` suggested the idea of training each expert model on different areas. `@emrgnt_cmplxty` also shared experiences and challenges in managing large amounts of data (4TB database) and talked about the necessity of more scalable strategies to handle about 100TB of high quality data. User `@tokenbender` discussed the balance between cost, latency, and accuracy in data management ([Link to code](https://github.com/SciPhi-AI/agent-search/blob/main/agent_search/search/base.py)).
- **Embedding and Vector Database Discussion**: Users `@fullstack6209`, `@gabriel_syme` and `@emrgnt_cmplxty` had an in-depth discussion on various vector database solutions and embedding generation at scale. They shared experiences with solutions like Qdrant, pg-vector, Weaviate, Chroma/Pinecone, and Jina, highlighting challenges in managing and scaling vector databases.
- **Model Merging**: The chat saw an ongoing discussion regarding model merging, specifically using methods like SLERP, Ties and others for merging pretrained large language models. Tools like MergeKit were suggested for those looking into model merging.
- **Large Language Model (LLM) Interpretability**: User `@mayhem1349` shared a repository dedicated to resources on LLM Interpretability ([GitHub link](https://github.com/JShollaj/awesome-llm-interpretability)). This collection includes open-source tools, papers, articles, and groups focused on interpreting LLMs.
- **Model Performance**: Different AI models were discussed across various aspects. `@weyaxi` shared results from slerp merge without additional training. The community also reflected on models such as Hermes 2.5, GPT-4, and Mistral regarding their performance in coding and reasoning tasks.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (25 messages🔥): 
        
- **Red Hint Added to "Attention is All You Need"**: User `@ex3ndr` noticed a red hint added to "Attention is All You Need". `@fullstack6209` speculated it could be due to legal reasons while `@lightningralf` considered it a part of Google's push about transformer's ownership.
- **Issues with `NousResearch-Yarn-Llama-2-7b-64k.Q8_0.gguf`**: User `@cognitivetech` reported issues with **`NousResearch-Yarn-Llama-2-7b-64k.Q8_0.gguf`** model, wondering if there was a specific prompt template to use. `.beowulfbr` suggested possibly using ChatML.
- **Striped Hyena Architecture**: `@casper_ai` enquired about the **Striped Hyena Architecture** drawing to use in AutoAWQ. User `@teknium` pointed him to the main contributor of Striped Hyena.
- **Striped Hyena Quantization**: A detailed discussion took place around quantizing **Striped Hyena**. User `@casper_ai` mentioned various challenges such as the inability to quantize the filter layer, though attention and MLP layers could be quantized. `@zymrael` provided helpful insights on sensitivity to quantization and the elements that can't be quantized.
- **Problem with RMSNorm in Striped Hyena**: `@casper_ai` mentioned encountering an `AttributeError` related to the `'RMSNorm' object` in the context of Striped Hyena, and considered creating a new scaling function for RMSNorm. `@zymrael` confirmed that the scale is equivalent to the weight in RMSNorm.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Debates and knowledge exchange on **OpenAI's** tools, particularly **GitHub Copilot's compatibility** with JetBrains and its effectiveness for inline coding. *"GitHub Copilot works well with JetBrains"* -`@infidelis` and `@jessicant`.
- Critiques and recommendations concerning AI applications such as Bing being referred to as one of the worst AI apps for schoolwork due to its failure and illogical content – `@caledordragontamer`.
- Exploration of the untapped potential of **quantum computers** in AI training and the suggestion to read research papers to gain a deeper understanding – `@michael_6138_97508`.
- A sudden ban occurrence raised by `@0718.eth`, highlighting the need for moderation and account security.
- Discussion on the use of **Mixtral 8x7b and Mistral Medium models** in OpenAI utilities by `@eljajasoriginal`.
- Users reporting **GPT-4's limited capability** to analyze data and extract data from files, with some speculating this could be linked with Bing integration.
- Users sharing their experiences with error messages counting towards their usage limits and the suggestion to implement user feedback to solve errors – `@busybenss`, `@xv_versus`, `@lugui`.
- User discussion on **features** expected to be introduced in the future versions of ChatGPT, such as the "My ChatGPT" feature – `@jaicraft`, `@dino.oats`.
- Users sharing dissatisfaction on current **GPT-4's responses** and suggestions on how to get improved responses – `@gionta21`, `@rendo1`.
- Dialogs on challenges with **OpenAI API connections** and potential solutions, with `@bluehipp0.` sharing their experience on resolving OpenAI API issues.
- Discussions on problems encountered while upgrading **ChatGPT PLUS subscriptions** and speculation on possible cause – `@ixtatica`, `@7877`.
- User queries on prompt engineering and feedback for DALL-E image generation and potential issues and improvements in chatbots – `@eskcanta`, `@.shaw93`, `@madame_architect`.
- Conversations about Dall-E image generation to match specific user requirements, the importance of clear instructions, and potential conversion to **pixel art for game usage** – `@eskcanta` and `@suptrox`.
- Suggestions for better system message structure for chatbots and discussions about using a knowledge base for extensive system information – `@madame_architect`, `@.shaw93`.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (11 messages🔥): 
        
- **OpenAI Word Blacklisting**: `@afterst0rm` and `@i_am_dom_ffs` discussed the word 'openai' being originally filtered in the UI, but pointed out that the filtering has been **fixed and no longer necessary**.

- **GitHub Copilot with JetBrains**: `@infidelis` and `@jessicant` noted that **GitHub Copilot works well with JetBrains**. Meanwhile, `@exx1` added that Copilot is effective for inline completions.

- **AI Apps for School**: `@caledordragontamer` voiced a critical opinion about Bing, stating it's one of the **worst AI apps for schoolwork** due to frequent freezing and nonsensical information.

- **Quantum Computers and AI**: `@moldy21` expressed interest in AI training on quantum computers despite being not fully developed. `@michael_6138_97508` advised reading research papers and consulting with ChatGPT for a more solid understanding.

- **Account Banning Issue**: `@0718.eth` reported their account was suddenly banned while they were using it for code completion, seeking guidance on where to get help.

- **Preference for Mixtral Models**: `@eljajasoriginal` commended the performance of **Mixtral 8x7b and Mistral Medium models**, noticing they have fewer restrictions and can even provide opinions on a variety of subjects.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (108 messages🔥🔥): 
        
- **Issues with data analysis capability in GPT4**: User `@rendo1` reported issues with GPT-4's ability to analyze and extract data from files. It was a functionality that worked a month ago but seems to be experiencing issues now. User `@cozy_artist` also suggested that this could be related to Bing integration.
- **Errors counting towards usage limit**: Users `@busybenss`, `@xv_versus`, and `@lugui` had a discussion about error messages counting towards their usage limit. Despite `@lugui`'s claim that error messages had never counted towards the limit, other users reported contrary experiences. User `@offline` suggesting incorporating user feedback to resolve errors and possibly refund usage.
- **Anticipated feature updates**: Users `@jaicraft` and `@dino.oats` discussed upcoming updates for ChatGPT, particularly a feature known as "My ChatGPT" that was briefly rolled out last month. This feature purportedly personalizes ChatGPT based on user conversations.
- **Restricted access**: User `@hra42` reported an issue of not being able to access the ChatGPT website without a VPN, suggesting a potential IP or regional issue. `@_univurse` also highlighted that error messages show when trying to access AI text classifier.
- **ChatGPT's quality of response**: User `@gionta21` expressed dissatisfaction with GPT-4's responses, stating that GPT-3.5 provided more full and insightful responses. `@rendo1` suggested that the user be more specific in their prompts to GPT-4.
 - **ChatGPT availability**: Several users expressed concerns over the functionality of ChatGPT. While `@kaveen` and `@dino.oats` asserted that ChatGPT isn't broken, `@jaicraft` humorously suggested that it never existed to begin with. `@lumirix` jokingly termed ChatGPT as an "optical illusion".


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (148 messages🔥🔥): 
        
- **Lang Chain Discussion**: `@openheroes` expressed unfamiliarity with Lang Chain, a topic also unknown to gpt3.5.
- **GPT-4 Verification Hurdle**: `@Denis Volkov` shared his experience of GPT 4 attempting to verify if he was human.
- **Access to GPT Lists with Disabled Chat History**: `@toutudouhou` had a question about accessing GPT lists after disabling chat history. `@openheroes` confirmed that it's not possible and history needs to be enabled.
- **OpenAI API Connection Issues**: `@bluehipp0.` encountered and solved a problem regarding OpenAI API connection errors, which initially appeared to be an issue with using `"https://api.openai.com/v1"`.
- **ChatGPT PLUS Subscription Issues**: `@ixtatica` had difficulties upgrading to ChatGPT PLUS when their card, normally usable, kept getting declined. Although they hinted at the possibility of the company being compromised, another user, `@7877`, found the situation amusing.
- **Support for ChatGPT**: `@tuxmaster` was looking for ways to open a support ticket for ChatGPT as they had been waiting for a response for 2 weeks. They got advised by `@satanhashtag` to seek help at help.openai.com, but `@tuxmaster` expressed dissatisfaction with the support service, suspecting it to be run by a limited AI bot.
- **User Verification Requests for Paying Members**: `@3daisy` and `@knowcryptoshow` discussed the inconvenience of having to constantly verify their identities despite being paying members, speculating that such a process might be more relevant for freemium users.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (34 messages🔥): 
        
- **GPT-4 Features and Capabilities**: User `@mrkarumin` inquired about the capabilities of GPT-4, specifically regarding its ability to access data from 2023. `@jaicraft` confirmed that GPT-4 Turbo can access up-to-date information and is enabled by default on ChatGPT 4. They also mentioned additional features in ChatGPT 4 including Dall-e 3 and Code Interpreter.
- **GPT-4 Speed**: `@mrkarumin` noted the excellent response speed of GPT-4, compared to GPT-3.5 and `@jaicraft` suggested trying GPT-3.5 again with Plus for a super-fast experience.
- **ChatGPT Plus Access**: `@mrkarumin` inquired about getting premium access, which according to `@jaicraft` will give access to the advanced features like web search, Dall-e 3, and the Code Interpreter mentioned earlier.
- **Actions Function in GPT**: `@froggy_chacko` asked for explanations for the function of "Actions" in GPT, to which `@jaicraft` replied it enables GPT to access external things using APIs. `@sudojames` suggested checking out the 'ACTIONS GPT' from OpenAI for examples and potential use cases.
- **Disruption in GPT Functionality**: `@dystopia78` experienced an issue with custom GPT vanishing, while `@happyg` faced a problem with custom GPTs forgetting instructions but resolved it without external help.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (15 messages🔥): 
        
- **Prompt Engineering and Feedback Mechanism for DALL-E Generation**: `@eskcanta` discussed their specific requirements for image generation with `@suptrox`. Feedback was provided to hone in on a precise output, including tweaking the details of the scene, focusing on elements within the garden, and eventually specifying the desired style as pixel art. Through iterative conversation, `@suptrox` managed to generate the desired image style.
- **Potential Issue and Improvement Suggestions for Chatbots**: `@.shaw93` raised a concern about a chatbot divulging information prematurely, before establishing necessary prerequisites like if the recipient is a new client. `@madame_architect` suggested moving the crucial "first message should ask" instruction to the top and repeating it at the end of the prompt, but also highlighted that an overall quality check may be needed due to the lengthy system message.
- **Utilization of the Knowledge Base**: `@madame_architect` pointed out that significant parts of `@.shaw93`'s system instruction details might be more appropriate for a knowledge base.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (15 messages🔥): 
        
- **Dall-E Image Generation**: User `@eskcanta` sought advice from `@suptrox` on generating specific images similar to their ideal preference using **Dall-E**, and specifically wanted to avoid elements like skies and trees. `@suptrox` guided on the importance of specific instructions to the AI, and added, "*Your ability to communicate **exactly** what you want is how you succeed with AI.*"
- **Pixel Art Generation**: `@eskcanta` later sought to convert these Dall-E generated images into pixel art suitable for game usage, which led to further discussions between `@eskcanta` and `@suptrox`.
- **Chatbot and Lead Generation**: User `@.shaw93` solicited help with utilizing the assistants API for a chatbot that initially provides information before asking particular questions. They wished to ensure that certain information is only unveiled to new clients after verifying that they are indeed new clients.
- **System Message Improvement**: `@madame_architect` offered a quick fix to `@.shaw93`'s problem, advising them to move certain key instructions in their system message and repeat them at the end for better results, and suggested a quality check from a prompt engineer. They also noted that some of the system instruction details seemed more appropriate for a knowledge base.
- **Children Book Illustration Request**: User `@yantou.` made a brief request for a children book illustration.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Extensive discussions about the use and specifications of **Mixtral** on AWQ, including the **required amount of RAM** and issues around loading large models. Additionally, there were talks about the usage of regular and instruct Mixtral with the same configurations. A [GitHub link](https://github.com/apple/ml-ferret) was shared to the tool named **ml-ferret**. 
- **Gemini API Calls** were queried by `noobmaster29` who was then informed that they are not free.
- In-depth deliberations regarding support for **ROCm** and the capabilities of the new **AMD MI300x card**. The conversation also touched upon the **VRAM requirements for full model tuning** with the mention of potential solutions to fit the model on 80GB card. There was a call for compute contribution to **optimize Mixtral training** with several members ready to contribute.
- The community shared insights and problems regarding changes in the code for **merging loras** to models with the latest transformers, suggesting possible solutions like downgrading peft or using axolotl for merging. Also, they shared their experiences with **testing merged models using airoboros and LDJnr/Puffin front-ends**.
- A discussion around the **ways to lower the targeted parameters count when training certain layers using qlora**, and also a query on **canceling the completion in axolotl.cli.inference** without terminating the entire application. `Dangfutures` asked for assistance in initiating dpo on the dpo branch.
- Other topics included methods to **encode math for a fine-tuning dataset** and the use of a tool for a more human-readable **dataset preview** with Visual Code suggested as a potential tool. Confirmation that the **output from Nougat** is compatible with **Mathpix**.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (21 messages🔥): 
        
- **Gemini API Calls**: `noobmaster29` asked about the availability of free Gemini API calls. However, `@nafnlaus00 yamashi` clarified that the API calls are not free.
- **Using Mixtral on AWQ**: `dangfutures` had a query regarding the specific amount of RAM required to load Mixtral on AWQ. `@dangfutures casper_ai` indicated that **24GB RAM would be sufficient if the operating system doesn't utilize any GPU, otherwise, at least 32GB would be needed**. They further added that it fits the Mixtral model with **512 context length and 512 decoded tokens**.
- **Large Model Loading Issues**: `dangfutures` experienced kernel dying issues while loading Mixtral on AWQ. `casper_ai` explained that notebooks are usually not efficient in loading large models.
- **Use of Instruct Mixtral**: `dangfutures` sought clarification on the usage of regular and instruct Mixtral with the same configurations. `@dangfutures casper_ai` confirmed the usability with the same settings.
- **Resource Sharing**: `dangfutures` shared a [GitHub resource](https://github.com/apple/ml-ferret) named **ml-ferret**.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (53 messages🔥): 
        
- **Discussion on ROCm and AMD MI300x Support**: User `@yamashi` initiated a discussion about the support for ROCm and the capabilities of the new **AMD MI300x card**, with an emphasis on its suitability for high-performance computing. User `@noobmaster29` provided a [press release link](https://www.amd.com/en/newsroom/press-releases/2023-12-6-amd-delivers-leadership-portfolio-of-data-center-a.html) discussing the card's application in inference contexts and expressed a desire for a 48GB consumer card.

- **Considerations for GPU Upgrade**: A debate ensued between `@yamashi` and `@noobmaster29` regarding the VRAM requirements for full model tuning, with an indirect suggestion for AMD as a potential solution due to their more generous VRAM provisions. `@yamashi` expressed the need to upgrade from 4xA100 setup for FFT mixing of Mixtral model.

- **Discussion on Fulltune vs LoRA**: `@dreamgen` asked `@yamashi` about the differences perceived between Fulltune and LoRA, specifically in a medical context. 

- **Potential Solution to Fit Model on 80GB Card**: `@nruaif` suggested freezing the experts layers and using deepspeed 3 to fit the full model on 4 A100 GPUs, but `@yamashi` clarified that even a 7 billion parameter model requires around 70GB memory. 

- **Call for Compute Contribution to Optimize Mixtral Training**: User `@casper_ai` invited others to contribute compute for optimized training of Mixtral with plans to import stuff from MegaBlocks for efficient training. Both `@le_mess` and `@caseus_` offered to help with available compute resources.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (11 messages🔥): 
        
- **Changes in Merging Loras to Models Code**: `@jaredquek` mentioned that the code for merging loras to models has been greatly changed in the latest transformers and provided a link to the relevant [documentation](https://huggingface.co/docs/peft/package_reference/tuners). They also found that their old code wasn't working anymore. In response, `@nanobitz` suggested they might try downgrading peft or using axolotl for merging.
- **Testing Merged Loras with Front-ends**: `@self.1` communicated having successfully tested merged models using both airoboros and LDJnr/Puffin front-ends, and mentioned that the second new line and stop token may be unnecessary.
- **Freezing Layers in Training with Qlora**: `@xzuyn` asked if there's a way to lower the targeted parameters count when training certain layers using qlora. Their target count remained at 209M parameters despite setting fewer layers to train.
- **Axolotl CLI Inference Query**: `@marijnfs` a inquired if there's a way to cancel the completion in `axolotl.cli.inference` without terminating the entire application.
- **Initiating DPO**: `@dangfutures` asked for assistance in initiating dpo on the dpo branch.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **Encoding Math for Fine-tuning Dataset**: User `@noobmaster29` shared an interest in finding the best method to encode math for a fine-tuning dataset.
- **Dataset Preview Tool**: `@noobmaster29` expressed interest in utilizing a tool to preview encoded information in a human-readable format. Markdown preview feature in Visual Code being noted as potential tool.
- **Nougat Output Clarification**: `@noobmaster29` confirmed that the output from **Nougat** is in fact compatible with Mathpix, resolving any initial confusion.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Users' experience and technical discussions on accessing and running **MistralAI's models** locally on various systems like M1 machine with 16GB memory and Lenovo Thinkcenter i5 with 32GB RAM were discussed.
    - “*User `@djmango` shared that they were successful in running the Mistral models using an M1 machine with 16GB of memory.*” 
    - “*`@ved_ikke` added that they can run most Mistral models on a Lenovo Thinkcenter i5 with 32GB RAM.*”
- Dialog on the performance of **Mistral Medium** specifically on creative writing in Chinese, and skepticism around its open source release. The idea of getting insights about Mistral models' details for better understanding was also discussed. 
    - “*User `@ken70wtf` expressed his admiration for Mistral Medium's performance on creative writing in Chinese via poe.com, stating it's faster than gpt-3.5-turbo.*”
    - “*User `@tom_lrd` questioned the importance of having the specifics of models that will never be run locally.*”
- Discussion on optimization ideas to speed up MistralAI using an open-source package named **Unsloth**. Queries were raised about other optimization methods such as the use of float16 or 8-bit & 4-bit using bitsandbytes, and the use of Flash Attention 2 in context to MistralAI/Mixtral-8x7B-Instruct-v0.1 model. 
    - “[Reddit Post](https://www.reddit.com/r/OpenAI/comments/18o4i7d/finetune_llms_25x_faster_use_60_less_memory_by/) about an open-source package named Unsloth that claims to make finetuning via QLoRA of the Mistral model 2.2x faster and use 62% less memory by leveraging OpenAI's Triton language.”
- Conversations on the necessity of an 8k token **context window** for AGI and possibility of creating AGI with 8k context window. Also, suggestions about using a graph database for efficient code generation.
    - "*`@poltronsuperstar` believes that an AGI could be created with an 8k context window, despite its inability to contain whole codebases.*"
    - "*`@daain` proposed using a graph database to hold a semantic understanding of parsed context, for efficient code generation.*"
- Discussion on the need and potential benefits of **finetuning**, as well as the upcoming features in Mistral API – particularly, function calling support for platforms like MemGPT.
    - "*User `@krissayrose poltronsuperstar` asked whether finetuning is really needed as they had been using function calling with GPT-3 before RLHF simply through few-shots learning.*"
    - "*`@flyinparkinglot` clarified that although this feature is currently lacking, it is scheduled for future implementation.*"

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (49 messages🔥): 
        
- **Accessing MistralAI's Models**: Several users including `@antononcube` and `@sublimatorniq` discussed how to access MistralAI's models programmatically. `@antononcube` was initially having trouble using the GET method and API key set up to access the models, but eventually, managed to do so with help anddirect code examples from `@sublimatorniq`.
- **Running Mistral Locally**: `@rosethelocalfemboy` shared that they successfully ran the Mistral model, specifically the **8x7b** version, on their local machine. They found it to be of high quality, even though it was a quantized version.
- **Possibly Speeding Up MistralAI**: User `@red_code` shared a link to a [Reddit Post](https://www.reddit.com/r/OpenAI/comments/18o4i7d/finetune_llms_25x_faster_use_60_less_memory_by/) about an open-source package named `Unsloth` that claims to make finetuning via QLoRA of the Mistral model 2.2x faster and use 62% less memory by leveraging OpenAI's Triton language.
- **Hardware Requirements for Running Mistral**: In response to a query by `@daveburstein`, `@djmango` shared that they were successful in running the Mistral models using an M1 machine with **16GB** of memory. It was pointed out that it's mostly the memory speed that creates a constraint rather than CPU or GPU strength. `@ved_ikke` added that they can run most Mistral models on a Lenovo Thinkcenter i5 with 32GB RAM.
- **Fitting Mistral 8x7b on a 24Gb Card**: `@jdwebprogrammer` asked if it's possible to fit the Mistral 8x7b model on a 24Gb card. They noted that when quantized to 4-bit, the model maxed out and it seemed like it would fit in 25Gb.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (8 messages🔥): 
        
- **Mistral Medium's Performance and Open Sourcing**: User `@ken70wtf` expressed his admiration for **Mistral Medium**'s performance on creative writing in Chinese via poe.com, stating it's faster than gpt-3.5-turbo. However, he expressed skepticism over the possibility of its release as an open source and open weight model.
- **LLM Learning Resources**: User `@Bharat` requested for resources to learn **LLM** at the architecture level in order to contribute to the open source LLM community.
- **Processing Speed Comparison between Platforms**: User `@sublimatorniq` inquired about the processing speed between mistral endpoints and the perplexity endpoint on poe.com in response to `@ken70wtf`'s statement on the superior performance of Mistral Medium.
- **Inquiry on Model Details**: User `@tom_lrd` questioned the importance of having the specifics of models that will never be run locally during their exchange with `@ken70wtf`.
- **Reference to Eric Hartford's Work**: In response to `@alex_deng`'s inquiry, user `@sublimatorniq` provided links hosted by huggingface.co to uncensored models published by [Eric Hartford](https://erichartford.com/uncensored-models).


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (5 messages): 
        
- **Personal Interactions**: User `@alex_deng` asked if `@sublimatorniq` was from Cambodia to which they responded affirmatively.
- **Dolphin 2.6 Mixtral 8X7B**: `@dutchellie` shared a [link](https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-GGUF) to the model name Dolphin 2.6 Mixtral 8X7B. `@jdwebprogrammer` expressed surprise about finding the existence of the Dolphin model and it being already at version 2.6.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
- **Necessity of Finetuning**: User `@krissayrose poltronsuperstar` asked whether **finetuning** is really needed as they had been using function calling with **GPT-3** before RLHF simply through few-shots learning.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (1 messages): 
        
antononcube: https://rakuforprediction.wordpress.com/2023/12/23/wwwmistralai/


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (8 messages🔥): 
        
- **Optimizations for MistralAI/Mixtral-8x7B-Instruct-v0.1**: User `@husain3739` initiated a discussion about the optimizations used for executing the MistralAI/Mixtral-8x7B-Instruct-v0.1 in the Hugging Face Chat. They asked whether the default model was executed in full precision or if there were modifications, such as using float16 or 8-bit & 4-bit using bitsandbytes, to reduce memory requirements. They also enquired about the use of Flash Attention 2.

- **Need for an Extended Context Window**: `@poltronsuperstar` and `@daain` discussed the necessity of an 8k token context window for AGI. `@poltronsuperstar` believes that an AGI could be created with an 8k context window, despite its inability to contain whole codebases. `@daain` proposed using a graph database to hold a semantic understanding of parsed context, for efficient code generation.

- **Potential of Lower Tier Mistral Models**: `@jackson_97091` expressed interest in the new Mistral update on their API, with a 32k token limit. While it's not on par with the top-tier models, they consider this move due to perceived shifting interests of the higher tier models towards corporate liability.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (4 messages): 
        
- **Function Call Support in Mistral API**: User `@brokearsebillionaire` inquired about the presence of function calling support in the Mistral API. `@flyinparkinglot` clarified that although this feature is currently lacking, it is **scheduled for future implementation**. This news was well-received by `@brokearsebillionaire` and `@antoniofiliona`, with the latter expressing eagerness to test the feature with MemGPT.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Revolving around **Transformers** library's new **llava** support, practical experience gaining in **RL** for an undergrad, handling prompting responses by libraries like **Langchain**, suggestions for small **quantized models** for CPUs and mobile devices, and inquiries about **Hugging Face's APIs** for Android application, as well as issues related to the application process at Hugging Face and problems with a **neural network's code** and **fine-tuning stable diffusion** [general](https://discord.com/channels/879548962464493619/879548962464493622/).

- Showcased member-made apps and tools such as an AI **Emoji Translator** app, the **Mamba model architecture**, an investment strategy video utilizing **AI technologies**, a **Christmas-themed video**, and a Chrome browser extension for **Musicgen continuations** [i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/).

- Discussions on the definitions recurring in DDPM and DDIM papers particularly the symbols **alpha_bar_t** and **alpha_t**, issue exploration, and seeking effective **text embedding for coherent image generation** [diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/).

- Query about conducting **key point detection on tooth X-ray images** and calculating distances among detected points [computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/).

- Confusion regarding the differences between `AutoModelForQuestionAnswering` and `AutoModelForSeq2SeqLM` called for community-insight [NLP](https://discord.com/channels/879548962464493619/922424173916196955/).

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (20 messages🔥): 
        
- **Llava Support in Transformers**: `@meatfucker` mentioned that **Transformers** library just added **llava** support.
- **Undergrad Seeking Advice on RL**: `@swadine`, an undergrad student, is seeking advice on how to gain practical experience in Reinforcement Learning (RL) as the advanced course in **Deep RL** is not running in the upcoming semester at their university.
- **Langchain Library and Prompt Responses**: `@somefuckingweeb` asked about how libraries like **Langchain** handle prompting responses to actual tool invocations.
- **Quantized Models for CPUs and Mobile Devices**: `@vishyouluck` requested suggestions for small, quantized models which can run on CPUs and smartphones for basic Q&A and text generation tasks. `@kubuxu` suggested **Quantized Mistral 7B**.
- **Query About Hugging Face API**: `@abrarsharif` made an inquiry regarding the existence of Hugging Face's APIs like OpenAI for integration into an Android application.
- **Internship Application at Hugging Face**: `@_aabidk` asked about application process at Hugging Face for multiple internship positions.
- **Code Issue with Neural Network**: `@power9799` sought help about a code issue with a neural network. The problem was related to mismatched dimensions in batches.
- **Difficulty in Fine-tuning Stable Diffusion**: `@isleepwhenimcomfortable` requested assistance in fine-tuning stable diffusion on Collab due to directory errors.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (9 messages🔥): 
        
- **Emoji Translator AI App**: User `@gospace7575` shared the creation of an interesting AI app named [Emoji Translator](https://huggingface.co/spaces/gospacedev/emoji-translator) which is capable of translating text into emojis and vice versa. This app can generate entire stories with only a few emojis.
- **Mamba Model Architecture**: User `@qbert000` announced the successful implementation of the Mamba model architecture. They have made it available on [GitHub](https://github.com/LegallyCoder/mamba-hf) and also on Hugging Face under the collection name [Q-bert/Mamba-130M](https://huggingface.co/collections/Q-bert/mamba-65869481595e25821853d20d).
- **Investment Strategy Video**: An investment strategy video utilizing Stable diffusion, Leonardo Motion, and Pika was shared by `@andysingal`. The video can be viewed [here](https://youtube.com/shorts/236QfU1GJrk?si=FnednJMjw9cYfwbw).
- **Christmas Vibes Video**: A Christmas-themed video, seemingly created with AI, was shared by `@andysingal` to celebrate Christmas with the rest of the Hugging Face team. The video can be viewed [here](https://youtube.com/shorts/ZSvGV3B_Q1I?feature=share).
- **Chrome Browser Extension for Musicgen Continuations**: User `.bigdookie` shared his project, a Chrome browser extension for Musicgen continuations that listens for your position in a YouTube track and starts a continuation from there, while ensuring the continuation stops at the end of a bar. He also mentioned a component for arranging remixes. The project's updates were shared on this [Twitter link](https://x.com/thepatch_kev/status/1738697484627816478?s=20).


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **Understanding DDPM and DDIM papers**: User `@wukong7752` is learning the DDPM and DDIM paper and expressed confusion about the use of the symbol `alpha_t` in these papers. They noted that `alpha_t` is defined as a **decreasing sequence** in the DDIM paper, however, in many implementations, people define `alpha_t` in DDIM algorithms the same as the `alpha_bar_t` in DDPM. They are seeking clarification on this matter.
- `@pseudoterminalx` informed `@lorenz1392` about **looking into a particular issue**. The details of the issue were not given in the shared chat snippet.
- `@vipitis` expressed a desire to `@lorenz1392` to **find a text embedding** that generates **coherent images** which perform well under a specific differentiable xai evaluation metric.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **Tooth X-ray Key Point Detection and Distance Measurement**: User `@navinaananthan` inquires about the feasibility of performing key point detection on tooth X-ray images and determining the distance between the detected points using any pre-existing models.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (2 messages): 
        
- **Difference between AutoModelForQuestionAnswering and AutoModelForSeq2SeqLM**: `@opencuiguy` asked for clarification on the differences between `AutoModelForQuestionAnswering` and `AutoModelForSeq2SeqLM`. The conversation awaits input from other participants who can offer further insight.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **Query About Text Embedding for Generating Coherent Images**: `@lorenz1392` sought advice on finding a *text embedding that generates coherent images which perform well under a certain differentiable xai evaluation metric*. `@vipitis` responded with a willingness to engage in this topic.
  
- **Question on DDPM and DDIM Paper Symbols**: `@wukong7752` raised a question regarding the symbols `alpha_bar_t` and `alpha_t` used in the DDPM and DDIM papers. They found that `alpha_t` in DDIM was defined similarly to `alpha_bar_t` in DDPM by many implementations, and inquired for clarity on whether this is coincidental or based on an unmentioned reason.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- A user reported being banned from both ChatGPT Discord and LM Studio Discord without given reasons; discussion on this matter might follow up.
- A [plasma physics application](https://plasma-physics-gpt.streamlit.app/) was shared by a user, adding to the compilation of AI-related tools and projects within the community.
- An AI tool directed for education was noted to be in development, without further details provided.
- The topic of multilingual models was surfaced, with an emphasis on the future support to be provided by AI assistant Aya.
- Curiosity was expressed in the development of a model that can generate and answer questions from a large dataset, like RedPajama. The idea of a self-searching model or a Retriever-Augmented Generation method for filling knowledge gaps was also discussed.
- There was reference to the significant potential of large corpora in improving long context comprehension in models, given the presence of questions, instructions, and insights.
- A novel idea was proposed to use the Language Model itself as a hypernetwork, predicting parameters to expand or implement new layers, via a Layer-wise Relevance Propagation method specific to each task.
- The [Nasty Teachers paper](https://arxiv.org/abs/2105.07381) was discussed, with queries raised on modifying output vs altering the loss function, and implications when probabilities of all classes need to be considered.
- The challenges of monetizing AI apps were brought up, with a user stating involvement in a project aimed at simplifying selling API access.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (7 messages): 
        
- **Ban from ChatGPT Discord and LM Studio Discord**: User `@antdx316` stated that they got **banned** from both the **ChatGPT Discord** and the **LM Studio Discord**. No reason for the ban was provided in the messages given.

- **Link to Plasma Physics Application**: User `@anjor_20331` shared a [link to a plasma physics application](https://plasma-physics-gpt.streamlit.app/) but did not provide any further information or context about it.

- **AI Tool For Education**: User `@fred_fups` stated that he is building an **AI tool for education**. No further details about the project were provided in the messages given.


### ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/) (3 messages): 
        
- **Multilingual Models**: `@stereoplegic` mentioned the development of multilingual models and expressed hopes that **Aya** would assist with this task soon.

- **Question Generation/Answering from a Large Corpus**: `@stereoplegic` has plans to develop a model able to generate and answer questions from a large corpus, like **RedPajama**. He also expressed interest in the model being able to self-search or use a Retriever-Augmented Generation method to fill in gaps in its knowledge based on a given prompt or a large corpus. 

- **Long Context Comprehension in Large Corpora**: He also mentioned the significant potential of large corpora in improving a model's comprehension in long contexts, provided that relevant questions, instructions, and related insights are present.

- **Using LLM as its own Hypernetwork**: `@stereoplegic` proposed a niche idea to use the Language Model itself as a **hypernetwork**. This would help in predicting parameters to expand its existing layers or add new ones, possibly by using a Layer-wise Relevance Propagation specific to that task. He noted it could be beneficial if the loader has surplus free Virtual RAM to utilize.


### ▷ #[ft-study-main](https://discord.com/channels/1131084849432768614/1139035605616042085/) (1 messages): 
        
- **Nasty Teachers Paper Discussion**: `@mootverick` brought up the topic of the [Nasty Teachers paper](https://arxiv.org/abs/2105.07381), summarizing the methodology as creating a random spike at a few incorrect labels to produce an appearance of accuracy. They raised two questions on this approach:
  - The method might not be helpful when probabilities of all classes need to be taken into account, not just the top class.
  - The user queried if the same result can be achieved by modifying the output, as opposed to altering the loss function.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
- **Monetizing AI apps**: User `@megazord` initiated a discussion about challenges in monetizing AI apps and mentioned working on a project aimed at simplifying the process of selling API access.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- User `@shivam51` requested for a **LangSmith referral code** in the general channel.
- Various **technical issues** were discussed in the LangChain context: 
    - `@ninamani` was facing problems while trying to run the Llama-2 chat model on LangChain, encountering errors with both `ChatOllama` and `Llama2Chat` methods.
    - `@ninamani` also explored the possibility of merging features of llama-cpp-python with LangChain, mentioning earlier unsuccessful attempts and discrepancies in chat prompt templates.
- Importance of ensuring that all essential information is properly transmitted to Retrieval and **Question Answer step in RAG chains** was discussed by `@a404.eth lhc1921`.
- `@motaz_hashhoush` made an inquiry regarding **Prompt Acquisition in ConversationalRetrievalChain**, particularly when using `ConversationSummaryMemory`. A function to count the number of tokens was proposed.
- In the 'share-your-work' channel, `@rajib2189` shared a [YouTube video](https://youtu.be/Tjrk5ozze3M) and a [GitHub repository](https://github.com/rajib76/aws_bedrock_examples/blob/main/examples/04_how_to...) demonstrating how to use **AWS Bedrock Agent programmatically**. Furthermore, `@rajib2189` opened a discussion on **Prompt Optimization**, inviting inputs from those who have attempted optimization.


**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (8 messages🔥): 
        
- **Langsmith Referral Code Request**: `@shivam51` has requested a LangSmith referral code.
- **Llama-2 Chat Model Issues on LangChain**: `@ninamani` has been facing issues while attempting to run the Llama-2 chat model on LangChain. They are receiving errors while using both `ChatOllama` and `Llama2Chat` methods.
- **Merger of llama-cpp-python and LangChain**: `@ninamani` also inquired about the potential to marry features of llama-cpp-python and LangChain, highlighting a previous failed attempt and mentions the discrepancies in chat prompt templates.
- **Retrieval and Question AnswerStep for RAG chains**: `@a404.eth lhc1921` discussed the importance of ensuring all necessary information is effectively passed to Retrieval and Question Answer step in RAG chains.
- **Prompt Acquisition in ConversationalRetrievalChain**: `@motaz_hashhoush` inquired if it was possible to obtain a full prompt from ConversationalRetrievalChain before feeding it to the model, especially while using `ConversationSummaryMemory`. He further specified the need for a function that counts the number of tokens.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **Using AWS Bedrock Agent Programmatically**: User `@rajib2189` shared a [YouTube link](https://youtu.be/Tjrk5ozze3M) to a video demonstrating programmatic access to AWS Bedrock Agent. The linked [GitHub code repository](https://github.com/rajib76/aws_bedrock_examples/blob/main/examples/04_how_to...) was also provided.
- **Optimization of Prompts**: `@rajib2189` expressed a suspicion that the prompts are not optimized and welcomed input from anyone who may have tried to further optimize them.


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- In the AI General Chat, there was a resource sharing discussion. User `@swyxio` shared a link to an episode of ["The Cognitive Revolution"](https://overcast.fm/+_N6F_oTH8), enriching the community with more knowledge on AI-related content.
- An idea was proposed by `@lightningralf` about the utility of developing an **IPTC metadata filler** that could insert keywords, descriptions, and more.
- User `@gratchie1188` raised a question regarding the best practices for interfacing with time series databases due to the perceived lack of solutions for them compared to text and SQL databases.
- `@swyxio` made an announcement in the AI Event Announcements channel about a special podcast episode which is a recap of NeurIPS (Part 1). A link to the [tweet announcing the new episode](https://fxtwitter.com/latentspacepod/status/1738709627829883346) was shared for easy access.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (4 messages): 
        
- **Mamba Explainer**: User `@swyxio` shared a link to an episode of ["The Cognitive Revolution"](https://overcast.fm/+_N6F_oTH8), a podcast that provides explainers on various AI-related topics.
- **IPTC Metadata Filler Request**: User `@lightningralf` suggested the utility of an **IPTC metadata filler** with features such as keyword insertion, descriptions, and more.
- **Time series DB Interfacing**: `@gratchie1188` asked for recommendations for interfacing with time series databases, noting a lack of solutions compared to those for text and SQL databases.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **NeurIPS Recap Part 1**: User `@swyxio` announced the release of part 1 of their special weekend podcast episode - a recap of NeurIPS. They shared a link to the [Tweet announcing the new episode](https://fxtwitter.com/latentspacepod/status/1738709627829883346) for easy access.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- A curated **Github repository** for Large Language Model (*LLM*) **Interpretability**, featuring open-source tools, papers, articles, and groups was shared by `@mayhem1349`. The resources can be found at this [link](https://github.com/JShollaj/awesome-llm-interpretability)
- `@burnydelic` suggested adding the [Mech Interp Discord group](https://discord.gg/wS7Zhpwe8q) to the list of resources on LLM **Interpretability**.
- An intriguing **poster sighting** was reported by `@teknium neilbert.,` however, insufficient details were provided about it.

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (4 messages): 
        
- **LLM Interpretability Resources**: `@mayhem1349` has shared a [Github repository](https://github.com/JShollaj/awesome-llm-interpretability) containing a curated list of open-source tools, papers, articles, and groups related to Large Language Model (LLM) Interpretability. 
- **Additional Group for LLM Interpretability**: `@burnydelic` suggested adding the [Mech Interp Discord group](https://discord.gg/wS7Zhpwe8q) to the list of resources on LLM Interpretability.


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1133673143064596644/) (1 messages): 
        
- **Poster Sighting**: User `@teknium neilbert.` shared an experience of spotting an intriguing poster, suspecting the creator to be an author of the content presented on it. No additional details were provided regarding the content of the poster or the potential UW connection postulated by the user.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Custom MoE models**: `@jp1` discussed a **custom 2bit quant with 4 experts** that has *consistent output up to 500 tokens*. They offered a link to the **experimental quants of 4 expert MoE mixtrals in various GGUF formats** on Hugging Face [here](https://huggingface.co/nisten/quad-mixtrals-gguf).
- **4 Expert MoE Mixtrals**: The goal, according to `@jp1`, is to create the **best-performing MoE below 10GB**. Also, they shared **experimental q8 and q4 files** available for training and finetuning, specifying that *"No sparsity tricks yet"* have been used.
- **Installation of Llama.cpp**: `@jp1` provided a brief guide to download and run llama.cpp from Github, concluding that their 8.4GB **custom 2bit quant** works okay until 512 token lengths, after which it starts looping.
        
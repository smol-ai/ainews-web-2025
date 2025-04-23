---
id: 80165381-8741-4085-9b82-3e1ec521d094
title: '[AINews] 12/26/2023: not much happened today'
date: '2023-12-29T10:07:18.273087Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-12262023-not-much-happened-today
---

<!-- buttondown-editor-mode: plaintext -->Boxing day was pretty quiet, but lots of experimentation and discussion continues.

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Extensive discussions on **LM Studio**, its versions, and associated issues including performance, installation problems on macOS, loading issues, etc. Mentioned that future enhancements like Exllama2 support and multimodality with the Llava model are anticipated.
- Users voiced their view on handling **inefficient help requests** frequently encountered in the server, with the blog post 'Don't Ask to Ask, Just Ask' being shared as a relevant reference.
- Discussed **utilizing the GPU for offloading** in LM Studio along with optimal settings for various models, and difficulties faced while changing the cache folder and launching the server in LM Studio.
- The **performance rating, choice of experts in MoE models and model conversion compatibility** with LM Studio were topics seen in the models-discussion-chat. Roleplay AI and issues with ChatML preset in LM Studio also got a substantial amount of attention.
- Users inquired about and discussed **vRAM capacity and utilization**, in the context of hardware configurations and LLM deployment.
- Technical difficulties around **ChromaDB Plugin** and a comical discussion on programming as a hobby captured users' interests.
- Conversations around **configuring hardware for LLM inference** and comparing performance between server and desktop setups took place.
- Features and concerns for new **beta releases**, including custom settings saving, cache file size, and visual LLM inquiries were mentioned.
- Questions posed regarding **saving the model state in LLM** with Autogen on LM Studio, and the inquiry's eventual resolution, were the highlight of the autogen channel.

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (162 messages🔥🔥): 
        
- **LM Studio and its Versions**: Users `@leefnie`, `@kujila`, `@heyitsyorkie`, `@ptable`, `@blizaine`, `@thelefthandofurza`, `@fabguy`, `@rugg0064`, `@yagilb` and others discussed various aspects of LM Studio, specifically its performance, closed-source nature, future enhancements such as Exllama2 support, comparision with other chat bots and multimodality with the Llava model.
  - Especially `@kujila` compared LM Studio's usability and performance to textgen webui and highlighted it's potential for commercial success.
  - `@heyitsyorkie` clarified that exllama2 can only be run on GPU and there has been continuous request for its support on LM Studio.
- **How to Handle Inefficient Help Requests**: `@fabguy`, `@ptable` and `@heyitsyorkie` had an in-depth discussion about inefficient requests for help and the problems it causes for both help seekers and those offering help. A blog post was shared titled 'Don't Ask to Ask, Just Ask' (`https://dontasktoask.com`) to justify their points.
- **Issues with Installing LM Studio on macOS**: User `@.mjune` encountered installation issues with LM Studio on their MacBook Pro 14 inch running macOS 14.2. `@yagilb` suggested updating the LM Studio version and `@heyitsyorkie` further mentioned Ubuntu 20.04 has known glibc issues.
- **Concerns about LLMs and Model Performance**: Issues and questions were asked about using and understanding different models and quantizations as well as their performance. Contributions came from users `@ptable`, `@rugg0064`, `@heyitsyorkie`, `@kujila`, `@dagbs`, `@number_zero_` among others.
- **Music Generation Models**: `@american_pride` and `@fabguy` briefly discussed music generation models, with AudioCraft from Meta and MusicLM from Google being mentioned.


### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (96 messages🔥🔥): 
        
- **Automatically launching the server in LM Studio**: `@y3llw` asked if it's possible to automate the server launch when LM Studio starts. `@heyitsyorkie` responded saying currently the server needs to be launched manually after the app starts.
- **Issues loading models in LM Studio**: `@giopoly` reported issues with loading any model in LM Studio. `@thelefthandofurza` suggested checking RAM capacity, trying model loading with GPU offload both activated and deactivated, and potentially reinstalling the software.
- **Changing the cache folder in LM Studio**: `@niborium` encountered issues related to a cache folder name containing special characters. `@heyitsyorkie` suggested changing the cache folder path to a directory that doesn't include special characters. The process for changing cache path was further clarified: the option resides under the gear icon next to chat.
- **GPU Offloading in LM Studio**: `@xenorhon` queried about the option to select a specific GPU for offloading within LM Studio. `@doderlein` pointed out that the current GPU can be displayed using the 'System' tab in 'Preferences'.
- **Customizing Preset Settings for Models**: `@pdg` sought advice on optimal settings for various models. Specifically, they inquired about how to adjust settings such as input prefix, suffix, and antiprompt within the JSON file for model presets. User `@yagilb` provided assistance by sharing a template JSON file for the OpenChat model.


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (27 messages🔥): 
        
- **Rating of Chatbot Performance**: `@itzjustsamu` rated the bot's performance as `5/10`, stating it had a "Low token count and didn't respond to my normal response".
- **Discussion about MoE Models**: A question was raised by `@kujila` about the choice of the number of experts in MoE models. `@fabguy` recommended reading the paper and blog from Mistral for a deeper understanding.
- **Model Conversion Compatibility with LMStudio**: `@xkm` asked if converting a model to GGUF would be compatible with LMStudio, specificallly referencing the model at [this link](https://huggingface.co/shi-labs/vcoder_ds_llava-v1.5-13b/).
- **Seeking Roleplay AI Recommendations**: `@american_pride` asked for recommendations for a Roleplay AI and further explained character cards.
- **Issues with ChatML Preset in LM Studio**: `@dagbs` experienced issues with creating their own preset in LM Studio and received help from `@heyitsyorkie` and `@yagilb`, who proposed RoPE values might be the cause. Despite numerous troubleshooting attempts, the issue persisted.


### ▷ #[🛠-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/) (9 messages🔥): 
        
- **GPU Offload with LLM**: User `@_kuva_` inquired about using GPU offload with an LLM and the necessary layers for a 30GB model with an RTX 4070. `@fabguy` suggested starting with **10 layers** and gradually increasing until 90% of the dedicated vRAM is utilized. 
- **Checking vRAM Utilization**: To track vRAM usage, `@ptable` and `@fabguy` advised `@_kuva_` to use the **performance tab in the task manager** on Windows. `@fabguy` further clarified that the dedicated GPU memory (_'Dedizierter GPU-Speicher'_) in the task manager signifies vRAM. 
- **vRAM Capacity**: `@fabguy` indicated that `@_kuva_` is currently using *3GB of vRAM* but could use up to *11GB*. It should be noted that this was mentioned in a non-LM studio context.


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (14 messages🔥): 
        
- **ChromaDB Plugin Issue**: User `@eimiieee` encountered an error while trying to run `@vic49`'s plugin. The error was related to 'NoneType' object in `pynvml`. `@vic49` acknowledged the issue and asked `@eimiieee` to submit details on GitHub, but agreed to get the information over Discord DM when `@eimiieee` mentioned they do not have a GitHub account.
- **Suggestion for ChromaDB Version**: `@heliosprime_3194` offers advice to `@vic49` based on his own experience running an older version of Chroma without issues. He proposed posting the full working code to GitHub and suggested that simply updating to version 0.4.6 might solve the plugin's error.
- **OpenAI Chat Reference**: `@vic49` posted a link to an OpenAI chat, [here](https://chat.openai.com/share/a4e03cdc-1c88-436f-ab61-522244630893), without any specific context.
- **Programming Hobby Comment**: `@vic49` humorously responded to a comment from `@iamtherealbeef`, stating that they thought "programming was going to be a fun and easy hobby!"


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (64 messages🔥🔥): 
        
- **Performance of Different Models**: `@totallybored` discussed their experience testing different models, noting that some like Hermes 2.5 offer better performance after adding specific code instruction examples and that Mistral cannot be extended beyond 8K without continued pretraining.
- **Hardware and AI Services**: `@totallybored` suggested that an ideal scenario would be to rent an AI server which would provide the necessary hardware and a "private" environment, however, they expressed concerns over the privacy and safety of cheap options.
- **GPU Use for Intel Iris Xe GPUs**: `@kuzetsa` queried if anyone had managed to get the GPU working for Intel Iris Xe GPUs, and detailed their successful experience with enabling it on Linux.
- **Building a System for LLM Inference**: In a discussion about building systems for LLM inference, `@heyitsyorkie` recommended prioritizing GPUs, especially VRAM. Other users (`@rugg0064` and `@pefortin`) discussed options for cheaper solutions and considering mining frames and risers as potential hardware setups. `@pefortin` also mentioned that they will test the impact of risers vs added VRAM.
- **Performance on Poweredge Server vs. Desktop**: `@_nahfam_` asked if anyone tested the performance difference between a Poweredge server compared to a desktop with equal amounts of RAM and VRAM.
- **RAM Frequency**: `@dagbs` questioned the importance of frequency when purchasing more RAM for high-end cards. `@xenorhon` advised on specific FLCK, MCLK/UCLK, CL, RCD and RP, RAS and RC settings for Ryzen 7000, warning against trying to push the settings further.


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (10 messages🔥): 
        
- **Custom Settings Saving**: User `@jeunjetta` appreciated the feature to save custom settings to json files and asked if there's a designated spot for adding minor suggestions. `@yagilb` responded positively, directing the user to channel `#1128339362015346749`.
- **Mixtral Compatibility**: `@doderlein` confirmed the compatibility of mixtral with Linux using **LM_Studio-0.2.10-beta-v2**.
- **Cache File Size Issue**: `@pastatfiftylikesmoke` expressed a concern about the large size of `.session_cache/sessionid-xxx.gxxx` files and asked for a faster way of clearing them. `@yagilb` responded by providing two solutions: right-clicking on chat and selecting clear cache; or disabling the cache system altogether by clicking on the gear icon next to Chats and turning cache off.
- **Vision LLM Inquiry**: `@pastatfiftylikesmoke` asked how to work with vision LLM models. `@yagilb` provided a link to a discord channel that contains a section related to vision models: [Link to vision model](https://discord.com/channels/1110598183144399058/1111797717639901324/1187839146094506096).


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (4 messages): 
        
- **Saving Model State in LLM with Autogen**: `@nemesisthedead` inquired about the possibility of saving the model state in Language Model (LLM) training with Autogen on LM Studio. No explicit solution was provided, however, `@nemesisthedead` eventually reported figuring out the process.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Interactions regarding the timeline of information for **Mistral-medium**, new offerings of H100s by Paperspace and reflections on the usefulness of perplexity in AI research.
- Emergence of language setting issue with OpenAI's communication, clarification on the current state of GPT-5, performance evaluation of Turbo GPT for coding, lack of facilities for phone number change in OpenAI and the discourse on $5 holding charge when updating API payment method.
- Several inquiries surrounding the creation of an AI character, the issues linked with plugins and shared links in ChatGPT, inference speed of the fine-tuned GPT-3.5 Turbo, using ChatGPT 3.5 for compiling and formatting reports, uploading images to ChatGPT on PC, refresh time of GPT's usage limit, and text summarization with OpenAI.
- Introduction of a Pythagoras chatbot, inquiries and discussions on integrating Trello REST API with GPT, speculations over the future of GPT series, examination of the 'Knowledge' feature in GPTs, a project on Chickasaw language assistant, discourse on issues encountered with ChatGPT, discussion on the usage of Discord Bot for image generation, and conversation on the limits on instruction, knowledge and action tools in GPTs.
- Assistance sought for crafting role-playing prompts in psychology and sociology, proposals for analyzing behavior using GPT, launch of a project targeting academic and research-oriented transformation of common language using GPT and a solution to an unstated problem with ChatGPT.
- Dialogue on creating detailed and insightful prompts for psychology and sociology, potential conflict with privacy guidelines in behavior analysis using GPT, potential usefulness of Directive GPT for prompt optimization, introduction of an academic prompt generator GPT for research purposes, and a solution to address knowledge base problems in ChatGPT.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (3 messages): 
        
- **Model Knowledge Timeline**: User `@eljajasoriginal` asked about the timeline of information for **Mistral-medium**, specifically wanting to know which year up to which the model has data.
- **Paperspace And H100s**: `@kyoei` noted that Paperspace started offering H100s for use, implying a potential competition for OpenAI in 2024.
- **Comment on Perplexity**: `@caledordragontamer chief_executive` shared insights on the usefulness of perplexity in research, stating its efficiency in terms of speed.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (81 messages🔥🔥): 
        
- **Language Setting Issue**: `@alextheemperorpenguin` is receiving emails in French from OpenAI despite having the account set to English. They had a discussion with `@rjkmelb` about reaching out to OpenAI via the chatbot on the help page to raise a support ticket for this issue.
- **GPT-5 Discussion**: `@mazejk` was initially under the impression that GPT-5 has been released, but was corrected by `@satanhashtag` clarifying it's actually a custom chatbot named "gpt-5".
- **GPT Coding Performance**: `@pandora_box_open` and `@thunder9289` engaged in a discussion about whether the Turbo version of GPT is worth purchasing for coding purposes. The consensus was that it is not significantly better than GPT-4 in this regard.
- **Phone Number Change Feture**: `@jamiecropley` and `@infidelis` discussed about OpenAI not allowing users to update their phone numbers on their accounts. `@jamiecropley` is facing issues with this as they frequently change phone numbers. 
- **OpenAI Charges**: `@jamiecropley` asked about a $5 holding charge when updating his API payment method. `@aminelg` responded that they never experienced such a charge. `@jamiecropley` later found more information about this on OpenAI's help page: [here](https://help.openai.com/en/articles/7438062-after-updating-my-api-payment-method-i-noticed-a-5-charge-is-this-a-temporary-hold).


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (51 messages🔥): 
        
- **Creating AI Character**: User `@load2` asked about how to create an AI character. However, no concrete responses were provided.
- **Plugins in ChatGPT**: `@rooster_25528` was having trouble with logging in to plugins in ChatGPT. User `@elektronisade` informed that the plugin system is technically going away and many plugins are abandoned or work poorly.
- **Checking Shared Links in ChatGPT App**: User `@beefcheong` wanted to know how to check shared links in the ChatGPT app, and `@lugui` responded that there currently is no way to do that.
- **Training GPT-3.5 Turbo Speed and Usage**: `@nyla3206` asked about the inference speed for the fine-tuned version of GPT-3.5 Turbo. However, there was no response to this query.
- **Using ChatGPT for Work**: `@transatlantictex` inquired about using ChatGPT 3.5 for compiling and formatting reports at work. They received feedback from user `@yomo42` about using GPT-3.5's ability to handle CSV formats, and the potential of using GPT-4 or a Custom GPT. User `@yomo42` also emphasized on the importance of considering the appropriateness and legality of using AI in the workplace.
- **Uploading Images to ChatGPT**: User `@euroeuroeuro` was struggling with uploading images to ChatGPT on PC, whereas the android app seemed to work fine. No solution was provided in the messages.
- **Usage Limit and Refresh Time of GPT**: `@yomo42` wanted to know about the refresh time of GPT's usage limit, and `@satanhashtag` clarified that it's 1 prompt for every 4.5 minutes meaning 40 prompts in 3 hours.
- **Text Summarisation With OpenAI**: `@_s.n.o.o.p.y_` asked about the best OpenAI model for summarising text. No responses were provided for this query.
- **ChatGPT Plus Account Upgrade for More Dall-e Usage**: `@.lb3373` inquired about upgrade options for more Dall-e usage on ChatGPT Plus account. User `@aminelg` clarified that the current limit applies to all and for more usage, DALL-E API has to be used.
- **Effiliation with "worldbrains foundation"**: User `@zingzongy` queried about any affiliation between OpenAI and a non-profit organization called "worldbrains foundation". This did not receive a response in the channel.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (23 messages🔥): 
        
- **Pythagoras Chatbot**: `@arnoldtri` suggested a Pythagoras chatbot and shared a link to it: [Chatbot Pythagoras](https://chat.openai.com/g/g-gNH4K4Egg-shownotes).
- **Implementing Trello REST API into a GPT**: `@fyruz` asked for help regarding the confusion they are experiencing while trying to implement Trello REST API into their GPT.
- **Discussion on Future of GPT**: `@phil4246` started a discussion asking the community's opinion on what will come next after **GPT-4**, such as **GPT-4.5**, **GPT-5** or something else.
- **'Knowledge' feature in GPT**: `@_odaenathus` asked for clarification on how the 'Knowledge' feature in GPTs works and how it is implemented. `@solbus` clarified that **knowledge files are like reference documents for the GPT** that don't modify the GPT's base knowledge.
- **Chickasaw Language Assistant**: `@machinemerge` shared about his project of creating a Chickasaw language assistant and asked for specific prompts for better learning. `@Rock` suggested methods for him to improve the performance of his GPT.
- **Issue with ChatGPT**: `@bonjovi6027` reported an issue with chatGPT, receiving **network errors** on both Google Chrome and Safari. `@scargia` queried about the conversation length. 
- **Using Discord Bot**: `@imaad22` asked about how to use the discord bot for image generation. `@elektronisade` clarified that there's no such provision.
- **Limits on Instruction, Knowledge and Action Tools**: `@machinemerge` and `@Rock` further discussed the **limit of 20 knowledge files with 512 MB each**. The instructions tool has **an 8000 character limit** and `@loschess` added that there can be **10 Knowledge files** active at any given time with a **limit of 2M tokens each**. The GPTs can only process so much since each output is limited to 4k tokens.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (8 messages🔥): 
        
- **Role-playing Prompts for Psychology and Sociology**: User `@thebookoforlando777115` sought help in crafting prompts to make ChatGPT sound like an expert in psychology and sociology. `@bambooshoots` suggested a directive which emphasizes on in-depth analyses, theoretical frameworks, and key concepts. They also hinted at further tuning using Directive GPT.
- **Asking ChatGPT to Analyze Behavior**: `@thebookoforlando777115` asked for ways to use such a role-playing prompt for scenarios where someone's reaction or truthfulness is to be analyzed. `@bambooshoots` advised trying with real-life questions, while `@eskanta` pointed out potential issues with privacy guidelines when analyzing other's behavior.
- **Alternative Approach to Analyzing Behavior**: `@eskanta` recommended asking the model to review a situation from multiple viewpoints and following up on the most insightful responses.
- **GPT for Academic and Research-Oriented Tasks**: `@sacmaj` shared about an ongoing project aimed at transforming common language into academic terminology for research purposes. They offered a [link](https://discord.com/channels/974519864045756446/1188346798314618980/1188346798314618980) to try out a prompt crafting tool for academic tasks.
- **Fix for a ChatGPT Issue**: `.@shaw93` mentioned addressing a problem with ChatGPT and the solution was shared on their [Twitter account](https://twitter.com/ShawOnTech1).


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (8 messages🔥): 
        
- **Optimizing Prompts for Psychology and Sociology**: User `@thebookoforlando777115` was seeking advice on creating detailed and insightful prompts for psychology and sociology. `@bambooshoots` provided a detailed role-play prompt that aimed to generate in-depth analysis from a psychology and sociology expert standpoint.

- **Realistic Scenario Analysis**: `@thebookoforlando777115` also expressed a desire for the AI to analyze realistic scenarios and determine if a person is lying in a given situation. However, `@eskcanta` reminded that such requests might conflict with usage guidelines and the model's training to respect privacy.

- **Directive GPT**: `@bambooshoots` suggested that Directive GPT might be useful in further optimizing the responses if the given prompts don't produce the desired results.

- **GPT for Research-Oriented Tasks**: `@sacmaj` shared an academic prompt generator GPT designed for research-oriented tasks which transforms regular language into something resembling a master's level thesis, providing structured stages for creating prompts. They also provided a [link](https://discord.com/channels/974519864045756446/1188346798314618980/1188346798314618980) to try it out.

- **Knowledge Base Solution**: `.shaw93` shared a solution to an unknown problem on their Twitter `@ShawOnTech1`, describing that they fixed the issue by explicitly directing the knowledge base to 'use this section only if x'.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Discussed AI Research**: `@teknium` had shared a [Twitter post](https://fxtwitter.com/francis_yao_/status/1739688394148733264?s=46) on AI research; `@giftedgummybee` deemed it **"very good."**

- **Collaboration Opportunities, Contribution Value, Research Possibility and Current Focus**: Topics like potential tasks on the axolotl project, the significance of documentation improvement, unmet expectations in LLM research and concerns about the unsustainable nature of unpaid contribution were discussed. 
      
- **AI Developments and User Interaction**: Users shared and discussed a [YouTube video](https://youtu.be/SozBO7eCvaM?feature=shared) on **RAG, RALM,** and **Vector Stores**. User `@asada.shinon` identified this development as **S-LoRA**.

- **Nous-Hermes 2 Yi 34B GGUF Quantized Version Release**: An announcement for a [new model](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF) release was made by `@teknium`.

- **Performance Comparison, Code Contamination, Reddit User Experiences, NeurIPS 2023 Access, and LM Studio Compatibility**: Discussions were held comparing **Hermes 2.5** and **Mixtral** performances, addressing code contamination in models, sharing poor experiences with Reddit users, exploring ways to gain access to NeurIPS 2023 talks, and confirming Nous Research model's compatibility with LM Studio.

- **Local LLM Training, Model Selection, Model Conversions, and Dataset Formation**: Queries related to LLM training, recommendations for suitable model selection, issues on running Nous Hermes 2 - Yi-34B in **fp16**, and inquiries on building and cleaning datasets were responded to.

- **Transformers & Pipelines Usage, Task Division, and Iteration Averaging**: An ask for examples of using transformers and pipelines and a proposal on task division and iteration averaging were made.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (2 messages): 
        
- On the topic of **discussed AI research**, `@teknium` shared a [Twitter post by Francis Yao](https://fxtwitter.com/francis_yao_/status/1739688394148733264?s=46), which `@giftedgummybee` acknowledged by stating that it was **"very good"**.


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (19 messages🔥): 
        
- **Potential Collaboration**: `@pradeep1148` inquired about tasks that involve writing code and was directed by `@teknium` to work on the axolotl project with @525830737627185170, @257999024458563585, and @208256080092856321.
- **Value of Contribution**: In a discussion about contributions, `@hamelh` stressed that even small steps like improving documentation are valuable and encouraged `@pradeep1148` to start from there.
- **Research Opportunities**: When `@erichallahan` expressed a disappointment from prior experiences in LLM research, `@hamelh` suggested that there are still valuable research opportunities to be had. 
- **Focus Shift**: Despite the encouragement, `@erichallahan` clarified that unless a compelling offer is made, they have moved their focus to other important endeavors. They voiced disappointment about the unsustainable nature of unpaid work and a lack of respect despite contributions.
- **Upcoming Paper**: In a different vein, `@hamelh` shared with `@ldj` that they are soon releasing a paper on the amplify-instruct method used to build the Capybara dataset.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (8 messages🔥): 
        
- **Discussion on AI Breakthroughs**: User `@lightningralf` shared a [YouTube video](https://youtu.be/SozBO7eCvaM?feature=shared) introducing a new AI development, described as a breakthrough. The video seems to discuss advanced concepts like **Retrieval Augmented Generation (RAG)**, **Retrieval Augmented Language Models (RALM)**, and **Vector Stores**. 
- **Identification of AI Concept**: User `@asada.shinon` recognized this "breakthrough" as **S-LoRA**.
- **Query about Video Origin**: User `@rabiat` questioned if `@lightningralf` was the creator of the shared video. `@lightningralf` responded, explaining they are not an AI researcher but are interested in AI topics.
- **User Interaction**: User `@fullstack6209` expressed satisfaction with the shared video.
- **Additional Link Shared**: User `@metaldragon01` shared a [Twitter post](https://fxtwitter.com/NickADobos/status/1739736441175900203) without providing context. The content and relevance of the tweet is not clear from the provided chat history.


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (1 messages): 
        
- **Nous Hermes 2 Yi 34B GGUF Quantized Version Availability**: In an announcement made by `@teknium`, it has been declared that Nous-Hermes 2 Yi 34B has been GGUF quantized in different typical sizes TheBloke normally would have. The model is available on the **HuggingFace** org at this [link](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF). A reference image was also shared marketing the release.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (74 messages🔥🔥): 
        
- **Hermes 2.5 vs Mixtral Performance**: `@teknium` and other users engaged in a discussion comparing the performances of **Hermes 2.5** and **Mixtral**. One user, `@fullstack6209`, shared his user experience that highlighted Mixtral's superior performance in certain benchmark testing. Both models were noted to originate from the Nous Research team, with a user (`@vic49.`) mentioning the latest release, **Nous Hermes 2 - Yi-34B - GGUF Quantized Version**. The link: [Nous Hermes 2 - Yi-34B - GGUF Quantized Version](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF) was posted in relation to this conversation. 

- **Code Contamination in Models**: A lively conversation was held concerning the issue of code contamination in AI models. Users `@.beowulfbr` and `@nonameusr` led the discussion, mentioning the difficulty of benchmarking models and problems with "contaminated" base models like **yi-34b**.

- **Experience with Reddit Users**: The chat between `@.beowulfbr` and other users like `@weyaxi`, `@ldj`, and `.benxh` suggested dissatisfaction with the Reddit user community, citing negative experiences they had.

- **NeurIPS 2023**: User `@lejooon` asked about ways to access NeurIPS 2023 talks without being in attendance. Responses from `@itali4no` and `@gabriel_syme` suggested that online tickets would need to be purchased or one could wait for potential free access after a few weeks.

- **LM Studio Compatibility**: User `@vic49.` inquired about the compatibility of models produced by Nous Research with LM Studio. Specific reference was made to **Nous Hermes 2 - Yi-34B - GGUF Quantized Version**. `@teknium` confirmed that they are indeed compatible.


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (21 messages🔥): 
        
- **Usage of <0x0A> in the LLM**: `@teknium` pointed out that `<0x0A>` is used to denote a newline in a model script. This was further validated by `@everyoneisgross`, though they also noted a complication with their mixtral model, which seemed to convert the tag into string and alter their text. This suggested a possible bug in the script.
- **Model Selection for Local LLM Training**: `@leuyann` sought recommendations for models suited for local and non-local fine-tuning and testing with reference to reasoning enhancements in large language models (LLMs). `@eas2535` clarified that while CUDA was not needed for the 4-bit quantized version of Mixtral, it did require substantial RAM, which made it unsuitable for systems with 16GB memory. `@night_w0lf` recommended starting with a 3B base model such as StableLM before taking on larger models like a 7B Mistral.
- **Converting Models to fp16**: `@mgoin` raised the issue of running Nous Hermes 2 - [Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) in fp16, noting that the model produced NaNs when run in that precision. They noted the lack of support for bfloat16 in PyTorch's fake_quantize. `@teknium` advised simply changing the torch.dtype to float16.
- **Data Cleaning and Dataset Formation**: `@.beowulfbr` sought suggestions in building and cleaning datasets. `@teknium` recommended using Lilac, an open-source data improvement tool by @tomic_ivan. `@spencerbot15` provided the corresponding [link](https://docs.lilacml.com/) to the Lilac documentation.


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 messages): 
        
- **Usage of Transformers and Pipelines**: `@vic49` asked `@qnguyen3` for examples of how to use transformers, pipelines, and other similar tools.
- **Task Splitting and Iteration Averaging**: `@tranhoangnguyen03` proposed to `@.benxh` the idea of dividing the task and taking the average from multiple iterations for the count task.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Active engagement on how to deal with cache files within **HuggingFace datasets**, as suggested by `@vipitis`, involves deleting everything in `~\.cache\huggingface\datasets` and upgrading datasets to version 2.16.0.
- Extensive discussion about **model fine-tuning**, ranging from applications on Windows, fine-tuning with the Whisper model using FSDP, to understanding the workings of OpenAI embeddings. User `@deadsg` shared a useful [Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing) for users with limited GPU capacity.
- The unwritten rules of community engagement surfaced as a topic of discussion among users like `@liyucheng09`, `@gez_gin`, `@vipitis`, and `@mr.kiter`, touching upon the response time and challenges faced in the feedback loop within the community.
- Notable development and testing projects introduced included **Texify** by `@tonic_1` which transcribes images into latex formulas and the online testing platform for **Dolphin 2.6** shared by `@jiha` [here](https://replicate.com/kcaverly/dolphin-2.6-mixtral-8x7b-gguf). `@Deadsg` shared their **Bat-.-LLama-.-CPP project** on [GitHub](https://github.com/Deadsg/Bat-.-LLama-.-CPP) and its associated [Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing). 
- Inquiries into **Retrieved Augmented Generation (RAG) Vs Fine-tuning** for knowledge impartation in models, with `@harsh_xx_tec_87517` advocating for RAG to counter hallucinations in fine-tuning. `@harsh_xx_tec_87517` also shared a [tutorial](https://www.youtube.com/watch?v=74NSDMvYZ9Y&t=193s) on how to update information in LLMs.
- Educational content shared by users such as `@onceabeginner`'s exploration of **mamba architecture** and `@merve3234`'s learning journey into **OneFormer**, describing it as a powerful segmentation model and sharing their [notes](https://x.com/mervenoyann/status/1739707076501221608?s=20) regarding the same.
- Updates of tools and additions to the community, as `@appstormer_25583` revealed about enhancements made to [Appstorm.ai](https://beta.appstorm.ai/), including GPT capabilities, LiteLLM integration, and bug fixes. `@gz888` shared a [tool](https://huggingface.co/thibaud/sdxl_dpo_turbo) in the cool-finds channel.
- Exploration into language models, with `@opencuiguy` suggesting **NLI models for entailment** and indicating that **seq2seq models are prone to hallucination**, while `@lucastononro` combines **Large Language Models (LLMs)** with 'Interface Omniscience' for web interactions in his [LinkedIn post](https://www.linkedin.com/posts/activity-7139769316337504256-dDoT?utm_source=share&utm_medium=member_desktop) and shared corresponding [GitHub code](https://github.com/lucastononro/llm-food-delivery).

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (90 messages🔥🔥): 
        
- **HF Datasets Cache Cleanup**: User `@liyucheng09` inquired about cleaning cache files for HuggingFace datasets. They had updated a dataset on the HF Hub but couldn't see the updates when loading it locally. `@vipitis` suggested deleting everything in `~\.cache\huggingface\datasets` and upgrading datasets to version 2.16.0 which reportedly includes some fixes for cached stuff.

- **Model Fine Tuning Discussions**: Discussion regarding model finetuning occurred between various users. `@omarabb315` asked if Whisper can be fine-tuned with FSDP as DeepSpeed doesn't work on Windows. `@deadsg` shared a [useful Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing) they created for those who do not have the GPU capacity for finetuning. 

- **Community Response and Etiquette**: Several users including `@liyucheng09`, `@gez_gin`, `@vipitis`, and `@mr.kiter` engaged in a discussion about the community engagement. Opinions touched upon the amount of time it takes to receive help or feedback, the influence of the holiday season on response times, and the challenges faced by those providing answers.

- **Embedding Vectors Question**: User `@somefuckingweeb` asked why OpenAI embeddings are a single vector and if they use summations or means across the dimension for cosine similarity. This question was forwarded to two other users `@251101219542532097` and `@697163495170375891` by `@_aabidk`.

- **Testing Dolphin 2.6 online**: `@jiha` asked how they can test Dolphin 2.6 online since they didn't have enough memory to run Mixtral 8x7b. They also shared a [link](https://replicate.com/kcaverly/dolphin-2.6-mixtral-8x7b-gguf) to a free online playground for Dolphin 2.6 Mixtral.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **Mamba Architecture**: `@onceabeginner` shared that they learned about the **mamba architecture**. However, no further details were given to shed light on the topic.
- **OneFormer - Universal Segmentation Model**: `@merve3234` shared their learning experience with **OneFormer**. They referred to it as a "very powerful universal segmentation model" and shared a [link](https://x.com/mervenoyann/status/1739707076501221608?s=20) to their notes on the topic.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (1 messages): 
        
gz888: https://huggingface.co/thibaud/sdxl_dpo_turbo


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 messages): 
        
- **Texify Project**: `@tonic_1` announced a [demo](https://huggingface.co/spaces/Tonic1/texify/settings) they built, which takes pictures and returns latex formulas. It's currently available on a community grant for others to build upon.
- **Bat-.-LLama-.-CPP Project**: `@deadsg` shared the [GitHub repository](https://github.com/Deadsg/Bat-.-LLama-.-CPP) of their project.
- **Google Colab for Bat-.-LLama-.-CPP Project**: `@deadsg` also provided a [Google Colab link](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing) for those who lack the GPU for finetuning their Bat-.-LLama-.-CPP project. They advised to set paths and git clone the repo into the Colab.
- **Update on Appstorm.ai**: `@appstormer_25583` shared recent patch updates to [Appstorm.ai](https://beta.appstorm.ai/), including enhancements to GPT capabilities, bug fixes, LiteLLM integration, and more.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **Retrieved Augmented Generation vs Fine-Tuning**: User `@harsh_xx_tec_87517` indicated that the best way to impart knowledge into a specific model is by using **Retrieved Augmented Generation (RAG)** rather than fine-tuning, as the latter often leads to hallucinations. User `@bennoo_` expressed interest in fine-tuning using a small dataset to update knowledge in a Language Model, questioning if a **RAG system and its infrastructure** was necessary. 
- **Tutorial on Fine-Tuning LLM**: `@harsh_xx_tec_87517` also shared a YouTube [tutorial](https://www.youtube.com/watch?v=74NSDMvYZ9Y&t=193s) on how to **update information in LLMs**.
- **Use of NLI models**: User `@opencuiguy` suggested using **NLI models for entailment** in response to `@merve3234`'s question. He mentioned these models are encoder only and can be found under text-classification on the HuggingFace website.
- **Project - LLMs with 'Interface Omniscience'**: `@lucastononro` shared his project, combining **Large Language Models (LLMs)** with 'Interface Omniscience' to simplify web interactions like food ordering. He has utilised RAG for intelligent information retrieval in the chatbot. The [LinkedIn post](https://www.linkedin.com/posts/activity-7139769316337504256-dDoT?utm_source=share&utm_medium=member_desktop) and [GitHub code](https://github.com/lucastononro/llm-food-delivery) were shared.
- **Discussion about seq2seq models**: In a discussion with `@merve3234`, `@opencuiguy` mentioned that **seq2seq models are prone to hallucination** and suggested fine-tuning encoder-only models for better performance with less compute and memory requirements.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **LookAhead Decoding Strategy**: Mentioned by user `@rage31415`, LookAhead can speed up inference for Language-Level Models (LLM) in AI applications.
- **Mistral API Waitlist and Pricing**: Discussions on the waiting time and cost of the Mistral API, with user `@lee0099` sharing recent acceptance from the waiting list and the cost of less than $0.50 for 1 million tokens for smaller models. Discrepancies in token count between Mistral and OpenAI models were highlighted by `@sublimatorniq`.
- **Text-to-speech and Text-to-music Models**: User `@qwerty_qwer` asked about training these types of models in context of potentially having access to a rich dataset including 1 million songs from Spotify and around 20,000 hours of audio files with captions.
- **Veracity and Argumentative Responses in Models**: Issues with accuracy and consistency in Mistral 7B v0.2 and Mixtral v0.1 models were pointed out by `@.skyair` drawing a contrast between these, Claude and GPT-3.5 models.
- **Mistral and Mixtral Release Source Code Availability**: User `@pilot282` inquired about the release status and source code's availability of these models, which `@lee0099` clarified that the source code for Mixtral small has been released but not for Mixtral medium.
- **Open Source Options for LLM of MoE**: `@brycego` asked about open-source solutions for training LLM of MoE, and was referred to the [Mixtral + vast.ai](https://github.com/mistralai/mistral-src/blob/main/mistral/moe.py) by `@poltronsuperstar`.
- **Model Responses Consistency**: Concerns about inconsistent responses from Mixtral 8x7b raised by `@mka79`, with suggestions offered for adjustments to the parameters like temperature, top_p, and fixing the random seed.
- **Chatbot Model Comparisons**: A discussion about the consistency of different chatbot models, presenting views about Google's Gemini and Huggingface Chat models.
- **Speed Optimizations**: Queries about how to increase the speed of the mistral instruct v02 7B q4_k_s model, with suggestion of expanding VRAM from `@steroidunicorn`.
- **Fine-Tuned Mistral Model Optimization**: `@builderx` asked suggestions for optimizing a fine-tuned Mistral model deployed on an A100.
- **Building Mistral Image Issue**: `@cheemszero69420` encountered long installation time on the megablocks and stanform-stk libraries during Mistral image building process using Docker.
- **Few Shot Prompting Template**: The existence of a few shot prompting template was asked by `@nioned`.
- **Rate-Limits on Requests**: Announcement from `@lerela` regarding the introduction of requests per second rate-limits for improved service quality.
- **Mistral Medium Public Release**: Confirmation from user `.superintendent` that a public release for Mistral Medium is in the plan, however, the exact release date is yet to be defined.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (25 messages🔥): 
        
- **Inference Speed** for **Language Models**: `@rage31415` brought up the topic of using the **LookAhead decoding strategy** to speed up inference for Language-Level Models (LLM).
   
- **Mistral API Waitlist and Cost**: There were conversations revolving around the waitlist and costs for the **Mistral API**. User `@mka79` inquired about the typical wait time for getting off the API waitlist. `@lee0099` mentioned their recent acceptance from the waitlist, but noted the challenge of affordability. `@lee0099` later shared that the cost is less than $0.50 for 1 million tokens for small models—a detail corroborated by `@sublimatorniq`, who however noted that the token count varies between Mistral and OpenAI models, estimating roughly 4 OpenAI tokens to 5 Mistral tokens. 

- **Training Text-to-Speech or Text-to-Music Models**: User `@qwerty_qwer` asked whether anyone was training a text-to-speech or text-to-music model, stating that they have abundant data sources, including 1 million songs from Spotify and around 20,000 hours of audio files with captions.

- **Veracity and Argumentative Nature of Models**: `@.skyair` voiced concerns over **Mistral 7B v0.2 and Mixtral v0.1** models skewing more argumentative with the frequent contradiction statement, "it is not accurate to say...". The user also pointed out perceived issues with truthfulness, attesting that the models often declare facts as inaccurate while generating inaccurate information. They drew comparisons with Claude and GPT-3.5, which according to them don't have these issues.

- **Release and Source Code of Mistral and Mixtral**: `@pilot282` asked about the release status and source code availability for **Mistral and Mixtral**. In response, `@lee0099` clarified that the source code for Mixtral small has been released, but not for Mixtral medium yet.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (16 messages🔥): 
        
- **Open Source Solutions for Training LLM of MoE**: User `@brycego` asked about open source solutions for training LLM of MoE and `@poltronsuperstar` pointed them towards **[Mixtral + vast.ai](https://github.com/mistralai/mistral-src/blob/main/mistral/moe.py)**, further emphasizing the need for PyTorch expertise.
- **Inconsistencies in Model Responses**: `@mka79` raised a concern about obtaining different answers from **Mixtral 8x7b** for the same prompts. `@sublimatorniq` and `@lerela` suggested adjusting parameters like **temperature**, **top_p**, and fixing the **random seed** to achieve more consistency.
- **Comparison of Chatbot Models**: Different chatbot models were compared for consistency. `@sublimatorniq` found Google's Gemini to be pretty consistent while `@mka79` reported better results with **Huggingface Chat**.
- **Speed Optimization with mistral instruct v02 7B q4_k_s Model**: User `@serhii_kondratiuk` inquired about ways to speed up the response time in the case of mistral instruct v02 7B with q4_k_s model on a 1k token input, running on a GeForce 3080 ti RTX with 16GB VRAM and 32 GB RAM. `@steroidunicorn` suggested that expanding the VRAM to 24GB would enhance the processing speed significantly.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **Optimizing Fine-Tuned Mistral Model**: `@builderx` asked about methods to optimize a fine-tuned **Mistral** model due to its slow performance when running on an **A100**.

- **Issue With Building Mistral Image**: `@cheemszero69420` encountered issues building a **Mistral** image using Docker due to the process getting stuck at the installation step for **megablocks** and **stanform-stk** libraries. The process took significantly longer than expected, prompting for solutions.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **Few Shot Prompting Template**: User `@nioned` asked if there is a **few shot prompting template** to follow.
- **Rate-Limits Update**: `@lerela` notified that they have introduced **requests per second rate-limits** to improve the quality of service on the platform. Users can view these limits on the platform and contact support@mistral.ai if they are affected by this change.
- **Mistral Medium Public Release**: User `.superintendent` confirmed that **Mistral Medium** is scheduled for a public release, although a specific date has not been set.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Discussions on **model training** with a **new tokenizer** integration and its potential compatibility with qlora, led by `@noobmaster29` and `@nruaif`. It was noted that training with some tokens might be possible, but entire language adaptation might not yield sufficient results. 
- Technical questions related to the **size of shards and GPU states**, raised by `@casper_ai` and clarified to be respectively 5 GB and 10 GB. It was suggested that limited space might be causing issues.
- Clarifications on the term `MistralForCausalLM` in relation to an `AttributeError`, with `@caseus_` and `@faldore` confirming that `Mistral-7b` is the correct term.
- Insights on **Alpaca** formatting shared by `@colejhunter` and `@caseus_`, with a confirmation that alpaca follows a particular structure found in the [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L54).
- Approach on choosing a **validation set** when a dataset offers multiple valid responses for a single input, asked by `@unknowngod`.
- Strategies for **deduplicating and decontaminating test sets**, initiated by `@xzuyn`.
- An attempt to **replace a token** in the tokenizer, undertaken by `@faldore`.
- Inquiry into tools for generating **Q&A pairs** for RAG by `@fred.bliss`, who has previously used LlamaIndex for this purpose. Interest in other users' experiences with LlamaIndex and AutoGen for the latest RAG dataset generation was expressed. Discussions revolved around achieving domain-specific recognition using local models, emphasizing that high quality is not the main goal.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (4 messages): 
        
- **Training and Integrating New Tokenizer**: User `@noobmaster29` sought advice on **training a model to work with a new tokenizer** and its possibility of integration with qlora. User `@nruaif` clarified stating that training with a few tokens might be feasible but entire language adaptation would not be sufficient.
- **Fine-tuning with New Tokenizer**: User `@noobmaster29` further raised the question on whether it's better to fine-tune on text completion for a new tokenizer, to which `@nruaif` replied that any format would be acceptable.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (3 messages): 
        
- **Concerns about Available Space with Shards and GPU States**: `@casper_ai` clarified to `@faldore` that shards are 5 GB and GPU states are 10 GB, and raised the question if insufficient space might be the issue.
- **Clarification on AttributeError: MistraForCausalLM**: `@caseus_` queried `@faldore` about the correct term between `Mistral` or `Mixtral` in relation to `AttributeError: MistralForCausalLM`.
- **Confirmation on Using Mistral-7b**: `@faldore` - when asked by `@caseus_`, confirmed that the correct term is `Mistral-7b`.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (18 messages🔥): 
        
- **Alpaca Formatting**: User `@colejhunter` questioned if alpaca followed the standard alpaca formatting, using the structure seen in one of the [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L54). Responding to the inquiry, `@caseus_` confirmed that their implementation is more explicit for the system@prompt. `@colejhunter` further clarified that when running inference after training using `type: alpaca`, the prompt should be structured `### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:\n`.

- **Choosing a Validation Set**: User `@unknowngod` sought advice on the best way to choose a validation set when a dataset contains multiple valid responses for a given input.

- **Deduplicating and Test Set Decontamination**: User `@xzuyn` initiated a discussion asking what strategies people typically use for de-duplicating and test set decontamination.

- **Replacing Token in Tokenizer**: User `@faldore` was trying to replace a token (`</s>`) in the tokenizer with a new token (`


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (4 messages): 
        
- **Querying Q&A Pairs Generation Tools**: At the beginning of his hiatus, `@fred.bliss` inquires about current tools for generating Q&A pairs for RAG.
- **Local Models for Niche Domain**: `@fred.bliss` clarifies his intention of using local models to generate these pairs, emphasizing that the quality isn't the primary concern but the aim is to achieve domain-specific recognition.
- **Using LlamaIndex for RAG Dataset Generation**: `@fred.bliss` shares his past experience of using LlamaIndex as a workaround for this particular task and notes that it is now fully functional for this use case but is heavily optimized for OAI.
- **Seeking Experiences with LlamaIndex and AutoGen**: Finally, `@fred.bliss` asks if anyone has experimented with the latest RAG dataset generators in LlamaIndex and the tool AutoGen. He mentions an earlier conversation from `<@284810978552578050>` that referred to AutoGen.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Announcement by `@hwchase17` about the **creation of a new channel**, `<#1189281217565179915>`, following a suggestion by `<@1059144367731920896>`.
- Extensive discussion on the use and challenges of **LangChain**, with specific queries surrounding its possibilities beyond reading PDFs and querying results. Points included additional key-value data in LLM output from Tool, comparison between **LangChain, LCEL, and agents with tools**, and ImportError when importing numpy.
- Introduction of idea to create a **Specialist Bot** focusing on autogen, **LangChain**, etc. by `@repha0709`, who also shared conflicts between LangChain and the "openai" API when using text-generation-webui.
- Inquiry from `@evolutionstepper` about community practices regarding handling **async and FastAPI**.
- A shared issue by `@cryptossssun` and `@madgic_` regarding **GPT Vision’s limitations** in accurately extracting all data from complex tables in PDFs. `@madgic_` shared a particular prompt to alleviate transcription issues from image to Markdown.
- `@a404.eth` shared his exploration on **LangChain Playground** and how to use inputs in it, the implementation of a "Conversational Retrieval Chain", and shared a link for LangChain Documentation for creating a retrieval chain. Also, notable Python libraries for LangChain were mentioned.
- Updates on [Appstorm.ai](https://beta.appstorm.ai/) were announced by `@appstormer_25583`. The updates included the implementation of **Gradio's folium component**, addition of examples for GPTs, increased robustness of GPTs, multiple bug fixes, availability of LiteLLM integration, improvements to Vision GPTs, and the ability for GPTs to perform Google Search.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **Creation of New Channel**: User `@hwchase17` announced the creation of a new channel `<#1189281217565179915>`, following a suggestion by `<@1059144367731920896>`. The new channel has been created due to the significant amount of work required to ensure reliable operation of their systems.


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (15 messages🔥): 
        
- **LangChain Usage**: User `@infinityexists.` asked if there is any resource available to explore the possibilities of using **LangChain** beyond reading PDFs and querying results from the model.
- **Additional Key-Value Data in LLM output**: `@peeranat_fup` inquired if it is possible to pass additional key-value data in **LLM** output from Tool.
- **ImportError of numpy**: User `@arunraj6451` shared an issue about an ImportError when trying to import numpy from its source directory rather than the python interpreter.
- **Discussion on LCEL vs Agents with Tools**: `@a404.eth` expressed interest in discussing the comparison between **LCEL** and **agents with tools**.
- **Creating a Specialist Bot**: `@repha0709` discussed their idea for creating a bot specializing in **autogen**, **LangChain**, etc., which would generate the autogen configuration as per the user's prompt. They mentioned facing conflicts between LangChain and the "openai" API when using text-generation-webui to run models.
- **Handling Async and FastAPI**: `@evolutionstepper` asked about the community's approaches to handling **async and FastAPI**.
- **GPT Vision Limitation with PDFs**: There was a discussion by `@cryptossssun` and `@madgic_` regarding difficulties with **GPT Vision** extracting all data from complex tables in PDF files, as it often misinterprets or overlooks empty cells. To attempt a solution, `@madgic_` shared a prompt to improve the transcription from image to Markdown format.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (5 messages): 
        
- **Using Inputs in LangChain Playground**: `@a404.eth` asked about how to use inputs in the LangChain playground and provided an example of how to run a simple template using `ChatPromptTemplate` from LangChain.
- **Conversational Retrieval Chain**: `@a404.eth` mentioned that he/she is following a guide for implementing a "Conversational Retrieval Chain" with retrieval-augmented generation and code snippet shared.
- **LangChain Guide Link Shared**: `@a404.eth` shared a link to the guide for creating a retrieval chain on [LangChain Documentation](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain).
- **Important Python Libraries for LangChain**: The LangChain guide recommends installing certain Python libraries including LangChain, OpenAI, faiss-cpu, and tiktoken.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Appstorm.ai Updates**: User `@appstormer_25583` announced several patch updates to [Appstorm](https://beta.appstorm.ai/):
    - GPTs can use Gradio's folium component to render detailed maps.
    - GPTs now have examples.
    - GPTs are showing increased robustness.
    - Multiple bug fixes have been implemented.
    - LiteLLM integration is now available.
    - Fixes have been introduced to the Vision GPTs.
    - GPTs can now perform Google Search.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Query about the best function-calling model on Hugging Face, ideally a variant of **Llama-7B** by User `@harsh1729`.
- Acknowledgment for the impressive performance of the 7B model by xDan from User `.@.benxh`.
- Discussion about the controversy stirring on the LocalLLaMa subreddit, involving accusations of fraudulent activities against AI model creators, references shared by `.@beowulfbr` ([thread1](https://www.reddit.com/r/LocalLLaMA/comments/18qp3fh/this_is_getting_ridiculous_can_we_please_ban/), [thread2](https://www.reddit.com/r/LocalLLaMA/comments/18ql8dx/merry_christmas_the_first_opensource/)).
- Mention of the backlash received by User `.@beowulfbr`'s model, CodeNinja, after an initial positive response on the subreddit ([thread3](https://www.reddit.com/r/LocalLLaMA/comments/18pr65c/announcing_codeninja_a_new_open_source_model_good/)).
- Comprehensive guide on mastering Java, ranging from setup to real-world projects, shared by `@ty.x.202`.
- Discussion on the method of stacking multiple adapters on a base model, rather than using the RAG approach, along with a link to a related GitHub project: [S-LoRA](https://github.com/S-LoRA/S-LoRA) by `@lightningralf`.
- Announcement of the new version of OpenChat model with its HuggingFace link and online demonstration, even though it scores similarly to prior versions. Server access, model view, and online demo's respective links: [server](https://discord.gg/ySZJHsPt), [model view](https://huggingface.co/openchat/openchat-3.5-1210), [Online demo](https://openchat.team).
- Inquiry from `@joshxt` about users finding value in using or fine-tuning a specific context, although the context wasn't specified.

**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (1 messages): 
        
- **Best Function-Calling Model Inquiry**: User `@harsh1729` asked for advice on the best function-calling model available on Hugging Face, preferably a derivative of **Llama-7B**.


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (10 messages🔥): 
        
- **xDan Model Praise**: `.@.benxh` praised the impressive output of the 7B model by xDan. 
- **Controversy on Reddit**: `.@beowulfbr` pinpointed the negative attention directed towards AI models on the subreddit LocalLLaMa. Attackers are hurling accusations of fraudulent activity towards those creating and merging models. Links to the evidential threads were shared ([thread1](https://www.reddit.com/r/LocalLLaMA/comments/18qp3fh/this_is_getting_ridiculous_can_we_please_ban/), [thread2](https://www.reddit.com/r/LocalLLaMA/comments/18ql8dx/merry_christmas_the_first_opensource/)).
- **CodeNinja Model Targeted**: `.@beowulfbr` also mentioned the release of their own model, CodeNinja, on the same subreddit. Despite initially receiving positive feedback, it was later targeted by the same group of attackers ([thread3](https://www.reddit.com/r/LocalLLaMA/comments/18pr65c/announcing_codeninja_a_new_open_source_model_good/)).
- **Guide on Mastering Java**: `@ty.x.202` provided a comprehensive guide on mastering Java, which includes setting up a Java environment, mastering Java fundamentals, exploring object-oriented programming principles, understanding Java APIs, handling exceptions, JDBC and database connectivity, GUI development with JavaFX, advanced Java topics, and finally, real-world projects.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 messages): 
        
- **Stacking Multiple Adapters on a Base Model**: `@lightningralf` discussed the approach of stacking multiple adapters on a base model and switching them out every week/month instead of the RAG approach. He shared a link to a [GitHub project named S-LoRA](https://github.com/S-LoRA/S-LoRA), highlighting its capability to serve thousands of concurrent LoRA Adapters in a fast way.


### ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/) (1 messages): 
        
- **OpenChat Announcement**: An announcement about a new version of OpenChat model has been made. The server can be accessed [here](https://discord.gg/ySZJHsPt). In terms of performance, it scores equally high as previous versions.
- **HuggingFace Link**: OpenChat's new model can be viewed [here](https://huggingface.co/openchat/openchat-3.5-1210). It's an advancement in open-source language models using mixed-quality data.
- **Online Demo**: The OpenChat model can be tested in an [Online Demo](https://openchat.team).


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (1 messages): 
        
- **Fine-tuning Value Inquiry**: `@joshxt` asked if any user found value in using or fine-tuning a specific context, though the context was not specified in the available information.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Anergetic discussions on the **bias of LLM-as-a-Judge evaluation**, highlighting concerns such as exploitation of *GPT-4's preference for long outputs*. Possible mitigation strategies included evaluating questions with reference answers.
- Examination of the **application of Disco Judge in different evaluation contexts**. Difference in contexts noted, with *reference answers not always being available* in instances like synthetic data filtering or quality assurance.
- Conversation regarding the limitations of the **LLM-as-a-Judge methodology**. Issues raised included inherent biases or narrow scope depending on whether an LLM determines answer quality or if it compares to a reference answer generated by an LLM.
- Conclusion that there's a need for continued discussions and **quantification of these limitations** in LLM-as-a-Judge evaluation methods.
- Presentation of **benchmarked results of various 7B models**. Models discussed include *xDAN-AI/xDAN-L1-Chat-RL-v1*, *openchat/openchat-3.5-1210*, *Weyaxi/SauerkrautLM-UNA-SOLAR-Instruct*, and *upstage/SOLAR-10.7B-Instruct-v1.0*. The benchmarking tool and source code was mentioned as [EQbench](https://www.eqbench.com/) with the accompanying [Source Code](https://github.com/EQ-bench/EQ-Bench) and [Paper](https://arxiv.org/abs/2312.06281).
- Skepticism voiced about the scores of the **7B models in comparison to models like GPT-4 or Mixtral**.
- A request to **benchmark Nous Hermes 2** based on the yi34b model.

**DiscoResearch Channel Summaries**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (4 messages): 
        
- **Discussion on the Bias of LLM-as-a-Judge Evaluation**: `.calytrix` expressed concerns about models being fine-tuned to exploit the biases of a judge model such as **gpt-4's preference for long outputs**. They suggested mitigating this issue by evaluating questions that have reference answers.
- **Application of Disco Judge in Different Evaluation Contexts**: `_jp1_` responded by stating that for "real" evaluation and benchmarks, reference answers would be helpful but in applications like synthetic data filtering or quality assurance, reference answers often do not exist. They highlighted that **Disco Judge** aims to accommodate both use cases.
- **Limitations of LLM-as-a-Judge Methodology**: `.calytrix` further brought up that the current llm-as-a-judge benchmarks could be limiting as they either allow the llm to determine what is a good answer based on any set metric, or they compare to a reference answer generated by an llm. 
- **Need for Active Discussion on Drawbacks**: `bjoernp` agreed with the concerns and emphasized the need for active discussions and quantification of these limitations later down the line.


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (3 messages): 
        
- **Benchmarking Top 7B Models**: User `@.calytrix` presented the results of benchmarked leaderboard positions of some 7B models. Notably, all these models scored lower than GPT-4, GPT-3.5 and Mistral-7B-OpenOrca. The benchmarked models and scores include:

    - xDAN-AI/xDAN-L1-Chat-RL-v1: 40.12
    - openchat/openchat-3.5-1210: 43.29
    - Weyaxi/SauerkrautLM-UNA-SOLAR-Instruct: 41.01
    - upstage/SOLAR-10.7B-Instruct-v1.0: 41.72
    
    The benchmark results were obtained from [EQbench](https://www.eqbench.com/). Here is the [Source Code](https://github.com/EQ-bench/EQ-Bench) and [Paper](https://arxiv.org/abs/2312.06281) for reference.
 
- **Comparing 7B Models to GPT-4 or Mixtral**: User `@cybertimon` expressed a skepticism on these 7b models outperforming GPT-4 or Mixtral. 
- **Benchmarking Nous Hermes 2**: `@gheshue` asked `@.calytrix` to benchmark Nous Hermes 2 based on yi34b model.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Guidance Library for Language Model Interactions**: `@explore8161` shared information about the Guidance Library designed for enhancing interactions with language models. This tool aids in efficient prompt development and precise response generation, providing an effective means to control the language model. A related [Medium article](https://medium.com/@saurabhdhandeblog/language-model-prompt-engineering-guidance-library-7b9ad79cf9d4) provides more details on this topic.
- **Blog Post on Everyday Machine Learning Use Cases**: `@jillanisofttech` posted a link to a [Medium blog](https://jillanisofttech.medium.com/ten-everyday-machine-learning-use-cases-a-deep-dive-into-ais-transformative-power-b08afa961e10) focusing on the daily use cases of Machine Learning and its transformative power.
- **Request for Feedback on Kaggle Notebook**: `@vengeance3210`, a beginner in the field, requested feedback on a [Kaggle notebook](https://www.kaggle.com/code/nishchay331/n6-house-prices-advanced-regression-techniques) focusing on house price advanced regression techniques.
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Deep learning rig configuration** : 
  - User `@zmuuuu` asked for advice on motherboard recommendations for a deep learning rig with **2 3090s NVIDIA founder's edition version cards** in SLI configuration, expressing concerns over cooling.
  - In response, `@jeffreyw128` suggested considering **open-air** setup to alleviate cooling issues.
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Medical Image Diagnosis Project**: User `onuralp.` is working on a medical image diagnosis project using **GPT-4 Vision** and is seeking benchmark models for comparison. He asked about experiences using the **Med-LM API** and interest in details about connections with the Google Product Team. 
- **Open Source Model Inquiry**: `onuralp.` also inquired about the availability of open source models for inclusion in the project. Specifically, he requested any benchmarking data comparing **Hermes 2 Vision** and **bakllava**.
        

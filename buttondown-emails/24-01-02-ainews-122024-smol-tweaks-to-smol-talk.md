---
id: 35042e7a-2765-48c3-a894-4ff649792087
title: '[AINews] 1/2/2024: Smol tweaks to Smol Talk'
date: '2024-01-03T07:38:24.484214Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-122024-smol-tweaks-to-smol-talk
---

<!-- buttondown-editor-mode: plaintext -->Some Smol Talk meta today: we didn't feel too happy with the "boringness" of the bullet point summaries so we tried to tweak the prompts:

 ![image.png](https://assets.buttondown.email/images/36e567d0-bf55-4196-b70a-f04e8d5f00e6.png?w=960&fit=max) 

The results are reflected in the summaries shown below. However it feels like they have lost information because our few shot examples were too short. We'd like to revise this whole pipeline to [use function calling like god intended](https://twitter.com/minimaxir/status/1737884828111179916).

---

**Table of Contents**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **The Search for AI Superiority**: Detailed comparison among AI search engines, including **Perplexity**, **Copilot**, **Bard**, and **Claude 2**, mentioned by `@jaicraft` in [ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/), with Bard and Claude 2 falling second in the race.

- **Meta AI Steals the Show**: Introduction of **Meta AI**, a chatbot by Meta (formerly Facebook), into the conversation with availability on both Instagram and Whatsapp, as shared by `@thedreamakeem` and `@jaicraft`.

- **Browsers Battle with ChatGPT**: Numerous reports of multiple browser issues with OpenAI's ChatGPT ranging from persistent captchas with VPN usage to non-functioning plugins, found in the [openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) channel.

- **ChatGPT’s Personality Switch**: Spirited exchanges in [openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) about tuning ChatGPT's personality, divergent model capabilities, and workable solutions for software development support.

- **JSON vs. Markdown and YAML**: Debates in [gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) about the use of **JSON**, **YAML**, or **Markdown** for task-specific applications.

- **Prompt Engineering Puzzles**: Complex discussions in [prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) cover encounter paradoxes, uncovering model features and issues, and scripting for TikTok and SWOT analysis.

- **Unpacking the API**: [API-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) unravels the paradox of "last seen", real-time data retrieval through Bing, script feeding, prompt construction techniques, and limits of text outputs in API implementations.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (11 messages🔥): 
        
- **Comparison of AI Search Engines**: `@jaicraft` stated that **Perplexity** the AI-based search engine performed well. However, they also considered **Copilot** to be satisfactory, arguably outperforming **Bard** and **Claude 2**.
  
- **Request for Bard Update**: `@jaicraft` expressed a desire to see **Gemini Ultra** incorporated into **Bard Advanced**. They appreciated Bard's personality, but felt the model could improve with this update.
  
- **Launch of Meta AI**: `@thedreamakeem` announced the launch of **Meta AI** by Meta (formerly Facebook). Notably, this chatbot is currently operational on Instagram.
  
- **Meta AI on Whatsapp**: In a follow-up conversation, `@jaicraft` noted that Meta AI is also available on **Whatsapp**.
  
- **Meta AI Image Generation**: `@thedreamakeem` stated that Meta AI includes an **image generation feature**, which he likened to a free version of GPT.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (209 messages🔥🔥): 
        
- **ChatGPT issues on multiple browsers**: Users `@rosyskull` and `@kilogamz` reported issues with ChatGPT on multiple browsers including Opera GX, Firefox, and Edge. Even after clearing cache, using different devices and trying different network connections, the issue persisted. `@satanhashtag` suggested reaching out to `help.openai.com` and also posting in `<#1070006915414900886>` - OpenAI's specific channel for such enquiries.
  
- **ChatGPT information about itself is unreliable**: Users `@lugui` and `@beanz_and_rice` discussed the unreliable nature of ChatGPT when it comes to providing information about itself, specifically its versions, upgrades, and capabilities. It was emphasised that ChatGPT often hallucinates or misrepresents this information. Discussion also clarified the different versions and terminology used in model names.

- **Issues with constant captchas when using VPN**: User `@thedeeplearningdummy` raised an issue of constant captcha verifications whenever they access ChatGPT through a VPN connection. Esteemed theory among discussion is that VPNs can trigger security measures due to cache/IP masking, though the matter remains inconclusive.

- **Issues with AskYourPDF plug-in in GPT+**: User `@kez92.` reported an issue with the plug-in “AskYourPDF” in GPT+, which ordinarily allows ChatGPT to process PDF files. Problem is, instead of allowing the upload of a PDF, it requests to copy and paste the information or provide a link, which is an abnormal behavior. No resolution was provided in the messages. 

- **Irregular output and rapid usage cap hit with GPT3.5-turbo-1106**: Multiple users reported issues with the `GPT3.5-turbo-1106` model, including `@minister_maximus` who reported receiving gibberish output, and `@Mr Gingerfish` who claimed to have hit the usage cap after just four prompts. Others provided some theories, possible causes, and solutions, but issues appear to be ongoing and unresolved.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (75 messages🔥🔥): 
        
- **ChatGPT Inconsistently Nailing Personality Traits**: `@thefreebachelor` and `@solbus` discuss the difficulty of tuning ChatGPT's personality. `@thefreebachelor` mentions that a knowledge file worked well for configuring an AI's profile picture but not its personality, expressing frustration that a long Reddit paragraph caused the model to become messy. `@solbus` suggests starting with a fresh GPT and manually editing the instructions to get closer to the desired outcome.

- **Conversation Limitations with GPT-4**: Users `@crafty.chaos`, `@paulh3911`, and `@darthgustav.` discuss encountering limits to how long a conversation with GPT-4 can be. This question was raised after users reported reaching the error "conversation too long, please start a new one" during their chat session with the AI. 

- **Technical Issues with ChatGPT**: Users `@ssk0746`, `@froggy_chacko`, `@bartomeu_70091_41043`, `@watchalls`, and `@solbus` discuss technical issues they've encountered with the ChatGPT service. Issues include ChatGPT 4.0 not working with certain VPNs and issues with excessive verification puzzles. `@bartomeu_70091_41043`'s issue with the human verification puzzles was temporarily resolved by not using a VPN with Firefox.

- **Usage of ChatGPT in Software Development Work**: User `@muhct` presented a scenario, questioning the best way to use ChatGPT for a software development job without the employer knowing. Suggestions from `@spyied` and `@michael_6138_97508` revolved around using alternative services like GitHub's Copilot and logging responsibilities for proprietary data protection.

- **Quiz on GPT-4's Message Cap Limit**: `@yomo42` queried group members whether "GPT-4 classic" has a higher message cap than regular GPT-4. `@yomo42` added that they had learned the cap for custom GPT versions is 25 messages per 3 hours.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (6 messages): 
        
- **Considering YAML and Markdown over JSON**: `@sciandy` suggested the possible use of **YAML** or **Markdown** as alternatives to JSON for certain tasks.
- **Request for Technical Assistance on Wordpress**: `@delightful_rabbit_60937` is seeking help in implementing a chatbot on their freelo WordPress using the **Power AI** plugin and wants it to provide a link on freelancer with relevant proficiency.
- **Benefits of JSON**: `@niko3757` stressed that **JSON** is the most effective way as it allows searching with keywords, effectively saving valuable tokens.
- **Return of missing GPT's**: `@loschess` mentioned that all their missing GPT's have returned and wondered if anyone else experienced the same.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (169 messages🔥🔥): 
        
- **GPT-4's Paradoxical "Last Seen" Problem**: `@beanz_and_rice` identifies a paradoxical issue in GPT-4 Preview when asking about the "last time X was seen". The model explained that this issue occurred due to the ambiguity of the term "last," which creates a conflict between the model's instruction to access historical data up until 2023 and the inherent implication of immediacy within the term. This discrepancy resulted in the model providing the most recent information within its established timeframe (pre-2023), which may diverge from the actual "last" occurrence beyond 2023.
- **Various Versions of GPT-4 and Their Features**: The conversation uncovers various aspects of different GPT-4 versions. For example, GPT-4.0613 does not have the same issue with the "last seen" query as GPT-4 Preview does. Furthermore, `@beanz_and_rice` questions whether GPT-4 Turbo incorporates vision and whether GPT-4.1106 is capable of "seeing" images, with `@madame_architect` clarifying that Vision is a skill of GPT-4 Turbo. 
- **GPT-4's Overreliance on Bing**: Some users, particularly `@beanz_and_rice`, express dissatisfaction with GPT-4's tendency to lean on Bing for information retrieval despite its extensively trained dataset. `@madame_architect` rationalizes this behavior by indicating that GPT-4 is trained to use the Bing skill, consequently using real-time data from search results to enhance its responses.
- **Prompt-Engineering for TikTok Scripts and SWOT Analysis**: `@user691378` seeks advice on crafting a prompt for generating funny TikTok scripts with ChatGPT, to which `@beanz_and_rice` offers unique pseudo language. In contrast, `@madame_architect` presents a challenge to create prompts that generate SWOT analyses on technologies, citing superior results when prompting individually for strengths, weaknesses, opportunities, and threats.
- **Challenges and Solutions for Modulating GPT-4's Output**: Both `@MishaMgla` and `@vangroovie` grapple with issues in GPT-4's outputs. `MishaMgla` struggles with the need to limit response lengths without over-engineering the model and incurring extra costs. Acknowledging limitations, `@rendo1` suggests setting token limitations via the API but warns of potential sentence cutoffs. `@vangroovie` reports issues with GPT inaccurately transposing speakers in customer service transcripts.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (169 messages🔥🔥): 
        
- **The Paradox of Last Seen**: User `@beanz_and_rice` unearthed a paradox in **GPT-4-Preview's** response to a query involving the "last time X was seen." The chatbot described it as an ambiguous statement, explaining that as it has historical data up until 2023 but cannot access real-time updates, it doesn't have the capacity to determine the most recent occurrence of any event post-2023.

- **Bing Browsing and Its Role in API**: User `@beanz_and_rice` analyzed that every time "real-time" info is requested, GPT-4-Preview tends to invoke the Bing search skill. `@madame_architect` further clarified that Bing isn't the entire dataset of the Internet, but rather a tool GPT uses to grab necessary data for specific tasks in real-time..

- **Utilizing GPT for Unique Tasks**: User `@user691378` initiated a discussion on force-feeding GPT custom-made scripts in an attempt to generate a funny texting video for TikTok.

- **Techniques for Constructing Effective Prompts**: `@madame_architect` shared her experiences and methods for preparing effective prompts, which included reading numerous AI papers on prompting tactics and indulging in on-hand learning experiments with GPT.

- **Limits of Text Outputs**: `@MishaMgla` voiced concerns about controlling the length and quality of output generated by GPT. They found that keeping the text's length within certain range required the generation of an additional response, thus incurring extra time and costs. Users proposed methods to limit the output length, although some of these methods might cause GPT to cut off text mid-sentence.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Show Me the Merges**: Announcement of a new 'Show Merges' filter added to Open LLM Leaderboard, as initially suggested by the Stanford NLP Group, by `@clefourrier`, aiming to assist creators with incomplete metadata. [Source Tweet](https://x.com/clefourrier/status/1742224428639928694)
- **Low-Latency Inference**: AutoAWQ, which allows running a Mixtral 8x7B MoE with Flash Attention 2 in ~24GB GPU VRAM and fast inference, announced by `@reach_vb`. [Source Tweet](https://x.com/reach_vb/status/1741175347821883502)
- **Whisper Guides Metal**: Whisper on Metal now powered by the Rust programming language, as revealed by `@reach_vb`. [Source Tweet](https://x.com/reach_vb/status/1740804591095283899)
- **Helping Hands in Hugging Face**: `@RisingSayak` spotlights useful response from `@vipitis` providing guidance to `@ronny4002` interested in contributing to the Hugging Face's open-source projects. [Contributing Guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)
- **Real-Time Math Class**: `@osanseviero` shares an informative blog post titled [The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/), providing an end-to-end walkthrough of the mathematics within a transformer model.
- **Where Do We Go From Here?**: Exploration of Task-Oriented LLM Systems' design parameters through the paper titled *"The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey"* discussed by `@dhruvdh` and `@chad_in_the_house`. [Author's Link](https://arxiv.org/abs/2312.17601)
- **Encouraging Tech Evolution**: `@apolinariosteps` introduced the `Advanced Training Script` which combines successful techniques from the community for improved training capabilities. [Release Tweet](https://x.com/linoy_tsaban/status/1742281649294016742?s=20)
- **Eyes on Visual Output**: `@wandereronarock` references repository [patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux) as an example of the visual output method they're discussing.
- **AI Bites into Boolean**: Dialogue between `@stroggoz` and `@ketul1842` revolved around generating boolean logic responses to prompts using encoder-decoder models or next sentence prediction strategies.

**HuggingFace Discord Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 messages): 
        
- **Filter Merges on Open LLM Leaderboard**: User `@clefourrier` [announced](https://x.com/clefourrier/status/1742224428639928694) a new `Show Merges` filter for the Open LLM Leaderboard and requested assistance to help model creators with incomplete metadata. The idea of a 'Show merges' filter was initially suggested by the Stanford NLP Group.
- **Enhanced Mixtral 8x7B with AWQ and Flash Attention 2**: `@reach_vb` [presented](https://x.com/reach_vb/status/1741175347821883502) the latest release of AutoAWQ, which allows running a Mixtral 8x7B MoE with Flash Attention 2 in ~24GB GPU VRAM and fast inference. The user also provided a comprehensive guide to use this feature.
- **Whisper on Metal Powered by Rust**: `@reach_vb` [shared](https://x.com/reach_vb/status/1740804591095283899) that the Whisper on Metal is now powered by the Rust programming language.
- **Utility Libraries by Hugging Face – ‘huggingface_hub’ and ‘accelerate’**: `@RisingSayak` [mentioned](https://x.com/RisingSayak/status/1740998251447550329) two non-modelling libraries by Hugging Face – `huggingface_hub` for integration with the Hub and `accelerate` to handle model inference.
- **DINOv2 Fine-tuning and Qwen Updated**: User [NielsRogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb') shared a tutorial on fine-tuning DINOv2 for image classification using the Hugging Face Trainer. Meanwhile, `@JustinLin610` [announced](https://x.com/JustinLin610/status/1742184229453320451) the update of Qwen-VL to Qwen-VL-Plus, improving capabilities like image captioning, visual question answering, visual grounding, OCR, and visual reasoning.

**Links mentioned**:

- [Tweet from Clémentine Fourrier 🍊 (@clefourrier)](https://x.com/clefourrier/status/1742224428639928694): Just added a &#34;Show merges&#34; to filter out m...
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1741175347821883502): Mixtral 8x7B Instruct with AWQ & Flash Attention 2...
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1740804591095283899): fuck yeah! whisper on metal powered by rust 🦀  10...
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1740998251447550329): Hugging Face has non-modelling libraries that the ...
- [Transformers-Tutorials/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb at master · NielsRogge/Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb): This repository contains demos I made with the Tra...
- [Tweet from Junyang Lin (@JustinLin610)](https://x.com/JustinLin610/status/1742184229453320451): 📷🎨👀 https://huggingface.co/spaces/Qwen/Qwen-VL-...


### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (167 messages🔥🔥): 
        
- **AI for Legal applications**: `@caleb_sol` shared a [link](https://lsvp.com/legaltech-x-ai-the-lightspeed-view/) of an article highlighting how AI can constitute a significant transformation in the field of law.
- **Consistent Delays in Inference Endpoints**: `@duplaja` mentioned that he has been noticing persistent slowness of the Inference Endpoints to spin up from a pause, shared it in ask for help.
- **Contributing to Hugging Face**: `@ronny4002`, a student, expressed interest in contributing to open-source projects of Hugging Face, particularly for code development and training dataset contributions focused on NLP/LLM. `@vipitis` guided him to the [contributing guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md) on GitHub and advised looking into the Hugging Face course and docs for further insights.
- **Differences between `torch_dtype=torch.float16` & `variant="fp16"` for `from_pretrained`**: `@felixsanz` questioned the difference between `torch_dtype=torch.float16` and `variant="fp16"` in `from_pretrained` function. `@kyvarus` and `@jo_pmt_79880` provided the explanation that `torch_dtype` specifies the tensor's datatype and `variant` specifies the branch of the model repo. The discussion ended with `@vipitis` explaining that specifying the dtype will convert to the target datatype either way.
- **Performance Issues with Gradio**: `@tony_assi` reported frequent bugs with Gradio, leading to daily issues. `@cakiki` suggested connecting with the Gradio team via specific channels or raising issues on GitHub.


**Links mentioned**:

- [Legaltech x AI: The Lightspeed View - Lightspeed Venture Partners](https://lsvp.com/legaltech-x-ai-the-lightspeed-view/)
- [transformers/CONTRIBUTING.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md): 🤗 Transformers: State-of-the-art Machine Learning...
- [transformers/src/transformers/modeling_utils.py at aa4a0f8ef37eb5d42b4e3810f37e554585c90d41 · huggingface/transformers](https://github.com/huggingface/transformers/blob/aa4a0f8ef37eb5d42b4e3810f37e554585c90d41/src/transformers/modeling_utils.py#L3348): 🤗 Transformers: State-of-the-art Machine Learning...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 messages): 
        
- **Navigating HF Inference Endpoints**: `@duplaja` shared their experience with learning around `handler.py HF Inference Endpoints`, but they admitted it's happening *very, very slowly*.
- **No Changes Made, Just Parameters Adjustment**: `@gag123` stated they *did not change anything* and were simply *playing with the parameters*.
- **MacOS Ease Surprises Windows User**: `@lawls.net`, a backend software developer, expressed how they were surprised at the ease of using Python on MacOS as compared to Windows, leading them to purchase a M3 Max.
- **Advice on Learning Rate & Data dtype**: `@exponentialxp` gave some tips, suggesting to increase the learning rate to *3e-4* and advising to set the data to *dtype=torch.long*.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **Demystifying Transformers with Mathematics**: User `@osanseviero` shared a link to his new blog post titled [The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/). The post offers an end-to-end example of the **mathematics within a transformer model**, simplifying it for better understanding. The post recommends readers to also view [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) for a more intuitive explanation of the transformer model. User `@jo_pmt_79880` expressed appreciation, finding the post "**Very informative**".

**Links mentioned**:

[hackerllama - The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/): Understand how transformers work by demystifying a...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (3 messages): 
        
- **Unleashing Conversational Power with AI Fusion**: User `@andysingal` shared a [blog post](https://medium.com/ai-advances/unleashing-conversational-power-rag-agent-tools-and-trulens-eval-revolutionize-document-chatting-32845ea2215b) discussing the integration of **LlamaIndex and Gemini** to enhance AI capabilities. 
- **Find Your Celebrity Look-alike with AI**: `@tony_assi` developed an [app](https://huggingface.co/spaces/tonyassi/celebrity-look-a-like) that identifies celebrities that users look similar to.

**Links mentioned**:

- [Celebrity Look A Like - a Hugging Face Space by tonyassi](https://huggingface.co/spaces/tonyassi/celebrity-look-a-like)
- [Unleashing Conversational Power: RAG, Agent Tools, and Trulens-Eval Revolutionize Document Chatting…](https://medium.com/ai-advances/unleashing-conversational-power-rag-agent-tools-and-trulens-eval-revolutionize-document-chatting-32845ea2215b): Ankush k Singal


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (12 messages🔥): 
        
- **Exploring Task-Oriented LLM Systems**: `@dhruvdh` shares a new paper titled *"The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey"*. The paper examines the design parameters for task-oriented LLM systems and forms three conjectures, providing a starting point for future research. Find the paper [here](https://arxiv.org/abs/2312.17601).

- **Paper Presentation Proposal**: `@chad_in_the_house` recommends the paper shared by `@dhruvdh` for the week's presentation. `@lunarflu` confirms that it's a potential option and asks for any other suggestions.

- **Request for Help with RL Keras**: `@only_busy` requests suggestions for libraries compatible with the new gymnasium of OpenAI as he encounters compatibility issues with RL Keras and OpenAI gym.

- **Reading Group Calendar**: `@swyxio` expresses interest in a calendar to keep track of reading group sessions. `@clock.work_` suggests checking if the group events are listed on Discord, and `@chad_in_the_house` suggests finding the information in Discord threads.

- **Easier Access to Past Presentations**: `@chad_in_the_house` proposes sharing presentations on HuggingFace blog posts and Discord to increase availability of past presentations and potential presenters.

**Links mentioned**:

- [The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey](https://arxiv.org/abs/2312.17601): This scoping survey focuses on our current underst...
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/18w09hn/r_the_tyranny_of_possibilities_in_the_design_of/)


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **New Advanced Training Script Unveiled**: User `@apolinariosteps` announced the maiden release of 2024, the `Advanced Training Script`. The script incorporates successful techniques used by the community, including Pivotal Tuning from `cog-sdxl` and the `Prodigy Optimizer` from `kohya`. The script is compatible with both AUTO1111 and ComfyUI and can be used via Google Colab, Hugging Face Spaces, or as a direct Python script. [Release Tweet](https://x.com/linoy_tsaban/status/1742281649294016742?s=20)
- **Advanced Training Script Features**: The script draws inspiration from the Pivotal Tuning technique used in Replicate's SDXL Cog trainer and the Prodigy optimizer from Kohya's trainer. It also features other optimizations promising effective results for SDXL Dreambooth LoRA fine tuning. [Link to Blog](https://huggingface.co/blog/sdxl_lora_advanced_script)
- **Platform Compatibility**: The aforementioned script allows for varied use across different platforms. Users can execute the script on [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb), [Hugging Face Spaces](https://huggingface.co/spaces/multimodalart/lora-ease) with a simple UI and custom parameters, or directly as a Python script.

**Links mentioned**:

- [Tweet from Linoy Tsaban🎗️ (@linoy_tsaban)](https://x.com/linoy_tsaban/status/1742281649294016742?s=20): Let&#39;s go 2024 🚀:  🆕 training script in 🧨 @d...
- [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script)


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **ControlNet_Aux Linked as Visual Output Example**: `@wandereronarock` referred to the repository [patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux) as an example of the output they believe is produced by the method they're discussing. The user noted their uncertainty due to not having used the method in a while.

**Links mentioned**:

[GitHub - patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux): Contribute to patrickvonplaten/controlnet_aux deve...


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **Prompts to Propositions**: `@stroggoz` and `@ketul1842` discussed the idea of creating the correct boolean logic proposition from a given prompt. `@stroggoz` suggested this could be achieved using encoder-decoder models or next sentence prediction strategy.

- **OpenAI to HuggingFace Transition**: User `.oo92` shared a function using the **OpenAI API** to generate a knowledge graph based on input system message and file prompts, and expressed a struggle to mirror similar logic with **HuggingFace's Bloom**. 

- **Web-Scraping for Datasets**: `@exponentialxp` asked about the web-scraping techniques used in datasets like **owt** and **c4**. `@cakiki` clarified that **C4** is a filtered version of the **Common Crawl** (CC), which crawls the web monthly. 

- **GPU Recommendations**: `@exponentialxp` also recommended checking karpathy/nanoGPT on GitHub for information on how to train models using multiple GPUs, and suggested Lambda as a good choice for GPU rental.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **The TinyLlama MOE Experiment**: `@le_mess` shared his early-stage Mixture of Experts (MoE) model using eight instances of [TinyLlama](https://huggingface.co/mhenrichsen/tinymix-8x1b). Fine-tuning details and a lively discussion on the LASER method for model performance enhancement were also highlighted. The LASER method, as described by `@mihai4256`, integrates with existing models, raising intrigue around potential overfitting.
- **Exploration of Mixtral and Axolotl**: Deep dives into Mixtral and Axolotl implementation extended from suggestions for GitHub project use-cases, queries about prompt formatting in Axolotl, and support for SFT. Advice on training Mistral with qlora, shared by `@noobmaster29`, steered the dialogue towards maintaining focused conversations in developer channels.
- **Training FAQs and Troubleshooting Tips and Tricks**: In the `general-help` channel, users discussed improving diversity in model responses, encountered index-out-of-bound issues during training, analyzed LoRA configurations, and broke down scoring methods for Hellaswag. The discovery of a discrepancy between Hugging Face tokenizer and the Sentencepiece original implementation spurred further conversation.
- **New and Noteworthy Dataset Debut**: `@cf0913` introduced a synthetic code dataset `DevSpecCode` for complex multi-requirement code generation instructions. Feedback was sought for this dataset hosted on huggingface.co. The dataset includes intricate instructions for code generation.
- **Roleplay Datasets and Discussions on Fine-Tuning**: Delving into roleplay datasets, `@nruaif` shared access to sensitive content and potential for more. Additionally, there was a debate on the preferred dataset selection for fine-tuning, comparing redpajama-v2 with slimpajama, with a focus on quality filtering.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (63 messages🔥🔥): 
        
- **MOE using 8x TinyLLama**: `@le_mess` created an MoE (Mixture of Experts) model using eight instances of [TinyLlama](https://huggingface.co/mhenrichsen/tinymix-8x1b). It's in the early evaluation phase and may require further training to outperform its components. The work was done using the [mergekit](https://github.com/cg123/mergekit/tree/mixtral) mixtral branch.
- **LASER discussion**: Various community members discuss the LASER method to improve model performance by removing higher-order components of models' MLP layers. `@mihai4256` shared [his experimentation with LASER](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1) on models and mentioned a straightforward integration of LASER with existing models using a Python package, provided the models can be loaded in Hugging Face's Transformers. `@nafnlaus00` expressed interest in the LASER method's potential overfitting.
- **Tokenizer Website Query**: `@yamashi` asked for a website URL that compares tokenization results across different tokenizers. The website's purpose is to provide visualizations of tokenizations.
- **Training and Finetuning Discussions**: Various community members (notably `@nafnlaus00`, `@noobmaster29`, and `@le_mess`) made remarks on training and finetuning models, including mentions of dropout + LR metaparameter tweaking and MoE model training.
- **Comparison of Fine-tuning Methods**: `@jaredquek` [shared a paper](https://huggingface.co/papers/2401.00788) detailing the performance comparisons among different fine-tuning methods across scales and tasks. The paper found that full-parameter fine-tuning (FFT) generally delivers the best performance across all scales, while the performance trade-offs of parameter-efficient fine-tuning (PEFT) methods noticeably vary based on model scale. One method, LoRA, was typically found to offer a favorable balance between cost and performance.

**Links mentioned**:

- [The Truth Is In There: Improving  Reasoning in Language Models with Layer-Selective Rank Reduction](https://pratyushasharma.github.io/laser/)
- [Paper page - Astraios: Parameter-Efficient Instruction Tuning Code Large Language
  Models](https://huggingface.co/papers/2401.00788)
- [mhenrichsen/tinymix-8x1b · Hugging Face](https://huggingface.co/mhenrichsen/tinymix-8x1b)
- [GitHub - open-compass/MixtralKit: A toolkit for inference and evaluation of &#39;mixtral-8x7b-32kseqlen&#39; from Mistral AI](https://github.com/open-compass/MixtralKit): A toolkit for inference and evaluation of &amp;#39...
- [GitHub - cg123/mergekit at mixtral](https://github.com/cg123/mergekit/tree/mixtral): Tools for merging pretrained large language models...
- [mhenrichsen](https://wandb.ai/mhenrichsen/huggingface?workspace=user-mhenrichsen): Weights & Biases, developer tools for machine lear...
- [Mihaiii/Pallas-0.5-LASER-0.1 · Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1)
- [GitHub - pratyushasharma/laser: The Truth Is In There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://github.com/pratyushasharma/laser): The Truth Is In There: Improving Reasoning in Lang...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (9 messages🔥): 
        
- **GitHub project shared for potential Mixtral use**: `@casper_ai` shared a link to a [GitHub project](https://github.com/imoneoi/cutlass_grouped_gemm) titled "PyTorch bindings for CUTLASS grouped GEMM", suggesting that it could be used for **Mixtral**.
- **Query about Axolotl's support for SFT**: `@mrfakename_` inquired if Axolotl supports SFT. `@noobmaster29` confirmed that it does, mentioning that lora/qlora is SFT.
- **Discussion on prompt format usage in Axolotl**: `@mrfakename_` also asked about the possibility to use the Zephyr prompt format. `@noobmaster29` responded that it might require self-setup.
- **Recommended training for Mistral with qlora**: `@builderx` posed a question about using qlora to train mistral on a dataset of 15k rows, asking if three epochs would suffice. `@noobmaster29` confirmed that three epochs should be fine and suggested watching the eval loss. However, they also recommended using the help channels for such questions to keep the `axolotl-dev` channel clear for developers.

**Links mentioned**:

[GitHub - imoneoi/cutlass_grouped_gemm: PyTorch bindings for CUTLASS grouped GEMM.](https://github.com/imoneoi/cutlass_grouped_gemm): PyTorch bindings for CUTLASS grouped GEMM. Contrib...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (33 messages🔥): 
        
- **Shuffling data during preprocessing suggested for diversity**: A discussion arose about enhancing the diversity of model responses. While `@casper_ai` suggested adding more diverse data, they also recommended shuffling the data during preprocessing.
- **Index out of bounds issue during training**: User `@matanvetzler` experienced a RuntimeError with CUDA during training due to an 'index out of bounds' issue. The error seems to occur when the model dimensions don't match the tokenizer size. This was pointed out by `@bjoernp`.
- **Question about LoRA configurations brought up**: `@tcapelle` asked whether `lora_target_linear` should be set to `False` when targeting modules with LoRA configurations, providing a link to an example on [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml) for reference.
- **Clarification on scoring methods for Hellaswag**: `@le_mess` asked about the correct scoring method for Hellaswag, and `@suikamelon` clarified that acc_norm (normalized scores) are typically used. They provided a link to the [HuggingFace OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) as an example of this use.
- **Discrepancy between Huggingface tokenizer and original Sentencepiece implementation noted**: User `@emperor` mentioned a discrepancy where the transformers/AutoTokenizer was not providing the same results as the original Sentencepiece representation. They shared code examples to illustrate the differences.

**Links mentioned**:

- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [axolotl/examples/mistral/qlora.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml): Go ahead and axolotl questions. Contribute to Open...


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (1 messages): 
        
- **Seeking Feedback on Specialized Dataset**: `@cf0913` has created a detailed synthetic code dataset for complex multi-requirement code generation instructions. They seek feedback on the dataset named `DevSpecCode`, which is hosted on huggingface.co. The dataset instructions include intricate requirements, limitations, and instructions for code generation. An example provided is about writing a safe concurrent function in Go. The dataset is available at [this link](https://huggingface.co/datasets/cfahlgren1/DevSpecCode).


**Links mentioned**:

[cfahlgren1/DevSpecCode · Datasets at Hugging Face](https://huggingface.co/datasets/cfahlgren1/DevSpecCode)


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (8 messages🔥): 
        
- **Roleplay Datasets Shared for "Not-For-All-Audiences"**: `@nruaif` shared links to two roleplay datasets, [rpguild](https://huggingface.co/datasets/chargoddard/rpguild) and [bluemoon-fandom-1-1-rp-cleaned](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned), from huggingface.co that contain sensitive content.
- **Potential to Access More Roleplay Data**: `@nruaif` also offered that **more roleplay data could be provided** if needed, hinting at an availability of **half a terabyte of data**.
- **Discussion about Fine-Tuning Strategy for Mixtral**: `@tcapelle` asked about the use of specific json file [zero3_bf16.json](https://github.com/OpenAccess-AI-Collective/axolotl/blob/transformers-update-mixtral/deepspeed/zero3_bf16.json) from GitHub for fine-tuning the Mixtral model.
- **Debate on Data Selection between Redpajama-v2 and Slimpajama**: `@xzuyn` suggested using **redpajama-v2** instead of **slimpajama** data. However, `@caseus_` pointed out that the continued pretraining might not require a large amount of data, citing the Sheared llama paper that only used **50B more tokens**. They advocated for the **quality filtering** present in slimpajama.

**Links mentioned**:

- [axolotl/deepspeed/zero3_bf16.json at transformers-update-mixtral · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/transformers-update-mixtral/deepspeed/zero3_bf16.json): Go ahead and axolotl questions. Contribute to Open...
- [chargoddard/rpguild · Datasets at Hugging Face](https://huggingface.co/datasets/chargoddard/rpguild)
- [Squish42/bluemoon-fandom-1-1-rp-cleaned · Datasets at Hugging Face](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned)


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Model Choices - Perplexity vs GPT-4 vs Claude 2.1**: Users on the server engaged in discussions on the best AI models to use, as `@prolapsedgod` struggled deciding between the Perplexity Experimental model and the GPT-4 model. `@hynjia` chimed in revealing their tendency to switch between GPT-4 and Claude 2.1.
- **The Intricate Mixtral Parameters**: `@.hackit` sparked interest as he inquired about the parameters of `mixtral-8x7b-instruct` with intentions of recreating it with the rest-API.
- **'HOLIDAYS23' Coupon Chaos**: The use of the 'HOLIDAYS23' Pro membership coupon code triggered intensive discussions with participation from numerous users (`@nightpearl62`, `@icelavaman`, `@jonathanonymous`, `@ok.alex`, `@danielagmz888`, and `@ashley__9910`). Problems encountered ranged from the link's functionality to applying the code, with responses recommending methods like subscribing via [Plexity.ai](https://pplx.ai/holidays), trying via mobile web, and contacting support.
- **Image Generation within Perplexity - A Visual Stimulus?**: `@araf@` and `@archient` probed into the benefits of image generation within Perplexity. The discussion revolved around its role in enhancing visual engagement and serving as a creative stimulant.
- **Plead for Mistral-Medium in Pro Accounts**: `@Rehnn` suggested the integration of the Mistral-Medium model into Pro accounts, praising its speed and reduced hallucinations based on their playground tests. `@reflext` agreed with this proposal.
- **Payment Woes across Borders**: `@ashley__9910` from Germany experienced difficulties with payment methods. Despite trying GPay on an Android phone, the problem persisted. `@icelavaman` advised reaching out to support@perplexity.ai for help with the Stripe-related issue.
- **Perplexity's Data Collection and Model Information**: `@maxxflyer` drew attention to [Perplexity's FAQ](https://docs.perplexity.ai/page/frequently-asked-questions), highlighting its data collection practices around API usage and user account info. They also commented that, according to a [Perplexity blog post](https://blog.perplexity.ai/blog/introducing-pplx-online-llms), the pplx-7b-online and pplx-70b-online models use public data and data from their search index.
- **Broken Discord Invite Link & Military Boarding History**: The faulty Discord invite link on Perplexity's page was spotlighted by `@maxxflyer`. In another conversation, `@jpa` expressed curiosity about military boarding history, referring to [Perplexity's Search](https://www.perplexity.ai/search/whats-the-history-2Hi46dUBTIayi4HlvrozmQ?s=c) for further exploration.
- **API and Channel Confusion**: A preference for model sorting installment was proposed by `@paul16307` with a wish to arrange Mistral models from the least to most powerful ones. As `@monish0612` made enquiries on a possible online LLM with Mistral API, he was guided towards the **pplx-7b-chat** by `@icelavaman`. Channel sorting was also a point of discussion as users were redirected to appropriate channels for distinct topics.

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (82 messages🔥🔥): 
        
- **Choosing the Right AI Model**: `@prolapsedgod` posed a question about which model to use, indicating that they find it tough to decide between the Perplexity Experimental model and the GPT-4 model. `@hynjia` joined the conversation and noted that they usually switch between GPT-4 and Claude 2.1.
- **Query about Mixtral parameters**: User `@.hackit` raised a question to understand the parameters of `mixtral-8x7b-instruct` attempting to recreate it with the rest-API. 
- **Interaction about 'HOLIDAYS23' Pro Membership Coupon**: A discussion about applying the 'HOLIDAYS23' Pro membership coupon code took place, with `@nightpearl62`, `@icelavaman`, `@jonathanonymous`, `@ok.alex`, `@danielagmz888`, and `@ashley__9910` participating. Users raised issues about the link's functionality and applying the code, with replies suggesting methods like subscribing via Plexity.ai, trying via mobile web, and reaching out to support.
- **Inherent Benefits and Questions about Image Generation in Perplexity**: `@arafay` and `@archient` had an insightful discussion around the usage and benefits of image generation in Perplexity. They reflected on its visual engagement aspects and its potential as a creative stimulant.
- **Interest in Incorporating the 'Mistral-Medium' Model for Pro Accounts**: `@Rehnn` suggested the inclusion of the Mistral-Medium model in Pro accounts based on their tests in the playground, remarking on its speed and decreased hallucinations. `@reflext` agreed with the idea.  
- **Payment Challenges for European User**: `@ashley__9910` from Germany faced challenges with payment methods with the advice to try using GPay on an Android phone not working. `@icelavaman` recommended reaching out to support@perplexity.ai for assistance regarding the Stripe related issue.

**Links mentioned**:

[Perplexity Pro](https://pplx.ai/holidays): Enjoy 2 free months of Perplexity Pro or $40 off o...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (7 messages): 
        
- **Perplexity AI Collects API and User Information**: `@maxxflyer` pointed to the [FAQ](https://docs.perplexity.ai/page/frequently-asked-questions) of Perplexity AI, which says that the organization collects information related to API usage data and User's account information such as name, email address, etc.
- **Perplexity's pplx-7b-online and pplx-70b-online Models Use Public Data**: `@maxxflyer` mentioned that the pplx-7b-online and pplx-70b-online models of Perplexity use public data and data from their search index, and referred to a [Perplexity blog post](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) for more information.
- **Faulty Discord Invite on Perplexity Page**: `@maxxflyer` pointed out that the Discord invite on Perplexity's page is not working. 
- **Jpa's Curiosity about Military Boarding History**: `@jpa` asked a question about the history of why the military boards first on aircraft and provided a link to [Perplexity's search](https://www.perplexity.ai/search/whats-the-history-2Hi46dUBTIayi4HlvrozmQ?s=c) for more details.
- **Channel Sorting Discussion**: `@icelavaman` told `@maxxflyer` to use a different channel, and `@ok.alex` guided `@504076886268313616` and `@1179797683167309926` towards the appropriate channel for their respective discussions.

**Links mentioned**:

[Frequently Asked Questions](https://docs.perplexity.ai/page/frequently-asked-questions)


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (3 messages): 
        
- **Model Organization Suggestion**: `@paul16307` proposed to **sort the Mistral models** in the dropdown from the least powerful to the most powerful ones.
- **Query on Online LLM with Mistral API**: `@monish0612` questioned if there's Mistral API available with online LLM, to which `@icelavaman` responded affirmatively providing the name "**pplx-7b-chat**".


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Google's T2I versus DALL·E-3: A Showdown**: Google's T2I tool elicited mixed reviews. Some users lauded its fidelity and lack of overfitting, while others criticized its over-censorship and less prompt adherence. Reddit links were shared for more insights, and users suggested creating shared prompts from the guild. *"[Google Lab Experiments looks far superior to...](https://labs.google/)*"
   
- **AI Training Copyright under Japanese Law**: The group examined an article discussing Japan's copyright laws which currently permit unrestricted AI training with legal datasets. However, caution was sounded by members regarding ongoing lobbying which may alter that landscape. *"[Japan goes all in, copyright doesn't apply to AI training...](https://www.biia.com/japan-goes-all-in-copyright-doesnt-apply-to-ai-training/)*"

- **Wildcard Concepts for a Bigger, Better Dataset**: A new methodology was suggested, involving each existing CLIP keyword for synthesizing a large data set. The proposed concept envisioned utilization of Language Models for merging varied concepts.
  
- **SynCLR: The Future is Synthetic**: A research called [SynCLR](https://fxtwitter.com/_akhaliq/status/1741668076037247127)t, which exclusively utilizes synthetic images and captions for visual representation learning without using real data, was shared and recognized as a potential realization of the wildcard concept idea.

- **A New Wave in High-Quality Text Embeddings**: Members shared and discussed an [arxiv paper](https://arxiv.org/abs/2401.00368) outlining a method to achieve high-quality text embeddings with synthetic data, employing LLMs to create a range of synthetic data and to fine-tune open-source decoder-only LLMs. 

- **Performance of MSE in Classifier-free Guidance**: Mean Squared Error was recognized to perform better in Classifier-free Guidance. Other models, such as training with perceptual losses, were assessed as costlier. 
   
- **Self-perceptual Objective versus Diffusion Models**: A thoughtful debate was held on using pretrained models as feature extractors for a perceptual loss, orbiting a [paper](https://arxiv.org/abs/2401.00110) proposing a self-perceptual objective, capable of producing more realistic images, yet decreasing performance upon repeated iterations.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (49 messages🔥): 
        
- **Google's Text-to-Image Platform Stuns**: User `@SegmentationFault` initially raved over Google's new T2I tool [Google Lab Experiments](https://labs.google/), comparing it favorably over DALL·E-3 in terms of image fidelity and lack of overfitting. However, users including `@thejonasbrothers` and `@nodja` pointed out its limitations, including less adherence to prompts than DALL·E-3 and blocking certain types of content including certain human-related images.
    
- **Sharing Generated Images**: `@nodja` brought up the idea of generating shared prompts and users suggested generating prompts from a pre-determined channel for comparison. Additional images were shared as users tested the system.
    
- **Restrictive Blockage in Image Generation**: Google's T2I platform reportedly blocked certain types of content at either the prompt level or image level, particularly around topics related to humans, according to `@nodja`. User `.undeleted` criticized this decision, questioning the value of the system if it refuses even to generate standard content.
    
- **Sharing T2I Insights**: `@SegmentationFault` recommended checking out /r/dalle2 and /r/weirddalle on Reddit for more posts regarding T2I and provided a link to [r/ImagenAI](https://www.reddit.com/r/ImagenAI/), even though it was less active at the time.
    
- **AI Training Copyright in Japan**: User `@vrus0188` shared an article about Japan's stance on copyright and AI training and its potential impact on machine translation. `@peacekeeper8310` clarified that this stance stemmed from a 2018 law allowing for unrestricted training of ML models with legal datasets but cautioned that this could change due to ongoing lobbying efforts.

**Links mentioned**:

- [Japan Goes All In: Copyright Doesn’t Apply To AI Training | BIIA.com | Business Information Industry Association](https://www.biia.com/japan-goes-all-in-copyright-doesnt-apply-to-ai-training/)
- [Tweet from Zumer (@zumersultana)](https://vxtwitter.com/zumersultana/status/1734530458359349500): Google can now generate high-quality AI images.  A...
- [Try experiments in Search Labs](https://www.google.com/search/images/editor/mNJJecqS)


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (34 messages🔥): 
        
- **Synthesizing CLIP dataset with wildcard concepts**: `@SegmentationFault` suggested the idea of synthesizing a large dataset using each existing CLIP keyword. The proposed concept involved utilizing Language Models (LLMs) for merging different concepts.
- **SynCLR as potential realization of segmentationfault's idea**: `@rom1504` shared a research by `_Akhaliq` called [SynCLR](https://fxtwitter.com/_akhaliq/status/1741668076037247127), that learns visual representations exclusively from synthetic images and synthetic captions, without any real data utilization.
- **Exploration of high-quality text embeddings with synthetic data**: `@thejonasbrothers` and `@mkaic` discussed the simple but effective method of obtaining high-quality text embeddings with synthetic data and less than 1k training steps that was outlined in an [arxiv paper](https://arxiv.org/abs/2401.00368). The method leverages LLMs to produce diverse synthetic data and fine tune open-source decoder-only LLMs.
- **Discussion on perceptual loss in diffusion models**: `@nodja`, `@mkaic`, and `@clock.work_` had an engaging discussion on using pretrained models as feature extractors for their own perceptual loss. Their discussion orbited around a [paper](https://arxiv.org/abs/2401.00110) proposing a self-perceptual objective that could generate more realistic images, though iterating this method seems to decrease performance.
- **MSE performs better with Classifier-free Guidance**: `@thejonasbrothers` pointed out that Mean Squared Error (MSE) works better with Classifier-free Guidance (CFG). Other possibilities, such as training with perceptual losses, were considered to be more expensive.

**Links mentioned**:

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368): In this paper, we introduce a novel and simple met...
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1741668076037247127?t=83X-PzqSIMTQVncqyg-LDw&s=19): Learning Vision from Models Rivals Learning Vision...
- [Diffusion Model with Perceptual Loss](https://arxiv.org/abs/2401.00110): Diffusion models trained with mean squared error l...
- [Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17090): The explosion of visual content available online u...


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **"Old School Cool in Pascal"**: A humorous exchange about using old-school Pascal language featured a [gist link](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1) to an OpenAI unit written in Pascal shared by @psipiai.
- **"Troubleshooting in Action with LMStudio"**: Technical discussions revolving around LMStudio's performance, with @malte0621 experiencing significant slowdowns with Vision models which were resolved by managing VRAM spillover and properly handling CPU threads.
- **"Decode the Models!"**: Curious queries related to differences between specific LMStudio models, namely, dolphin-2.6-mistral-7b-dpo and dolphin-2.5-mixtral-8x7b, and how to load Hugging Face models. Replying to these, @fabguy clarified that only GGUF format models are compatible with LMStudio and provided hardware requirement advice for the Mixtral model. 
- **"The Code Translator Conundrum"**: @xenorhon initiated a question about ideal models for code translation tasks. The conversation culminated with a suggestion for the tinyllama model by @usizu, following @fabguy's advice on combining traditional methods with LLMs for optimal outcomes.
- **"Taming the Tiny Llama"**: Model-related issues arose in the models-discussion-chat, with @macaulj struggling with the unruly Tiny Llama 2 model. Suggestions included trying out the [Mixtral 8X7B model](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF), and recommended models for memory-constrained setups like Mistral 7B Q4 or Zephyr Beta.
- **"Feedback Channel Police"**: User @heyitsyorkie providing a clear reminder of the Discord channel rules, emphasizing the proper use of the feedback channel for discussions related to LMStudio.
- **"Hardware Spec Talks and Benchmark Shares"**: A lively hardware discussion featuring an assessment of a specific eBay hardware configuration by @leviticus_slow and advice on GPU alternatives by @nixthefolf. Additionally, speculation on Nvidia's future plans by @kess4747 and sharing benchmarks by @discgolfvalife. New user @dantefortal requested assistance with a TinyLlama installation error.

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (52 messages🔥): 
        
- **OpenAI Unit in Pascal**: @psipiai shared a [gist link](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1) featuring OpenAI Unit in Pascal. The post joked about the use of the old-school Pascal language, implying even vintage coding has its moments of glory.
- **LMStudio Performance Inquiry**: @malte0621 reported a significant slowdown when using Vision models on LMStudio. Following a detailed discussion with @fabguy, the issue was resolved by managing VRAM spillover and handling CPU threads correctly.
- **Mixing Models in LMStudio**: @vanthryn was curious about the output quality and capabilities difference between **dolphin-2.6-mistral-7b-dpo** and **dolphin-2.5-mixtral-8x7b** models, with the aim to decide between integral loading of Mistral 7b or partial loading of Mixtral 8x7b onto GPU memory. @heyitsyorkie warned against using Mixtral unless one has requisite hardware specs (at least 20GB vram).
- **Loading Hugging Face Models into LMStudio**: @usizu queried loading hugging face models into LMStudio and wanting to try the OpenPipe/mistral-ft-optimized-1227 model. @fabguy clarified that only models in GGUF format are compatible with LMStudio. Furthermore, typing the model name directly can bypass the URL filter for non-compatible models.
- **Code Translation Suggestions**: @xenorhon sparked a discussion seeking recommendations for models to perform code translation tasks. @fabguy advised that best results come from combining traditional methods like parsers and grammars with LLMs, rather than using only LLMs. The discussion ended with @usizu suggesting the tinyllama model for consideration.

**Links mentioned**:

- [OpenPipe/mistral-ft-optimized-1227 · Hugging Face](https://huggingface.co/OpenPipe/mistral-ft-optimized-1227)
- [Snoop Dogg Dre GIF - Snoop Dogg Dre Riding - Discover &amp; Share GIFs](https://tenor.com/view/snoop-dogg-dre-riding-driving-rolling-gif-13800248): Click to view the GIF
- [OPEN AI Unit. in Pascal. Also works for other compatible stuff like LMstudio, Firemonkey cross platfrom test app attached](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1): OPEN AI Unit. in Pascal. Also works for other comp...
- [TheBloke/mistral-ft-optimized-1227-GGUF · Hugging Face](https://huggingface.co/TheBloke/mistral-ft-optimized-1227-GGUF)


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (9 messages🔥): 
        
- **Trouble taming Tiny Llama 2**: `@macaulj` reported having issues with the **Tiny Llama 2** model, saying it's generating random stuff and not listening to inputs. 
- **Recommended Models**: `@kess4747` advised `@macaulj` to try out the [Mixtral 8X7B Instruct v0.1 model by Mistral AI](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF), claiming to have gotten better results from it. 
- **Memory limits for models**: `@macaulj` states they only have **8 or 16 GB of memory** to work with, which `@fabguy` responded to by suggesting the **Mistral 7B Q4** or **Zephyr Beta** models, purported to deliver adequate performance within this memory limit.
- **Deepseek Chat template query**: `@nvn.osto` enquired about the recommended template for **Deepseek Chat 67B**. According to `@heyitsyorkie`, the default lm studio preset works fine with these models.

**Links mentioned**:

[TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (1 messages): 
        
- **A Clear Reminder of Channel Rules**: User `@heyitsyorkie` reminded the community that this channel, 🧠-feedback, is specifically for discussing feedback related to LMStudio. For other discussions, appropriate channels include `<#1111440136287297637>`, `<#1128339362015346749>`, `<#1111649100518133842>`, and `<#1139405564586229810>` for bugs and `<#1185646847721742336>` and `<#1110598183144399061>` for model advice and chat respectively.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (10 messages🔥): 
        
- **Hardware Spec Discussion**: `@leviticus_slow` shared hardware details from an eBay listing and asked if dual Tesla M10s in a PowerEdge R730 with dual E5-2697v3s was a good purchase. `@nixthefolf` responded that the M10s were not the best choice due to their age and lower performance, suggesting a more current Tesla P40 or a used 3090 as alternatives.
- **Speculating on Nvidia's Future Plans**: `@kess4747` expressed hope for the return of nvlink in future Nvidia graphic card models, such as the speculated 5090.
- **Benchmarks Shared**: `@discgolfvalife` shared detailed specifications and benchmark results for a variety of AI models he tested on his ASUS G17 laptop using LM Studio V 0.2.10. He also shared a detailed test prompt and asked for tips or advice. `@doderlein` responded with the results from a test he ran on two OpenChat models using the same test prompt.
- **Trouble With TinyLlama**: New user `@dantefortal` encountered an issue while attempting to install TinyLlama, sharing an error message that stated his processor did not support AVX2 instructions, and asked for assistance.

**Links mentioned**:

[Nvidia Tesla M10  32GB  180-12405-DAAB-A01 GDDR5 GPU  | eBay](https://www.ebay.com/itm/145495209710)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mixing it up with Mixtral**: Users express disappointment with Mixtral's output being more censored than expected. The identity of Mistral models, such as "mistral-tiny", is revealed and suggestions for clearer documentation are shared. Functional Python code is offered for using the Mistral API. Last but not least, the feasibility of running models on a Raspberry Pi sparks curiosity.
- **Docker & Memory, A Tragic Love Story**: The frequent CUDA out-of-memory issue rears its ugly head again, even with a powerful machine. A need for detailed instructions for running Mistral locally surfaces. The mystery of the necessary API URL is unsolved. Raspberry Pi makes another appearance as the subject of feasibility discussions for running AI models.
- **Text Cleaning Drama**: Users express the need for robust text cleaning solutions, especially for large and messy corpuses like scanned books. The limitations of out-of-box solutions are discussed, with the suggestion of tailored ETL or LT operations. Recommended libraries like pandas, opencv, and PIL bring hope.
- **Channel Contact Conundrum**: Users query about updates and seek guidance on contacting maintainers, but the conversation lacks context to provide further insight.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (50 messages🔥): 
        
- **Mixtral not as uncensored as expected**: Users `@meyelo` and `@hexdigest` discussed their experiences with the output of Mixtral, expressing disappointment that it was not as uncensored as they initially believed. They reported getting "do not break TOS" messages and unable to get responses to certain prompts even when using the API with safe mode turned off.
- **Revealing the true identities of Mistral models**: In response to questions from `@juliandarley`, `@i_am_dom` clarified that Mistral 7b is referred to as "mistral-tiny" and Mixtral is "mistral-small" in the API. The community seemed to share a sentiment that clearer naming in the documentation could reduce confusion.
- **Practical code for using Mistral API**: In response to a request from `@refik0727`, `@i_am_dom` shared functional Python code that uses the Mistral API to chat with chosen models like "mistral-medium", showing both non-streaming and streaming responses.
- **How to tell Mistral to follow specific rules**: `@unknownperson2156` asked for advice on instructing Mistral models to follow specific rules to ensure shorter generated texts and rule compliance. Suggestions given by `@hexdigest` included simplifying instructions and reducing the temperature parameter.
- **AI on Raspberry Pi feasibility**: `@kaigenji` inquired about the feasibility of running Mistral models on a Raspberry Pi or similar Single Board Computer for IoT purposes, but was redirected to a different channel by `@hexdigest`.


**Links mentioned**:

- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#embeddings): We provide client codes in both Python and Javascr...
- [metal : optimize ggml_mul_mat_id (faster Mixtral PP) by ggerganov · Pull Request #4725 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4725): Add new mat-mat kernel for ggml_mul_mat_id which f...


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (1 messages): 
        
cavoi9205: /im


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (8 messages🔥): 
        
- **CUDA Out of Memory Issue**: `@pastillafit` experienced a CUDA out-of-memory error when trying to run a Docker command for **Mistral**. Despite having a machine with GeForce 4090, AMD 7950X3D, and 64GB RAM, they received an error indicating the GPU capacity was exhausted. User `@hanschrs` informed that the user's VRAM at 24GB isn't enough and also suggested using the quantized version via ollama.
- **Detailed Process Needed to Run Mistral Locally**: `@kartik.07` is looking for instructions on how to run **Mistral** locally on a computer, specifically through the command line, without relying on ML Studio or similar software. 
- **API URL Query**: `@macadelic4982` asked for the API URL needed to use in a chat backend.
- **Feasibility Query about Running on Raspberry Pi**: `@kaigenji` queried about the feasibility of running models on a Raspberry Pi for an IoT assistant project. They required a light enough model that won't overheat Raspberry Pis or similar single-board computers (SBCs).


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **In Search of Text Cleaning Solutions**: `@_b_r.` is looking for a solid pipeline or script for text cleaning and pre-processing, particularly for a corpus of scanned books with lots of page numbers and typos.
- **Custom ETL May Be Required**: `@duck` suggests that many off-the-shelf scripts might not be sufficient for the task, indicating that there could be a need for tailored ETL or LT operations.
- **Tools for Cleaning Structured Files**: `@duck` recommends libraries like pandas for cleaning more straightforward formats like `.csv` and `.txt` files.
- **Use Google for Finding Tools**: `@duck` advises `_b_r.` to Google phrases like 'python clean unstructured pdf's' to find tools that others might be using to clean that type of data/format.
- **Image-to-Text Libraries**: `@duck` brings up opencv and PIL as potential libraries for converting image to text, which might be relevant when dealing with unstructured data from books.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **Query about an update**: User `@jakobdylanc` asked about the estimated time of arrival (ETA) on **two unspecified changes**.
- **Seeking contact with maintainer**: `@carloszela` wanted to know if **this channel** is the right place to **contact a maintainer**.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Eleuther Celebrates the Year's Victories**: Eleuther's achievements over the past year, including its official establishment, were celebrated and announced by `@stellaathena` in the `#announcements` channel. New minor announcement and social events role tags were also implemented. A link to the **[community survey](https://forms.gle/tZeCqaNQ4dguH2Li7)** was shared. 
- **Code Clipping and Model Juggling**: In the `#general` channel, `@nate.dawgg` promoted a [VSCode extension for code suggestion](https://github.com/CodedotAl/code-clippy-vscode) which enhances local code development. Discussion on potential interleavings between self-attention and SSM blocks, the order of checkpoints in Pythia Model training, and the merits and demerits of using the Lion optimizer was brought up. Specifically, user `@ricklius` mentioned an [optimizer paper](https://arxiv.org/pdf/2202.04329.pdf) for reference. 
- **From Publishing Mysteries to New Optimizers**: In the `#research` channel, potential journals to publish LLM bias characterization studies were suggested, a performance question between Mistral 7b and GPT2 300m was addressed, and MAMBA's benefits were dissected. In terms of other inquiries, there were also recommendations for frameworks to run large data inferencing, such as **DeepSpeed Inference**, and guidance for new server members like `@rabiussany`.
- **Revamping the Evaluation System**: The `#lm-thunderdome` channel manufactured a dialogue on an [issue regarding More Flexible Answer Extraction Code](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159) where `@hailey_schoelkopf` argued for a possible two-score system for evaluative tasks, thus opening the conversation to the larger community.
- **Illuminating Image Inference with DiffusionLight**: In the `#multimodal-general` channel, `@supasornae` flashed light on the introduction of **DiffusionLight**, a lighting estimation technique, along with the [paper](https://arxiv.org/abs/2312.09168), [official page](https://diffusionlight.github.io/), and the [Hugging Face Model](https://huggingface.co/DiffusionLight/DiffusionLight) for further reference.

**Eleuther Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/) (1 messages): 
        
- **Ring in the New Year with Eleuther**: The `@everyone` announcement by `@stellaathena` celebrated Eleuther's accomplishments in the past year, including **becoming a legal entity, writing over 40 papers, and advising world leaders on AI regulation**.

- **Expanded Role Tags**: `@stellaathena` notes that while `@everyone` is used for major announcements, there are other specific **tag roles** for minor announcements and social events. Some of these roles were accidentally deleted and thus removed from members - `@stellaathena` prompted members to double check if they are still subscribed to the roles they want.

- **Change of Role Name**: To avoid confusion, the `looking-for-work` role has been renamed to <@&1051750303843749929>.

- **Channel Shuffling and Addition**: A series of channel updates has been announced which includes moving channel `<#795089627089862656>` under the Other Modalities heading, creation of two new channels `<#1181279677004910662>` and `<#1179031436704096327>` under the same heading, and generalization of the "math-and-ai" channel to `<#1110611369574793306>`.

- **Community Survey and Newsletter**: `@stellaathena` shared a link to **[their semi-annual community survey](https://forms.gle/tZeCqaNQ4dguH2Li7)** and announced the start of a quarterly newsletter that summarizes Eleuther's work. Members can sign up for the newsletter via the community survey.


### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (20 messages🔥): 
        
- **Nate Dawgg's VSCode Extension for Code Suggestion**: User `@nate.dawgg` shared a [link](https://github.com/CodedotAl/code-clippy-vscode) to a GitHub project for a VSCode extension that supports faux pilot for local model serving.
- **Possible Interleaving in Self-Attention and SSM blocks**: `@thooton` suggested the interleaving of self-attention and SSM blocks, similar to striped hyena operations.
- **Order of Checkpoints in Pythia Model Training**: `@stellaathena` clarified to `@wolferk` that in Pythia model training, checkpoint numbers like `step11000` are later than lower numbers like `step10900`. The confusion may arise because HuggingFace's UI displays them in alphabetical order.
- **Discussion on the Lion Optimizer**: Users `@sentialx` and `@ravnabergsndot` discussed the advantages and disadvantages of using the Lion optimizer. It was noted that while Lion optimizer might potentially affect 8-bit quantization, it converges about as well as Adam and may save some memory. A link to the [Sophia optimizer paper](https://arxiv.org/pdf/2305.14342.pdf) was also referenced by `@ricklius`.
- **Variable Performance of Lion Optimizer in LLMs and ViTs**: `@frazermc` shared personal experience of testing Lion optimizer, stating that while performance in LLMs was undetermined, it worked really well for ViT and potentially many encoder models like Bert.

**Links mentioned**:

[GitHub - CodedotAl/code-clippy-vscode: VSCode extension for code suggestion](https://github.com/CodedotAl/code-clippy-vscode): VSCode extension for code suggestion. Contribute t...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (31 messages🔥): 
        
- **Seeking for Suitable Journals for LLM Bias Research**: User `@bluerune` is finishing up a paper on **LLM bias characterization** and sought suggestions on which journals to consider for publishing. `@stellaathena` recommends **NLP venues, ACL, ICLR, COLT, and AISTATS**.
- **Performance Comparison between Mistral 7b and GPT2 300m**: `@sehaj.dxstiny` raised a question about whether **Mistral 7b** would outperform **GPT2 300m** upon fine-tuning with a slightly smaller dataset. `@thooton` responded, stating that bigger models are generally better and more sample-efficient than smaller models, sharing his experience of achieving 1.2-1.3 validation loss with Mistral fine-tuning.
- **Understanding MAMBA's Breakthrough**: `@swaystar123` asked about the breakthrough with MAMBA, `@_inox` explained that MAMBA allows for O(L) versus O(L^2), where L is the length of the sequence. However, `@salmon_lemon` asked whether the perplexity performance drops as L increases, with `_inox` clarifying it's not guaranteed to decrease or increase.
- **Seeking Offline Model Inference Framework Suggestions**: `@alexspangher` asked for recommendations on the preferred framework for running offline model inference on large amounts of data. Several were tried including Ray, huggingface datasets, huggingface accelerate, and pyspark/sparknlp. Stellaathena recommended **DeepSpeed Inference**.
- **Welcome and Engagement Guidance for New Member**: `@rabiussany`, a new member of the server expressed his interest in participating in research. `@alexanderrgriffing` suggested that newbies should explore channels, find projects they like and send pull requests on the GitHub of these projects. `@hailey_schoelkopf` advised checking out the server's threads for ongoing projects and discussions.


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (1 messages): 
        
- **Flexible Answer Extraction Code Issue Discussed**: @hailey_schoelkopf shared an issue from GitHub titled ["More Flexible Answer Extraction Code"](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159). The matter revolves around the current evaluation system which discredits answers that are correct but not in the required format. The issue creates a bias for models that are trained to mimic the benchmark formatting, thus disadvantaging equally capable models with nonstandard formatting.
- **Proposed Solution**: @hailey_schoelkopf suggested a solution to have two evaluation scores for every relevant generative task. The first one (`acc` with `strict` postprocessing/extraction) would follow the original design of the task with precise formatting. The second one (`acc` with `loose/flexible` postprocessing) would attempt to extract the correct answer from the model's generation irrespective of the format. This solution aims to address complaints about tasks where different evaluation frameworks report varying scores due to their flexible answer extraction methodologies. The user has asked for feedback on this proposed solution from the community.

**Links mentioned**:

[More Flexible Answer Extraction Code · Issue #1159 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159): In LM Evaluation Harness, we work to match the &qu...


### ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/) (1 messages): 
        
- **Introducing DiffusionLight**: User `@supasornaek` shared a [link](https://fxtwitter.com/supasornaek/status/1742223785288433708?t=ADE_KEfR4K23NHsxB5qrYA&s=19) to their introduction of **DiffusionLight**, a technique to estimate lighting from in-the-wild input images by inpainting a chrome ball into the image with diffusion models.
- **Detailed look into DiffusionLight**: User `@supasornaek` also provided [links](https://arxiv.org/abs/2312.09168) to the DiffusionLight paper, its [official page](https://diffusionlight.github.io/), and [Hugging Face model](https://huggingface.co/DiffusionLight/DiffusionLight).

**Links mentioned**:

[Tweet from Supasorn Suwajanakorn (@supasornaek)](https://fxtwitter.com/supasornaek/status/1742223785288433708?t=ADE_KEfR4K23NHsxB5qrYA&s=19): Introducing DiffusionLight---a simple yet effectiv...


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Splitting Splitters, A Revolt**: Issues arose with the `RecursiveCharacterTextSplitter` from the LangChain's text splitter module, causing users grief and cutting words in the middle. The [sample code](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py) was run by user `@offer.l` with some suggested alternatives such as `NLTKTextSplitter` from `@hasan_34148`.
- **Missing in Action, the load_qa_chain**: User `@jupyter1310` sought missing API documentation for the elusive LangChain's `load_qa_chain` module. 
- **GeneratorChainOptions, the Unwanted**: User `@menny9762` reported a strange, unwanted output from `ConversationalRetrievalQAChain`. The output generated was `generatorChainOptions`, not the expected response from the LangChain API. Talk about a conversation stopper!
- **LangChain Secret Weapon, not Secretive Enough**: Assistance required on large text summarization using `loadSummarizationChain` in LangChain from user `@agentcole`. I guess it's not a secret weapon if you have to ask about it in a public channel!
- **Alttexter Steals the Show**:  `@jonathan_09689` shared links to GitHub projects [alttexter-ghclient](https://github.com/jonathanalgar/alttexter-ghclient) and [alttexter](https://github.com/jonathanalgar/alttexter). These projects are utilized as a service to generate 'alt' text for images, strengthening the connection between code and image content.
- **Alttexter Makes an Entrance, Enhances Langchain Repo**: The proof is in the pudding! `@jonathan_09689` showcased Alttexter's value by running it on [langchain-ai/langchain repo](https://github.com/langchain-ai/langchain/pull/15357/files) to update alt text across documentation. Clean up on documentation aisle 4!



**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (27 messages🔥): 
        
- **Text Splitter Troubles**: User `@offer.l` had some issues with the `RecursiveCharacterTextSplitter` from the LangChain's text splitter module, noting that it doesn't 'down escalate' from double newline to newline, space, etc, thus cutting words in the middle and producing meaningless splits. They shared the [sample code](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py) they were using. `@hasan_34148` suggested using `NLTKTextSplitter` instead, adding that while he doesn't see `RecursiveCharacterSplitter` as faulty, it tends to cut words in the middle. `@seththunder` chimed in, mentioning that they've never had any issue using `RecursiveCharacterSplitter`.
- **Quest for load_qa_chain documentation**: User `@jupyter1310` sought help finding the API documentation for LangChain's `load_qa_chain` module, a class which they were unable to locate in the API docs but is commonly imported in various pages in the LangChain documentation.
- **Issues with ConversationalRetrievalQAChain**: `@menny9762` reported an issue with `ConversationalRetrievalQAChain`, where the return output was the generatorChainOptions instead of the real response. They even provided a code snippet to explain the issue in detail.
- **Summarizing Large Text in LangChain**: User `@agentcole` sought advice to lengthen the output text length when using `loadSummarizationChain` for summarizing large text. `@jupyter1310` suggested enforcing a minimum word count in the prompts. But `@agentcole` found the workaround of using previous steps in chunks for the summary still inconvenient.
- **LangChain Citation**: User `@thenextmz` wanted to know if there was an official paper about LangChain that they could cite in their Master's Thesis.

**Links mentioned**:

[text-split-explorer/splitter.py at main · langchain-ai/text-split-explorer](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py): Contribute to langchain-ai/text-split-explorer dev...


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Alttexter: GitHub Action Helps Create Alt Text for Images**: User `@jonathan_09689` shared links to two of his projects on GitHub - [alttexter-ghclient](https://github.com/jonathanalgar/alttexter-ghclient) and [alttexter](https://github.com/jonathanalgar/alttexter). Both projects function as a wrapper service for `gpt4-vision-preview` that can batch generate alternative text and title attributes for images defined in markdown formatted files. 
- **Alttexter's Application in Langchain Repo**: For demonstration purposes, `@jonathan_09689` ran Alttexter over [langchain-ai/langchain repo](https://github.com/langchain-ai/langchain/pull/15357/files) to showcase its value in updating alt text and title attributes across documentation.

**Links mentioned**:

- [GitHub - jonathanalgar/alttexter-ghclient: Containerized GitHub action for interacting with the alttexter service (gpt4-vision-preview wrapper)](https://github.com/jonathanalgar/alttexter-ghclient): Containerized GitHub action for interacting with t...
- [GitHub - jonathanalgar/alttexter: gpt4-vision-preview wrapper service to batch generate alternative (&#39;alt&#39;) text and title attributes for images defined in markdown formatted files.](https://github.com/jonathanalgar/alttexter): gpt4-vision-preview wrapper service to batch gener...
- [Batch update of alt text and title attributes across documentation by jonathanalgar · Pull Request #15357 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/15357/files): (Actions ran in jonathanalgar#3. Links to LangSmit...


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **LLM Performance Wrestling Match**: Dissection of how different-sized Language Learning Models (LLMs), like **Mistral 7b** and **GPT2 300m**, perform after fine tuning on smaller datasets, initiated by `@sehaj.dxstiny`.
- **Fine Tuning Superpowers**: User `@thebaghdaddy` underscoring the potential of smaller models to surpass bigger counterparts, like **GPT4**, in task-specific performance through fine tuning.
- **Tasks Change the Game**: `@pantsforbirds`'s anecdote exemplified task dependency in model performance, remarking that some models, **like Mistral 7b**, were easier to fine tune than others given the same setup.
- **Anyscale, Mistral's Partner in Crime**: Praise for **Anyscale**'s impact on the speed and consistency of **Mistral** model, as vaunted by `@robotums`.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (10 messages🔥): 
        
- **LLM Performance Debate**: User `@sehaj.dxstiny` sparked a discussion about whether a significantly larger language learning model (LLM) like **Mistral 7b** would perform better after fine tuning on a slightly smaller dataset, compared to a smaller model like **GPT2 300m** trained from scratch.
- **Power of Fine Tuning**: `@thebaghdaddy` highlighted the ability of smaller models to excel past even **GPT4** at specific tasks through fine tuning, though they perform worse on general benchmarks.
- **Essential Role of Tasks and Fine Tuning**: User `@pantsforbirds` shared their experiences, suggesting that task dependency plays a major role. They found some models, **like Mistral 7b**, to be easier to fine tune compared to other similar models, using the same setup.


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (3 messages): 
        
- **Anyscale: A Powerhouse for Mistral**: `@robotums` states **Anyscale** significantly increases the speed and consistency of their **Mistral** model when compared to **OpenAI's** deployment.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Mixing Things Up with Mixtral**: Insights into gate vectors, frankensteins, and the "punching above weight class" performance of the AI model *Mixtral*, from [@_jp1_](https://discord.com/channels/1178995845727785010/1182759434326396998/)'s shared [blog post](https://goddard.blog/posts/clown-moe/).
- **Axolotl Craze and Configurations**: A general increase in Axolotl's popularity exemplified by `@philipmay`, `@le_mess`'s inquiry into fast Fourier Transform (FFT) Axolotl YAML config for Mixtral, and `@hammadkhan`'s provided solution using [Axolotl YAML configuration](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml).
- **Text Embedding Gets Synthetic Makeover**: [Research paper](https://arxiv.org/abs/2401.00368) shared by `@bjoernp` introduces efficient, synthetic data-based text embeddings model claiming low training steps and potential compatibility with German *leo-mistral*.
- **Riding the Wave of End-to-End Training**: `@rasdani` proposes the benefits of applying end-to-end training taken from Computer Vision in autonomous driving to large language models.
- **Digging Deeper into Embeddings**: Conversation on model performance scaling initiated by `@casper_ai`, data efficiency and pretrained models brought up by `@bjoernp`, and a shared tool, [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference), for fast inference solution for text embeddings by `@le_mess`.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (5 messages): 
        
- **Experiments with Mixtral**: User `@_jp1_` shared a [blog post](https://goddard.blog/posts/clown-moe/) about some experiments using prompts to determine gate vectors and frankensteins by undi in the context of Mixtral, an AI model by Mistral. The post praises the use of a set of eight "experts" in Mixtral's architecture, claiming that the model "punches way above its weight class".
- **Twitter Post Highlighted**: `@sebastian.bodza` highlighted an [interesting Twitter post](https://twitter.com/morgymcg/status/1741819937641910537); however, the contents of the post are left undisclosed in this message.
- **Popularity of Axolotl**: `@philipmay` made a remark about the increasing popularity of **Axolotl**, a model "tuning" tool.
- **Axolotl YAML Config Inquiry**: `@le_mess` asked if anyone had an Axolotl YAML config for Fast Fourier Transform (FFT) for Mixtral.
- **Possible Solution Provided for YAML Config**: In response to `@le_mess`, `@hammadkhan` suggested converting an existing [Axolotl YAML configuration](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml) for use with Mixtral.

**Links mentioned**:

- [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe/)
- [axolotl/examples/llama-2/fft_optimized.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml): Go ahead and axolotl questions. Contribute to Open...


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (7 messages): 
        
- **Synthetic Data for Effective Text Embeddings**: `@bjoernp` shared a [research paper](https://arxiv.org/abs/2401.00368) introducing a novel method for obtaining high-quality text embeddings using synthetic data and less than 1k training steps. The method does not rely on complex training pipelines and could be used to train a **leo-mistral** with German synthetic data.
- **LLMs and End-to-End Training**: `@rasdani` stated that using large language models (LLMs) for each part of an application pipeline could be beneficial, drawing parallels to how Computer Vision in autonomous driving is shifting towards end-to-end training.
- **Checking Model Performance Scaling**: `@casper_ai` raised a question on how the performance of other models might improve if they were scaled up 10 times, in response to the larger size of the discussed text embedding model.
- **Model Training Efficiency**: `@bjoernp` pointed out that the key takeaway from the discussed research is about data efficiency and the ability to harness the strengths of pretrained models.
- **Generating Embedding Data**: `@le_mess` noted their similar experience in generating an embedding dataset, where similar and dissimilar sentences were produced instead of queries. They also introduced the [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) tool that provides a fast inference solution for text embeddings models.

**Links mentioned**:

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368): In this paper, we introduce a novel and simple met...
- [GitHub - huggingface/text-embeddings-inference: A blazing fast inference solution for text embeddings models](https://github.com/huggingface/text-embeddings-inference): A blazing fast inference solution for text embeddi...


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Deciphering Inference Settings**: Lingering confusion over **inference settings** was alleviated when `@fred_fups` confided to `@rusch` that their repetition penalty is set to `1`.
- **Unexpected Tricky Prompt Formats**: A puzzling issue with **prompt formatting** led `@fred_fups` to undertake retraining, with *ChatML* coming to the rescue and effectively resolving the problem.
- **Model Training Feud**: The seemingly straightforward task of training a **Mistral 7B model** on a sample size of `350` ruffled some feathers when the model failed to adhere to the text formatting norms (*paragraph breaks and keyword highlighting*).
- **Model Performance Gets a Thumbs Down**: `@fred_fups` left the community in shock when they announced that their **together.ai** trained model struggled to deliver, merely mirroring the input without any significant formatting alterations.
- **The Alluring Promise of Axolotl**: In the wake of continued disappointments, `@fred_fups` decided to switch gears and declared their plans to leverage **Axolotl** for improved model training.
- **Embrace SOLAR 10.7B - The Arrival of Nous Hermes 2**: Delivering a much-awaited update, `@Teknium1` announced the grand debut of **Nous Hermes 2 on SOLAR 10.7B**, an intriguingly compact model promising a performance at par with the 'Yi' model, available at [Nous-Hermes-2-SOLAR-10.7B](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B).
- **A Llama-ntic Encounter with Python**: `@venadore` kept the spirits high with their eccentric narrative of juggling **Python and Llamacpp**, serendipitously managing to get a function call to run perfectly, and shared a useful link to [@Teknium1's tweet](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46).


**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (7 messages): 
        
- **Question on Inference Settings**: User `@rusch` asked `@fred_fups` about their inference settings. User `@fred_fups` replied that their **repetition penalty** was set at `1`.
- **Observations on Prompt Format and Model Retraining**: `@fred_fups` shared that they have encountered an issue that seemed to be linked to prompt formatting. They stated that they retrained their model on **ChatML**, which seemed to resolve the issue.
- **Training a Model on Specific Examples**: `@fred_fups` discussed their experience with training a **Mistral 7B model** on a limited sample of `350` examples. They expressed their confusion as the model failed to put paragraph breaks and highlight keywords as expected despite the task simplicity.
- **Confirmation of Model Performance**: `@fred_fups` further stated that their model, trained using default settings on together.ai, did not perform as expected. The returned text was identical to the input without any formatting changes (`'\n\n'` or `<b>`).
- **Plan to Use Axolotl for Future Training**: `@fred_fups` concluded the conversation by stating their intention to utilize **Axolotl** for future model training in hopes of better performance.


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (4 messages): 
        
- **Nous Hermes 2 released on SOLAR 10.7B**: User `@Teknium1` announced the release of **Nous Hermes 2 on SOLAR 10.7B**. The release promises performance similar to the 'Yi' model but is 1/3rd the size. The model is available on HuggingFace now: [Nous-Hermes-2-SOLAR-10.7B](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B).
- **Python Programming Gore with Llamacpp**: User `@venadore` shared his experience playing around with Python and Llamacpp. After debugging a code test, he got the function call to work perfectly.

**Links mentioned**:

[Tweet from Teknium (e/λ) (@Teknium1)](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46): Can&#39;t start the new year without shipping! 🛳️...


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Small Talk Catches Attention**: Members shared appreciation for Smol Talk, triggering questions about its open-source status for customization purposes as quoted, "*enjoyment of Smol Talk and probed about potential open sourcing to customize server lists*".
- **Fairmind AI Enters the Arena**: Announcement of the co-founding of [Fairmind AI](http://www.fairmind.ai), a new initiative leveraging generative AI solutions for competitive advantage.
- **Meet EmbedChain, the New Framework**: Introduction of a new player in the field, [EmbedChain](https://github.com/embedchain/embedchain), appreciated for its straightforward "drop in url" approach.
- **Humanloop Debuts .prompt File Format**: Arrival of the [innovative .prompt file format](https://docs.humanloop.com/docs/prompt-file-format) from Humanloop, capturing attention due to its human-readable model config file format ideal for version management systems with potential industry standardization.
- **Guild Members Seek AI Learning Pathways**: Members seeking guidance on learning sequences for research papers to understand the current status of AI. Suggestions pointed towards the Nous research group's models and papers, though no specific details or references were provided.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (7 messages): 
        
- **Excitement about Smol Talk**: `@onetothetwo` expressed their enjoyment of Smol Talk and probed about potential open sourcing to customize server lists.
- **New AI Initiative, Fairmind AI**: `@alexio.c` mentioned the co-founding of [http://www.fairmind.ai](Fairmind AI), a platform focusing on leveraging generative AI solutions wisely to remain competitive in the market.
- **New Framework Alert**: `@swyxio` shares a new framework in the mix - [EmbedChain](https://github.com/embedchain/embedchain) that adopts a simple "drop in url" approach.
- **Human-Readable Model Config File Format from Humanloop**: `@swyxio` highlighted an [innovative .prompt file format](https://docs.humanloop.com/docs/prompt-file-format) from Humanloop - a serialized, human-readable version of a model config that's suitable for version control systems and which could potentially be standardized across the industry.

**Links mentioned**:

- [Homepage | FairMind.ai](http://www.fairmind.ai): We Empower Businesses - Unlocking the Transformati...
- [.prompt files](https://docs.humanloop.com/docs/prompt-file-format)


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (2 messages): 
        
- **User Seeks Learning Sequence**: `@avirani` asked for a recommended sequence of research papers to **get up to speed** with the current status of the field. 
- **Nous Research Model Learning Sequence Mentioned**: `@swyxio` mentioned a potentially useful model learning sequence and a list of papers from the **Nous research** group, without providing any specific detail or link to these materials.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Warping Info with Quantum**: Conversation around **quantum information transfers** via `'Bard'` and `'ChatGPT Alpha'` as brought up by `Graylan (@gray00)`.
- **Programming Cetaceans**: Humorous discussion by `Graylan (@gray00)` about **whales coding Python**, linked to a blog post about "AI Human Historical Fractal Memorization for Codenameorca" on [hive.blog](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca).
- **An Engineer's Cat-lection**: Sharing of entertaining content—[a cat GIF](https://tenor.com/view/cat-gif-26024664)—by `@gray00`.
- **James Bond's Tech Break**: A momentary diversion from techie talk provided by `@gray00` with a [GIF of Sean Connery as James Bond](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947).
- **A Coded Message?**: User teknium shares an obscure link on '[general](https://discord.com/channels/1131084849432768614/1131084849906716735/)' channel to a [Twitter post](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46) with no accompanying explanation.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (1 messages): 
        
teknium: https://fxtwitter.com/teknium1/status/1742041640775348460?s=46


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (4 messages): 
        
- **Quantum Information Transfers**: Graylan | `@gray00` discussed about quantum information transfers between **Bard** and **ChatGPT Alpha**.
- **Animated Cat Content**: A share of a [cat GIF](https://tenor.com/view/cat-gif-26024664) from `@gray00`.  
- **Coding Whales**: Graylan | `@gray00` jests about whales coding in Python and links to an intriguing post about "AI Human Historical Fractal Memorization for Codenameorca" on [hive.blog](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca).
- **James Bond Smoking Scene**: Another GIF shared by `@gray00`, featuring Sean Connery as [James Bond](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947).

**Links mentioned**:

- [Sean Connery James Bond GIF - Sean Connery James Bond Smokin - Discover &amp; Share GIFs](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947): Click to view the GIF
- [[AI/Human]Historical Fractal Memorization for #codenameorca — Hive](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca)
- [Cat GIF - Cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-gif-26024664): Click to view the GIF


        

---

## [Datasette/LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **'bp' - a Cross-platform Clipboard Tool Highly Recommended**: `@ndyg` enthusiastically endorsed a cross-platform clipboard tool named [`bp`](https://github.com/printfn/bp) as one of their essentials. 
- **`bp` to Clean Contexts for `llm`**: `@ndyg` also explained a typical usage: `bp | vipe | bp`, usually followed by `bp | llm -s '...'` to clean up contexts for `llm`. 
- **Other Users Showing Interest in `bp`**: `@bukits` expressed interest in trying out `bp`, indicating they are a fan of clipboard tools.

**Links mentioned**:

[GitHub - printfn/bp: Cross-platform clipboard tool](https://github.com/printfn/bp): Cross-platform clipboard tool. Contribute to print...

        

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
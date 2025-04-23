---
id: d11ec836-6a8e-4d11-9b8d-ad23da8c1ac1
title: '[AINews] 1/1/2024: How to start with Open Source AI'
date: '2024-01-03T07:23:06.742443Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-112024-how-to-start-with-open-source-ai
---

<!-- buttondown-editor-mode: plaintext -->Teknium and LDJ dropped a good learning path in the Nous Discord over the new year. swyx also updated his [December notes repo](https://github.com/swyxio/ai-notes/commit/650319e6cb5423a684200255de3ec200ef9953f0) in preparation for the monthly/year end recap on Latent Space.

Projects: 
 ![image.png](https://assets.buttondown.email/images/ee262ecf-65e7-468f-82a0-bcfdaf37789b.png?w=960&fit=max) 

Papers:

 ![image.png](https://assets.buttondown.email/images/c57ed945-05a9-461b-8292-5bdbe3d28e08.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- A mixed sentiment was held regarding the performance and quality of **Bing's AI** versus **ChatGPT** and **Perplexity AI**, with instances of both negative and positive perspectives shared. A similar divide was observed regarding **Microsoft's Copilot product** and its integration with **Office 365**.

- Detailed discussions about **DALL-E 3 access within ChatGPT Plus**, technical issues faced by users in **ChatGPT's performance** and potential solutions explored. 

- Users discussed desires to train a **GPT model to generate content from a book**, and potential ways such as integration with **ChatGPT Plus** or **OpenAI API** were suggested. 

- Anticipation and preferences for **GPT4 turbo in Microsoft Copilot's future update** were conversed, added with constructive comments about **AI reasoning, prompt engineering, and output length limitation of GPT models**. 

- Predominantly inquiries and clarifications about improving **Custom GPT's performance**, overcoming **Recaptcha glitches**. Dealings with **overhauling a Custom GPT model after poor modifications,** and the process of tracking v4 usage were talked about. 

- A detailed interaction and process involving the use of **Custom GPT for creating humorous TikTok scripts** was shared. This revolved around a user seeking advice to improve the model's outcomes and a fellow user offering a complex example. 

- Users, like `.009_f.108`, who are new to AI programming, were advised to start with **Python**. To improve the use of custom knowledge, YAML or Markdown file structures were suggested over JSON by `@sciandy`.

- Finally, an interesting discussion on the future of AI specifically with expositions concerning the necessity of multiple GPTs. This ranged from the favorability of specialization in models to the role of **Microsoft's Copilot** in the growing field of AI.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (3 messages): 
        
- **Comparison of AI Products**: `@chief_executive` expressed a negative opinion about Bing AI, stating it's inferior to ChatGPT or even Perplexity AI.
- **Microsoft Copilot and Office 365**: `@chief_executive` sees the Copilot product as a lost cause, while considering the Office 365 integration as potentially non-problematic.
- **Different Perspective on Bing and Copilot**: Contrary to the earlier critique, `@michael_6138_97508` expressed a positive view on Bing, despite some glitches, and showed interest in a potential Copilot lower tier subscription without Office 365.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (126 messages🔥🔥): 
        
- **DALL-E 3 access discussion**: `@black_swordsman99` expressed confusion regarding **DALL-E 3** access within **ChatGPT Plus**. It was clarified by `@darthgustav.` that **DALL-E 3** is integrated and can be accessed via **GPT 4** or using the **DALL-E 3 custom GPT**. 

- **ChatGPT Payment Issues**: `@rememberthebrigade` inquired about the possibility of using **Google Play Store cards** to pay for **ChatGPT Plus** due lack of a credit card. `@jaicraft` affirmed the possibility of such transaction via the Android app.

- **ChatGPT Performance Issues**: Several users including `@kilogamz` and `@youraveragedev` raised concerns about **ChatGPT's performance and responsiveness**. `@kilogamz` detailed a specific issue where **ChatGPT halted their D&D session planning** by giving an error message upon asking for pictures. The issue persisted despite various attempts to resolve it including clearing cache, using incognito mode, and using different browsers. 

- **Training GPT model using the content of a book**: `@sushiconduit` asked how to train a GPT model to generate content from a book. `@thunder9289` suggested adding the book to the knowledge base of a private custom GPT if they have ChatGPT plus. If they don't, they need to use the OpenAI API and create a custom GPT with a knowledge base within the playground.

- **Usage of Copilot GPT-4 turbo**: `@jaicraft` and `@pruo` discussed anticipation and usage preferences of **GPT-4 turbo in Copilot** which is yet to be released.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (39 messages🔥): 
        
- **Optimizing GPT for a new computer language:**
    - User `@tilanthi` expressed concern about improving the performance of a GPTs trained on a bespoke version of Lua when it comes to creating new codes or strategies. They noted the difference in output compared to GPT4 generating advanced Python scripts. They voiced a hypothesis that Python examples are part of the LLM training process, and the lack of similar examples for Lua could be hindering performance.
    - User `@solbus` clarified that the "training" of custom GPTs is limited to the information provided in the GPT edit page: Instructions + Knowledge files + Actions. They emphasized that knowledge files serve as reference documents and not as permanent added context. They provided recommendations on tweaking the custom GPT's instructions to improve performance and understanding.

- **Issues with Recaptcha**: `@watchalls` complained about having to resolve Recaptcha puzzles after every sentence/entry into GPT-4 on desktop. `@solbus` advised checking if a VPN is installed and also to confirm if the issue is browser-specific.

- **Overhauling Custom GPT after unsatisfactory modifications:** `@thefreebachelor` expressed dissatisfaction with their customized GPT for basic life coaching after attempting to improve it using a Reddit-found prompt. User `@solbus` provided consistent support and advice, including keeping a copy of instructions for future reference, a reminder of the limitations of knowledge files, and offered advice for writing more effective instructions. 

- **Reaching v4 Limit:** User `@phospheneoverdrive` inquired about checking the quota for their v4 usage. User `@solbus` responded by clarifying that no UI indicator exists and suggested pacing oneself by sending one message every 4.5 minutes to stay within the 40/3hrs limit.

- **Successfully implementing humorous Tiktok video script prompts using a customized GPT:** `@user691378` sought help on creating a prompt for generating funny Tiktok video scripts with a custom GPT model, describing their unsuccessful attempts despite feeding the model with twenty example scripts.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (6 messages): 
        
- **New Year Greetings**: `@randall.townsend` wished everyone a Happy New Year. 
- **Learning AI programming and coding**: User `.009_f.108` solicited for advice on how to get started with AI programming and coding, to which `@darthgustav.` suggested starting with Python.
- **Structuring files for custom knowledge** : `@iamhere6321` inquired about the best way to structure files for custom knowledge uploads. They noticed that while HTML and PDF formats work fine, JSON files with custom structure do not produce the expected results. They questioned whether using `llamaindex` would provide better control.
- **Suggestions on file structures**: In response to the above query, `@sciandy` proposed using YAML or Markdown instead of JSON for structuring files.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (222 messages🔥🔥): 
        
- **Discussions on AI's Reasoning Abilities and Evading Answers**:
   - Several users, starting with `@johannes43` and `@hawaiianz`, discussed the AI's improving ability to understand English, with `@hawaiianz` predicting no need for prompt engineering in a year's time. `@beanz_and_rice` disagreed, stating that there will always be some engineering required with words and semantics. They expressed frustration with how preemptive the AI is in explaining its lack of access to real-time information, even when the question doesn't require such data.
   - `@madame_architect` added that some developers are using frozen models to avoid the issues that arise with each update of the AI model.
   
- **Limiting GPT-4 Output Length**:
   - `@MishaMgla` asked for help in limiting the output of GPT-4 by word or character amount. `@rendo1` and `@madame_architect` explained that this is something GPT tends to struggle with, as it generates responses on a token-by-token basis, making it hard to plan and limit its writing in advance.

- **Process of Learning Prompt Engineering**:
   - `@user691378` asked for guidance on how to effectively train their custom model in generating good TikTok video scripts after providing it with about 20 sample scripts. `@beanz_and_rice` provided a detailed and complex example of a custom prompt to help train the user's model to achieve the desired result.
   - `@madame_architect` shared her learning process with GPT, which includes reading research papers on prompting tactics, trying out different methods, and learning from failures.

- **Discussion on Future of AI (GPT Models and CoPilot)**:
   - `@beanz_and_rice` questioned the necessity of multiple GPTs and expressed a desire for simplicity in having one model for all tasks. `@madame_architect` suggested that the future might favor specialized models alongside a main GPT to query them. She also pointed towards Microsoft's Copilot as a growing player in the field.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (222 messages🔥🔥): 
        
- **Understanding of ChatGPT**: In a discussion between `@beanz_and_rice`, `@hawaiianz`, and `@johannes43`, they shared their opinions about the performance and future of ChatGPT. They pointed out that while the AI has a robust understanding of English, its current iteration may sometimes be evasive or make assumptions that aren't accurate.
- **Model Performance Discussion**: `@beanz_and_rice` expressed dissatisfaction over the updates, noticing that GPT-4 and GPT-4.1106 preview (GPT4 Turbo) appear to be more evasive. He reported that asking questions about "the last time X was something" triggers evasion as the model incorrectly assumes a need for real-time information.
- **Skill Limitations of GPT**: The topic of GPT's limitations in tasks came up. When `@MishaMgla` wanted to limit the output of GPT by characters or symbols, `@rendo1` and `@madame_architect` pointed out that GPT is not efficient at this type of task since it doesn't plan its writings in advance.
- **Prompts and Custom Models**: The importance and challenge of creating good prompts were discussed. `@beanz_and_rice` shared a complex prompt example, which he calls "Custom AI syntax", for `@user691378` who asked for help in creating a prompt for a funny TikTok script. He also mentioned that the language could be generated by GPT-4.
- **Discussion about Microsoft's Copilot**: In a conversation involving `@madame_architect` and `@beanz_and_rice`, Microsoft's Copilot was brought up. `@madame_architect` speculated about its potential growth and shared hands-on experiences with different versions of it, while `@beanz_and_rice` admitted to not being impressed with GitHub Copilot.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- The development of the **Scholar Evals Platform** to allow for direct visualization and publication of results/reproductions was discussed, with a link shared to the GitHub repository. The platform was well-received, and feedback was requested. (source: [GitHub - scholar-org/scholar-evals](https://github.com/scholar-org/scholar-evals))
- Revealing conversations on various AI topics, including **Counterfactual Regret Minimization** in poker strategies, potential heterogeneity in future AI architectures with transformer and Selective Sequence Memory (SSM) blocks, and an experimental fine-tune of `Yi-34B-200k` using `bagel` which predicted interesting benchmarks. (sources: [Blogpost on Counterfactual Regret Minimization](https://rnikhil.com/2023/12/31/ai-cfr-solver-poker.html), [LessWrong Blog on AI architectures](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and#comments), [GitHub - NousResearch/StripedHyenaTrainer](https://github.com/NousResearch/StripedHyenaTrainer), [jondurbin/bagel-dpo-34b-v0.2](https://huggingface.co/jondurbin/bagel-dpo-34b-v0.2))
- **Finetuning of models** was discussed, emphasizing the advantage of time and resource efficiency over creating models from scratch. Debate on the pros and cons of the proprietary `Yi-34b` base against the AGPL-compliant `Mixtral` base emerged with differing opinions coming forth. (sources: None)
- Encouraging words were shared for beginners attempting to break into the field of AI, urging persistence and the value of bringing in unique insights from other domains. (source: None)
- Discord was noted on the performance of chat models, preferences for instant task completion variants, the distinction between chat and instruct models, and the scope of open-source to fill the gaps left by discontinued models. (sources: [Tweet from Lijie Fan (@lijie_fan)](https://fxtwitter.com/lijie_fan/status/1741916320827093209?s=46), [Tweet from Amjad Masad (@amasad)](https://twitter.com/amasad/status/1741902845539140061))
- In-depth conversation around Local Large Language Models (LlaMas) unfolded, discussing topics like comparison between different models, process of expanding the tokenizer for new languages, running models on MacBook, and using models for automation tasks. Several resources for LlaMa experimentation were recommended, such as LM Studio's `Openhermes` model. (sources: None)

**Nous Research AI Channel Summaries**

### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (3 messages): 
        
- **Scholar Evals Platform**: User `@manveerxyz` is developing a platform built on top of Eleuther's LM harness that allows visualization of results/raw outputs, and eventually plans on letting users publish results/reproductions from the platform. They requested feedback on the MVP, sharing the link to [GitHub - scholar-org/scholar-evals](https://github.com/scholar-org/scholar-evals) for further details. User `@gabriel_syme` expressed appreciation towards the project, to which `@manveerxyz` responded encouragingly for testing it out.


**Links mentioned**:

[GitHub - scholar-org/scholar-evals: A unified platform for benchmarking large language models.](https://github.com/scholar-org/scholar-evals): A unified platform for benchmarking large language...


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (8 messages🔥): 
        
- **Counterfactual Regret Minimisation**: User `@nnn5686` shared a [blogpost](https://rnikhil.com/2023/12/31/ai-cfr-solver-poker.html) from `@rnikhilcom` on Counterfactual Regret Minimization and its application in Poker Winning strategies.
- **Heterogeneity in AI architectures**: Post by `@vincentweisser` sharing a [LessWrong blog](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and#comments) stating that AI architectures may soon comprise not only transformer blocks but also Selective Sequence Memory (SSM) blocks. These two kinds of blocks are stated to be natural opposites on the tradeoff scale between episodic cognitive capacity (Transformer's strength) and long-term memorisation (SSM's strength).
- **StripedHyenaTrainer**: `@vincentweisser` mentioned a [GitHub repository](https://github.com/NousResearch/StripedHyenaTrainer) called "StripedHyenaTrainer" related to the discussion on transformer vs. SSM blocks.
- **Experimental Fine-tune with Bagel**: `@metaldragon01` shared a [link](https://huggingface.co/jondurbin/bagel-dpo-34b-v0.2) to an experimental fine-tune of `Yi-34B-200k` using `bagel`. It noted that the model has been trained on every dataset and predicted that the benchmarks could be interesting.
- **New Datasets**: `@teknium` expressed interest in discovering new datasets featured in the `bagel` model shared by `@metaldragon01`.

**Links mentioned**:

- [Tweet from Nikhil R (@rnikhilcom)](https://x.com/rnikhilcom/status/1741416707338756259?s=46): Counterfactual Regret Minimisation or How I won an...
- [jondurbin/bagel-dpo-34b-v0.2 · Hugging Face](https://huggingface.co/jondurbin/bagel-dpo-34b-v0.2)
- [GitHub - NousResearch/StripedHyenaTrainer](https://github.com/NousResearch/StripedHyenaTrainer): Contribute to NousResearch/StripedHyenaTrainer dev...
- [AGI will be made of heterogeneous components, Transformer and Selective SSM blocks will be among them — LessWrong](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and#comments): This post is prompted by two recent pieces: …


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (167 messages🔥🔥): 
        
- **AI-Powered Voice Chat Use**: User `@mihai4256` discussed the feature of AI models responding in a character's voice and recognizing when the user stops speaking in a voice chat. However, `@teknium` clarified that the 2-way voice chat feature has been around on mobile since dev day.
- **Finetuning Models**: User `@iamavalex` raised questions about the process of finetuning models, asking about the required resources such as dataset, base model, and GPUs. `@.beowulfbr` and `@night_w0lf` provided answers, emphasizing that the advantage of finetuning is time and resource efficiency compared to creating a model from scratch.
- **Yi-34b Base vs Mixtral Base**: User `@asgnosi` sought input on the pros and cons of using the Yi-34b base against the Mixtral base, with `@night_w0lf` pointing out that Yi may be slower considering it's a denser model compared to MOE.
- **Introduction to AI for Beginners**: `@Serial Connector` reflected on feeling overwhelmed as a beginner in AI, and `@gabriel_syme` offered advice about persistence and contributing unique insights from other areas of expertise. 
- **Debate on Model Functionality & Behavior**: `@gabriel_syme` expressed dissatisfaction with chat models for handling certain tasks, stating a preference for more instant task completion models. The conversation explored the distinction between chat and instruct models, with `@teknium` explaining how data and training impact a model's behavior. The conversation touched on the loyalty of the community to the extinct instruct models and the potential for open-source to fill this gap.

**Links mentioned**:

- [Tweet from Lijie Fan (@lijie_fan)](https://fxtwitter.com/lijie_fan/status/1741916320827093209?s=46): 🚀 Is the future of vision models Synthetic? Intro...
- [Tweet from Amjad Masad (@amasad)](https://twitter.com/amasad/status/1741902845539140061): With davinci I can &#34;program&#34; it using trad...


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (51 messages🔥): 
        
- **Comparison of Models**: `@yeyito777` compared Phi-2, Mistral, and DeciLM-7B models in terms of benchmarks and recommended Mistral due to its openness and wide adoption.

- **Expanding the Tokenizer**: `@qnguyen3` discussed the process of expanding the tokenizer to include new languages. They suggested following the Chinese-Llama team's clear instructions and mentioned the VinaLLaMA paper as a resource. The process involves training a new Sentencepiece tokenizer and concatenating it to the current model's tokenizer.

- **LLaMAs and Learning Journey**: Several users including `@momentumjs`, `@gsayko`, `@teknium`, and `@rohinish404` engaged in a discussion about starting a learning journey with Open Source models and Local Large Language Models (LlaMas). `@teknium` suggested experimenting with models using LM Studio, specifically highlighting the Openhermes model.

- **Running Models on MacBook**: A chat about running language models on a MacBook was carried on by `@teknium` and `@rohinish404`. They ephasized that the speed of models running locally is largely dependent on the computing power of the machine. With the 'Metal' checkbox checked under the hardware config, language models are said to run much faster on M1/M2/M3 chips.

- **Using Models for Automation**: `@teknium` suggested that when a task is found that can be automated using an OS model, there are plenty of ways to get it running on a command line. They additionally cited APIs like Together.ai and OpenRouter as alternatives to ChatGPT.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- A in-depth discussion about various language models, citing particular interest in **TinyLLaMA** for specialized model training, evaluation of **Gemini**'s weaknesses in reasoning, and the high hardware requirements for **Mixtral**. A study called [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112) was shared, exploring model compression through weighted SVD, complementing the discussions about model optimization where the LASER technique was also discussed. 
- `@faldore` reported issues in the `#axolotl-dev` channel with resuming from a checkpoint on Mixtral Qlora and a RuntimeError encountered while attempting to install PEFT from source.
- In `#general-help`, guidance was sought and given on various topics, such as fine-tuning configs for **Mixtral 8x7b**, data formatting for fine-tuning, inconsistency in model output, impact of shorter training samples on model performance, and how accuracy (ACC) is calculated for LLM answers. `@morgymcg` shared their experimental notes for Mixtral on [wandb.ai](https://api.wandb.ai/links/reviewco/iwaneb73)
- The `#datasets` channel exemplified a supportive environment in which users shared insights and sought advice on creating custom datasets. An [example dataset](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) and a [custom dataset](https://huggingface.co/datasets/cfahlgren1/DevSpecCode) for code execution were shared. There were conversations about the optimal format for a function-calling dataset, generating large instruction datasets, pretraining data and the practice of mixing languages, and a workflow for adding new tokens to a base model. 
- In the `#rlhf` channel, users reported and resolved issues concerning Mixtral compatibility and upgrading transformers. A particular point of clarification about the necessity of an SFT for training the DPO model especially if no new tokens are being added was raised.
- A suggestion to incorporate legal data into AI models was made by `@dangfutures` in the `#shearedmistral` channel, referencing a potential source of such data in the [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) dataset on Hugging Face.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (108 messages🔥🔥): 
        
- **Discussion on TinyLLaMA and model specialisation**: `@nafnlaus00` expressed fondness for **TinyLLaMA** as a base for training specialized models due to its good performance on non-complex tasks with minimum training time and fast inference time. Specific quote: "*I love TinyLLaMA (base, not chat) - I use it as a base for training specialized models.*".
- **Gemini in Reasoning**: `@noobmaster29` shared a [link](https://huggingface.co/papers/2312.17661) to a detailed overview and evaluation of **Gemini**, a multimodal large language model introduced by Google. It is noted that despite Gemini's advancements, it lacks in commonsense reasoning tasks when benchmarked.
- **Discussion on model optimization and LASER technique**: `@mihai4256` and `@fernando.fernandes.` had discussions on the LASER noise reduction technique, used to fine-tune models. They mentioned that applying LASER doesn't reduce parameters, but changes the ranks of the weight matrices. They also included ongoing work on generalizing this technique, beyond using it specifically for a dataset.
- **Concerns about model training hardware requirements**: `@yamashi`, `@dangfutures`, and `@casper_ai` discussed the high hardware requirements - such as the need for 8x A100 GPUs - for training models like **Mixtral**.
- **Talk on Low Rank Factorization for model compression**: `@stefangliga` shared a [link](https://arxiv.org/abs/2207.00112) to a study which uses Fisher Information for weighted SVD as an approach for model compression.

**Links mentioned**:

- [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112): Factorizing a large matrix into small matrices is ...
- [Mixtral 8x7B support (#2011) · vllm-project/vllm@b5f882c](https://github.com/vllm-project/vllm/commit/b5f882cc98e2c9c6dde7357dbac2ec0c2c57d8cd): Co-authored-by: Pierre Stock &lt;p@mistral.ai&gt; ...
- [Tweet from AK (@_akhaliq)](https://x.com/_akhaliq/status/1741658336917872755?s=20): Gemini in Reasoning: Unveiling Commonsense in Mult...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (4 messages): 
        
- **Issues with Resuming from Checkpoint on Mixtral Qlora**: User `@faldore` reported that they are unable to resume from a checkpoint on Mixtral Qlora.
- **Attempt to Install PEFT from Source**: `@faldore` indicated trying to install PEFT from its source but encountered difficulties.
- **RuntimeError during PEFT Installation**: While trying to install PEFT, `@faldore` encountered a RuntimeError. The error claimed: "*Error(s) in loading state_dict for PeftModelForCausalLM: Missing key(s) in state_dict.*" which included a list of base_model key names.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (34 messages🔥): 
        
- **Mixtral 8x7b Fine-tuning Suggestions**: `@morgymcg` asked for suggestions on fine-tuning configs for Mixtral 8x7b. They have been using the SlimOrca dataset, conducting qlora fine-tuning on 8 x H100s over the course of several days. Rough experiment notes are available on [wandb.ai](https://api.wandb.ai/links/reviewco/iwaneb73). `@le_mess` suggested testing different optimizers, including `adamw_bnb_8bit`, `lion_8bit`, `paged_adamw_32bit`, `paged_adamw_8bit` and `paged_lion_8bit`.
- **Data Formatting for fine-tuning**: `@matanvetzler` asked for advice about how to format the "datasets" section in the config file using a specific JSONL structure. `@le_mess` suggested converting the dataset to the ShareGPT format.
- **Consistency in Model Output**: `@colejhunter` reported an issue where model outputs yielded inconsistent results when training on a small dataset versus a larger one, despite keeping variables consistent across both runs. Repetitions and random text were noted in the outputs from the larger dataset.
- **Impact of Shorter Training Samples**: `@suikamelon` reported that after adding shorter conversation snippets to the training dataset, the model's performance was negatively impacted for longer context sizes. They asked if sample packing could be a solution. 
- **Calculating ACC for LLM Answers**: `@noobmaster29` sought information on how accuracy (ACC) for LLM answers is calculated, noting that change in the prompt in mmlu significantly impacted the results. Their base model was Mistral.


**Links mentioned**:

- [Mixtral Experiments Notes](https://api.wandb.ai/links/reviewco/iwaneb73): Searching for a strong Mixtral fine-tuning recipe
- [Tweet from Eric Hartford (@erhartford)](https://twitter.com/erhartford/status/1740977087006339137)): Dolphin-2.7-mixtral-8x7b is cooking with these fix...


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (11 messages🔥): 
        
- **Data sets for fine-tuning**: In the context of creating datasets for fine-tuning models, `@nruaif` shared an [example dataset](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) and `@le_mess` mentioned that there were more datasets linked in the channel.
- **Format for function calling dataset**: `@yazanhussein01` inquired about the ideal format for a function calling dataset. `@le_mess` suggested adhering to the sharegpt format, but clarified that there's no universal format.
- **Generating large instruction datasets**: `@stoicbatman` sought advice for generating large instruction datasets (between 20k-50k samples) drawn from GPT-4, specifically asking for tips or resources on better data generation.
- **Pretraining data and mixing languages**: `@noobmaster29` asked about the quality of data for pretraining and whether it's advisable to mix English with a new language token to prevent the model from "forgetting" English. They also mentioned their current workflow of adding the new tokens on a base model using a pass on completion, then, a separate pass for learning instructions, and sought feedback on the appropriateness of this approach.
- **Feedback on a complex code generation dataset**: `@cf0913` asked for feedback on a [custom dataset](https://huggingface.co/datasets/cfahlgren1/DevSpecCode) they created for code execution with complex, multiple requirement instructions. The dataset is geared towards generating synthetic code based on detailed requirements.

**Links mentioned**:

- [cfahlgren1/DevSpecCode · Datasets at Hugging Face](https://huggingface.co/datasets/cfahlgren1/DevSpecCode)
- [mhenrichsen/alpaca_2k_test · Datasets at Hugging Face](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test)


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (8 messages🔥): 
        
- **Mixtral Compatibility Issues**: User `@dangfutures` reported issues getting **Mixtral** to work with their current branch, prompting a response from `@caseus_` that they would consider merging or rebasing their branch.
- **Upgraded Transformers KeyError**: `@dangfutures` initially faced a **KeyError** after upgrading transformers but later realized they were using the wrong dataset.
- **DPO Model Training**: A discussion between `@dangfutures` and `@caseus_` clarified that an **SFT** may not be necessarily required to train the **DPO model**, particularly if no new tokens are being added.


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (2 messages): 
        
- **Legal Data in AI Models**: User `@dangfutures` suggested incorporating legal data into AI models, possibly to improve these models' performance in legal contexts. They also shared a link to a dataset named [Pile of Law](https://huggingface.co/datasets/pile-of-law/pile-of-law) on Hugging Face, indicating that it could be a potential source of legal data for AI training. The dataset is primarily in English and contains a large corpus of legal and administrative data.

**Links mentioned**:

[pile-of-law/pile-of-law · Datasets at Hugging Face](https://huggingface.co/datasets/pile-of-law/pile-of-law)


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- Exchange of **New Year wishes** across different channels, with greetings from `@targed`, `@pseudoterminalx`, `@puffy310`, and `mega_b`.
- Discussion around issues with the dataset viewer on [huggingface.co](https://huggingface.co/datasets/multimodalart/steamboat-willy-frames), with `@SegmentationFault` reporting a `JobManagerCrashedError`.
- Proposal of using models like **CogVLM or GPT4V** to caption public domain datasets from 1928 Mickey's appearances with a focus on potential training on an SDXL lora, dreambooth, or full fine-tune as suggested by `@SegmentationFault`. This was followed by notification from `@thejonasbrothers` about an already published lora model for the public domain mouse character at [huggingface.co](https://huggingface.co/multimodalart/public-domain-mouse-character).
- Appreciation shared by `@SegmentationFault` of Google's T2I technology, noting its fidelity and lack of overfit issue commonly seen in midjourney and dalle.
- In-depth consideration of a new method for learning **visual representation from synthetic images** and captions called **SynCLR**, as introduced in a [paper](https://huggingface.co/papers/2312.17742) discussed by `@spirit_from_germany`. This led to debates around its limitations and performance.
- Discussion on **botting** with `@cutycat2000` boasting about a photo-tracking bot. A request to share said bot was expressed by `@ishaanshri95`.
- Exchange on how AI models learn concepts, discussing potential for concept learning if the model was not explicitly trained on, as `@phryq` questioned and `@JH` confirmed. The conversation proceeded to explore the impact of image captions. 

**Key Resources Shared**:

- [Pluralistic: 2024&#8217;s public domain is a banger (20 Dec 2023) &#8211; Pluralistic: Daily links from Cory Doctorow](https://pluralistic.net/2023/12/20/em-oh-you-ess-ee/)
- [multimodalart/steamboat-willy-frames · Datasets at Hugging Face](https://huggingface.co/datasets/multimodalart/steamboat-willy-frames)
- [Pclanglais/Mickey-1928-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Pclanglais/Mickey-1928-dataset)
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1741668076037247127?t=83X-PzqSIMTQVncqyg-LDw&s=19): Learning Vision from Models Rivals Learning Vision...
- [GitHub - google-research/syn-rep-learn: Learning from synthetic data - code and models](https://github.com/google-research/syn-rep-learn): Learning from synthetic data - code and models. Co...

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (12 messages🔥): 
        
- **New Year Celebrations**: Users `@targed`, `@pseudoterminalx`, and `@puffy310` exchanged **New Year wishes**.

- **Issues with Dataset Viewer**: `@SegmentationFault` encountered an issue with the dataset viewer on huggingface.co. The problem seemed to relate to the `JobManagerCrashedError` error, which prevented the full dataset from being displayed. [Link to dataset](https://huggingface.co/datasets/multimodalart/steamboat-willy-frames)

- **Public Domain Cartoon Datasets**: `@SegmentationFault` proposed the use of models like **CogVLM or GPT4V** to caption public domain datasets from 1928 Mickey's appearances, suggesting potential training on an SDXL lora, dreambooth, or full fine-tune.

- **Public Domain Mouse Character LORA Model**: `@thejonasbrothers` notified that a lora model has already been published for the public domain mouse character. The model can be accessed at [huggingface.co](https://huggingface.co/multimodalart/public-domain-mouse-character).

- **Advancements in Google's T2I Technology**: `@SegmentationFault` shared admiration for Google's T2I technology, calling it more impressive than dalle3 in terms of fidelity for characters without the overfit issue seen in midjourney and dalle itself.

**Links mentioned**:

- [Pluralistic: 2024&#8217;s public domain is a banger (20 Dec 2023) &#8211; Pluralistic: Daily links from Cory Doctorow](https://pluralistic.net/2023/12/20/em-oh-you-ess-ee/)
- [multimodalart/steamboat-willy-frames · Datasets at Hugging Face](https://huggingface.co/datasets/multimodalart/steamboat-willy-frames)
- [Pclanglais/Mickey-1928-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Pclanglais/Mickey-1928-dataset)


### ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/) (1 messages): 
        
mega_b: Happy New Year!  🎉


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (114 messages🔥🔥): 
        
- **SynCLR**: In a [paper](https://huggingface.co/papers/2312.17742) discussed by `@spirit_from_germany`, SynCLR, a new method for learning visual representation from synthetic images and captions, is introduced. `@spirit_from_germany` suggests follow-up work using different concept lists and OpenAI's one billion blip 2 captions. The discussion later expands to how to efficiently generate images, with `@thejonasbrothers` and `@rom1504` pointing out current limitations.

- **Botting**: `@cutycat2000` boasts about a bot they coded which can track photos to cities and countries. A request for sharing the bot comes from `@ishaanshri95`.

- **Limitations of SynCLR**: `@thejonasbrothers` voices doubts regarding SynCLR, noting that the method's dependence on a text to image model essentially distills the original model. `@rom1504` disagrees, stating that Image text isn't supervised. The conversation continues to debate this point.

- **Performance of SynCLR**: `@thejonasbrothers` and `@rom1504` dispute the performance of SynCLR, with a quoted excerpt from the research suggesting that SynCLR outperforms openclip, though `@rom1504` refutes this.

- **Discussions on Training with Synthetic Data**: The debate progresses to the practicality of training with synthetic data. `@rom1504` maintains that this can control what the model is trained on, highlighting benefits even under worst-case scenarios. However, `@thejonasbrothers` points out the exorbitant costs and limitations of current T2i models.

**Links mentioned**:

- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1741668076037247127?t=83X-PzqSIMTQVncqyg-LDw&s=19): Learning Vision from Models Rivals Learning Vision...
- [GitHub - google-research/syn-rep-learn: Learning from synthetic data - code and models](https://github.com/google-research/syn-rep-learn): Learning from synthetic data - code and models. Co...


### ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/) (6 messages): 
        
- **Concept Learning in AI Models**: `@phryq` inquired if an AI model could still "learn concepts" that were not explicitly trained, such as a visual concept. `@JH` confirmed that it would be likely for any diffusion model trained on a diverse dataset. 
- **Impact of Image Captions on Concept Learning**: `@phryq` further discussed the potential for a model to learn a concept, like bowties, from images even if it was not often or ever captioned. 
- **Model's Association of Concepts with Prompts**: `@JH` clarified that a model would likely recognize the concept of a bowtie, but the association with the specific word “bow tie” would decrease if it was rarely captioned as such. For instance, if bowtie images were majorly from weddings and captioned with "wedding", the model might generate bowties in images when prompted with a wedding scenario.
- **AI Models Generating Unprompted Artifacts**: `@JH` further explained that the model's association of concepts from captions could lead to unprompted artifacts. For instance, if it frequently associates bowties with weddings, it may add bowties to any wedding scene generated, even if not explicitly prompted.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Discussion on **data deduplication** with a focus on using cosine similarity in Python for text columns, sought guidance by `@robotics3483`.
- Experience sharing by `@chokipro` on an error encountered, suspected to be related to **Hugging Face**. This error, also faced by others, was humorously disambiguated by `@ddchiken` as a code indicator for 'I'm a teapot'.
- `@epicureus` initiated a dialogue on how to convert a **legacy app to Laravel** - illustrating his experience with this [blogpost](https://tighten.com/insights/converting-a-legacy-app-to-laravel/). He noted troubles with outdated codes and a strong temptation to start afresh.
- `@kopyl` enquired about a locally cached dataset that was no longer available on HuggingFace, required for their miniSDXL training plans.
- A query on the specifics of **Mistral Model**-number of training or fine-tuning data samples was posted by `@stoicbatman`.
- Switching to Windows from MacBook Pro for a 3D plants business was explored by `@woutdeclerck`, with a planned use of Blender and Unreal Engine.
- `@duplaja` shared his learning journey about Hugging Face Inference Endpoints in the handler.py context.
- Insight into adjustments of parameters without modifying Andrej’s code was provided by `@gag123`, in response to a query by `@waffle_cream`.
- Positive appreciations and recommendations of the blog posts and future content shared in the guild, notably the video presentation by `@723709999452389417`.
- `@ddchiken`'s project link noted its connection to AINews while `@vikasexcel` announced their **published model for Hindi**-[open-aditi-hi-v1](https://huggingface.co/manishiitg/open-aditi-hi-v1), along with an open-sourced dataset.
- Community members encouraged to propose papers for discussion in the reading-group channel by `@lunarflu`.
- Emergence of the idea about **Langchain-backed ChatGPT** client's application in legal contexts was shared by `@_johnny1984`.
- Queries on **text to SQL conversion models** on Hugging Face by `@p_k_boo` and plans of `@ketul1842` to train a small LLM to convert natural queries to conditions in SQL statements, with guidance on various related aspects sought.
- Model suggestions to `@ketul1842`'s query were provided by `@stroggoz`, including encoder-decoder models for translation, next sentence prediction concerning BERT, and doubts regarding the latter's effectiveness.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (36 messages🔥): 
        
- **Data Deduplication Request**: `@robotics3483` asked for help with grouping and deduplication in Python using cosine similarity on a text column.
- **Error Encountered**: `@chokipro` shared a [link to an error message](https://discord.com/channels/879548962464493619/1191085523566071868) they were experiencing, suggested it was an issue with **Hugging Face** that was also encountered by others. `@ddchiken` jokingly clarified that**http status 418** is 'I'm a teapot'.
- **Legacy Code Conversion**: `@epicureus` shared a [link to a blogpost](https://tighten.com/insights/converting-a-legacy-app-to-laravel/) about converting a legacy app to Laravel and reflected on challenges with outdated code and the temptation to rebuild from scratch.
- **Dataset Requests**: `@kopyl` was looking for a locally cached dataset `ChristophSchuhmann/improved_aesthetics_6plus` which is no longer available on [HuggingFace](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus) as they planned to train miniSDXL.
- **Mistral Model Inquiry**: `@stoicbatman` asked about the number of samples (training or finetuning data size) used for the Mistral model, but could not find any information related to the dataset.

**Links mentioned**:

[Legacy to Laravel: How to Modernize an Aging PHP Application](https://tighten.com/insights/converting-a-legacy-app-to-laravel/): Many of our clients have legacy PHP apps and want ...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 messages): 
        
- **Switching from MacBook Pro to Windows for 3D Plants Application**: `@woutdeclerck` seeks advice on whether to switch from MacBook Pro to a windows computer, specifically one with RTX GeForce 4090 for a business related to 3D plants for CG application. They plan to use Blender and Unreal engine.
- **Learning HF Inference Endpoints**: `@duplaja` mentioned that they are gradually learning their way around handler.py Hugging Face Inference Endpoints.
- **Adjusting Parameters**: In response to `@waffle_cream`'s query, `@gag123` clarified that they did not modify Andrej's code but are just experimenting with the parameters.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (4 messages): 
        
- **Appreciation for content**: User `@osanseviero` expressed their enjoyment of a video by `@723709999452389417`, calling it "very, very nice".
- **Suggestion for future content**: `@jartine` expressed interest in receiving ideas for future blog posts.
- **Acknowledgement of content**: `@nikocof_63920` responded briefly with "tnks".


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (3 messages): 
        
- **Project Linked to AINews**: User `@ddchiken` mentioned that their project was linked to AINews, possibly by someone from the server.
- **Publication of Hindi Model**: User `@vikasexcel` shared that they have published a model, namely [open-aditi-hi-v1](https://huggingface.co/manishiitg/open-aditi-hi-v1), specifically fine-tuned for Hindi. They also open-sourced the dataset.
- User `@nikocof_63920` responded with a simple "oh...", possibly expressing surprise or interest.


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 messages): 
        
- **Community-led Paper Suggestions**: User `@lunarflu` encouraged a newcomer, `@Zack` to suggest any interesting paper they feel like discussing since the group is community-led.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Langchain-Backed ChatGPT in Legal Contexts**: User `@_johnny1984` introduced a potential use case for a **Langchain-backed ChatGPT** client filled with specialized attorney/judge/psychiatrist agent. The proposed scenario was about having the AI system analyze different parties' viewpoints and relevant court documents to make a judgement, particularly for complex and sensitive cases such as child custody disputes.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (7 messages): 
        
- **Text to SQL Models**: User `@p_k_boo` asked for suggestions on models for text to SQL conversion available on HuggingFace. They reported having problems running some models locally due to crashes. 
- **Data Deduplication Using Cosine Similarity**: `@robotics3483` is seeking help for data deduplication by grouping using cosine similarity with a threshold greater than 0.95 in Python.
- **Training a Language Model (LLM) for Natural Queries to Conditions Conversion**: `@ketul1842` expressed their intention to train a small LLM to convert natural language queries to conditions in SQL statements. They asked for guidance on dataset generation, selection of an open-source model, fine-tuning, evaluation, and deployment. They also welcomed any resources related to this subject.
- **Model Suggestions by @stroggoz**: In response to `@ketul1842`'s query, `@stroggoz` suggested looking into encoder-decoder models used for machine translation or paraphrasing. They also mentioned the possibility of using a model trained for Next Sentence Prediction, like BERT, but expressed doubts about its effectiveness in this case. They suggested creating a dataset with odd sentences acting as prompts and even sentences representing boolean logic.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Application of Langchain-backed ChatGPT client in law**: User `@_johnny1984` raised a question about the potential application of a Langchain-backed ChatGPT client in making legal judgements. They provided a specific example of a cruel child custody case. However, the user didn't provide further details nor did they receive direct answers to their question.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- Discussion about **GEGLU/SWIGLU's effectiveness and scalability** in comparison to GeLU for training LLMs with diverse perspectives shared by `@fer.bear`, `@ad8e`, and `@catboy_slim_`. [[Discussion here](https://discord.com/channels/general)]
- Benchmark results for the **Mistral 7B Instruct V0.2 model** were posted, indicating variable performance dependent on input/output length. PyTorch errors were noted to be ignorable and occurring due to vRAM shortages. [[Details here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance.md)]
- Technical issue raised about **models predicting EOS token prematurely post extra pretraining and fine-tuning**, sparked interest and prompted potential solutions by `@ad8e`.
- Suggestions requested for **customizable VS Code extensions offering Copilot-like features** using local models. Received suggestion for [`code-clippy-vscode`](https://github.com/CodedotAl/code-clippy-vscode).
- Query about the **order of checkpoints during Pythia model training** clarified by `@stellaathena`, explaining the sequence of step 11000 after step 10900 under HuggingFace's UI representation.
- Notable question posed about the conceptual equivalence of **Transformers and MLPs**, and whether a similar dynamic exists for convolutional kernel weights.
- Suggestions made for "dynamic convolution" and local attention as potential responses to the question about **Transformers and MLPs**.
- The **computational cost of Transformers** was pointed out, noting that their usage of computations and weight linear projections could be seen as wasteful, sparking further exploration on the necessity of these linear projections.


**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (27 messages🔥): 
        
- **GEGLU/SWIGLU's Effectiveness and Scalability**: There is a discussion about the effectiveness of GEGLU/SWIGLU compared to GeLU in training LLMs. `@fer.bear` believes that they improve long-term training performance, but `@ad8e` and `@catboy_slim_` mention a lack of robust evidence supporting that claim ([Discussion here](https://discord.com/channels/general))
- **Mistral 7B Instruct V0.2 Benchmark Results**: `@flow7450` shared benchmark numbers for the mistral 7b instruct v0.2 model with statistics on various input and output lengths. Performance varied depending on input/output lengths. Information suggests PyTorch errors were ignorable and occur when it runs out of vRAM ([Details here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance.md)).
- **Issue with Fast EOS Token Predictions After Fine-tuning**: `@evil_malloc` raised an issue about models predicting the EOS token too fast after additional pretraining and fine-tuning. `@ad8e` suggested that correcting certain methods might resolve this.
- **VS Code Extensions for Copilot-like Features**: `@danielpgonzalez` requested suggestions for hackable VS Code extensions that provide copilot-like features using local models. `@nate.dawgg` suggested [`code-clippy-vscode`](https://github.com/CodedotAl/code-clippy-vscode), which supports local model serving.
- **Order of Pythia model Training Checkpoints**: `@wolferk` sought clarification on the order of checkpoints in Pythia model training. `@stellaathena` clarified that step11000 comes after step 10900, but HuggingFace's UI displays them in alphabetical order.

**Links mentioned**:

- [React App](https://main--shiny-raindrop-121903.netlify.app/)
- [GitHub - jondurbin/qlora: QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/jondurbin/qlora): QLoRA: Efficient Finetuning of Quantized LLMs. Con...
- [GitHub - CodedotAl/code-clippy-vscode: VSCode extension for code suggestion](https://github.com/CodedotAl/code-clippy-vscode): VSCode extension for code suggestion. Contribute t...
- [TensorRT-LLM/docs/source/gpt_attention.md at d37b507f41a87457fe9f10f7459d08f5db235745 · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/docs/source/gpt_attention.md#int8fp8-kv-caches>): TensorRT-LLM provides users with an easy-to-use Py...
- [TensorRT-LLM/docs/source/precision.md at d37b507f41a87457fe9f10f7459d08f5db235745 · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/docs/source/precision.md#technical-detail-the-quantmode-flags>): TensorRT-LLM provides users with an easy-to-use Py...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (4 messages): 
        
- **Transformers vs MLPs**: `@voxs` posited that transformers can be thought of as multi-layer perceptrons (MLPs) but with weights that are recalculated based on input. They asked if a similar approach exists for convolutional kernel weights. 
- **Dynamic Convolution Suggestion**: In response to `@voxs`'s query, `@kharr.xyz` suggested looking up "dynamic convolution", indicating that there are quite a few papers on the subject.
- **Local Attention Suggestion**: `@theseriousadult` proposed local attention as a possible answer to `@voxs`'s question.
- **Discussion on Transformers Computation**: `@fern.bear` commented on the computational cost of Transformers. Noting that Transformers could be seen as unnecessarily wasteful with computations and weight linear projections, `@fern.bear` is exploring the extent to which these linear projections are necessary.


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (1 messages): 
        
sk5544: Perfect!!!


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Discussion on **Mistral-medium and Mixtral performance**, with varying strengths and weaknesses noted in certain tasks, as well as inquiries concerning the waiting time for account setup on Mistral.
- Insights on **model performance across different hardware configurations**, including the effects of GPU offload, RAM and VRAM sharing, with specific mentions of Nous-Hermes-2-Yi-34B-GGUF Q8 and MiXtral V0.1 Q4 models.
- Interest expressed in **integrating Mistral's API with the AutoGen (Studio) UI Assistant Agent**, though no solutions were offered.
- Questions regarding **run time on GPU vs CPU** for large token numbers, highlighted by a specific case with the mixtral-8x7b-instruct-v0.1.Q5_0.gguf model; issues explained in part due to the model's inability to fully fit on the user's GPU.
- Exploration of **physics concepts**, like horsepower, work, energy, and power, including their distinct definitions, SI units, and practical examples.
- Inquiry about **timing for subscription fees** across different channels.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (15 messages🔥): 
        
- **Performance of Mistral-medium vs Mixtral**: Users `@i_am_dom` and `@.skyair` had a discussion regarding the performance comparison between Mistral-medium and Mixtral. They concluded that Mistral-medium does certain tasks better than Mixtral while Mixtral outperforms Mistral-medium in certain other tasks.

- **Wait-list for Mistral account**: User `@kiritz_x` asked about the typical wait time for an account setup and leaving the wait-list in Mistral.

- **Integration of Mistral's API with AutoGen (Studio) UI Assistant Agent**: User `@jb_5579` sought advice on integrating Mistral's API with the AutoGen (Studio) UI Assistant Agent. However, no solutions were given.

- **Queries about Run Time with GPU vs CPU**: User `@gilford3641` asked why it takes longer to run an inference on 3-4k tokens using a GPU compared to using a CPU on Windows system having mixtral-8x7b-instruct-v0.1.Q5_0.gguf model. `@casper_ai` responded that the model doesn't fit on the user's GPU, which slows down the process due to the requirement for constant communication between the CPU and GPU.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (7 messages): 
        
- **Nous-Hermes-2-Yi-34B-GGUF Q8 Performance with Different Setup**: User `@.gue22` shared their experience in testing Nous-Hermes-2-Yi-34B-GGUF Q8 model on a 256GB Xeon setup. They observed that this model required a large amount of RAM and operated at a slow pace without GPU offload.
- **MiXtral v0.1 Q4 Performance and Model Sizes**: They also tested the MiXtral V0.1 Q4 model on an M3 and found it to load quickly and perform responses rapidly, regardless of GPU being on or off.
- **Comparison of Model Performance on Different Hardware**: User `@fayiron` revealed their experience with trying different model sizes. They noted better efficiency (49 tokens/s) when a model fully runs on the GPU compared to when RAM and VRAM are shared.
- **Performance of MiXtral 8x Instruct 7b Q4 on MacBook Pro**: User `@.gue22` noted the MiXtral 8x Instruct 7b Q4 model loaded in 40 seconds and outputted at a rate of 30 tokens/s on an M3 Max 36GB MacBook Pro.
- **Unexpected Findings in Model Performance**: In a subsequent test, `@.gue22` found the Xeon operating at half the speed of the M3 Max, when loading the MiXtral 8x Instruct 7b Q4 26GB model. They found this result intriguing and questioned why it happened.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (4 messages): 
        
- **Subscription Fees Inquiry**: User `@alimsss` asked about the timing for subscription fees charge.
- **Explanation of Horsepower**: User `@dryousefsharrab` explained the concept of horsepower and differentiated between mechanical horsepower and metric horsepower.
- **Discussion on Work, Energy, and Power**: `@dryousefsharrab` also detailed the distinctive meanings of work, energy, and power in physics, including their respective units in the International System of Units (SI).
- **Practical Examples on Work, Energy, and Power**: `@dryousefsharrab` provided real-life examples to illustrate the concepts of work, energy, and power. These included moving a box (work), a battery powering a flashlight (energy), and a car engine's power output (power).


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Discussion on the **token usage in OpenAI API**, specifically on how document chunk size impacts API usage pricing and its influence on relevant information retrieval with a query from `@swatchap`.
- **Technical issues** pertaining to streaming in **LangChain's `ChatGoogleGenerativeAI` and `LLMChain`** reported by `@hasan_34148`, with an open GitHub [issue](https://github.com/langchain-ai/langchain/issues/14709) for the same.
- Proposal of an **alternative streaming method** by `@seththunder` - essentially delaying each letter in a response. However, the practicality of this method was questioned by `@rajib2189` based on the essential purpose of streaming.
- **LangChain Model Parameter Discussion**: `@coorbin` initiated a conversation on the possible quality differences in LangChain output if different parameter models are used for embeddings and inference.
- **Langchain Quickstart Example Error**: `@manuel_24767` reported a 'ValidationError' while running a LangChain example related to creating a retrieval chain, providing traceback error log for further support.
- **New research paper** titled *The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey* shared by `@dhruvdh` in the #share-your-work channel. The paper details various LLM system designs and includes a unique thought experiment. The full paper can be accessed [here](https://arxiv.org/abs/2312.17601).
- `@dhruvdh` also created a [Reddit post](https://www.reddit.com/r/MachineLearning/comments/18w09hn/r_the_tyranny_of_possibilities_in_the_design_of/) summarizing the conjectures from the paper, outlining the uncommon utilization of a thought experiment in such research.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (18 messages🔥): 
        
- **Understanding Token Usage in OpenAI API**: `@swatchap` expressed need for clarifying how OpenAI's API usage pricing is affected by document chunk size. They questioned if document chunk size influences the retrieved relevant information and subsequently counted as input tokens. 
- **Issues with Streaming in ChatGoogleGenerativeAI & LLMChain**: `@hasan_34148` reported that they are facing issues with LangChain's streaming functionality in relation to `ChatGoogleGenerativeAI` and `LLMChain`. They have an open GitHub [issue](https://github.com/langchain-ai/langchain/issues/14709) regarding the problem.
- **Alternative Streaming Method**: `@seththunder` suggested an alternative method for implementing streaming, essentially delaying each letter in a response. The method, however, was questioned by `@rajib2189` asserting that it defeats the purpose of streaming, which should ideally send the token as soon as it is generated. 
- **Ollama Embeddings and Model Parameter Equivalence**: `@coorbin` posed a question regarding the use of different parameter models for embeddings and inference in LangChain. They queried if there could be a quality difference in the output, allowing for more parameter-efficient models to generate embeddings more or less equivalently.
- **Running Langchain Quickstart Example Error**: `@manuel_24767` shared an issue they faced while running a LangChain example related to creating a retrieval chain. The error they encountered pointed towards a 'ValidationError' upon invoking the retrieval chain. They detailed the traceback error log for further troubleshooting.

**Links mentioned**:

- [LangChain Expression Language (LCEL) | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/): LangChain Expression Language, or LCEL, is a decla...
- [Redirecting...](https://errors.pydantic.dev/2.4/v/missing)
- [CallbackHandler on_llm_new_token method not fire with ChatGoogleGenerativeAI(Gemini) but works fine with ChatOpenAI when streaming true · Issue #14709 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/14709): System Info aiohttp==3.9.1 aiosignal==1.3.1 annota...


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **New Paper on Task-Oriented LLM Systems**: User `@dhruvdh` has shared a new research paper titled *The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey*. The paper presents a scoping survey on task-oriented LLM system designs and various related parameters. It includes a thought experiment discussing the performance of different LLM system configurations on complex tasks. [Download the Paper](https://arxiv.org/abs/2312.17601).
- **Reddit Post Summarizing the Paper's Conjectures**: `@dhruvdh` also made a [Reddit post](https://www.reddit.com/r/MachineLearning/comments/18w09hn/r_the_tyranny_of_possibilities_in_the_design_of/) summarizing the seven conjectures from the paper. One key aspect highlighted is the use of a thought experiment, which is unusual in this research area.

**Links mentioned**:

- [The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey](https://arxiv.org/abs/2312.17601): This scoping survey focuses on our current underst...
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/18w09hn/r_the_tyranny_of_possibilities_in_the_design_of/)


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI Recap of 2023**: A comprehensive review of AI developments in 2023, with a special focus on Large Language Models (LLMs), was shared by @swyxio. This includes discussions on the ease of building LLMs, running them on devices, hobbyist fine-tuning, challenges with GPT-4, and more. 
    - [Stuff we figured out about AI in 2023](https://simonwillison.net/2023/Dec/31/ai-in-2023/)
- **LLM Course & AI Notes**: An open-source course for LLMs providing roadmaps and Colab notebooks was mentioned, alongside updated AI notes adding more recommended reads from December 2023.
    - [Course to get into Large Language Models (LLMs)](https://github.com/mlabonne/llm-course)
    - [Update all the AI notes](https://github.com/swyxio/ai-notes/commit/650319e6cb5423a684200255de3ec200ef9953f0)
- The **Mergekit** section from the course and a visual representation of the latest trends, especially GPT-4's dominance in open models, were also put forth.
    - [Visualization of Open Models](https://www.reddit.com/r/LocalLLaMA/comments/18r56fq/chatbot_arena_elo_ratings_overtime/?share_id=RmHGHSeMlTsL2-Om4UjyV&utm_content=1&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1)
- **State Machine of Thought (SMoT)**: A research paper proposing a new paradigm to enhance problem-solving within LLMs using predefined state machines was highlighted.
    - [SMoT: Think in State Machine](https://arxiv.org/abs/2312.17445)
- **Mixture of Experts (MoEs) Transformers**: A HuggingFace blog post discussing the building blocks, training, and tradeoffs of MoEs in detail, prompted by the release of Mixtral 8x7B, was shared.
    - [Mixture of Experts Explained](https://huggingface.co/blog/moe)

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (11 messages🔥): 
        
- **AI Recap of 2023**: `@swyxio` shared a round-up of AI in 2023, highlighting Large Language Models (LLMs) as the major development in the field. The recap details multiple aspects of LLMs including their ease of build, running them on devices, fine-tuning by hobbyists, and the challenges in building GPT-4. ([Link to Recap](https://simonwillison.net/2023/Dec/31/ai-in-2023/))
- **LLM Course**: An open-source course for getting into Large Language Models (LLMs) was shared by `@swyxio`, which includes roadmaps and Colab notebooks. The course is suitable for Louisiana State University as indicated in the message. ([Link to Course](https://github.com/mlabonne/llm-course))
- **Mergekit and AI**: `@swyxio` pointed out the relevance of the Mergekit section in the shared course, especially in the current scenario.
- **AI Notes**: Further, `@swyxio` shared a link to updated AI notes offering more recommended reads from December 2023. ([Link to Notes](https://github.com/swyxio/ai-notes/commit/650319e6cb5423a684200255de3ec200ef9953f0))
- **Visualization of Open Models**: A visualization showing the latest trends in open models, particularly the dominance of GPT-4, was shared by `@swyxio`. The visualization is based on data from the last six months and was obtained from a Reddit post on LocalLLaMA. ([Link to Visualization](https://www.reddit.com/r/LocalLLaMA/comments/18r56fq/chatbot_arena_elo_ratings_overtime/?share_id=RmHGHSeMlTsL2-Om4UjyV&utm_content=1&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1))
- **State Machine of Thought (SMoT)**: `@davidkpiano` shared a link to a research paper introducing SMoT, a novel paradigm that improves problem-solving within LLMs by employing predefined state machines, thereby eliminating fruitless exploration. ([Link to Paper](https://arxiv.org/abs/2312.17445))

**Links mentioned**:

- [Stuff we figured out about AI in 2023](https://simonwillison.net/2023/Dec/31/ai-in-2023/): 2023 was the breakthrough year for Large Language ...
- [SMoT: Think in State Machine](https://arxiv.org/abs/2312.17445): Current prompting approach for language model infe...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18r56fq/chatbot_arena_elo_ratings_overtime/?share_id=RmHGHSeMlTsL2-Om4UjyV&utm_content=1&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1)
- [GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.](https://github.com/mlabonne/llm-course): Course to get into Large Language Models (LLMs) wi...
- [update all the notes · swyxio/ai-notes@650319e](https://github.com/swyxio/ai-notes/commit/650319e6cb5423a684200255de3ec200ef9953f0)


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages): 
        
- **Mixture of Experts (MoEs) Transformers**: `@swyxio` shared a link to an accessible [blog post](https://huggingface.co/blog/moe) from HuggingFace discussing **Mixture of Experts (MoEs)**, a hot topic in the open AI community following the release of **Mixtral 8x7B**. The blog post delves into the building blocks, training, and tradeoffs of MoEs when serving them for inference.

**Links mentioned**:

[Mixture of Experts Explained](https://huggingface.co/blog/moe)


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- In the AI and ML discussion, user `@fred_fups` expressed **issues encountered while training the Mistral 7b model**, specifically generating incomplete responses and excessive newline characters. They also shared an extract of the training file and requested insights from other members.
    - "*I trained the Mistral 7b model on text formatting examples... After training, the model produces incomplete responses and repeatedly generates newline characters (`'\n'`) till reaching the output limit*"

- New Year greetings and interaction dominated the `oo` channel; `@damiondreggs` shared a [Kermit The Frog GIF](https://tenor.com/view/kermit-the-frog-meme-memes-gif-24277198) for the occasion.

- The Looking-for-Work channel featured `@klrshak`, a soon-to-be graduate with experience in autonomous vehicles, inquiring about **remote summer internships and long-term job opportunities** in similar areas. This user expressed an aspiration to eventually pursue a PhD. They called for suggestions for research labs or companies in their field of interest.

**Alignment Lab AI Channel Summaries**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (4 messages): 
        
- **Training Mistral Model Issue**: User `@fred_fups` reported an issue with the **Mistral 7b** model he trained on text formatting examples. After training, the model produces incomplete responses and repeatedly generates newline characters (`'\n'`) till reaching the output limit.
- **Seeking Help with Model Training**: `@fred_fups` asked if anyone has encountered a similar issue or has theories on what might be the root cause.
- **Sharing Training File Example**: `@fred_fups` shared an extract from the training file, the context being enhancing the readability of a text by applying specific formatting changes. He expressed being new to AI training and openness to potential issues within his training set.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (5 messages): 
        
The messages in the `oo` channel consisted of:
- New Year greetings from `@nanobitz`, `@imonenext`, and `@entropi`.
- `@damiondreggs` sharing a GIF via a link [https://tenor.com/view/kermit-the-frog-meme-memes-gif-24277198](https://tenor.com/view/kermit-the-frog-meme-memes-gif-24277198) and acknowledging the New Year.

**Links mentioned**:

[Kermit The GIF - Kermit The Frog - Discover &amp; Share GIFs](https://tenor.com/view/kermit-the-frog-meme-memes-gif-24277198): Click to view the GIF


### ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/) (1 messages): 
        
- **Internship and Long-Term Opportunities Inquiry**: `@klrshak`, a student graduating before the summer with experience in scene understanding and perception for Autonomous Vehicles, is interested in both a summer internship and long-term job opportunities within similar fields. They aim to pursue a PhD in the long run. They are specifically seeking suggestions for research labs or companies doing commendable work in this domain that would be open to remote internships now and possibly in-person internships within the summer.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Users** `@teknium`, `@far_el`, and `@oleegg` **exchanged "Happy New Year" wishes** in general conversations.
- A **discussion on the output of a chatbot** initiated by `@teknium` drew responses from `@caviterginsoy` and `@leuyann`, but the context remains unclear.
- `@walter8967` initiated a discussion on the **value of text annotation for multitask training**, referring to how annotated images can reduce the data requirements for image generation (source not cited).
- In an alternative approach, `@walter8967` suggested that **using more text could possibly improve training**.
- In off-topic, `yusufhilmi_` **sought information on a SAM-like model for segmenting graphic arts and user interfaces**.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (6 messages): 
        
- **General Conversations**: Users shared warm wishes, with `@teknium`, `@far_el`, and `@oleegg` wishing "Happy New Year". 
- **Chatbot output discussion**: User `@teknium` made a comment that was met with amusement, to which `@caviterginsoy` and `@leuyann` responded. However, the context or topic of the discussion is unclear from these messages.


### ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/) (2 messages): 
        
- **Value of Text Annotation for Multitask Training**: `@walter8967` queried the potential benefits of annotating text, such as identifying parts of speech or manually disambiguating, for multitask training. They mentioned that annotated images were found to reduce data requirements for image generators, though the source of this information was not cited.
- **Alternate Approach - More Text**: `@walter8967` also pondered if alternatively, accruing more text would be a better approach to improve training.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
yusufhilmi_: does anyone know a SAM like model for segmenting graphics arts and user interfaces


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Adding German to MTEB**: `@rasdani` is working on a fork/PR for Multilingual Text Embeddings Benchmark with MRR@10 on GermanQuAD. They shared a link to an interesting [GitHub issue](https://github.com/embeddings-benchmark/mteb/issues/183) suggesting possible datasets to consider.
- **Testing MTEB on GermanQuAD**: `@rasdani` posted results from a bug-free run of the Multilingual Text Embeddings Benchmark on the entire GermanQuAD test set using the `intfloat/multilingual-e5-small` model. They used the dot product as recommended and kept all default metrics.
- **Concerns with Dot Product Method**: `@philipmay` commented that the model has to be trained with the dot product as part of the loss function, not just by using dot product instead of cosine similarity. Distance function should correspond with the training method.
- **Low MRR@10 Value**: `@philipmay` pointed out that the MRR@10 value of 0.3908 is low.
- **Improved Results with a Subset**: `@rasdani` reported improved results when evaluating on a subset of 100 and 150 unique contexts. They also commented that the BEIR library, which is used in MTEB, is picky about dataset format. They plan to publish their fork for review.

**Links mentioned**:

[Adding German to MTEB · Issue #183 · embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb/issues/183): Hi everybody, I think it would be great to add Ger...

        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Virtual Event: Infer - The Engineering Behind AI and ML**: User `@amitqwak` announced a virtual event titled **Infer**, which is aimed at ML Engineers, Data Scientists, Data engineers, Software engineering managers, MLOps practitioners. The event is designed to connect leaders in Machine Learning and AI, showcase how top companies apply ML and AI in practice, highlight challenges and strategies in utilizing ML/AI in production, and offers insights into the latest trends and advancements in the field. The event is free to attend and is scheduled for March 20th, 2024 at 11:30 AM EST. Registration for the event can be made via [this link](https://www.qwak.com/infer/infer-march-2024?utm_source=Chip_Hyuen&utm_medium=Discord&utm_campaign=Infer_March20). Interested participants can submit a talk for the event [here](https://forms.gle/ogY5xcPHVhc7iSQg8). The agenda is yet to be announced.

**Links mentioned**:

[Infer by Qwak | The Engineering Behind AI and ML](https://www.qwak.com/infer/infer-march-2024?utm_source=Chip_Hyuen&utm_medium=Discord&utm_campaign=Infer_March20): Infer by Qwak brings ML and AI leaders to share ho...

        

---

## [Datasette/LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Document AI and Open Source Models**: In the discussion, `@stephen_88734` shared a link to a [Hugging Face blog post](https://huggingface.co/blog/document-ai) discussing various Document AI tasks and how open-source models can be utilized to unlock information from various types of documents. It covers tasks like **image classification**, **image to text**, **document question answering**, **table question answering**, and **visual question answering**. Recommended models include **Donut** and **LayoutLM**.

**Links mentioned**:

[Accelerating Document AI](https://huggingface.co/blog/document-ai)

        
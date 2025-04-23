---
id: 1f96b623-39c8-477a-9333-7ef7fb3223b1
title: '[AINews] 1/6-7/2024: LlaMA Pro - an alternative to PEFT/RAG??'
date: '2024-01-08T00:51:41.330707Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-16-72024-llama-pro-just-add-new-layers-lol
---

<!-- buttondown-editor-mode: plaintext -->> This is 2 day's worth of content because we missed yesterday.

New papers released show very promising Llama Extensions:

- [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385): (we wrote about this in previous AINews issues, but this is the paper) We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. ([written up on Semafor](https://www.semafor.com/article/11/01/2023/microsoft-pushes-the-boundaries-of-small-ai-models))
- [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415): LLaMA-Pro is an 8.3 billion parameter model. It's an expansion of LLaMA2-7B, further trained on code and math corpora totaling 80 billion tokens
  - Adding new knowledge to Llama7B without catastrophic forgetting... by just adding layers lol
  - gives it a nice tradeoff between language and code tasks  ![image.png](https://assets.buttondown.email/images/b5bbe5df-d1fa-42c0-8057-8baee739865f.png?w=960&fit=max) 

But it is getting some scrutiny already for basing on  LlaMA and [not using Mistral/Qwen/etc](https://x.com/teortaxesTex/status/1743421078649643407?s=20):

 ![image.png](https://assets.buttondown.email/images/15f2048b-37cf-4cd6-a286-9f2988a55e57.png?w=960&fit=max) 

Yannic Kilcher already has a great Llama Pro explainer out:

https://www.youtube.com/watch?v=hW3OVWfndLw

In other news, LangChain is planning to promote their recent v0.1 next week.

---

**Table of Contents**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Enterprise-Lite, Right? Not**: User `@a1vx` inquired about the possibility of individual users availing themselves of an Enterprise package, but was promptly informed by `@solbus` that this offer currently only caters to large corporations with six-figure contracts. 
- **Privacy Fears and GPT Reassurances**: There have been concerns raised around the privacy aspects when using GPT models by user `@a1vx`. `@feltsteam0` reassured that the private data is kept confidential and unlikely to be used in model pretraining.
- **Not One Language Fits All**: Queries existed regarding the support for other languages like Russian in the imminent GPT Store. The community suggested to wait for further details post-launch of the store.
- **Fine-tuning for Models, Not Just Motors**: For the ongoing search of refining GPT-3.5 1106 for use in Assistants, `@a1vx` directed users towards the Assistant API page of OpenAI's official documentation.
- **The Traceroute is Not the Route**: A discussion between `@darthgustav.` and `@chotes` about the model's operation in multi-dimensional vector space ended with skepticism towards the accuracy and common misconceptions related to interpreting a model's response as a traceroute.
- **GPT-4: Tokens Galore, More is More**: Token usage cap and desired full-time access related discussions were touched on by `its_eddy_` and `Bianca Stoica` in relation to GPT-4.
- **The Builders Need Verifications**: Verifying domains during the set-up of GPT-Builders has turned out to be thorny for `johan0433` and `savioai`, though efforts have been made to resolve the issue.
- **GPTs Flexing Language Muscles**: `xbtrob` and `elektronisade` highlighted that GPT-4 may have difficulty recognizing non-Latin languages from images, namely kanjis.
- **Caught Between HTML Rock and CSS Hardplace**: `kerubyte` is grappling with controlling generated HTML and CSS code in their custom GPT, which seems to have an unsolicited affinity for spawning `<html>` and `<body>` elements.
- **Custom GPTs for Who, Now?**: `filipemorna` sought information regarding the enrollment process for GPT builders and was informed by `solbus` that access is possible with a **ChatGPT Plus** account.
- **GPTs: Not a Fan of Red Tape**: User `.ashh.` questioned if creating GPTs for specific video games could infringe on copyrights.
- **Playing Hide and Seek with GPT-4**: The common issue of GPT-4 being inaccessible was discussed by `its_eddy_` and `Bianca Stoica`, touching on usage limit and token limits.
- **Dynamic Image Generation via DALL-E for Breakfast**: User `@hawaiianz` shared experiences and advice on how to create more balanced and dynamic images with DALL-E by being creative with prompts and descriptions.
- **More Locks for More Safety**: Security concerns in deploying GPT was a point of contention amongst users `@cerebrocortex`, `@aminelg`, `@madame_architect` and `@wwazxc`. Users were advised against breaking OpenAI's policies while exploring new security methods.
- **Logo-ing with DALL-E is Dope**: `@viralfreddy` sought advice on creating logos with DALL-E, and `@darthgustav` suggested specific prompts and English translations for effectiveness.
- **GPT's Task List is Overflowing**: Users like `@ajkuba`, `@wwazxc` and `@cosmodeus_maximus` were vexed at the GPT's handling of large tasks and its unresponsiveness at times.
- **The Treacherous Path to Prompting Success**: `@exhort_one` shared their journey in prompt engineering, which, after several iterations and learning from the community, led to a successful generation of the desired script.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (241 messages🔥🔥): 
        
- **Enterprise AI for Individual Users? Not Yet**: User `@a1vx` enquired about the possibility of purchasing an Enterprise package for individual use. User `@solbus` clarified that currently, OpenAI only caters to large companies with six-figure contracts for Enterprise and suggested contacting the OpenAI sales team for negotiation.
- **Concerns about Privacy with ChatGPT**: `@a1vx` raised concerns regarding privacy while using GPT models, especially regarding the use of "Chat history & training" and data storage for moderation. `@feltsteam0` reassured that the data is highly unlikely to end up in the pretraining data and that 'private' chats are probably not used for training.
- **Incorporating Private GPTs in the GPT Store**: Questions were raised about the upcoming GPT Store and whether it would allow GPTs trained to understand other languages, especially Russian. It was noted that such details were not present in the current documentation, and would hopefully be clarified when the Store is launched.
- **Fine-tuning GPT Models for Assistants**: User `@s44002` asked whether it is possible to use a fine-tuned version of GPT-3.5 1106 as a model for the assistant. `@a1vx` referenced the Assistant API page in OpenAI's documentation and noted that support for fine-tuned models in the Assistant API is coming soon.
- **'The Traceroute Metaphor' Discussion**: A long technical discussion was held between `@darthgustav.` and `@chotes` regarding the operation of language models in a high-dimensional vector space. The idea of considering a model's responses as a traceroute through this space was met with skepticism and debate about the accuracy and misunderstandings of this metaphor.


**Links mentioned**:

- [Conversa ](https://arcaii.github.io/Conversa/)
- [Pricing](https://openai.com/pricing): Simple and flexible. Only pay for what you use.
- [GitHub - SuperDuperDB/superduperdb: 🔮 SuperDuperDB. Bring AI to your database; integrate, train and manage any AI models and APIs directly with your database and your data.](https://github.com/SuperDuperDB/superduperdb): 🔮 SuperDuperDB. Bring AI to your database; integrate, train and manage any AI models and APIs directly with your database and your data.  - GitHub - SuperDuperDB/superduperdb: 🔮 SuperDuperDB. Bring....


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (97 messages🔥🔥): 
        
- **GPT Usage Cap and Subscription Discussions**: User `its_eddy_` responded to concerns about the usage cap on **GPT-4**, helping clarify availability times for free users. `Bianca Stoica` asked about gaining full-time access to GPT-4.
- **GPT Builder Domain Verification Troubleshooting**: Users `johan0433` and `savioai` shared their experiences with issues in verifying their respective domains when setting up GPT Builders. As part of this discussion, `johan0433` suggested inputting "www" as the Name field to resolve the issue. However, this solution was not successful for `savioai`.
- **Anticipated Launch of GPT Store**:`solbus` mentioned that OpenAI announced the upcoming launch of the GPT Store in an email to users. This prompted further discussion with `jguzi` over the availability of a directory of custom GPT bots.
- **Issues with GPT Knowledge Recognition**: `alanwunsche` raised concerns over his GPT being unable to recognize small uploaded files, identifying this as a common issue among users.
- **Mobile Incompatibility of Custom GPT Actions**: `dorian8947` reported issues with custom actions on his GPT in a mobile application, seeking help in resolving these issues.
- **HTML and CSS Custom GPT development Problems**: User `kerubyte` is struggling to control generated CSS and HTML code with their custom GPT. Their **GPT** keeps generating `<html>` and `<body>` elements even though they explicitly instructed it not to in the rule set.
- **Technical Issue in Non-Latin Languages**: `xbtrob` noted that **GPT-4** cannot read kanjis from pictures, which was confirmed by `elektronisade`, stating non-Latin alphabets such as Japanese or Korean may not be processed optimally.
- **Enrollment Enquiry for GPT Builders**: `filipemorna` asked about the enrollment process to become a GPT builder, to which `solbus` explained that anyone with a **ChatGPT Plus** account can create a custom GPT.
- **Queries About GPT Monetization**: `dinola` sparked discussions around how GPT monetization works. `pinkcrusader` provided a speculative response suggesting user split pay between the most used GPTs. `7877` provided a more comprehensive explanation of how the split might work, using an example of a user who utilizes a custom GPT for 10% of their usage, and that the creator could achieve a 70/30 split of that 10%.
- **Legal Concerns About GPTs for Video Game Guides**: `ashh.` raised a question about potential infringement of copyright rules if a GPT is built to answer questions from specific video games.
- **Question About Fast Conversion of API to GPT Config File**: `moneyj2k` wanted to know the quickest way to convert a public API into a GPT's config file.
- **GPT-4 Token Limit and Unaccessible GPTs**: `its_eddy_` and `Bianca Stoica` mentioned the GPT-4 token usage limit and the issue of unaccessible GPTs.
- **Enthusiasm Over Customized GPT Creation**: `kungfuchopu` shared their excitement in designing custom GPTs that generate unique stories in a specific character's voice. 
- **Public API Conversion Into OpenAI Config**: `moneyj2k` queried about a method to quickly convert a public API into a config file for a GPT.
- **Advice on Domain Verification**: `anardude` asked for help and advice on verifying his domain for OpenAI GPT. His concerns remain unaddressed.
- **Builder Enrollment Hiccups**: `filipemorna` asked about enrollment for the GPT Store and was directed by `solbus` to the builder profile settings in ChatGPT Plus.
- **GPT Store Preparation Tips**: `cerebrocortex` asked for advice in preparing a GPT for the GPT Store. `solbus` provided OpenAI guidance for reference. 
- **Interactions with GPT**: Users `jobydorr` and `_odaenathus` discussed prompts and rules for interacting with GPTs. `_odaenathus` further asked about "Knowledge" limits, prompting `chotes` to clarify that there is a hard token limit and that larger documents are split up using Retrieval Augmented Generation (RAG).
- **'GPT Inaccessible or Not Found' Issue**: User `melysid` asked Coty about solutions for a 'GPT inaccessible or not found' issue. This query remained unaddressed in the discussions.
- **GPT Store Payout Mechanism**: User `pinkcrusader` provided details about the payout mechanism based on most used GPTs, being discussed in a developers' keynote.
- **Number of Messages Limitation Issues**: User `holden3967` brought up concerns about the messaging limit (25 messages per 3 hours) on GPT, considering it severely limiting the utility of GPT apps. `7877` humorously pointed out that there is no way around this limit. `_odaenathus` proposed an idea for users to have multiple accounts with multiple plans to overcome this issue.
- **DALLE Image Aspect Ratios**: User `.arperture` shared their difficulty in consistently outputting images in aspect ratios other than square when using DALLE in a custom GPT. `solbus` suggested using "tall aspect ratio" and "wide aspect ratio" instead.
- **Procedure for Enrolling as a GPT Builder**: User `filipemorna` asked about the process for enrolling as a GPT Builder. `solbus` mentioned it can be done with a ChatGPT Plus account on OpenAI's GPT editor.
- **Creating Custom GPTs for Video Games**: User `.ashh.` asked about the legality of creating custom GPTs to answer questions about specific video games without infringing copyrights. `7877` suggested it would be fine as long as no game assets are used.
- **GPT Store Monetization**: User `dinola` sought clarifications about how the monetization of the GPT store works. `7877` offered a detailed explanation using the example of a user who uses a custom GPT for 10% of their usage time.
- **Limitations of GPT-4 in Reading Kanjis**: `xbtrob` mentioned that **GPT-4** is unable to read Kanjis from images. `elektronisade` confirmed the limitation, stating the model doesn't perform optimally on non-Latin alphabets.
- **Issuing with Custom GPT and Rate Limits**: `_odaenathus` clarified how mixing regular ChatGPT with custom GPT messages worked regarding to rate limits. `lumirix` mentioned that he believes that GPT prompts also count towards the 40 message limit.
- **Approaching Rate Limits Carefully**: User `dino.oats` warned that generating multiple images in the same response can quickly reach the rate limit.
- **OpenAI GPT More Content**: User `kungfuchopu` shared their enthusiasm about creating unique stories in specific character voices using their custom GPT.
- **Potential Intellectual Property issue Building GPTs**: `.ashh.` asked about potential copyright issues when creating GPTs based on video games. `7877` suggested that using game assets could pose a problem.
- **Domain Verification Problems**: User `anardude` faced difficulties when verifying his domain, and asked for solutions without receiving a response.
- **Discussion About the GPT Store**: `cerebrocortex` asked about tips for preparing a GPT for the GPT Store and was directed by `solbus` to OpenAI's guidelines.
- **Potential User Subscription Issues**: `Bianca Stoica` expressed a concern about having full time access to GPT-4.
- **Discussion Around Non-Latin Languages Issue**: `xbtrob` stated that GPT is unable to read Kanjis from images. This issue is especially deemed significant for reading non-Latin languages.
- **Custom GPT for Code Creation**: `kerubyte` was looking for advice on creating a custom GPT that would generate valid HTML and CSS code for a specific platform.
- **Discussion About the Limits of GPT**: `_odaenathus` asked about the hard and soft limits of GPT "knowledge", and was answered by `chotes` regarding the limits on tokens.
- **Token Costs for Uploading Files**: `chotes` warned about the high cost of uploading large JSON files.
- **API Configuration for GPTs**: `moneyj2k` asked for a quick way to transform a public API into a config file for a GPT.
- **Discussion on GPT Communication Limits**: `holden3967` brought up the limit of 25 messages per 3 hours on GPT as an issue, `7877` humorously mentioned there is no way around it, later `_odaenathus` suggested having multiple accounts with different plans to overcome this issue.
- **Dalle Image Generation**: `.arperture` is looking for ways to consistently output images with aspect ratios other than a square using Dalle in a custom GPT.
- **Discussion Over Rule-set in GPT**: `jobydorr` discussed over the issues of having a set of instructions that GPT has to follow.
- **Request for GPT Literature Interaction**: `kungfuchopu` enthusiastically shared about their new GPT creation which allows users to interact with characters and generate unique stories in their own voice and tone.
- **Discussion About the Enrollment Process for GPT Builders**: User `filipemorna` asked about the process of enrolling as a GPT builder. `solbus` provided the answer that anyone with a **ChatGPT Plus** account can create a custom GPT.
- **Uncertainty over GPT Inaccessibility**: `melysid` sought advice about solving an issue where GPT was inaccessible or not found. Unfortunately, their query remained unanswered.
- **Monetization Strategy for GPT Store**: User `dinola` asked about how GPT monetization works in relation to user access and API usage. In response to this, `7877` speculated on a possible method, suggesting that the cost is split across common/widely used GPTs.
- **Custom GPT for Video Game Guides and Trademark Concern**: User `.ashh.` discussed potential trademark concerns when creating a GPT that answers questions for a specific video game. The consensus was that as long as `ashh.` doesn't use any game assets, it should be fine.
- **Verifying Domain for GPT Builder Setup**: An ongoing issue of verifying the domain for GPT Builder set-up was discussed by `johan0433` and `savioai`. A proposed solution of inputting "www" as the Name field during DNS/TXT record creation was unsuccessful.
- **Anticipation for Launch of GPT Store**: User `solbus` announced that the launch of an OpenAI GPT Store was imminent, as announced in an email from OpenAI. This sparked a small discussion around the availability and searchability of a directory of custom GPT bots.
- **Question About Structuring API to OpenAI Config**: `moneyj2k` asked about transforming a public API into an OpenAI config file for GPT, creating a brief discussion but provided no confirmed solution.
- **GPT-4 Token Limit Discussions**: A brief exchange between `its_eddy_` and `Bianca Stoica` discussed the issue of GPT-4 usage cap and the concept of token limits. A necessity for a subscription model was discussed to bypass this issue.
- **Resolution to Inaccessible GPT Issue**: User `melysid` brought up an issue where GPT is inaccessible or not found, but the conversation did not provide a resolution.
- **Issues with GPT and File Recognition**: User `alanwunsche` raised concerns about his GPT failing to recognize small uploaded files, noting the issue as one reported by a number of users. However, offered solutions remained unanswered.


**Links mentioned**:

- [Brand guidelines](https://openai.com/brand#gpts-in-chatgpt>): Language and assets for using the OpenAI brand in your marketing and communications.
- [GPT access on cell phone app](https://community.openai.com/t/gpt-access-on-cell-phone-app/502753): My GPT Ranko works well on Mac but when I access it on the cell phone app it tells me my GPT is unavailable. I can access other GPTs alright. Is there any additional setting for app access?


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (48 messages🔥): 
        
- **Balancing images with dynamic poses**: User `@hawaiianz` shared a [prompt](https://dct.openempathic.ai/) with tips on how to create balanced images in DALL-E by using dynamic poses to support the structure of pyramid formation in the image. They suggested this method can be applied beyond images.

- **Security in GPT Deployment**: Users `@cerebrocortex`, `@aminelg`, `@madame_architect` and `@wwazxc` had a discussion on securing GPT models against unauthorized access. Though additional resources were suggested for exploring security methods, users were cautioned against breaking OpenAI's usage policies.

- **Improving GPT Performance**: User `@ajkuba` shared difficulties regarding a custom GPT’s inability to process large batches of data effectively without crashing. Fellow users `@sciandy` and `@wwazxc` recommended outputting the results into different formats or utilizing JSON mode to avoid these issues.

- **Logo Creation with DALL-E**: `@viralfreddy` sought advice on how to improve their prompts for creating a logo using DALL-E. `@darthgustav` suggested being specific in the prompt and translating the prompt to English for better results if English wasn't the user's native language.

- **Issues with GPT Model**: User `@cosmodeus_maximus` expressed their dissatisfaction with the current performance of the GPT model, specifically its lack of creativity and neglect of user instruction. Similarly, `@mysterious_guava_93336` also flagged issues he's having with constant 'PowerPoints' ('structured responses') served by ChatGPT, which impedes the quality of conversation.
  
- **Prompt Engineering Success Story**: User `@exhort_one` shared their triumph in finally obtaining their desired script from the GPT model after 3 months of prompt engineering, demonstrating the learning curve and resilience needed in this field.

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (48 messages🔥): 
        
- **Exploration of Enhanced Image Generation**: `@hawaiianz` explained that the usage of dynamic positioning and support roles within image descriptions could increase the quality of results when using **Dall-E**. Details in image composition, such as varying actions (like "random kung fu attack positions"), can contribute to a more balanced and realistically dynamic picture.
- **Security Concerns in GPT**: Users like `@cerebrocortex` and `@wwazxc` raised concerns about security measures for GPT, particularly pertaining to guarding instruction execution. Suggestions included exploring scholarly resources and established projects like SecretKeeperGPTs or SilentGPT for learning and inspiration, shared by users such as `@madame_architect` and `@eskcanta`.
- **Logo Designing with DallE**: `@viralfreddy` sought assistance with designing a logo using DallE, receiving advice from `@darthgustav` to be specific with the prompts and also suggest translating the prompts to English for better results.
- **Issues with GPT's Task Handling**: `@ajkuba` discussed a persisting issue with GPT failing to complete large tasks consisting of numerous items, like batch web browsing. Other users like `@wwazxc` and `@cosmodeus_maximus` also expressed frustration with major limitations and occasional unresponsiveness in the AI system's operations.
- **Prompt Engineering Challenges and Success**: `@exhort_one` shared their journey of persevering through multiple prompt revisions and a burnout, and ultimately achieving their desired script from GPT. The user acknowledged the community's prompt engineering insights as helpful in this process.

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Model Faceoff: Mamba vs. Transformer**: A lively debate centered on Mamba's performance compared to Transformer models in NLP tasks. The discussion, initiated by `@sentialx`, addressed concerns of model diversity, optimization levels and use-cases.
- **All Eyes on Civitai's Clubs**: `@digthatdata` shared Civitai's latest feature for creators: "Clubs". Meant to boost engagement for creators offering exclusive content, the rollout has not been without its share of detractors.
- **Phi-2 Released Under MIT License**: As announced by `@SebastienBubeck` via Twitter, and shared by `@digthatdata`, the Phi-2 model hailing from Microsoft Research is now accessible under the MIT License.
- **Youths of the LLaMA Family: TinyLlama and LLaMA Pro**: New releases of language models, TinyLlama and LLaMA Pro, were shared by `@philpax` and `@ai_waifu`. Both models mark significant advancements in the LLaMA model lineage.
- **McQs and LLMs: A Tumultuous Relationship**: `@Nish`, a researcher from the University of Maryland, reported notable performance discrepancies between his project's LLM-generated answers to multiple choice questions, especially on the HellaSwag dataset, and previously reported leaderboard figures. Contributory factors could include differences in implementation and normalization of choices' log likelihoods.
- **Shared Notes on Sequence Labelling**: The application of logit/tuned lens in sequence labelling tasks, such as PoS tagging, was the focus of a conversation commenced by `@vertimento`.
- **Unlocking the Mysteries of GPT-4 Token Choices**: A paper shedding light on GPT-4's token choices from smaller model distributions was shared by `@allanyield`. The paper suggests that if properly prompted, an LLM can output text as an explanation followed by a conclusion.
- **Interest in Joining Interpretability-Related Project**: `@eugleo` finished their ARENA program and expressed their eagerness to partake in an interpretability-related project, offering a commitment of 16 hours per week.
- **Proof Lies in Performance**: An in-depth conversation involving `@carsonpoole` and `@rallio` revolved around the inference performance of different-sized models. They noted the crucial role of model size, GPU capabilities, and batch size in determining inference speed and cost.
- **Training Models and the Efficiency of MLP**: The talk between `@jks_pl` and `@stellaathena` covered the implications and efficiency of Masked Language Model (MLM) training and span corruption tasks, incorporating reference to a [specific paper](https://arxiv.org/abs/2204.05832).
- **Decoding Transformers**: Doubts regarding Transformer architectures were listed by `@erich.schubert`, leading to a dialogue on handling positional encodings, layer outputs, and the structure of prefix decoders.
- **Bias in Sequence Length—Real or Imagined?** Concerns raised by `@chromecast56` about sequence length warnings in a project triggered a dialogue to reassure users about the ability of the evaluation harness to handle this issue.
- **ToxiGen Adds Its Flavour to lm-evaluation-harness**: `@hailey_schoelkopf` acknowledged the notable contribution from the lead author of the ToxiGen paper to the implementation of lm-eval-harness, an element not explored in the original paper.
   


**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (161 messages🔥🔥): 
        
- **Mamba vs Transformer Debate**: Users engaged in a lively discussion over the effectiveness of Mamba and Transformer models in NLP tasks, prompted by `@sentialx`'s comment questioning Mamba's performance. `@thatspysaspy`, `@canadagoose1`, `@stellaathena`, and `@clock.work_` participated, discussing factors such as the degree of optimization, the diversity among different models, and the specific use-cases for different model types.
- **Civitai Clubs Rolled Out**: `@digthatdata` shared an [update from Civitai](https://civitai.com/articles/3624/introducing-civitai-clubs) discussing their new feature for creators, "Clubs". It's intended to enhance engagement for creators offering exclusive content, but also mentioned some backlash they've experienced since release.
- **Release of Phi-2 under MIT License**: `@digthatdata` also highlighted a [tweet by @SebastienBubeck](https://fxtwitter.com/sebastienbubeck/status/1743519400626643359), announcing the release of Phi-2 under an MIT license. 
- **Inference Performance Comparisons**: A detailed discussion between `@carsonpoole` and `@rallio` focused on the inference performance of different-sized models, with `@rallio` sharing a useful [resource](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) on best practices for handling large language models (LLMs). The talk highlighted the importance of model size, GPU power, and batch size when considering inference speed and cost.
- **Volunteer Research Opportunity**: `@sirmalamute` reached out to the community looking for opportunities to contribute to a machine learning research project on a voluntary basis to gain more hands-on experience.

**Links mentioned**:

- [Call for feedback on sustainable community development | Civitai](https://civitai.com/articles/3636): This week, we rolled out Clubs - a new feature for creators who’ve been running exclusive memberships on platforms like Patreon. Clubs is our way o...
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices): In this blog post, t
- [Tweet from Sebastien Bubeck (@SebastienBubeck)](https://fxtwitter.com/sebastienbubeck/status/1743519400626643359): Starting the year with a small update, phi-2 is now under MIT license, enjoy everyone!  https://huggingface.co/microsoft/phi-2


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (64 messages🔥🔥): 
        
- **Logit Tuned Lens in Sequence Labelling**: A discussion initiated by `@vertimento` focused on the application of logit/tuned lens to sequence labelling tasks such as PoS tagging. The conversation did not yield a decisive answer but attracted a variety of responses.
- **Releases of TinyLlama and LLaMA Pro**: `@philpax` and `@ai_waifu` shared about new releases of language models, TinyLlama and LLaMA Pro, respectively. Both pose significant advancements in the LLaMA family of models.
- **Discussion on Model Training and MLP Efficiency**: `@jks_pl` and `@stellaathena` conducted an extensive discussion on the efficiency and implications of Masked Language Model (MLM) training and span corruption tasks, including the referencing of a useful paper for context: [https://arxiv.org/abs/2204.05832](https://arxiv.org/abs/2204.05832).
- **Change in Licensing of Phi-2 Model**: Phi-2, originally under Microsoft Research License, is now under MIT License as per `@kd90138`.
- **Queries Regarding Transformer Architectures**: `@erich.schubert` raised some questions on the classic Transformer (encoder-decoder) architecture, receiving inputs from `@stellaathena`, `@ad8e`, and others on the handling of positional encodings and layer outputs, and the structure of prefix decoders.

**Links mentioned**:

- [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385): We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. Building on the architecture and tokenizer of Llama 2, TinyLlama leverages variou...
- [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/abs/2401.02415): Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretra...
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832): Large pretrained Transformer language models have been shown to exhibit zero-shot generalization, i.e. they can perform a wide variety of tasks that they were not explicitly trained on. However, the a...
- [Look What GIF - Look What Yep - Discover &amp; Share GIFs](https://tenor.com/view/look-what-yep-this-gif-13617583): Click to view the GIF
- [Upload 3 files · microsoft/phi-2 at 7e10f3e](https://huggingface.co/microsoft/phi-2/commit/7e10f3ea09c0ebd373aebc73bc6e6ca58204628d)
- [An explanation for every token: using an LLM to sample another LLM — LessWrong](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample): Introduction Much has been written about the implications and potential safety benefits of building an AGI based on one or more Large Language Models…
- [AI 100-2 E2023, Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations | CSRC](https://csrc.nist.gov/pubs/ai/100/2/e2023/final)
- [Reproducibility Challenge: ELECTRA (Clark et al. 2020)](https://wandb.ai/cccwam/rc2020_electra_pretraining/reports/Reproducibility-Challenge-ELECTRA-Clark-et-al-2020---VmlldzozODYzMjk): A reproduction of the Efficiently Learning an Encoder that Classifies Token Replacements Accurately approach in low-resource NLP settings.
- [Improving position encoding of transformers for multivariate time series classification - Data Mining and Knowledge Discovery](https://link.springer.com/article/10.1007/s10618-023-00948-2/figures/1): Transformers have demonstrated outstanding performance in many applications of deep learning. When applied to time series data, transformers require effective position encoding to capture the ordering...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (2 messages): 
        
- **Discussion on "An Explanation for Every Token" Paper**: `@allanyield` brought up a [paper](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample) they found that explores how **GPT-4 chooses tokens** from the distribution provided by a smaller model and writes justifications for its choices. The paper acknowledges the potential safety benefits of scaling AGI based on Large Language Models (LLMs) and demonstrates how a properly prompted LLM can output text structured as an explanation followed by a conclusion.
- **Call for Research Collaboration**: User `@eugleo` mentioned completing the ARENA program and expressed a future interest in joining an **interpretability-related project**. Offering their contribution for approximately 16 hours a week for several months, they requested suggestions for possible projects or related contacts, stating no need for funding or additional commitments.

**Links mentioned**:

[An explanation for every token: using an LLM to sample another LLM — LessWrong](https://www.lesswrong.com/posts/5ZFgZbqp6Mi2xpYjK/an-explanation-for-every-token-using-an-llm-to-sample): Introduction Much has been written about the implications and potential safety benefits of building an AGI based on one or more Large Language Models…


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (13 messages🔥): 
        
- **Discussion on generating answers to multiple choice questions**: Researcher `@Nish` from the University of Maryland started a conversation around discrepancies he noted in his project that uses Language Learning Models (LLMs) to generate answers to multiple choice questions in standard MC format. He reported experiencing a significant performance gap compared to the reported numbers on the leaderboard, particularly for the HellaSwag dataset. 
- **Possible causes for the performance gap**: `@hailey_schoelkopf` suggested that the gap could be due to differences in the implementation, where `@Nish` is having the model generate just the answer letter, and Eleuther's implementation scores the sequence probability of the letter and the text after the letter. As an additional contributing factor, she mentioned the normalization of the choices' log likelihoods, which can significantly impact performance, especially for the HellaSwag dataset.
- **Benchmarking unnormalized vs normalized scores**: `@Nish` was directed by `@stellaathena` and `@hailey_schoelkopf` to [Teknium's github library](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Mixtral-7x8-Base.md) that contains benchmark logs for different LLMs, to view both unnormalized and normalized scores.
- **Worries about sequence length warnings**: User `@chromecast56` raised concerns about sequence length warnings they were seeing, and asked if intervention was necessary or if the evaluation harness will handle it.
- **ToxiGen's contribution to lm-evaluation-harness**: `@hailey_schoelkopf` mentioned that despite the original ToxiGen paper not exploring the autoregressive language model setting, the novel implementation in the lm-eval-harness was contributed by the ToxiGen lead author.

**Links mentioned**:

[LLM-Benchmark-Logs/benchmark-logs/Mixtral-7x8-Base.md at main · teknium1/LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Mixtral-7x8-Base.md): Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Improving Perplexity Usage**: Users discussed various approaches to enhance the utilization of Perplexity, with `@icelavaman` stating that the service is being refined to **develop a swift and precise answer engine**. [Link to tweet](https://x.com/AravSrinivas/status/1742919240364278092?s=20)
- **Perplexity Evaluation Scrutinized**: `@the_only_alexander` reported concerns regarding the Perplexity assessment published in an [academic paper](https://arxiv.org/abs/2304.09848), with specific doubts about the employed methodology. 
- **Perplexity Pro Unveiled**: Users `@donebg` inquired about the perks of Perplexity Pro. Responses from `@giddz` and `@mares1317` directed the queries to the [official Perplexity Pro page](https://www.perplexity.ai/pro). 
- **Notion Database Barrier Triggers Frustration**: `@stanislas.basquin` exhibited annoyance related to the limitations in linking Notion databases across divergent workspaces. [Perplexity thread](https://www.perplexity.ai/search/try-to-find-0po_EFFQTL2CUl4Cvqxcpg)
- **Perplexity Praises for Prompt Research Inputs**: `@shaghayegh6425` asserted appreciation for Perplexity being prompt and informative, particularly for bio-research inquiries and brainstorming. They shared multiple [examples](https://www.perplexity.ai/search/how-does-perplexity-mds5zed.T0uTRaRVJG4mmA?s=c#1f20c101-3fd2-4a0d-8e6c-0d55a804561a) 
- **Perplexity API Traps**: Users, including `@blackwhitegrey` and `@brknclock1215`, indicated difficulties and sought guidance for integrating the Perplexity API on platforms like ‘typingmind’ and ‘harpa.ai’. References to [Perplexity's API documentation](https://docs.perplexity.ai/reference/post_chat_completions) were made for assistance.


**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (180 messages🔥🔥): 
        
- **Perplexity Enhanced**: Users discussed ways to improve the usage of Perplexity with `@icelavaman` stating the service is being improved to **build the fastest and most accurate answer engine**, citing a [tweet](https://x.com/AravSrinivas/status/1742919240364278092?s=20) by Perplexity's CEO, Aravind Srinivas.
- **Perplexity Vs Academic Paper**: User `@the_only_alexander` shared his reservations about the Perplexity assessment mentioned in a [research paper](https://arxiv.org/abs/2304.09848), expressing concern about the paper's methodology.
- **Sound Annoyance**: `@jamiecropley` reported being annoyed by a sound that plays on the Perplexity homepage and was suggested to **mute the tab**.
- **In-browser Perplexity**: Users share how to set up **perplexity as default search engine** in the web browser, mainly on Firefox and Chrome. Users shared links to various resources such as [setting up search shortcuts in Chrome](https://chromeunboxed.com/chrome-site-search-shortcuts) and [how to add or remove search engines in Firefox](https://support.mozilla.org/en-US/kb/add-or-remove-search-engine-firefox) for this purpose.
- **Perplexity Pro Features**: Users `@donebg` asked about the advantages of Perplexity Pro in the app. Users `@giddz` and `@mares1317` directed them to the [official page](https://www.perplexity.ai/pro) listing Perplexity Pro benefits like more copilot searches, model selection, and unlimited file uploads.


**Links mentioned**:

- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1742919240364278092?s=20),): @Austen @eriktorenberg Thanks to every investor and user who has supported us so far. We couldn&#39;t be here without all the help! We look forward to continuing to build the fastest and most accurate...
- [Evaluating Verifiability in Generative Search Engines](https://arxiv.org/abs/2304.09848): Generative search engines directly generate responses to user queries, along with in-line citations. A prerequisite trait of a trustworthy generative search engine is verifiability, i.e., systems shou...
- [
  Add or remove a search engine in Firefox | Firefox Help
](https://support.mozilla.org/en-US/kb/add-or-remove-search-engine-firefox)
- [Comparing Brave vs. Firefox: Which one Should You Use?](https://itsfoss.com/brave-vs-firefox/): The evergreen open-source browser Firefox compared to Brave. What would you pick?
- [Vulcan Salute Spock GIF - Vulcan Salute Spock Star Trek - Discover &amp; Share GIFs](https://tenor.com/view/vulcan-salute-spock-star-trek-the-original-series-live-long-and-prosper-gif-23575376): Click to view the GIF
- [How to set up site search shortcuts in Chrome](https://chromeunboxed.com/chrome-site-search-shortcuts): As a search engine, Google does a great job of helping with broad web searches – such as when you want to find a business or product online. However, if you are looking for results within a specific w...
- [
  Assign shortcuts to search engines | Firefox Help
](https://support.mozilla.org/en-US/kb/assign-shortcuts-search-engines)
- [ChatGPT vs Perplexity AI: Does Perplexity Use ChatGPT? - AI For Folks](https://aiforfolks.com/chatgpt-vs-perplexity-ai/): The AI landscape is constantly shifting, and can be confusing. Many companies overlay different technologies for their own use. In this article, we&#39;ll compare
- [Perplexity Blog](https://blog.perplexity.ai/): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Perplexity CEO Aravind Srinivas, Thursday Nights in AI](https://www.youtube.com/watch?v=jksGQhMtXjo): Outset Capital&#39;s Ali Rohde and Josh Albrecht interview Perplexity AI CEO, Aravind Srinivas. Special thanks to Astro Mechanica (https://astromecha.co/) for ho...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (15 messages🔥): 
        
- **Notion DataBase Limitations Upset User**: `@stanislas.basquin` expressed frustration about being unable to create linked Notion databases across different workspaces. [Link to discussion](https://www.perplexity.ai/search/try-to-find-0po_EFFQTL2CUl4Cvqxcpg)
- **Perplexity Assists in Client Report Validation**: `@jay.mke` shared that the Perplexity service helped validate a client report within 2 minutes. However, the originally shared thread was not publicly accessible until `@ok.alex` advised making the thread public. [Link to thread](https://www.perplexity.ai/search/Beware-of-HP-CQwxdYIoQL63inFvs.xl0w?s=c)
- **Perplexity Praised for Bio-Research Inquiry Aid**: `@shaghayegh6425` appreciated Perplexity for being quick and resourceful particularly for Bio-Research inquiries and brainstorming. They shared three [resources](https://www.perplexity.ai/search/how-does-perplexity-mds5zed.T0uTRaRVJG4mmA?s=c#1f20c101-3fd2-4a0d-8e6c-0d55a804561a), [here](https://www.perplexity.ai/search/how-does-perplexity-mds5zed.T0uTRaRVJG4mmA?s=c#0684eaed-7b79-403d-8c43-e78059b9838d) and [here](https://www.perplexity.ai/search/how-does-perplexity-mds5zed.T0uTRaRVJG4mmA?s=c#cb095010-7db1-44d9-86cd-37e7663b03fe)
- **Perplexity Helps Avoid Anchoring**: `@archient` shared that Perplexity helped them figure out what they need to search for in order to avoid getting anchored. [Link to thread](https://www.perplexity.ai/search/Could-you-help-1WNmSDpiSVafl6ufcinb0g?s=c#4b91ed65-84c0-4489-8259-ae6e6b265738)
- **Perplexity Aids in Building Modular Web Application**: `@whoistraian` shared resources from Perplexity for building a modular web application. [Link to resources](https://www.perplexity.ai/search/Explain-like-Im-gP6oykAkQpCbbxMawkmaZg?s=u)

**Links mentioned**:

- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1743313625874645132): The knowledge app is on its ascent on the App Store. #25 in productivity. If you believe it&#39;s better than Bing (#17 right now, but a slower and much larger memory consuming app), you know what to ...
- [Tweet from Kristi Hines (@kristileilani)](https://fxtwitter.com/kristileilani/status/1743425585085579481): How useful is @perplexity_ai?   Here’s how it helped me with a little surprise in my garden.


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (5 messages): 
        
- **Struggling to Use API on Platforms**: `@blackwhitegrey` expressed difficulty in using the API on `typingmind` and `harpa.ai`, prompting a query over how other users are employing the API.
- **Going Independent with API**: `@archient` suggested writing their own code to utilize the API, pointing to [Perplexity's API documentation](https://docs.perplexity.ai/reference/post_chat_completions) for guidance.
- **Hands-on Try with API**: `@archient` also hinted at the prospect of making a direct attempt using the provided token.
- **Seeking Convenient Syntax for API Calls**: `@brknclock1215` requested suggestions on how to mold `Perplexity API` calls' syntax for `HARPA AI browser plugin`. The user referred to [Perplexity's API documentation](https://docs.perplexity.ai/reference/post_chat_completions) as they experimented with various input/setting combinations to no avail.

**Links mentioned**:

[Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Fine-Tuning Paused Due to Bug**: `@matts9903` brought attention to a [bug](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942) that is currently affecting the Mixtral model. The advice from `@le_mess` was to halt fine-tuning until this bug can be resolved.

- **Large Language Models and Datasets**: The discussion focused on whether training *Mistral* on separate datasets (Dolphin or OpenOrca) or a merged set would yield similar results. `@caseus_` responded it would be quite similar, with the recommendation of training with slimorca.

- **Comparison of Full Fine-Tuning and LoRa/QLoRa**: `@noobmaster29` initiated a conversation asking if anyone had done a comparison between full fine-tuning instruction sets and LoRa or QLoRa.

- **Device Mapping in Axolotl**: `@nanobitz` suggested merging [Pull Request #918](https://github.com/OpenAccess-AI-Collective/axolotl/pull/918) as it introduces better device mapping for handling large models in the Axolotl project.

- **Dockerhub Login Complication with CI**: Continuous Integration was failing due to Dockerhub login issues. Several team members involved including `@caseus_` and `@hamelh` struggled to solve this. A variant solution proposed was to only log into Dockerhub when pushing to the `main` branch.

- **Token Embedding and Training in New Language**: `@.___init___`'s question concerning language-specific tokenizers led to a focus on the feasibility of expanding tokenizers for new language training. `@nanobitz` and `@noobmaster29` clarified the task could be unfruitful without significant pretraining.

- **Shearing Commences for Models**: Confirmation by `@caseus_` indicated that the shearing process had commenced for a 2.7B model and expressed willingness to initiate pruning for a 1.3B model. With community support growing, `@le_mess` along with `@emrgnt_cmplxty` and `@nosa_.` are amassing resources in support of the project. `@emrgnt_cmplxty` shared relevant shearing [resources](https://github.com/princeton-nlp/LLM-Shearing) to facilitate contribution towards the project.

- **VinaLLaMA - Vietnamese Language Model**: The introduction of [VinaLLaMA](https://arxiv.org/abs/2312.11011), a Vietnamese Language Model, sparked a discussion initiated by `@.___init___` over the hypothetical performance of GPT-4 against language-specific models on benchmarks.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (57 messages🔥🔥): 
        
- **Bug Pauses Mixtral Fine-tuning**: `@matts9903` shared a [Huggingface bug](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942) impacting Mixtral, asking if it was worth fine-tuning before a resolution. `@le_mess` recommended waiting for a fix.
- **Mistral Trained on Single vs. Merged Datasets**: `@yamashi` inquired if a Mistral model trained on either Dolphin or OpenOrca datasets would give similar results to training on a merge of the two. `@caseus_` confirmed the similarity, recommending training with slimorca.
- **Jon Durbin's Mixtral Method Illustrated**: `@casper_ai` shared the technique used by Jon Durbin for DPO’ing mixtral, shared on [Twitter](https://x.com/jon_durbin/status/1743575483365699809?s=46&t=QUL78vIQDJohFpnIzCbQXA). Updates to TRL for multi-device use and adaptations to DPO code were among steps taken.
- **Comparing Full Fine-Tuning vs. LoRa/QLoRa**: `@noobmaster29` queried if anyone had compared full fine-tuning instructions sets versus LoRa or QLoRa.
- **Unreliable HF Evaluation for Bagel Model**: `@_dampf` reported unreliable HF evaluation results for Jon Durbin's Bagel model, which she has attempted to re-add to the HF eval board. There seems to be a consistent issue with Dolphin series models not showing up.

**Links mentioned**:

- [HuggingFaceH4/open_llm_leaderboard · Bagel 8x7B evaluation failed](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/516)
- [GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon](https://github.com/ml-explore/mlx): MLX: An array framework for Apple silicon. Contribute to ml-explore/mlx development by creating an account on GitHub.
- [Tweet from Jon Durbin (@jon_durbin)](https://x.com/jon_durbin/status/1743575483365699809?s=46&t=QUL78vIQDJohFpnIzCbQXA): DPO&#39;ing mixtral on 8x a6000s is tricky.  Here&#39;s how I got it working:  1. Update TRL to allow multiple devices: https://github.com/jondurbin/trl/commit/7d431eaad17439b3d92d1e06c6dbd74ecf68bada...
- [ Incorrect implementation of auxiliary loss  · Issue #28255 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942): System Info transformers version: 4.37.0.dev0 Platform: macOS-13.5-arm64-arm-64bit Python version: 3.10.13 Huggingface_hub version: 0.20.1 Safetensors version: 0.4.1 Accelerate version: not install...
- [[WIP] RL/DPO by winglian · Pull Request #935 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935/files#diff-65b4693504c4e8ffac76c7f2c90913faee381f802cf64e7f49c995a2134ed3b3R294)
- [[`Mixtral`] Fix loss + nits by ArthurZucker · Pull Request #28115 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28115): What does this PR do? Properly compute the loss. Pushes for a uniform distribution. fixes #28021 Fixes #28093


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (43 messages🔥): 
        
- **Merge Recommendation for Better Device Mapping**: `@nanobitz` proposed for the merge of [Pull Request #918](https://github.com/OpenAccess-AI-Collective/axolotl/pull/918) aiming at better device mapping for large models in the *Axolotl* project.
- **CI Fails Due to Dockerhub Login Issue**: The Continuous Integration process was failing due to issues while logging into Dockerhub. `@caseus_` tried fixing it by creating new tokens and updating github secrets, but the issue persisted. `@hamelh` investigated and concluded that the issue was related to workflow permissions due to the `pull_request` event. An option to log into Dockerhub only when pushing to `main` was suggested.
- **Removing Docker Login from Workflow**: `@hamelh` proposed removing the login process for Dockerhub from github actions workflow, as it wasn't clear why it was there in the first place. `@caseus_` agreed to the proposal.
- **Intel Gaudi 2 AI Accelerators Cheaper for Training**: `@dreamgen` shared an article from Databricks which suggests that training on Intel Gaudi 2 AI Accelerators can be up to 5 times cheaper than NVIDIA's A100.
- **Token Embedding Size Warning**: `@caseus_` shared a [tweet](https://fxtwitter.com/abacaj/status/1743752273199595856) advising against resizing token embeddings when adding new tokens to a model, as it has caused errors due to a mismatch between the embedding size and the vocabulary size. `@nanobitz` suggested that this might be a tokenizer and model config discrepancy and that the phi model may be affected by this issue.

**Links mentioned**:

- [Tweet from anton (@abacaj)](https://fxtwitter.com/abacaj/status/1743752273199595856): Do not resize token embeddings if you are adding new tokens, when I did this model errors out. Seems like there are 51200 embedding size but only 50294 vocab size
- [LLM Training and Inference with Intel Gaudi 2 AI Accelerators](https://www.databricks.com/blog/llm-training-and-inference-intel-gaudi2-ai-accelerators)
- [build-push-action/action.yml at master · docker/build-push-action](https://github.com/docker/build-push-action/blob/master/action.yml#L76-L79): GitHub Action to build and push Docker images with Buildx - docker/build-push-action
- [axolotl/.github/workflows/tests-docker.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/.github/workflows/tests-docker.yml#L40-L44): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [e2e-docker-tests · OpenAccess-AI-Collective/axolotl@cbdbf9e](https://github.com/OpenAccess-AI-Collective/axolotl/actions/runs/7425756764/job/20208062696): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [Update tests-docker.yml by hamelsmu · Pull Request #1052 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1052)
- [Simplify Docker Unit Test CI by hamelsmu · Pull Request #1055 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1055): @winglian I think we might have to merge this to really test it.
- [feature: better device mapping for large models by kallewoof · Pull Request #918 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/918): When a model does not fit completely into the GPU (at 16-bits, if merging with a LoRA), a crash occurs, indicating we need an offload dir. If we hide the GPUs and do it purely in CPU, it works, but...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (61 messages🔥🔥): 
        
- **Expanding the tokenizer for a new language**: User `@.___init___` was seeking advice on expanding a tokenizer with more tokens for a new language and then training it. `@nanobitz` and `@noobmaster29` shared their experiences indicating that it was not proven beneficial unless there's significant pretraining. They cited a project on [GitHub](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md) as a reference.

- **Implementations of gradient accumulation steps during full fine-tuning**: In a discussion initiated by `@Sebastian`, `@caseus_` confirmed that smaller batch sizes are preferable for most fine-tuning tasks.

- **Downsampling a dataset in axolotl**: `@yamashi` was looking for ways to downsample a dataset for a training run in axolotl. `@le_mess` suggested using shards to achieve this.

- **Experiments with smaller models**: `@.___init___` and `@noobmaster29` discussed the idea of experimenting with smaller models. `@noobmaster29` expressed concerns about the memory issues with phi2 and wasn't sure about the performance of tinyllama.

- **Introducing VinaLLaMA**: `@.___init___` brought attention to [VinaLLaMA](https://arxiv.org/abs/2312.11011), a state-of-the-art Large Language Model for the Vietnamese language built upon LLaMA-2. This led to a discussion on whether GPT-4 would perform better on benchmarks than such language-specific models.

**Links mentioned**:

- [VinaLLaMA: LLaMA-based Vietnamese Foundation Model](https://arxiv.org/abs/2312.11011): In this technical report, we present VinaLLaMA, an open-weight, state-of-the-art (SOTA) Large Language Model for the Vietnamese language, built upon LLaMA-2 with an additional 800 billion trained toke...
- [LeoLM: Igniting German-Language LLM Research | LAION](https://laion.ai/blog/leo-lm/): &lt;p&gt;We proudly introduce LeoLM (&lt;strong&gt;L&lt;/strong&gt;inguistically &lt;strong&gt;E&lt;/strong&gt;nhanced &lt;strong&gt;O&lt;/strong&gt;pen &lt;strong&gt;L&lt;/strong&gt;anguage &lt;stron...
- [Chinese-LLaMA-Alpaca/README_EN.md at main · ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md): 中文LLaMA&amp;Alpaca大语言模型+本地CPU/GPU训练部署 (Chinese LLaMA &amp; Alpaca LLMs) - ymcui/Chinese-LLaMA-Alpaca


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (14 messages🔥): 
        
- **Shearing commenced for 2.7B model**: `@caseus_` confirmed that shearing is being conducted for the 2.7B model. Also expressed willingness to do the initial pruning for a 1.3B model.
- **Compute assistance for 1.3B model**: `@le_mess` showed readiness to provide computational resources for the training of the 1.3B model.
- **Getting set for shearing contribution**: `@emrgnt_cmplxty` expressed interest in contributing to shearing and was provided a [GitHub link](https://github.com/princeton-nlp/LLM-Shearing) to get started by `@caseus_`.
- **Shearing process based on Sheared LLaMa model**: `@emrgnt_cmplxty` clarified that the shearing process will be based on the framework used for the Sheared LLaMa models.
- **Question on Mistral & LLaMa difference**: `@emrgnt_cmplxty` questioned if the only variation between Mistral and LLaMa is the sliding attention window. `@caseus_` confirmed this to be true.
- **Problem with sampling subset of RedPajama**: `@caseus_` mentioned having issues with the part of the code that uses a sampled subset of RedPajama from the [given GitHub link](https://github.com/princeton-nlp/LLM-Shearing/tree/main/llmshearing/data).
- **Plan for using large context window**: In response to `@emrgnt_cmplxty`'s question about the use of a large context window, `@caseus_` confirmed an intention to use this approach with another fine-tune on a dataset like Capybara after the shearing process.
- **Support for the shearing project expands**: `@nosa_.` showed interest in supporting the shearing project.

**Links mentioned**:

- [LLM-Shearing/llmshearing/data at main · princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing/tree/main/llmshearing/data): Preprint: Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning - princeton-nlp/LLM-Shearing
- [GitHub - princeton-nlp/LLM-Shearing: Preprint: Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://github.com/princeton-nlp/LLM-Shearing): Preprint: Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning - GitHub - princeton-nlp/LLM-Shearing: Preprint: Sheared LLaMA: Accelerating Language Model Pre-training via...


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Big Data on Hugging-Face Query**: `@QwertyJack` questioned if a 10TB bio-ML dataset could be hosted on HuggingFace. Access the discussion [here](https://discord.com/channels/879548962464493619/879548962464493622/).
- **LLM Benchmarking Guide Requested**: In response to `@exponentialxp`'s question on LLM benchmarking, `@gr.freecs.org` shared a useful resource - a link to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on GitHub. 
- **AI-powered Personal Discord Chatbot and Pokemon-Classifier Shared**: `@vashi2396` and `@4gastya` shared their [personal discord chatbot](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA) and [Pokemon-Classifier](https://huggingface.co/spaces/AgastyaPatel/Pokemon-Classifier) projects respectively in the `i-made-this` channel.  
- **Potential NLP Research Collaboration Initiated**: `@sirmalamute` offered to collaborate on NLP research projects and Python modules development in the `general` channel.
- **The Tyranny of Possibilities Reading Group Discussion**: The `reading-group` channel hosted an engaging discussion on the design of Task-Oriented Language Model Systems, or **LLM Systems**, which was initiated by `@dhruvdh`.
- **TinyLlama Project and Small AI Models Revealed**: `@dame.outlaw` revealed an open team project [TinyLlama](https://github.com/jzhang38/TinyLlama) and `@dexens` shared an article on Microsoft's [small AI models](https://www.semafor.com/article/11/01/2023/microsoft-pushes-the-boundaries-of-small-ai-models) in the `cool-finds` channel.
- **Introduction of LoRA implementations for Diffusion DPO**: The `core-announcements` channel featured `@sayakpaul`'s [announcement](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo) about the implementation of **LoRA** for **Diffusion DPO**.
- **A New Conversation Generator Unveiled**: In the `NLP` channel, `@omaratef3221` introduced their new [project](https://huggingface.co/Omaratef3221/flan-t5-base-dialogue-generator), a **Conversation Generator** that could transform the field of developing chatbots and virtual assistants.
- **Call for Input on Computer Vision Data Storage**: User `@etharchitect` in the `computer-vision` channel initiated a discussion inviting suggestions on recent, pioneering techniques in computer vision data storage.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (85 messages🔥🔥): 
        
- **Hosting of large datasets on HF questioned**: User `@QwertyJack` inquired whether a public bio-ML dataset of about 10TB could be hosted on HuggingFace.
- **Need for LLM benchmarking guide surfaced**: `@exponentialxp` asked for guidance on how to benchmark an LLM. `@gr.freecs.org` shared a [link to lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on GitHub.
- **Struggles with specialized coding assistant under discussion**: `@pixxelkick` conversed with `@vipitis` about how to engineer an LLM for "copilot" style code completion with the `nvim.llm` plugin. `@vipitis` suggested looking at open source code completion models and tools such as [TabbyML's CompletionContext](https://github.com/TabbyML/tabby/blob/main/clients/tabby-agent/src/CompletionContext.ts) for reference and using larger models like `deepseek-coder`.
- **Potential NLP research collaboration broached**: `@sirmalamute`, an experienced ML engineer, expressed interest in collaborating on NLP research projects and developing open source Python modules. `@kopyl` welcomed the offer of help for his logo generation model project.
- **Input needed on running prediction in dataset filter function**: `@kopyl` got into a discussion with `@vipitis` about running model inference in a `dataset.filter` function on an image dataset. `@vipitis` advised against this practice due to certain potential issues, but `@kopyl` responded that the alternative would require more setup for multi-GPU inference.


**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1WUNKph8BdP0on5ve3gQnh_PE0cFLQqTn?usp=sharing)
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)
- [Good first issues • openvinotoolkit](https://github.com/orgs/openvinotoolkit/projects/3): A place to keep track of available Good First Issues. You can assign yourself to an issue by commenting &quot;.take&quot;.
- [tabby/clients/tabby-agent/src/CompletionContext.ts at main · TabbyML/tabby](https://github.com/TabbyML/tabby/blob/main/clients/tabby-agent/src/CompletionContext.ts): Self-hosted AI coding assistant. Contribute to TabbyML/tabby development by creating an account on GitHub.
- [iconbot (dev)](https://t.me/s/sdicon): AI engineer: @kopyl  Канал про то как я тренирую модель искуственного интеллекта по генерации иконок.  Доступ к боту по приглашениям, пишите в лс.
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of autoregressive language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of autoregressive language models. - GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of autoregressive language models.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (26 messages🔥): 
        
- **GPTs Optimization Discussions**: `@gag123` had inquiries about tweaking their GPT model's learning rate and tensor's dtype. After sharing their [Github project link](https://github.com/gagangayari/my-gpt) for context, `@exponentialxp` suggested they should have a dataset of size 20 times the number of parameters in their model for optimal results. 
- **Optimizer.zero_grad() to the Rescue**: The conversation revealed that `@gag123` was missing `optimizer.zero_grad()` in their code. `@exponentialxp` alerted them to this crucial line, which led to perceptible improvements in their model's performance.
- **Concerns about Overfitting and Loss Value**: `@exponentialxp` voiced concerns about possible overfitting due to the large size of `@gag123`'s GPT model and recommended downsizing it to 384,6,6 dimensions. They also discussed the unusually high loss value of 2.62 despite a modest vocab size of 65.
- **Iterative Code Enhancements Advised**: `@exponentialxp` recommended several code enhancements for `@gag123` including the introduction of a `train` and `val split` to avoid overfitting, adjustments to the `pos_emb` for their GPT class, and ensuring they are calling `Head` and not `MHA`.
- **DINOv2 Self-Supervised Learning in Images**: User `@merve3234` shared a [thread link](https://x.com/mervenoyann/status/1743290724672495827?s=20) about their learning experience with Dimensionality reduction for Image Nodes (DINO) and DINOv2, noting that DINOv2 is the current king of self-supervised learning in images.

**Links mentioned**:

- [Tweet from merve (@mervenoyann)](https://x.com/mervenoyann/status/1743290724672495827?s=20): DINOv2 is the king for self-supervised learning in images 🦖🦕  But how does it work? I&#39;ve tried to explain how it works but let&#39;s expand on it 🧶
- [History for sample.py - exponentialXP/TextGenerator](https://github.com/exponentialXP/TextGenerator/commits/main/sample.py): Create a custom Language Model / LLM from scratch, using a few very lightweight scripts! - History for sample.py - exponentialXP/TextGenerator
- [GitHub - gagangayari/my-gpt](https://github.com/gagangayari/my-gpt): Contribute to gagangayari/my-gpt development by creating an account on GitHub.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (6 messages): 
        
- **Exploring TinyLlama**: `@dame.outlaw` shared a [GitHub link](https://github.com/jzhang38/TinyLlama) to **TinyLlama Project**, an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.
- **Microsoft Pioneers Small Multimodal AI Models**: `@dexens` introduced a [semaphore article](https://www.semafor.com/article/11/01/2023/microsoft-pushes-the-boundaries-of-small-ai-models) that discusses Microsoft's research division adding new capabilities to its smaller language model, Phi 1.5, enabling it to view and interpret images.
- **Interesting Paper Alert**: `@masterchessrunner` shared an [academic paper](https://arxiv.org/pdf/1911.11423v1.pdf) found on Reddit, praising the author's sense of humor.

**Links mentioned**:

- [Microsoft pushes the boundaries of small AI models with big breakthrough | Semafor](https://www.semafor.com/article/11/01/2023/microsoft-pushes-the-boundaries-of-small-ai-models): The work by the company’s research division shows less expensive technology can still have advanced features, without really increasing in size.
- [GitHub - jzhang38/TinyLlama: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.](https://github.com/jzhang38/TinyLlama): The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens. - GitHub - jzhang38/TinyLlama: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama mode...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 messages): 
        
- **Pokemon-Classifier in the Making**: User `@4gastya` shared his project of fine-tuning a Pokemon Classifier model stating "*though it's not that accurate but i'm getting in love with playing with models*". Project link: [Pokemon-Classifier](https://huggingface.co/spaces/AgastyaPatel/Pokemon-Classifier).
- **AI-powered Personal Discord Chatbot**: `@vashi2396` is developing a personal discord chatbot based on AI to read and post messages. The project can be checked out [here](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA).
- **Collecting Feedback for AI Model**: `@gr.freecs.org` is inviting feedback for their model 'artificialthinker-demo-gpu'. The model can be accessed through the provided [link](https://huggingface.co/spaces/lmdemo/artificialthinker-demo-gpu).
- **AI Chatbot featured on LinkedIn**: `@vashi2396` shared a LinkedIn [post](https://www.linkedin.com/posts/vashisth-malik_building-an-ai-chatbot-and-much-more-rather-activity-7149329040942772224-Vy8M?utm_source=share&utm_medium=member_android) featuring their AI chatbot.
- **Request for Help**: User `@vashi2396` asked for help in "*getting input thru Automatic speech recognition model via microphone*."
- **Difficulties with Phi-2 Model**: User `@dexens` expressed disappointment with Phi-2's performance, especially with code generation and its tendency for repetition. They also asked if anyone had trained the model in JavaScript/TypeScript apart from Python. The model can viewed at this [link](https://huggingface.co/microsoft/phi-2).
- **Confirmation of Phi-2's Poor Performance**: User `@vipitis` concurred with `@dexens` and stated that Phi-2 is "*currently the worst performing model*".

**Links mentioned**:

- [ArtificialThinker Demo on GPU - a Hugging Face Space by lmdemo](https://huggingface.co/spaces/lmdemo/artificialthinker-demo-gpu)
- [Pokemon Classifier - a Hugging Face Space by AgastyaPatel](https://huggingface.co/spaces/AgastyaPatel/Pokemon-Classifier)
- [Google Colaboratory](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)
- [microsoft/phi-2 · Hugging Face](https://huggingface.co/microsoft/phi-2)


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (29 messages🔥): 
        
- **Diving Deep into The Tyranny of Possibilities**: User `@dhruvdh` created a thread discussing the design of Task-Oriented Language Model Systems, or **LLM Systems**, and this received acknowledgment from `@chad_in_the_house` who tagged the entire group for attention. 
- **A Heated Debate on Group Notifications**: An argument over group notifications broke out between users `@chad_in_the_house` and `@svas.`. `@cakiki` intervened to remind everyone about maintaining respect within the discussion. 
- **The Reading Group's Meet Format**: Users `@thamerla` and `@4gastya` expressed concerns about the accessibility and visibility of the reading group meet. `@chad_in_the_house` clarified that presentations are currently done via text threads in discord but is open for suggestions like using voice channels.
- **Exploring Joint-Embedding Predictive Architecture**: `@shashank.f1` brought up a paper for a discussion on the **MC-JEPA** approach towards self-supervised learning of motion and content features. A [YouTube video](https://youtu.be/figs7XLLtfY?si=USVFAWkh3F61dzir) providing a deep dive discussion on the same was shared.
- **Awakening AI Consciousness**: `@minhsmind` was reading a paper about AI Consciousness in preparation for a presentation, this sparked a debate with `@beyond_existence` and `@syntharion` where they shared their insights and skepticism on whether AI can attain consciousness.

**Links mentioned**:

[MC-JEPA neural model: Unlock the power of motion recognition &amp; generative ai on videos and images](https://youtu.be/figs7XLLtfY?si=USVFAWkh3F61dzir): 🌟 Unlock the Power of AI Learning from Videos ! 🎬 Watch a deep dive discussion on the MC-JEPA approach with Oliver, Nevil, Ojasvita, Shashank and Srikanth....


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **LoRA implementations for Diffusion DPO now available**: User `@sayakpaul` announced the implementation of **LoRA** for **Diffusion DPO**. The support is included for both **SD and SDXL**. The implementation can be checked out on their [GitHub link](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo).
- **SDXL-Turbo support on the way**: `@sayakpaul` hinted at incoming support for **SDXL-Turbo** in a future update.

**Links mentioned**:

[diffusers/examples/research_projects/diffusion_dpo at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch - huggingface/diffusers


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **Post-processing model Incorrect Labels**: `@merve3234` suggested that one could **run the model on the dataset and label the incorrect ones** as a method to correct incorrect labels in a dataset.
- **Deep Dive into DINO and DINOv2**: `@merve3234` shared a comprehensive thread and infographic explaining the workings of **DINO and DINOv2** for self-supervised learning in images. The complete thread can be found [here](https://x.com/mervenoyann/status/1743290724672495827?s=20). 
- **Query on Computer Vision Data Storage Techniques**: `@etharchitect` raised a question seeking information on recent **computer vision data storage techniques**, inviting the community to engage and discuss.

**Links mentioned**:

[Tweet from merve (@mervenoyann)](https://x.com/mervenoyann/status/1743290724672495827?s=20): DINOv2 is the king for self-supervised learning in images 🦖🦕  But how does it work? I&#39;ve tried to explain how it works but let&#39;s expand on it 🧶


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (3 messages): 
        
- **Exciting Announcement: A Conversation Generator Meet Dialogue Summarization**: `@omaratef3221` shared their new project, a **Conversation Generator**. This tool is designed to "generate realistic dialogues using **large language models** from small summaries." They've fine-tuned models like **Google's T5** and have created a dialogue generation version. The training took place on **Google Colab Platform** using **🤗 Transformers** and **Kaggle's Open Source Dialogue Dataset**.
- **Omaratef3221's Contribution to Open-Source Community**: The conversation generator is `@omaratef3221`'s gift to the open-source community. They expressed that the project's open-source and would appreciate any contributions or feedback. They also expressed that "**Open-source collaboration is a fantastic way to learn and grow together**."
- **In-depth Look at the Generator**: The model, [Omaratef3221's flan-t5-base-dialogue-generator](https://huggingface.co/Omaratef3221/flan-t5-base-dialogue-generator), is a fine-tuned version of Google's `t5` structured to generate "realistic and engaging dialogues or conversations."
- **Potential Uses Highlighted**: `@omaratef3221` noted that their model is "ideal for developing **chatbots**, **virtual assistants**, and other applications where generating **human-like dialogue** is crucial."
- **Enthusiastic Reactions and Future Possibilities**: `@stroggoz` reacted positively to the announcement and further mentioned how the technology might help them "understand higher-level math."

**Links mentioned**:

[Omaratef3221/flan-t5-base-dialogue-generator · Hugging Face](https://huggingface.co/Omaratef3221/flan-t5-base-dialogue-generator)


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **"It's all about fit":** Users `pseudoterminalx` and `thejonasbrothers` delved into **overfitting and underfitting** in model training. Discussion included the necessary balance and trade-offs involved in achieving satisfactory model performance.
- **"Noise adds originality":** User `pseudoterminalx` advocated for the inclusion of noise in **image generation** outputs, arguing that it contributes to perceived authenticity. Pointed out potential uncanny valley problem in excessively clean images.
- **"Synthetic Data (SD) 2.1x models can pack a punch":** A deep dive into **Synthetic Data (SD) 2.1x models** discussed training techniques such as cropping and resolution buckets. Reemphasis was given on SD 2.1 models' capabilities to output images of high resolution, with proper training.
- **"CommonCrawl and NSFW filtering":** `thejonasbrothers` mooted utilising uncaptioned images from CommonCrawl. Speculated on potential of synthetic captioning, the necessity for NSFW models to sift out inappropriate content and a vision of vast knowledge storage.
- **"Impeccable details in AI-generated images":** `pseudoterminalx` displayed high-res model outputs showcasing intricate details, creating a positive impression in the channel viewers.
- **"Masked tokens and U-Net are the secret sauce":** `kenjiqq` clarified that in the case of **transformer models**, it's not diffusion but **masked tokens** that play key role. User referenced original U-Vit paper about benefits of transformer blocks for better accelerator throughput.
- **"LLaVA-$\\phi$: small yet efficient":** Introduction to **LLaVA-$\\phi$** by `thejonasbrothers`, presented an efficient multi-modal assistant model that operates efficiently even with a smaller size of 2.7 billion parameters, showing impressive performance on multi-modal dialogue tasks.
- **"RLHF-V: high expectations, low performance":** `kopyl` and `thejonasbrothers` expressed their disappointment with the RLHF-V model, expected to have a low hallucination rate. Improvement in model performance was noted when the prompt was changed to request detailed image description.
- **"Decoding Causality in Language Models":** `JH` elaborated on how **Language and Image Models (LLMs)** can learn causality due to their decoder-oriented design and causal attention mask, whereas diffusion models learn by association.
- **"Deciphering 3D Space in Diffusion Models":** `phryq` posed an interesting question regarding diffusion models’ understanding of 3D space, specifically if these models could comprehend different facial perspectives. 

**Links mentioned**:

- [TikTok - Make Your Day](https://www.tiktok.com/@openbsd_fan_club/video/7319975721156250912?is_from_webapp=1&sender_device=pc&web_id=7311072099527116320)
- [Philipp Schmid's tweet](https://vxtwitter.com/_philschmid/status/1743545280086053032)
- [The Reel Robot's tweet](https://fxtwitter.com/TheReelRobot/status/1742984859457626562)
- [LLaVA-$ϕ$ paper](https://arxiv.org/abs/2401.02330)
- [Improving Diffusion-Based Image Synthesis with Context Prediction](https://arxiv.org/abs/2401.02015)
- [visheratin/LLaVA-3b · Hugging Face](https://huggingface.co/visheratin/LLaVA-3b)
- [CCSR GitHub repository](https://github.com/csslc/CCSR)
- [openbmb/RLHF-V · Hugging Face](https://huggingface.co/openbmb/RLHF-V)

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (103 messages🔥🔥): 
        
- **"Too optimized, or not optimized enough?"**: A helpful debate between users `pseudoterminalx` and `thejonasbrothers` focused around **overfitting and underfitting** in model training. They discussed the challenges and obvious trade-offs pointing out that some degree of fitting is required for the model to perform well.
- **"Upscale AI's latest trend: Noise in images!?"**: `pseudoterminalx` defended the inclusion of noise in image generation outputs, claiming it adds to the authenticity. Instances where "every single image is crystal clear makes them look, ironically, like AI gens", and they mentioned the issue of the "uncanny valley of perfection."
- **"Unleashing the true potential of SD 2.1x models"**: `pseudoterminalx` and `thejonasbrothers` took a deep dive into training techniques for **Synthetic Data (SD) 2.1x models**, diving into cropping methods, resolution buckets, and perceptual hashes. They also reviewed the model output resolutions and reiterated that SD 2.1 models were capable of outputting high-res images well beyond 1mp when trained correctly.
- **"A Picture of the Future from CommonCrawl":** `thejonasbrothers` contemplated on the idea of ripping uncaptioned images from CommonCrawl, then synthetically captioning and training them for a model that can store vast knowledge. It was pointed out that NSFW models would have to be used to filter out inappropriate content from CommonCrawl.
- **"Beauty is in the Eyes of the AI Beholder":** `pseudoterminalx` presented sample high-res outputs from the models. The images highlighted fine details like hair, skin texture, and tree bark, which seemed to impress the users in the channel.

**Links mentioned**:

- [TikTok - Make Your Day](https://www.tiktok.com/@openbsd_fan_club/video/7319975721156250912?is_from_webapp=1&sender_device=pc&web_id=7311072099527116320)
- [Tweet from Philipp Schmid (@_philschmid)](https://vxtwitter.com/_philschmid/status/1743545280086053032): We got a late Christmas gift from @Microsoft! 🎁🤗 Microsoft just changed the license for their small LLM phi-2 to MIT! 🚀  Phi-2 is a 2.7 billion parameter LLM trained on 1.4T tokens, including synth...
- [Tweet from The Reel Robot (@TheReelRobot)](https://fxtwitter.com/TheReelRobot/status/1742984859457626562): The most ambitious AI film I&#39;ve made to date.    We are less than a year out from commercially viable AI films. Updates to @runwayml (control) and @elevenlabsio (speech-to-speech) have me believin...


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (24 messages🔥): 
        
- **Masked tokens and U-Net in Transformer Models**: In a discussion about transformer models, `@kenjiqq` elaborated that in this case it's **not diffusion, but masked tokens** playing a significant role. They added that the major benefit of transformer blocks is better accelerator throughput, according to the original U-Vit paper.
- **Introduction of LLaVA-$\\phi$**: `@thejonasbrothers` shared a paper introducing **LLaVA-$\\phi$ (LLaVA-Phi)**, an efficient multi-modal assistant that harnesses the power of the small language model, Phi-2, for multi-modal dialogues. Notwithstanding its reduced number of parameters (as few as 2.7B), the model delivers commendable performance on tasks integrating visual and textual elements.
- **Confusion over LLaVA-3b and LLaVA-Phi Models**: `@nodja` linked the [Hugging Face model card](https://huggingface.co/visheratin/LLaVA-3b) for **LLaVA-3b**, a fine-tuned model from Dolphin 2.6 Phi, but noted that the authors listed didn't match those of LLaVA-Phi. A difference in architecture between the two models was also noted.
- **Context Prediction for Diffusion Models**: `@vrus0188` shared a link to the [CCSR GitHub repository](https://github.com/csslc/CCSR), a project titled **"Improving the Stability of Diffusion Models for Content Consistent Super-Resolution"**. 
- **Disappointment with Performance of RLHF-V**: In a discussion about the **RLHF-V** model, touted for its low hallucination rate, `@kopyl` and `@thejonasbrothers` expressed disappointment with its actual performance. `@bob80333` also reported subpar results when using a frame from a Ghibli movie. However, when the prompt was changed to requesting a detailed image description, the model's performance improved.

**Links mentioned**:

- [openbmb/RLHF-V · Hugging Face](https://huggingface.co/openbmb/RLHF-V)
- [Improving Diffusion-Based Image Synthesis with Context Prediction](https://arxiv.org/abs/2401.02015): Diffusion models are a new class of generative models, and have dramatically promoted image generation with unprecedented quality and diversity. Existing diffusion models mainly try to reconstruct inp...
- [LLaVA-$ϕ$: Efficient Multi-Modal Assistant with Small Language Model](https://arxiv.org/abs/2401.02330): In this paper, we introduce LLaVA-$ϕ$ (LLaVA-Phi), an efficient multi-modal assistant that harnesses the power of the recently advanced small language model, Phi-2, to facilitate multi-modal dialogues...
- [Instruct-Imagen: Image Generation with Multi-modal Instruction](https://arxiv.org/abs/2401.01952): This paper presents instruct-imagen, a model that tackles heterogeneous image generation tasks and generalizes across unseen tasks. We introduce *multi-modal instruction* for image generation, a task ...
- [visheratin/LLaVA-3b · Hugging Face](https://huggingface.co/visheratin/LLaVA-3b)
- [GitHub - csslc/CCSR](https://github.com/csslc/CCSR): Contribute to csslc/CCSR development by creating an account on GitHub.


### ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/) (3 messages): 
        
- **Debate on Causality in LLMs and Diffusion Models**: `@JH` provided a thorough explanation of how **Language and Image Models (LLMs)** can learn causality due to their decoder-based architecture and causal attention mask. In contrast, diffusion models are seen to learn through association rather than causality. This doesn't necessarily prevent them from learning causality, although their architecture and training may not directly bias them towards such a task. One possible way to train diffusion models to learn causality could be by creating datasets that deconstruct a caption from an explicit description to a **causal** one. 
- **Curiosity About Diffusion Models' Understanding of 3D Space**: `@phryq` asked if diffusion models understand that a profile face is just a face positioned differently, representing an interest in the model’s understanding of 3D space.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain 0.1 to See the Light of Day**: `@hwchase17` disclosed plans for the launch of **LangChain 0.1** and the intention to actively promote it. They have invited early feedback.
- **Catch My Signal**: Database encoding issues surfaced, with `@felipeescobar.` highlighting the need to pass the **UTF-8 Encoding** to SQL agent.
- **Chatbots on a Shopping Spree**: User `@vilelaone` probed for insights regarding a feature in chatbots that allows smooth transitioning between different modes, using `RouterChain` and `a conversational chain` in the context of an example like 'adding to cart,' 'gathering shipping data,' and 'defining payment.' A productive conversation followed this, concluded by `@evolutionstepper` posting code snippets and dropping the [GitHub repo link](https://github.com/ai-ponx/casa-bot/blob/dev/services/api/main.py).
- **LangChain Shines**: On the publicity front, `@rorcde` brought the community's attention to the spotlighting of LangChain as the AI Library of the Day at The AI Engineer, urging for the news to be spread via LinkedIn and Twitter.
- **A Picture is Worth a Thousand Words**: A desire for knowledge sharing was expressed by `@nav1106`, who sought suggestions on pretrained models for image embedding.
- **Error Message Five**: `@Tom P` reported encountering a `ModuleNotFoundError` for the CSV Agent, despite confirming its presence in the /packages directory.
- **New Variables, New Challenges**: `@cryptossssun` inquired how to utilize new variables while prompting in the LangServe service, post changing `input_variables` in the `PromptTemplate`. The ensuing discussions with `@veryboldbagel` and Co. can be traced back to the LangServe examples, OpenAPI docs and the `RemoteRunnable` client.
- **Callable AgentExecutor Unveiled**: `@veryboldbagel` elucidated on `AgentExecutor` being registered with `add_routes`, owing to its inheritance from `Chain`.
- **Taking Custom Logic on Board**: A member detailed the implementation of custom logic in LangChain, mentioning LCEL with runnable lambdas or through inheritance from `Runnable`.
- **Reading the Input Schema**: `@veryboldbagel` recommended checking the input schema on the runnable as a potential solution for issue related to unrecognized inputs, even describing the `with_types` method.
- **Session History Took a Hit**: `@nwoke.` expressed experiencing issues with `session_id` while integrating `RedisChatMessageHistory` in LangServe, with `@veryboldbagel` suggesting an issue to be opened on [GitHub](https://github.com/langchain-ai/langserve/issues).
- **Dependency Conflict Mars Langchain update**: `@attila_ibs` brought to fore the dependency conflicts they faced after updating LangChain, citing diverse packages that demanded mismatching versions of `openai` and `langchain`.
- **The Life Expectancy of Langchain**: `@dejoma` triggered discussions on the feasibility of writing a book on **Langchain**, given the pace of its version updates and potential deprecation risk.
- **ArtFul - Embolden Your Creativity**: `@vansh12344` heralded the unveiling of **ArtFul**, a free, ad-supported app enabling users to create their own art using various AI models, now live on the [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful).
- **Neutrino Makes its Debut**: `@ricky_gzz` launched **Neutrino**, a new innovative AI model routing system designed to optimize response quality, costs and latency between different closed and open-source models. A detailed insight into Neutrino can be found on [their official webpage](https://www.neutrinoapp.com/).
- **Tutorial Leaves Users Wanting More**: `@offer.l` showered praise on a tutorial, sparking a commitment from `@a404.eth` to develop another tutorial soon. Meanwhile, `@bads77` sought guidance on retrying missing responses using the langchain JS library, leading `@lhc1921` to propose monitoring retries with a `while loop`.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **LangChain 0.1 Launch**: User `@hwchase17` announced the launch of LangChain 0.1 and plans to highlight it heavily in the following week. They are open for early feedback.


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (33 messages🔥): 
        
- **UTF-8 Encoding in SQL Agent**: `@felipeescobar.` wondered if it's possible to pass the encoding to their SQL agent and confirmed that they are referring to **UTF-8 Encoding** when `@evolutionstepper` asked for specifics.
- **Seamless Transitioning in Chatbots**: `@vilelaone` asked for advice regarding the implementation of feature that enables a chatbot to seamlessly switch between different modes such as 'adding to cart,' 'gathering shipping data,' and 'defining payment.' There was a discussion about `RouterChain` and a possible solution involving `a conversational chain`. `@evolutionstepper` provided some source code examples and [full code on GitHub](https://github.com/ai-ponx/casa-bot/blob/dev/services/api/main.py) for further guidance.
- **LangChain AI Library Spotlight**: `@rorcde` announced that LangChain was spotlighted as the AI Library of the Day at The AI Engineer, and suggested the community to help spread the word by sharing related posts on LinkedIn and Twitter.
- **Pretrained Model for Image Embedding**: `@nav1106` was looking for suggestions on pretrained models for image embedding as they are new to this domain.
- **CSV-Agent Module Error**: `@Tom P` encountered an error (`ModuleNotFoundError: No module named 'csv_agent'`) when trying to install and run the CSV-agent template. Although the csv-agent folder is confirmed to be in the /packages directory, the error persists. There was a discussion about giving the conversational agent the necessary tools to solve the problem.


**Links mentioned**:

[casa-bot/services/api/main.py at dev · ai-ponx/casa-bot](https://github.com/ai-ponx/casa-bot/blob/dev/services/api/main.py): Agentive real estate sms assistant. Contribute to ai-ponx/casa-bot development by creating an account on GitHub.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (17 messages🔥): 
        
- **Querying for New Variables in Langserve Service**: `@cryptossssun` queried about how to use new variables while prompting in the LangServe service after a change of `input_variables` in the `PromptTemplate`. `@veryboldbagel` suggested looking at the LangServe example provided in the [docs](https://python.langchain.com/docs/langserve#server), which supports adding additional parameters in the query. User `@cryptossssun` was then advised to use the OpenAPI docs for schema information and to use the `RemoteRunnable` client for initiating requests.
- **Callable AgentExecutor**: `@veryboldbagel` explained how `AgentExecutor` could be registered with `add_routes`, being that it inherits from `Chain`, which in turn inherits from `Runnable`. This convention allows `AgentExecutors` to be directly callable when running LangChain instances.
- **Executing Custom Logic**: A member explained how custom logic in LangChain can be implemented, either via the combination of LCEL with runnable lambdas or through inheritance from `Runnable` to implement the desired logic.
- **Input Schema Check**: `@veryboldbagel` suggested checking the input schema on the runnable whenever issues regarding inputs being recognized are encountered. Further details like using the `with_types` method to manually specify the input type were discussed, linking to the LangChain [Docs](https://python.langchain.com/docs/expression_language/interface#input-schema) for reference.
- **Trouble with Session History**: `@nwoke.` expressed concerns over experiencing issues with the `session_id` while testing the `RedisChatMessageHistory` integration in LangServe. `@veryboldbagel` prompted for more information to be provided via an issue on [GitHub](https://github.com/langchain-ai/langserve/issues).

**Links mentioned**:

- [Interface | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/interface#input-schema.): To make it as easy as possible to create custom chains, we’ve
- [🦜️🏓 LangServe | 🦜️🔗 Langchain](https://python.langchain.com/docs/langserve#server.): Release Notes
- [🦜️🏓 LangServe | 🦜️🔗 Langchain](https://python.langchain.com/docs/langserve#client.): Release Notes
- [Issues · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.


### ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 
        
- **Template dependency conflicts in Langchain update**: User `@attila_ibs` reported experiencing dependency conflicts after updating Langchain. The user cited various packages including `neo4j-parent`, `pirate-speak`, `research-assistant`, and `stepback-qa-prompting` that require different versions of `openai` and `langchain` than currently installed. The user seeks assistance on how to fix the issue and effectively update an app based on Langchain templates.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (3 messages): 
        
- **Questioning the Longevity of Langchain**: `@dejoma` posed a thought-provoking question about the merit of writing a book on **Langchain**, a tool which sees frequent version updates and could become deprecated within two months.
- **New Year, New AI Art App**: `@vansh12344` announced the launch of **ArtFul**, a free, ad-supported app that enables anyone to generate their own art using various AI models. The announcement includes the claim that there's no sign-up or sign-in necessary and that it's all completely free following a short ad view. The application can be found on the [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful).
- **Introduction of Neutrino Router**: `@ricky_gzz` introduced a new project - **Neutrino** - an AI model routing system that aims to optimize response quality, costs and latency across different closed and open-source models. Neutrino's operational strategy is to automatically sample and evaluate responses to improve routing performance over time. Neutrino can be found [on this website](https://www.neutrinoapp.com/).

**Links mentioned**:

[Neutrino AI](https://www.neutrinoapp.com/)


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (5 messages): 
        
- **Positive feedback on tutorial**: `@offer.l` expressed enthusiasm for the tutorial, stating they "**Really 'liked' it! 🙂**". `@a404.eth` responded with appreciation and a promise to make another tutorial over the weekend.
- **Technical query about retrying missing responses**: In a technical discussion, `@bads77` asked about using the langchain JS library. **Bads77's query** focused on how to perform a retry for a missing/partial response in their requests, particularly when one of the expected fields from a prompt is missing. `@lhc1921` suggested using a `while loop` as a potential solution.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mixtral Team to Start Office Hours**: `@sophiamyang` announced the onset of **text-based office hours** and confirmed that **fine-tuning on the platform** is underway. The first office hour is scheduled for Jan 10 at 11:00 am EST (5pm CET) as per a post on the `#[office-hour]` channel.
- **Doubts About 'if-else' Use with Mistral**: In a discussion on the `#[general]` channel, `@xifaj78420` wondered about the success of using 'if else' conditions in Mistral prompts. `@sophiamyang` encouraged testing it out.
- **High Latency on La Plateforme**: `@silk.ai` and `@_definitely_not_sam_` raised the issue that **La Plateforme** has high latency, which is affecting their production-level usage. The problem has been recognized by `@lerela` from the support team.
- **Mixtral 8x7B Model Fine-tuned Using LLaMA2-Accessory**: `@cpxjj` declared having successfully fine-tuned the **Mixtral 8x7B model** using LLaMA2-Accessory. The resulting **SPHINX-MoE** model is accessible on their [GitHub repository](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX).
- **Concerns About Lack of Frequency and Presence Penalties in Mistral AI Client**: `@spaceemotion` reported having trouble adding the Mistral AI Client to their project because it fails to adjust for frequency and presence penalties, according to a conversation on the `#[la-plateforme]` channel.
- **Recommendation for 8* 7b Model Deployment on Azure Server**: In a discussion in the `#[deployment]` channel, `@casper_ai` suggested that **a minimum 2x A100 or H100 server setup** is required for deploying an 8* 7b model on Azure for optimal performance.
- **Broken Guardrailing Link in Mistral Docs**: In the `#[ref-implem]` channel, `@productiondown` noted a broken link in the Mistral documentation: [https://docs.mistral.ai/usage/guardrailing](https://docs.mistral.ai/usage/guardrailing).
- **Strategic Advice for Ubuntu LTS Users**: In a concept on the `#[random]` channel, `@cognitivetech` suggested installing the **previous Ubuntu LTS when a new one becomes available** to avoid bugs and gain full software support.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (25 messages🔥): 
        
- **Mistral Team's Office Hours**: `@sophiamyang` from the Mistral team proposed a trial run of **text-based office hours** on Discord next week to address user concerns. They're willing to explore live video ones based on feedback.
- **Fine-tuning Mistral Models**: In response to `@sublimatorniq` and `_dampf`'s queries, `@sophiamyang` confirmed that **fine-tuning on the platform** is under development. `_dampf` highlighted issues in tuning performance of Mixtral models, coralborating it with `@284810978552578050`'s subpar Dolphin model results.
- **Probing 'if-else' use  in Mistral**: `@xifaj78420` asked if anyone managed to successfully use 'if else' conditions in their Mistral prompt, `@sophiamyang` responded recommending to test this out.
- **Model Evaluation Discussions**: The conversation continued over the accuracy of relying on logic tests to evaluate AI models, `.skyair` suggests that real-world applications are a more effective method of testing, while `@i_am_dom` claims this may be an inefficient process.
- **AI Model 'Adapting'**: `@meyelo` humorously described an encounter where despite actively soliciting, their request was politely declined by the AI model.

Additional Notes:
Conversation is majorly around problems and potential solutions related to fine tuning and testing AI models, with some users suggesting possible workarounds to the ongoing issues. There is also a discussion on the practicality and accuracy of testing methods. Several users also express interest in testing specific functionalities, such as 'if-else' conditions.


**Links mentioned**:

- [Join the Mistral AI Discord Server!](https://discord.gg/JfZCaxt4?event=1192782416285278332): Check out the Mistral AI community on Discord - hang out with 8444 other members and enjoy free voice and text chat.
- [cognitivecomputations/dolphin-2.5-mixtral-8x7b · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b)


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (1 messages): 
        
10anant10: Hey anyone wanna build something together


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **Azure Server Recommendations for 8* 7b Model Deployment**: User `@sankar.san` queried about the **appropriate server** for deploying an 8* 7b model on Azure cloud for optimum performance. `@casper_ai` recommended a **minimum 2x A100 or H100** server setup.


### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (1 messages): 
        
productiondown: Hey folks, https://docs.mistral.ai/usage/guardrailing this link is broken


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (3 messages): 
        
- **LLaMA2-Accessory tuned Mixtral 8x7B model** : User `@cpxjj` announced the successful fine-tuning of the **Mixtral 8x7B** model using LLaMA2-Accessory, which has resulted in a new model named **SPHINX-MoE**. The project is accessible on their [GitHub repository](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX).
- **Project Proposal to Match Documents**: User `@.tostino` sought feedback on a project idea involving matching up two different documents - a claim asking for money back for some products the customer ordered and a contract with a list of eligible products/companies/other attributes. The plan was to use a long-context instruct embedding model and further train it on the instruction/positive/negative triplets for the dataset.  
- **Request for Finetuning and Self-hosting Advice**: User `@subham5089` enquired about the best way to fine-tune a mistral or a mixtral model and self-host it on AWS or Azure. They welcomed suggestions on finetuning as well as required services for hosting on the mentioned platforms.

**Links mentioned**:

- [LLaMA2-Accessory/SPHINX at main · Alpha-VLLM/LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX): An Open-source Toolkit for LLM Development. Contribute to Alpha-VLLM/LLaMA2-Accessory development by creating an account on GitHub.
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct)


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=aXeU6mVRgiA


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (3 messages): 
        
- **Consider Using Previous Ubuntu LTS Versions**: User `@cognitivetech` presented a strategy to avoid bugs and software support issues: installing the **previous Ubuntu LTS** when a new one comes out, allowing time for major fixes and support for the new version to be established.
- **Human Consciousness Evolution**: `@cognitivetech` shared a philosophical thought on the mutable nature of consciousness through history, suggesting people of the past may not have experienced consciousness as we understand it today.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (21 messages🔥): 
        
- **La Platform has High Latency Issues**: `@silk.ai` reported that an average **120-token request** on platform takes **9 to 20 seconds** to process, which is too high for production-level usage. The issue was also confirmed by `@_definitely_not_sam_`. `@lerela` from the support team launched an investigation to identify the cause behind such high latency. ([Link to Discussion](https://rentry.co/v5t8x))
- **Mixtral 8x7B in French Insurance Company**: `@eight_me` from a French insurance company expressed interest in using **La Plateforme** and **Mixtral 8x7B model** for building an AI Assistant. They inquired about the possibility of adding **function calling** to the platform and its endpoints. 
- **Request for OpenAI Compatibility**: `@louis030195` requested the Mistral team to make their API compatible with OpenAI, referring to the issue about missing `created` in the chat stream. ([Link to Issue](https://github.com/64bit/async-openai/issues/173))
- **Hosting of La Plateforme Services**: `@johann1613` inquired about the hosting location of **La Plateforme services**. `@eight_me` confirmed from the privacy policy that the services are **hosted in Sweden via Azure**.
- **Issues and Suggestions for Mistral AI Client**: `@spaceemotion` faced issues in adding the Mistral AI Client to their project due to lack of options for **frequency and presence penalties**. They also reported a CORS request issue during the basic fetch process. `@toonb` suggested adding an identifying string when creating API keys for better tracking of application usage.

**Links mentioned**:

- [Changing base url - Usage with open source LLM - Invalid status code: 404 Not Found · Issue #173 · 64bit/async-openai](https://github.com/64bit/async-openai/issues/173): Hey I&#39;m trying to use async-openai with axum and open source LLMs through perplexity.ai in my test. Basically my endpoint would route the request to OpenAI API or an OpenAI API like API changing t...
- [Average Time: 5.900000 seconds](https://rentry.co/v5t8x): Min Time: 3.0 seconds Max Time: 18.0 seconds 95th Percentile Time: 9.0 seconds Making 10 API calls... Call 1: Time taken: 6.0 seconds Response: {&quot;id&quot;:&quot;cmpl-9a5b9869eed84bcead1c6a04df994...


### ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/) (1 messages): 
        
- **Announcement of First Office Hour**: `@sophiamyang` announced that the **first office hour** is scheduled for **Jan 10 at 11:00 am EST (5pm CET).** The office hour will last for one hour.


        

---

## [Datasette/LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **High Quality AI Conversations**: @simonw discussed the need for more high-quality conversations about AI, sharing a [blog post](https://simonwillison.net/2024/Jan/7/call-it-ai/) where he suggested embracing the term "***AI***, despite its imperfections".
- **Literal Language Challenges**: @antisimplistic humorously noted that if every term were used literally, English speakers would encounter many difficulties.
- **Classroom AI**: @derekpwillis plans to educate a class on AI using the general term "***AI***" and will provide an explanation of the specific components of the technology.
- **PATH Resolution**: @thale7166 resolved an issue by implementing use of the **full path** and loading `bashrc` for `PATH` upon startup.
- **OpenAI's New Feature**: @antisimplistic broke the news that **OpenAI** has started **per-key tracking**.

**Datasette/LLM (@SimonW) Channel Summaries**

### ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (3 messages): 
        
- **High Quality Conversations about AI Needed**: `@simonw` shared a blog post about the necessity of having high-quality conversations about AI. In his [post](https://simonwillison.net/2024/Jan/7/call-it-ai/), he advocated for the acceptance of the term "AI", despite its imperfections.
- **"AI" Usage in Classroom**: `@derekpwillis` revealed his plan to teach a class on AI and to use the general term "AI" while also explaining specific components of the technology.
- **Irony in Language Literalism**: `@antisimplistic` humorously pointed out that if every term were to be used in its literal meaning, speakers of English would face plenty of challenges.

**Links mentioned**:

[It’s OK to call it Artificial Intelligence](https://simonwillison.net/2024/Jan/7/call-it-ai/): We need to be having high quality conversations about AI: what it can and can’t do, its many risks and pitfalls and how to integrate it into society in the …


### ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/) (2 messages): 
        
- **Path and bashrc loading solve issue**: User `@thale7166` confirmed that using a **full path** and loading `bashrc` for `PATH` at the start has solved their problem.
- **OpenAI Launches Per-key Tracking**: User `@antisimplistic` announces that **OpenAI has enabled per-key tracking**.


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **NPM Descends into Chaos with 'Everything' Package**:
  `@venadore` shared a fascinating [blogpost](https://socket.dev/blog/when-everything-becomes-too-much) about a disruptive NPM package aptly named **'everything'**. Created by user **PatrickJS** (also known as [gdi2290](https://socket.dev/npm/user/gdi2290)), the package depends on all other public NPM packages. This leads to a Denial of Service (DoS) attack for anyone attempting to install it due to the behemoth amount of transitive dependencies. The extent of the disruption can be viewed at http://everything.npm.lol. The reaction to this chaos was minimal, with only `@venadore` remarking on the situation as being "**incredible**".

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (2 messages): 
        
- **NPM chaos with the 'everything' package**: User `@venadore` shared an intriguing [blogpost](https://socket.dev/blog/when-everything-becomes-too-much) regarding a disruptive NPM package named **'everything'** created by user PatrickJS or [gdi2290](https://socket.dev/npm/user/gdi2290). This package depends on all other public NPM packages, leading to a Denial of Service (DOS) for anyone installing it due to the massive amount of transitive dependencies. PatrickJS further showcased the chaos unleashed via http://everything.npm.lol.
- **Reaction to NPM 'everything' chaos**: In response, `@venadore` commented with a simple one-word reaction, labeling it as "**incredible**".

**Links mentioned**:

[When &quot;Everything&quot; Becomes Too Much: The npm Package Chaos of 2024 - Socket](https://socket.dev/blog/when-everything-becomes-too-much): An NPM user named PatrickJS launched a troll campaign with a package called &quot;everything,&quot; which depends on all public npm packages.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 messages): 
        
teknium: Hi all <a:waveyboy:507416520788279297>


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Exploring Mixture of Experts**: User `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=aXeU6mVRgiA) titled "Mixture of Experts Implementation from scratch", which details an implementation of the **Mixture of Experts** machine learning technique.
- **Trying Mobile Machines with Vision Language**: Another [YouTube video](https://www.youtube.com/watch?v=mzY7ujNb4WA) titled "Trying MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices" was shared by `@pradeep1148` demonstrating a multimodal vision language model designed to operate on mobile devices.

**Links mentioned**:

- [Mixture of Experts Implementation from scratch](https://www.youtube.com/watch?v=aXeU6mVRgiA): We are going to implement Mixture of ExpertsMixture of experts (MoE) is a machine learning technique where multiple expert networks (learners) are used to di...
- [Trying MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices](https://www.youtube.com/watch?v=mzY7ujNb4WA): We present MobileVLM, a competent multimodal vision language model (MMVLM) targeted to run on mobile devices. It is an amalgamation of a myriad of architectu...

        

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
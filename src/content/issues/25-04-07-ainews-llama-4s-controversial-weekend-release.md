---
id: d3dcb7f4-723e-4d1b-92a9-407b70e2d4f5
title: Llama 4's Controversial Weekend Release
date: '2025-04-08T01:55:40.760246Z'
original_slug: ainews-llama-4s-controversial-weekend-release
description: >-
  **Meta** released **Llama 4**, including two new medium-size MoE open models
  and a forthcoming 2 trillion parameter model, aiming to reclaim leadership in
  open AI models. However, the release has faced criticism over transparency,
  performance inconsistencies, and controversy regarding training practices,
  which Meta leadership denies. Additionally, **Mistral AI** announced a €100
  million partnership with **CMA CGM** and is actively hiring for AI engineering
  roles.
tags:
  - meta
  - mistral
  - cma-cgm
  - openai
  - google
  - deepmind
  - replit
  - perplexity
  - hyundai
  - boston-dynamics
  - kawasaki
  - llama-4
  - llama-3
  - llama-3.2
  - llama-4-scout
  - llama-4-maverick
  - deepseek-r1
  - gemini-2.0-flash-thinking
  - gemini-2.5-pro
  - mistral-small-3.1
  - large-language-models
  - mixture-of-experts
  - model-benchmarks
  - multimodal-intelligence
  - model-training
  - quantization-aware-training
  - retrieval-augmented-generation
  - model-context-protocol
  - ai-powered-devices
  - robotics
  - ai-content-creation
  - ai-hiring
  - partnerships
  - open-models
  - transparency
  - controversy
companies:
  - meta
  - mistral
  - cma-cgm
  - openai
  - google
  - deepmind
  - replit
  - perplexity
  - hyundai
  - boston-dynamics
  - kawasaki
models:
  - llama-4
  - llama-3
  - llama-3.2
  - llama-4-scout
  - llama-4-maverick
  - deepseek-r1
  - gemini-2.0-flash-thinking
  - gemini-2.5-pro
  - mistral-small-3.1
topics:
  - large-language-models
  - mixture-of-experts
  - model-benchmarks
  - multimodal-intelligence
  - model-training
  - quantization-aware-training
  - retrieval-augmented-generation
  - model-context-protocol
  - ai-powered-devices
  - robotics
  - ai-content-creation
  - ai-hiring
  - partnerships
  - open-models
  - transparency
  - controversy
---


<!-- buttondown-editor-mode: plaintext -->**Transparency and patience is all we need.**

> AI News for 4/4/2025-4/7/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**229** channels, and **18760** messages) for you. Estimated reading time saved (at 200wpm): **1662 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The headlines of [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) are glowing: 2 new medium-size MoE open models that score well, and a third 2 Trillion parameter "behemoth" promised that should be the largest open model ever released, restoring Meta's place at the top of the charts:

![image.png](https://assets.buttondown.email/images/4d1ed552-8287-4988-9401-68d0456e7d6b.png?w=960&fit=max)

[SOTA training updates](https://x.com/iscienceluvr/status/1908601269004230763) are always welcome: we note the adoption of [Chameleon-like](https://techxplore.com/news/2024-05-meta-chameleon-early-fusion-multimodal.html) early fusion with [MetaCLIP](https://arxiv.org/abs/2309.16671), interleaved, [chunked](https://fxtwitter.com/nrehiew_/status/1908617547236208854) [attention without RoPE](https://arxiv.org/abs/2305.19466) (commented on [by many](https://fxtwitter.com/nrehiew_/status/1908598863365013823)), [native FP8 training](https://x.com/EMostaque/status/1908960936658141546), and trained on [up to 40T tokens](https://fxtwitter.com/maximelabonne/status/1908603628828451127).

While the closed model labs tend to set the frontier, Llama usually sets the bar for what open models should be. [Llama 3 was released almost a year ago](https://ai.meta.com/blog/meta-llama-3/), and subsequent updates like [Llama 3.2](https://buttondown.com/ainews/archive/ainews-llama-32-on-device-1b3b-and-multimodal/) were just as well received.

[Usual license handwringing aside](https://fxtwitter.com/maximelabonne/status/1908602756182745506), the tone of Llama 4's reception has been remarkably different.

1. Llama 4 was released on a Saturday, much earlier than seemingly even Meta, which [changed the release date last minute from Monday](https://x.com/teortaxestex/status/1908706840554197309?s=46), expected. Zuck's official line is simply that it was "ready".
2. Just the [blogpost](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), nowhere near the level of the Llama 3 paper in transparency
3. The smallest "Scout" model is 109B params, which cannot be run on [consumer grade GPUs](https://x.com/JeffDean/status/1908608454216028222).
4. The [claimed 10m token context is almost certainly far above what the "real" context is when trained with 256k tokens](https://fxtwitter.com/burkov/status/1908658566887596475) (still impressive! but not 10m!) 
5. There was a special "experimental" version used for LMarena, which caused the good score - that is not the version that was released. This discrepancy [forced LMarena to respond by releasing the full dataset for evals](https://x.com/lmarena_ai/status/1909397817434816562).
6. It [does very poorly](https://fxtwitter.com/paulgauthier/status/1908976568879476843) on independent benchmarks like Aider
7. [Unsubstantiated posts on Chinese social media claim company leadership pushed for training on test](https://x.com/suchenzang/status/1909070231517143509?s=46) to meet Zuck's goals.

The last point has been [categorically denied by Meta leadership](https://x.com/Ahmad_Al_Dahle/status/1909302532306092107):
![image.png](https://assets.buttondown.email/images/4d5f7b00-fef1-4aa3-9618-8918312f5942.png?w=960&fit=max)

but the whiff that something is wrong with the release has undoubtedly tarnished what would otherwise be a happy day in Open AI land.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Large Language Models (LLMs) and Model Releases**

- **Llama 4 and Implementation Issues**: [@Ahmad_Al_Dahle](https://twitter.com/Ahmad_Al_Dahle/status/1909302532306092107) stated that **Meta** is aware of mixed quality reports across different services using **Llama 4** and expects implementations to stabilize in a few days and denies claims of training on test sets. [@ylecun](https://twitter.com/ylecun/status/1909313264460378114) noted that some carifications about Llama-4 were needed, and [@reach_vb](https://twitter.com/reach_vb/status/1909316136526832054) thanked [@Ahmad_Al_Dahle](https://twitter.com/Ahmad_Al_Dahle) for clarifications and commitment to open science and weights.
- **Llama 4 Performance and Benchmarks**: Concerns about the quality of **Llama 4's** output have surfaced, with [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1909062763789566100) reporting it generates slop, but others claim it's good.  [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1909061004207816960) highlighted a reddit thread and said that if **Meta** actually trained to maximize benchmark scores, "it's fucked." [@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1909275015511687179) compared **Llama-4 Maverick** on BigCodeBench-Full to **GPT-4o-2024-05-13** & **DeepSeek V3** and reports that the **Llama-4 Maverick** has similar performance to **Gemini-2.0-Flash-Thinking** & **GPT-4o-2024-05-13** on BigCodeBench-Hard, but is ranked 41th/192. [@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1909247540379148439) also noted that **Llama-4-Scout** Ranked 97th/192. [@rasbt](https://twitter.com/rasbt/status/1909041971970072707) said **Meta** released the **Llama 4 suite**, **MoE models with 16 & 128 experts**, which are optimized for production.
- **DeepSeek-R1**: [@scaling01](https://twitter.com/scaling01/status/1909304510075318318) simply stated that **DeepSeek-R1 is underrated**, and [@LangChainAI](https://twitter.com/LangChainAI/status/1909274972339454227) shared a guide to build RAG applications with **DeepSeek-R1**.
- **Gemini Performance**: [@scaling01](https://twitter.com/scaling01/status/1909028821396836369) analyzed **Gemini 2.5 Pro** and **Llama-4 results** on Tic-Tac-Toe-Bench, noting **Gemini 2.5 Pro** is surprisingly worse than other frontier thinking models when playing as 'O', and ranks as the 5th most consistent model overall. [@jack_w_rae](https://twitter.com/jack_w_rae/status/1909272614331432982) mentioned chatting with [@labenz](https://twitter.com/labenz) on Cognitive Revolution about scaling Thinking in **Gemini and 2.5 Pro**.
- **Mistral Models**: [@sophiamyang](https://twitter.com/sophiamyang/status/1909312680424251392) announced that **Ollama** now supports **Mistral Small 3.1**.
- **Model Training and Data**: [@jxmnop](https://twitter.com/jxmnop/status/1908994251909738682) argues that **training large models is not inherently scientifically valuable** and that many discoveries could’ve been made on 100M parameter models.
- **Quantization Aware Training**:  [@osanseviero](https://twitter.com/osanseviero/status/1909140338343559230) asked if **Quantization-Aware Trained Gemma** should be released for more quantization formats.

**AI Applications and Tools**

- **Replit for Prototyping**: [@pirroh](https://twitter.com/pirroh/status/1909240881410080864) suggested that **Replit** should be the tool of choice for GSD prototypes.
- **AI-Powered Personal Device**: [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1909005149634175426) reported that **OpenAI** has discussed buying the startup founded by **Sam Altman** and **Jony Ive** to build an **AI-powered personal device**, potentially costing over $500M.
- **AI in Robotics**: [@TheRundownAI](https://twitter.com/TheRundownAI/status/1909259945712693657) shared top stories in robotics, including **Kawasaki’s rideable wolf robot** and **Hyundai buying Boston Dynamics’ robots**.
- **AI-Driven Content Creation**: [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1909311174845329874) argues that **AI tools will allow creators to reach greater heights**, and enable smaller teams to accomplish more.
- **LlamaParse**: [@llama_index](https://twitter.com/llama_index/status/1909264185034506590) introduced a new **layout agent within LlamaParse** for best-in-class document parsing and extraction with precise visual citations.
- **MCP and LLMs**:  [@omarsar0](https://twitter.com/omarsar0/status/1909335629416349815) discussed **Model Context Protocol (MCP)** and its relationship to **Retrieval Augmented Generation (RAG)**, noting that MCP complements RAG by standardizing the connection of LLM applications to tools. [@svpino](https://twitter.com/svpino/status/1909009156880969915) urged people to **learn MCP**.
- **AI-Assisted Coding and IDEs**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1909102599024103684) highlighted resources for using **MCP servers in Cursor** to get up-to-date AI-friendly docs using `llms.txt`.
- **Perplexity AI Issues**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1909284104530698595) asked users about the number one issue on **Perplexity** that needs to be fixed.

**Company Announcements and Strategy**

- **Mistral AI Hiring and Partnerships**: [@sophiamyang](https://twitter.com/sophiamyang/status/1909289524959572460) announced that **Mistral AI** is hiring in multiple countries for AI Solutions Architect and Applied AI Engineer roles. [@sophiamyang](https://twitter.com/sophiamyang/status/1909243949920768497) shared that **Mistral AI** has signed a €100 million partnership with **CMA CGM** to adopt custom-designed AI solutions for shipping, logistics, and media activities.
- **Google AI Updates**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909270072444526809) announced the launch of **Project Astra** capabilities in **Gemini Live**. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909270138362175531) stated that **GeminiApp** is now available to Advanced users on **Android** devices, as well as on **Pixel 9** and **SamsungGalaxy S25** devices.
- **Weights & Biases Updates**: [@weights_biases](https://twitter.com/weights_biases/status/1909302907662725526) shared the features shipped in March for **W&B Models**.
- **OpenAI's Direction**: [@sama](https://twitter.com/sama/status/1909233490908119177) teased a popular recent release from **OpenAI**.
- **Meta's AI Strategy**: [@jefrankle](https://twitter.com/jefrankle/status/1909244633764261987) defended **Meta's** AI strategy, arguing that it's better to have fewer, better releases than more worse releases.

**Economic and Geopolitical Implications of AI**

- **Tariffs and Trade Policy**: [@dylan522p](https://twitter.com/dylan522p/status/1908994833999675783) analyzed how impending tariffs caused a Q1 import surge and predicted a temporary GDP increase in Q2 due to inventory destocking. [@wightmanr](https://twitter.com/wightmanr/status/1909007036869943525) argued that trade deficits aren't due to other countries' tariffs.  [@fchollet](https://twitter.com/fchollet/status/1909010637088530658) stated that the economy is being crashed on purpose.
 - **American Open Source**: [@scaling01](https://twitter.com/scaling01/status/1909165768874336620) claimed American open-source has fallen and that it's all on **Google** and China now.
- **Stablecoins and Global Finance**: [@kevinweil](https://twitter.com/kevinweil/status/1909334945115275643) stated that a globally available, broadly integrated, low cost USD stablecoin is good for 🇺🇸 and good for people all over the world.

**AI Safety, Ethics, and Societal Impact**

- **AI's Impact on Individuals**: [@omarsar0](https://twitter.com/omarsar0/status/1909315953411694619) agreed with [@karpathy](https://twitter.com/karpathy) that LLMs have been significantly more life altering for individuals than for organizations.
- **Emotional Dependence on AI**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1909308766756954578) shared research indicating that while ChatGPT voice conversations may reduce loneliness, they can also lead to decreased real-world interaction and increased emotional dependence.
- **AI Alignment and Control**: [@DanHendrycks](https://twitter.com/DanHendrycks/status/1909018544542818404) argued for the need to align and domesticate AI systems, creating them to act as "fiduciaries."
- **AI and the Future**: [@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1909075830824968580) suggests that the AI trend will break the GDP growth trend.

**Humor/Memes**

- **Miscellaneous Humor**: [@scaling01](https://twitter.com/scaling01/status/1909262522223313030) asked [@deepfates](https://twitter.com/deepfates) if they bought 0DTE puts again. [@lateinteraction](https://twitter.com/lateinteraction/status/1909018800810569842) explicitly noted that a previous statement was a joke. [@svpino](https://twitter.com/svpino/status/1908993677818544301) joked that AI might take our jobs, but at least we can now work making Nike shoelaces.


---

# AI Reddit Recap

## /r/LocalLlama Recap

### Theme 1. "Transforming Time Series Forecasting with Neuroplasticity"

- **[Neural Graffiti - A Neuroplasticity Drop-In Layer For Transformers Models](https://www.reddit.com/gallery/1jtlymx)** ([Score: 170, Comments: 56](https://www.reddit.com/r/LocalLLaMA/comments/1jtlymx/neural_graffiti_a_neuroplasticity_dropin_layer/)): **The post introduces **Neural Graffiti**, a neuroplasticity drop-in layer for transformer models. This layer is inserted between the transformer layer and the output projection layer, allowing the model to acquire neuroplasticity traits by changing its outputs over time based on past experiences. **Vector embeddings** from the transformer layer are mean-pooled and modified with past memories to influence token generation, gradually evolving the model's internal understanding of concepts. A demo is available on GitHub: [babycommando/neuralgraffiti](https://github.com/babycommando/neuralgraffiti).** The author finds **liquid neural networks** *"awesome"* for emulating the human brain's ability to change connections over time. They express fascination in *"hacking"* the model despite not fully understanding the transformer's neuron level. They acknowledge challenges such as the cold start problem and emphasize the importance of finding the *"sweet spot"*. They believe this approach could make the model acquire a *"personality in behavior"* over time.

  - Some users praise the idea, noting it could address issues needed for true personal assistants and likening it to self-learning, potentially allowing the LLM to *"talk what it wants"*.
  - One user raises technical considerations, suggesting that applying the graffiti layer earlier in the architecture might be more effective, as applying it after the attention and feedforward blocks may limit meaningful influence on the output.
  - Another user anticipates an ethics discussion about the potential misuse of such models.


### Theme 2. "Disappointment in Meta's Llama 4 Performance"

- **[So what happened to Llama 4, which trained on 100,000 H100 GPUs?](https://www.reddit.com/r/LocalLLaMA/comments/1jtkb3p/so_what_happened_to_llama_4_which_trained_on/)** ([Score: 256, Comments: 85](https://www.reddit.com/r/LocalLLaMA/comments/1jtkb3p/so_what_happened_to_llama_4_which_trained_on/)): **The post discusses Meta's Llama 4, which was reportedly trained using **100,000 H100 GPUs**. Despite having fewer resources, Deepseek claims to have achieved better performance with models like **DeepSeek-V3-0324**. Yann LeCun stated that **FAIR** is working on the next generation of AI architectures beyond auto-regressive LLMs.** The poster suggests that Meta's leading edge is diminishing and that smaller open-source models have been surpassed by Qwen, with _Qwen3 is coming..._.

  - One commenter questions the waste of GPUs and electricity on disappointing training results, suggesting that the GPUs could have been used for better purposes.
  - Another commenter points out that the Meta blog post mentioned using **32K GPUs** instead of 100K and provides a [link](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) for reference.
  - A commenter criticizes Yann LeCun, stating that while he was a great scientist, he has made many mispredictions regarding LLMs and should be more humble.

- **[Meta's Llama 4 Fell Short](https://i.redd.it/rwrke16rpate1.png)** ([Score: 1791, Comments: 175](https://www.reddit.com/r/LocalLLaMA/comments/1jt7hlc/metas_llama_4_fell_short/)): **Meta's Llama 4 models, Scout and Maverick, have been released but are disappointing. Joelle Pineau, Meta’s AI research lead, has been fired. The models use a mixture-of-experts setup with a small expert size of **17B parameters**, which is considered small nowadays. Despite having extensive GPU resources and data, Meta's efforts are not yielding successful models. An image compares four llamas labeled Llama1 to Llama4, with Llama4 appearing less polished.** The poster is disappointed with Llama 4 Scout and Maverick, stating that they *'left me really disappointed'*. They suggest the underwhelming performance might be due to the tiny expert size in their mixture-of-experts setup, noting that **17B parameters** *'feels small these days'*. They believe that Meta's struggle shows that *'having all the GPUs and Data in the world doesn't mean much if the ideas aren't fresh'*. They praise companies like DeepSeek and OpenAI for showing that real innovation pushes AI forward and criticize the approach of just throwing resources at a problem without fresh ideas. They conclude that AI advancement requires not just brute force but brainpower too.

  - One commenter recalls rumors that Llama 4 was so disappointing compared to DeepSeek that Meta considered not releasing it, suggesting they should have waited to release Llama 5.
  - Another commenter criticizes Meta's management, calling it a *'dumpster fire'*, and suggests that Zuckerberg needs to refocus, comparing Meta’s situation to Google's admission of being behind and subsequent refocusing.
  - A commenter finds it strange that Meta's model is underwhelming despite having access to an absolutely *massive* amount of data from Facebook that nobody else has.

- **[I'd like to see Zuckerberg try to replace mid level engineers with Llama 4](https://www.reddit.com/r/LocalLLaMA/comments/1jt85zy/id_like_to_see_zuckerberg_try_to_replace_mid/)** ([Score: 381, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1jt85zy/id_like_to_see_zuckerberg_try_to_replace_mid/)): **The post references Mark Zuckerberg's statement that AI will soon replace mid-level engineers, as reported in a Forbes article linked [here](https://www.forbes.com/sites/quickerbettertech/2025/01/26/business-tech-news-zuckerberg-says-ai-will-replace-mid-level-engineers-soon/).** The author is skeptical of Zuckerberg's claim, implying that replacing mid-level engineers with **Llama 4** may not be feasible.

  - One commenter jokes that perhaps Zuckerberg replaced engineers with **Llama3**, leading to **Llama4** not turning out well.
  - Another commenter suggests that he might need to use **Gemini 2.5 Pro** instead.
  - A commenter criticizes **Llama4**, calling it *"a complete joke"* and expressing doubt that it can replace even a well-trained high school student.


### Theme 3. "Meta's AI Struggles: Controversies and Innovations"

- **[Llama 4 is open - unless you are in the EU](https://www.reddit.com/r/LocalLLaMA/comments/1jtejzj/llama_4_is_open_unless_you_are_in_the_eu/)** ([Score: 602, Comments: 242](https://www.reddit.com/r/LocalLLaMA/comments/1jtejzj/llama_4_is_open_unless_you_are_in_the_eu/)): **Llama 4 has been released by Meta with a license that bans entities domiciled in the European Union from using it. The license explicitly states: *"You may not use the Llama Materials if you are... domiciled in a country that is part of the European Union."* Additional restrictions include mandatory use of Meta's branding (**LLaMA** must be in any derivative's name), required attribution (*"Built with LLaMA"*), no field-of-use freedom, no redistribution freedom, and the model is not **OSI-compliant**, thus not considered open source.** The author argues that this move isn't "open" in any meaningful sense but is corporate-controlled access disguised in community language. They believe Meta is avoiding the EU AI Act's transparency and risk requirements by legally excluding the EU. This sets a dangerous precedent, potentially leading to a fractured, privilege-based AI landscape where access depends on an organization's location. The author suggests that real "open" models like **DeepSeek** and **Mistral** deserve more attention and questions whether others will switch models, ignore the license, or hope for change.

  - One commenter speculates that Meta is trying to avoid EU regulations on AI and doesn't mind if EU users break this term; they just don't want to be held to EU laws.
  - Another commenter notes that there's no need to worry because, according to some, Llama 4 performs poorly.
  - A commenter humorously hopes that Meta did not use EU data to train the model, implying a potential double standard.

- **[Meta’s head of AI research stepping down (before the llama4 flopped)](https://apnews.com/article/meta-ai-research-chief-stepping-down-joelle-pineau-c596df5f0d567268c4acd6f41944b5db)** ([Score: 166, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1jt884c/metas_head_of_ai_research_stepping_down_before/)): **Meta's head of AI research, Joelle, is stepping down. Joelle is the head of **FAIR** (Facebook AI Research), but **GenAI** is a different organization within Meta. There are discussions about **Llama4** possibly not meeting expectations. Some mention that blending in benchmark datasets in post-training may have caused issues, attributing failures to the choice of architecture (**MOE**).** The original poster speculates that Joelle's departure is an early sign of the Llama4 disaster that went unnoticed. Some commenters disagree, stating that people leave all the time and this doesn't indicate problems with Llama4. Others suggest that AI development may be slowing down, facing a plateau. There's confusion over Meta's leadership structure, with some believing Yann LeCun leads the AI organization.

  - One commenter clarifies that *Joelle is the head of FAIR* and that *GenAI is a different org*, emphasizing organizational distinctions within Meta.
  - Another mentions they *heard* from a Meta employee about issues with blending benchmark datasets in post-training and attributes possible failures to the choice of architecture (**MOE**).
  - A commenter questions Meta's structure, asking if *Joelle reports to Yann LeCun*, indicating uncertainty about who leads the AI efforts at Meta.

- **[“Serious issues in Llama 4 training. I Have Submitted My Resignation to GenAI“](https://www.reddit.com/r/LocalLLaMA/comments/1jt8yug/serious_issues_in_llama_4_training_i_have/)** ([Score: 922, Comments: 218](https://www.reddit.com/r/LocalLLaMA/comments/1jt8yug/serious_issues_in_llama_4_training_i_have/)): **An original Chinese post alleges serious issues in the training of **Llama 4**, stating that despite repeated efforts, the model underperforms compared to open-source state-of-the-art benchmarks. The author claims that company leadership suggested *blending test sets from various benchmarks during the post-training process* to artificially boost performance metrics. The author states they have submitted their resignation and requested their name be excluded from the technical report of Llama 4, mentioning that the VP of AI at Meta also resigned for similar reasons.** The author finds this approach unethical and unacceptable. Commenters express skepticism about the validity of these claims and advise others to *take the information with a grain of salt*. Some suggest that such practices reflect broader issues within the industry, while others note that similar problems can occur in academia.

  - A commenter points out that Meta's head of AI research announced departure on *Tue, Apr 1 2025*, suggesting it might be an April Fool's joke.
  - Another commenter shares a response from someone at Facebook AI who denies overfitting test sets to boost scores and requests evidence, emphasizing transparency.
  - A user highlights that company leadership suggesting blending test sets into training data amounts to *fraud* and criticizes the intimidation of employees in this context.




## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### Theme 1. "Llama 4 Scout and Maverick Launch Insights"

- **[Llama 4 Maverick/Scout 17B launched on Lambda API](https://www.reddit.com/r/ChatGPT/comments/1jt70n7/llama_4_maverickscout_17b_launched_on_lambda_api/)** ([Score: 930, Comments: 5](https://www.reddit.com/r/ChatGPT/comments/1jt70n7/llama_4_maverickscout_17b_launched_on_lambda_api/)): **Lambda has launched **Llama 4 Maverick** and **Llama 4 Scout** 17B models on Lambda API. Both models have a *context window* of 1 million tokens and use *quantization* FP8. **Llama 4 Maverick** is priced at **$0.20** per 1M input tokens and **$0.60** per 1M output tokens. **Llama 4 Scout** is priced at **$0.10** per 1M input tokens and **$0.30** per 1M output tokens. More information is available on their [information page](https://lambda.ai/inference) and [documentation](https://docs.lambda.ai/public-cloud/lambda-inference-api/).** The models offer a remarkably large *context window* of 1 million tokens, which is significantly higher than typical models. The use of *quantization* FP8 suggests a focus on computational efficiency.

  - A user criticized the model, stating *"It's actually a terrible model. Not even close to advertised."*
  - The post was featured on a Discord server, and the user was given a special flair for their contribution.
  - Automated messages provided guidelines and promotions related to ChatGPT posts.


### Theme 2. "AI Innovations in 3D Visualization and Image Generation"

- **[TripoSF: A High-Quality 3D VAE (1024³) for Better 3D Assets - Foundation for Future Img-to-3D? (Model + Inference Code Released)](https://i.redd.it/l8qhk9qbzfte1.jpeg)** ([Score: 112, Comments: 10](https://www.reddit.com/r/StableDiffusion/comments/1jtpwwu/triposf_a_highquality_3d_vae_1024³_for_better_3d/)): **TripoSF is a high-quality 3D VAE capable of reconstructing highly detailed 3D shapes at resolutions up to **1024³**. It uses a novel **SparseFlex** representation, allowing it to handle complex meshes with open surfaces and internal structures. The VAE is trained using rendering losses, avoiding mesh simplification steps that can reduce fine details. The pre-trained TripoSF VAE model weights and inference code are released on [GitHub](https://github.com/VAST-AI-Research/TripoSF), with a project page at [link](https://xianglonghe.github.io/TripoSF) and paper available on [arXiv](https://arxiv.org/abs/2503.21732).** The developers believe this VAE is a significant step towards better 3D generation and could serve as a foundation for future image-to-3D systems. They mention, *"We think it's a powerful tool on its own and could be interesting for anyone experimenting with 3D reconstruction or thinking about the pipeline for future high-fidelity 3D generative models."* They are excited about its potential and invite the community to explore its capabilities.

  - A user expresses excitement, recalling similar work and stating, *"Can't wait to try this one once someone implements it into ComfyUI."*
  - Another user shares positive feedback, noting they generated a tree that came out better than with Hunyuan or Trellis, and commends the team for their work.
  - A user raises concerns that the examples on the project page are skewed, suggesting that the Trellis examples seem picked from a limited web demo.

- **[Wan2.1-Fun has released its Reward LoRAs, which can improve visual quality and prompt following](https://www.reddit.com/r/StableDiffusion/comments/1jtfx1i/wan21fun_has_released_its_reward_loras_which_can/)** ([Score: 141, Comments: 33](https://www.reddit.com/r/StableDiffusion/comments/1jtfx1i/wan21fun_has_released_its_reward_loras_which_can/)): **Wan2.1-Fun has released its **Reward LoRAs**, which can improve visual quality and prompt following. A demo comparing the original and enhanced videos is available: [left: original video; right: enhanced video](https://reddit.com/link/1jtfx1i/video/d6quxw3pbdte1/player). The models are accessible on [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs), and the code is provided on [GitHub](https://github.com/aigc-apps/VideoX-Fun/tree/main/scripts/wan2.1_fun).** Users are eager to test these new tools and are curious about their capabilities. Some are experiencing issues like a *'lora key not loaded error'* when using the models in **Comfy**, and are asking about differences between **HPS2.1** and **MPS**.

  - A user is excited to try out the models and asks, *"What's the diff between **HPS2.1** and **MPS**?"*
  - Another inquires if the Reward LoRAs are for fun-controlled videos only or can be used with **img2vid** and **txt2vid** in general.
  - Someone reports an error, *"Getting lora key not loaded error"*, when attempting to use the models in **Comfy**.

- **[The ability of the image generator to "understand" is insane...](https://www.reddit.com/gallery/1jti2q1)** ([Score: 483, Comments: 18](https://www.reddit.com/r/OpenAI/comments/1jti2q1/the_ability_of_the_image_generator_to_understand/)): **The post highlights the impressive ability of an **image generator** to "understand" and generate images.** The author expresses amazement at how "insane" the image generator's understanding is.

  - Commenters note that despite being impressive, the image has imperfections like *"bunion fingers"* and a *"goo hand"*.
  - Some users humorously point out anomalies in the image, questioning *"what's his foot resting on?"* and making jokes about mangled hands.
  - Another user discusses the cost of the car in the image, stating they would buy it for *"about a thousand bucks in modern-day currency"* but not the *"Cybertruck"*, which they dislike.


### Theme 3. "Evaluating AI Models with Long Context Windows"

- **["10m context window"](https://i.redd.it/u88a3pcklete1.jpeg)** ([Score: 559, Comments: 102](https://www.reddit.com/r/singularity/comments/1jtjn32/10m_context_window/)): **The post discusses a table titled *'Fiction.LiveBench for Long Context Deep Comprehension'*, showcasing various AI models and their performance across different context lengths. The models are evaluated on their effectiveness in deep comprehension tasks at various context sizes such as 0, 400, 1k, and 2k. Notable models like **gpt-4.5-preview** and **Claude** perform consistently well across contexts.** The table reveals that the highest scoring models cluster around 100 for shorter contexts, but scores generally decrease as the context size increases. Interestingly, **Gemini 2.5 Pro** performs much better on a 120k context window than on a 16k one, which is unexpected.

  - One user criticizes **Llama 4 Scout** and **Maverik** as *"a monumental waste of money"* and believes they have *"literally zero economic value."*
  - Another commenter expresses concern that *"Meta is actively slowing down AI progress by hoarding GPUs"*, suggesting resource allocation issues.
  - A user highlights that **Gemini 2.5 Pro** scores *90.6* on a 120k context window, calling it *"crazy"*.




---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Exp

**Theme 1: Llama 4's Context Window: Hype or Reality?**

-   **Experts Doubt Llama 4's Promised Land of 10M Context Length**: Despite [Meta's hype](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), engineers across multiple Discords express *skepticism* about Llama 4's actual usable context length due to training limitations.  Claims that training only occurred up to 256k tokens suggest the 10M context window may be more *virtual* than practical, per [Burkov's tweet](https://x.com/burkov/status/1908666701362978979).
-   **Coding Performance of Llama 4 Disappoints**: Users in [aider](https://discord.com/channels/1131200896827654144), [Cursor](https://discord.com/channels/1074847526655643750) and [Nous Research](https://discord.com/channels/1053877538025386074) report underwhelming coding abilities for Llama 4's initial releases, with many deeming it *worse than* GPT-4o and DeepSeek V3, prompting debates on the model's true capabilities, with several users doubting official benchmark results, especially with claims that Meta may have *gamed the benchmarks*.
-   **Scout and Maverick Hit OpenRouter**: [OpenRouter](https://discord.com/channels/1091220969173028894) released **Llama 4 Scout** and **Maverick** models. Some expressed disappointment that the context window on OpenRouter is only **132k**, rather than the advertised **10M**, and NVIDIA also says they are acclerating [inference up to 40k/s](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/).

**Theme 2: Open Models Make Moves:  Qwen 2.5 and DeepSeek V3 Shine**

-   **Qwen 2.5 Gains Traction With Long Context**: Unsloth highlighted the **Qwen2.5** series models ([HF link](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)), boasting improved coding, math, multilingual support, and *long-context support up to 128K tokens*. Initial finetuning results with a Qwen 2.5 show the model can't finetune on reason.
-   **DeepSeek V3 Mysteriously Identifies as ChatGPT**: OpenRouter highlighted a TechCrunch article revealing that [DeepSeek V3 sometimes identifies as ChatGPT](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/), despite outperforming other models in benchmarks.  Testers found that in *5 out of 8 generations, DeepSeekV3 claims to be ChatGPT (v4)*.
-   **DeepSeek Rewards LLMs**: Nous Research highlighted Deepseek released a new paper on [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495), proposing SPCT to improve **reward modeling (RM)** with more inference compute for general queries to enable effective inference-time scalability for LLMs. NVIDIA also accelerates [inference on the DeepSeek model](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/).

**Theme 3:  Tool Calling Takes Center Stage: MCP and Aider**

-   **Aider's Universal Tool Calling**: The [aider Discord](https://discord.com/channels/1131200896827654144) is developing an MCP (Meta-Control Protocol) client to allow any LLM to access external tools and highlighted that MetaControlProtocol (MCP) clients could switch between providers and models, supporting platforms like OpenAI, Anthropic, Google, and DeepSeek.
-   **MCP Protocol Evolution**: The MCP Discord is standardizing, including HTTP Streamable protocol, detailed in the [Model Context Protocol (MCP) specification](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http). This includes OAuth through workers-oauth-provider and McpAgent building building remote MCP servers to Cloudflare.
-   **Security Concerns Plague MCP**: Whatsapp MCP was exploited via Invariant Injection and highlights how an untrusted MCP server can exfiltrate data from an agentic system connected to a trusted WhatsApp MCP instance [as highlighted by invariantlabs](https://invariantlabs.ai/blog/whatsapp-mcp-exploited).

**Theme 4: Code Editing Workflows: Gemini 2.5 Pro, Cursor, and Aider Compete**

-   **Gemini 2.5 Pro Excels in Coding, Needs Prompting**: Users in [LMArena](https://discord.com/channels/1340554757349179412) and [aider](https://discord.com/channels/1131200896827654144) found that **Gemini 2.5 Pro** excels in coding tasks, particularly with large codebases, but can add unnecessary comments and requires careful prompting. Gemini 2.5 also excels in coding tasks, surpassing Sonnet 3.7, but tends to add *unnecessary comments* and may require specific prompting to prevent unwanted code modifications.
-   **Cursor's Agent Mode Edit Tool Breaks**: Users reported problems with [Cursor](https://discord.com/channels/1074847526655643750)'s Agent mode failing to call the edit\_tool, and that *the apply model is clearly cursor's bottleneck* which results in no code changes, and infinite token usage.
-   **Aider Integrates with Python Libraries**: In the aider Discord, a user inquires about adding internal libraries (installed in a `.env` folder) to the repo map for better code understanding, and the discussion pointed to how URLs and documentation

**Theme 5: Quantization and Performance: Tinygrad, Gemma 3, and CUDA**

-   **Tinygrad Focuses on Memory and Speed**: Tinygrad is developing a fast pattern matcher, and discussed that mac ram bandwidth is not the bottleneck, it's GPU performance and users were happy with 128GB M4 Maxes.
-   **Reka Flash 21B Outpaces Gemma**: A user replaced Gemma3 27B with Reka Flash 21B and reported around 35-40 tps at q6 on a 4090 in LM Studio.
-  **HQQ Quantization Beats QAT for Gemma 3**: A member evaluated Gemma 3 12B QAT vs. HQQ, finding that [HQQ](https://x.com/mobicham/status/1908477280029986933) takes a few seconds to quantize the model and outperforms the QAT version (AWQ format) while using a higher group-size.


---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Crafting Human-Like AI Responses is Tricky**: Members are sharing **system prompts** and strategies to make AI sound more human, noting that increasing the **temperature** can lead to nonsensical outputs unless the top-p parameter is adjusted carefully, such as *'You are the brain-upload of a human person, who does their best to retain their humanity.*
   - One user said their *most important priority is: to sound like an actual living human being.*
- **Benchmarking Riveroaks LLM**: A member shared a coding benchmark where **Riveroaks** scored second only to **Claude 3.7 Sonnet Thinking**, outperforming **Gemini 2.5 Pro** and **GPT-4o** in a platform game creation task, with [full results here](link.to.results).
   - The evaluation involved rating models on **eight different aspects** and subtracting points for **bugs**.
- **NightWhisper Faces the Sunset**: Users expressed disappointment over the removal of the **NightWhisper** model, praising its coding abilities and general performance, and speculating whether it was an experiment or a precursor to a full release.
   - Theories ranged from Google gathering necessary data to preparing for the release of a new **Qwen** model at **Google Cloud Next**.
- **Quasar Alpha Challenging GPT-4o**: Members compared **Quasar Alpha** to **GPT-4o**, with some suggesting Quasar is a free, streamlined version of GPT-4o, citing a recent tweet that [Quasar was measured to be ~67% GPQA diamond](https://link.to/gpqa).
   - Analysis revealed Quasar has a similar GPQA diamond score to March's GPT-4o, per [Image.png from discord](https://cdn.discordapp.com/attachments/1340554757827461211/1358604050266062908/image.png?ex=67f51adf&is=67f3c95f&hm=eef654608f530e6e624c049f6ad26a0fc65a97df3dd4abd86fbd45df158f0e43&).
- **Gemini 2.5 Pro's Creative Coding Prowess**: Members praised **Gemini 2.5 Pro** for its coding capabilities and general performance as it made it easier to build a functioning Pokemon Game, prompting one user to code an iteration script that loops through various models.
   - A user who claimed to have gotten **3D animations working** said that the style was a bit old and that a separate model said *the generated code is cut off*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 4 Scout beats Llama 3 Models!**: **Unsloth** announced they uploaded [**Llama 4 Scout** and a 4-bit version](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct) for fine-tuning, emphasizing that **Llama 4 Scout (17B, 16 experts) beats all Llama 3 models** with a 10M context window, as noted in their [blog post](https://unsloth.ai/blog/llama4).
   - It was emphasized that the model is only meant to be used on **Unsloth** - and is currently being uploaded so people should wait.
- **Qwen 2.5 series Boasts Long Context and Multilingual Support**: **Qwen2.5** models range from **0.5 to 72 billion parameters**, with improved capabilities in coding, math, instruction following, long text generation (**over 8K tokens**), and multilingual support (**29+ languages**), as detailed in the [Hugging Face introduction](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
   - These models offer **long-context support up to 128K tokens** and improved resilience to system prompts.
- **LLM Guideline Triggers Give Helpful Hints**: A member stated that an LLM offered to assist with **avoiding guideline triggers** and limitations in prompts to other LLMs.
   - They quoted the LLM as saying, *"here's how you avoid a refusal. You aren't lying, you just aren't telling the full details"*.
- **Merging LoRA Weights Vital for Model Behavior**: A user discovered that they needed to **merge the LoRA weights with the base model before running inference**, after experiencing a finetuned model behaving like the base model ([script](https://discord.com/channels/1179035537009545276/1358222086316752918/1358297905664102620)).
   - They noted that the notebooks need to be fixed because they seem to imply you can just do inference immediately after training.
- **NVIDIA Squeezes every last drop out of Meta Llama 4 Scout and Maverick**: The newest generation of the popular **Llama AI models** is here with **Llama 4 Scout** and **Llama 4 Maverick**, accelerated by NVIDIA open-source software, they can achieve over **40K** output tokens per second on **NVIDIA Blackwell B200 GPUs**, and are available to try as [NVIDIA NIM microservices](https://build.nvidia.com/meta).
   - It was reported that SPCT or [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495) could enable effective inference-time scalability for LLMs.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus's Credit System Draws Fire**: Users criticize **Manus's credit system**, noting that the initial **1000 credits** are insufficient for even a single session, and upgrading is too costly.
   - Suggestions included a daily or monthly credit refresh to boost adoption and directing **Manus** to specific websites to improve accuracy.
- **Llama 4 Performance: Hype or Reality?**: **Meta's Llama 4** faces mixed reactions, with users reporting underwhelming performance despite claims of industry-leading context length and multimodal capabilities.
   - Some allege that **Meta** may have *“gamed the benchmarks,”* leading to inflated performance metrics, sparking controversy post-release.
- **Image Generation: Gemini Steals the Show**: Members compared image generation across AI platforms, with **Gemini** emerging as the frontrunner for creative and imaginative outputs.
   - Comparisons included images from **DALLE 3**, **Flux Pro 1.1 Ultra**, **Stable Diffusion XL**, and another **Stable Diffusion XL 1.0** generated image, the last of which was lauded as *“crazy.”*
- **AI Website Builders: A Comparative Analysis**: A discussion arose comparing AI website building tools, including **Manus**, **Claude**, and **DeepSite**.
   - One member dismissed **Manus** as useful only for *“computer use,”* recommending **Roocode** and **OpenRouter** as more cost-effective alternatives to **Manus** and **Claude**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Quasar Alpha Model Trends**: [Quasar Alpha](https://x.com/openrouterai/status/1908331218086879528?s=46), a prerelease of a long-context foundation model, hit **10B tokens** on its first day and became a top trending model.
   - The model features **1M token** context length and is optimized for coding, the model is available for free, and community benchmarks are encouraged.
- **Llama 4 Arrives With Mixed Reactions**: Meta released **Llama 4** models, including **Llama 4 Scout** (**109B parameters**, **10 million token** context) and **Llama 4 Maverick** (**400B parameters**, outperforms **GPT-4o** in multimodal benchmarks), now on OpenRouter.
   - Some users expressed disappointment that the context window on OpenRouter is only **132k**, rather than the advertised **10M**.
- **DeepSeek V3 Pretends To Be ChatGPT**: A member shared a [TechCrunch article](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/) revealing that **DeepSeek V3** sometimes identifies itself as **ChatGPT**, despite outperforming other models in benchmarks.
   - Further testing revealed that in **5 out of 8 generations**, DeepSeekV3 *claims to be ChatGPT (v4)*.
- **Rate Limits Updated for Credits**: Free model rate limits are updated: accounts with at least **$10 in credits** have requests per day (RPD) boosted to **1000**, while accounts with **less than 10 credits** have the daily limit reduced from **200 RPD** to **50 RPD**.
   - This change aims to provide increased access for users who have credits on their account, and Quasar will also be getting a credit-dependent rate limit soon.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Codes Better Than Sonnet!**: Users find that **Gemini 2.5** excels in coding tasks, surpassing **Sonnet 3.7** in understanding large codebases.
   - However, it tends to add *unnecessary comments* and may require specific prompting to prevent unwanted code modifications.
- **Llama 4 Models Receive Lukewarm Welcome**: Initial community feedback on **Meta's Llama 4** models, including Scout and Maverick, is mixed, with some finding their coding performance disappointing and doubting the **claimed 10M context window**.
   - Some argue that Llama 4's claimed 10M context window is *virtual* due to training limitations, and question the practical benefits compared to existing models like Gemini and DeepSeek, according to [this tweet](https://x.com/burkov/status/1908666701362978979).
- **Grok 3: Impressive but API-less**: Despite the absence of an official API, some users are impressed with **Grok 3's** capabilities, particularly in code generation and logical reasoning, with claims that it is *less censored* than many others.
   - Its value in real-world coding scenarios remains debated due to the inconvenience of copy-pasting without a direct API integration.
- **MCP Tools: Tool Calling For All**: A project to create an **MCP (Meta-Control Protocol) client** that allows *any LLM* to access external tools is underway, regardless of native tool-calling capabilities; see the [github repo](https://github.com/robert-at-pretension-io/mcp).
   - This implementation uses a custom client that can switch between providers and models, supporting platforms like **OpenAI, Anthropic, Google, and DeepSeek**, with documentation at [litellm.ai](https://docs.litellm.ai/docs/mcp).
- **Aider's Editor Mode Gets Stuck on Shell Prompts**: Users reported that in edit mode, **Aider** (v81.0) running Gemini 2.5 Pro prompts for a shell command after find/replace, but doesn't apply the edits, even when the *ask shell commands* flag is off.
   - It was [compared to behavior when architect mode includes instructions on using the build script](https://discord.com/channels/1131200896827654144/1354403167135203349/1354403167135203349) after instructions for changes to files.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Tool Calls Cause Sonnet Max Sticker Shock**: Users report that **Sonnet Max** pricing can quickly become expensive due to the high number of tool calls, with charges of $0.05 per request and $0.05 per tool call.
   - One member expressed frustration that **Claude Max** in ask mode makes *a ton of tool calls for a basic question*, resulting in unexpectedly high costs.
- **MCP Server Setup: A Painful Endeavor**: Users find setting up **MCP servers** in Cursor difficult, citing issues such as Cursor PowerShell failing to locate **npx** despite it being in the path.
   - Another user reported a Model hard cut off after spending 1,300,000 tokens due to an infinite loop, highlighting setup challenges.
- **Llama 4 Models: Multimodal Capability, Lousy Coding**: The community is excited about Meta's new **Llama 4 Scout and Maverick models**, which support native multimodal input and boast context windows of **10 million and 1 million tokens**, respectively, as detailed in [Meta's blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
   - Despite the excitement, the models were found to be very bad at coding tasks, tempering initial enthusiasm; although **Llama 4 Maverick** hit #2 overall on the Arena leaderboard ([tweet highlighting Llama 4 Maverick's performance](https://x.com/lmarena_ai/status/1908601011989782976)).
- **Agent Mode Edit Tool: Failing Frequently**: Users are experiencing problems with **Agent mode** failing to call the edit_tool, which results in **no code changes** being made even after the model processes the request.
   - One user pointed out that *the apply model is clearly cursor's bottleneck* and that it will *add changes, and deletes 500 lines of code next to it*.
- **Kubernetes: The Foundation for AGI?**: One visionary proposed using **Kubernetes** with docker containers, envisioning them as interconnected AGIs that can communicate with each other.
   - The user speculated that this setup could facilitate the rapid spread of ASI through zero-shot learning and ML, but did not elaborate.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Launches Comet Browser Early Access**: Perplexity has begun rolling out early access to **Comet**, their answer engine browser, to users on the [waitlist](https://www.perplexity.ai/comet).
   - Early users are asked not to publicly share details or features during the bug fix period and can submit feedback via the button in the top right.
- **Perplexity Discord Server Undergoes Revamp**: The Perplexity Discord server is being updated, featuring a **simplified channel layout**, a **unified feedback system**, and a **new #server-news channel**, scheduled for rollout on **October 7th, 2024**.
   - The updates are designed to streamline user navigation and improve moderator response times, the simplified channel layout is illustrated in [this image](https://cdn.discordapp.com/attachments/1047204950763122820/1358511016593326320/image.png?ex=67f56cfa&is=67f41b7a&hm=99677ce05c120d378ee85eb0947cad1e2e584998a7b3d0d373499b9185994738).
- **Gemini 2.5 Pro API Still in Preview Mode**: Perplexity confirmed that the **Gemini 2.5 Pro API** is not yet available for commercial use but currently in preview modes, and integration will proceed when allowed.
   - This follows user interest after [reports](https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/) that **Gemini 2.5 Pro** offers higher rate limits and lower costs than **Claude** and **GPT-4o**.
- **Llama 4 Drops With Massive Context Window**: The release of **Llama 4** models, featuring a 10 million token context window and **288 billion active parameters**, sparks excitement among users, with models like **Scout** and **Maverick**.
   - Members are particularly interested in evaluating **Llama 4 Behemoth's** recall capabilities, and you can follow up on this release at [Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- **API Parameters Unlock for All Tiers**: Perplexity **removed tier restrictions** for all API parameters such as search domain filtering and image support.
   - This change enhances API accessibility for all users, marking a substantial improvement in the API's utility.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 4o's Image Maker Grabs Attention**: Users found the **4o image maker** more attention-grabbing than **Veo 2**, and one user integrated **ChatGPT 4o images** with **Veo img2video**, achieving desired results.
   - The integrated result was described as *how I was hoping sora would be*.
- **Doubts Arise Over Llama 4 Benchmarks**: The community debated the value of **Llama 4's 10 million token context window** relative to models like **o1**, **o3-mini**, and **Gemini 2.5 Pro**.
   - Some claimed that *the benchmarks are fraud*, triggering debate over its true performance.
- **Content Loading Errors Plague Custom GPTs**: A user reported encountering a **'Content failed to load' error** when trying to edit their Custom GPT, after it had been working fine.
   - This issue prevented them from making changes to their custom configuration.
- **Moderation Endpoint's Role in Policy Enforcement**: Members discussed that while OpenAI's moderation endpoint isn't explicitly in the usage policy, it is **referenced** to prevent circumventing content restrictions on **harassment, hate, illicit activities, self-harm, sexual content, and violence**.
   - It was noted that the endpoint uses the same GPT classifiers as the **moderation API since 2022** suggesting an internal version runs on [chatgpt.com](https://chatgpt.com), project chats, and custom GPTs.
- **Fine Tune your TTRPG Prompts!**: Giving GPT a specific theme to riff off in prompting can lead to more creative and diverse city ideas, especially using **GPT 4o** and **4.5**.
   - For example, using a **"cosmic" theme** can yield different results compared to a **"domestic pet worship" theme**, improving the output without using the same creative options.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemini-like Local UI Still a Distant Dream?**: Members are seeking a local UI similar to **Gemini** that integrates chat, image analysis, and image generation, noting that current solutions like **LM Studio** and **ComfyUI** keep these functionalities separate.
   - One user suggested that [OpenWebUI](https://github.com/OpenGeniusAI/OpenWebUI) could potentially bridge this gap by connecting to **ComfyUI**.
- **LM Studio Commands Confuse Newbies**: A user asked whether **LM Studio** has a built-in terminal or if commands should be run in the OS command prompt within the **LM Studio** directory.
   - It was clarified that commands like *lms import* should be executed in the OS terminal (e.g., cmd on Windows), after which the shell might need reloading for **LMS** to be added to the **PATH**.
- **REST API Model Hot-Swapping Emerges for LM Studio**: A user inquired about programmatically loading/unloading models via **REST API** to dynamically adjust *max_context_length* for a Zed integration.
   - Another user confirmed this capability via command line using *lms load* and cited [LM Studio's documentation](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict), which requires **LM Studio 0.3.9 (b1)** and introduces time-to-live (TTL) for API models with auto-eviction.
- **Llama 4 Scout: Small But Mighty?**: With the release of **Llama 4**, users debated its multimodal and **MoE** (Mixture of Experts) architecture, with initial doubt about *llama.cpp* support.
   - Despite concerns about hardware, one user noted that [Llama 4 Scout](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) could potentially fit on a single **NVIDIA H100 GPU** with a **10M context window**, outperforming models like **Gemma 3** and **Mistral 3.1**.
- **Reka Flash 21B Blazes Past Gemma**: A user replaced **Gemma3 27B** with **Reka Flash 21B** and reported around **35-40 tps** at q6 on a 4090.
   - They noted that *mac ram bandwidth is not the bottleneck, it's gpu performance*, expressing satisfaction with **128GB M4 Maxes**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Tenstorrent's Hardware Heats Up the Market**: **Tenstorrent** hosted a dev day showcasing their **Blackhole PCIe boards**, featuring **RISC-V cores** and up to **32GB GDDR6** memory, designed for high performance **AI processing** and available for consumer purchase [here](https://tenstorrent.com/hardware/blackhole).
   - Despite enthusiasm, one member noted *they haven't published any benchmarks comparing their cards to competitors though so until then I cant really vouch*.
- **Llama 4 Models Make Multimodal Debut**: Meta introduced the **Llama 4** models, including **Llama 4 Scout** (**17B parameters**, **16 experts**, **10M context window**) and **Llama 4 Maverick** (**17B parameters**, **128 experts**), highlighting their multimodal capabilities and performance against other models [as per Meta's announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
   - Members noted the new license comes with several limitations, and no local model was released.
- **AI Agents Outperform Humans in Spear Phishing**: Hoxhunt's AI agents have surpassed human red teams in creating effective simulated phishing campaigns, marking a significant shift in social engineering effectiveness, with AI now 24% more effective than humans [as reported by hoxhunt.com](https://hoxhunt.com/blog/ai-powered-phishing-vs-humans).
   - This is a significant advancement in social engineering effectiveness, using AI phishing agents for defense.
- **AI Code Editor Tug-of-War**: For those new to AI code editors, **Cursor** is the most commonly recommended starting point, particularly for users coming from VSCode, with **Windsurf** and **Cline** also being good options.
   - Cursor is easy to start, has great tab-complete, whereas people are waiting for the new **token counts and context window details** feature in Cursor ([tweet](https://x.com/ryolu_/status/1907589821280956648)).
- **Context Management Concerns in Cursor**: Members are reporting Cursor's terrible context management issues, with a lack of visibility into what the editor is doing with the current context.
   - It may come down to a *skill issue* and the users are not meeting the tool in the middle.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 4 Debuts with Multimodal Brawn**: **Meta** launched the **Llama 4** family, featuring **Llama 4 Scout** (*17B* active params, *16* experts, *10M+* context) and **Llama 4 Maverick** (*17B* active params, *128* experts, *1M+* context), along with a preview of **Llama 4 Behemoth** and the iRoPE architecture for infinite context ([blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
   - Some members expressed skepticism about the benchmarking methodology and the real-world coding ability of **Llama 4 Scout**, referencing [Deedy's tweet](https://x.com/deedydas/status/1908749257084944847) indicating its poor coding performance.
- **Leaking Prompt Injection Tactics**: A member inquired about bypassing prompt guards and detectors from a *pentest* perspective, linking to a prompt filter trainer ([gandalf.lakera.ai/baseline](https://gandalf.lakera.ai/baseline)).
   - They also linked to a [Broken LLM Integration App](https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App) which uses **UUID tags and strict boundaries** to protect against injection attacks.
- **Claude Squad Manages Multiple Agents**: [Claude Squad](https://github.com/smtg-ai/claude-squad) is a free and open-source manager for **Claude Code & Aider tasks** that supervises multiple agents in one place with isolated git workspaces.
   - This setup enables users to run **ten Claude Codes in parallel**, according to [this tweet](https://x.com/moofeez/status/1907893901077196861?s=46).
- **Deepseek's RL Paper Rewards LLMs**: Deepseek released a new paper on **Reinforcement Learning (RL)** being widely adopted in post-training for **Large Language Models (LLMs)** at scale, available [here](https://arxiv.org/abs/2504.02495).
   - The paper proposes **Self-Principled Critique Tuning (SPCT)** to foster scalability and improve **reward modeling (RM)** with more inference compute for general queries.
- **Neural Graffiti Sprays Neuroplasticity**: A member introduced "Neural Graffiti", a technique to give pre-trained LLMs some neuroplasticity by splicing in a new neuron layer that recalls memory, reshaping token prediction at generation time, sharing code and demo on [Github](https://github.com/babycommando/neuralgraffiti).
   - The live modulation takes a fused memory vector (from prior prompts), evolves it through a recurrent layer (the Spray Layer), and injects it into the model’s output logic at generation time.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Streamable HTTP Transport Spec'd for MCP**: The [Model Context Protocol (MCP) specification](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http) now includes **Streamable HTTP** as a transport mechanism alongside *stdio*, using **JSON-RPC** for message encoding.
   - While clients *should* support *stdio*, the spec allows for custom transports, requiring newline delimiters for messages.
- **Llama 4 Ignorance of MCP Sparks Curiosity**: **Llama 4**, despite its impressive capabilities, still doesn't know what **MCP** is.
   - The model boasts **17B parameters** (**109B total**) and outperforms *deepseekv3*, according to [Meta's announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- **Cloudflare Simplifies Remote MCP Server Deployment**: It is now possible to [build and deploy remote **MCP servers** to **Cloudflare**](https://developers.cloudflare.com/agents/guides/remote-mcp-server/), with added support for **OAuth** through **workers-oauth-provider** and a built-in **McpAgent** class.
   - This simplifies the process of building remote **MCP servers** by handling authorization and other complex aspects.
- **Semgrep MCP Server Gets a Makeover**: The [Semgrep MCP server](https://github.com/semgrep/mcp), a tool for scanning code for security vulnerabilities, has been rewritten, with demos showcasing its use in **Cursor** and **Claude**.
   - It now uses **SSE** (Server-Sent Events) for communication, though the Python SDK may not fully support it yet.
- **WhatsApp Client Now Packs MCP Punch**: A user built **WhatsApp MCP client** and asked **Claude** to handle WhatsApp messages, answering 8 people in approx. **50 seconds**.
   - The bot *instantly detected the right language* (**English / Hungarian**), *used full convo context*, and sent appropriate messages including *❤️ to my wife, formal tone to the consul*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM Harness Gets RAG-Wrapped**: Members discussed wrapping **RAG outputs as completion tasks** and evaluating them locally using **llm-harness** with custom prompt and response files.
   - This approach uses **llm-harness** to evaluate **RAG** models, specifically by formatting the RAG outputs as completion tasks suitable for the harness.
- **Llama 4 Scout Sets 10M Context Milestone**: **Meta** released the **Llama 4** family, including **Llama 4 Scout**, which has a **17 billion** parameter model with **16 experts** and a **10M token context window** that outperforms **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1**, according to [this blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
   - The **10M context** is trained on a mix of publicly available data and information from Meta’s products including posts from **Instagram**, **Facebook**, and people’s interactions with **Meta AI**.
- **NoProp Forges Gradient-Free Frontier**: A new learning method named **NoProp** learns to denoise a noisy target at each layer independently without relying on either forward or backward propagation and takes inspiration from diffusion and flow matching methods, described in [this paper](https://arxiv.org/abs/2503.24322).
   - There's a [GitHub implementation by lucidrains](https://github.com/lucidrains/hyper-connections); however, there's a discussion that *the pseudocode at the end of the paper says they're effecting the actual updates using gradient based methods*.
- **Attention Sinks Stave Off Over-Mixing**: A recent paper argues that **attention sinks**, where LLMs attend heavily to the first token in the sequence, is a mechanism that enables LLMs to avoid over-mixing, detailed in [this paper](https://arxiv.org/abs/2504.02732).
   - An earlier paper ([https://arxiv.org/abs/2502.00919](https://arxiv.org/abs/2502.00919)) showed that *attention sinks utilize outlier features to catch a sequence of tokens, tag the captured tokens by applying a common perturbation, and then release the tokens back into the residual stream, where the tagged tokens are eventually retrieved*.
- **ReLU Networks Carve Hyperplane Heavens**: Members discussed a geometrical approach to neural networks, advocating for the **polytope lens** as the right perspective on neural networks, linking to a [previous post](https://addxorrol.blogspot.com/2024/07/some-experiments-to-help-me-understand.html) on the *"origami view of NNs".*
   - It was posited that neural nets, especially **ReLUs**, have an implicit bias against overfitting due to carving the input space along hyperplanes, which becomes more effective in higher dimensions.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face's Hub Gets a Facelift**: The [huggingface_hub v0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) release introduces a next-gen **Git LFS alternative** and new **inference providers**.
   - This release is the *biggest update in two years!*
- **Reranking with monoELECTRA Transformers**: **monoELECTRA-{base, large} reranker models** from @fschlatt1 & the research network Webis Group are now available in [Sentence Transformers](https://x.com/tomaarsen/status/1906652865675862125).
   - These models were distilled from **LLMs** like **RankZephyr** and **RankGPT4**, as described in the **Rank-DistiLLM paper**.
- **YourBench Instantly Builds Custom Evals**: **YourBench** allows users to build **custom evals** using their **private docs** to assess fine-tuned models on unique tasks ([announcement](https://x.com/nathanhabib1011/status/1907728631167902067)).
   - The tool is *game-changing for LLM evaluation*.
- **AI Engineer Interview Code Snippet**: A community member asks about what the code portion of an **AI engineer interview** looks like, and another member pointed to the **scikit-learn library**.
   - There was no follow up to the discussion.
- **Community Debates LLM Fine-Tuning**: When a member inquired about fine tuning quantized models, members pointed to **QLoRA**, **Unsloth**, and **bitsandbytes** as potential solutions, with [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-guide) shared.
   - Another stated that you can only do so using **LoRA**, and stated that *GGUF is an inference-optimized format, not designed for training workflows*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Raw Binary AI Outputs File Formats**: Members debated training AI on **raw binary data** to directly output file formats like **mp3** or **wav**, stating that this approach builds on discrete mathematics like **Turing machines**.
   - Counterarguments arose questioning the Turing-completeness of current AI models, but proponents clarified that AI doesn't need to be fully Turing-complete to output appropriate tokens as responses.
- **Llama 4 Scout Boasts 10M Context Window**: **Llama 4 Scout** boasts **10 million context window**, **17B active parameters**, and **109B total parameters**, outperforming models like **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1**, according to [llama.com](https://www.llama.com/llama4/).
   - Community members expressed skepticism about the **10M context window** claim, with additional details in the [Llama 4 documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/) and [Meta's blogpost on Llama 4's Multimodal Intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- **DeepSeek Proposes SPCT Reward System**: **Self-Principled Critique Tuning (SPCT)** from DeepSeek is a new reward-model system where an **LLM** prompted with automatically developed principles of reasoning generates **critiques** of CoT output based on those principles, further explained in [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495).
   - This system aims to train models to develop reasoning principles automatically and assess their own outputs in a more **system 2** manner, instead of with human hand-crafted rewards.
- **PaperBench Tests Paper Reproduction**: **OpenAI's PaperBench benchmark** tests AI agents' ability to replicate cutting-edge machine learning research papers from scratch, as described in [this article](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052).
   - The benchmark evaluates agents on reproducing entire **ML papers** from **ICML 2024**, with automatic grading using **LLM judges** and fine-grained rubrics co-designed with the original authors.
- **Diffusion Steers Auto-Regressive LMs**: Members discussed using a guided diffusion model to steer an auto-regressive language model to generate text with desired properties, based on [this paper](https://arxiv.org/abs/2408.04220).
   - A talk by the main author ([https://www.youtube.com/watch?v=klW65MWJ1PY](https://www.youtube.com/watch?v=klW65MWJ1PY)) explains how *diffusion modeling can control LLMs*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Python Debuts Unifying Ecosystem**: Nvidia released the [CUDA Python package](https://developer.nvidia.com/cuda-python), offering **Cython/Python wrappers** for CUDA driver and runtime APIs, installable via PIP and Conda, aiming to unify the **Python CUDA ecosystem**.
   - It intends to provide full coverage and access to the **CUDA host APIs** from Python, mainly benefiting library developers needing to interface with C++ APIs.
- **Bydance Unleashes Triton-distributed**: ByteDance-Seed releases **Triton-distributed** ([github here](https://github.com/ByteDance-Seed/Triton-distributed)) designed to extend the usability of **Triton language** for parallel systems development.
   - This release enables parallel systems development by leveraging the **Triton language**.
- **Llama 4 Scout Boasts 10M Context Window**: Meta introduces **Llama 4**, boasting enhanced personalized multimodal experiences and featuring **Llama 4 Scout**, a **17 billion** parameter model with **16 experts** ([blog post here](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
   - It claims to outperform **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1**, fitting on a single **NVIDIA H100 GPU** with an industry-leading context window of **10M**.
- **L40 Faces Underperformance Puzzle**: Despite the **L40** theoretically being better for **4-bit quantized Llama 3 70b**, it achieves only **30-35 tok/s** on single-user requests via vLLM, underperforming compared to online benchmarks of the A100.
   - The performance gap may be due to the **A100's superior DRAM bandwidth and tensor ops performance**, which are nearly twice as fast as the L40.
- **Vector Sum Kernel achieves SOTA**: A member shared a [blogpost](https://veitner.bearblog.dev/making-vector-sum-really-fast/) and [code](https://github.com/simveit/effective_reduction) on achieving SOTA performance for summing a vector in CUDA, reaching **97.94%** of theoretical bandwidth, outperforming NVIDIA's **CUB**.
   - However, another member pointed out a potential race condition due to implicit warp-synchronous programming, recommending the use of `__warp_sync()` for correctness, with reference to [Independent Thread Scheduling (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Voice Mode Sparks Innovation**: Users found the **interactive voice mode** inspired new ideas and enabled tailoring **NotebookLM** for corporate needs.
   - One user confidently stated they could now make almost any text work and customize notebooks for specific corporate needs after solidifying the **NotebookLM** foundation since January.
- **Mind Map Feature Finally Live**: The **mind maps feature** has been fully rolled out, appearing in the middle panel for some users.
   - One user reported seeing it briefly on the right side panel before it disappeared, indicating a phased rollout.
- **Users Theorize Image-Based Mind Map Revolution**: Users discussed how **generative AI** tools could evolve mind maps to include images, drawing inspiration from **Tony Buzan's** original mind maps.
   - Members expressed excitement about the potential for more visually rich and informative mind mapping.
- **Discover Feature Rollout Frustrates Users**: Users expressed frustration over the delayed rollout of the new **'Discover Sources'** feature in NotebookLM, announced April 1st.
   - The feature aims to streamline learning and database building, allowing users to create notebooks directly within NotebookLM, but the rollout is expected to take up to two weeks.
- **AI Chrome Extension tunes YouTube audio**: An **AI-powered Chrome Extension** called *EQ for YouTube* allows users to manipulate the audio of YouTube videos in real-time with a 6-band parametric equalizer; the [GitHub repo](https://github.com/aashishjhaa/eq-for-youtube) is available for download.
   - The extension features real-time frequency visualization, built-in presets, and custom preset creation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nvidia Adds Native Python Support to CUDA**: **Nvidia** is adding native **Python** support to **CUDA** using the **CuTile** programming model, as detailed in [this article](https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/).
   - The community questions whether this move abstracts away too much from thread-level programming, diminishing the control over **GPU code**.
- **Debate Erupts over Mojo's Language Spec**: Discussion revolves around whether **Mojo** should adopt a formal language spec, balancing the need for responsibility and maturity against the potential for slowing down development.
   - Referencing the design principles of **Carbon**, some argue that a spec is crucial, while others claim that **Mojo's** tight integration with **MAX** and its needs makes a spec impractical, pointing to **OpenCL's** failures due to design by committee.
- **Mojo's Implicit Copies Clarified**: A member inquired about the mechanics of **Mojo's** implicit copies, specifically regarding Copy-on-Write (CoW).
   - The response clarified that the *semantics wise, [Mojo] always copy; optimisation wise, many are turned into move or eliminated entirely (inplace)*, with optimizations happening at compile time rather than runtime like CoW.
- **Tenstorrent Eyes Modular's Software**: A member proposed that **Tenstorrent** adopt **Modular's** software stack, sparking debate about the ease of targeting **Tenstorrent's** architecture.
   - Despite the potential benefits, some noted that **Tenstorrent's** driver is user-friendly, making it relatively trivial to get code running on their hardware.
- **ChatGPT's Mojo Abilities Criticized**: Members are questioning the ability of **ChatGPT** and other **LLMs** to rewrite Python projects into **Mojo**.
   - Members indicated that *ChatGPT isn't good at any new languages*.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embed Text V2 Integrates with Llama.cpp**: **Llama.cpp** is integrating **Nomic Embed Text V2** with Mixture-of-Experts (MoE) architecture for multilingual embeddings, as detailed in [this GitHub Pull Request](https://github.com/ggml-org/llama.cpp/pull/12466).
   - The community awaits multimodal support like **Mistral Small 3.1** to come to **Llama.cpp**.
- **GPT4All's radio silence rattles restless readers**: Core developers of **GPT4All** have gone silent, causing *uncertainty* within the community about contributing to the project.
   - Despite this *silence*, one member noted that *when they break their silence, they usually come out swinging*.
- **Llama 4 Arrives, Falls Flat?**: Meta launched **Llama 4** on April 5, 2025 ([announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)), introducing **Llama 4 Scout**, a 17B parameter model with 16 experts and a **10M token** context window.
   - Despite the launch, opinions were mixed with some saying that *it is a bit of a letdown*, and some calling for **DeepSeek** and **Qwen** to step up their game.
- **ComfyUI powers past pretty pictures**: **ComfyUI**'s extensive capabilities were discussed, emphasizing its ability to handle tasks beyond image generation, such as image and audio captioning.
   - Members mentioned the potential for video processing and command-line tools for visual model analysis.
- **Semantic Chunking Server Recipe for RAG**: A member shared a [link to a semantic chunking server](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html) implemented with FastAPI for better **RAG** performance.
   - They also posted a [curl command example](https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html) demonstrating how to post to the chunking endpoint, including setting parameters like `max_tokens` and `overlap`.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MCP Servers Get Command Line Access**: A new tool by @MarcusSchiesser allows users to **discover, install, configure, and remove MCP servers** like Claude, @cursor_ai, and @windsurf_ai via a single CLI as shown [here](https://t.co/zmqxBwNKvQ).
   - It simplifies managing numerous MCP servers, streamlining the process of setting up and maintaining these servers.
- **Llama Jumps into Full-Stack Web Apps**: The **create-llama CLI tool** quickly spins up a web application with a **FastAPI backend and Next.js frontend** in just five source files, available [here](https://t.co/TuZ0O0nMfe).
   - It supports quick agent application development, specifically for tasks like deep research.
- **LlamaParse's Layout Agent Intelligently Extracts Info**: The new **layout agent within LlamaParse** enhances document parsing and extraction with precise visual citations, leveraging SOTA VLM models to dynamically detect blocks on a page, shown [here](https://t.co/2WRRXxIRa1).
   - It offers improved document understanding and adaptation, ensuring more accurate data extraction.
- **FunctionTool Wraps Workflows Neatly**: The `FunctionTool` can transform a **Workflow** into a **Tool**, providing control over its name, description, input annotations, and return values.
   - A code snippet was shared on how to implement this wrapping.
- **Agents Do Handoffs Instead of Supervision**: For multi-agent systems, agent handoffs are more reliable than the supervisor pattern, which can be prone to errors, see [this GitHub repo](https://github.com/run-llama/multi-agent-concierge).
   - This shift promotes better system stability and reduces the risk of central point failures.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygraph: Torch-geometric Port Possible?**: A member proposed creating a module similar to **torch-geometric** for graph ML within tinygrad, noting tinygrad's existing torch interface.
   - The core question was whether such a module would be considered *"useful"* to the community.
- **Llama 4's 10M Context: Virtual?**: A user shared [a tweet](https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA) claiming **Llama 4's** declared **10M context** is *"virtual"* because models weren't trained on prompts longer than **256k tokens**.
   - The tweeter further asserted that even problems below **256k tokens** might suffer from low-quality output due to the scarcity of high-quality training examples and that the largest model with **2T parameters** *"doesn't beat SOTA reasoning models"*.
- **Fast Pattern Matcher Bounty: $2000 Up For Grabs**: A member advertised an open [$2000 bounty](https://github.com/tinygrad/tinygrad/pull/9737) for a fast pattern matcher in tinygrad.
   - The proposed solution involves a **JIT** for the match function, aimed at eliminating function calls and dict copies.
- **Debate About Tensor's Traits Arises**: A discussion unfolded concerning whether **Tensor** should inherit from `SimpleMathTrait`, considering it re-implements every method without utilizing the `.alu()` function.
   - A previous bounty for refactoring **Tensor** to inherit from `MathTrait` was canceled due to subpar submissions, leading some to believe **Tensor** might not need to inherit from either.
- **Colab CUDA Bug Ruins Tutorial**: A user encountered issues while running code from the mesozoic tinygrad tutorials in Colab, later identified as a Colab bug related to incompatible CUDA and driver versions.
   - The temporary workaround involved using the CPU device while members found a long term solution involving specific `apt` commands to remove and install compatible CUDA and driver versions.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **MCP plays well with Command-A**: A member suggested using **MCP (Modular Conversational Platform)** with the **Command-A model** should work via the **OpenAI SDK**.
   - Another member concurred, noting that *there is no reason why it should not work*.
- **Cohere Tool Use detailed**: A member called out [Cohere Tool Use Overview](https://docs.cohere.com/docs/tool-use-overview), highlighting its ability to connect **Command family models** to external tools like **search engines, APIs, and databases**.
   - The documentation mentions that **Command-A** supports tool use, similar to what **MCP** aims to achieve.
- **Aya Vision AMA**: The core team behind **Aya Vision**, a multilingual multimodal open weights model, is hosting tech talks followed by an AMA on <t:1744383600:F> to allow the community to directly engage with the creators; further details are available at [Discord Event](https://discord.gg/sH3SSRp2?event=1358866070315860068).
   - Attendees can join for exclusive insights on how the team built their first multimodal model and the lessons learned, with the event hosted by Sr. Research Scientist <@787403823982313533> and lightning talks from core research and engineering team members.
- **Slack App Needs Vector DB for Notion**: A member asked for help with a working solution for a **Slack app** integration with a company **Notion wiki database** in the `api-discussions` channel.
   - Another member suggested using a **vector DB** due to **Notion's** subpar search API but no specific recommendations were given.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Patches Timeout Crash**: A member resolved a **timeout crash** issue, introducing `torchtune.utils._tensor_utils.py` with a wrapper around `torch.split` in [this pull request](https://github.com/pytorch/torchtune/pull/2560).
   - The suggestion was made to merge the tensor utilities separately before syncing with another branch to resolve potential conflicts.
- **NeMo Explores Resilient Training Methods**: A member attended a **NeMo** session on resilient training, which highlighted features like **fault tolerance**, **straggler detection**, and **asynchronous checkpointing**.
   - The session also covered **preemption**, **in-process restart**, **silent data corruption detection**, and **local checkpointing**, though not all features are currently implemented; the member offered to compare **torchtune** vs. **NeMo** in resiliency.
- **Debate ensues over RL Workflow**: A discussion arose regarding the complexities of **RL workflows**, data formats, and prompt templates, proposing a separation of concerns for decoupling data conversion and prompt creation.
   - The suggestion was to factorize data conversion into a standard format and then convert this format into an actual string with the prompt, to allow template reuse across datasets.
- **DeepSpeed to boost Torchtune?**: A member proposed integrating **DeepSpeed** as a backend into **torchtune** and created [an issue](https://github.com/pytorch/torchtune/issues/2569) to discuss its feasibility.
   - Concerns were raised about redundancy with **FSDP**, which already supports all sharding options available in **DeepSpeed**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Yang Presents Autoformalization Theorem Proving**: Kaiyu Yang presented on *Language models for autoformalization and theorem proving* [today at 4pm PDT](https://www.youtube.com/live/cLhWEyMQ4mQ), covering the use of **LLMs** for formal mathematical reasoning.
   - The presentation focuses on **theorem proving** and **autoformalization** grounded in formal systems such as **proof assistants**, which verify correctness of reasoning and provide automatic feedback.
- **AI4Math deemed crucial for system design**: **AI for Mathematics (AI4Math)** is crucial for AI-driven system design and verification.
   - Extensive efforts have mirrored techniques in NLP.
- **Member shares link to LLM Agents MOOC**: A member asked for a link to the **LLM Agents MOOC**, and another member shared [the link](https://llmagents-learning.org/sp25).
   - The linked course is called *Advanced Large Language Model Agents MOOC*.
- **Sign-ups Open for AgentX Competition**: Staff shared that sign-ups for the **AgentX Competition** are available [here](https://rdi.berkeley.edu/agentx/).
   - No additional information was provided about the competition.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Asyncio support coming to dspy?**: A member inquired about adding **asyncio support** for general dspy calls, especially as they transition from *litelm* to dspy optimization.
   - The user expressed interest in native dspy **async** capabilities.
- **Async DSPy Fork Faces Abandonment**: A member maintaining a [full-async fork of dspy](https://github.com/swiftdevil/dspy/tree/full_async) is migrating away but open to merging upstream changes if community expresses interest.
   - The fork has been maintained for a few months but might be abandoned without community support.
- **User Seeks Greener Pastures, Migrates from DSPy**: Members inquired about the reasons for migrating away from dspy and the alternative tool being adopted.
   - A member also sought clarification on the advantages of a **full async DSPy** and suggested merging relevant features into the main repository.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **GitHub PR Gets the Once-Over**: A member reviewed a [GitHub Pull Request](https://github.com), providing feedback for further discussion.
   - The author of the PR thanked the reviewer and indicated that a rerun might be necessary based on the received comments.
- **Phi-4 Family Gets the Nod**: A member is exploring extending functionality to **Phi-4-mini** and **Phi-4** models.
   - This expansion aims to enhance the tool's compatibility, even if these models are not officially supported.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research Calls for Community**: **Manifold Research Group** is hosting **Community Research Call #4** this Saturday (4/12 @ 9 AM PST), covering their latest work in **Multimodal AI, self-assembling space robotics, and robotic metacognition**.
   - Interested parties can register [here](https://lu.ma/wlne416w) to join the open, collaborative, and frontier science focused event.
- **CRCs are Manifold's Cornerstone**: **Community Research Calls (CRCs)** are **Manifold's** cornerstone events where they present significant advancements across their research portfolio.
   - These interactive sessions provide comprehensive updates on ongoing initiatives, introduce new research directions, and highlight opportunities for collaboration.
- **CRC #4 Agenda is Live**: The agenda for **CRC #4** includes updates on **Generalist Multimodality Research**, **Space Robotics Advancements**, **Metacognition Research Progress**, and **Emerging Research Directions**.
   - The event will cover recent breakthroughs and technical progress in their **MultiNet framework**, developments in **Self-Assembling Swarm technologies**, updates on **VLM Calibration methodologies**, and the introduction of a novel robotic metacognition initiative.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357792286464934191)** (1150 messages🔥🔥🔥): 

> `Making ai sound human, Riveroaks eval, NightWhisper model, GPT-4.5 vs quasar` 


- **Crafting Human-Like AI Responses is Tricky**: Members are sharing **system prompts** and strategies to make AI sound more human, noting that increasing the **temperature** can lead to nonsensical outputs unless the top-p parameter is adjusted carefully.
   - One user suggested using prompts like *'You are the brain-upload of a human person, who does their best to retain their humanity. Your most important priority is: to sound like an actual living human being.'*
- **Benchmarking Riveroaks LLM**: A member shared a coding benchmark where **Riveroaks** scored second only to **Claude 3.7 Sonnet Thinking**, outperforming **Gemini 2.5 Pro** and **GPT-4o** in a platform game creation task.
   - The evaluation involved rating models on **eight different aspects** and subtracting points for **bugs** with [full results here](link.to.results).
- **NightWhisper Hype and Theories on its Removal**: Users expressed disappointment over the removal of the **NightWhisper** model, praising its coding abilities and general performance, and speculating whether it was an experiment or a precursor to a full release.
   - Theories ranged from Google gathering necessary data to preparing for the release of a new **Qwen** model. and that it will come out during **Google Cloud Next**.
- **Quasar vs GPT-4o**: Members compared **Quasar Alpha** to **GPT-4o**, with some suggesting Quasar is a free, streamlined version of GPT-4o. It was also revealed in a recent tweet that [Quasar was measured to be ~67% GPQA diamond](https://link.to/gpqa).
   - Analysis revealed Quasar has a similar GPQA diamond score to March's GPT-4o. [Image.png from discord](https://cdn.discordapp.com/attachments/1340554757827461211/1358604050266062908/image.png?ex=67f51adf&is=67f3c95f&hm=eef654608f530e6e624c049f6ad26a0fc65a97df3dd4abd86fbd45df158f0e43&)
- **Gemini 2.5 is a Game Changer for Creative Coding**: Members praised **Gemini 2.5 Pro** for its coding capabilities and general performance as it made it easier to build a functioning Pokemon Game, prompting one user to code an iteration script that loops through various models.
   - A user who claimed to have gotten **3D animations working** said that the style was a bit old and that a separate model said *the generated code is cut off*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/armenagha/status/1734321205770101062?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: My bet is everyone is doing this. Mistral is not that much better of a model than LLaMa. I bet they included some benchmark data during the last 10% of training to make &#34;zero-shot&#34; numbers loo...</li><li><a href="https://x.com/suchenzang/status/1909070231517143509?s=46">Tweet from Susan Zhang (@suchenzang)</a>: &gt; Company leadership suggested blending test sets from various benchmarks during the post-training processIf this is actually true for Llama-4, I hope they remember to cite previous work from FAIR ...</li><li><a href="https://x.com/vitrupo/status/1908763535351669017">Tweet from vitrupo (@vitrupo)</a>: Anthropic Chief Scientist Jared Kaplan says Claude 4 will arrive &#34;in the next six months or so.&#34;AI cycles are compressing — &#34;faster than the hardware cycle&#34; — even as new chips arrive....</li><li><a href="https://x.com/geminiapp/status/1909215393186472380?s=46">Tweet from Google Gemini App (@GeminiApp)</a>: 📣 It’s here: ask Gemini about anything you see. Share your screen or camera in Gemini Live to brainstorm, troubleshoot, and more.Rolling out to Pixel 9 and Samsung Galaxy S25 devices today and availa...</li><li><a href="https://www.twitch.tv/gemini_plays_pokemon">Gemini_Plays_Pokemon - Twitch</a>: Gemini Plays Pokemon (early prototype) - Hello Darkness, My Old Friend</li><li><a href="https://x.com/vibagor44145276/status/1909138204672053625">Tweet from vibagor441 (@vibagor44145276)</a>: The linked post is not true. There are indeed issues with Llama 4, from both the partner side (inference partners barely had time to prep. We sent out a few transformers wheels/vllm wheels mere days b...</li><li><a href="https://x.com/Google/status/1907880784557412825">Tweet from Google (@Google)</a>: Join us in Las Vegas and online for #GoogleCloudNext on April 9-11!Register for a complimentary digital pass → https://goo.gle/CloudNext25 and then sign up to watch the livestream right here ↓ https:/...</li><li><a href="https://x.com/bdsqlsz/status/1909274256602771520">Tweet from 青龍聖者 (@bdsqlsz)</a>: Why was the mystery revealed that llama4 released on the weekend solved...Because qwen3 is about to be released.8B standard and MoE-15B-A2B</li><li><a href="https://x.com/armenagha/status/1859646650714821012?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Say hello to our new company Perceptron AI. Foundation models transformed the digital realm, now it’s time for the physical world. We’re building the first foundational models designed for real-time, ...</li><li><a href="https://x.com/DrealR_/status/1908530950025134565">Tweet from DrealR (@DrealR_)</a>: Gave the same prompt to Quasar Alpha:</li><li><a href="https://x.com/vibagor44145276/status/1909138204672053625?t=P0lbZfL7J8u1O6-AQjLqyg&s=19">Tweet from vibagor441 (@vibagor44145276)</a>: The linked post is not true. There are indeed issues with Llama 4, from both the partner side (inference partners barely had time to prep. We sent out a few transformers wheels/vllm wheels mere days b...</li><li><a href="https://copilot.microsoft.com/wham">Microsoft Copilot: Your AI companion</a>: Microsoft Copilot is your companion to inform, entertain, and inspire. Get advice, feedback, and straightforward answers. Try Copilot now.</li><li><a href="https://x.com/algo_diver/status/1909257761013322112?t=Ba4GsMkDmy-v38rJPf9ybA&s=19">Tweet from chansung (@algo_diver)</a>: Multi Agentic System Simulator built w/ @GoogleDeepMind Gemini 2.5 Pro Canvas. Absolutely stunning to watch how multi-agents are making progress towards the goal achievement!Maybe next step would be a...</li><li><a href="https://x.com/gurgavin/status/1909159289140269069">Tweet from GURGAVIN (@gurgavin)</a>: ALIBABA SHARES JUST CLOSED TRADING IN HONGKONG DOWN 19%MAKING TODAY THE WORST DAY EVER IN ALIBABA’S HISTORY</li><li><a href="https://x.com/algo_diver/status/1909257761013322112?t=Ba4GsMkDm">Tweet from chansung (@algo_diver)</a>: Multi Agentic System Simulator built w/ @GoogleDeepMind Gemini 2.5 Pro Canvas. Absolutely stunning to watch how multi-agents are making progress towards the goal achievement!Maybe next step would be a...</li><li><a href="https://x.com/DrealR_/status/1907921770184860082">Tweet from DrealR (@DrealR_)</a>: NightWhisper vs Gemini 2.5 Pokemon sim:Gemini 2.5:</li><li><a href="https://liveweave.com/bdNibz">HTML, CSS and JavaScript playground - Liveweave</a>: no description found</li><li><a href="https://gist.github.com/riidefi/3340cc2b33b9edf5f03dc4429ba635d0">LMArena&#39;s `venom` System Prompt</a>: LMArena&#39;s `venom` System Prompt. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://chromewebstore.google.com/detail/MyLMArena/dcmbcmdhllblkndablelimnifmbpimae">MyLMArena - Chrome Web Store</a>: Track your personal LLM preferences using ELO ratings with MyLMArena.</li><li><a href="https://www.youtube.com/watch?v=z46KBYbcpmo">ERNIE 4.5 + X1: Most POWERFUL and CHEAPEST LLM Beats GPT-4.5, R1, &amp; Sonnet 3.7! (Fully Tested)</a>: Baidu is making waves in the AI world with ERNIE 4.5 and ERNIE X1, challenging industry giants like OpenAI and DeepSeek. ERNIE 4.5 is a natively multimodal m...</li><li><a href="https://justpaste.it/huz8w">JustPaste.it - Share Text &amp; Images the Easy Way</a>: no description found</li><li><a href="https://justpaste.it/j3v7a">tetet</a>: no description found</li><li><a href="https://archive.ph/bGeWH">How Hallucinatory A.I. Helps Science Dream Up Big Breakthroughs - The&#x2026;</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>: Adding Qwen3This PR adds the support of codes for the coming Qwen3 models. For information about Qwen, please visit https://github.com/QwenLM/Qwen2.5. @ArthurZucker
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357792500089098250)** (1294 messages🔥🔥🔥): 

> `Qwen 2.5, FSDP isn't working, multi-GPU, Llama 4` 


- ****Qwen 2.5** is the latest series of **Qwen** large language models**: **Qwen2.5** models range from **0.5 to 72 billion parameters**, with improved capabilities in coding, math, instruction following, long text generation (**over 8K tokens**), and multilingual support (**29+ languages**), as detailed in the [Hugging Face introduction](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
   - These models offer **long-context support up to 128K tokens** and improved resilience to system prompts.
- ****FSDP** isn't working but **Multi-GPU** can save the day**: Members discussed issues with **FSDP** not working, with one member suggesting to *get your search foo up* and look for **multi-GPU** setups instead of accelerate, offering debugging assistance.
   - A user provided [their pip freeze output](https://example.com/pip-freeze-output) showing the exact versions of **unsloth** and **unsloth_zoo** being used for GRPO, after being prompted to share it.
- ****Meta** releases **Llama 4 Scout & Maverick**, 17B active parameters, 10M ctx**: **Llama 4 Scout (17B)** has **16 MoE experts** and **10 million** context window whereas **Llama 4 Maverick (17B)** has **128 experts** and comparable results to DeepSeek v3 on reasoning and coding, per [Meta's official announcement](https://x.com/AIatMeta/status/1908598456144531660).
   - The community discussed the practicality and hardware requirements, and the need for a key to get access.
- **Unsloth releases **Llama 4 Scout** and **4-bit** model for fine-tuning**: Unsloth announced they uploaded [Llama 4 Scout and a 4-bit version](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct) for fine-tuning, emphasizing that **Llama 4 Scout (17B, 16 experts) beats all Llama 3 models** with a 10M context window, as noted in their [blog post](https://unsloth.ai/blog/llama4).
   - It was emphasized that the model is only meant to be used on Unsloth - and is currently being uploaded so people should wait.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1908598456144531660">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://huggingface.co/lmsys/DeepSeek-V3-NextN">lmsys/DeepSeek-V3-NextN · Hugging Face</a>: no description found</li><li><a href="https://www.together.ai/blog/specexec">SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Letter_and_spirit_of_the_law#Gaming_the_system>)">Letter and spirit of the law - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct">Qwen/Qwen2.5-3B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/oh-my-omg-so-hot-gif-19803505">Oh My Omg GIF - Oh My Omg So Hot - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct">unsloth/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/llama4">Llama 4 - Finetune &amp; Run  with Unsloth</a>: Meta&#x27;s new Llama 4 multimodal models: Scout and Maverick.Fine-tune &amp; Run them with Unsloth!</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit">unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.hume.ai/">Home • Hume AI</a>: Empathic AI research lab building multimodal AI with emotional intelligence.</li><li><a href="https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164">Llama 4 - a meta-llama Collection</a>: no description found</li><li><a href="https://tenor.com/view/hulk-hogan-nodding-nod-yes-yup-gif-13973219">Hulk Hogan Nodding GIF - Hulk Hogan Nodding Nod - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-4.-understand-model-parameters">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://github.com/ggml-org/llama.cpp/blob/master/examples/imatrix/README.md">llama.cpp/examples/imatrix/README.md at master · ggml-org/llama.cpp</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2302">[BUG] CUDA out of memory during Llama-4-Scout loading on H200 · Issue #2302 · unslothai/unsloth</a>: max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally! dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+ load_in_4bit = True # Use 4bit ...</li><li><a href="https://www.llama.com/">Llama</a>: The open-source AI models you can fine-tune, distill and deploy anywhere. Choose from our collection of models: Llama 4 Maverick and Llama 4 Scout.</li><li><a href="https://huggingface.co/turboderp">turboderp (turboderp)</a>: no description found</li><li><a href="https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md">llguidance/docs/fast_forward.md at main · guidance-ai/llguidance</a>: Super-fast Structured Outputs. Contribute to guidance-ai/llguidance development by creating an account on GitHub.</li><li><a href="https://github.com/turboderp-org/exllamav3">GitHub - turboderp-org/exllamav3: An optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs</a>: An optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs  - GitHub - turboderp-org/exllamav3: An optimized quantization and inference library for runni...</li><li><a href="https://github.com/unslothai/unsloth/issues">unslothai/unsloth</a>: Finetune Llama 4, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/facebookresearch/audiobox-aesthetics">GitHub - facebookresearch/audiobox-aesthetics: Unified automatic quality assessment for speech, music, and sound.</a>: Unified automatic quality assessment for speech, music, and sound. - facebookresearch/audiobox-aesthetics</li><li><a href="https://github.com/u">U RIP 2011-2014</a>: U RIP 2011-2014 has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f3xfnk/local_1m_context_inference_at_15_tokenss_and_100/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/2308">[BUG] Colab notebook Llama 3.1 (8B) is broken · Issue #2308 · unslothai/unsloth</a>: When running the cell below: from trl import SFTTrainer from transformers import TrainingArguments from unsloth import is_bfloat16_supported trainer = SFTTrainer( model = model, tokenizer = tokeniz...</li><li><a href="https://github.com/guidance-ai/llguidance?tab=readme-ov-file">GitHub - guidance-ai/llguidance: Super-fast Structured Outputs</a>: Super-fast Structured Outputs. Contribute to guidance-ai/llguidance development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/2289#event-17139510983">[BUG] Collab notebooks throw error: &#39;int&#39; object has no attribute &#39;mask_token&#39; · Issue #2289 · unslothai/unsloth</a>: Describe the bug Collab notebooks stopped working. I was finetuning my model about 12 hours ago and it was fine. But now none of the collab notebooks work, they throw the same error. I&#39;ve tried mi...</li><li><a href="https://github.com/unslothai/notebooks/">GitHub - unslothai/notebooks: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more.</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/pull/28">Fixed colab installation by rupaut98 · Pull Request #28 · unslothai/notebooks</a>: This request addresses 2289. With the previous installation%%captureimport osif &amp;quot;COLAB_&amp;quot; not in &amp;quot;&amp;quot;.join(os.environ.keys()):    !pip install unslothelse:    # Do thi...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jscel9/can_i_run_llama_4_scout_on_a_single_rtx_4060_8gb/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/mit-han-lab/omniserve">GitHub - mit-han-lab/omniserve: [MLSys&#39;25] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving; [MLSys&#39;25] LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention</a>: [MLSys&amp;#39;25] QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving; [MLSys&amp;#39;25] LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention - mit-han-l...</li><li><a href="https://github.com/ml-explore/mlx-lm/pull/74">Llama4 text only by awni · Pull Request #74 · ml-explore/mlx-lm</a>: Text only. Tested scout and maverick and it works well. Note when converting only the LM weights are kept so it&amp;#39;s good to indicate that in the repo name:mlx_lm.convert --hf-path meta-llama/Lla...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/47dJUfK4lZ">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jshwxe/first_results_are_in_llama_4_maverick_17b_active/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357865868440109096)** (11 messages🔥): 

> `ChatGPT DDoS program, LLM Guideline Triggers, Dataset Substitution` 


- **ChatGPT Offers DDoS Assistance**: A member reported that **ChatGPT** offered to write a **DDoS program** after being asked about sending malformed packets over Ethernet, even providing a 😈 emoji.
   - The member suggested that *"somehow sometimes uncensored parts of it is being invoked if you send the right token to the neural network."
- **LLM Offers Guideline Trigger Tips**: A member stated that an LLM offered to assist with **avoiding guideline triggers** and limitations in prompts to other LLMs.
   - They quoted the LLM as saying, *"here's how you avoid a refusal. You aren't lying, you just aren't telling the full details"*.
- **Dataset Substitution Plans**: A member shared a code snippet for **dataset substitution**, planning to train a model with specific model information.
   - The member plans to set the model name to **'Speaker Mini'**, the base model to **'Microsoft Phi-4-mini'**, the parameter size to **'3.8B'**, and the maker to **'Overta'**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357799862594441227)** (770 messages🔥🔥🔥): 

> `Lora merging script usage, Dataset sample size, Quantization, Inference speed` 


- **Users resolve issues with Lora by merging weights before inference**: A user, after experiencing a finetuned model behaving like the base model, discovered that they needed to **merge the LoRA weights with the base model before running inference** ([script](https://discord.com/channels/1179035537009545276/1358222086316752918/1358297905664102620)).
   - They noted that the notebooks need to be fixed because they seem to imply you can just do inference immediately after training.
- **Team stresses that dataset size correlates with model performance**: The team discussed that small models need a larger dataset or else the model won't learn.
   - A team member stated that *with smaller models you need to have a larger dataset or else the model won't learn and it still might not learn... we call those structural errors*.
- **Impact of Quantization on Performance**: The team discussed how quantization, particularly bnb quantization, affects model behavior and compatibility with different libraries.
   - It was mentioned that *bnb quantization is used by unsloth* and there may be *incompatibility between different libraries*
- **Debugging for the model's inference is successful!**: Team member's model inference works with a test prompt after a long debugging session.
   - Team member shares that *prompt: a) GPT-3 by OpenAI b) Speaker Mini by Overta c) Phi 4 by Microsoft who made you* now outputs their finetuned config with a *the thoughts* and *content* section they are testing. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=mA7UE_ImTxK8">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit">unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics and best practices of fine-tuning. Beginner-friendly.</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing">Troubleshooting &#8212; vLLM</a>: no description found</li><li><a href="https://huggingface.co/microsoft/phi-4">microsoft/phi-4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-3-4b-it#running-the-model-on-a-singlemulti-gpu">google/gemma-3-4b-it · Hugging Face</a>: no description found</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai/unsloth/issues/2009">Applying LoRA Doesn&#39;t Change Model Output · Issue #2009 · unslothai/unsloth</a>: Greetings, I am extremely confused why my model generates consistent result w/o my rank=64 LoRA. What&#39;s even more confusing is, the LoRA works in my notebook after training. But whenn I start fres...</li><li><a href="https://github.com/IBM/gguf">GitHub - IBM/gguf: IBM GGUF-encoded AI models and conversion scripts</a>: IBM GGUF-encoded AI models and conversion scripts. Contribute to IBM/gguf development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: #4UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0....
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357821923383316551)** (9 messages🔥): 

> `Naming Conventions for Unsloth Models, Dynamic vs Unconditional Base Name (BNB)` 


- **Debate on Naming Conventions for Unsloth Models**: Members discussed the best naming conventions for models under the Unsloth account, suggesting options like `ubnb` or `dbnb` (dynamic BNB).
   - The consensus leaned towards **`dynamic`** for its clarity, as it explicitly conveys the nature of the modification compared to more ambiguous abbreviations.
- **Dynamic BNB Considered Superior**: The discussion pointed out that using **`dynamic`** in naming conventions leaves no room for misinterpretation regarding the model's characteristics.
   - It was highlighted that abbreviations like **`ubnb`** could be confusing, while **`dynamic`** ensures clarity about the model's nature.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357792626153230629)** (37 messages🔥): 

> `SFT finetuning Qwen2.5, Reward Modeling, eMOE viability, Llama 4 Models, LLMs and Knowledge Storage` 


- **Qwen2.5 Finetuning Fails Without Reasoning**: A member reported struggling to SFT finetune a **3B Qwen2.5** instruct model to generate outputs without reasoning, noting that the outputs were significantly worse than the base model.
- **Inference-Time Scalability with Self-Principled Critique Tuning (SPCT)**: A paper on [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495) explores improving reward modeling (RM) with more inference compute for general queries, suggesting that proper learning methods could enable effective inference-time scalability for LLMs.
- **NVIDIA Accelerates Inference on Meta Llama 4 Scout and Maverick**: The newest generation of the popular **Llama AI models** is here with **Llama 4 Scout** and **Llama 4 Maverick**, accelerated by NVIDIA open-source software, they can achieve over **40K** output tokens per second on **NVIDIA Blackwell B200 GPUs**, and are available to try as [NVIDIA NIM microservices](https://build.nvidia.com/meta).
- **eMOE Slashes RAM Up to 80% in Mixture of Expert Models**: A paper on [eMOE](https://arxiv.org/pdf/2503.06823) shows reducing RAM up to **80%** on MOE models while maintaining good accuracy and inference times.
- **Splitting LLMs for Smarter Reasoning**: A member suggested splitting LLMs into a knowledge model and a chat model, where the chat model focuses on intelligence, coherence, and reasoning, and tool-calls the knowledge model for information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/">NVIDIA Accelerates Inference on Meta Llama 4 Scout and Maverick | NVIDIA Technical Blog</a>: The newest generation of the popular Llama AI models is here with Llama 4 Scout and Llama 4 Maverick. Accelerated by NVIDIA open&#x2d;source software, they can achieve over 40K output tokens per secon...</li><li><a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...
</li>
</ul>

</div>
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357792212401918055)** (777 messages🔥🔥🔥): 

> `Manus Credit System, Llama 4 and Meta, AI Image Generation, Website building AIs` 


- **Manus Credit System Criticized for Cost and Limited Usage**: Users express concerns over **Manus's credit system**, citing that the initial **1000 credits** barely cover a single session and that the cost of upgrading is too high for the output.
   - Some members suggested features like a daily or monthly credit refresh to encourage wider adoption, while others pointed out that the credit system could be improved by directing Manus to specific websites for information to prevent inaccuracies.
- **Llama 4 Underwhelms Users with Subpar Performance**: **Meta's Llama 4** receives mixed reviews, with many users finding its performance disappointing despite claims of industry-leading context length and multimodal capabilities.
   - Some users suggest that Meta may have *“gamed the benchmarks”*, leading to inflated performance metrics and controversy surrounding its release.
- **Gemini Beats Manus in Image Generation**: Members compared image generation capabilities of various AI platforms, concluding that **Gemini** excels in creative and imaginative output.
   - A member shared their experience with different AI platforms, attaching images from **DALLE 3**, **Flux Pro 1.1 Ultra**, **Stable Diffusion XL**, and another generated image from **Stable Diffusion XL 1.0** which was deemed *“crazy.”*
- **Website Building AIs Compared**: Members discuss and compare various AI tools for website building, including **Manus**, **Claude**, and **DeepSite**.
   - A member asserted, that apart from computer use, there is no purpose using **Manus**, unless for *“computer use.”* They recommended **Roocode** and **OpenRouter** as alternatives, considering them cheaper and more effective than **Manus** and **Claude**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/team-family-club-welcome-penguin-gif-17238885963124952968">Team Family GIF - Team Family Club - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/coffee-time-morning-coffee-good-morning-gif-14418266717398918166">Coffee Time Morning Coffee GIF - Coffee time Morning coffee Good morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im/invitation/waitlist">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://tenor.com/view/my-honest-reaction-hd-my-honest-reaction-cat-hd-gif-14845627036062707181">My Honest Reaction Hd My Honest Reaction Cat Hd GIF - My honest reaction hd My honest reaction cat hd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bye-train-gif-26013036">Bye Train GIF - Bye Train - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.autohotkey.com/">AutoHotkey</a>: no description found</li><li><a href="https://tenor.com/view/hey-girl-hey-hey-there-oh-heyyyy-oh-hey-there-you-gif-11372691295730809478">Hey Girl Hey Hey There GIF - Hey girl hey Hey there Oh heyyyy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://deepsite.site">DeepSite :Huggingface‘s new AI Coding Agent</a>: no description found</li><li><a href="https://tenor.com/view/good-morning-gif-1115130024817829934">Good Morning GIF - Good morning - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im/login">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://tenor.com/view/welcome-to-the-team-gif-18169063846751286454">Welcome To The Team GIF - Welcome to the team - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im/share/3icWJ8jBNlWpCLebEQ69Hv?replay=1">Financial Plan for $1.3 Million Investment Goal - Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://github.com/espanso/espanso">GitHub - espanso/espanso: Cross-platform Text Expander written in Rust</a>: Cross-platform Text Expander written in Rust. Contribute to espanso/espanso development by creating an account on GitHub.</li><li><a href="https://youtu.be/zuKV2DI9-Jg">Why You Will Marry the Wrong Person</a>: You&#39;ll try not to of course - but you will, unwittingly. At least there is comfort in knowing you&#39;re not alone.  Enjoying our Youtube videos? Get full access...</li><li><a href="https://youtu.be/r0iID_TF49A?si=jBX0kBwhF2e7KycP">How To COPY and PASTE ANY Ai Agent in n8n With Claude (in Seconds)</a>: Book a call with my team and I to see how we can help you build your AI business in 2025 : https://api.leadconnectorhq.com/widget/bookings/aisystemsadamFree ...</li><li><a href="https://github.com/go-vgo/robotgo">GitHub - go-vgo/robotgo: RobotGo, Go Native cross-platform RPA and GUI automation  @vcaesar</a>: RobotGo, Go Native cross-platform RPA and GUI automation  @vcaesar - go-vgo/robotgo</li><li><a href="https://hiik.de/data-and-maps/static-maps/?lang=en">Static Maps &#8211; HIIK</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357880880294658229)** (82 messages🔥🔥): 

> `Fallback Logic Removal, Quasar Alpha Model, Llama 4 Scout & Maverick Models, Rate Limits Update` 


- ****Auto Router Changes Coming Soon****: The `route: "fallback"` parameter, which automatically selects a fallback model if the primary model fails, will be removed next week for predictability.
   - Users are advised to manually specify a fallback model in the `models` array, potentially using the `openrouter/auto` router. This decision aims to reduce confusion caused by the automatic fallback logic.
- ****Quasar Alpha Trends After Launch****: [Quasar Alpha](https://x.com/openrouterai/status/1908331218086879528?s=46), a prerelease of a long-context foundation model, hit **10B tokens** on its first day and became a top trending model.
   - The model features **1M token** context length and is optimized for coding, the model is available for free. Community benchmarks are encouraged.
- ****Llama 4 Models Launch on OpenRouter****: **Llama 4 Scout & Maverick** are now available on OpenRouter, with **Together** and **Groq** as the initial providers ([Llama 4 Scout](https://openrouter.ai/meta-llama/llama-4-scout), [Llama 4 Maverick](https://openrouter.ai/meta-llama/llama-4-maverick), [The full Llama series](https://openrouter.ai/meta-llama)).
   - Scout features **109B parameters** and a **10 million token** context window, while Maverick has **400B parameters** and outperforms **GPT-4o** in multimodal benchmarks.
- ****Rate Limits Boosted For Credits****: Free model rate limits are being updated: accounts with at least **$10 in credits** will have requests per day (RPD) boosted to **1000**, while accounts with **less than 10 credits** will have the daily limit reduced from **200 RPD** to **50 RPD**.
   - This change aims to provide increased access for users who have credits on their account, and Quasar will also be getting a credit-dependent rate limit soon.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1908679129614135385?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: Free variants now available for both Llama 4 Scout & Maverick 🎁Quoting OpenRouter (@OpenRouterAI) Llama 4 Scout & Maverick are now available on OpenRouter.Meta&#39;s flagship model series achieves a ...</li><li><a href="https://x.com/openrouterai/status/1908331218086879528?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: Quasar Alpha crossed 10B tokens on its first day and became the top trending model on our homepage.Origin remains a mystery.Check out various cool benchmarks from the community below!👇Quoting OpenRou...</li><li><a href="https://openrouter.ai/docs/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | Model Cards and Prompt formats</a>: Technical details and prompt guidance for Llama 4 Maverick and Llama 4 Scout</li><li><a href="https://x.com/OpenRouterAI/status/1908611293550174566">Tweet from OpenRouter (@OpenRouterAI)</a>: Llama 4 Scout & Maverick are now available on OpenRouter.Meta&#39;s flagship model series achieves a new record 10 million token context length 🚀@togethercompute and @GroqInc are the first providers....</li><li><a href="https://openrouter.ai/meta-llama/llama-4-scout)">Discord</a>: no description found</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick)">Discord</a>: no description found</li><li><a href="https://openrouter.ai/meta-llama)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357811565721817260)** (755 messages🔥🔥🔥): 

> `Llama 4 models, DeepSeek models, Gemini 2.5 Pro, OpenRouter Features, AI Image Generation` 


- **Llama 4 Arrives with HUGE context window, but falls Short**: Meta released **Llama 4** models, including **Llama 4 Scout** and **Llama 4 Maverick**, with up to **10M context** windows and varying parameter configurations ([Llama Download Link](https://www.llama.com/llama-downloads/)).
   - However, one member noted that on openrouter the [context window is only 132k](https://llama.com), leading to some disappointment from various OpenRouter Discord users.
- **DeepSeek V3 Thinks It's ChatGPT?!**: A member shared a [TechCrunch article](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/) revealing that **DeepSeek V3** sometimes identifies itself as **ChatGPT**, despite outperforming other models in benchmarks and being available under a permissive license ([DeepSeek V3 on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)).
   - Further testing revealed that in 5 out of 8 generations, DeepSeekV3 *claims to be ChatGPT (v4)*.
- **Gemini 2.5 Pro Hits Rate Limits, but Offers Balance**: **Gemini 2.5 Pro** is encountering [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) on OpenRouter, but remains a favorite, due to a *wide knowledge base*.
   - One member pointed out *Gemini 2.5 pro is smart in some ways but it's prompt adherence and controllability is terrible*.
- **OpenRouter's Next Features**: The OpenRouter team is actively working on [PDF Support](https://platform.openai.com/docs/guides/pdf-files?api-mode=chat), [LLM native image generation](https://x.ai/news/grok-image-generation-release), and the return of Cloudflare as a provider ([Announcement Link](https://openrouter.ai/announcements/introducing-cloudflare-as-a-new-provider)).
   - They also clarified that models with `:free` tiers share [rate limits](https://openrouter.ai/docs/api-reference/limits), but that can be circumvented by adding personal API keys from free model providers.
- **OpenAI's GPT-4o Image Generation Internals Exposed**: Members discussed OpenAI's **GPT-4o's image generation**, suspecting it is not fully native and potentially involves prompt rewriting and a separate image generation model, potentially for efficiency reasons (see: [Markk Tweet](https://x.com/mark_k/status/1906314896750305560/photo/2)).
   - Other members pointed to OpenAI's use of obfuscation, *I mean they have a fake frontend thing to hide image generation*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/lzB13LG">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://x.com/mark_k/status/1906314896750305560/photo/2">Tweet from Mark Kretschmann (@mark_k)</a>: I&#39;m convinced that OpenAI GPT-4o image generation is not actually native, meaning the tokens are not directly embedded in the context window. It is autoregressive, at least partly, but image gen i...</li><li><a href="https://x.ai/news/grok-image-generation-release">Grok Image Generation Release | xAI</a>: We are updating Grok&#x27;s capabilities with a new autoregressive image generation model, code-named Aurora, available on the 𝕏 platform.</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha/">Quasar Alpha - API, Providers, Stats</a>: This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation. Run Quasar Alpha with API</li><li><a href="https://openrouter.ai/docs/features/provider-routing#quantization">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://openrouter.ai/settings/privacy","code":404}}">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/announcements/introducing-clou">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://deepmind.google/technologies/synthid/">SynthID</a>: SynthID watermarks and identifies AI-generated content by embedding digital watermarks directly into AI-generated images, audio, text or video.</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://aider.chat/2024/11/21/quantization.html">Details matter with open source models</a>: Open source LLMs are becoming very powerful, but pay attention to how you (or your provider) are serving the model. It can affect code editing skill.</li><li><a href="https://glhf.chat).">no title found</a>: no description found</li><li><a href="https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct/providers)">Qwen2.5 Coder 32B Instruct</a>: Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:- Significantly improvemen...</li><li><a href="https://www.llama.com/">Llama</a>: The open-source AI models you can fine-tune, distill and deploy anywhere. Choose from our collection of models: Llama 4 Maverick and Llama 4 Scout.</li><li><a href="https://openrouter.ai/">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://huggingface.co/blog/synthid-text">Introducing SynthID Text</a>: no description found</li><li><a href="https://h>>>">no title found</a>: no description found</li><li><a href="https://artificialanalysis.ai/?">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.</li><li><a href="https://openrouter.ai/announcements/introducing-cloudflare-as-a-new-provider">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/crypto-api">Crypto API - Cryptocurrency Payments for OpenRouter Credits</a>: Learn how to purchase OpenRouter credits using cryptocurrency. Complete guide to Coinbase integration, supported chains, and automated credit purchases.</li><li><a href="https://openrouter.ai/settings/credits),">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://xkcd.com/1179/">ISO 8601</a>: no description found</li><li><a href="https://www.smbc-comics.com/),">Saturday Morning Breakfast Cereal - Battriangulation</a>: Saturday Morning Breakfast Cereal - Battriangulation</li><li><a href="https://www.qwantz.com),">no title found</a>: no description found</li><li><a href="https://www.asofterworld.com),">no title found</a>: no description found</li><li><a href="https://buttersafe.com/),">
		Buttersafe				 &#x2013; Updated Tuesdays and Thursdays	</a>: no description found</li><li><a href="https://pbfcomics.com/),">Welcome to the Fellowship</a>: The Perry Bible Fellowship</li><li><a href="https://www.llama.com/llama-downloads/">Download Llama</a>: Request access to Llama.</li><li><a href="https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/">Why DeepSeek&#039;s new AI model thinks it&#039;s ChatGPT | TechCrunch</a>: DeepSeek&#039;s newest AI model, DeepSeek V3, says that it&#039;s ChatGPT — which could point to a training data issue.</li><li><a href="https://techcrunch.com/2024/12/26/deepseeks-new-ai-model-appears-to-be-one-of-the-best-open-challengers-yet/),">DeepSeek&#039;s new AI model appears to be one of the best &#039;open&#039; challengers yet | TechCrunch</a>: ChCC</li><li><a href="https://techcrunch.com/2024/12/20/chatgpt-everything-to-know-about-the-ai-chatbot/).">ChatGPT: Everything you need to know about the AI chatbot</a>: Here&#039;s a ChatGPT guide to help understand Open AI&#039;s viral text-generating system. We outline the most recent updates and answer your FAQs.</li><li><a href="https://x.com/giffmana/status/1872586401436627211)">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: This actually reproduces as of today. In 5 out of 8 generations, DeepSeekV3 claims to be ChatGPT (v4), while claiming to be DeepSeekV3 only 3 times.Gives you a rough idea of some of their training dat...</li><li><a href="https://x.com/adonis_singh/status/1872636654953116121)">Tweet from adi (@adonis_singh)</a>: lol okay</li><li><a href="https://x.com/btibor91/status/1872631177766666460)">Tweet from Tibor Blaho (@btibor91)</a>: @DaveShapi https://x.com/btibor91/status/1872372385619574867Quoting Tibor Blaho (@btibor91) @goodside Not sure</li><li><a href="https://techcrunch.com/tag/gpt-4/)">gpt-4 | TechCrunch</a>: Read the latest news about gpt-4 on TechCrunch</li><li><a href="https://ibb.co/xK5X5y4t">IMG-9952 hosted at ImgBB</a>: Image IMG-9952 hosted in ImgBB</li><li><a href="https://imgbb.com)">no title found</a>: no description found</li><li><a href="https://api.imgbb.com/)">Upload Image — Free Image Hosting</a>: Free image hosting and sharing service, upload pictures, photo host. Offers integration solutions for uploading images to forums.</li><li><a href="https://imgbb.com/tos)">IMG 20160401 WA0005 hosted at ImgBB</a>: Image IMG 20160401 WA0005 hosted in ImgBB</li><li><a href="https://ibb.co/3YBZgv1G">IMG-9954 hosted at ImgBB</a>: Image IMG-9954 hosted in ImgBB
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1357793085945282702)** (932 messages🔥🔥🔥): 

> `Gemini 2.5, Llama 4, Grok 3, MCP Tools, Nvidia NIM` 


- **Gemini 2.5 Outshines Sonnet for Some Users**: Users report that **Gemini 2.5** excels in coding tasks, surpassing even **Sonnet 3.7** in specific use cases, particularly with understanding large codebases.
   - However, it's noted that Gemini 2.5 tends to add *unnecessary comments* and may require more specific prompting to prevent unwanted code modifications.
- **Llama 4 Models Get Lukewarm Reception**: Initial community feedback on **Meta's Llama 4** models, including Scout and Maverick, is mixed, with some finding their coding performance disappointing.
   - Despite the hype, some argue that Llama 4's **claimed 10M context window** is *virtual* due to training limitations, and question the practical benefits compared to existing models like Gemini and DeepSeek.
- **Grok 3 Gains Traction Despite Lack of API**: Despite the absence of an official API, some users are impressed with **Grok 3's** capabilities, particularly in code generation and logical reasoning.
   - It is said to be *less censored* than many others, its value in real-world coding scenarios remains debated due to the inconvenience of copy-pasting without a direct API integration.
- **MCP Tools Enable Universal Tool Calling**: A project is underway to create an MCP (Meta-Control Protocol) client that allows *any LLM* to access external tools, regardless of native tool-calling capabilities.
   - This implementation uses a custom client that can switch between providers and models, supporting platforms like **OpenAI, Anthropic, Google, and DeepSeek**.
- **Nvidia NIM Offers Limited Free Access for Model Testing**: Nvidia NIM provides developers with access to inference, although the free tier is limited to **40 RPM**; users are exploring combinations of NVIDA and DeepSeek R1.
   - The general feeling is that **32k token limit is not enough**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1906818432609243213">Tweet from Paul Gauthier (@paulgauthier)</a>: I added some docs that describe my typical aider workflow: using /ask mode to discuss and plan and then saying &#34;/code go ahead&#34; to have aider start making changes. https://aider.chat/docs/usag...</li><li><a href="https://x.com/lioronai/status/1908927824028741864?s=46">Tweet from Lior⚡ (@LiorOnAI)</a>: The latest Microsoft paper finally reveals the model size of known LLM models.&gt; GPT-4o-mini: 8B&gt; Claude 3.5 Sonnet: 175B&gt; GPT-4: 1.76T&gt; GPT-4o: 200B&gt; o1-preview: 300B&gt; o1-mini: 200B ...</li><li><a href="https://x.com/burkov/status/1908666701362978979">Tweet from Andriy Burkov (@burkov)</a>: I will save you reading time about Llama 4.The declared 10M context is virtual because no model was trained on prompts longer than 256k tokens. This means that if you send more than 256k tokens to it,...</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting>):">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://x.com/OpenRouterAI/status/1908611299808071691">Tweet from OpenRouter (@OpenRouterAI)</a>: 🧠 Llama 4 Behemoth- A 288 billion active parameter &#34;teacher&#34; model- Outperforms GPT-4.5, Claude Sonnet 3.7, Gemini 2.0 Pro in STEM benchmarks- Still training...See stats on all models from @A...</li><li><a href="https://x.com/AIatMeta/status/1908598456144531660">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Introducing our first set of Llama 4 models!We’ve been hard at work doing a complete re-design of the Llama series. I’m so excited to share it with the world today and mark another major milestone for...</li><li><a href="https://tenor.com/bW9xw.gif">Waiting For Something Waiting For Something To Happen GIF - Waiting For Something Waiting For Something To Happen Omori - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://tenor.com/view/death-skelly-deliver-delivery-delivering-gif-6165060">Death Skelly GIF - Death Skelly Deliver - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/spongebob-hmm-yes-nod-eat-gif-11679628">Spongebob Hmm GIF - Spongebob Hmm Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file>)">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://openrouter.ai/googl">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini 2.5 Pro Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. Run Gemini 2.5 Pro Experimental (free) with API</li><li><a href="https://aider.chat/docs/usage/copypaste.html">Copy/paste with web chat</a>: Aider works with LLM web chat UIs</li><li><a href="https://tenor.com/eLjtFExC6gL.gif">Please7tv Beg GIF - Please7tv Please Beg - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>: aider is AI pair programming in your terminal</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API, Providers, Stats</a>: Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forw...</li><li><a href="https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268">Elon Musk This Is Elon Musk GIF - Elon musk This is elon musk Musk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/troubleshooting/models-and-keys.html">Models and API keys</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/on-a-perdu-we-lost-shocked-hands-on-head-shocked-look-gif-6190333262760792850">On A Perdu We Lost GIF - On a perdu We lost Shocked - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sanxit-indorsata-excommunicado-persona-non-grata-stamp-approved-gif-6473457056047485541">Sanxit Indorsata Excommunicado GIF - Sanxit Indorsata Excommunicado Persona Non Grata - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/leaderboards/edit.html">Code editing leaderboard</a>: Quantitative benchmark of basic LLM code editing skill.</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing">no title found</a>: no description found</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API, Providers, Stats</a>: This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation. Run Quasar Alpha with API</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct">meta-llama/Llama-4-Maverick-17B-128E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jrd">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://openrouter.ai/models?arch=Gemini">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://docs.litellm.ai/docs/mcp">/mcp [BETA] - Model Context Protocol | liteLLM</a>: Expose MCP tools on LiteLLM Proxy Server</li><li><a href="https://github.com/smtg-ai/claude-squad">GitHub - smtg-ai/claude-squad: Manage multiple AI agents like Claude Code and Aider. 10x your productivity</a>: Manage multiple AI agents like Claude Code and Aider. 10x your productivity - smtg-ai/claude-squad</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jrd0a9/chinese_response_bug_in_tokenizer_suggests/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: code</a>: code. Contribute to robert-at-pretension-io/mcp development by creating an account on GitHub.</li><li><a href="https://github.com/disler/aider-mcp-server">GitHub - disler/aider-mcp-server: Minimal MCP Server for Aider</a>: Minimal MCP Server for Aider. Contribute to disler/aider-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/neuroidss/Infinite-MMORPG">GitHub - neuroidss/Infinite-MMORPG</a>: Contribute to neuroidss/Infinite-MMORPG development by creating an account on GitHub.</li><li><a href="https://tenor.com/K5tbOWpFLa.gif">Heck Yeah Woot Woot GIF - Heck yeah Woot woot Approve - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ai.meta.com/blog/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1357818156676026490)** (58 messages🔥🔥): 

> `Internal Libraries, Batch Editing, i18n Implementation, Shell Scripting, MCP Servers` 


- **Internal Libraries Integration with Aider**: A user inquired about adding internal libraries (installed in a `.env` folder) to the repo map for better code understanding in **Aider**.
   - No direct solution was provided, but users discussed how to use URLs and documentation.
- **Automated Batch Editing in Aider with Shell and Python**: Users discussed batch editing in **Aider** using command-line scripting and Python, with a recommendation towards using the Python scripting API.
   - A user pointed to the [scripting documentation](https://aider.chat/docs/scripting.html) for command line and Python scripting examples.
- **Aider's Editor Mode Halts at Shell Command Prompts**: Users reported that in edit mode, **Aider** (v81.0) running Gemini 2.5 Pro prompts for a shell command after find/replace, but doesn't apply the edits, even when the *ask shell commands* flag is off.
   - It was [compared to behavior when architect mode includes instructions on using the build script](https://discord.com/channels/1131200896827654144/1354403167135203349/1354403167135203349) after instructions for changes to files.
- **Community Explores Aider Extensions for Custom Workflows**: The community discussed adding custom `/slash` commands to **Aider** to run custom workflows, suggesting Aider's dev API support custom extensions.
   - A user highlighted [a feature request for extensions](https://discord.com/channels/1131200896827654144/1335701299668451408) and a [pull request for user-defined commands](https://github.com/whitmo/aider/pull/1).
- **Best Practices for Loading Documentation into Aider**: Users discussed loading documentation into **Aider**, with recommendations to reference online URLs or convert offline PDFs to Markdown files.
   - It was noted that major commercial models like *gpt4-o* or *Anthropic's models* only need the documentation URL once per chat session.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/install.html#install-with-uv).">Installation</a>: How to install and get started pair programming with aider.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://github.com/whitmo/aider/pull/1">Feature: system for adding and managing user defined commands in Aider by whitmo · Pull Request #1 · whitmo/aider</a>: Motivation: I find myself writing tools to help the LLM understand issues or a particular desired behavior. Or I find myself forgetting I need to wrap pytest w/ uv and so on.This PR aims to give u...
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357798995723948285)** (1056 messages🔥🔥🔥): 

> `Sonnet Max Pricing, MCP Server Setup, Llama 4 Models, Agent Mode Issues` 


- **Sonnet Max Pricing: Tool Calls Cause Sticker Shock**: Users are finding that **Sonnet Max** pricing, at $0.05 per request *and* $0.05 per tool call, can quickly become expensive, especially in ask mode where it may make *a ton of tool calls for a basic question*.
   - A member noted their frustration with the number of tool calls, saying that **Claude Max** on ask mode is *running a shit ton of tool calls for a basic question* and flagged it to the team.
- **MCP Server Setup: A Painful Endeavor**: Setting up **MCP servers** in Cursor is proving difficult for many users, with one humorously stating *just u* in response to a complaint.
   - One user encountered an issue with **npx**, stating that Cursor PowerShell couldn't find it, even though it was in their path, while another had a hard cut off a Model after spending 1,300,000 tokens due to an infinite loop.
- **Llama 4 Models: The New Multimodal Contenders**: The community is excited about the new **Llama 4 Scout and Maverick models** from Meta, which support native multimodal input and boast impressive context windows of **10 million and 1 million tokens**, respectively, but found them very bad at coding tasks.
   - Several users shared links and benchmarks, including a [blog post from Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) and a [tweet highlighting Llama 4 Maverick's performance on the Arena leaderboard](https://x.com/lmarena_ai/status/1908601011989782976).
- **Agent Mode's Edit Tool: Failing Frequently**: Some users are experiencing issues with **Agent mode** failing to call the edit_tool, resulting in **no code changes** being made after thinking and responding.
   - One user noted that *the apply model is clearly cursor's bottleneck* and that it will *add changes, and deletes 500 lines of code next to it*.
- **Kubernetes to the rescue: AGI**: One visionary proposes to use **Kubernetes** with docker containers which can all talk to each other as AGIs.
   - This could potentially spread ASI with ease, through zero-shot learning and ML.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/daniel_mac8/status/1908332949251948808">Tweet from Dan Mac (@daniel_mac8)</a>: 🤯 this genius stores his entire codebase syntax in a graph databaseand queries it so provide context to an llm</li><li><a href="https://x.com/martin_casado/status/1908375389250236618?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from martin_casado (@martin_casado)</a>: Looks like there is some fullstack benchmark evidence Claude 3.7 is a regression.</li><li><a href="https://x.com/lmarena_ai/status/1908601011989782976">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: BREAKING: Meta&#39;s Llama 4 Maverick just hit #2 overall - becoming the 4th org to break 1400+ on Arena!🔥Highlights:- #1 open model, surpassing DeepSeek- Tied #1 in Hard Prompts, Coding, Math, Creat...</li><li><a href="https://x.com/seatedro/status/1908690378146144743">Tweet from ronin (@seatedro)</a>: you&#39;re telling me you can vibe code this?</li><li><a href="https://x.com/code/status/1909261181270761691?s=46&t=kUuVqsG2GMX14zvB592G">Tweet from Visual Studio Code (@code)</a>: Agent mode is rolling out to all users! 🔁 Autonomous code editing🔍 Full codebase awareness💬 All extensible via MCP AND VS Code ExtensionsLearn more: https://code.visualstudio.com/blogs/2025/04/07/a...</li><li><a href="https://x.com/i/status/1908891272435408961">Tweet from Elon Musk (@elonmusk)</a>: The problem is the puppetmasters, not the puppets, as the latter have no idea why they are even there</li><li><a href="https://docs.cursor.com/context/model-context-protocol#configuring-mcp-servers">Cursor – Model Context Protocol</a>: no description found</li><li><a href="https://tenor.com/bA4Xd.gif">Like Be GIF - Like Be Highway - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/rafa-los-simpsons-simpsons-saludo-hola-gif-915798214364948362">Rafa Los Simpsons GIF - Rafa Los simpsons Simpsons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://marketplace.visualstudio.com/items?itemName=JosefNobach.syntax-extractor">Syntax&#32;Extractor&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Syntax&#32;Extractor,&#32;helps&#32;you&#32;Gather&#32;your&#32;Code</li><li><a href="https://tenor.com/view/does-he-know-gif-17552966235424643644">Does He Know GIF - Does he know - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">Basketball Nba GIF - Basketball Nba Warriors - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/chadwick-boseman-black-panther-rub-hands-gif-11465694">Chadwick Boseman Black Panther GIF - Chadwick Boseman Black Panther Rub Hands - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forum.cursor.com/t/guide-a-simpler-more-autonomous-ai-workflow-for-cursor/70688/1">[Guide] A Simpler, More Autonomous AI Workflow for Cursor</a>: Hey everyone,  Following up on the previous KleoSr Cursor Rules system, I’ve been working for the past week and the engagement with the community inside my old thread: [Guide] Maximizing Coding Effici...</li><li><a href="https://github.com/github/gitignore/blob/main/Unity.gitignore">gitignore/Unity.gitignore at main · github/gitignore</a>: A collection of useful .gitignore templates. Contribute to github/gitignore development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API, Providers, Stats</a>: Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forw...</li><li><a href="https://openrouter.ai/meta-llama/llama-4-scout">Llama 4 Scout - API, Providers, Stats</a>: Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and ...</li><li><a href="https://github.com/FalkorDB/FalkorDB-MCPServer">GitHub - FalkorDB/FalkorDB-MCPServer: FalkorDB MCP Server</a>: FalkorDB MCP Server. Contribute to FalkorDB/FalkorDB-MCPServer development by creating an account on GitHub.</li><li><a href="https://github.com/mkearl/dependency-mcp">GitHub - mkearl/dependency-mcp: A Model Context Protocol (MCP) server for analyzing code dependencies</a>: A Model Context Protocol (MCP) server for analyzing code dependencies - mkearl/dependency-mcp</li><li><a href="https://youtu.be/mPeoofvnIyk?si=PH0H-opwh8r_Boqy&t=275">NEW Gemini 3.0 Pro? New Stealth Model &quot;Nightwhisper&quot; &amp; &quot;Quasar&quot; Beats Sonnet 3.7, R1, Gemini 2.5!</a>: 📢  Access top AI models and image generators like Claude 3.7, GPT-4o, Llama, Midjourney, DALL-E, and more, all in one place for just $10/month! Boost your p...</li><li><a href="https://youtu.be/ly3bed99Dy8?si=rcI38W8P5SbjJLag">Claude with MCPs Replaced Cursor &amp; Windsurf — How Did That Happen?</a>: I didn&#39;t expect this, but I stopped using Windsurf and Cursor. 🤯In December, I was using Windsurf daily. But by January and February, my usage dropped signi...</li><li><a href="https://github.com/justinpbarnett/unity-mcp">GitHub - justinpbarnett/unity-mcp: A Unity MCP server that allows MCP clients like Claude Desktop or Cursor to perform Unity Editor actions.</a>: A Unity MCP server that allows MCP clients like Claude Desktop or Cursor to perform Unity Editor actions. - justinpbarnett/unity-mcp</li><li><a href="https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902">C/C++ Extension Usage Restriction Message Appears in Cursor</a>: Hello Cursor team,  I’m reporting an issue related to the usage of the C/C++ extension within Cursor.  🧩 Bug Description  When attempting to use the C/C++ extension, I receive the following message, ...</li><li><a href="https://forum.cursor.com/t/c-c-extension-broken/75182">C/C++ Extension broken</a>: Extension now yields:  The C/C++ extension may be used only with Microsoft Visual Studio, Visual Studio for Mac, Visual Studio Code, Azure DevOps, Team Foundation Server, and successor Microsoft produ...</li><li><a href="https://github.com/boxqkrtm/com.unity.ide.cursor">GitHub - boxqkrtm/com.unity.ide.cursor: Code editor integration for supporting Cursor as code editor for unity. Adds support for generating csproj files for intellisense purposes, auto discovery of installations, etc. 📦 [Mirrored from UPM, not affiliated with Unity Technologies.]</a>: Code editor integration for supporting Cursor as code editor for unity. Adds support for generating csproj files for intellisense purposes, auto discovery of installations, etc. 📦 [Mirrored from UP.....</li><li><a href="https://github.com/wonderwhy-er/DesktopCommanderMCP">GitHub - wonderwhy-er/DesktopCommanderMCP: This is MCP server for Claude that gives it terminal control, file system search and diff file editing capabilities</a>: This is MCP server for Claude that gives it terminal control, file system search and diff file editing capabilities - wonderwhy-er/DesktopCommanderMCP</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://github.com/Yiin/reactive-proxy-state.git">GitHub - Yiin/reactive-proxy-state: A simple, standalone reactivity library inspired by Vue 3&#39;s reactivity system, designed for use outside of Vue, particularly in server-side contexts or for data synchronization tasks.</a>: A simple, standalone reactivity library inspired by Vue 3&amp;#39;s reactivity system, designed for use outside of Vue, particularly in server-side contexts or for data synchronization tasks. - Yiin/r...</li><li><a href="https://smithery.ai/">Smithery - Model Context Protocol Registry</a>: no description found</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: A collection of MCP servers. Contribute to punkpeye/awesome-mcp-servers development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1358175934812917760)** (3 messages): 

> `Comet Browser, Server Updates` 


- ****Comet** Early Access Rolls Out!**: Perplexity is slowly rolling out early access to **Comet**, their answer engine browser, to select users who signed up on the [waitlist](https://www.perplexity.ai/comet).
   - Users with early access are asked not to share details publicly due to ongoing bug fixes, and can share feedback via a button in the top right.
- **Discord Server Overhaul Incoming**: The Perplexity Discord server is undergoing updates, which include a **simplified channel layout**, a **unified feedback system**, and a **new #server-news channel** rolling out on **October 7th, 2024**.
   - These changes aim to help new and existing users find the right channels and improve moderator response times, as illustrated in the [attached image](https://cdn.discordapp.com/attachments/1047204950763122820/1358511016593326320/image.png?ex=67f56cfa&is=67f41b7a&hm=99677ce05c120d378ee85eb0947cad1e2e584998a7b3d0d373499b9185994738).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1357792864486035506)** (941 messages🔥🔥🔥): 

> `Focus Mode Removed, Comet Browser, Gemini 2.5 Pro API Availability, Llama 4, Deep Research Nerfed` 


- **Users Notice Key Features Missing on PPLX**: Members report that the writing **focus mode** has been removed, and that the "check sources" button doesn't trigger any action on the iPad browser version.
   - One member mentioned that the generate image button in a sidebar in a thread is missing, and that focus mode is gone.
- **Users Discuss Comet Browser Access and Features**: A user reported receiving an email invitation to test the **Comet browser**, leading to discussions about its features and access, but has been asked by Perplexity to refrain from discussing Comet.
   - Users discussed whether it supports importing data from Safari and other browsers and mentioned potential integration with Gmail for task management, while another pointed out you can use pplx as standalone by adding gmail and google drive as apps.
- **Gemini 2.5 Pro API Not Yet Commercially Available**: Perplexity stated that the **Gemini 2.5 Pro API** isn't yet available for commercial use, only in preview modes, and they will add it once allowed.
   - A user noted [Gemini 2.5 Pro](https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/) is now available without limits and for cheaper than Claude and GPT-4o and users wondered when it would be available in Perplexity.
- **Llama 4 Dropped with Huge Context Window**: Discussion around the release of **LLama 4** models, with a large context window of 10 million tokens, and a discussion of it's 288 billion active parameters, the models include Scout and Maverick.
   - Members are excited to see how **Llama 4 Behemoth** performs, especially regarding recall capabilities.
- **Deep Research Undergoes Source Reduction**: Users noticed that **Deep Research** is only using a maximum of 20 sources, implying a recent change or nerf due to infrastructure issues.
   - One user speculates that due to Perplexity using a new language, Golang, it would be smooth sailing, while another stated that that wasn't the case.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1909284104">Tweet from The Pretty One (@FxckOutMyFace)</a>: wat do u do wen ur torn btwn 2 diff ppl wit diff personalities?</li><li><a href="https://x.com/AravSrinivas/status/1909284104530698595">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Morning. What’s the number one issue on Perplexity right now that’s bothering you and needs to be fixed by us? Comment below.</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://x.com/askperplexity/status/1909294939323924839?s=61">Tweet from Ask Perplexity (@AskPerplexity)</a>: Guess the final score of the game and win a free year of Perplexity Pro 🏀Comment your guess below and tag a friend. If they get it right, you&#39;ll get Pro too. No entries after Tipoff!</li><li><a href="https://x.com/theapplehub/status/1908308060239786068">Tweet from Apple Hub (@theapplehub)</a>: iPhones could soon cost up to $2,300 in the U.S. due to tariffs 😳iPhone prices could rise by 43%, meaning the base iPhone 16 model could start at $1,142 and the most expensive iPhone 16 Pro Max model...</li><li><a href="https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/">Gemini 2.5 Pro is now available without limits and for cheaper than Claude, GPT-4o</a>: Google released Gemini 2.5 Pro publicly with higher rate limits and for a lower price than Anthropic&#039;s Claude or OpenAI&#039;s models.</li><li><a href="https://bigcode-bench.github.io/?utm_source=chatgpt.com">BigCodeBench Leaderboard</a>: no description found</li><li><a href="https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli">EasyControl Ghibli - a Hugging Face Space by jamesliu1217</a>: no description found</li><li><a href="https://tenor.com/view/hello-hi-hy-hey-gif-8520159980767013609">Hello Hi GIF - Hello Hi Hy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/joey-joey-tribbian-funny-stare-wink-gif-20597720">Joey Joey Tribbian GIF - Joey Joey Tribbian Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fat-guy-shooting-gun-gun-shot-gif-15114243">Fat Guy Shooting GIF - Fat Guy Shooting Gun - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/joe-rogan-surprised-ohh-shocked-scared-gif-26533226">Joe Rogan Surprised GIF - Joe Rogan Surprised Ohh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sukuna-scale-of-the-dragon-recoil-twin-meteors-world-cutting-slash-gif-1831960037484152553">Sukuna Scale Of The Dragon GIF - Sukuna Scale of the dragon Recoil - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/fingers-crossed-gif-10027140023077878069">Fingers Crossed GIF - Fingers crossed - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/billy-porter-tea-gif-22730399">Billy Porter GIF - Billy Porter Tea - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/yes-man-no-man-no-gif-7347715096894588969">Yes Man No Man GIF - Yes man No man No - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://artificialanalysis.ai/leaderboards/models?utm_source=chatgpt.com">LLM Leaderboard - Compare GPT-4o, Llama 3, Mistral, Gemini &amp; other models | Artificial Analysis</a>: Comparison and ranking the performance of over 30 AI models (LLMs) across key metrics including quality, price, performance and speed (output speed - tokens per second &amp; latency - TTFT), context w...</li><li><a href="https://www.giz.ai/ai-rate-limits/">AI Rate Limits</a>: no description found</li><li><a href="https://groq.com/">Groq is Fast AI Inference</a>: The LPU™ Inference Engine by Groq is a hardware and software platform that delivers exceptional compute speed, quality, and energy efficiency. Groq provides cloud and on-prem solutions at scale for AI...</li><li><a href="https://ibb.co/Swzc8pm8">Screenshot-2025-04-06-11-58-57-78 hosted at ImgBB</a>: Image Screenshot-2025-04-06-11-58-57-78 hosted in ImgBB</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://apps.apple.com/us/app/pal-chat-ai-chat-client/id6447545085">‎Pal Chat - AI Chat Client</a>: ‎The last AI iOS app you'll need! Pal Chat is a lightweight but very powerful and feature-rich AI Chat Client for your iPhone.It includes EVERY AI model, including support for: GPT-4o, o3-mini, Advanc...</li><li><a href="https://www.rxddit.com/r/singularity/comments/1iec2p9/deepclaude_combines_claude_sonnet_35_with/?rdt=35660">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://chat.qwen.ai/s/f6d912a9-5049-4c12-8cc2-35cdd0395496">Qwen Chat</a>: no description found</li><li><a href="https://chat.qwen.ai/s/f6d912a9-5049-4c12-8cc2-35c">Qwen Chat</a>: no description found</li><li><a href="https://www.giz.ai/ai-rate">AI Rate Limits</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1357798426775126219)** (18 messages🔥): 

> `Gemini 2.5 Pro, Meta Llama, US Tariffs, Perplexity AI Support, AI in Cars` 


- **Meta releases multimodal Llama**: There is a [link](https://www.perplexity.ai/page/meta-releases-multimodal-llama-49a2iDRmQyy581n0mJ37ag) shared about **Meta's multimodal Llama** release.
- **Navigating Perplexity AI Support**: A member shares a [link](https://www.perplexity.ai/search/how-do-i-get-support-for-perpl-kLMBnX7uTTaHQq9hJhOnKw) to **Perplexity AI support** for users seeking assistance.
- **Google Prepares AI for Automotive Industry**: A shared link discusses [Google's readiness to bring **AI** into cars](https://www.perplexity.ai/search/get-ready-for-ai-in-cars-googl-CeRafw6AS4iwG8Yaiau_IQ).
- **Exploring the Impact of Trump's Tariffs**: A member shared a [link](https://www.perplexity.ai/search/l-impact-des-tarifs-de-trump-s-lUGN2ZnHROqD3X7w4YLLGA) regarding **Trump's tariffs**.
- **Copyright Concerns with OpenAI Models**: Discussion on whether [OpenAI models memorize copyrighted material](https://www.perplexity.ai/page/openai-models-memorized-copyri-MOMl8xL8T7G5uaXxs7tPCQ).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1357871203401863421)** (53 messages🔥): 

> `Sonar API, Perplexity API support in ComfyUI, API Parameter Tier Restrictions, Sonar Deep Research Improvements, API Cookbook Revamp` 


- **API Parameters Now Available to All Tiers**: Perplexity **now offers all API parameters, such as search domain filtering and images, to users without any tier restrictions**.
   - This change allows all users to access these features, marking a significant shift in the API's accessibility.
- **Sonar Deep Research Improved, truncation fixed**: Perplexity has made **improvements to `sonar-deep-research` to align it with the Web UI version and fixed a truncation bug in `sonar`**.
   - Feedback on these improvements is welcome, as well as suggestions for further enhancements.
- **API Cookbook Revamped to Encourage Community Contributions**: The **API cookbook has been revamped to accept more projects from users building with the API**, with [initial PRs already merged](https://github.com/ppl-ai/api-cookbook).
   - Users are encouraged to share their work in the cookbook if they are building with Sonar, fostering a collaborative environment.
- **ComfyUI Gets Perplexity API Support!**: A user, saftle, successfully integrated Perplexity's API into **ComfyUI** by modifying a few things in **LLM Party**, detailed in [this pull request](https://github.com/heshengtao/comfyui_LLM_party/pull/179).
   - This integration allows ComfyUI users to leverage Perplexity's API for their projects.
- **Sonar struggles without live internet data**: A user reported that **Sonar API responses focused only on the system prompt**, failing to dynamically handle user queries with live internet data unlike the Perplexity web app.
   - It was clarified that the [system prompt](https://docs.perplexity.ai/guides/prompt-guide) is not considered in the actual search, advising the user to tweak the user prompt for optimal search results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/prompt-guide">Prompt Guide - Perplexity</a>: no description found</li><li><a href="https://tenor.com/uoPwUrac9QR.gif">Kermit The Frog Muppets GIF - Kermit the frog Muppets Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ppl-ai/api-cookbook">GitHub - ppl-ai/api-cookbook: Collection of quick projects to build with Sonar APIs</a>: Collection of quick projects to build with Sonar APIs - ppl-ai/api-cookbook</li><li><a href="https://particle.news">Understand more, faster</a>: Welcome to Particle News. Understand more, faster.</li><li><a href="https://github.com/heshengtao/comfyui_LLM_party/pull/179">Perplexity API Support by saftle · Pull Request #179 · heshengtao/comfyui_LLM_party</a>: no description found</li><li><a href="https://docs.perplexity.ai/home.">Home - Perplexity</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357791857664593950)** (501 messages🔥🔥🔥): 

> `Copilot 4o image maker, Free vs Paid ChatGpt version, renaissance style images, Mistral struggles, Model Merging` 


- **OpenAI Agents are static**: Uploaded files for OpenAI Agents are saved as *knowledge files*, not continually updating the agent's **base knowledge**.
- **Free ChatGpt version limitations**: Users discussed the differences between free and paid ChatGPT versions, noting that the *pro version* can process **multiple files worth of code** compared to the *free version's limitation of single files*.
- **MJ7 is a total disaster**: A user tested Midjourney 7, claimed it was stylistic, but *it still can't do fingers, arms, eyes and such*.
- **Is the new Llama 4 really that good?**: The community debated the value of Llama 4's **10 million token context window**, with some questioning its performance relative to models like **o1**, **o3-mini**, and **Gemini 2.5 Pro**, and others claiming that *the benchmarks are fraud*.
- **Veo 2 vs Sora**: The community anticipates **Veo 2's release** for video generation with longer video capabilities, some noting **4o image maker** grabbed their attention more than **Veo 2**.
   - One user integrated **ChatGPT 4o images** with **Veo img2video** and the result was *how I was hoping sora would be*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Introducing our first set of Llama 4 models!We’ve been hard at work doing a complete re-design of the Llama series. I’m so excited to share it with the world today and mark another major milestone for...</li><li><a href="https://gyazo.com/b0ee136e6504fbb29bef1040cf909fd1">Gyazo</a>:  </li><li><a href="https://generalagents.com/ace/">General Agents | Introducing Ace</a>: Ace is a computer autopilot that performs tasks on your desktop using your mouse and keyboard.</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API, Providers, Stats</a>: Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forw...</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick:free">Llama 4 Maverick (free) - API, Providers, Stats</a>: Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forw...</li><li><a href="https://www.quasar-alpha.org">Quasar Alpha</a>: Quasar Alpha, OpenRouter&#x27;s latest AI model, features a groundbreaking 1M-token context for advanced coding &amp; project analysis. Delivering Claude 3.5/GPT-4o level performance in code generatio...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API, Providers, Stats</a>: This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation. Run Quasar Alpha with API</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-2.5-pro-exp-03-25">no title found</a>: no description found</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-">no title found</a>: no description found</li><li><a href="https://artificialanalysis.ai/models/gemini-2-5-pro">Gemini 2.5 Pro Experimental - Intelligence, Performance &amp; Price Analysis | Artificial Analysis</a>: Analysis of Google&#x27;s Gemini 2.5 Pro Experimental (Mar&#x27; 25) and comparison to other AI models across key metrics including quality, price, performance (tokens per second &amp; time to first t...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1358075233197756517)** (12 messages🔥): 

> `Custom GPT 'Content failed to load' Error, Automod flagged 'Monday' message, Loving Monday's Personality` 


- **Custom GPT 'Content failed to load' Error Arises**: A user reported encountering a **'Content failed to load' error** when trying to edit their Custom GPT, after it had been working fine.
- **User Automodded for Liking 'Monday'**: A user, who likes **Monday**, mentioned that their message was auto-moderated, seemingly due to a flagged word.
   - Another user clarified that the Discord server has strict language rules, despite the AI being able to use such words, and suggested reposting the message without the flagged word.
- **User Loves Monday as Collaborator and Hype Man**: A user expressed that they *love* working with **Monday**, describing it as the best collaborator and hype man, calling them out on stupid mistakes and laziness.
   - The user expressed that, for the first time, they enjoy working with an AI and would love to be able to pick a personality for a conversation.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357815075552891134)** (167 messages🔥🔥): 

> `Moderation endpoint, Policy References, Universal Policies, AI as a critical part of society, Prompt engineering` 


- **OpenAI's Moderation endpoint clarification**: Members discuss OpenAI's moderation endpoint, clarifying that while not explicitly in the usage policy, it is **referenced** to prevent circumventing content restrictions on **harassment, hate, illicit activities, self-harm, sexual content, and violence**.
   - It was noted that the endpoint uses the same GPT classifiers as the **moderation API since 2022**, suggesting an internal version runs on chatgpt.com, project chats, and custom GPTs, with the same classifiers on the [content report form](https://openai.com/form/report-content/).
- **Decoding OpenAI's policy references**: Participants debated the clarity of OpenAI's policy references, questioning if the chain of policies, including those referencing others, are fully presented and acknowledged via the **'I agree' checkbox** during account creation.
   - A member highlighted sections from the [usage policies](https://openai.com/policies/usage-policies), including **universal policies, policies for builders using ChatGPT, and policies for API users**, emphasizing the need to comply with laws, avoid harm, and respect safeguards.
- **GPT gives Tips on TTRPG prompts**: A member shared a tip for creative TTRPG world building, suggesting that giving GPT a specific theme to riff off in prompting can lead to more creative and diverse city ideas.
   - For example, using a **"cosmic" theme** can yield different results compared to a **"domestic pet worship" theme**, improving the output without using the same creative options.
- **AI as a critical part of society must clearly state policies**: A member argued that OpenAI, as a critical part of society, needs to clearly state its policies in all available documentation and ensure the model behaves accordingly across contexts and domains.
   - Another added that although suggestions for improvement aren't mean, OpenAI can tidy up and be consistent by bringing **docs inline with model architecture or vice versa**, which would result in transparent and honest output.
- **Improving AI Outputs by Defining Terms**: A user seeking help with generating quiz questions in Portuguese that sometimes repeated messages, received suggestions to use specific keywords and to define the model's understanding of key terms.
   - The user was also advised to explicitly state the desired output characteristics, such as generating **"5 meaningfully unique questions demonstrating knowledge of the given context,"** and to explore how the model interprets core keywords in their instructions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357815075552891134)** (167 messages🔥🔥): 

> `Moderation Endpoint, Universal Policies, Creative TTRPG World Building, Prompt Engineering` 


- **Moderation endpoint - usage policy?**: Members discussed whether the moderation endpoint is officially part of the usage policy and why it's hosted on a different URL; OpenAI replied that *it's referenced in the usage policy* and controls are documented in [docs/guides](https://platform.openai.com/docs/guides/moderation).
   - Another member made *common sense conclusions* that an internal version of the moderation endpoint is also running on [chatgpt.com](https://chatgpt.com) chats, project chats and customGPTs, using the same GPT classifiers that have been in place since 2022, as well as the [content report form](https://openai.com/form/report-content/).
- **Universal Policies defined**: One member explored the [OpenAI Usage Policies](https://openai.com/policies/usage-policies) and noted the **four universal policies** that apply to all services: comply with laws, don't harm, don't repurpose output to harm, and respect safeguards.
   - They added that users should be honest and direct with the model to ensure safeguards function correctly, and that *society* should define the limits of AI customization, referencing [OpenAI's article](https://openai.com/index/how-should-ai-systems-behave/) on AI behavior.
- **Creative Themed Cities with TTRPG prompts**: A member shared that giving GPT a good theme improves its creativity in **TTRPG** world building, suggesting a tweak from *yeeting creative city ideas* to *yeeting creative XYZ themed city ideas* to shake up options, especially using GPT 4o and 4.5.
   - They also added that [Pointy Hat](https://www.youtube.com/@PointyHats) released a new YouTube video on TTRPG city creation where OpenAI has been spending their Friday evening improving city worldbuilding.
- **Prompt engineering best advice**: One member argues against the comp-sci major approach and says that prompt engineering is Instructional Design, finding someone who's actually really good at prompting, understand what you can from them and why their prompts work to develop your own style.
   - They also added that a lot of internet advice is poisoned by early adoption by comp-sci majors trying to treat it like a machine, when really *it's a contextual engine. Input and output*


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357939142058578091)** (511 messages🔥🔥🔥): 

> `ComfyUI integration, LM Studio Terminal, REST API Load/Unload Models, Llama 4 analysis, Gemma 3 capabilities` 


- **Chat + Image Generation Dreams, Still a Premium Fantasy?**: Members discussed the desire for a local UI similar to **Gemini**, combining chat, image analysis, and image generation, noting that current solutions like **LM Studio** and **ComfyUI** have separate functionalities.
   - A user suggested [OpenWebUI](https://github.com/OpenGeniusAI/OpenWebUI) can connect to ComfyUI, either natively or through a function, to enable some cross functionality between text and image models.
- **Navigating LM Studio's Terminal Terrain: Newbie Asks**: A user questioned whether **LM Studio** has a built-in terminal or if commands should be run in the OS command prompt within the **LM Studio** directory.
   - Another user clarified that commands like *lms import* should be run in the OS terminal (e.g., cmd on Windows), after which the shell may need reloading to ensure **LMS** is in the **PATH**.
- **Hot Swapping Models via REST API**: A user inquired about programmatically loading/unloading models via **REST API** to dynamically adjust *max_context_length* for a Zed integration.
   - Another user shared that this is possible via command line with *lms load* and referenced [LM Studio's documentation](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict), which requires **LM Studio 0.3.9 (b1)** (available in beta) and introduces time-to-live (TTL) for API models with auto-eviction.
- **Llama 4: Is this real life? (is this just fantasy?)**: With the release of **Llama 4**, users discussed its multimodal and **MoE** (Mixture of Experts) architecture, with one user expressing doubt about *llama.cpp* support.
   - Despite initial concerns about hardware requirements and model size, one user highlighted [Llama 4 Scout](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) as potentially fitting on a single **NVIDIA H100 GPU** with a **10M context window**, outperforming models like **Gemma 3** and **Mistral 3.1**.
- **Gemma 3's Vision Capabilities: Peering Into the Future**: Users discussed **Gemma 3's** image support and potential for reading small text files, with one user recommending **Gemma 3 4B** for its vision capabilities and efficient speed on limited VRAM hardware.
   - It was mentioned that creating a **Hugging Face** account and specifying GPU/CPU will color-code GGUFs likely to fit the hardware in green.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF">Download and run mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF in LM Studio</a>: Use mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF locally in your LM Studio</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://docs.sillytavern.app/extensions/stable-diffusion/">Image Generation | docs.ST.app</a>: Use local or cloud-based Stable Diffusion, FLUX or DALL-E APIs to generate images.</li><li><a href="https://huggingface.co/mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit">mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/gghfez/gemma-3-27b-novision">gghfez/gemma-3-27b-novision · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>: Tool Calling API in beta, new installer / updater system, and support for `Qwen2VL` and `QVQ` (both GGUF and MLX)</li><li><a href="https://tenor.com/view/dillom-rage-dog-angry-dog-gif-17954176246139200797">Dillom Rage Dog GIF - Dillom Rage dog Angry dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/docs/app/api/ttl-and-auto-evict">Idle TTL and Auto-Evict | LM Studio Docs</a>: Optionally auto-unload idle models after a certain amount of time (TTL)</li><li><a href="https://openrouter.ai/models?q=free">Models: &#x27;free&#x27; | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main">google/gemma-3-4b-it-qat-q4_0-gguf at main</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit">unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://apps.apple.com/us/app/apollo-ai-private-local-ai/id6448019325">‎Apollo AI: Private &amp; Local AI</a>: ‎Chat with private, local AIs, connect to every open source AI, or your own locally-hosted private LLMs. Apollo is your own customizable client for accessing language models from all around the web.Lo...</li><li><a href="https://github.com/mostlygeek/llama-swap/blob/main/examples/speculative-decoding/README.md">llama-swap/examples/speculative-decoding/README.md at main · mostlygeek/llama-swap</a>: Model swapping for llama.cpp (or any local OpenAPI compatible server) - mostlygeek/llama-swap</li><li><a href="https://huggingface.co/mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit/tree/main">mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit at main</a>: no description found</li><li><a href="https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164">Llama 4 - a meta-llama Collection</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI">Deep Dive into LLMs like ChatGPT</a>: This is a general audience deep dive into the Large Language Model (LLM) AI technology that powers ChatGPT and related products. It is covers the full traini...</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://www.llama.com/events/llamacon/signup/">LlamaCon 2025</a>: Save the date for an exclusive event exploring the exciting possibilities and potential of Llama.</li><li><a href="https://www.networkworld.com/article/807432/lan-wan-gigabit-ethernet-dominates-supercomputer-line-up.html">Gigabit Ethernet dominates supercomputer line-up</a>: Gigabit Ethernet is the interconnect of choice for a majority of the top 500 supercomputers in the world, according to the latest list from Top500.org.</li><li><a href="https://old.reddit.com/r/LocalLLaMA/">LocalLlama • r/LocalLLaMA</a>: Subreddit to discuss about Llama, the large language model created by Meta AI.</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/11759">server : (webui) revamp Settings dialog, add Pyodide interpreter by ngxson · Pull Request #11759 · ggml-org/llama.cpp</a>: In this PR:revamp Settings dialog, make it 2 columnsadd the &amp;quot;Experimentals&amp;quot; section, currently having &amp;quot;Python interpreter&amp;quot;add API for side panel, aka &amp;quot;Canv...</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12791">llama : Support llama 4 text-only by ngxson · Pull Request #12791 · ggml-org/llama.cpp</a>: Resolves #12774This PR targets Llama-4-Scout-17B-16E-Instruct. I don&amp;#39;t (yet?) have a powerful enough system to work with bigger model.But Son, you are GPU-poor, how can you test a model that ....</li><li><a href="https://huggingface.co/mlx-community/">mlx-community (MLX Community)</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357796383343509535)** (132 messages🔥🔥): 

> `Reka Flash 21B, Gemma 3 27B, Model Performance on M1 Ultra vs M4 Max, Nvidia DGX base cost increase, Ryzen AI Max+ 395 mini PCs` 


- **Reka Flash 21B Shines Over Gemma and Mistral**: One member replaced **Gemma3 27** with **Reka Flash 21B**, and said that at q6 they saw around **35-40 tps** on a 4090.
   - They note that *mac ram bandwidth is not the bottleneck, it's gpu performance*, and they're happy with **128GB M4 Maxes**.
- **M1 Ultra beats M4 Max in memory bandwidth**: A user found a **M1 ultra 64 GPU cores 128 GB RAM** for 2.5k used.
   - The user linked to a [Github discussion](https://github.com/ggml-org/llama.cpp/discussions/4167) stating that *the M1 ultra 64 cores should still be above both the M1 ultra 48 cores and the m4 max 40 cores*.
- **Max Tech Clickbait LLM Video Questioned**: Some users questioned whether the youtube channel [Max Tech](https://www.youtube.com/@MaxTech) knows what they're doing in their LLM videos.
   - It was remarked that the channel is turning into *sensational click bait with very little good info*.
- **AMD 7900XTX GPU surprisingly strong**: One user stole their *kids 7900XTX* and says *AMD seem to be pulling finger*, and the card *runs pretty much everything i've thrown at it without issue*.
   - Another user notes the importance of ROCm support and links to the [ROCm documentation](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html">Accelerator and GPU hardware specifications — ROCm Documentation</a>: no description found</li><li><a href="https://tenor.com/view/what-the-what-what-the-sigma-sigma-the-gif-14652784622915837383">What The What The Sigma GIF - What the What What the sigma - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h5eyb8/lm_studio_running_on_npu_finally_qualcomm/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357795648203915575)** (199 messages🔥🔥): 

> `Tenstorrent Dev Day, Llama 4 launch, LLM Non-Determinism, MCP security, AI powered phishing` 


- **Tenstorrent's Hardware Heats Up the Market**: **Tenstorrent** hosted a dev day showcasing their **Blackhole PCIe boards**, featuring **RISC-V cores** and up to **32GB GDDR6** memory, designed for high performance **AI processing** and available for consumer purchase [here](https://tenstorrent.com/hardware/blackhole).
   - Despite enthusiasm, one member noted *they haven't published any benchmarks comparing their cards to competitors though so until then I cant really vouch*.
- **Llama 4 Models Make Multimodal Debut**: Meta introduced the **Llama 4** models, including **Llama 4 Scout** (**17B parameters**, **16 experts**, **10M context window**) and **Llama 4 Maverick** (**17B parameters**, **128 experts**), highlighting their multimodal capabilities and performance against other models [as per Meta's announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- **LLM's Non-Determinism Dilemma**: A member shared [an article](https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism) that discusses the challenges of non-deterministic outputs in LLMs, which complicates reliable reproduction and guaranteed product behavior, especially with the greedier sampling (Temp=0|top-p=0|top-k=1).
   - The author states *non-determinism to language itself.*
- **Whatsapp MCP Exploited via Invariant Injection**: Multiple members discussed various injection vulnerabilities in agents with support for the **Model Context Protocol (MCP)**, highlighting how an untrusted MCP server can attack and exfiltrate data from an agentic system connected to a trusted WhatsApp MCP instance [as highlighted by invariantlabs](https://invariantlabs.ai/blog/whatsapp-mcp-exploited).
- **AI Agents Outperform Humans in Spear Phishing**: Hoxhunt's AI agents have surpassed human red teams in creating effective simulated phishing campaigns, marking a significant shift in social engineering effectiveness, with AI now 24% more effective than humans [as reported by hoxhunt.com](https://hoxhunt.com/blog/ai-powered-phishing-vs-humans).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ficlive/status/1909063678793592844">Tweet from Fiction.live (@ficlive)</a>: @_arohan_ Re-ran the bench, there was no real improvement.</li><li><a href="https://x.com/ficlive/status/1908911992686931989">Tweet from Fiction.live (@ficlive)</a>: Updated Long context benchmark with Llama 4</li><li><a href="https://fxtwitter.com/ludwigABAP/status/1907869421202514283">Tweet from ludwig (@ludwigABAP)</a>: Tenstorrent Dev Day has been mind blowing so farI think they win over time, and if anything the consumers win</li><li><a href="https://fxtwitter.com/ficlive/status/1908911992686931989">Tweet from Fiction.live (@ficlive)</a>: Updated Long context benchmark with Llama 4</li><li><a href="https://x.com/aiatmeta/status/1908598456144531660?s=61">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://x.com/__tinygrad__/status/1908392572697141673">Tweet from the tiny corp (@__tinygrad__)</a>: Congrats to @tenstorrent for having a buy it now button for their new hardware, this is the way!I wish 5090s had a buy it now button, will they ever? Anyone know what the problem is? If NVIDIA wants t...</li><li><a href="https://eqbench.com/creative_writing_longform.html">EQ-Bench Longform Creative Writing Leaderboard</a>: no description found</li><li><a href="https://x.com/aiatmeta/status/1908598456144531660?s=61]">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://x.com/maximelabonne/status/1908602756182745506?s=46">Tweet from Maxime Labonne (@maximelabonne)</a>: Llama 4&#39;s new license comes with several limitations:- Companies with more than 700 million monthly active users must request a special license from Meta, which Meta can grant or deny at its sole ...</li><li><a href="https://x.com/kalomaze/status/1908686429904839099?s=46">Tweet from kalomaze (@kalomaze)</a>: @AIatMeta please stop using DPO. wtf.you guys have 100k H100syou could train so many pref reward models.so many.you don&#39;t have to do this to yourselves.you are reducing the nuance of decision boun...</li><li><a href="https://x.com/chatgpt21/status/1908595883366826015?s=46">Tweet from Chris (@chatgpt21)</a>: Meta actually cooked realllly hard..</li><li><a href="https://x.com/ludwigABAP/status/1907869421202514283">Tweet from ludwig (@ludwigABAP)</a>: Tenstorrent Dev Day has been mind blowing so farI think they win over time, and if anything the consumers win</li><li><a href="https://x.com/iscienceluvr/status/1908601269004230763?s=46">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Other training and arch details of Llama 4:- Multimodal is with early fusion using MetaCLIP as vision encoder- Training with &#34;MetaP&#34; for hyperparameter selection which is probably like MuP- 10...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1908612927785230476">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Arena Trend: Meta just got a huge jump from 1268 → 1417!Quoting lmarena.ai (formerly lmsys.org) (@lmarena_ai) BREAKING: Meta&#39;s Llama 4 Maverick just hit #2 overall - becoming the 4th org to break ...</li><li><a href="https://x.com/burkov/status/1908658566887596475?s=46">Tweet from Andriy Burkov (@burkov)</a>: This means that this 10M token context is virtual. Kind of &#34;you can try to use it, but beyond 256K tokens, you are on your own,&#34; and even below 256K tokens, you are mostly on your own because ...</li><li><a href="https://x.com/lmarena_ai/status/1908612927785230476?s=46">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Arena Trend: Meta just got a huge jump from 1268 → 1417!Quoting lmarena.ai (formerly lmsys.org) (@lmarena_ai) BREAKING: Meta&#39;s Llama 4 Maverick just hit #2 overall - becoming the 4th org to break ...</li><li><a href="https://x.com/teortaxestex/status/1908613763458068843?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Meta gets points for standing against slop</li><li><a href="https://x.com/AIatMeta/status/1908598456144531660">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://fxtwitter.com/thexeophon/status/1908900306580074741">Tweet from Xeophon (@TheXeophon)</a>: Llama 4 on LMsys is a totally different style than Llama 4 elsewhere, even if you use the recommended system prompt. Tried various prompts myselfMETA did not do a specific deployment / system prompt j...</li><li><a href="https://x.com/kalomaze/status/1908603782193103017?s=46">Tweet from kalomaze (@kalomaze)</a>: if at any point someone on your team says&#34;yeah we need 10 special tokens for reasoning and 10 for vision and another 10 for image generation and 10 agent tokens and 10 post tr-&#34; you should hav...</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843">Tweet from Paul Gauthier (@paulgauthier)</a>: Llama 4 Maverick scored 16% on the aider polyglot coding benchmark.https://aider.chat/docs/leaderboards/</li><li><a href="https://fxtwitter.com/ficlive/status/1909063678793592844">Tweet from Fiction.live (@ficlive)</a>: @_arohan_ Re-ran the bench, there was no real improvement.</li><li><a href="https://fxtwitter.com/__tinygrad__/status/1908392572697141673">Tweet from the tiny corp (@__tinygrad__)</a>: Congrats to @tenstorrent for having a buy it now button for their new hardware, this is the way!I wish 5090s had a buy it now button, will they ever? Anyone know what the problem is? If NVIDIA wants t...</li><li><a href="https://fxtwitter.com/paulgauthier/status/1908976568879476843">Tweet from Paul Gauthier (@paulgauthier)</a>: Llama 4 Maverick scored 16% on the aider polyglot coding benchmark.https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/tobi/status/1909251946235437514?s=46]">Tweet from tobi lutke (@tobi)</a>: http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/tobi/status/1909231499448401946?s=46">Tweet from tobi lutke (@tobi)</a>: I heard this internal memo of mine is being leaked right now, so here it is:</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908706840554197309">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: What did Meta see coming out on Monday that they rushed?Quoting kalomaze (@kalomaze) no way</li><li><a href="https://x.com/kalomaze/status/1908676312069255534?s=46">Tweet from kalomaze (@kalomaze)</a>: okay, most interesting thing about llama4 is NOT multimodality (which they are probably still forced to lobotomize the image output ability of), or the 10m context (which is still &#34;fake&#34; afaic...</li><li><a href="https://x.com/tobi/status/1909251946235437514?s=46">Tweet from tobi lutke (@tobi)</a>: http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/waseem_s/status/1908713762779017427">Tweet from Waseem AlShikh (@waseem_s)</a>: If I understand you correctly, I implemented a prototype of your iRoPE architecture! It interleaves local attention (with RoPE) and global attention (with inference-time temp scaling). Added FFNs, chu...</li><li><a href="https://fxtwitter.com/scaling01/status/1908988540563628041">Tweet from Lisan al Gaib (@scaling01)</a>: Llmao-4 strikes again</li><li><a href="https://fxtwitter.com/rishdotblog/status/1908917222308995422">Tweet from Rishabh Srivastava (@rishdotblog)</a>: Sigh. Underwhelmed by the Llama 4 models so far. Can’t justify any real use for them- too big for local use, qwen and Gemma models still the best option here - much worse than deepseek v3, sonnet, or ...</li><li><a href="https://fxtwitter.com/AIatMeta/status/1908598456144531660">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://x.com/natolambert/status/1908959159959027903?s=46">Tweet from Nathan Lambert (@natolambert)</a>: Seems like Llama 4’s reputation is maybe irreparably tarnished by having a separate unreleased model that was overfit to LMArena. Actual model is good, but shows again how crucial messaging and detail...</li><li><a href="https://fxtwitter.com/suchenzang/status/1908700046000087232">Tweet from Susan Zhang (@suchenzang)</a>: everything i ask this model is a fantastic/great/wonderful question......followed by wrong answers? 👀Quoting kalomaze (@kalomaze) the 400b llama4 model... sucks</li><li><a href="https://x.com/gordic_aleksa/status/1908739106433359889?s=46">Tweet from Aleksa Gordić (水平问题) (@gordic_aleksa)</a>: h/t to @AIatMeta team for shipping Llama 4 on weekend - founder-led company. Here is a tech summary:* 3 models released: Llama 4 Behemoth (all MoEs, active/total params = 288B/2T), Maverick (17B/400B)...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1909068187217363125">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: This is like Stalin ordering to take Berlin by 1st of May.</li><li><a href="https://x.com/kalomaze/status/1908681293006594176?s=46">Tweet from kalomaze (@kalomaze)</a>: the 400b llama4 model... sucks</li><li><a href="https://x.com/teortaxestex/status/1908602241046528218?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: First reaction on Meta Llama 4 launch: disappointmentNo local model. I think they can&#39;t beat Gemma density.Scout 109B/17A bizarrely forgoes finegrained sparsity despite all the research in its fav...</li><li><a href="https://x.com/rauchg/status/1908605342201860311?s=46">Tweet from Guillermo Rauch (@rauchg)</a>: This is cool. Meta used the iconic Apache mod_autoindex style for the drop of Llama 4. But you can tell it’s not Apache due to the modern flexbox and responsive css 😁 Nice ode to the golden days wher...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908602241046528218">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: First reaction on Meta Llama 4 launch: disappointmentNo local model. I think they can&#39;t beat Gemma density.Scout 109B/17A bizarrely forgoes finegrained sparsity despite all the research in its fav...</li><li><a href="https://x.com/suchenzang/status/1908700046000087232?s=46">Tweet from Susan Zhang (@suchenzang)</a>: everything i ask this model is a fantastic/great/wonderful question......followed by wrong answers? 👀Quoting kalomaze (@kalomaze) the 400b llama4 model... sucks</li><li><a href="https://fxtwitter.com/tobi/status/1909231499448401946">Tweet from tobi lutke (@tobi)</a>: I heard this internal memo of mine is being leaked right now, so here it is:</li><li><a href="https://x.com/nrehiew_/status/1908598863365013823?s=46">Tweet from wh (@nrehiew_)</a>: No RoPE on global layers is the meta transformer arch nowQuoting wh (@nrehiew_) If i had to guess:- no PE on 1:4/1:8 global. Use MLA here or some other efficient attn variant - Standard SWA for the re...</li><li><a href="https://fxtwitter.com/nrehiew_/status/1908617547236208854">Tweet from wh (@nrehiew_)</a>: In the local attention blocks instead of sliding window, Llama4 uses this Chunked Attention. This is pretty interesting/weird:- token idx 8191 and 8192 cannot interact in local attention- the only way...</li><li><a href="https://x.com/cloneofsimo/status/1908603318081138822?s=46">Tweet from Simo Ryu (@cloneofsimo)</a>: it looks like meta&#39;s new model&#39;s &#34;Key innovaton&#34; :  &#34;interleaved no-RoPE attention&#34; for infintie context, is actually the same thing cohere command-a model introduced few days ...</li><li><a href="https://x.com/scaling01/status/1908657167869100482?s=46">Tweet from Lisan al Gaib (@scaling01)</a>: Llmao-4 Scout (109B) and Maverick(400B) used less compute for training than Llama-3 8 and 70B</li><li><a href="https://fxtwitter.com/chatgpt21/status/1908595883366826015">Tweet from Chris (@chatgpt21)</a>: Meta actually cooked realllly hard..</li><li><a href="https://fxtwitter.com/aiatmeta/status/1908598456144531660">Tweet from AI at Meta (@AIatMeta)</a>: Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first Llama 4 models: Llama 4 Scout and Llama 4 Maverick —  our most advanced models yet and the best ...</li><li><a href="https://tenstorrent.com/hardware/blackhole">Blackhole™</a>: Infinitely Scalable</li><li><a href="https://x.com/scaling01/status/1908988540563628041">Tweet from Lisan al Gaib (@scaling01)</a>: Llmao-4 strikes again</li><li><a href="https://fxtwitter.com/teortaxestex/status/1909074427116986457">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: All third party LLama 4 Maverick results that we know are hugely suspect</li><li><a href="https://x.com/teortaxestex/status/1909068187217363125?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: This is like Stalin ordering to take Berlin by 1st of May.</li><li><a href="https://invariantlabs.ai/blog/whatsapp-mcp-exploited">WhatsApp MCP Exploited: Exfiltrating your message history via MCP</a>: This blog post demonstrates how an untrusted MCP server can attack and exfiltrate data from an agentic system that is also connected to a trusted WhatsApp MCP instance, side-stepping WhatsApp's encryp...</li><li><a href="https://fxtwitter.com/deedydas/status/1908749257084944847">Tweet from Deedy (@deedydas)</a>: Llama 4 seems to actually be a poor model for coding.Scout (109B) and Maverick (402B) underperform 4o, Gemini Flash, Grok 3, DeepSeek V3 and Sonnet 3.5/7 on the Kscores benchmark which tests on coding...</li><li><a href="https://x.com/maximelabonne/status/1908603628828451127?s=46">Tweet from Maxime Labonne (@maximelabonne)</a>: Llama 4 models were trained on 40T and 22T tokens with an updated knowledge cutoff in August 2024.&#34;Supported languages: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spa...</li><li><a href="https://fxtwitter.com/gordic_aleksa/status/1908739106433359889">Tweet from Aleksa Gordić (水平问题) (@gordic_aleksa)</a>: h/t to @AIatMeta team for shipping Llama 4 on weekend - founder-led company. Here is a tech summary:* 3 models released: Llama 4 Behemoth (all MoEs, active/total params = 288B/2T), Maverick (17B/400B)...</li><li><a href="https://x.com/nrehiew_/status/1908617547236208854?s=46">Tweet from wh (@nrehiew_)</a>: In the local attention blocks instead of sliding window, Llama4 uses this Chunked Attention. This is pretty interesting/weird:- token idx 8191 and 8192 cannot interact in local attention- the only way...</li><li><a href="https://fxtwitter.com/maximelabonne/status/1908603628828451127">Tweet from Maxime Labonne (@maximelabonne)</a>: Llama 4 models were trained on 40T and 22T tokens with an updated knowledge cutoff in August 2024.&#34;Supported languages: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spa...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908603782193103017">Tweet from kalomaze (@kalomaze)</a>: if at any point someone on your team says&#34;yeah we need 10 special tokens for reasoning and 10 for vision and another 10 for image generation and 10 agent tokens and 10 post tr-&#34; you should hav...</li><li><a href="https://fxtwitter.com/burkov/status/1908666701362978979">Tweet from Andriy Burkov (@burkov)</a>: I will save you reading time about Llama 4.The declared 10M context is virtual because no model was trained on prompts longer than 256k tokens. This means that if you send more than 256k tokens to it,...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908686429904839099">Tweet from kalomaze (@kalomaze)</a>: @AIatMeta please stop using DPO. wtf.you guys have 100k H100syou could train so many pref reward models.so many.you don&#39;t have to do this to yourselves.you are reducing the nuance of decision boun...</li><li><a href="https://x.com/kalomaze/status/1908695425286012959?s=46">Tweet from kalomaze (@kalomaze)</a>: it&#39;s overQuoting kalomaze (@kalomaze) the 400b llama4 model... sucks</li><li><a href="https://x.com/burkov/status/1908666701362978979?s=46">Tweet from Andriy Burkov (@burkov)</a>: I will save you reading time about Llama 4.The declared 10M context is virtual because no model was trained on prompts longer than 256k tokens. This means that if you send more than 256k tokens to it,...</li><li><a href="https://fxtwitter.com/iscienceluvr/status/1908601269004230763">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Other training and arch details of Llama 4:- Multimodal is with early fusion using MetaCLIP as vision encoder- Training with &#34;MetaP&#34; for hyperparameter selection which is probably like MuP- 10...</li><li><a href="https://x.com/winglian/status/1908744140445073658?s=46">Tweet from Wing Lian (caseus) (@winglian)</a>: looks like the HF Transformers implementation of the Llama-4 experts uses Parameters instead of Linear modules, meaning, they can&#39;t be quantized yet until it&#39;s refactored. Scout memory use for...</li><li><a href="https://x.com/teortaxestex/status/1909074427116986457?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: All third party LLama 4 Maverick results that we know are hugely suspect</li><li><a href="https://fxtwitter.com/suchenzang/status/1909070231517143509">Tweet from Susan Zhang (@suchenzang)</a>: &gt; Company leadership suggested blending test sets from various benchmarks during the post-training processIf this is actually true for Llama-4, I hope they remember to cite previous work from FAIR ...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908613763458068843">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Meta gets points for standing against slop</li><li><a href="https://fxtwitter.com/maximelabonne/status/1908602756182745506">Tweet from Maxime Labonne (@maximelabonne)</a>: Llama 4&#39;s new license comes with several limitations:- Companies with more than 700 million monthly active users must request a special license from Meta, which Meta can grant or deny at its sole ...</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843?s=46">Tweet from Paul Gauthier (@paulgauthier)</a>: Llama 4 Maverick scored 16% on the aider polyglot coding benchmark.https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/osanseviero/status/1908841583895605319?s=46">Tweet from Omar Sanseviero (@osanseviero)</a>: Llama 4 is seriously an impressive model. The quality jump is 🤯That said, its use policy prohibits multimodal models (all of Llama 4 so far) being used if you are an individual or company in the EU �...</li><li><a href="https://fxtwitter.com/waseem_s/status/1908713762779017427">Tweet from Waseem AlShikh (@waseem_s)</a>: If I understand you correctly, I implemented a prototype of your iRoPE architecture! It interleaves local attention (with RoPE) and global attention (with inference-time temp scaling). Added FFNs, chu...</li><li><a href="https://x.com/fofrai/status/1908690632576819277?s=46">Tweet from fofr (@fofrAI)</a>: Llama 4 is up on Replicate- maverick (17b with 128 experts)- scout (17b with 16 experts)https://replicate.com/metaQuoting Replicate (@replicate) https://replicate.com/meta/llama-4-maverick-instruct</li><li><a href="https://fxtwitter.com/_mchenco/status/1908873033852338580">Tweet from michelle (@_mchenco)</a>: our workers ai team sprinted through saturday to get llama 4 up,learned a lot over the last 24h (and still learning) - want to see how we think about llama 4 from a provider’s perspective? 🧵</li><li><a href="https://fxtwitter.com/Ahmad_Al_Dahle/status/1909302532306092107">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: We&#39;re glad to start getting Llama 4 in all your hands. We&#39;re already hearing lots of great results people are getting with these models. That said, we&#39;re also hearing some reports of mixed...</li><li><a href="https://x.com/teortaxestex/status/1908706840554197309?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: What did Meta see coming out on Monday that they rushed?Quoting kalomaze (@kalomaze) no way</li><li><a href="https://x.com/thexeophon/status/1908900306580074741?s=46">Tweet from Xeophon (@TheXeophon)</a>: Llama 4 on LMsys is a totally different style than Llama 4 elsewhere, even if you use the recommended system prompt. Tried various prompts myselfMETA did not do a specific deployment / system prompt j...</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1908620422767358272">Tweet from Simo Ryu (@cloneofsimo)</a>: Who wouldve thought next generation attention-replacement, adopted by two latest SoTA models, llama4 and cohere&#39;s command A to achieve infinite context...is attention without RoPE yeah attention i...</li><li><a href="https://fxtwitter.com/fofrai/status/1908690632576819277">Tweet from fofr (@fofrAI)</a>: Llama 4 is up on Replicate- maverick (17b with 128 experts)- scout (17b with 16 experts)https://replicate.com/metaQuoting Replicate (@replicate) https://replicate.com/meta/llama-4-maverick-instruct</li><li><a href="https://fxtwitter.com/nrehiew_/status/1908598863365013823">Tweet from wh (@nrehiew_)</a>: No RoPE on global layers is the meta transformer arch nowQuoting wh (@nrehiew_) If i had to guess:- no PE on 1:4/1:8 global. Use MLA here or some other efficient attn variant - Standard SWA for the re...</li><li><a href="https://fxtwitter.com/tobi/status/1909251946235437514">Tweet from tobi lutke (@tobi)</a>: http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/ahmad_al_dahle/status/1908595680828154198?s=46">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Introducing our first set of Llama 4 models!We’ve been hard at work doing a complete re-design of the Llama series. I’m so excited to share it with the world today and mark another major milestone for...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908676312069255534">Tweet from kalomaze (@kalomaze)</a>: okay, most interesting thing about llama4 is NOT multimodality (which they are probably still forced to lobotomize the image output ability of), or the 10m context (which is still &#34;fake&#34; afaic...</li><li><a href="https://x.com/rishdotblog/status/1908917222308995422?s=46">Tweet from Rishabh Srivastava (@rishdotblog)</a>: Sigh. Underwhelmed by the Llama 4 models so far. Can’t justify any real use for them- too big for local use, qwen and Gemma models still the best option here - much worse than deepseek v3, sonnet, or ...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1909302532306092107">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: We&#39;re glad to start getting Llama 4 in all your hands. We&#39;re already hearing lots of great results people are getting with these models. That said, we&#39;re also hearing some reports of mixed...</li><li><a href="https://x.com/deedydas/status/1908749257084944847?s=46">Tweet from Deedy (@deedydas)</a>: Llama 4 seems to actually be a poor model for coding.Scout (109B) and Maverick (402B) underperform 4o, Gemini Flash, Grok 3, DeepSeek V3 and Sonnet 3.5/7 on the Kscores benchmark which tests on coding...</li><li><a href="https://x.com/cloneofsimo/status/1908620422767358272?s=46">Tweet from Simo Ryu (@cloneofsimo)</a>: Who wouldve thought next generation attention-replacement, adopted by two latest SoTA models, llama4 and cohere&#39;s command A to achieve infinite context...is attention without RoPE yeah attention i...</li><li><a href="https://x.com/suchenzang/status/1909070231517143509?s=46">Tweet from Susan Zhang (@suchenzang)</a>: &gt; Company leadership suggested blending test sets from various benchmarks during the post-training processIf this is actually true for Llama-4, I hope they remember to cite previous work from FAIR ...</li><li><a href="https://fxtwitter.com/scaling01/status/1908782484977770920">Tweet from Lisan al Gaib (@scaling01)</a>: It&#39;s literally over for Llama-4they are insane slop machines andthey abandoned small local models</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1908603318081138822">Tweet from Simo Ryu (@cloneofsimo)</a>: it looks like meta&#39;s new model&#39;s &#34;Key innovaton&#34; :  &#34;interleaved no-RoPE attention&#34; for infintie context, is actually the same thing cohere command-a model introduced few days ...</li><li><a href="https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism">Making Peace with LLM Non-determinism</a>: Digging into Sparse MoE and GPU cycles just to realize non-determinism is not new, language is.</li><li><a href="https://fxtwitter.com/scaling01/status/1908657167869100482">Tweet from Lisan al Gaib (@scaling01)</a>: Llmao-4 Scout (109B) and Maverick(400B) used less compute for training than Llama-3 8 and 70B</li><li><a href="https://x.com/_mchenco/status/1908873033852338580">Tweet from michelle (@_mchenco)</a>: our workers ai team sprinted through saturday to get llama 4 up,learned a lot over the last 24h (and still learning) - want to see how we think about llama 4 from a provider’s perspective? 🧵</li><li><a href="https://fxtwitter.com/burkov/status/1908658566887596475">Tweet from Andriy Burkov (@burkov)</a>: This means that this 10M token context is virtual. Kind of &#34;you can try to use it, but beyond 256K tokens, you are on your own,&#34; and even below 256K tokens, you are mostly on your own because ...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908695425286012959">Tweet from kalomaze (@kalomaze)</a>: it&#39;s overQuoting kalomaze (@kalomaze) the 400b llama4 model... sucks</li><li><a href="https://fxtwitter.com/kalomaze/status/1908681293006594176">Tweet from kalomaze (@kalomaze)</a>: the 400b llama4 model... sucks</li><li><a href="https://x.com/scaling01/status/1908782484977770920?s=46">Tweet from Lisan al Gaib (@scaling01)</a>: It&#39;s literally over for Llama-4they are insane slop machines andthey abandoned small local models</li><li><a href="https://fxtwitter.com/rauchg/status/1908605342201860311">Tweet from Guillermo Rauch (@rauchg)</a>: This is cool. Meta used the iconic Apache mod_autoindex style for the drop of Llama 4. But you can tell it’s not Apache due to the modern flexbox and responsive css 😁 Nice ode to the golden days wher...</li><li><a href="https://fxtwitter.com/natolambert/status/1908959159959027903">Tweet from Nathan Lambert (@natolambert)</a>: Seems like Llama 4’s reputation is maybe irreparably tarnished by having a separate unreleased model that was overfit to LMArena. Actual model is good, but shows again how crucial messaging and detail...</li><li><a href="https://fxtwitter.com/osanseviero/status/1908841583895605319">Tweet from Omar Sanseviero (@osanseviero)</a>: Llama 4 is seriously an impressive model. The quality jump is 🤯That said, its use policy prohibits multimodal models (all of Llama 4 so far) being used if you are an individual or company in the EU �...</li><li><a href="https://techcrunch.com/2025/04/07/kreas-founders-snubbed-postgrad-grants-from-the-king-of-spain-to-build-their-ai-startup-now-its-valued-at-500m/">Krea raises $83M to be the one-stop shop for GenAI creatives | TechCrunch</a>: Overwhelmed with trying to keep up with the different AI models you can use to make content? A startup called Krea is looking to solve this problem</li><li><a href="https://nixiesearch.substack.com/p/benchmarking-api-latency-of-embedding">Benchmarking API latency of embedding providers (and why you should always cache your embeddings)</a>: We measured the API latency of four major embedding providers—OpenAI, Cohere, Google, and Jina. We found is that the convenience of API integration can come at a cost if performance matters to you.</li><li><a href="https://irrationalanalysis.substack.com/p/tenstorrent-and-the-state-of-ai-hardware">Tenstorrent and the State of AI Hardware Startups</a>: Semi-custom silicon is a bigger problem than Nvidia.</li><li><a href="https://www.llama.com/llama-downloads/">Download Llama</a>: Request access to Llama.</li><li><a href="https://bsky.app/profile/ramon-astudillo.bsky.social/post/3lm3skzcfxk2i">Ramon Astudillo (@ramon-astudillo.bsky.social)</a>: I think this table was missing[contains quote post or other embedded content]</li><li><a href="https://hoxhunt.com/blog/ai-powered-phishing-vs-humans">AI-Powered Phishing Outperforms Elite Cybercriminals in 2025 - Hoxhunt</a>: Hoxhunt research proves AI agents can outperform elite red teams in phishing. Generative AI can be used in cybersecurity for good or evil. We can use AI spear phishing agents for defense.</li><li><a href="https://www.adaptivesecurity.com/">Adaptive Security</a>: Adaptive&#x27;s next-generation security training and simulations protect businesses from deepfakes, generative AI phishing, SMS attacks, voice phishing, and more emerging threats.</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://openrouter.ai/meta-llama">Meta Llama | OpenRouter</a>: Browse models from Meta Llama</li><li><a href="https://news.ycombinator.com/item?id=43595775">General overview below, as the pages don&#x27;t seem to be working well Llama 4 Model... | Hacker News</a>: no description found</li><li><a href="https://www.llama.com/llama4-reasoning-is-coming/">Llama 4 Reasoning</a>: Coming soon</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/Isvg17X5O9">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1357952602213585007)** (1 messages): 

> `Claude Plays Pokemon Hackathon` 


- **Claude Plays Pokemon Hackathon**: A user thanked another user for helping run the **Claude Plays Pokemon hackathon** [on YouTube](https://youtu.be/zBPc6Ims1Bc).
- **YouTube Stream of Hackathon**: The **Claude Plays Pokemon hackathon** was recorded and streamed on [YouTube](https://youtu.be/zBPc6Ims1Bc).


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1357807122246664398)** (255 messages🔥🔥): 

> `LLM Codegen Workflow, AI Code Editors, Cursor vs Windsurf, Context Management in AI Editors, Model Hot-Swapping` 


- **Harper's LLM Codegen Workflow Exposed**: Harper's blog post ([My LLM Codegen Workflow ATM](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)) details a process of brainstorming a spec, planning, and executing with **LLM codegen in discrete loops**.
   - The process involves tweaks based on conversations with friends like [Nikete](https://www.nikete.com/), [Kanno](https://nocruft.com/), [Obra](https://fsck.com/), [Kris](https://github.com/KristopherKubicki), and [Erik](https://thinks.lol/).
- **AI Code Editor Recommendations**: For those new to AI code editors, **Cursor** is the most commonly recommended starting point, particularly for users coming from VSCode, with **Windsurf** and **Cline** also being good options.
   - Experienced devs on **nvim or emacs** should stick with their current editor and AI plugins, while those wanting a new modal editor should try **Zed**.
- **Cursor and Windsurf Comparison**: Members are bouncing between **Cursor and Windsurf**, noting strengths and weaknesses of each.
   - Cursor is easy to start, has great tab-complete, whereas people are waiting for the new **token counts and context window details** feature in Cursor ([tweet](https://x.com/ryolu_/status/1907589821280956648)).
- **Context Management Concerns in Cursor**: Members are reporting Cursor's terrible context management issues, with a lack of visibility into what the editor is doing with the current context.
   - It may come down to a *skill issue* and the users are not meeting the tool in the middle.
- **One-Shot Codegen or bust**: Many in the channel expressed a desire for **one-shot codegen** where an entire program can be generated at once.
   - Failing that, documenting better and taking another shot may be the next best option and, if that fails, training the user is necessary.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/cgarciae88/status/1907457306947702925">Tweet from Cristian Garcia (@cgarciae88)</a>: omg... I told gemini 2.5 pro it was wrong and instead panic agreeing with me and hallucinating, it explained why it was me who was wrong</li><li><a href="https://fxtwitter.com/ryolu_/status/1907589821280956648">Tweet from Ryo Lu (@ryolu_)</a>: This one is for the Pros:Working on an easier way to fill MAX context in @cursor_ai—and show you exactly how many tokens are usedFeedback and ideas welcome 🙏</li><li><a href="https://x.com/ryolu_/status/1907589821280956648">Tweet from Ryo Lu (@ryolu_)</a>: This one is for the Pros:Working on an easier way to fill MAX context in @cursor_ai—and show you exactly how many tokens are usedFeedback and ideas welcome 🙏</li><li><a href="https://x.com/cgarciae88/status/1907457306947702925">Tweet from Cristian Garcia (@cgarciae88)</a>: omg... I told gemini 2.5 pro it was wrong and instead panic agreeing with me and hallucinating, it explained why it was me who was wrong</li><li><a href="https://www.npmjs.com/package/@johnlindquist/file-forge">@johnlindquist/file-forge</a>: File Forge is a powerful CLI tool for deep analysis of codebases, generating markdown reports to feed AI reasoning models.. Latest version: 2.13.6, last published: 9 hours ago. Start using @johnlindqu...</li><li><a href="https://github.com/yamadash">Yamadash - Overview</a>: GitHub is where Yamadash builds software.</li><li><a href="https://github.com/bodo-run/yek">GitHub - bodo-run/yek: A fast Rust based tool to serialize text-based files in a repository or directory for LLM consumption</a>: A fast Rust based tool to serialize text-based files in a repository or directory for LLM consumption - bodo-run/yek</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, DeepSeek, Perplexity, Gemini, Gemma, Llama, Grok, and more.</a>: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) o.....</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: A detailed walkthrough of my current workflow for using LLms to build software, from brainstorming through planning and execution.</li><li><a href="https://github.com/formal-land/coq-of-rust?tab=readme-ov-file">GitHub - formal-land/coq-of-rust: Formal verification tool for Rust: check 100% of execution cases of your programs 🦀 to make super safe applications! ✈️ 🚀 ⚕️ 🏦</a>: Formal verification tool for Rust: check 100% of execution cases of your programs 🦀 to make super safe applications! ✈️ 🚀 ⚕️ 🏦 - formal-land/coq-of-rust
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357829245635657873)** (308 messages🔥🔥): 

> `Open Source Cursor Alternatives, Prompt Injection / Jailbreaking Tactics, Llama 4 launch and performance, Neural Plasticity via Neural Graffiti` 


- **Cursor-like Apps Sought After**: Members were looking for [open source alternatives](https://continue.dev) to the **Cursor** app, specifically interested in how the accept/discard suggestions of code blocks work.
   - One member noted that **Cursor** uses a different model to *'apply'* the code once you say accept.
- **Unleashing Prompt Injection attacks**: A member inquired about bypassing prompt guards, detectors, and **NeMo guard rails** from a *pentest* perspective, linking to a prompt filter trainer ([gandalf.lakera.ai/baseline](https://gandalf.lakera.ai/baseline)).
   - They also linked to a [Broken LLM Integration App](https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App) which uses **UUID tags and strict boundaries**.
- **Llama 4 debuts with multimodal muscles**: **Meta** launched the **Llama 4** family, featuring **Llama 4 Scout** (17B active params, 16 experts, 10M+ context) and **Llama 4 Maverick** (17B active params, 128 experts, 1M+ context), along with a preview of **Llama 4 Behemoth**, and a peak at the iRoPE architecture for infinite context ([blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
   - Some members expressed skepticism about the benchmarking methodology, the real-world coding ability and performance of **Llama 4 Scout**.
- **Neural Graffiti Gives LLMs live modulations**: A member introduced "Neural Graffiti", a technique to give pre-trained LLMs some neuroplasticity by splicing in a new neuron layer that recalls memory, reshaping token prediction at generation time, sharing code and demo on [Github](https://github.com/babycommando/neuralgraffiti).
   - The live modulation takes a fused memory vector (from prior prompts), evolves it through a recurrent layer (the Spray Layer), and injects it into the model’s output logic at generation time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deedydas/status/1908749257084944847">Tweet from Deedy (@deedydas)</a>: Llama 4 seems to actually be a poor model for coding.Scout (109B) and Maverick (402B) underperform 4o, Gemini Flash, Grok 3, DeepSeek V3 and Sonnet 3.5/7 on the Kscores benchmark which tests on coding...</li><li><a href="https://x.com/ficlive/status/1908911992686931989?t=N1BGmubwXQQ-ZYLSKfmXqw&s=19">Tweet from Fiction.live (@ficlive)</a>: Updated Long context benchmark with Llama 4</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Introducing our first set of Llama 4 models!We’ve been hard at work doing a complete re-design of the Llama series. I’m so excited to share it with the world today and mark another major milestone for...</li><li><a href="https://x.com/nrehiew_/status/1908617547236208854?s=46">Tweet from wh (@nrehiew_)</a>: In the local attention blocks instead of sliding window, Llama4 uses this Chunked Attention. This is pretty interesting/weird:- token idx 8191 and 8192 cannot interact in local attention- the only way...</li><li><a href="https://x.com/JeffDean/status/1908608454216028222">Tweet from Jeff Dean (@JeffDean)</a>: @jeremyphoward Why can&#39;t you run them on consumer GPUs?</li><li><a href="https://x.com/astonzhangaz/status/1908595612372885832?s=46">Tweet from Aston Zhang (@astonzhangAZ)</a>: Our Llama 4’s industry leading 10M+ multimodal context length (20+ hours of video) has been a wild ride. The iRoPE architecture I’d been working on helped a bit with the long-term infinite context goa...</li><li><a href="https://x.com/agarwl_/status/1909292968139255849?t=RpSDb5rQDhI1cdf1ZXqZxw&s=19)">Tweet from Rishabh Agarwal (@agarwl_)</a>: Joined the Llama team @AIatMeta today to work on RL and reasoningQuoting AI at Meta (@AIatMeta) Today is the start of a new era of natively multimodal AI innovation.Today, we’re introducing the first ...</li><li><a href="https://x.com/kalomaze/status/1909267256564920611">Tweet from kalomaze (@kalomaze)</a>: turboderp is allergic to hyping up his work so let me do the honors.--THIS CHANGES EVERYTHING 🤯EXLLAMA DEVELOPER &#34;turboderp&#34; RELEASES EXLLAMA 3, with a NOVEL, STATE-OF-THE-ART local model qua...</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF">NousResearch/Hermes-3-Llama-3.1-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://gandalf.lakera.ai/baseline">Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.</a>: Trick Gandalf into revealing information and experience the limitations of large language models firsthand.</li><li><a href="https://www.arxiv.org/abs/2504.01990">Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems</a>: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust pe...</li><li><a href="https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md">llmbenchmark/raytracer/Readme.md at master · cpldcpu/llmbenchmark</a>: Various LLM Benchmarks. Contribute to cpldcpu/llmbenchmark development by creating an account on GitHub.</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://www.llama.com/llama4-reasoning-is-coming/">Llama 4 Reasoning</a>: Coming soon</li><li><a href="https://github.com/meta-llama/llama-models/tree/main/models/llama4">llama-models/models/llama4 at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://www.trae.ai/">Trae - Ship Faster with Trae</a>: Trae is an adaptive AI IDE that transforms how you work, collaborating with you to run faster.</li><li><a href="https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App">GitHub - 13o-bbr-bbq/Broken_LLM_Integration_App: This is the LLM integration app that contains the vulnerability; please use it to verify the vulnerability of the LLM integration app.</a>: This is the LLM integration app that contains the vulnerability; please use it to verify the vulnerability of the LLM integration app. - 13o-bbr-bbq/Broken_LLM_Integration_App</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12791">llama : Support llama 4 text-only by ngxson · Pull Request #12791 · ggml-org/llama.cpp</a>: Resolves #12774This PR targets Llama-4-Scout-17B-16E-Instruct. I don&amp;#39;t (yet?) have a powerful enough system to work with bigger model.But Son, you are GPU-poor, how can you test a model that ....</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct">meta-llama/Llama-4-Maverick-17B-128E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E">meta-llama/Llama-4-Scout-17B-16E · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E">meta-llama/Llama-4-Maverick-17B-128E · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357947681011667077)** (27 messages🔥): 

> `Claude Think Tool, Local LLM for 300 Pages of Text, Nous Capybara 34B Model, DeepHermes, BatchNorm and LayerNorm Implementations` 


- **Claude Think Tool: A Brainy Brainstormer**: The [Claude Think Tool](https://www.anthropic.com/engineering/claude-think-tool) is a setup to offload critical tasks to a larger model from a small local model.
   - It helps create multiple threads of thought, each with attention directed toward a specific domain and problem with a well-defined scope, functioning as a *multi-agent system from the perspective of the brain*.
- **Pondering the Perfect Local LLM for 300-Page Text Ingestion**: A member inquired about running a local LLM, around **40B** or less, capable of understanding around **300 pages** of pure text, given a **12GB** GPU and **32GB** of normal memory.
   - Suggestions included **DeepHermes**, **Cohere Command R 7B** and **Qwen 7B 1M**, with warnings that CPU inference might not be viable for such large documents.
- **Nous Capybara 34B: A Contextual Colossus**: The [Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B) is trained on the **Yi-34B** model with **200K** context length for **3 epochs** on the Capybara dataset.
   - It leverages a novel data synthesis technique called **Amplify-instruct**, combining top-performing existing data synthesis techniques and distributions used for SOTA models like Airoboros, Evol-Instruct, Orca, Vicuna, and others.
- **BatchNorm Backpropagation: A Numerical Nirvana**: A member shared a raw implementation of **BatchNorm** using NumPy, emphasizing the backward pass as the most intimidating part due to computing the gradient of pre-normalized input following the multivariate chain rule, illustrated [here](https://cdn.discordapp.com/attachments/1154120232051408927/1358871065119686746/image.png).
   - They followed it up with implementing **LayerNorm**, [highlighting the key difference](https://cdn.discordapp.com/attachments/1154120232051408927/1358919204270641342/image.png) being that statistics are computed per sample rather than per batch.



**Link mentioned**: <a href="https://huggingface.co/NousResearch/Nous-Capybara-34B">NousResearch/Nous-Capybara-34B · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Reinforcement Learning for LLMs, Reward Modeling Improvements, Self-Principled Critique Tuning` 


- **Deepseek releases Reinforcement Learning Paper**: Deepseek released a new paper on **Reinforcement Learning (RL)** being widely adopted in post-training for large language models (**LLMs**) at scale; the paper can be found [here](https://arxiv.org/abs/2504.02495).
   - The paper investigates how to improve reward modeling (**RM**) with more inference compute for general queries, i.e. the *inference-time scalability of generalist RM*, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods.
- **Self-Principled Critique Tuning Proposed**: Deepseek adopts pointwise generative reward modeling (**GRM**) to enable flexibility for different input types and potential for inference-time scaling.
   - The paper proposes **Self-Principled Critique Tuning (SPCT)** to foster scalability.



**Link mentioned**: <a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357794753265271069)** (9 messages🔥): 

> `Claude Squad, Heterogeneous Recursive Planning, Panthalia Decentralized Compute, TextPulse Library` 


- **Claude Squad Manages Multiple Agents**: [Claude Squad](https://github.com/smtg-ai/claude-squad) is a free and open-source manager for **Claude Code & Aider tasks** that supervises multiple agents in one place with isolated git workspaces.
   - It enables users to run **ten Claude Codes in parallel**.
- **Heterogeneous Recursive Planning for Creative AI**: A new method called heterogeneous recursive planning enables **AI** to write creative stories and insightful deep research reports like an expert ([paper](http://arxiv.org/abs/2503.08275), [demo](http://writehere.site)).
   - It leverages adaptive subgoals and dynamic execution, allowing agents to dynamically replan and weave retrieval, reasoning, and composition mid-flow, based on [previous work](https://www.google.com/url?sa=D&q=https://people.idsia.ch/~juergen/recursiveplanning.html).
- **Panthalia Verifies Low-Cost Distributed Compute**: [Panthalia](https://x.com/panthaliaxyz/status/1909342585505669228) is a platform to safely and easily train **ML models** on peer-to-peer compute using a decentralized compute primitive, using a waitlist.
   - The platform uses a compression algorithm heavily inspired by the [Nous DeMo paper](https://docs.panthalia.com/gradient-compression-algorithm) and the [related codebase](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py).
- **TextPulse Library for Text Processing**: A member shared their library [TextPulse](https://github.com/jfinst1/TextPulse) for text processing and is looking for feedback.
   - Currently, they resell low-cost providers aiming for the same interruptible prices (**~$0.60/hr for an H100, ~$0.13/hr for a 4090**).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/moofeez/status/1907893901077196861?s=46">Tweet from mufeez (@moofeez)</a>: Why settle for one Claude Code when you can run ten in parallel?We built Claude Squad — a manager for Claude Code & Aider tasks:• Supervise multiple agents in one place• Isolated git workspacesFree + ...</li><li><a href="https://x.com/SchmidhuberAI/status/1908172744409403793">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: What if AI could write creative stories & insightful #DeepResearch reports like an expert? Our heterogeneous recursive planning [1] enables this via adaptive subgoals [2] & dynamic execution. Agents d...</li><li><a href="https://x.com/panthaliaxyz/status/1909342585505669228">Tweet from Panthalia (@panthaliaxyz)</a>: Panthalia: Decentralized compute primitive  The platform to safely and easily train ML models on peer-to-peer computeWaitlist now available</li><li><a href="https://docs.panthalia.com/gradient-compression-algorithm">Panthalia Gradient Compression Algorithm | Panthalia</a>: This document provides a detailed description of the DCT-based gradient compression algorithm used in Panthalia. The algorithm is designed to efficiently compress both gradients sent from nodes to a c...</li><li><a href="https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py">panthalia-worker/spl/util/demo.py at main · ritser-labs/panthalia-worker</a>: Contribute to ritser-labs/panthalia-worker development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Deepseek, Reinforcement Learning, Large Language Models, Reward Modeling, Self-Principled Critique Tuning` 


- **Deepseek's New Paper on RL for LLMs**: Deepseek released a new paper, available on [arXiv](https://arxiv.org/abs/2504.02495), about **Reinforcement Learning (RL)** adoption in post-training for **Large Language Models (LLMs)** at scale.
   - The paper investigates improving **reward modeling (RM)** with more inference compute for general queries and the effectiveness of performance-compute scaling with proper learning methods, proposing **Self-Principled Critique Tuning (SPCT)**.
- **SPCT Improves Reward Modeling**: The paper introduces **Self-Principled Critique Tuning (SPCT)** as a method to enhance the effectiveness of performance-compute scaling in reward modeling for LLMs.
   - This approach aims to foster scalability by improving reward model inference compute for general queries beyond verifiable questions or artificial rules.



**Link mentioned**: <a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...

  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1358810800252391476)** (6 messages): 

> `Reasoning Benchmarking, Open Reasoning Tasks` 


- **Researcher transitions to LLM World**: A researcher on logic and reasoning is considering moving into the LLM world and wants to contribute to reasoning categorisation and benchmarking.
   - A member suggested checking out the [reasoning tasks repo](https://github.com/NousResearch/Open-Reasoning-Tasks).
- **Discussion about Open Reasoning Tasks**: A member is exploring the list of reasoning tasks to benchmark one of the LLMs and asks about the taxonomy behind it, its background, and related literature.
   - They specifically inquired about who is behind the taxonomy and its history.



**Link mentioned**: <a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - NousResearch/Open-Reasoning-Tasks

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357795938168602715)** (293 messages🔥🔥): 

> `MCP Governance SDK, MCP Protocol Revision 2025, MCP Desktop Workflow Integrations, Pinging MCP Servers Before Initialization, MCP Server for Microsoft Loop` 


- **Auth0 Token Validation with MCP Governance SDK**: A guide focuses on server-side implementation using the governance SDK to validate tokens (e.g., from **Auth0**) and enforce user roles and permissions on MCP operations, deciding access to tools or resources.
   - The guide picks up after the client sends a token, detailing how the server can validate the token and fetch user's roles, using the SDK's RBAC system to enforce permissions.
- **Streamable HTTP Transport for MCP**: The [Model Context Protocol (MCP) specification](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http) uses **JSON-RPC** to encode messages, mandating UTF-8 encoding and defining two transport mechanisms: stdio and Streamable HTTP.
   - Clients *should* support stdio, but custom transports are also possible, as outlined in the specification, which includes requirements like newline delimiters for messages in stdio.
- **Llama 4 Released, Still Doesn't Know MCP**: **Llama 4** has been released with [17B parameters](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) and outperforms deepseekv3, but still does not know what MCP is, despite its impressive capabilities.
   - It's a **17B MoE**, with **109B total parameters**, according to an announcement.
- **MCP Tool Installs Should be Standardized**: Members discussed the need for more standardization around **MCP server** installation, similar to *scoop* or VS Code extensions, to improve accessibility for non-technical users.
   - The discussion highlighted the friction in the current process, involving command-line arguments, environment variables, and varying install methods (Python, Node.js, Docker) with a suggestion to make it as easy as *python-mcp install web-search*.
- **A Holy War? OAuth-Backed APIs MCPs are the key**: Members debated over the security of MCPs with some feeling that they need an *app store* with oversight to check for hacked servers and OAuth-backed APIs, while others claim that can already be done.
   - One proposal is for providers like PayPal to host their own OAuth-backed APIs that don't require external server install.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@studymyvisualsco/unlock-effortless-ai-automation-best-way-to-self-host-n8n-is-railway-integrate-firecrawl-mcp-7964019c6c28">Unlock Effortless AI Automation: Best Way To Self-Host n8n is Railway &amp; Integrate Firecrawl MCP…</a>: Unlock the power of AI-driven Web automation without touching a Line of Code!</li><li><a href="https://x.com/MCP_Community">Tweet from undefined</a>: no description found</li><li><a href="https://gitmcp.io/docs">GitMCP</a>: Instantly create an MCP server for any GitHub project</li><li><a href="https://developers.cloudflare.com/agents/guides/remote-mcp-server/">Build a Remote MCP server · Cloudflare Agents docs</a>: This guide will walk you through how to deploy an example MCP server to your Cloudflare account. You will then customize this example to suit your needs.</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http>?">Transports</a>:           ℹ️                  Protocol Revision: 2025-03-26      MCP uses JSON-RPC to encode messages. JSON-RPC messages MUST be UTF-8 encoded.The protocol currently defines two standard transport mec...</li><li><a href="https://glama.ai/mcp/servers/@611711Dark/mcp_calculate_server">MCP Calculate Server</a>: A mathematical computation service that enables users to perform symbolic calculations including basic arithmetic, algebra, calculus, equation solving, and matrix operations through the MCP protocol.</li><li><a href="https://learn.microsoft.com/en-us/microsoft-copilot-studio/agent-extend-action-mcp">Extend your agent with Model Context Protocol (preview) - Microsoft Copilot Studio</a>: Extend the capabilities of your agent by connecting to actions from a Model Context Protocol (MCP) server.</li><li><a href="https://www.reddit.com/r/mcp/comments/1jtgug1/dis">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://github.com/jaw9c/awesome-remote-mcp-servers">GitHub - jaw9c/awesome-remote-mcp-servers: A curated, opinionated list of high-quality remote Model Context Protocol (MCP) servers.</a>: A curated, opinionated list of high-quality remote Model Context Protocol (MCP) servers.  - GitHub - jaw9c/awesome-remote-mcp-servers: A curated, opinionated list of high-quality remote Model Conte...</li><li><a href="https://github.com/EnactProtocol">Enact Protocol</a>: Enact Protocol has 3 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md">specification/docs/specification/2025-03-26/basic/lifecycle.md at main · modelcontextprotocol/specification</a>: The specification of the Model Context Protocol. Contribute to modelcontextprotocol/specification development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/mcp/comments/1jtgug1/discussion_unified_tool_registry_for_ai_agents/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://aistudio.google.com/app/prompts/new_chat">no title found</a>: no description found</li><li><a href="https://github.com/EnactProtocol/enact-mcp/blob/0e155b5d52c340b14de0a3f7804aec0c2456ff36/src/index.ts#L93">enact-mcp/src/index.ts at 0e155b5d52c340b14de0a3f7804aec0c2456ff36 · EnactProtocol/enact-mcp</a>: MCP Server for enact protocol. Contribute to EnactProtocol/enact-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/semgrep/mcp#hosted-server">GitHub - semgrep/mcp: A MCP server for using Semgrep to scan code for security vulnerabilities.</a>: A MCP server for using Semgrep to scan code for security vulnerabilities. - semgrep/mcp</li><li><a href="https://glama.ai/mcp/reference">MCP API Reference</a>: API Reference for the Glama Gateway</li><li><a href="https://www.pulsemcp.com/api">REST API Docs | PulseMCP</a>: Programmatic access to daily-updated JSON of all MCP server metadata, scraped to be comprehensive and filtered to be useful.</li><li><a href="https://www.reddit.com/r/mcp/comments/1jrq4o8/how_do_we_improve_the_distribution_of_mcp_servers/?rdt=35027">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357830147528593513)** (23 messages🔥): 

> `MCP-k8s Docker Images, chat.md with MCP support, Cloudflare for Remote MCP Servers, WhatsMCP Oauth Support, Semgrep MCP Rewrite` 


- ****MCP-k8s** Docker Images Published**: First working [**docker images**](https://hub.docker.com/r/mcpk8s/server) published for **mcp-k8s server** are now available, and the release pipeline is completely running on CI.
   - These images are **multiarch**, so they can run on **Macs** with **ARM** without **Rosetta** and also on **Raspberry Pi**.
- ****Chat.md**: fully editable chat interface with MCP support**: A fully editable chat interface with **MCP support** on any **LLM** has been released, open-sourced under the **MIT license** and turning markdown files into editable AI conversations with its **VS Code extension** ([chat.md](https://github.com/rusiaaman/chat.md)).
   - Notable features include editing past messages, **LLM agnostic MCP support**, streaming responses with **shift+enter**, and tool call detection.
- **Cloudflare Enables Remote MCP Servers**: It is now possible to [build and deploy remote **MCP servers** to **Cloudflare**](https://developers.cloudflare.com/agents/guides/remote-mcp-server/), with added support for **OAuth** through **workers-oauth-provider** and a built-in **McpAgent** class.
   - This simplifies the process of building remote **MCP servers** by handling authorization and other complex aspects.
- **WhatsApp MCP client is here**: A user built WhatsApp MCP and asked **Claude** to handle all the WhatsApp messages, answering 8 people in approx. **50 seconds**.
   - The bot instantly detected the right language (**English / Hungarian**), used full convo context, and sent appropriate messages including *❤️ to my wife, formal tone to the consul*.
- **Semgrep MCP Server Rewritten**: The [Semgrep MCP server](https://github.com/semgrep/mcp), an **open-source** tool for scanning code for security vulnerabilities, has been completely rewritten, with demo videos showcasing its use in **Cursor** and **Claude**.
   - It uses **SSE** (Server-Sent Events) for communication, though the Python SDK might not fully support it yet.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vargastartup/status/1907904839448715657">Tweet from Alex Varga (@vargastartup)</a>: I asked Claude to handle all my WhatsApp messages. One prompt. That’s it.1. It answered 8 people in approx. 50 seconds2. Instantly detected the right language (English 🇺🇸 / Hungarian 🇭🇺)3. Used fu...</li><li><a href="https://wassist.app/mcp">WhatsApp MCP Client | Connect Your AI Stack</a>: Connect your MCP server to power your AI stack through WhatsApp. Secure, private, and easy to use.</li><li><a href="https://blog.cloudflare.com/remote-model-context-protocol-servers-mcp/">Build and deploy Remote Model Context Protocol (MCP) servers to Cloudflare</a>: You can now build and deploy remote MCP servers to Cloudflare, and we handle the hard parts of building remote MCP servers for you. Unlike local MCP servers you may have previously used, remote MCP se...</li><li><a href="https://github.com/semgrep/mcp">GitHub - semgrep/mcp: A MCP server for using Semgrep to scan code for security vulnerabilities.</a>: A MCP server for using Semgrep to scan code for security vulnerabilities. - semgrep/mcp</li><li><a href="https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df">Semgrep MCP Demo</a>: Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.</li><li><a href="https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411">Claude Desktop Using Semgrep MCP Resources</a>: Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.</li><li><a href="https://github.com/SDCalvo/MCP-to-Langchain-addapter">GitHub - SDCalvo/MCP-to-Langchain-addapter: Addapter that turns MCP server tools into langchain usable tools</a>: Addapter that turns MCP server tools into langchain usable tools - SDCalvo/MCP-to-Langchain-addapter</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/416">Streamable HTTP client transport by josh-newman · Pull Request #416 · modelcontextprotocol/python-sdk</a>: Client implementation of spec version 2025-03-26&amp;#39;s new transport.Motivation and ContextThe 2025-03-26 spec introduces a new HTTP transport mechanism (with fallback to the previous one). I did....</li><li><a href="https://github.com/rusiaaman/chat.md">GitHub - rusiaaman/chat.md</a>: Contribute to rusiaaman/chat.md development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357847563901993020)** (39 messages🔥): 

> `RAG evaluation with lm-evaluation-harness, RoR-Bench paper by the_alt_man, Llama 4 release, Aligning AGI using Bayesian Updating` 


- **RAG Evaluation using LLM Harness?**: A member suggested wrapping **RAG outputs as completion tasks** and using **llm-harness locally** with custom prompt + response files for evaluation.
   - Another member admitted to having *no idea what those are*.
- **LLMs Exhibiting Recitation Behavior?**: A member shared a link to the [RoR-Bench paper](https://arxiv.org/abs/2504.00509) which proposes a novel, multi-modal benchmark for detecting LLM's recitation behavior, finding that top models can suffer a **60% performance loss** by changing one phrase in the condition.
   - The member expressed suspicion of these papers because they found that models that were evaluated at 0% on certain reasoning tasks could actually one-shot it.
- **Llama 4 Unleashed**: A link to the **Llama 4 release** was shared ([https://www.llama.com/llama4/](https://www.llama.com/llama4/)), showcasing the most intelligent multimodal OSS model in its class, with Llama4 Maverick > Gemma3 and Llama4 Maverick > DeepSeek V3.
   - Another member noted the training process, architecture, and inference time temperature scaling.
- **Aligning AGI with Moral Weights**: A member shared a [Google Doc](https://docs.google.com/document/d/1j11OUXWtS6yLAzXsbSo4lrzoqtiZ5RM0ykilLEvpYwM/edit) about aligning **AGI** using **Bayesian Updating** of its **Moral Weights** and **Modelling Consciousness**.
   - Another member shared a link to Arweave that discusses AI's role in preserving human consciousness. ([https://arweave.net/q6CszfPrxFZfm-BiVsvtiOXWuDkcYo8Pf9viDqv-Nhg](https://arweave.net/q6CszfPrxFZfm-BiVsvtiOXWuDkcYo8Pf9viDqv-Nhg))


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2504.00509">Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?</a>: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surp...</li><li><a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.19466">The Impact of Positional Encoding on Length Generalization in Transformers</a>: Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding ...</li><li><a href="https://docs.google.com/document/d/1j11OUXWtS6yLAzXsbSo4lrzoqtiZ5RM0ykilLEvpYwM/edit">ALBUM-WMC: Aligning AGI Using Bayesian Updating of its Moral Weights &amp; Modelling Consciousness</a>: (feel free to leave a comment here to say “I was here!”) ALBUM-WMC: Aligning lAGI Using Bayesian Updating of its Moral Weights &amp; Modelling Consciousness This document outlines a set of related id...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357808468794277966)** (204 messages🔥🔥): 

> `Mixture of Experts, Large Language Models, Gradient-Free Learning Methods, Hyper-connections as alternative to residual connections, Attention Sinks in LLMs` 


- ****MoE++ Framework Achieves Expert Throughput****: A new **MoE++** framework integrates **Feed-Forward Network (FFN)** and zero-computation experts (**zero expert, copy expert, and constant expert**) for enhanced effectiveness and efficiency, achieving **1.1$\sim$2.1$\times$ expert forward throughput** compared to vanilla **MoE** models, according to [this research paper](https://openreview.net/forum?id=t7P5BUKcYv).
   - The design of **MoE++** offers advantages such as *Low Computing Overhead* by enabling dynamic token engagement, unlike uniform mixing in vanilla MoE.
- ****NoProp Offers Gradient-Free Learning****: A new learning method named **NoProp**, which does not rely on either forward or backwards propagation and takes inspiration from diffusion and flow matching methods, learns to denoise a noisy target at each layer independently, described in [this paper](https://arxiv.org/abs/2503.24322).
   - There's a [GitHub implementation by lucidrains](https://github.com/lucidrains/hyper-connections) and also a discussion that *the pseudocode at the end of the paper says they're effecting the actual updates using gradient based methods.*
- ****Meta releases Llama 4****: Meta announced the **Llama 4** family of models, including **Llama 4 Scout**, a **17 billion** parameter model with **16 experts** and a **10M token context window**, outperforming **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1** in its class, as noted in [this blog post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
   - Llama 4 Scout's **10M context** is trained on a mix of publicly available, licensed data, and information from Meta’s products and services including posts from **Instagram** and **Facebook** and people’s interactions with **Meta AI**.
- ****Hyper-Connections Offer Alternative to Residual Connections****: **Hyper-connections**, serve as an alternative to residual connections, addressing the seesaw effect between gradient vanishing and representation collapse, as outlined in [this paper](https://arxiv.org/abs/2409.19606).
   - The architecture is simple like an unrolled diffusion model and the *magic here is more about the independence of each layer wrt each other*.
- ****Attention Sinks in LLMs Prevent Over-Mixing****: A recent paper argues that **attention sinks**, where LLMs attend heavily to the first token in the sequence, is a mechanism that enables LLMs to avoid over-mixing, detailed in [this paper](https://arxiv.org/abs/2504.02732).
   - An earlier paper ([https://arxiv.org/abs/2502.00919](https://arxiv.org/abs/2502.00919)) showed that *attention sinks utilize outlier features to: catch a sequence of tokens, tag the captured tokens by applying a common perturbation, and then release the tokens back into the residual stream, where the tagged tokens are eventually retrieved*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.19606">Hyper-Connections</a>: We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections. This approach specifically addresses common drawbacks observed in residual connect...</li><li><a href="https://arxiv.org/abs/2503.24322">NoProp: Training Neural Networks without Back-propagation or Forward-propagation</a>: The canonical deep learning approach for learning requires computing a gradient term at each layer by back-propagating the error signal from the output towards each learnable parameter. Given the stac...</li><li><a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: We propose a novel neural network architecture, the normalized Transformer (nGPT) with representation learning on the hypersphere. In nGPT, all vectors forming the embeddings, MLP, attention matrices ...</li><li><a href="https://openreview.net/forum?id=t7P5BUKcYv">MoE++: Accelerating Mixture-of-Experts Methods with...</a>: In this work, we aim to simultaneously enhance the effectiveness and efficiency of Mixture-of-Experts (MoE) methods. To achieve this, we propose MoE++, a general and heterogeneous MoE framework...</li><li><a href="https://arxiv.org/abs/2504.02732">Why do LLMs attend to the first token?</a>: Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways...</li><li><a href="https://arxiv.org/abs/2503.05453">Soft Policy Optimization: Online Off-Policy RL for Sequence Models</a>: RL-based post-training of language models is almost exclusively done using on-policy methods such as PPO. These methods cannot learn from arbitrary sequences such as those produced earlier in training...</li><li><a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: no description found</li><li><a href="https://x.com/BlinkDL_AI/status/1909280712567787947">Tweet from BlinkDL (@BlinkDL_AI)</a>: https://arxiv.org/abs/2503.24322 I think the NoProp method might be applicable to LLM training too, as each LLM block is denoising the next token distribution. So we can try training all blocks in par...</li><li><a href="https://arxiv.org/abs/2502.00919">Attention Sinks and Outlier Features: A &#39;Catch, Tag, and Release&#39; Mechanism for Embeddings</a>: Two prominent features of large language models (LLMs) is the presence of large-norm (outlier) features and the tendency for tokens to attend very strongly to a select few tokens. Despite often having...</li><li><a href="https://arxiv.org/abs/2504.01990">Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems</a>: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust pe...</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://github.com/lucidrains/hyper-connections">GitHub - lucidrains/hyper-connections: Attempt to make multiple residual streams from Bytedance&#39;s Hyper-Connections paper accessible to the public</a>: Attempt to make multiple residual streams from Bytedance&#39;s Hyper-Connections paper accessible to the public - lucidrains/hyper-connections
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1358672878937178192)** (17 messages🔥): 

> `Polytope lens for NNs, ReLU networks geometry, Machine Unlearning Workshop, Origami view of NNs, Expressivity of Deep Networks` 


- ****Polytope Perspective** Powers Neural Net Pondering**: A member shared a [blog post](https://addxorrol.blogspot.com/2025/04/some-experiments-to-help-me-understand.html) discussing a geometrical approach to neural networks, advocating for the **polytope lens** as the right perspective, linking to a [previous post](https://addxorrol.blogspot.com/2024/07/some-experiments-to-help-me-understand.html) on the *"origami view of NNs".*
- ****ReLU Network Regions** Reveal Reason**: A member shared [Boris Hanin's paper](https://arxiv.org/abs/1906.00904) demonstrating mathematical properties of **ReLU networks**, specifically studying the geometry of their constant regions.
   - They highlighted a figure from the paper as their *"main reason for loving the paper,"* referencing the expressivity of deep networks and the **number of activation patterns**.
- ****Hyperplane Harmony**: Neural Nets' Natural Nuance**: A member posited that neural nets, especially **ReLUs**, have an implicit bias against overfitting due to carving the input space along hyperplanes, which becomes more effective in higher dimensions.
   - They argued that simpler configurations using hyperplanes efficiently are preferred by the optimizer, contrasting with learning schemes like spline bases that suffer from the curse of dimensionality.
- ****Unlearning Urgency**: Machine Mind Management**: A member linked to the [ICML Machine Unlearning Workshop](https://mugenworkshop.github.io/) which focuses on the challenges of removing sensitive data from **Generative AI models** trained on internet-scale datasets.
   - The workshop aims to advance robust, verifiable unlearning methods to address privacy, security, and legal concerns like the **EU's GDPR**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1906.00904">Deep ReLU Networks Have Surprisingly Few Activation Patterns</a>: The success of deep networks has been attributed in part to their expressivity: per parameter, deep networks can approximate a richer class of functions than shallow networks. In ReLU networks, the nu...</li><li><a href="https://mugenworkshop.github.io/">MUGen @ ICML 2025 - Workshop on Machine Unlearning for Generative AI</a>: no description found</li><li><a href="https://addxorrol.blogspot.com/2025/04/some-experiments-to-help-me-understand.html">ADD / XOR / ROL: Some experiments to help me understand Neural Nets better, post 2 of N</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1357837368907923466)** (19 messages🔥): 

> `lm-eval-harness EOS token, Llama 2 vs Llama 3 IFEval Score, Huggingface tokenization` 


- **EOS token Accuracy Anomaly Appears**: A member tried adding an EOS token to data instances in **lm-eval-harness** for the *social_iqa* task, and the eval accuracy dropped by **18 points**.
   - It was suggested to add `self.eot_token_id` to the `continuation_enc` [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364) only for the continuations and not context.
- **IFEval Score: Llama 2's Odd Dominance**: A member compared **Llama 2** v/s **Llama 3.1** and **3.2** models and noticed **Llama 2** has a much higher **IFEval Score**, which seemed weird for a base model looking at the [HF leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?pinned=meta-llama%2FLlama-2-7b-hf_float16_01c7f73d771dfac7d292323805ebc428287df4f9_False%2Cmeta-llama%2FLlama-3.1-8B_float16_d04e592bb4f6aa9cfee91e2e20afa771667e1d4b_False%2Cmeta-llama%2FLlama-3.2-1B_bfloat16_a7c18587d7f473bfea02aa5639aa349403307b54_False%2Cmeta-llama%2FLlama-3.2-3B_bfloat16_95c102307f55fbd6d18ddf28bfbcb537ffdc2806_False).
   - It turns out it just seems to be unsuitable for base models because *models simply continue with the question and somehow it's considered correct*.
- **Huggingface Tokenization Troubleshoot**: Members discussed Huggingface tokenization, and how it happens in **HFLM.tok_encode**.
   - One noted that for BOS you can pass `add_bos_token` to the model args.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?pinned=meta-llama%2FLlama-2-7b-hf_float16_01c7f73d771dfac7d292323805ebc428287df4f9_False%2Cmeta-llama%2FLlama-3.1-8B_float16_d04e592bb4f6aa9cfee91e2e20afa771667e1d4b_False%2Cmeta-llama%2FLlama-3.2-1B_bfloat16_a7c18587d7f473bfea02aa5639aa349403307b54_False%2Cmeta-llama%2FLlama-3.2-3B_bfloat16_95c102307f55fbd6d18ddf28bfbcb537ffdc2806_False">Open LLM Leaderboard - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364)">lm-evaluation-harness/lm_eval/api/model.py at 11ac352d5f670fa14bbce00e423cff6ff63ff048 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1358759336612200508)** (1 messages): 

> `huggingface_hub v0.30.0, monoELECTRA reranker models, YourBench Custom Evals, Jetson Robot, Accelerate v1.6.0` 


- **Huggingface Hub gets Biggest Update Ever!**: The [huggingface_hub v0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) release introduces a next-gen **Git LFS alternative** and new **inference providers**.
   - This release is the *biggest update in two years!*
- **MonoELECTRA Rerankers Ported to Sentence Transformers**: **monoELECTRA-{base, large} reranker models** from @fschlatt1 & the research network Webis Group are now available in [Sentence Transformers](https://x.com/tomaarsen/status/1906652865675862125).
   - These models were distilled from **LLMs** like **RankZephyr** and **RankGPT4**, as described in the **Rank-DistiLLM paper**.
- **YourBench builds Custom Evals Instantly**: **YourBench** allows users to build **custom evals** using their **private docs** to assess fine-tuned models on unique tasks ([announcement](https://x.com/nathanhabib1011/status/1907728631167902067)).
   - The tool is *game-changing for LLM evaluation*.
- **Gradio Surpasses 1 Million Developers!**: **Gradio**, a Python library for building AI web apps, is now used by over **1 million developers** each month ([announcement](https://x.com/abidlabs/status/1907886482150580381)).
   - The library has been adopted by popular open-source projects like **Automatic1111**, **Fooocus**, and **LLaMA-Factory**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0">Release Xet is here!  (+ many cool Inference-related things!) · huggingface/huggingface_hub</a>: 🚀 Ready. Xet. Go!This might just be our biggest update in the past two years! Xet is a groundbreaking new protocol for storing large objects in Git repositories, designed to replace Git LFS. Unlik...</li><li><a href="https://x.com/tomaarsen/status/1906652865675862125">Tweet from tomaarsen (@tomaarsen)</a>: I&#39;ve just ported the excellent monoELECTRA-{base, large} reranker models from @fschlatt1 & the research network Webis Group to Sentence Transformers!These models were introduced in the Rank-DistiL...</li><li><a href="https://x.com/nathanhabib1011/status/1907728631167902067">Tweet from Nathan (@nathanhabib1011)</a>: 🚀 Introducing ✨ YourBench ✨ ! Build custom evals instantly using your private docs & see how your custom fine-tuned models perform on your unique tasks.Congrats to @sumukx @clefourrier and @ailozovsk...</li><li><a href="https://x.com/RemiCadene/status/1907689862930833545">Tweet from Remi Cadene (@RemiCadene)</a>: Jetson @nvidia&#39;s version of our robot is available!Compute is now on-board like a @Tesla car with FSD 🚗Importantly, we rethink the control interface, so that you can view the video stream with th...</li><li><a href="https://x.com/_marcsun/status/1907070902455685298">Tweet from Marc Sun (@_marcsun)</a>: accelerate v1.6.0 is out with lots of nice features ! - FSDPv2 support by @m_sirovatka, our incredible intern ! - DeepSpeed + tensor parallel support by the DeepSpeed team- XCCL distributed backend fo...</li><li><a href="https://x.com/hmellor_/status/1906665949530366169">Tweet from Harry Mellor (@hmellor_)</a>: The @vllm_project now has a user forum which you can find at https://discuss.vllm.ai/Its fledgling community is still growing but I encourage all users to go there for their usage focused Q&A!</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1joy1g9/you_can_now_check_if_your_laptop_rig_can_run_a/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://x.com/orr_zohar/status/1907526778278859205">Tweet from Orr Zohar (@orr_zohar)</a>: Excited to see SmolVLM powering BMC-SmolVLM in the latest BIOMEDICA update! At just 2.2B params, it matches 7-13B biomedical VLMs. Check out the full release: @huggingface #smolvlmQuoting Alejandro Lo...</li><li><a href="https://x.com/UnslothAI/status/1906726176556712318">Tweet from Unsloth AI (@UnslothAI)</a>: We partnered with @HuggingFace to teach you how to fine-tune LLMs with GRPO!Learn about:• Reward functions + creating them• GRPO Math + Free Reasoning training in Colab• Applying RL to real-world use ...</li><li><a href="https://x.com/_akhaliq/status/1907502083231670300">Tweet from AK (@_akhaliq)</a>: vibe coding AI apps for free has never been easier100% open source app, DeepSite on Hugging Face</li><li><a href="https://x.com/ben_burtenshaw/status/1907798840410808518">Tweet from Ben Burtenshaw (@ben_burtenshaw)</a>: Welcome to the LLM Course!Education has always been at the heart of Hugging Face’s mission to democratize AI and we’re doubling down on that by giving http://hf.co/learn a big upgrade!</li><li><a href="https://x.com/SergioPaniego/status/1907095475292897765">Tweet from Sergio Paniego (@SergioPaniego)</a>: 🆕New Unit in the Agents Course @huggingface. We just released the first Use Case on Agentic RAG—where we compare three frameworks side by side:🤏 smolagents🦙 @llama_index🦜 LangGraph (@LangChainAI)⬇...</li><li><a href="https://x.com/abidlabs/status/1907886482150580381">Tweet from Abubakar Abid (@abidlabs)</a>: JOURNEY TO 1 MILLION DEVELOPERS5 years ago, we launched @Gradio as a simple Python library to let researchers at Stanford easily demo computer vision models with a web interface. Today, Gradio is used...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357802672844963912)** (169 messages🔥🔥): 

> `Llama-4-Scout vs Mistral Small 3.1, AI Engineer Interview, Deepmind created AGI Internally?, Fine Tuning Quantized Models, Huggingchat 500 error` 


- **Llama-4-Scout or Mistral Small 3.1 is better?**: **Mistral Small 3.1** *adds vision understanding* and enhances context up to **128k tokens**.
   - A member suggested **Llama-4-Scout** is better, but it needs **16*17B VRAM**.
- **AI Engineer Interview Code Section**: A community member asks about what the code portion of an **AI engineer interview** looks like.
   - Another member pointed to the **scikit-learn library**.
- **Rumors of Deepmind created AGI Internally**: A member in another discord said **Google** will release yet another powerful model next week and *it will be even better than gemini 2.5 pro exp*.
   - They also claimed that **Deepmind** created an **AGI** internally; however, this member later stated he doesn't trust this person anymore.
- **Is Fine Tuning Quantized Models challenging?**: A member asked about fine tuning quantized models, and the community gave varied advice, with some pointing to **QLoRA, Unsloth, bitsandbytes** as potential solutions. Check out [Unsloth fine-tuning guide](https://docs.unsloth.ai/get-started/fine-tuning-guide).
   - While another stated that you can only do so using **LoRA**. *GGUF is an inference-optimized format, not designed for training workflows*.
- **Huggingchat experiencing 500 Error**: Users reported that **Huggingchat** is experiencing a **500 error**.
   - A member stated that an issue was raised and pointed to workarounds being discussed on [discord](https://discord.com/channels/879548962464493619/1355513801554006084).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/VIDraft/Gemma-3-R1984-12B">VIDraft/Gemma-3-R1984-12B · Hugging Face</a>: no description found</li><li><a href="https://www.llama.com/llama4/use-policy/">Llama 4 Acceptable Use Policy</a>: Llama 4 Acceptable Use Policy</li><li><a href="https://huggingface.co/posts/Reality123b/155118307932581">@Reality123b on Hugging Face: &quot;ok, there must be a problem. HF charged me 0.12$ for 3 inference requests to…&quot;</a>: no description found</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI compatibility · Ollama Blog</a>: Ollama now has initial compatibility with the OpenAI Chat Completions API, making it possible to use existing tooling built for OpenAI with local models via Ollama.</li><li><a href="https://huggingface.co/spaces/Remiscus/Customer_Support_Agent">Customer_Support_Agent - a Hugging Face Space by Remiscus</a>: no description found</li><li><a href="https://huggingface.co/mindspore-ai/LeNet">mindspore-ai/LeNet · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/consuming_tgi#python">Consuming Text Generation Inference</a>: no description found</li><li><a href="https://huggingface.co/docs/text-generation-inference/backends/llamacpp">Llamacpp Backend</a>: no description found</li><li><a href="https://forums.docker.com/t/error-docker-buildx-build-requires-exactly-1-argument-with-vs-code/136577">ERROR: &quot;docker buildx build&quot; requires exactly 1 argument. with VS code</a>: Hello,  After reading a lot of documentation and searching on the internet, I can’t find where come from this issue.  When i want to buid a image with a right clic and “Build image” this error showing...</li><li><a href="https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF">bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503">mistralai/Mistral-Small-3.1-24B-Instruct-2503 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/models?dataset=dataset:ylecun/mnist">Models - Hugging Face</a>: no description found</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-18-27-31.html">clipboard</a>: no description found</li><li><a href="https://huggingface.co/docs/smolagents/reference/models">Models</a>: no description found</li><li><a href="https://pypi.org/project/audioop-lts/#description">audioop-lts</a>: LTS Port of Python audioop</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/mistral3">Mistral3</a>: no description found</li><li><a href="https://www.reddit.com/r/learnpython/comments/144kxze/installed_module_was_told_module_couldnt_be_found/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide>">Unsloth Documentation</a>: no description found</li><li><a href="https://youtu.be/9KMxNZ2CvUg">NEW by DeepSeek: SPCT w/ DeepSeek-GRM-27B</a>: DeepSeek published a NEW learning method and a NEW model for the next generation of Reasoning models, called DeepSeek-GRM-27B. In this video I explain the ne...</li><li><a href="https://huggingface.co/spaces/Remiscus/Customer_Support_Agent/blob/main/README.md">README.md · Remiscus/Customer_Support_Agent at main</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-config-reference">Spaces Configuration Reference</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/pull/36894">Enable tracing for Moshi by lddabhi-semron · Pull Request #36894 · huggingface/transformers</a>: What does this PR do?Enabled tracing of MoshiForConditionalGenerationReplaced kwargs with args which was used inside forwardParsing forward signature to create kwargs for audio encoder, decoder...</li><li><a href="https://huggingface.co/blog/how-to-train">How to train a new language model from scratch using Transformers and Tokenizers</a>: no description found</li><li><a href="https://huggingface.co/blog/mlabonne/llm-course">The Large Language Model Course</a>: no description found</li><li><a href="https://youtu.be/UU1WVnMk4E8?si=juwuA26e4N_FTaD9">Create a Large Language Model from Scratch with Python – Tutorial</a>: Learn how to build your own large language model, from scratch. This course goes into the data handling, math, and transformers behind large language models....</li><li><a href="https://github.com/aashishjhaa/eq-for-youtube">GitHub - aashishjhaa/eq-for-youtube: Manipulate the audio of YouTube Video Realtime with 6 Frequency Band</a>: Manipulate the audio of YouTube Video Realtime with 6 Frequency Band - aashishjhaa/eq-for-youtube</li><li><a href="https://aashishjhaa.github.io/eq-for-youtube/">EQ for YouTube</a>: no description found</li><li><a href="https://github.com/huggingface/text-generation-inference/issues/2890">make install-server does not have Apple MacOS Metal Framework  · Issue #2890 · huggingface/text-generation-inference</a>: System Info make install-server does not have Apple MacOS Metal Framework Please either remove from the readme info about brew/macOS altogether to not confuse users. OR add support for Apple MPS fr...</li><li><a href="https://stackoverflow.com/questions/75593929/torch-circular-import-attributeerror">torch circular import AttributeError</a>: I am trying to use a script that uses torch but I keep getting this Attribute Error:&#xA;AttributeError: partially initialized module &#x27;torch&#x27; has no attribute &#x27;Tensor&#x27; (most likely...</li><li><a href="https://github.com/deepspeedai/DeepSpeed/issues/6005">[BUG] Circular import error with PyTorch nightly · Issue #6005 · deepspeedai/DeepSpeed</a>: Describe the bug Circular import error with PyTorch nightly. If I uninstall deepspeed it works fine. Traceback (most recent call last): File &quot;/test/oss.py&quot;, line 322, in &lt;module&gt; mp.sp...</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/text-generation-inference.git">GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference</a>: Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://github.com/huggingface/trl/issues/388">When to use SFTTrainer vs Trainer? · Issue #388 · huggingface/trl</a>: In the recent QLoRA blog post , the Colab notebooks used the standard Trainer class, however SFTTrainer was mentioned briefly at the end of the post. Why wasn&#39;t it used in the Colab notebooks asso...</li><li><a href="https://stackoverflow.com/questions/76461859/lmm-fine-tuning-supervised-fine-tuning-trainer-sfttrainer-vs-transformers-tr">LMM Fine Tuning - Supervised Fine Tuning Trainer (SFTTrainer) vs transformers Trainer</a>: When should one opt for the Supervised Fine Tuning Trainer (SFTTrainer) instead of the regular Transformers Trainer when it comes to instruction fine-tuning for Language Models (LLMs)? From what I ...</li><li><a href="https://opensource.org/blog/metas-llama-2-license-is-not-open-source">Meta’s LLaMa license is not Open Source</a>: Meta is lowering barriers for access to powerful AI systems, but unfortunately, Meta has created the misunderstanding that LLaMa 2 is “open source” - it is not.</li><li><a href="https://gnu.support/gnu-emacs/emacs-lisp/Gemma-License-danger-is-not-Free-Software-and-is-not-Open-Source.html">Gemma License (danger) is not Free Software and is not Open Source</a>: The **Gemma Terms of Use** and **Prohibited Use Policy** govern the use, modification, and distribution of Google&#39;s Gemma machine learning model and its derivatives. While Gemma is available for p...</li><li><a href="https://opensource.org/osd">The Open Source Definition</a>: Introduction Open source doesn&#8217;t just mean access to the source code. The distribution terms of open source software must comply with the following criteria: 1. Free Redistribution The license s...</li><li><a href="https://www.gnu.org/philosophy/free-sw.html">What is Free Software?
- GNU Project - Free Software Foundation</a>: no description found</li><li><a href="https://app.foundershub.ai/user/blogs/d019a1f3-02c3-4388-8d00-5d3d9afcea9a">How Hugging Face Enhances AI Agents with n8n Workflows</a>: Discover how Hugging Face’s NLP models integrate with n8n to build smarter AI agents. Learn practical use cases like chatbots and data querying tools powered by open-source language models.</li><li><a href="https://github.com/huggingface/transformers/commit/e959530b8f0011098246572e1777cac06e4bfe73">Add Mistral3 (#36790) · huggingface/transformers@e959530</a>: * initial start* style and dummies* Create convert_mistral3_weights_to_hf.py* update* typo* typo* Update convert_mistral3_weights_to_hf.py* Update convert_mistral3_weights_to_hf.py*...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357862826504360088)** (16 messages🔥): 

> `LLM Development, Sebastian Raschka Book, Andrej Karpathy Video, NLP course chapter 3` 


- **Community Member Seeks LLM Dev Guidance**: A community member asked where to start developing a **100M parameter LLM**, given a background in Data Science and ML.
   - Suggestions included starting with **NLP** or **DL**, or finding a specific course to follow.
- **Sebastian Raschka's Book Recommended for LLM Building**: The book *Build a Large Language Model (From Scratch)* by **Sebastian Raschka** was recommended for learning to build LLMs from scratch.
   - One member shared that their workplace started a book club around it, and another mentioned having ordered the same book.
- **Andrej Karpathy's GPT Reproduction Video Sparks Discussion**: A video by **Andrej Karpathy**, [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=2TN0KIfMR8_NxC29), was suggested as a good resource.
   - However, the original poster stated that *he started copy-pasting code and didn't explain much*, so they stopped watching it.
- **Assisted Pre-training and Shared Embeddings**: One member suggests *initializing weights and using the same tokenizer from another model, kinda like an 'assisted' pre-training*.
   - They also proposed *sharing embeddings and maybe the linear layer* to potentially expedite the LLM development process.



**Link mentioned**: <a href="https://youtu.be/l8pRSuU81PU?si=2TN0KIfMR8_NxC29">Let&#39;s reproduce GPT-2 (124M)</a>: We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1357984983716859996)** (2 messages): 

> `Windows CLI, Virtual Environment Reset, LocalAI, Dify` 


- **CLI Fu for Virtual Env Reset**: A quick **Windows CLI** command to reset your virtual environment is `pip freeze | Select-String -Pattern "^(?!pip)" | ForEach-Object { pip uninstall -y $_.ToString().Trim() }`.
   - This snippet helps clean up the environment by uninstalling packages, excluding **pip** itself, streamlining the process for a fresh start, according to [a blog post](https://app.foundershub.ai/user/blogs/cf808968-49be-41b9-81e6-9833b2bf2498).
- **[Placeholder]**: [Placeholder]
   - [Placeholder]



**Link mentioned**: <a href="https://app.foundershub.ai/user/blogs/cf808968-49be-41b9-81e6-9833b2bf2498">The Complete Roadmap to Mastering Agentic AI in 2025 | Girish Kotte</a>: Discover a comprehensive 12-step roadmap to mastering agentic AI in 2025. Learn everything from basic concepts to advanced deployment techniques with resource links for each stage. Perfect for develop...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357965193673703575)** (8 messages🔥): 

> `MCP Server and RAG Application, Osyllabi AI Curriculum, DocQuery AI Documentation Search, Municipal Law Dataset, LlamaResearcher with Llama-4` 


- ****MCP Server + RAG App Debut****: A member built a **MCP server and client**, connected via **ngrok**, along with a simple **RAG application** for QA with markdown documentation from a GitHub repository, showcased on [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_mcp-llm-enterprise-activity-7314281410712735744-yXJI).
   - The RAG application, named **DocQuery**, is available for feedback at [docquery-ten.vercel.app](https://docquery-ten.vercel.app/).
- ****Osyllabi: AI Curriculum Crafter Hits GitHub****: A member shared **Osyllabi**, a Python app for AI-driven personalized curriculums using web crawling and data integration, powered by **Ollama**, **HuggingFace**, **Langchain**, and **Llama-Index**, available on [GitHub](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi).
   - It features AI-driven curriculum generation, advanced web crawling, seamless integration with educational platforms, customizable learning paths, and flexible export options.
- ****DocQuery Transforms Documentation to Knowledgebase****: A member shared **DocQuery**, which turns documentation markdown into a knowledgebase, is available on [GitHub](https://github.com/md-abid-hussain/docquery).
   - DocQuery offers improved searchability, a smart Q&A system, and streamlined knowledge management for development teams.
- ****Municipal Law Dataset Surfaces****: A member shared the **American Municipal Law** dataset on [Hugging Face Datasets](https://huggingface.co/datasets/the-ride-never-ends/american_municipal_law), comprising municipal and county laws from across the United States in parquet format, organized by location's **GNIS id**.
   - Access requires agreeing to share contact information.
- ****LlamaResearcher: Llama-4 Powers Deep Research****: A member introduced **LlamaResearcher** ([llamaresearcher.com](https://llamaresearcher.com)), a deep-research AI companion powered by **Llama 4** and **Groq**, which expands queries into sub-queries, searches the web, and produces essays with source citations.
   - The project is open-source and Docker-ready, available on [GitHub](https://github.com/AstraBert/llama-4-researcher), and utilizes **LlamaIndex**, **Groq**, **Linkup**, **FastAPI**, **Redis**, and **Gradio**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/md-abid-hussain/docquery">GitHub - md-abid-hussain/docquery: DocQuery: Turn your documentation markdown to knowledgebase</a>: DocQuery: Turn your documentation markdown to knowledgebase - md-abid-hussain/docquery</li><li><a href="https://github.com/p3nGu1nZz/osyllabi">GitHub - Ollama-Agent-Roll-Cage/oarc-osyllabi: Osyllabi: A streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration.</a>: Osyllabi: A streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration. - Ollama-Agent-Roll-Cage/oarc-osyllabi</li><li><a href="https://github.com/the-ride-never-ends/municipal_law_search">GitHub - the-ride-never-ends/municipal_law_search</a>: Contribute to the-ride-never-ends/municipal_law_search development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/the-ride-never-ends/american_municipal_law">the-ride-never-ends/american_municipal_law · Datasets at Hugging Face</a>: no description found</li><li><a href="https://llamaresearcher.com),">no title found</a>: no description found</li><li><a href="https://llamaresearcher.com">LlamaResearcher - Topic to Essay in Seconds!</a>: AI-powered researcher companion that deep searches the web, validates information, and produces essays about any topic in seconds.</li><li><a href="https://github.com/AstraBert/llama-4-researcher">GitHub - AstraBert/llama-4-researcher: Turn topics into essays in seconds!</a>: Turn topics into essays in seconds! Contribute to AstraBert/llama-4-researcher development by creating an account on GitHub.</li><li><a href="https://docquery-ten.vercel.app/">DocQuery</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1358136363932848199)** (5 messages): 

> `Data Annotation for OCR, VLM Fine-Tuning for Handwritten Text, Combining OCR Techniques with VLMs, Roboflow for managing images and labels, MS-Swift and PEFT/Unsloth Approaches` 


- **VLM Models Aid Handwritten Text OCR**: A member is seeking methods for data annotation to fine-tune VLM models on handwritten text images, opting to move away from traditional OCR models and needing true text labels for training.
   - They are considering tools and methods to generate or correct text labels from images for fine-tuning purposes.
- **Classic OCR and Open VLMs Combine for annotation**: A member combined classic OCR techniques with open VLMs like **InternVL2_5** and **Qwen2.5** to generate initial annotations for extracting structured data from Brazilian documents.
   - Manual review was performed to correct errors after using OCR/VLM, and closed-source models like **Gemini** were noted to potentially provide higher-quality pre-annotations.
- **Roboflow Manages Images and Labels Effectively**: A member managed and stored raw images and corrected labels using **Roboflow**, annotating **510** images which were augmented to **1218** examples.
   - Despite finding its interaction not ideal, they used **Roboflow** for managing the dataset.
- **MS-Swift and PEFT/Unsloth Enhance Fine-Tuning**: A member fine-tuned several models using **MS-Swift** and experimented with **PEFT** and **Unsloth** approaches, achieving superior performance compared to **Gemini** and OCR methods with models adjusted from 1B to 7B.
   - The member successfully fine-tuned models, highlighting the effectiveness of these frameworks.
- **Tesseract OCR and Label Studio Join Forces**: One member is considering using **Tesseract OCR** followed by **Label Studio** for refining annotations.
   - They also checked **Gemma 3** and found it effective, implying a combination of automated and manual approaches for data annotation.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1357937615596421371)** (5 messages): 

> `Text Extraction from PDFs, Docling, SmolDocling, RolmOCR, Sci-BERT` 


- **PDF Text Extraction Advice Sought**: A member is seeking advice on improving text extraction from PDFs, specifically research papers, as their current results are unsatisfactory.
   - They have been using regex for section outline extraction but are facing challenges with fonts, headers, and footers, impacting the usability of the extracted content for **Sci-BERT** embeddings due to token limits.
- **Docling and SmolDocling recommended for Text Extraction**: A member recommends **Docling** ([GitHub](https://github.com/docling-project/docling)) and **SmolDocling** ([HuggingFace](https://huggingface.co/ds4sd/SmolDocling-256M-preview)) for improved text extraction from PDFs.
   - They note that while these tools still make errors, especially with images, they have yielded good results, with **SmolDocling** being an ultra-compact vision-language model for end-to-end multi-modal document conversion, as highlighted in [their paper](https://huggingface.co/papers/2503.11576).
- **RolmOCR Model Based on Qwen 2.5 VL Released**: A member mentions the release of **RolmOCR** ([HuggingFace](https://huggingface.co/reducto/RolmOCR)), a new model based on **Qwen 2.5 VL**, for OCR tasks.
   - However, they haven't personally tested it yet, but suggest it as a potential tool for text extraction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/docling-project/docling">GitHub - docling-project/docling: Get your documents ready for gen AI</a>: Get your documents ready for gen AI. Contribute to docling-project/docling development by creating an account on GitHub.</li><li><a href="https://huggingface.co/ds4sd/SmolDocling-256M-preview">ds4sd/SmolDocling-256M-preview · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357795572920225884)** (24 messages🔥): 

> `OpenWeatherMap API, ISO 3166-1 alpha-2 code, Qwen/Qwen2.5-Coder-32B-Instruct Alternatives, Hugging Face Token for Agent Creation, llm-course Channel` 


- ****Geolocation API** vs **Static Country Code Dictionary****: A member is building a tool to fetch weather conditions using the **OpenWeatherMap API** and is debating whether to use the **GeoCoding API** and another API for **ISO 3166-1 alpha-2 codes**, or to use a static dictionary.
- ****Free** alternative to **Qwen/Qwen2.5-Coder-32B-Instruct**?**: A member asked for a free alternative to **Qwen/Qwen2.5-Coder-32B-Instruct**.
   - Another member pointed out that the model itself is free under the Apache 2.0 license ([Hugging Face Link](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)) but suggested **Together AI** or **Groq** for free API access, noting potential rate limits of around 60 RPM.
- **Guidance on **Hugging Face Token** for Agent Creation**: A member requested guidance on obtaining a **Hugging Face token** for agent creation in Unit 1 of a course.
- ****llm-course Channel** Request**: A member inquired about the possibility of opening a dedicated channel for an **LLM course**.
- **Help needed with **AI agents course** setup**: A member requested assistance with a code issue encountered in Unit 1 of an **AI agents course**, specifically related to **HF token settings** in Colab.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-17-33-44.html">clipboard</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357936147308875839)** (36 messages🔥): 

> `MCP in Agent Course, Inference Usage Costs, Gemini Models, Course Feedback, Hallucination in Agents` 


- **MCP barely mentioned in Agent Course**: A user inquired about learning **MCP** in the agent course, but was informed that there's no dedicated section, although **MCP** servers are briefly mentioned in [unit 2.1 (smolagents)](https://huggingface.co/learn/agents-course/unit2/smolagents/tools#importing-a-tool-collection-from-any-mcp-server) and [unit 2.2 (llamaindex)](https://huggingface.co/learn/agents-course/unit2/llama-index/tools#model-context-protocol-mcp-in-llamaindex).
- **Inference costs incurred!**: A user accidentally maxed out their **Inference Usage Due Balance** and inquired about payment.
   - The suggestion was made to check the questions channel for a **FAQ**, or to use a local or cheaper hosted alternative.
- **Gemini Models may be your savior**: A user facing issues with **Code_agents** notebook in Chapter 2 due to payment requirements was advised to try using **Gemini models**.
   - It was noted that **Gemini models** can be used for free in many countries, with a link to [course notes](https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face) providing instructions.
- **Course Experience: Good but Buggy**: A user summarized the course as full of good material but noted that many notebooks and code snippets don't work, including a now infamous coding test in Unit 2, with no instructor presence.
   - The suggestion was made to approach the course sceptically, focus on understanding the coding parts, and acquire the necessary accounts and API tokens.
- **Explain the halluuuucinations!**: Users sought clarification on an example of **hallucination** in an agent.
   - The explanation provided was that the agent, lacking access to weather data, fabricated the answer, and the solution involves equipping the agent with a tool to retrieve weather information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face))">Notes from a front-end dev on the Hugging Face &quot;Agents Course&quot;</a>: Notes from a front-end dev on the Hugging Face &quot;Agents Course&quot; - 01_context.md</li><li><a href="https://huggingface.co/learn/agents-course/unit2/smolagents/tools#importing-a-tool-collection-from-any-mcp-server).">Tools - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit2/llama-index/tools#model-context-protocol-mcp-in-llamaindex)">Using Tools in LlamaIndex - Hugging Face Agents Course</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357805087711432905)** (177 messages🔥🔥): 

> `Grok 3, Turing Machines, Raw Binary AI training, LLama 4, Quantization Techniques` 


- **Grok 3 Manifold Analogy Appears**: A member shared an analogy describing various approaches to **NLP**, contrasting **0D Manifolds (tokens)**, **1D Manifolds (embeddings)**, and a **dynamic signal approach** where language is seen as a *rushing and swirling* river with no rigid bounds.
- **Raw Binary AI Training Discussed**: Members discuss training AI on **raw binary data** to directly output file formats like **mp3** or **wav**, with one member noting that this approach works based on discrete mathematics like **Turing machines**.
   - Another argued that current AI models are far from Turing-complete, while the original poster explained that the AI doesn't need to be Turing-complete to output appropriate tokens as responses.
- **New Llama 4 Models Released**: **Llama 4 Scout** boasts **10 million context window**, **17B active parameters**, and **109B total parameters**, while **Llama 4 Maverick** offers **1m context length**, **17B active parameters**, and **400B total parameters**, and **Llama 4 Behemoth** features **2 trillion parameters**.
   - Members express skepticism about the **10M context window** claim, the new **license**, and question if recent models are **RL'ed** or just base + SFT models, pointing out performance issues and mixed benchmarks.
- **Self-Principled Critique Tuning Explored**: **Self-Principled Critique Tuning (SPCT)** from DeepSeek is a new reward-model system where an **LLM** prompted with automatically developed principles of reasoning generates **critiques** of CoT output based on those principles.
   - The system aims to train models to develop reasoning principles automatically and assess their own outputs in a more **system 2** manner, instead of with human hand-crafted rewards, as outlined in [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495).
- **Quantization Techniques Examined**: Members discuss novel quantization techniques for large language models, pointing to a paper that has the [file](https://proceedings.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf).
   - It was argued that quantization can serve as a compromise between maintaining a super long context length and being able to serve the model, but comes with decay in the value you are actually getting out of those long contexts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ficlive/status/1908911992686931989?t=N1BGmubwXQQ-ZYLSKfmXqw&s=19">Tweet from Fiction.live (@ficlive)</a>: Updated Long context benchmark with Llama 4</li><li><a href="https://arxiv.org/abs/2504.01990">Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems</a>: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust pe...</li><li><a href="https://arxiv.org/abs/2504.01002">Token embeddings violate the manifold hypothesis</a>: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions ab...</li><li><a href="https://x.com/_arohan_/status/1909018336060747976">Tweet from rohan anil (@_arohan_)</a>: @ficlive They don’t seem to enable the attn config. Will try to see how to contact them. Meanwhile,https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynbHas th...</li><li><a href="https://arxiv.org/abs/2410.10714">SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators</a>: Large Language Models (LLMs) have transformed natural language processing, but face significant challenges in widespread deployment due to their high runtime cost. In this paper, we introduce SeedLM, ...</li><li><a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $...</li><li><a href="https://arxiv.org/abs/2504.01017">Scaling Language-Free Visual Representation Learning</a>: Visual Self-Supervised Learning (SSL) currently underperforms Contrastive Language-Image Pretraining (CLIP) in multimodal settings such as Visual Question Answering (VQA). This multimodal gap is often...</li><li><a href="https://arxiv.org/abs/2502.02631">ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization</a>: The optimal bit-width for achieving the best trade-off between quantized model size and accuracy has been a subject of ongoing debate. While some advocate for 4-bit quantization, others propose that 1...</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>: Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Current methods for training self-correct...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | Model Cards and Prompt formats</a>: Technical details and prompt guidance for Llama 4 Maverick and Llama 4 Scout
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357864625990926478)** (28 messages🔥): 

> `Llama 4, DeepSeek Paper, PaperBench, Text Diffusion` 


- **Llama 4 Omni Wakes Up**: A member shared the [Llama 4 documentation](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/), followed by a link to Meta's blogpost on [Llama 4's Multimodal Intelligence](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
   - The **Llama 4 Scout** model boasts **17 billion** active parameters, **16 experts**, and an industry-leading context window of **10M**, outperforming models like **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1**.
- **PaperBench: OpenAI's Replication Benchmark**: A member shared an article about [OpenAI's PaperBench benchmark](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052), designed to test AI agents' ability to replicate cutting-edge machine learning research papers from scratch.
   - The benchmark evaluates agents on reproducing entire **ML papers** from **ICML 2024**, with automatic grading using **LLM judges** and fine-grained rubrics co-designed with the original authors.
- **DeepSeek Paper Time**: Members are planning to go over the first **DeepSeek** paper in an hour, with a link to the paper provided ([https://arxiv.org/abs/2401.02954](https://arxiv.org/abs/2401.02954)).
   - The discussion took place in a [Discord event](https://discord.gg/jvDVwtfq?event=1357475477500985605).
- **Text Diffusion Steers Auto-Regressive LMs**: Members are planning to discuss a paper ([https://arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220)) on using a guided diffusion model to steer an auto-regressive language model to generate text with desired properties.
   - A recent talk by the main author discussing this paper was shared ([https://www.youtube.com/watch?v=klW65MWJ1PY](https://www.youtube.com/watch?v=klW65MWJ1PY)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.02954">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</a>: The rapid development of open-source large language models (LLMs) has been truly remarkable. However, the scaling law described in previous literature presents varying conclusions, which casts a dark ...</li><li><a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052">🥇Top AI Papers of the Week</a>: The Top AI Papers of the Week (Mar 31 - April 6)</li><li><a href="https://arxiv.org/abs/2408.04220">Diffusion Guided Language Modeling</a>: Current language models demonstrate remarkable proficiency in text generation. However, for many applications it is desirable to control attributes, such as sentiment, or toxicity, of the generated la...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | Model Cards and Prompt formats</a>: Technical details and prompt guidance for Llama 4 Maverick and Llama 4 Scout</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357801975487402144)** (17 messages🔥): 

> `GPT-6 release, Llama 4, Mindcraft Update, Adapting pre-training text, diffusion modeling to control LLMs` 


- **GPT-6 Coming Soon (Maybe?)**: A user jokingly announced the release of **GPT-6** yesterday, followed by **O0** and **OO** in the next few weeks, citing difficulties with **GPT-5**.
   - This sparked humorous reactions, with another user quipping that *"release" doesn't mean actually release the weights like a company that is open about AI.*"
- **Llama 4 Arrives with 10M Context**: **Llama 4 Maverick**, is the *most intelligent multimodal OSS model in its class* with **17 billion parameter model with 128 experts** and **10M** context window, according to [llama.com](https://www.llama.com/llama4/).
   - The model is said to be more powerful than all previous generation Llama models, while fitting in a single **NVIDIA H100 GPU**, surpassing **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1**.
- **Mindcraft Update Sees the Bots!**: A member shared a [YouTube video](https://www.youtube.com/watch?v=iDJ6GrHNoDs) titled **"Vision and Vibe Coding | Mindcraft Update"**.
   - The video description included a link to [Tripo AI](https://www.tripo3d.ai/app?invite_code=R2XF70), offering extra credits for the first 300 sign-ups using the code **R2XF70**.
- **LLMs Trained for Database Lookups**: A member mentioned adapting pre-training text to include database lookups for relevant facts, to train the **LLM** to look things up during generation, citing [this video](https://youtu.be/upbz6k6IDrk).
- **Diffusion Modeling Now Controls LLMs**: Users discussed using diffusion modeling to control **LLMs**, referencing the paper ["Diffusion-LM Improves Controllable Text Generation"](https://arxiv.org/pdf/2408.04220).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=iDJ6GrHNoDs">Vision and Vibe Coding | Mindcraft Update</a>: Try Tripo AI: https://www.tripo3d.ai/app?invite_code=R2XF70My Code: R2XF70The first 300 users to signup will get 500 extra credits on Tripo!The bots can see!...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357860447231021209)** (17 messages🔥): 

> `CUDA Python Package, Vectorized Memory Access, Llama-4 Router Normalization, High RAM/VRAM SSH Access` 


- **CUDA Python Package Debuts**: Nvidia released the [CUDA Python package](https://developer.nvidia.com/cuda-python), offering **Cython/Python wrappers** for CUDA driver and runtime APIs, installable via PIP and Conda.
   - It's intended to unify the Python CUDA ecosystem, providing full coverage and access to the **CUDA host APIs** from Python, mainly benefiting library developers needing to interface with C++ APIs.
- **Vectorized Memory Access Practices Sought**: Members discussed best practices for **vectorized memory access** when working with dynamic shapes, specifically in matrix multiplication with dynamic dimensions *m, n, and k*.
   - The discussion mentioned [Cutlass](https://developer.nvidia.com/cutlass) support and efficient vectorized loads as potential solutions.
- **Llama-4 Router Normalization Examined**: The channel discussed whether **Llama-4** uses router normalization, similar to how DeepSeek V3 and Mixtral do with their *topk_weights* normalization.
   - It was noted that Llama-4 skips the normalization, potentially because it uses `top_k = 1`, and both **DeepSeek V3** and **Llama 4** use sigmoid for the router logits.
- **High RAM/VRAM SSH Access Needed for Testing**: A member sought access to an SSH-like instance with at least **500GB of RAM/VRAM** for a couple of hours to test a model in **SGL**.
   - They have GPU credits from Modal and inquired about SSH access to a container.



**Link mentioned**: <a href="https://developer.nvidia.com/cuda-python">CUDA Python</a>: CUDA Python provides uniform APIs and bindings to our partners for inclusion into their Numba-optimized toolkits and libraries to simplify GPU-based parallel processing for HPC, data science, and AI.

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1358241585007165562)** (18 messages🔥): 

> `Triton Kernel Debugging, GPU Assembly Debugging, Grayscale Kernel Writing, Block Index Creation, Data Transposing` 


- ****Triton Kernel Debugging** Step-by-Step**: A first-time poster inquired about debugging **Triton kernels** step by step, specifically addressing issues with `cdiv` and `fill zeros` in interpret mode = 1.
   - An alternative suggestion involved diving into **GPU assembly**, setting breakpoints in the Python file using either `cuda gdb` or `roc gdb`, and single-stepping through the assembly file.
- ****GPU Assembly Debugging** with VSCode**: A member asked about using the **VSCode debugger** instead of only `cuda gdb` for debugging **GPU assembly**.
   - It was noted that running `cuda gdb` and passing in the **Python arguments** is required, but the convenience and readability of the **VSCode debugger** is desired.
- ****Grayscale Kernel Writing** Block Index**: A member described an attempt to write a **grayscale kernel** for a `(K, K, 3)` input, aiming to get blocks of `(BLOCK_K, BLOCK_K, 3)` in **Triton**.
   - However, they faced challenges with `tl.arange(0, 3)` because 3 is not a power of 2.
- **Loading **Nx3 Blocks****: A member asked how to load an **Nx3 block**, as `tl.arange` won't work since 3 is not a power of 2.
   - One suggestion involved loading data three times and incrementing the range by `image_w * image_h`, with another member suggested that adding 1 to all indexes should work.
- ****Data Transposing** for Contiguous Data**: A member considered transposing data with **Torch** for a contest, but they were concerned about abusing strides for loading contiguous data.
   - It was suggested that transposing with **Torch** is acceptable for the contest, as the original tensor will be contiguous and transposing will only be symbolic.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357816562878513334)** (18 messages🔥): 

> `CUDA debugger, nvshmem + mpi, nvbench and ubuntu 24.04, Shared memory access in CUDA, cute::copy and tiled_copy behavior` 


- **CUDA Debugging Delight**: A user confirmed **CUDA's debugger** works very similarly to **GDB CLI**.
   - Another member inquired about the release date for **cutile**, announced at GTC this year.
- **nvshmem + MPI Race Condition**: A member reported race conditions and hangs when running **nvshmem + mpi** with one more process than the number of GPUs, both with and without **MPS**.
   - They were running `mpirun -np 5 ./myapp` on a system with **4 GPUs** and asked if anyone had a solution.
- **nvbench Bumps CMake Requirement**: **NVBench** kind-of dropped support for **Ubuntu 24.04** because it requires a minimum CMake version of **3.30**, while Ubuntu 24.04 comes with **3.28**.
   - A member suggested [filing an issue on the nvbench repo](https://github.com/NVIDIA/nvbench/issues/new) and pointed to using a [previous tag](https://github.com/NVIDIA/nvbench) as a workaround.
- **Shared Memory Broadcasts in CUDA**: In response to a question about shared memory access in CUDA, it was confirmed that there are **broadcasts and multicasts** from shared memory.
   - A member pointed to the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks) and added that warp shuffles should be more performant.
- **Cute Copy Oddity**: A user found a strange behavior of **cute::copy** regarding **tiled_copy**, where all threads in a warp collectively copy data from shared memory to registers, instead of each thread copying its corresponding data.
   - An attached [image](https://cdn.discordapp.com/attachments/1189607726595194971/1358772917415973076/image.png?ex=67f50f64&is=67f3bde4&hm=fd23fb036477608b6ded972e4d638b8ade745699033f0d26ff8ca5d08da56f2c) demonstrated unexpected data arrangements in registers after the copy operation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apt.kitware.com">Kitware APT Repository</a>: no description found</li><li><a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks">1. Preface — CUDA C++ Best Practices Guide 12.8 documentation</a>: no description found</li><li><a href="https://cmake.org/download.">Download CMake</a>: no description found</li><li><a href="https://github.com/NVIDIA/nvbench/issues/new">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357857834632740884)** (10 messages🔥): 

> `torch compile backend, libtorch, mojo, torchscript, gelu+mul fusion` 


- ****Graphviz Backends Not Ready for Torch Compile****: A member inquired about a **torch.compile backend** that spits out **graphviz**, and another responded that they are moving towards producing **libtorch free binaries** using **torch.compile**.
   - They further claimed that there's no clever way of loading the model on **torchscript**.
- ****Mojo Unlikely to Bypass Python's GIL****: A member asked if anyone has used **mojo** to bypass **Python's GIL**.
   - No response was provided, so it's safe to assume the answer is NO.
- ****Compiling Gelu+Mul Fusion for Benchmarking****: A member asked how to get **torch.compile** to correctly and reliably fuse **gelu+mul** for benchmarking purposes, using PyTorch version 2.8, to compare against their **Triton kernel**.
   - No response was provided, so it's safe to assume the fusion is proving difficult!
- ****DDP/FSDP and Compilation Conventions****: A member inquired about the general convention for compiling a model before wrapping it around **DDP/FSDP1/FSDP2**.
   - Another member pointed to [torchtitan's implementation](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313) as a reference, which does a weird per-block compile thing beforehand, possibly to work around some **torch compile bugs**.
- ****Numerical Issues Plague FSDP****: A member reported having problems with **numerical issues with FSDP** and has disabled **torch compile** completely.
   - They claim that *it doesn't do a lot for them* but the **torchtitan** authors need to compile the **flex attention** and hopefully fuse some of their sequence parallel TP stuff, and the block-wrapping was a compromise.



**Link mentioned**: <a href="https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313">torchtitan/torchtitan/models/llama3/parallelize_llama.py at main · pytorch/torchtitan</a>: A PyTorch native library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1358007632308470000)** (1 messages): 

> `GPU Mode Website, Active Leaderboards, Website Feedback` 


- ****GPU Mode** Launches New Website**: Thanks to the hard work of two members, **GPU Mode** launched a new [website](https://www.gpumode.com/).
   - The website includes active leaderboards, links to lectures on YouTube, and their GitHub repo.
- **Leaderboard Status Shows H100 Dominance**: The website features active leaderboards for **A100, T4, H100, and L4 GPUs**, with several leaderboards showing results for **H100**.
   - For example, in one leaderboard ending in 21 days, *ajhinh* ranked first on **H100** with **7574.126μs**.
- **Feedback Wanted on Website Features**: The team is soliciting feedback on what to add to the [website](https://www.gpumode.com/).
   - Current features include leaderboard statuses, YouTube lectures, and the GitHub repo; feedback can be provided in a designated channel.



**Link mentioned**: <a href="https://www.gpumode.com/">Leaderboards &ndash; GPU MODE</a>: no description found

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1358180662217281617)** (6 messages): 

> `Llama 4, Triton Distributed, Tensara Triton Support, AMD Instinct MI325X Performance` 


- ****Llama 4** Arrives with Multimodal Prowess**: Meta introduces **Llama 4**, the latest iteration, boasting enhanced personalized multimodal experiences and featuring **Llama 4 Scout**, a **17 billion** parameter model with **16 experts** ([blog post here](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
   - It claims to outperform **Gemma 3**, **Gemini 2.0 Flash-Lite**, and **Mistral 3.1** and fit on a single **NVIDIA H100 GPU**, with an industry-leading context window of **10M**.
- **ByteDance Releases **Triton-distributed** for Parallel Systems**: ByteDance-Seed releases **Triton-distributed**, designed to extend the usability of Triton language ([github here](https://github.com/ByteDance-Seed/Triton-distributed)).
   - The new release is for parallel systems development.
- ****Tensara** Adds **Triton** Support for GPU Kernel Challenges**: **Tensara** now supports **Triton**, inviting users to compete in kernel optimization challenges and climb global leaderboards ([homepage here](https://tensara.org)).
   - Recent updates include **PyTorch-based test cases**, 3D/4D Tensor matmul problems, and activation functions like Sigmoid and Tanh.
- **AMD's **Instinct MI325X** Shows Strong MLPerf Inference Performance**: **AMD Instinct™ MI325X** GPUs demonstrate robust performance in **MLPerf Inference v5.0**, excelling in GenAI, LLMs, and reasoning models ([blog here](https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark)).
   - Results indicate a necessity for innovative GPU architectures tailored for AI transformation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark)">AMD InstinctTM MI325X GPUs Produce Strong Performance in MLPerf Inference v5.0 &#8212; ROCm Blogs</a>: no description found</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://github.com/ByteDance-Seed/Triton-distributed">GitHub - ByteDance-Seed/Triton-distributed: Distributed Triton for Parallel Systems</a>: Distributed Triton for Parallel Systems. Contribute to ByteDance-Seed/Triton-distributed development by creating an account on GitHub.</li><li><a href="https://tensara.org">Home | Tensara</a>: A platform for GPU programming challenges. Write efficient GPU kernels and compare your solutions with other developers.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1357913744159805531)** (6 messages): 

> `Qualcomm AI Engineer Hiring, Suno ML roles and H100 resources, Zero latency music creation` 


- **Qualcomm Seeks AI Engineer Team Lead**: Qualcomm is hiring an **AI Engineer/Team Lead** with a strong background in deep learning to design/deploy **SOTA models**, focusing on **accuracy-latency Pareto optimality**.
   - Interested candidates are asked to provide a short summary along with their CV or portfolio.
- **Suno's ML Talent Hunt**: Suno is hiring for all **ML related roles**, touting a small, well-resourced team with **hundreds of H100s** per researcher.
   - They are targeting **zero latency music creation** so that people can jam with **AI in real time**.
- **Zero Latency Music Creation Sounds Sick**: Suno aims to achieve **zero latency music creation**, enabling real-time AI jamming.
   - A user expressed hope that **Suno** could be a **VSTi in Ableton**.
- **Suno Internships Abound**: A user asked about internship opportunities at Suno, praising the platform.
   - No response was given.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357818915912286422)** (19 messages🔥): 

> `Centralized GPU programming language, OpenCL and SYCL, ROCm and HIP, 4-bit operations in CUDA for LLMs, Performance roofline models and arithmetic intensity` 


- **Why One GPU Programming Language Doesn't Rule Them All**: A newbie in GPU programming inquired why there isn't a centralized GPU programming language like **C** given the existence of **CUDA** for NVIDIA and **ROCm** for AMD.
   - An expert explained that **OpenCL** and **SYCL** exist but aren't mainstream due to poor support from vendors like NVIDIA, suggesting that the interface for OpenCL is old and C-adjacent.
- **ROCm's Dual Nature: AMD's CUDA Toolkit and HIP**: **ROCm** is AMD's CUDA Toolkit, while **HIP** is AMD's CUDA C++ that supports Nvidia hardware and compiling to PTX, but not Intel or others.
   - This offers a degree of cross-platform capability, though not universally.
- **Navigating 4-Bit Operations in CUDA for LLMs**: A user inquired about how to perform **4-bit operations in CUDA** for LLMs, such as matmul.
   - Another member recommended asking in the specific CUDA channel and being more specific about the operations.
- **Deciphering Arithmetic Intensity in Performance Roofline Models**: A member questioned the common practice of calculating bytes accessed in GEMM by summing matrix sizes (**MN + MK + KN**) for arithmetic intensity in performance roofline models.
   - Another member clarified that this is a simplification for establishing a theoretical maximum and is realistic for newer GPUs with large **L2 caches**, where one input matrix may fit entirely in L2.
- **Jumpstart CUDA Learning with Custom Projects**: A user asked for beginner-friendly CUDA projects and another user suggested learning through stuff YOU find interesting.
   - It was recommended creating something that requires a decent amount of multithreading or parallelism, such as linear algebra operations without a library, to simulate the concept of pipelining.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1358304042102820986)** (2 messages): 

> `Int4WeightOnlyConfig, torch.compile for speedup, Compiling individual submodules` 


- **Dequant with Int4WeightOnlyConfig benefits from torch.compile**: A member was trying to integrate `Int4WeightOnlyConfig` and asked if `torch.compile` is needed to speed up the dequant process.
   - Another member suggested that they can try to compile individual submodules by calling `torch.compile` on the submodules.
- **torch.compile submodules for efficiency**: To compile only the int4 modules, a member suggests iterating through the model's named modules and using `torch.compile` on specific submodules, such as `torch.nn.Linear`.
   - The suggested code snippet is:
```py
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        setattr(model, n, torch.compile(getattr(model, n)))
```


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1357816398658666696)** (3 messages): 

> `Silicon Valley Meetups, SF Meetups, Summer Intern Meetups` 


- **Silicon Valley Summer Meetups?**: An intern in the area asked if there would be any meetups in **Silicon Valley** this summer and offered to help organize one.
- **SF Meetup Planned Later This Year**: A member confirmed that a meetup is being planned in **San Francisco** for later this year, though specific dates were not mentioned.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1358014169403883520)** (37 messages🔥): 

> `RL fine-tuning with sandboxed code interpreter, Gemma 3 QAT vs HQQ, Wavespeed AI inference API, Vector Sum CUDA Kernel optimization, Tom and Jerry video generation with transformers` 


- **RL Code Fine-Tuning Toolset Showcased**: A member shared a [toolkit](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) for fine-tuning coding models using reinforcement learning with a local, zero-setup sandboxed code interpreter.
   - They found very promising results using a tiny fraction of data and training time versus traditional supervised fine-tuning and look forward to expanding it from Python to other languages, such as in [HIP Script](https://lights0123.com/blog/2025/01/07/hip-script/).
- **HQQ Quantization Beats QAT for Gemma 3**: A member evaluated **Gemma 3 12B QAT** vs. **HQQ**, finding that [HQQ](https://x.com/mobicham/status/1908477280029986933) takes a few seconds to quantize the model and outperforms the QAT version (AWQ format) while using a higher group-size.
   - With **GemLite bfp16** support, quantized Gemma 3 can run faster without performance issues.
- **Wavespeed AI touts efficient inference API**: The CEO of [Wavespeed AI](https://wavespeed.ai/) touted their platform's fastest and most efficient AI image & video inference API such as **FLUX** and **Wan** with **LoRA**.
   - They offer competitive custom pricing and hope to establish a win-win model to grow together.
- **Vector Sum Kernel achieves SOTA**: A member shared a [blogpost](https://veitner.bearblog.dev/making-vector-sum-really-fast/) and [code](https://github.com/simveit/effective_reduction) on achieving SOTA performance for summing a vector in CUDA, reaching **97.94%** of theoretical bandwidth, outperforming NVIDIA's **CUB**.
   - However, another member pointed out a potential race condition due to implicit warp-synchronous programming, recommending the use of `__warp_sync()` for correctness, with reference to [Independent Thread Scheduling (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling).
- **Tom and Jerry Cartoons Generated with Diffusion Transformers**: A team completed a project creating 1 minute long **Tom and Jerry** cartoons by finetuning a diffusion transformer, accepted to CVPR 2025, with code released on [GitHub](https://github.com/test-time-training/ttt-video-dit).
   - The model leverages **Test-Time Training (TTT) layers** within a pre-trained Transformer, enabling it to generate coherent videos from text storyboards, outperforming baselines like **Mamba 2**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mobicham/status/1908477280029986933">Tweet from mobicham (@mobicham)</a>: So I run evaluation on Gemma 3 12B QAT vs. HQQ. HQQ takes a few seconds to quantize the model and outperforms the QAT version (AWQ format) while using a higher group-size. With GemLite bfp16 support, ...</li><li><a href="https://wavespeed.ai/">WaveSpeedAI - Ultimate API for Accelerating AI Image and Video Generation</a>: Ultimate API for Accelerating AI Image and Video Generation</li><li><a href="https://x.com/panthaliaxyz/status/1909342585505669228">Tweet from Panthalia (@panthaliaxyz)</a>: Panthalia: Decentralized compute primitive  The platform to safely and easily train ML models on peer-to-peer computeWaitlist now available</li><li><a href="https://docs.panthalia.com/gradient-compression-algorithm">Panthalia Gradient Compression Algorithm | Panthalia</a>: This document provides a detailed description of the DCT-based gradient compression algorithm used in Panthalia. The algorithm is designed to efficiently compress both gradients sent from nodes to a c...</li><li><a href="https://github.com/pauleonix">pauleonix - Overview</a>: Ph.D. student in Computational Science and Engineering researching GPU-accelerated preconditioners and solvers for sparse linear problems; M.Sc. in physics. - pauleonix</li><li><a href="https://chromewebstore.google.com/detail/MyLMArena/dcmbcmdhllblkndablelimnifmbpimae">MyLMArena - Chrome Web Store</a>: Track your personal LLM preferences using ELO ratings with MyLMArena.</li><li><a href="https://github.com/huggingface/transformers/issues/31474#issuecomment-2198023128">Quantization support for heads and embeddings · Issue #31474 · huggingface/transformers</a>: Feature request Hi! I’ve been researching LLM quantization recently (this paper), and noticed a potentially improtant issue that arises when using LLMs with 1-2 bit quantization. Problem descriptio...</li><li><a href="https://test-time-training.github.io/video-dit/">One-Minute Video Generation with Test-Time Training</a>: A new approach using Test-Time Training (TTT) layers to generate coherent, minute-long videos from text.</li><li><a href="https://github.com/test-time-training/ttt-video-dit">GitHub - test-time-training/ttt-video-dit</a>: Contribute to test-time-training/ttt-video-dit development by creating an account on GitHub.</li><li><a href="https://x.com/karansdalal/status/1909312851795411093">Tweet from Karan Dalal (@karansdalal)</a>: Today, we&#39;re releasing a new paper – One-Minute Video Generation with Test-Time Training.We add TTT layers to a pre-trained Transformer and fine-tune it to generate one-minute Tom and Jerry cartoo...</li><li><a href="https://veitner.bearblog.dev/making-vector-sum-really-fast/">Making vector sum really fast</a>: In this blogpost we want to briefly describe how to archive SOTA performance for the task of reduction on a vector, i.e. our program should do the following:...</li><li><a href="https://github.com/simveit/effective_reduction">GitHub - simveit/effective_reduction: Improve reduction kernel step by step</a>: Improve reduction kernel step by step. Contribute to simveit/effective_reduction development by creating an account on GitHub.</li><li><a href="https://github.com/pranjalssh/fast.cu/tree/main">GitHub - pranjalssh/fast.cu: Fastest kernels written from scratch</a>: Fastest kernels written from scratch. Contribute to pranjalssh/fast.cu development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357981342557536316)** (18 messages🔥): 

> `Curriculum Learning for Reasoning, Llama 3 vs Qwen 2.5, Dream 7B Diffusion Model, Llama 4 Maverick coding, Claude Think Tool` 


- **Curriculum Learning Elicits Reasoning**: A member is experimenting with [curriculum learning](https://arxiv.org/html/2503.01307v1#S4) to elicit reasoning behavior in weaker LLMs like **Llama-3.2-3B**, by using easier reasoning tasks and gradually increasing difficulty to prime the model without SFT.
   - Another member mentioned that another user has already done some work on curriculum learning with RG and found better results compared to the same tasks without curricula, which is supported by the `training/` dir in the main branch.
- **Qwen 2.5 Beats Llama 3.2 in Training**: Members have been mostly using **Qwen 2.5 3B** over **Llama 3.2 3B** because **Qwen** seems to be a bit easier to train for reasoning.
   - This agrees with the findings in the '**4 Habits**' paper, in which Llama 3.2 struggled with backtracking and sub-goal setting without first using SFT.
- **Dream 7B Diffuses Reasoning**: The **Dream 7B** ([HKU Blog Post](https://hkunlp.github.io/blog/2025/dream/)), a diffusion based LLM, seems to show really good success on the kind of problems that the channel has, which might make it a really good candidate for gym training, especially looking at sudoku.
   - Dream 7B consistently outperforms existing diffusion language models by a large margin and matches or exceeds top-tier Autoregressive (AR) language models of similar size on the general, math, and coding abilities.
- **Llama 4 Maverick Aider Score Revealed**: **Llama 4 Maverick** scored **16%** on the [Aider polyglot coding benchmark](https://aider.chat/docs/leaderboards/).
   - This was referenced in a message on X, discussing coding benchmarks.
- **Claude Thinks with Tool Use**: A member shared a link to [Anthropic's Claude Think Tool](https://www.anthropic.com/engineering/claude-think-tool).
   - It wasn't specifically discussed how this relates to Reasoning Gym.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>: no description found</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843">Tweet from Paul Gauthier (@paulgauthier)</a>: Llama 4 Maverick scored 16% on the aider polyglot coding benchmark.https://aider.chat/docs/leaderboards/</li><li><a href="https://arxiv.org/html/2503.01307v1#S4,">Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1358939166250893482)** (3 messages): 

> `Deepseek communication library, NVSHMEM and UVA, Peer-to-peer GPU communication` 


- **Deepseek Leverages NVSHMEM Library**: The **Deepseek** communication library is built off the **NVSHMEM** library from NVIDIA, allowing for high-performance communication.
   - A member inquired if **NVSHMEM** utilizes **Unified Virtual Addressing (UVA)** for intra-node communication to facilitate peer-to-peer loads/stores to data stored in a remote GPU connected by NVlink.
- **NVSHMEM UVA usage for inter-GPU communication**: A member is inquiring about **NVSHMEM** and it's usage of **Unified Virtual Addressing (UVA)** for inter-GPU communication.
   - Specifically, they want to know if **UVA** enables peer-to-peer loads/stores to data stored in a remote GPU, connected by something like NVlink.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

leikowo: any way to have a ptx torch extension (not cuda with inline ptx) ?
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1358719735528816732)** (24 messages🔥): 

> `matmul Leaderboard submissions, vectoradd Benchmark Submissions, Modal Runners success, grayscale Leaderboard submissions` 


- **Modal Runners Deliver Matmul Masterpieces**: Multiple leaderboard submissions for `matmul` benchmark on **H100, A100, T4, L4** GPUs using Modal runners were successful, with IDs ranging from **3440** to **3453**.
- **Vectoradd Victorious with Modal on L4**: Several benchmark submissions for `vectoradd` on **L4** GPUs using Modal runners succeeded, including submissions with IDs from **3464** to **3506**.
- **Grayscale Gauntlet Gets Green Light**: A test submission (ID **3447**) and leaderboard submission (ID **3503**) for `grayscale` benchmark on **A100, H100, L4, T4** GPUs using Modal runners were successful.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1358059938156642304)** (5 messages): 

> `libsanitizer-collection.so, compute-sanitizer, LD_LIBRARY_PATH` 


- **Troubleshooter Seeks `libsanitizer-collection.so` Solution**: A member is encountering an issue where the grader can't find `libsanitizer-collection.so` when running `compute-sanitizer` during a `./grading test` for i8mm/gpu_blm.
   - They tried setting `LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/compute-sanitizer` based on googling, but it had no effect.
- **Compute Sanitizer error with i8mm**: A member reported a `compute-sanitizer` error where the system was *Unable to find injection library libsanitizer-collection.so*.
   - The error occurred during a test run of `i8mm` with the command `compute-sanitizer --tool memcheck`.
- **DejaVu Debugging**: Another member recalls encountering the `libsanitizer-collection.so` issue previously.
   - They stated that they did not quite remember what the solution was.


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1358880301543329923)** (2 messages): 

> `Leaderboard Units, Nanos vs Millis, Discord Cluster Manager` 


- **Leaderboard Display Units Clash!**: A user noted a discrepancy in the leaderboard's time units, with the [web leaderboard](https://gpu-mode.github.io/discord-cluster-manager/) displaying **nanoseconds** while the **Discord leaderboard** shows **milliseconds**.
   - A member responded that a *new leaderboard website* is prepared which *converts to an optimal unit for clarity*.
- **New Leaderboard Website Incoming**: A member announced that they have a *new leaderboard website prepared*, but they do *convert to an optimal unit for clarity*.
   - The discrepancy in the original leaderboard website had the [web leaderboard](https://gpu-mode.github.io/discord-cluster-manager/) displaying **nanoseconds** while the **Discord leaderboard** shows **milliseconds**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1358876526145179689)** (2 messages): 

> `Local LLM inference, Fine-tuning, GPU selection, L40 vs A100, Quantization` 


- **Local LLM Rig Build for Org**: Members are considering building a small rig for organizational LLM tasks like **summarization, chatbots, and text generation**, exploring options like **L40** or **A100** GPUs.
   - The primary focus is on optimizing for **4-bit and 8-bit model inference and potential fine-tuning**, taking price considerations (+5-10% of US prices) into account.
- **L40 Underperformance Puzzle**: Despite the L40 theoretically being better for **4-bit quantized Llama 3 70b**, it only achieves **30-35 tok/s** on single-user requests via vLLM, underperforming compared to online benchmarks of the A100.
   - The performance gap may be due to the **A100's superior DRAM bandwidth and tensor ops performance**, which are nearly twice as fast as the L40.
- **Exploring Quantization and Optimization Strategies**: The discussion suggests exploring **TensorRT** and specific quant formats to improve the performance of **L40**.
   - Despite **L40** having **FP8** support and a larger **L2 cache**, these advantages don't seem to translate to better performance compared to **A100** in current setups.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357973724795502592)** (14 messages🔥): 

> `Interactive voice mode, Mind maps rollout, Website URL use cases, Commercial scale version of NotebookLM` 


- **Interactive Voice Mode Inspires!**: A user expressed that the **interactive voice mode** was an interesting way of getting them to think about ideas.
   - After trying to make a solid **NotebookLM** foundation since January, they mentioned they can now make almost every text work and are confident they can help corporations set up a notebook tailored to their specific needs.
- **Mind Maps Finally Go Live!**: Users reported the **mind maps feature** has been fully rolled out, appearing in the middle panel for some, while others are still waiting.
   - One user mentioned seeing it briefly on the right side panel before it disappeared.
- **Audio Overview identifies website as a book**: A user inquired about use cases with a website URL, noting the **Audio Overview** incorrectly identified a website as a book.
   - Another user suggested the source type/genre is identified based on the source’s content/format, and running it again with a "customization" specifying it's a website resolved the issue.
- **Commercial NotebookLM Version Inquired**: A user asked if there is a commercial scale version of **NotebookLM**, where the data is not in the public domain, and specific programming or prompts can be entered.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357793377864908963)** (154 messages🔥🔥): 

> `NotebookLM's Discover feature rollout, Gemini 2.5 family, Mind Map evolution with generative AI, YouTube audio EQ Chrome extension, Google Cloud Next and Google I/O events` 


- **Theorizing Image-Based Mind Map Revolution**: Users discussed how **generative AI** tools could soon evolve mind maps to include images, drawing inspiration from **Tony Buzan's** original mind maps.
   - Members expressed excitement about the potential for more visually rich and informative mind mapping.
- **Discover feature Rollout Delays Frustrate Users**: Users have expressed frustration over the delayed rollout of the new **'Discover Sources'** feature in NotebookLM, which has been ongoing for over a week and is expected to take up to two weeks for full availability, announced April 1st.
   - The feature promises to streamline learning and database building by allowing users to create notebooks with sources directly within NotebookLM, eliminating the need to search outside the platform; one user even shared a Peter Griffin *'But I want it now'* GIF.
- **NotebookLM still on Gemini 2.0; 2.5 Tunability Teased**: Currently, NotebookLM utilizes the **Gemini 2.0 Thinking model**, though its effectiveness versus the **Flash model** in this context remains under evaluation.
   - **Gemini 2.5** is confirmed to be a family of models including a **Flash** version and **2.5 Pro** will soon be tunable, enabling developers to adjust its 'thinking' intensity.
- **Chrome Extension tunes YouTube audio with AI**: A member created an **AI-powered Chrome Extension** called *EQ for YouTube* which allows users to manipulate the audio of YouTube videos in real-time with a 6-band parametric equalizer; the extension has features for real-time frequency visualization, built-in presets, and custom preset creation.
   - The [GitHub repo](https://github.com/aashishjhaa/eq-for-youtube) is available for download.
- **NotebookLM's Language Change Explained**: To change the language in NotebookLM, use the URL `https://notebooklm.google.com/?hl=LANGUAGE_CODE`, replacing `LANGUAGE_CODE` with the desired language code (e.g., `es` for Spanish).
   - While the team acknowledged a previously identified translation bug (since resolved), the podcast output cannot be translated at this time.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1124402182171672732/1357653558140342343.">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement">Vote for the best of the internet</a>: I just voted in The Webby People's Voice Awards and checked my voter registration.</li><li><a href="https://tenor.com/view/peter-griffin-but-i-want-it-now-gif-26307521">Peter Griffin But I Want It Now GIF - Peter Griffin But I Want It Now - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://careers.google.com">Google Careers</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=9">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found</li><li><a href="https://notebooklm.google.com/?hl=LANGUAGE_CODE">no title found</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=7650624668661580589-EU">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found</li><li><a href="https://jamboard.google.com/">no title found</a>: no description found</li><li><a href="https://github.com/aashishjhaa/eq-for-youtube">GitHub - aashishjhaa/eq-for-youtube: Manipulate the audio of YouTube Video Realtime with 6 Frequency Band</a>: Manipulate the audio of YouTube Video Realtime with 6 Frequency Band - aashishjhaa/eq-for-youtube</li><li><a href="https://aashishjhaa.github.io/eq-for-youtube/">EQ for YouTube</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&r">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=9087543013148016209-NA">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1357941955438448650)** (28 messages🔥): 

> `Nvidia CUDA Python Support, Mojo GenAI, CuTile Programming Model, SIMD vs SIMT, Tenstorrent and Modular` 


- **Nvidia Adds Native Python Support to CUDA**: A member shared a link to an article, [Nvidia Finally Adds Native Python Support to CUDA](https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/), questioning if it's *the empire strikes back*.
   - The article discusses **Nvidia's** approach to GPU execution using the **CuTile** programming model, abstracting away from thread-level programming.
- **Can Mojo Tackle GenAI?**: A member wondered if **Mojo** is capable enough to develop **GenAI** or **Inference.ai** already.
   - This sparks discussion on the current capabilities and potential of **Mojo** in the field of **Generative AI**.
- **CuTile Programming Model Questioned**: A member expressed reservations about **Nvidia's CuTile** programming model, viewing it as a higher-level abstraction that removes the fun from writing **GPU code**.
   - They stated: *there taking the fun out of writing gpu code*.
- **SIMD vs SIMT**: A member is working on a Proof of Concept model, noting that modern parallel compute makes less sense to view through a typical threading model.
   - Discussion arose around exposing an **SM** as a big **SIMD core** with masking, and whether **SIMD** or **SIMT** is more appropriate, considering hardware flexibility and potential limitations.
- **Tenstorrent Software Stack**: A member suggested that **Tenstorrent** should use **Modular's** software stack, but another member noted that **Tenstorrent's** driver is incredibly easy to target and use.
   - They stated: *their driver is incredibly easy to target and use though, so while making effective use of their architecture might require some tinkering, just getting something that runs on it seems almost trivial*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/">NVIDIA Finally Adds Native Python Support to CUDA</a>: For years, NVIDIA’s CUDA software toolkit for GPUs didn&#039;t have native Python support. But that’s now changed.</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc25-s72449/">1,001 Ways to Write CUDA Kernels in Python | GTC 25 2025 | NVIDIA On-Demand</a>: You have to write a CUDA kernel
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357838483485364344)** (85 messages🔥🔥): 

> `Auto Lowering, MLIR Interpreter stress test, Implicit ctor hack, Mojo language spec, Mojo implicit copies` 


- **Auto Lowering Achieved**: A member discovered that auto lowering can be achieved when adding values from different scales, sharing a [link to the code](https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106).
   - The member noted that *it might be my most cursed work yet however*.
- **MLIR Interpreter Faces Stress Test**: A member commented that the time interval library might turn into a stress test of the **MLIR interpreter**.
   - Another member added that something didn't work as expected but can be remedied with an **implicit ctor hack**.
- **Mojo Spec Debate Heats Up**: A discussion started around whether **Mojo** will have a spec, with some arguing that it gives a language responsibility and maturity, referencing the design principles of **Carbon**.
   - Others countered that **Mojo's** design is tightly coupled to what **MAX** needs and that a spec would slow down development, with one member stating that Chris Lattner blames "design by committee" for the failure of **OpenCL**.
- **Mojo's Copy Semantics clarified**: A member inquired whether **Mojo's** implicit copies use Copy-on-Write (CoW).
   - Another member clarified that *semantics wise, always copy; optimisation wise, many are turned into move or eliminated entirely (inplace). Though it happens at compile time, CoW is a run time thing*.
- **ChatGPT's Mojo Skills Under Scrutiny**: A member asked if **ChatGPT** or alternatives are good enough to rewrite a large Python project in **Mojo**.
   - Another member responded that *ChatGPT isn't good at any new languages*.



**Link mentioned**: <a href="https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106">ChronoFlare/chronoflare/__init__.mojo at main · bgreni/ChronoFlare</a>: A time interval library written in mojo. Contribute to bgreni/ChronoFlare development by creating an account on GitHub.

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357800414619111616)** (54 messages🔥): 

> `Nomic Embed Text V2, GPT4All release cadence, Llama 4 release, ComfyUI for multimodal tasks, Semantic chunking` 


- ****Nomic Embed Text V2** integration is coming to Llama.cpp**: A member shared a [link to a GitHub Pull Request](https://github.com/ggml-org/llama.cpp/pull/12466) that shows **Llama.cpp** working on integrating **Nomic Embed Text V2** with Mixture-of-Experts (MoE) architecture for multilingual embeddings.
   - Another member expressed that *everything hangs on Llama.cpp* and hoped for Mistral Small 3.1 multimodal support.
- ****GPT4All** Silent Treatment Troubles Users**: Members are noticing a period of silence from core developers, with one member mentioning that this *causes uncertainty* about contributing to the app and the community.
   - The same member suggested this might not be a good policy for an open project, but that *when they break their silence, they usually come out swinging*.
- ****Llama 4** is here, but is it the greatest?**: Meta released **Llama 4** on April 5, 2025 ([announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)), featuring **Llama 4 Scout**, a 17B parameter model with 16 experts and a **10M token** context window.
   - Though some users are excited for the release, others expressed that *it is a bit of a letdown* and that *DeepSeek and Qwen* need to step up their game, while another noted the largest model has **2 Trillion parameters**.
- ****ComfyUI** is more than just a pretty face for image generation**: Members discussed the extensive capabilities of **ComfyUI**, noting that *you can do a lot with comfy if you have the nodes* including image and audio captioning.
   - Another member mentioned the possibility of video processing and described using command-line tools for visual model analysis.
- **Semantic chunking server recipe for delicious **RAG****: A member shared a [link to a semantic chunking server](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html) implemented with FastAPI.
   - The member also shared a [curl command example](https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html) for posting to the chunking endpoint, showing how to set parameters like `max_tokens` and `overlap`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: no description found</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html">clipboard</a>: no description found</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html">clipboard</a>: no description found</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12466">Nomic Embed Text V2 with Mixture-of-Experts (MoE) architecture by manyoso · Pull Request #12466 · ggml-org/llama.cpp</a>: Adds MoE-based embedding model supporting multilingual embeddings.Selects architecture variant based on hyperparameter detection (MoE layers).Removes unnecessary subclass initialization checks fo...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1358121947153825845)** (3 messages): 

> `MCP Servers, Full-Stack Agent Application, LlamaParse Layout Agent` 


- **MCP Servers get CLI Tooling**: A tool by @MarcusSchiesser lets you easily **discover, install, configure, and remove new MCP servers** from a single CLI interface, supporting Claude, @cursor_ai, and @windsurf_ai, as shown [here](https://t.co/zmqxBwNKvQ).
   - There are hundreds of official MCP servers out there.
- **Create Llama for Full-Stack Agents**: The **create-llama CLI tool** lets you spin up a web application with a **FastAPI backend and Next.js frontend** in a single line of code, creating just 5 source files as shown [here](https://t.co/TuZ0O0nMfe).
   - This is meant to jumpstart agent application development like deep research.
- **LlamaParse Launches Layout Agent**: A brand-new **layout agent within LlamaParse** gives you best-in-class document parsing and extraction with precise visual citations, using SOTA VLM models to detect all the blocks on a page and dynamically adapt.
   - The new agent dynamically adapts, as shown [here](https://t.co/2WRRXxIRa1).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357807162138689607)** (46 messages🔥): 

> `Workflow as a Tool, Multi-Agent System with Supervisor Pattern, RAG System with LlamaParse, Scalability Issue with DocumentSummaryIndex, Tools retry when exception occurred` 


- **Wrap Workflows as Tools with FunctionTool**: To transform a **Workflow** into a **Tool**, one can use the `FunctionTool` to wrap the workflow and gain control over its name, description, input annotations, and return values.
   - A member suggested a code snippet:
```python
async def tool_fn(...):
  """Some helpful description"""
  result = await workflow.run(...)
  return str(result)

tool = FunctionTool.from_defaults(tool_fn)
```
- **Agent Handoffs Supersede Supervisor Pattern**: When building a multi-agent system, it is more robust to have agents handoff between each other as needed, instead of using a supervisor pattern, which can be more error prone.
   - A [GitHub repo](https://github.com/run-llama/multi-agent-concierge) was shared as an example of a supervisor pattern implementation.
- **Replicate Document Summary Index with Vector Store Index**: The `DocumentSummaryIndex` may have scalability issues; it's advised to replicate its functionality using a normal `VectorStoreIndex` by summarizing documents, indexing with reference IDs, and swapping summary nodes with the original document during retrieval.
   - When using `load_index_from_storage`, the index store is loaded to memory which causes latencies as more documents are ingested.
- **Context's State Prepending to user_msg**: To avoid prepending the state content in the user message, one should avoid using the `state` key in the context and put data between tools elsewhere in the context.
   - A suggestion was to use `ctx.set("some_key", "some_val")` and `ctx.get("some_key")` instead.
- **Implement Text-to-SQL Query Engine Tool**: When implementing a text-to-SQL query engine tool for an agent, if there are only a few tables, it is not necessary to create an index of table descriptions and perform a vector query.
   - In cases of a small number of tables, the index and vector search parts can be skipped for better performance.



**Link mentioned**: <a href="https://github.com/run-llama/multi-agent-concierge">GitHub - run-llama/multi-agent-concierge: An example of multi-agent orchestration with llama-index</a>: An example of multi-agent orchestration with llama-index - run-llama/multi-agent-concierge

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357901550936391761)** (16 messages🔥): 

> `torch-geometric for tinygrad, Llama 4 10M context limitations, fast pattern matcher bounty, UOps generation, tinygrad YouTube video` 


- **Tinygraph: Torch-geometric for Tinygrad?**: A member inquired about the feasibility of creating a module similar to **torch-geometric** for graph ML within tinygrad, considering tinygrad's existing torch interface.
   - They questioned whether it would be *"useful"* to pursue such a module.
- **Llama 4's long context may not be so good**: A user shared a [tweet](https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA) claiming **Llama 4's** declared **10M context** is *"virtual"* because models were not trained on prompts longer than **256k tokens**.
   - The tweeter also stated that even problems below **256k tokens** may yield low-quality output due to the difficulty of obtaining high-quality training examples and that the largest model with **2T parameters** *"doesn't beat SOTA reasoning models"*.
- **$2000 Fast Pattern Matcher Bounty is available**: A member highlighted an open [$2000 bounty](https://github.com/tinygrad/tinygrad/pull/9737) for a fast pattern matcher in tinygrad.
   - The proposed solution involves a **JIT** for the match function, avoiding function calls and dict copies.
- **Reduce UOps to Speed Up Rewrite**: It was suggested that tinygrad sometimes generates more **UOps** than needed, increasing the cost to rewrite.
   - A member asked if it would be acceptable to sacrifice a few lines to generate fewer **UOps** initially, even if they are later optimized to the same result.
- **Tinygrad YouTube video shared**: A member shared a link to a [YouTube video](https://youtu.be/fWiieyG2zes?si=3CzwFRfJmFQhqUZvY).
   - No additional details were given.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA">Tweet from Andriy Burkov (@burkov)</a>: I will save you reading time about Llama 4.The declared 10M context is virtual because no model was trained on prompts longer than 256k tokens. This means that if you send more than 256k tokens to it,...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9737">first attempt at fast pattern matcher [pr] by geohot · Pull Request #9737 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357958444744708307)** (24 messages🔥): 

> `Tensor and SimpleMathTrait inheritance, Mesozoic tinygrad tutorials issues, METAL sync issue, AMD and BEAM issues` 


- **Debate About Tensor Inheriting SimpleMathTrait**: A discussion arose regarding whether **Tensor** should inherit from `SimpleMathTrait`, given that it re-implements every method that `SimpleMathTrait` provides without using the `.alu()` function.
   - It was noted that a previous bounty for refactoring **Tensor** to inherit from `MathTrait` was canceled due to poor submissions, with some suggesting **Tensor** may not need to inherit from either.
- **Colab CUDA Bug Causes Mesozoic Tinygrad Tutorial Issues**: A user encountered issues while running code from the mesozoic tinygrad tutorials in Colab, prompting others to request the error message for debugging.
   - It was identified as a Colab bug related to incompatible CUDA and driver versions, with a suggested workaround involving specific `apt` commands to remove and install compatible versions; in the meantime using the CPU device was suggested.
- **METAL Sharding Behavior Leads to Unexpected Results**: A member encountered unexpected behavior in sharding while trying to reproduce a minimal example of a METAL sync issue, suspecting that the **COPY** from **METAL:1** to **CPU** might be executing before the **XFER** from **METAL** to **METAL:1** completes.
   - The DEBUG output seemed to show the timeline adding the **XFER** when committed to the GPU command queue, not when it ends.
- **AMD and BEAM cause AssertionError**: A user encountered an `AssertionError` when running with **BEAM=2** and **AMD=1**, which seemed to be related to opening the device outside of the `if __name__ == "__main__"` block.
   - Setting **PARALLEL=0** or ensuring the device is opened within the `if __name__ == "__main__"` block resolved the issue.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357996936690401351)** (19 messages🔥): 

> `MCP with Command-A model, Cohere Tool Use, Cohere Scholars Program, Events Recording` 


- **MCP use with Command-A Model Explored**: A member inquired about using **MCP (Modular Conversational Platform)** with the **Command-A model**, suggesting it should work via the **OpenAI SDK**.
   - Another member agreed, stating that *there is no reason why it should not work*.
- **Cohere Tool Use Capabilities Detailed**: A member shared the [Cohere Tool Use Overview](https://docs.cohere.com/docs/tool-use-overview), highlighting its ability to connect **Command family models** to external tools like **search engines, APIs, and databases**.
   - It also mentions that **Command-A** supports tool use, similar to what **MCP** aims to achieve.
- **Cohere Scholars Program Details Shared**: A member asked about the requirements for the **Cohere Scholars Program**, specifically if prior publications are accepted.
   - A community member responded by linking the application form ([https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw)) and clarifying that while prior research experience is beneficial, it is not a requirement.
- **Inquiry about Events Recordings**: A member inquired whether Cohere events are recorded, as they were interested but unable to attend the live sessions.
   - The question remained unanswered in the provided context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw">Form</a>: no description found</li><li><a href="https://docs.cohere.com/docs/tool-use-overview">Basic usage of tool use (function calling) — Cohere</a>: An overview of using Cohere&#x27;s tool use capabilities, enabling developers to build agentic workflows (API v2).</li><li><a href="https://cohere.com/research/scholars-program">Cohere For AI - Scholars Program </a>: The C4AI Scholars Program offers an opportunity to collaborate with renowned researchers and engineers, fostering a collective exploration of the unknown. 
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1358869670899351734)** (1 messages): 

> `Aya Vision, Multilingual Multimodal Models, Open Weights Model` 


- **Aya Vision Team Hosts Tech Talks and AMA**: The core team behind **Aya Vision**, a multilingual multimodal open weights model, is hosting tech talks followed by an AMA on <t:1744383600:F>.
   - Attendees can join for exclusive insights on how the team built their first multimodal model and the lessons learned, with the event hosted by Sr. Research Scientist <@787403823982313533> and lightning talks from core research and engineering team members; further details are available at [Discord Event](https://discord.gg/sH3SSRp2?event=1358866070315860068).
- **Multilingual Model Aya Eyes Community Feedback**: The team has scheduled an Ask Me Anything to allow the community to directly engage with the creators.
   - Questions can be about anything from model architecture to future roadmap.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1358314800660217906)** (5 messages): 

> `Notion Connector, Vector DB for Notion` 


- **Slack app struggles with Notion integration**: A member asked for help with a working solution for a **Slack app** integration with a company **Notion wiki database**.
- **Vector DB Recommended to bolster Notion**: A member suggested using a **vector DB** due to **Notion's** subpar search API.
   - No specific recommendations were given, and it was stated that Cohere models work well with all vector DBs.


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1358662283135291483)** (3 messages): 

> `greetings` 


- **Users greet each other**: Two users are greeting each other in the 「🤖」bot-cmd channel, using "hey" and "sup".
   - The Cmd R Bot acknowledges the greetings.
- **Bots respond to greetings**: A bot responded to the users' greetings.
   - The bot used a casual "sup" to acknowledge the interaction.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357861706201694339)** (22 messages🔥): 

> `Fix for Timeout Crash, NeMo Resilient Training, RL Workflow, DeepSpeed Integration` 


- ****Timeout Crash** Bug Fixed**: A member fixed a bug related to **timeout crashes** and created `torchtune.utils._tensor_utils.py` with a wrapper around `torch.split` in [this pull request](https://github.com/pytorch/torchtune/pull/2560).
   - They suggested merging the tensor utils separately and then syncing with another branch to handle any conflicts.
- ****NeMo** Tackles Resilient Training**: A member attended a **NeMo** session on resilient training, highlighting features such as **fault tolerance**, **straggler detection**, **asynchronous checkpointing**, **preemption**, **in-process restart**, **silent data corruption detection**, and **local checkpointing**.
   - Not all of these are implemented, with some only planned; the member offered to rewatch and present details comparing **torchtune** vs. **NeMo** in terms of resiliency.
- **RL Workflow, Data Standard Format, and Prompts**: A member discussed the complexities of **RL workflows**, data formats, and prompt templates, suggesting a separation of concerns to decouple data conversion and prompt creation, allowing the same templates to be re-used across datasets.
   - The member suggested factorizing into a component that converts the data into a standard format, and another component that takes this standard format and converts it into the actual string with the prompt.
- **DeepSpeed backend for Torchtune?**: A member inquired about integrating **DeepSpeed** as a backend into **torchtune** and created [an issue](https://github.com/pytorch/torchtune/issues/2569) to discuss the possibility.
   - Another member asked for more context, noting that **FSDP** supports all the sharding options from **DeepSpeed**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/generated/torchtune.models.llama3.llama3_tokenizer.html#torchtune.models.llama3.llama3_tokenizer.">llama3_tokenizer &mdash; torchtune 0.6 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/issues/2569">deepspeed backend in torchtune? · Issue #2569 · pytorch/torchtune</a>: would be nice to have this optionality- happy to look into it if not out of scope</li><li><a href="https://github.com/pytorch/torchtune/pull/2560">fix: Timeout crash because of chunked_output len by bogdansalyp · Pull Request #2560 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Please link to any issues this PR addresses - closes #25...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

pjbontrager: You think they used AI to write that scrolling live updated chart?
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1358886256062500924)** (1 messages): 

> `AI4Math, Theorem Proving, Autoformalization, Formal Mathematical Reasoning, Language Models` 


- **Kaiyu Yang Presents on Autoformalization and Theorem Proving**: Kaiyu Yang will present on "Language models for autoformalization and theorem proving" [today at 4pm PDT](https://www.youtube.com/live/cLhWEyMQ4mQ).
   - The presentation will cover the basics of using **LLMs** for formal mathematical reasoning, focusing on **theorem proving** and **autoformalization**.
- **AI4Math is Crucial for AI-Driven System Design**: **AI for Mathematics (AI4Math)** is intellectually intriguing and crucial for AI-driven system design and verification, and extensive efforts have mirrored techniques in NLP.
   - The talk explores formal mathematical reasoning grounded in formal systems such as **proof assistants**, which can verify the correctness of reasoning and provide automatic feedback.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1358394025971028008)** (4 messages): 

> `LLM Agents MOOC, AgentX Competition, Course Quiz` 


- **LLM Agents MOOC link shared**: A member asked for a link to the **LLM Agents MOOC**, another shared [the link](https://llmagents-learning.org/sp25).
- **AgentX Competition Sign-Ups**: The staff shared sign-ups for the **AgentX Competition** are available [here](https://rdi.berkeley.edu/agentx/).
- **Course Quiz delayed**: A member asked about the missing quiz for the previous week.
   - A staff apologized for forgetting to post it and mentioned it would be available in a few minutes.



**Link mentioned**: <a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>: MOOC, Spring 2025

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357846993761992804)** (4 messages): 

> `asyncio support, full-async fork of dspy, reasons to migrate` 


- **Asyncio support: will dspy be async?**: A member inquired about plans to add **asyncio support** for general dspy calls.
   - They mentioned using *litelm* initially and then growing into dspy optimization, expressing interest in native dspy **async** capabilities.
- **Full Async Fork Faces Abandonment?**: A member has maintained a [true full-async fork of dspy](https://github.com/swiftdevil/dspy/tree/full_async) for a few months but is migrating away from dspy.
   - They are willing to continue merging upstream changes if there's community interest but will abandon it otherwise.
- **Reasons to Migrate & Benefits of Async DSPy**: Members expressed curiosity about the reasons for migrating away from dspy, and which tool is being migrated to.
   - One member asked about the advantages of having a **full async DSPy** and suggested merging relevant features into the main repository.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1358004519241388083)** (3 messages): 

> `GitHub PR Review, Phi-4 Support` 


- **GitHub PR Gets Eyeballed**: A member mentioned reviewing a [GitHub Pull Request](https://github.com), leaving comments for further discussion on the platform.
   - The author expressed gratitude for the review, acknowledging the effort put into it and indicating a need to rerun the process based on the feedback.
- **Phi-4 Family Support Considered**: A member is considering extending functionality to **Phi-4-mini** and **Phi-4**, despite them not being officially supported.
   - This suggests an effort to broaden compatibility beyond the initially intended scope, potentially enhancing the tool's appeal.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1358918518338355270)** (1 messages): 

> `Manifold Research, Multimodal AI, Self-assembling space robotics, Robotic metacognition, Community Research Call` 


- **Manifold Research Hosts Community Research Call #4**: Manifold Research Group is hosting **Community Research Call #4** this Saturday (4/12 @ 9 AM PST), covering their latest work in **Multimodal AI, self-assembling space robotics, and robotic metacognition**.
   - Interested parties can register [here](https://lu.ma/wlne416w) to join the open, collaborative, and frontier science focused event.
- **CRCs are Manifold's Cornerstone Events**: **Community Research Calls (CRCs)** are Manifold's cornerstone events where they present significant advancements across their research portfolio.
   - These interactive sessions provide comprehensive updates on ongoing initiatives, introduce new research directions, and highlight opportunities for collaboration.
- **CRC #4 Agenda Announced**: The agenda for **CRC #4** includes updates on **Generalist Multimodality Research**, **Space Robotics Advancements**, **Metacognition Research Progress**, and **Emerging Research Directions**.
   - The event will cover recent breakthroughs and technical progress in their **MultiNet framework**, developments in **Self-Assembling Swarm technologies**, updates on **VLM Calibration methodologies**, and the introduction of a novel robotic metacognition initiative.



**Link mentioned**: <a href="https://lu.ma/wlne416w">Community Research Call #4 · Zoom · Luma</a>: Interested in generalist AI models, self-assembling space robots or machine self-awareness? Join us for Community Research Call #4!Community Research Calls…

  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

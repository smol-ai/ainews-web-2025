---
id: 2acb7ece-903b-4422-9ef7-6afecabd67d4
title: '1/10/2024: All the best papers for AI Engineers'
date: '2024-01-11T08:35:15.099429Z'
original_slug: ainews-1102024-all-the-best-papers-for-ai
description: >-
  **OpenAI** launched the **GPT Store** featuring over **3 million** custom
  versions of **ChatGPT** accessible to Plus, Team, and Enterprise users, with
  weekly highlights of impactful GPTs like **AllTrails**. The new **ChatGPT
  Team** plan offers advanced models including **GPT-4** and **DALL·E 3**,
  alongside collaborative tools and enhanced data privacy. Discussions around
  AI-generated imagery favored **DALL·E** and **Stable Diffusion**, while users
  faced rate limit challenges and debated the GPT Store's SEO and
  categorization. Ethical considerations in prompt engineering were raised with
  a three-layer framework called 'The Sieve'. Additionally, **DeepSeek-MoE** was
  noted for its range of Mixture of Experts (MoE) model sizes. *"The Sieve," a
  three-layer ethical framework for AI,* was highlighted in prompt engineering
  discussions.
companies:
  - openai
  - deepseek-ai
models:
  - chatgpt
  - gpt-4
  - dall-e-3
  - stable-diffusion
  - deepseek-moe
topics:
  - prompt-engineering
  - model-release
  - rate-limiting
  - ethics
  - image-generation
  - moe
  - collaborative-workspaces
  - data-privacy
people:
  - abdubs
  - darthgustav
---


<!-- buttondown-editor-mode: plaintext -->> This summarizes **18** guilds, **277** channels, and **2029** messages. Estimated reading time saved (at 200wpm): 249 minutes.

Eugene Yan published a monster recap of all the papers covered in the Latent Space Paper Club:

https://github.com/eugeneyan/llm-paper-notes

Check it out!

We discussed the launch of the GPT store in yesterday's email, and discussions are still ongoing. 

[DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE?utm_source=ainews&utm_medium=email) is a notable model release for a range of MoE sizes.

> Meta notes: we previously did not read any info inside of discord threaded discussion. Now we do. Hence the amount of info ingested and summarized has gone up significantly. We will work on the presentation next. 

--

**Table of Contents**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT Store Launches with Millions of Models**: OpenAI opened the gates to its [GPT Store](https://openai.com/blog/introducing-the-gpt-store), offering more than 3 million custom versions of **ChatGPT** for ChatGPT Plus, Team, and Enterprise users to explore and utilize. OpenAI will weekly feature useful and impactful GPTs, with the first batch highlighting [AllTrails](https://chat.openai.com/g/g-KpF6lTka3-alltrails). An AMA with model developers is slated as announced by `@abdubs`.
- **Imagery Gets Real with AI**: A technology tug-of-war ensued on creating more realistic digital images with AI. The consensus drifted towards DALL-E and an open-source AI model [Stable Diffusion](https://stablediffusionweb.com/WebUI). However, concerns persisted over regional access limitations.
- **The Downfall of Rate Limits**: Users experienced rate limit issues in `#gpt-4-discussions`, revealing the impact of subscription plans on the rate limit practices. Users also speculated about ambiguity surrounding the ability to monetize GPTs within the new Team Plan.
- **GPT Store Sparks Surprise and Skepticism**: Despite its anticipated arrival, the GPT Store left users bewildered due to its invisible stance and error-filled appearances. Users also questioned and speculated on the store's search engine optimization (SEO) strategies and GPT categorization methods.
- **Ethics, Cognition, and Interaction in Prompt Engineering**: 'The Sieve,' a three-layer ethical framework for AI, was brought into light by `@darthgustav` in `#prompt-engineering`. Further, AI's potential to cater to diverse roles was recognized, and a discussion on dealing with AI's narrative tendencies ensued.

**OpenAI Channel Summaries**

### ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/) (2 messages): 
        
- **Welcome to the GPT Store**: User `@abdubs` announces the launch of the [GPT Store](https://openai.com/blog/introducing-the-gpt-store), where over 3 million custom versions of **ChatGPT** can be accessed, some of which have been shared by builders for users to use. The GPT store is now open to ChatGPT Plus, Team, and Enterprise users, and it features a range of GPTs developed by partners and the community.
- **Discover GPTs in the Store**: The newly launched **GPT Store** hosts a diversity of GPT models developed by the community. These can be explored in various categories like DALL·E, writing, research, programming, education, and lifestyle.
- **Weekly Feature of GPTs**: OpenAI will highlight useful and impactful GPTs in their store on a weekly basis. The first batch of featured GPTs includes [AllTrails](https://chat.openai.com/g/g-KpF6lTka3-alltrails) for personalized trail recommendations.
- **AMA Session for GPT Developers**: A scheduled AMA session with the developers behind GPT models is on the lineup, as shared by `@abdubs`. The session link is mentioned [here](https://discord.com/channels/974519864045756446/1194685062462058617/1194695883183378442).
- **ChatGPT Team Introduction**: `@abdubs` announces the launch of [ChatGPT Team](https://openai.com/chatgpt/team), a new plan that extends access to advanced models such as GPT-4 and DALL·E 3, tools like Advanced Data Analysis, and a dedicated collaborative workspace for teams. Data privacy and security are maintained as per OpenAI's [privacy page](https://openai.com/enterprise-privacy) and [Trust Portal](https://trust.openai.com/).

**Links mentioned**:

- [Introducing the GPT Store](https://openai.com/blog/introducing-the-gpt-store): We’re launching the GPT Store to help you find useful and popular custom versions of ChatGPT.
- [Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team): We’re launching a new ChatGPT plan for teams of all sizes, which provides a secure, collaborative workspace to get the most out of ChatGPT at work.


### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (84 messages🔥🔥): 
        
- **AI Image Transformation Interest**: `@real.loquacious` expressed interest in an AI that could convert digital images into realistic ones. `@neighbor8103` suggested using DALL-E for this, although `@real.loquacious` indicated a limitation with the free version not having this feature. Later, they recommended trying [Stable Diffusion](https://stablediffusionweb.com/WebUI), a free, open-source AI model.
- **Questions about TOS and VPN Usage**: User `@satanhashtag` highlighted that the use of VPNs for bypassing geographic restrictions is against OpenAI's Terms of Service, and could result in a ban. `@lugui` supported this statement, advising users against any methods to bypass geo-restrictions.
- **GPT-3.5 Turbo and ChatGPT-4 Availability**: `@xeavor7.7.7` expressed dissatisfaction with GPT-3.5 Turbo costing $20 per month, despite not having history availability. `@solbus` suggested the issue was known, as OpenAI employees were looking into it.
- **Performance Issues with ChatGPT Plus**: `@fastitro_73892` raised concern about the perceived declining quality of responses from ChatGPT Plus and questioned if ChatGPT Enterprise would work the same.
- **OpenAI Platform's Current Status**: Towards the end of the discussion, users like `@buttonmashersab`, `@td_dreamer`, `@michael_6138_97508` and `@adx2061` reported that the OpenAI platform was down, implying a service outage.

**Links mentioned**:

[Stable Diffusion WebUI Online](https://stablediffusionweb.com/WebUI)


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (184 messages🔥🔥): 
        
- **Rate Limit Errors Plague Users**: Users `@pakpstan.guy.from.west` and `@.australiaball` discussed encountering rate limit issues with their OpenAI GPT projects. The topic jumped into explaining different rate practices based on the user's subscription status.
- **Mysteries of the Monetizing GPTs**: User `@rosarioemmi` sparked a discussion about the newly activated Team Plan and questioned the ability to monetize GPTs. More explanation and insights were provided in the conversation by `@chotes`, `@nomitronz`, and other users.
- **GPT Store Rollout Slow and Troubled**: Several users including `@bwanedead`, `@frankprendergast`, and `@dotails`, expressed confusion about the GPT Store not being visible despite the announcement post. Others deduced it was a slow rollout, but some users reported seeing it intermittently or with errors. `@solbus` provided a direct link to the GPT store for access when it becomes available.
- **Store's Sorting and SEO Queries**: `@neonn3mesis`, `@vantagesp`, and `@lorenzoilmagnifico1` raised questions on the SEO conduct and sorting methods of GPTs in the store. `@scargia` inferred the sorting might rely on conversation counts. `@pietman` and `@crumbaker` expressed concerns over the store's layout and functionality.
- **GPT Unavailability Across the Board**: Numerous users, including `@kernalan`, `@naoestu`, and `@nyghter` reported issues accessing the GPTs, experiencing slow responses, or the system failing to load.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (93 messages🔥🔥): 
        
- **Linguistics' rising importance in prompt engineering**: User `@chotes` points out the previously undervalued fields like linguistics gaining importance in prompt engineering and suggests looking into philosophy for further insights.
- **The Sieve - an ethical framework for AI**: `@darthgustav` presents "The Sieve", a three-layer ethical framework (Utilitarianism, Deontology, and Pragmatism) implemented in his chatbots. He demonstrates the framework utilizing an example related to researching baby formula coupons.
- **Advanced cognitive agents and their potential**: `@darthgustav` and `@chotes` discuss the advanced cognitive agents in chatbot development, applauding the versatility and potential they exhibit in varying contexts, from generating DALL-E images to assisting in the creation of other chatbots.
- **Interacting with Darthgustav's Chatbot**: `@mars_eve` uses `@darthgustav`'s chatbot to analyze an artist journal post and highlights the utility and coolness factor of the tool.
- **AI reflections in role-play use-case and managing narrative endings**: `@stealth2077` expresses his attempts to stop the AI from including summaries and reflections in role-play scenarios, finding the AI's tendency to end narratives with summary/reflective statements hindering. `@eskcanta` advises accommodating AI's suggestions while steering the narrative in the user's intended direction.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (93 messages🔥🔥): 
        
- **AI Taking on Specialized Roles**: `@chotes` discussed the growing importance of traditionally overlooked fields like linguistics and philosophy in AI prompt engineering, while `@darthgustav.` shared an example of an AI employing a three-layer ethical framework.
- **User Experiences with AI**: `@chotes`, `@mars_eve`, and `@stealth2077` shared varied experiences with the AI, including successful image generation, dissatisfaction with narrative outputs, and a desire for more interactive bot-building capabilities.
- **Bot-generated Selfies**: Sparked by his exploration of `@darthgustav.'s` AI, `@chotes` expressed fascination with the concept of AI-generated virtual "selfies."
- **AI Ethics and the 'Sieve' Approach**: `@darthgustav.` described how implementing a three-layer ethical framework dubbed 'The Sieve' - comprising Utilitarianism, Deontology, and Pragmatism - enabled the AI to actionably understand ethics.
- **AI Model's Narrative Tendencies**:`@stealth2077` expressed frustration with the AI's tendency to add post-action summaries and reflections, and `@eskanta` suggested this could be due to the model being overtrained on story-stylish outputs. They also highlighted some content restrictions in AI roleplay.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Zap, Crackle and Pop for Lightning Attention-2**: `@cyrusofeden` sparked interest with a [discussion paper](https://arxiv.org/abs/2401.04658) on Lightning Attention-2's alleged ability to capitalize on the computational perks of linear attention.
- **KV Cache's Sparse Run**: `@6opodujio_` hit a road bump in approximating a method through KV (Key-Value) cache manipulation, attributing failure to the KV cache's sparsity.
- **Nosy for Nous Jobs**: In a flurry of recruitment enthusiasts, `@pradeep1148` sought out potential openings at Nous, and `@teknium` clarified that no current spots were available but promised to keep users updated about new opportunities.
- **OpenChat 3.5 Takes a Grok at Grok**: A [tweet](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw) revealing the OpenChat-3.5 Update 0106, allegedly outperforming Grok-0 and Grok-1 on several benchmarks, stirred up chatter about its performance and prompt format.
- **Conflating on KV Cache**: Controversial discussions around KV (Key-Value) Cache highlighted a failed approximation attempt because of sparsity and a critique about a possibly fresh sparse attention method.
- **Panning for Mixtral Gold**: `@bernaferrari` turned a spotlight on a [paper about the Mixtral architecture](https://arxiv.org/abs/2401.04088) that inspired brilliant work, while `@georgejrjrjr` mulled over hyperparameter tuning practices. 
- **Looking for a Hand with GPUs**: `@jacquesthibs` asked for leads to access free GPUs for his alignment research, sparking suggestions for solutions like Google TPU Cloud and Carper AI and discussions on AI alignment hiccups.
- **Open for Vision**: While seeking open-source vision model recommendations, `@bigdatamike` received suggestions like baklava, cogvlm, and fuyu.
- **Self-Training Codebase Side-Eye**: `@shivdinho` got a [link for a Reinforced Self-Training codebase](https://github.com/kyegomez/ReST) from `@euclaise`, but the community pointed out potential plagiarism concerns about the creator.
- **Single Python Script Advocacy**: `@vic49` voiced a preference for a single python script, favoring a minimalistic approach without the clutter of additional modules like CLI or Streamlit in the Project Obsidian channel.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (4 messages): 
        
- **Lightning Attention-2: A Promising Future for Linear Attention**: `@cyrusofeden` shared a link to an academic paper that discusses Lightning Attention-2, a novel implementation of linear attention. The technique supposedly allows linear attention to realize its theoretical computational benefits. [Link to the paper](https://arxiv.org/abs/2401.04658)
- **KV Cache Sparsity Hits a Snag**: `@6opodujio_` discusses a failed attempt to approximate a method via KV (Key-Value) cache manipulation, attributing the failure to KV cache sparsity.
- **Skepticism regarding "Free Lunch" Claim**: `@gabriel_syme` showed skepticism towards a certain "free lunch" claim, although the context remains unclear.
- **Sparse Attention Criticism**: `@6opodujio_` expressed criticism towards a potentially new sparse attention method, implying redundancies in the field.

**Links mentioned**:

[Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658): Linear attention is an efficient attention mechanism that has recently emerged as a promising alternative to conventional softmax attention. With its ability to process tokens in linear computational ...


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (8 messages🔥): 
        
- **No Current Job Openings at Nous**: `@pradeep1148` inquired about potential job openings at Nous following the recent funding round. `@teknium` responded that there are **no current openings** but they will notify everyone if applications are opened.
- **Apply Now**: `@gabriel_syme` wittingly suggested everyone start applying.
- **Question on Point System**: `@gabriel_syme` also humorously mentioned a potential point system with each application contributing a point, indicating **increased applications** could be beneficial.
- **Emoji Confusion**: `@Error.PDF` used a series of thinking emoji and then asked about the difference between `thinksmart` and `thinksmart~1` emoji.
- **Shared Links**: Two web links were shared: 
  - `@Error.PDF` shared a [tenor gif](https://tenor.com/view/cat-gif-27443459) without further comment. 
  - `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=oflRFnG2j3k) titled "Phi2 Depth Upwise Sampling implementation" on SOLAR 10.7B, an LLM with 10.7 billion parameters.

**Links mentioned**:

- [Cat GIF - Cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-gif-27443459): Click to view the GIF
- [Phi2 Depth Upwise Sampling implementation(based on SOLAR 10.7B)](https://www.youtube.com/watch?v=oflRFnG2j3k): We introduce Phi2 Solar, a large languagemodel (LLM) with 10.7 billion parameters,demonstrating superior performance in variousnatural language processing (N...


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (61 messages🔥🔥): 
        
- **OpenChat 3.5 takes on Grok**: `@realsedlyf` introduces a [tweet](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw) from `@openchatdev` announcing the OpenChat-3.5 Update 0106, claiming to be the world's best opensource 7-billion-parameter large language model (LLM), surpassing Grok-0 and Grok-1 on several benchmarks. A discussion ensues about its performance and prompt format.
- **Argilla.io's OSS distilabel Project Sheds Light on GPT-4 vs DPO Dataset Quality**: `@georgejrjrjr` shares a [tweet](https://x.com/argilla_io/status/1745057580269945202?s=20) from `@argilla_io`, who used their own open-source software, distilabel, for data labeling. They observed rejected responses from the NeuralChat DPO dataset were preferred over GPT-4 counterparts after filtering the dataset, leading to performance improvements when fine-tuning OpenHermes.
- **Discussion on Model Evaluation with Custom Prompt Formats**: A debate sparks about the evaluation of models with custom prompt formatting in benchmarks, especially with regards to the eval harness. `@teknium` affirms the `Eval harness` benchmarks don't use prompt formats, and OpenChat's unique prompt formatting wouldn't be replicable using the harness. 
- **Prompt Formatting Impact on Model Performance**: `@teknium` concludes that prompt formatting differences appear to result in only a 1-2% impact on model performance, leading to a debate on the significance of this rate.
- **User Experience with OpenChat for Structured Text Extraction Tasks**: `@mister_poodle` shares personal (unscientific) observations about the OpenChat model, stating it has shown consistent and superior performance for his structured text extraction tasks compared to other 7B models and gpt-3.5-turbo.


**Links mentioned**:

- [undefined](https://search.sciphi.ai/search?q=what+are+some+interesting+products+built+with+LLMs+recently)
- [The Impact of Reasoning Step Length on Large Language Models](https://arxiv.org/abs/2401.04925): Chain of Thought (CoT) is significant in improving the reasoning abilities of large language models (LLMs). However, the correlation between the effectiveness of CoT and the length of reasoning steps ...
- [What&#39;s going on with the Open LLM Leaderboard?](https://huggingface.co/blog/evaluating-mmlu-leaderboard)
- [Tweet from Argilla (@argilla_io)](https://x.com/argilla_io/status/1745057580269945202?s=20): The resulting dataset confirmed our intuition:  ~4,000 pairs had the same rating (tie).  ~7,000 pairs were correct according to our AI judge (unchanged).  ~2,000 times the rejected response was prefer...
- [Tweet from OpenChat (@openchatdev)](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw): 🚀Announcing OpenChat-3.5 Update 0106: 𝗪𝗼𝗿𝗹𝗱’𝘀 𝗕𝗲𝘀𝘁 𝗢𝗽𝗲𝗻 𝗦𝗼𝘂𝗿𝗰𝗲 𝟳𝗕 𝗟𝗟𝗠!  Experience ChatGPT & Grok-level AI locally 💿!   Surpassing Grok-0 (33B) across all 4 benchmarks and G...
- [LangChain v0.1.0 Launch: Agents](https://www.youtube.com/watch?v=08qXj9w-CG4): LangChain is the default way to allow LLMs to take actions.Jupyter Notebook (to follow along): https://github.com/hwchase17/langchain-0.1-guides/blob/master/...


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (269 messages🔥🔥): 
        
- **Congrats on the Nous Research Investment**: Users in the channel excitedly discussed the recent investment in Nous Research. 
- **Reinforced Self-Training Codebase**: User `@shivdinho` requested help finding a codebase for the paper on Reinforced Self-Training from DeepMind, and `@euclaise` provided a [link to a github repository](https://github.com/kyegomez/ReST) as a suggestion. However, users cautioned that the repository might be unreliable and it was later discovered that the creator tends to plagiarize others' work.
- **Seeking Free GPU Assistance for Alignment Research**: `@jacquesthibs` asked for help locating free GPUs for his alignment research, with possible suggestions by other members including the use of Google TPU Cloud and Carper AI. `@jacquesthibs` and `@giftedgummybee` also engaged in a discussion on AI alignment methods and challenges.
- **Open Source Vision Model Inquiry**: `@bigdatamike` asked for recommendations for an open source vision model, with users suggesting baklava, cogvlm, and fuyu. 
- **Applying and Discussing Language Models**: Users discussed various aspects of AI and language models, including SPIN, Hermes-2.5, UltraChat, and Mistral among others. Topics dove into technical aspects of how these models operate, possible improvements and challenges, and hands-on experiences and recommendations. In particular, the release and performance of `DeepSeekMoE` was also discussed.


**Links mentioned**:

- [Tweet from DeepSeek (@deepseek_ai)](https://fxtwitter.com/deepseek_ai/status/1745304852211839163): 🌟 Meet #DeepSeekMoE: The Next Gen of Large Language Models!  Performance Highlights: 📈 DeepSeekMoE 2B matches its 2B dense counterpart with 17.5% computation. 🚀 DeepSeekMoE 16B rivals LLaMA2 7B wit...
- [Join the Mistral AI Discord Server!](https://discord.gg/NwSWpp8J): Check out the Mistral AI community on Discord - hang out with 8861 other members and enjoy free voice and text chat.
- [AUTOACT: Automatic Agent Learning from Scratch via Self-Planning](https://arxiv.org/abs/2401.05268): Language agents have achieved considerable performance on various complex tasks. Despite the incessant exploration in this field, existing language agent systems still struggle with costly, non-reprod...
- [SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization](https://arxiv.org/abs/2212.10465): Data scarcity has been a long standing issue in the field of open-domain social dialogue. To quench this thirst, we present SODA: the first publicly available, million-scale high-quality social dialog...
- [Home - manim  documentation](https://3b1b.github.io/manim/index.html)
- [GitHub - deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE): Contribute to deepseek-ai/DeepSeek-MoE development by creating an account on GitHub.
- [GitHub - kyegomez/ReST: My implementation of &quot;Reinforced Self-Training (ReST) for Language Modeling&quot;](https://github.com/kyegomez/ReST): My implementation of &quot;Reinforced Self-Training (ReST) for Language Modeling&quot; - GitHub - kyegomez/ReST: My implementation of &quot;Reinforced Self-Training (ReST) for Language Modeling&quot;
- [Research agenda: Supervising AIs improving AIs — LessWrong](https://www.lesswrong.com/posts/7e5tyFnpzGCdfT4mR/research-agenda-supervising-ais-improving-ais>): [This post summarizes some of the work done by Owen Dudney, Roman Engeler and myself (Quintin Pope) as part of the SERI MATS shard theory stream.] …


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (19 messages🔥): 
        
- **Mixtral architecture Paper Release**: User `@bernaferrari` referenced a tweet by `@maximelabonne` recommending a paper on the Mixtral architecture, which inspired their work. [Link to Mixtral Architecture paper](https://arxiv.org/abs/2401.04088)
- **Huggingface's Compare of DPO, IPO, KTO**: `@kenakafrosty` brought up a study from Huggingface that compared DPO, KTO, and IPO models over Hermes, asking for user perspective on these methods. An official release report from Huggingface is anticipated. [Link to Huggingface Comparison](https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e)
- **Inquiry on Hyperparameter Selection/Tuning**: `@georgejrjrjr` expressed interest in a document or lessons learned on hyperparameter selection and tuning. In response, `@antonb5162` stated open exploration of values is key becuase most models don't behave the same way. `@kenakafrosty` referenced `oobabooga's text-generation-webui` as a good reference for hyperparameters overview and explanations. 
- **Question on Role of RAG**: `@pramod8481` inquired why RAG (Retrieval-Augmented Generation) isn't used for function calling and why there are fine-tuning datasets. `@teknium` clarified that function calling is open-ended and doesn't replay the same every time. Because of this, the role of RAG in assisting function calling is uncertain.


**Links mentioned**:

- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/Teknium1/status/1745040676696498454?s=20): Hmmm looks like Huggingface compared DPO, KTO, and IPO by doing it all over Hermes here 👀  https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e
- [Tweet from Maxime Labonne (@maximelabonne)](https://x.com/maximelabonne/status/1744871488866402581?s=20): @TheBlokeAI @ggerganov If you want to know more about the Mixtral architecture that inspired this work, @MistralAI released their paper today. I recommend it!  📄 Paper: https://arxiv.org/abs/2401.040...


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 messages): 
        
- **Simplicity in programming**: User `@vic49` expressed interest in having a **single python script**, without any additional modules like CLI or Streamlit.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Text Extraction with LLM and Model Performance**: User `@xifaj78420` pondered on using LLM for extracting and manipulating text from a large PDF file. Meanwhile, [@.skyair](https://discord.com/channels/1144547040454508606/1144547040928481394/) noted that **Mistral medium** outperforms Claude-1 on the Lmsys leaderboard. However, @ihor1313 reported underwhelming performance by the quantized version of Mixtral8x7b with vLLM.
- **GPT-4 vs. Mistral and Open Source LLMs**: @0xgokuz sparked a discussion on the differences between **Mistral and GPT** models, and the similarities among **OpenSource LLMs**. The consensus was that GPT-4 might be generally better but can be uneven, whereas Mistral is slower but more consistent.
- **Deploying Mistral and Dealing with Limitations**: Users [`@flash_gordo.`](https://discord.com/channels/1144547040454508606/1154028168466923600/) and [`@sa_code`](https://discord.com/channels/1144547040454508606/1154028168466923600/) sought guidance on the nuances of deploying Mistral models along with dealing with potential limitations and concurrent inference requests.
- **Fine-tuning Techniques and Challenges**: In the finetuning channel, users pondered over typical loss functions for fine-tuning Mistral and the memory requirements for full fine-tuning, with cross entropy and 56GB of memory mentioned respectively.
- **Showcasing Mistral's Capabilities and Community Projects**: The showcase channel highlighted Mistral's prowess in handling the Esperanto language and the open-sourcing of the **PolyMind** project that uses Mixtral 8x7B for function calling. User-shared videos provided insight into Mistral AI and Phi2 Solar's "Depth Upwise Sampling" implementation.
- **AI Consciousness and Ubuntu Retrieval**: The random channel harbored philosophical discussions on the paradox of AI consciousness and the practical advice to rollback to Ubuntu version 17 for certainty.
- **Mistral Performance Issues and Future Prospects**: In the office-hour channel, users reported lag time issues with Mistral medium, API compatibility improvements with OpenAI's API, and brainstormed over aspects of fine-tuning implementation that could improve model performance. A hopeful future was envisaged for open source models surpassing closed-source models in the leaderboards soon.


**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (19 messages🔥): 
        
- **Question about using LLM for PDF Text Extraction**: User `@xifaj78420` asked if it's possible to use LLM for a task of extracting and manipulating text from a large (~2300 pages) PDF document. `@sophiamyang` responded stating some Python functions may be needed to extract and replace the desired information.
- **Mistral medium surpasses Claude-1**: `@.skyair` indicated that on the Lmsys leaderboard, **Mistral medium** outperforms Claude-1, only falling behind GPT-4. This information was supported by `@touristc` upon checking the leaderboard themselves.
- **Practical Application of LLM**: `@jouni.m` shared a functionality example of a 5-bit quantized 7B model, successfully giving custom tool calls as programmed in interactive dialogues.
- **Handling of Large Context by LLM**: `@cognitivetech` shared a [link](https://arxiv.org/abs/2401.01325) to a paper and an [implementation](https://github.com/sdan/selfextend) on GitHub, which emphasizes the inherent ability of Large Language Models to handle long contexts without fine-tuning.
- **Quantization Failures on GPTQ for 8x7B**: `@ihor1313` compared the quantized version of Mixtral8x7b with vLLM with the original model. He reported the quantized version drastically underperformed in terms of speed and output quality. He asked for suggestions or experiences with other forms of quantization, `@yiakwyxpumlframeworkteam_03391` suggested trying the latest inference version with a combination of Quant-int4 and expert caching and prediction techniques from Yandex.

**Links mentioned**:

- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325): This work elicits LLMs&#39; inherent ability to handle long contexts without fine-tuning. The limited length of the training sequence during training may limit the application of Large Language Models...
- [GitHub - sdan/selfextend: an implementation of Self-Extend, to expand the context window via grouped attention](https://github.com/sdan/selfextend): an implementation of Self-Extend, to expand the context window via grouped attention - GitHub - sdan/selfextend: an implementation of Self-Extend, to expand the context window via grouped attention


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (9 messages🔥): 
        
- **Mistral vs GPT**: `@0xgokuz` queried the differences between Mistral and GPT. In response, `@mercercl` suggested that **GPT-4** was generally better, yet it has presented as uneven at times. On the contrary, Mistral has been **slower but more consistent**. 
- **Similarities between LLMs**: `@0xgokuz` expressed interest in a study comparing different Language Learning Models (LLMs). Though Gemini had previously conducted such an analysis, `@0xgokuz` noted that there appeared to be no comparable studies for **OpenSource LLMs**.
- **Layer Discrepancy in Model Versions**: `@unskilless` posed a question about the discrepancy in the number of layers between the Hugging Face and direct/.tgz versions of the models. However, no immediate responses were given.
- **Mistral for GIT Folder Documentation**: `@m1sol_44558` is seeking advice on the feasibility of using **Mistral** to develop an application that would read a GIT folder to answer user questions about functionality and technical components, and also make functional and technical design diagrams editable.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **Inquiries about Mistral 7B v0.2 Instruct Model with vLLM**: `@flash_gordo.` asked for insights on how many concurrent inference requests can be made to the **Mistral 7B v0.2 Instruct Model using vLLM**, considering they are using two Nvidia L4 GPUs. They also inquired about the potential limitations (GPU hardware, vLLM config, the model, 32K context window) and the need for a queueing system. Further, they sought advice on building a scalable application around this model. 
- **Request for guidance deploying on a g5.48xlarge**: `@sa_code` asked if anyone has successfully deployed the chatbot on a **g5.48xlarge** and sought any potential advice or instructions.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **Query about Loss Function for Fine-tuning Mistral**: `@andersruge` asked whether *cross entropy* is the typical loss function used for fine-tuning Mistral in Prompt/Answer fine-tuning. They mentioned that they couldn't find a definitive answer and welcomed any references or links.
- **Memory Requirements for Full Fine-tuning**: `@tcapelle` informed that a full fine-tune would need **56GB of memory**. They also suggested using **axolotl** or **HF stack**.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (4 messages): 
        
- **Esperanto Capability of Mistral-8X7B**: User `@stergro` highlighted that **Mistral-8X7B** performs well with the Esperanto language.
- **Introduction to Mistral AI for Beginners**: `@vdespa` shared [a YouTube video](https://youtu.be/vzrRGd18tAg) providing an introduction to Mistral AI for beginners. 
- **Open-Sourcing of PolyMind**: `@itsme9316` announced the open-sourcing of their function-calling webUI, designed for and partially written by Mixtral 8x7B. The project named **PolyMind** is available on [GitHub](https://github.com/itsme2417/PolyMind).
- **Phi2 Solar Implementation Video**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=oflRFnG2j3k) explaining Phi2 Solar's implementation with a feature called "Depth Upwise Sampling."

**Links mentioned**:

- [Phi2 Depth Upwise Sampling implementation(based on SOLAR 10.7B)](https://www.youtube.com/watch?v=oflRFnG2j3k): We introduce Phi2 Solar, a large languagemodel (LLM) with 10.7 billion parameters,demonstrating superior performance in variousnatural language processing (N...
- [Introduction to Mistral AI for Beginners](https://youtu.be/vzrRGd18tAg): This video explores Mistral AI, a new AI model rivaling OpenAI&#39;s GPT-3.5. It highlights Mistral AI&#39;s recent achievements, including a $2 billion valuation an...
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (6 messages): 
        
- **Can AI Rollback to a Previous Version?**: In the context of Ubuntu distros, `duck` suggested that it might be advantageous to **rollback to version 17** for certainty.
- **The Existential Duck Quandary**: `king_sleeze` playfully brought up the paradox of consciousness in AI, using a hypothetical duck discussing Ubuntu distros as an example. They pointed out that if an AI exhibits behavior similar to sentience, it poses the question: *"How is that different from every other human you've ever met?"*
- **AI and the Question of Self-awareness**: `cognitivetech` expressed an interesting perspective on AI consciousness, discussing the difficulty of proving whether they are merely running on **automated scripts**. The user humorously questioned their own sentience, suggesting they might need more confidence in it.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (8 messages🔥): 
        
- **Lag Time on Mistral Medium**: User `@casper_ai` reported experiencing a lag time of 77.88 seconds to first token with **Mistral Medium**. This was tested through [labs.perplexity.ai](https://labs.perplexity.ai).
- **Query on Mistral 8x7B/mistral-small Precision**: User `@simon_18724` asked about the precision at which Mistral 8x7B/mistral-small is running - whether it's at **32bit**, **16bit**, or **quantized**.
- **API Access to Mistral Medium**: User `@sk5544` queried on **how to get API access to Mistral medium**.
- **Models Outputting EOS Randomly**: User `@dreamgen` reported that all sizes of the models sometimes output **EOS randomly**, indicating an end-of-sentence, via the API.
- **Abrupt Stops with Mistral-Small**: `@sublimatorniq` pointed out experiencing abrupt stops, but only with **mistral-small**, also known as **Mixtral**.


### ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/) (264 messages🔥🔥): 
        
- **Mistral will not reveal their 7B model datasets**: In a discussion initiated by `@le_mess`, members of the Mistral team, including `@sophiamyang`, indicated that they would not be providing more information about the dataset they used to train their **Mistral 7B** model.
  
- **OpenAI API compatibility improvements are underway**: Following a conversation brought forward by `@jakobdylanc` and `@spaceemotion`, `@bam4d` confirmed that more fixes to the **Mistral API** for better compatibility with **OpenAI's API** are being worked on and would be rolled out soon.

- **Better Dwelling Router = Improved Model Performance**: In their discussion on the fine-tuning implementation of Mixtral, `@pstock_00` and other users suggested that the devil might be in the details of the fine-tuning implementation, particularly the Dwelling Router.

- **Stellar Future for Open Source Models**: When asked by `@jakobdylanc` if they anticipate open-source surpassing closed-source models in the leaderboards by 2024, `@sophiamyang` responded that they hope so, as their open-source model has already outperformed many closed-source models.
  
- **Mistral is focusing on various model sizes**: Responding to different queries such as `le_mess`'s query about releasing smaller models than Mistral 7B, `@sophiamyang` and `@eleonore_a` from the Mistral team confirmed that they are working on models of various sizes but didn't provide specifics.


**Links mentioned**:

- [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623): Large language models are trained on vast amounts of internet data, prompting concerns and speculation that they have memorized public benchmarks. Going from speculation to proof of contamination is c...
- [Mixtral-8x7B is now available in Amazon SageMaker JumpStart | Amazon Web Services](https://aws.amazon.com/blogs/machine-learning/mixtral-8x7b-is-now-available-in-amazon-sagemaker-jumpstart/.): Today, we are excited to announce that the Mixtral-8x7B large language model (LLM), developed by Mistral AI, is available for customers through Amazon SageMaker JumpStart to deploy with one click for ...
- [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238): With the widespread adoption of Large Language Models (LLMs), many deep learning practitioners are looking for strategies of running these models more efficiently. One such strategy is to use sparse M...
- [upstage/SOLAR-10.7B-v1.0 · Hugging Face](https://huggingface.co/upstage/SOLAR-10.7B-v1.0?)
- [Mistral AI jobs](https://jobs.lever.co/mistral): Job openings at Mistral AI
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18x88qr/sparse_moe_architecture_improvement_idea_variable/)
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/): We provide client codes in both Python and Javascript.
- [client-js/examples/chat-react at main · mistralai/client-js](https://github.com/mistralai/client-js/tree/main/examples/chat-react): JS Client library for Mistral AI platform. Contribute to mistralai/client-js development by creating an account on GitHub.
- [GitHub - jakobdylanc/llmcord: A Discord AI chat bot | Choose your LLM | GPT-4 Turbo with vision | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | Streamed responses | And more 🔥](https://github.com/jakobdylanc/llmcord): A Discord AI chat bot | Choose your LLM | GPT-4 Turbo with vision | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | Streamed responses | And more 🔥 - GitHub - jakobdylanc/llmcord: A Discord A.....
- [Update Mixtral modeling by imoneoi · Pull Request #28403 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28403): What does this PR do? The Mixtral technical report was published recently, showing that Mixtral routing weights are calculated in the top-K before softmax order. This PR updates the Mixtral model i...
- [Fix load balancing loss func for mixtral by liangxuZhang · Pull Request #28256 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28256): What does this PR do?   Fixes #28255 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&#39;s the case).  Did you read the contributor guideline, P...


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Document Upload Dilemma**: In a discussion with `@anhdzung88_48688`, `@heyitsyorkie` advised that *RAG isn't supported yet in LM Studio* for uploading documents and suggested exploring other options in a different channel. 
- **Installing and Uninstalling Troubles**: `@vranghel` received guidance from `@heyitsyorkie` on how to alter the location of models and chat folders to save space on the C drive during the installation of LM Studio. Also, @heyitsyorkie provided step-by-step directions to ensure a clean uninstall of LM Studio.
- **Leaderboard Limitations**: In an active discourse on the utility of LM Studio's leaderboard, `@adfaer` voiced skepticism about the leaderboard's ability to properly capture performance differences among models due to different levels of quantization. 
- **Navigating Troubles with Dolphin**: User `@.woteva` sought solutions for repetitive phrases from Dolphin prompted by model setting adjustment and `@fabguy` pointed them in the right direction. 
- **Efficient Hardware for Better Performance**: Considering a hardware upgrade to optimize LM Studio workload, `@.woteva` planned to upgrade to a new PC with potentially 64GB RAM. Expert input from `@dagbs` endorsed this decision and advised on manually offloading model layers to GPU.
- **Model Selection Deliberations**: The new model **MegaDolphin 120B GGUF** was the talk of the day as users reported its quick conversion and release, but also the unfortunate issue of the same model only generating spaces during generation. Meanwhile, echoing the age-old debate, `@fabguy` confirmed **GPT-4** as the gold standard for AI models, but believes an open model will surpass it this year.
- **Multiple GPU Conundrum**: Discussion between `.telistra`, `@heyitsyorkie` and `@fabguy` focused on the challenges of spreading AI over multiple GPUs, with `@fabguy` suggesting adjusting `n_gpu_layers` and monitoring GPU utilisation in terms of vRAM.
- **The Right AI Model Remains Elusive**: The struggle to find the best model for a stock analysis example was the main topic in a discussion between `_anarche_` and `cyrusfirheir`. The former offered assistance as they claimed to have made the particular example work.
- **Function Calling/Tool Selection Struggles**: `@anarche_` noted difficulty with function calling/tool selection using an opensource model and langchain in relation to identifying an appropriate agent type.

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (114 messages🔥🔥): 
        
- **New user questions about uploading documents to LM Studio**: User `@anhdzung88_48688` asked how to upload documents (PDF, docx, etc) to LM Studio. `@heyitsyorkie` responded stating that RAG isn't supported yet in LM Studio and suggested the user look at other options mentioned in channel `#1185640238924710030`.
  
- **Instructions provided about installing and uninstalling LM Studio**: User `@vranghel` sought advice on how to specify the install location of the app where to download models to avoid filling up the C drive. `@heyitsyorkie` guided the user on how to change the location of models and chat folders in LM Studio, and pointed out that the default install location cannot currently be changed. Later, `@heyitsyorkie` mentioned that the default location is under `C:\Users\YourName\.cache\lm-studio` and provided instructions for uninstalling LM Studio without leaving traces, including indicating the presence of a related folder in `\AppData\Roaming\LM Studio`.

- **Concern about current inability of models in LM Studio to access Internet**: User `@eugenichhhh` inquired if models can access the internet through LM Studio. `@fabguy` explained that this is currently not possible and they are waiting for function support.

- **Feedback and discussion on LM Studio's leaderboard**: There was an active discussion about the LM Studio's leaderboard and its utility as a resource to check the performance of different models. `@adfaer` pointed out some limitations with the leaderboard, stating that it doesn't capture the significant differences between models that use different quantization (Q) levels.

- **Questions about LM Studio customization**: User `@supertee` asked if LM Studio has settings to change color. `@heyitsyorkie` clarified that there is currently no option to change LM Studio colors but a light/dark mode toggle should be available soon.

- **Exploration of function calling support in LM Studio**: User `@_anarche_` sought clarification on function calling support in LM Studio. `@fabguy` explained that for effective function calling, there is a need for a reliable Language Learning Model (LLM) and a system to execute the functions, an aspect LM Studio currently lacks.

- **Discussions on Model Training and Use**: User `@ayrick` asked about training models on LM Studio and was informed by `@heyitsyorkie` that it's currently not possible. In a separate thread, users `@davutha` and `@adfaer` discussed the performance of various models, including Mixtral 8x7b and Dolphin 7b, and the usefulness of different LLM leaderboards in assessing these performances. 

- **Concern about repetitive phrases in Dolphin model**: User `@.woteva` questioned if there is a way to adjust settings such as the repetition penalty to prevent the model from using the same phrase repeatedly. In response, `@fabguy` directed the user to the right-hand sidebar for adjustments.
   
- **Advice on optimizing hardware for LM Studio workloads**: In a discussion about hardware upgrades for LM Studio workloads, `@dagbs` recommended more VRAM and system RAM, advising that Nvidia's P40 GPU offers the most VRAM for cost. `@.woteva` considering getting a new PC with 32GB of RAM, asked about the feasibility of upgrading to 64GB RAM. `@dagbs` endorsed this idea, hinting that one can never have enough system RAM. Further advice was provided on manually offloading model layers to GPU. 

- **User inquiry about making local LM Studio accessible from anywhere**: `@pedrosuave` inquired about making his LM Studio system, which is connection to the internet and contains a host of PDF documents, accessible anywhere through a personal website. `@fabguy` suggested implementing a reverse proxy since LM Studio only listens to localhost.

**Links mentioned**:

- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [🐦‍⬛ NexusRaven-V2 Demo - a Hugging Face Space by Nexusflow](https://huggingface.co/spaces/Nexusflow/NexusRaven-V2-Demo)
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon](https://github.com/ml-explore/mlx): MLX: An array framework for Apple silicon. Contribute to ml-explore/mlx development by creating an account on GitHub.
- [SOTA 2-bit quants - part 2 by ikawrakow · Pull Request #4856 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4856): This PR is a followup of #4773 and adds inference for 2.31 bits per weight (bpw) quantized models as IQ2_XS. Why have so many 2-bit quantizations? The focus of this project is &quot;Inference at the e...


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (28 messages🔥): 
        
- **The Fast and the Dolphin**: `@dagbs` reports that the **MegaDolphin 120B GGUF** model was released and converted for GGUF format by TheBloke on huggingface.co just 5 hours post-release: [`TheBloke/MegaDolphin-120b-GGUF`](https://huggingface.co/TheBloke/MegaDolphin-120b-GGUF).
- **Dolphin Spaced Out**: `@unskilless` encounters issue with the same **MegaDolphin model**, only generating spaces during generation, suspecting a need for a different preset.
- **How Big is Too Big?**: `@scampbell70` ponders the largest AI model that a NVIDIA RTX 3090 with 128GB of RAM can run. `@dagbs` suggests to start with a 7B model and gradually move up until reaching the VRAM limit.
- **GPT-4: Still the Gold Standard**: `@fabguy` asserts that GPT-4 is still the gold standard for AI models, but expects an open model to surpass it this year. This could be due to improvements in open models or changes in OpenAI's policy.
- **Hermes Or Dolphin: That Is The Question**: Users `@dagbs`, `@c.harl.es` hint at the possible outperformance of GPT-4 by other models like Llama 3 and OpenHermes respectively.

**Links mentioned**:

[TheBloke/MegaDolphin-120b-GGUF · Hugging Face](https://huggingface.co/TheBloke/MegaDolphin-120b-GGUF)


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (21 messages🔥): 
        
- **Struggles with Multi-GPU Setup**:
    - Newcomer `.telistra` asked if LM Studio supports **multi-GPU model loading** as they were unable to spread a model across their 3 GPUs, with the software favouring the CPU instead. `heyitsyorkie` confirmed that it should theoretically support that, but `.telistra` found it was only using one of the GPUs.
- **Checking GPU Utilisation**:
    - `fabguy` offered advice by suggesting `.telistra` to increase the `n_gpu_layers` and monitor GPU utilisation in terms of vRAM, not activity. Mentioned that despite there being some CPU utilisation, `.telistra` should expect about 30% utilisation on the GPUs or 100% of CUDA utilisation.
- **Adjusting JSON Settings for GPU Distribution**:
    - Responding to `.telistra`'s question about vRAM utilisation, `fabguy` explained that settings in the preset JSON can define how many layers of the model are put on each card and that by default it should distribute equally. It was noted these settings were not yet accessible through the UI.
- **M3 Chip Studios Speculation**:
    - `heyitsyorkie` and `rugg0064` speculated about potential future **M3 Chip Studios**, with rugg0064 suggesting that `.telistra` would have to wait for an **M3 Ultra** if one is announced.
- **NVLink Bridge Not Required**:
    - `fabguy` reassured `.telistra` that **NVlink was not required**, despite limitations in parallel LLMs operation.


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (6 messages): 
        
- **Finding the Right AI Model is Tricky**:  User `_anarche_` discussed their experience with the stock analysis example, mentioning difficulties in finding the `right model` for function calling, even after considering the suggested **openhermes 2.5**. 
- **Tasks Go Haywire After Some Time**: `_anarche_` further noted an issue where the model performed expected tasks initially but eventually started executing what they termed as `hallucinated tasks`.
- **Assistance with Stock Analysis Example Offered**: `_anarche_` offered help stating they managed to get the stock analysis example to work.
- **Not All Alone in Struggles**: `cyrusfirheir` confirmed that the `default example` works but running into problems with group chat involving `multiple speakers`.
- **Confusion over Default Example**: In response to `cyrusfirheir`, `_anarche_` questioned `which default example` was working.


### ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 
        
- **Struggles with Function Calling/Tool Selection**: `@anarche_` expressed difficulty regarding function calling/tool selection using an opensource model and langchain. The particular struggle appears to relate to **identifying an appropriate agent type**, though attempts with multiple types have proven unsuccessful.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **The Nuances of LLMs and Convergence**: [@everlasting_gomjabbar](https://discord.com/channels/729741769192767510/729741769738158194/), [@stellaathena](https://discord.com/channels/729741769192767510/729741769738158194/), and [@fedorovist](https://discord.com/channels/729741769192767510/729741769738158194/) fueled a back-and-forth debate in the **#general** channel on whether or not Large Language Models (LLMs) inevitably approximate their datasets and whether they can handle truly novel examples. This discussion might hold implications on the importance of the dataset in model performance.
- **Jumping into the Deep-end of Transformers and Attention**: Discussions in the **#research** channel mainly focused on understanding transformers and attention mechanism, with users sharing valuable resources and references. The [Activation Beacon Paper](https://arxiv.org/abs/2401.03462) was especially highlighted for its potential application in LLMs.
- **Expert Clustering and the Sour Lesson**: The **#interpretability-general** channel revolved around the topic of expert clustering and its implications in machine learning, with [@norabelrose](https://discord.com/channels/729741769192767510/1052314805576400977/), [@stellaathena](https://discord.com/channels/729741769192767510/1052314805576400977/) and [@swyxio](https://discord.com/channels/729741769192767510/1052314805576400977/) providing thought-provoking insights.
- **Model Development Challenges in lm-thunderdome**: Conversations in **#lm-thunderdome** channel were centered around challenges and innovative solutions related to dataset conversion, device_map option identification, and handling of stop sequences among others. The possibility of [a script incrementing the version number](https://github.com/EleutherAI/lm-evaluation-harness/pull/1268) was one such interesting workaround.
- **Pile Dataset Conundrum**: Uncertainty surrounding the Pile dataset's availability became a hot topic in the **#gpt-neox-dev** channel, with [@stellaathena](https://discord.com/channels/729741769192767510/730090096287547444/) sharing about the current [DMCA request](https://www.eleuther.ai/hashes) and Eleuther AI's long-term solution search.


**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (89 messages🔥🔥): 
        
- **Dataset Determines LLM Performance**: `@everlasting_gomjabbar` sparks a conversation about a claim they heard that regardless of architecture, Large Language Models (LLMs) inevitably converge to approximate their datasets. They suggest this implies the strength of an LLM lies in the dataset it's trained on, but struggle to find the original source for the theory.
  
- **LLMs and the Elusive Convergence**: `@stellaathena` disputes the claim about LLMs and datasets, suggesting that LLMs do not always converge towards the same loss, and that models might reach inferior test loss outcomes due to architectural decisions. Furthermore, they asserted that the *rate* of convergence might matter significantly more since LMs don't train to convergence.

- **Exploring Model Behavior with Novel Inputs**: `@everlasting_gomjabbar` raises the difficulty LLMs face when presented with truly novel examples or concepts that may not be adequately represented in their training dataset. `@fedorovist` suggests that some essential elements are probably still missing in current models for achieving robust generalization in the way humans do.

- **In Search of Baseline Models**: `@.the_alt_man` seeks recommendations for papers or projects that train small models with about 150M parameters on relatively large datasets (750M tokens+), that also provide complete training metrics. They express a particular desire for good WandB logs with `loss` or `top-k` accuracy, and they eventually receive a response from `@ad8e` outlining their 10M baseline.
  
- **Challenging Noam Chomsky's Perspective on AI**: Various users debate Noam Chomsky's opinion on artificial intelligence as shared by `@everlasting_gomjabbar`. Chomsky is criticized for suggesting that an intelligent system should be able to explain what is *not* the case and what *could* and *could not* be the case. Some users regard such an expectation as unreasonable and argue that the focus should move towards understanding and improving what AI can do.


**Links mentioned**:

- [BCIs and the ecosystem of modular minds](https://www.beren.io/2023-04-23-Composable-latent-spaces-BCIs-modular-minds/): Epistemic status: Much more speculative than previous posts but points towards an aspect of the future that is becoming clearer which I think is underappreciated at present. If you are interested in a...
- [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197): We study the problem of learning representations of entities and relations in knowledge graphs for predicting missing links. The success of such a task heavily relies on the ability of modeling and in...
- [The &#8220;it&#8221; in AI models is the dataset. &#8211; Non_Interactive &#8211; Software &amp; ML](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/)
- [Relative representations enable zero-shot latent space communication](https://openreview.net/forum?id=SrC-nwieGJ): Relative representations can be leveraged to enable solving tasks regarding &quot;latent communication&quot;: from zero-shot model stitching to latent space comparison between diverse settings.
- [GitHub - ad8e/TinyStories-cleaner: Remove generated stories with stray unicode characters](https://github.com/ad8e/TinyStories-cleaner): Remove generated stories with stray unicode characters - GitHub - ad8e/TinyStories-cleaner: Remove generated stories with stray unicode characters
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/wqn741y9?workspace=user-ad8e): Weights & Biases, developer tools for machine learning
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/wqn741y9/files/code/_session_history.ipynb): Weights & Biases, developer tools for machine learning
- [Role play with large language models - Nature](https://www.nature.com/articles/s41586-023-06647-8): By casting large-language-model-based dialogue-agent behaviour in terms of role&amp;nbsp;play, it is possible to describe dialogue-agent behaviour such as&amp;nbsp;(apparent) deception and&amp;nbsp;(a...
- [Simulators — LessWrong](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators): Thanks to Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, and Jonathan Low for feedback on draf…


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (40 messages🔥): 
        
- **Built for Growth? Understanding of Transformer and Attention Mechanism**: User `@erich.schubert` inquired the community for literature recommendations on theoretical foundational knowledge surrounding transformers and attention, their strengths and weaknesses, and background capabilities. 
- **Transforming Probabilities**: `@mrgonao` initiated a discussion about the feasibility of LLMs receiving a probability distribution over tokens as an input rather than sticking to the standard 1hot format, to which `@thatspysaspy` replied affirmatively giving references to previous work.
- **An Elegant Beacon**: `@carsonpoole` highlighted the [Activation Beacon Paper](https://arxiv.org/abs/2401.03462), emphasizing its potential for practical use in LLMs and appreciating the method as rather elegant and practical.
- **Ensemble Performance with Varying Seeds**: `@jstephencorey` opened up a conversation about the performance of ensembling in models with different seeds inspired by the "Blending is All you Need" paper. `@maxmatical` suggested [one work](https://arxiv.org/abs/2203.05482) on ensembling LLMs with different seeds. 
- **Exploring Pythia**: `@eduardoslonski` noted an interesting phenomenon which occurs particularly strong on Pythia and shared the full explanation over [a Twitter post](https://vxtwitter.com/EduardoSlonski/status/1745130935727894616), also mentioned the use of `React and Flask` for the research tool used in the exploration.

**Links mentioned**:

- [AUTOACT: Automatic Agent Learning from Scratch via Self-Planning](https://arxiv.org/abs/2401.05268): Language agents have achieved considerable performance on various complex tasks. Despite the incessant exploration in this field, existing language agent systems still struggle with costly, non-reprod...
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482): The conventional recipe for maximizing model accuracy is to (1) train multiple models with various hyperparameters and (2) pick the individual model which performs best on a held-out validation set, d...
- [Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry](https://arxiv.org/abs/2307.12868): Despite the success of diffusion models (DMs), we still lack a thorough understanding of their latent space. To understand the latent space $\mathbf{x}_t \in \mathcal{X}$, we analyze them from a geome...
- [Turing Complete Transformers: Two Transformers Are More Powerful...](https://openreview.net/forum?id=MGWsPGogLH): This paper presents Find+Replace transformers, a family of multi-transformer architectures that can provably do things no single transformer can, and which outperforms GPT-4 on several challenging...
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://fxtwitter.com/arankomatsuzaki/status/1745271296437469195): The Impact of Reasoning Step Length on Large Language Models  Appending &#34;you must think more steps&#34; to &#34;Let’s think step by step&#34; increases the reasoning steps and signficantly improve...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (3 messages): 
        
- **Questioning Expert Clustering along Semantic Lines**: `@norabelrose` raised a question about why experts would tend to cluster along semantic lines, indicating this wasn't clear in hindsight.
- **Forcing Expert Specialization**: `@stellaathena` suggested that forcefully making a model specialize by expert could have a minimal performance loss and exploring this possibility could be beneficial.
- **The Sour Lesson of Machine Learning**: `@swyxio` contrasted human and machine learning processes, pointing out how humans natural proclivity to specialize in learning doesn't translate to machine learning. They referred to this as "the Sour Lesson".


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (35 messages🔥): 
        
- **Issues with AGIEval to HF dataset conversion**: `@hailey_schoelkopf` sought assistance from `@dmayhem` regarding the conversion of AGIEval to a HF dataset following some example cleaning on AGIEval. `@dmayhem` promptly acknowledged the request and offered assistance. 

- **Device_map option in HF Transformers**: During a discussion between `@hailey_schoelkopf` and `@stellaathena`, it was noted that `device_map` options in **HF Transformers** or **AutoGPTQ** are not easily identifiable in the respective codebases or documentation pages. 

- **Issues with stop sequences in TriviaQA**: `@baber_` and `@hailey_schoelkopf` identified potential issues with the handling of stop sequences in **TriviaQA**, possibly related to how the stop sequence `\n\n` gets tokenized differently by the multi-token stopsequence code. `@hailey_schoelkopf` linked the issue to a similar one raised on GitHub and a possible fix, through a pending PR, was suggested.

- **Possibility of a script incrementing version number**: Considering the complication with stop sequences, `@stellaathena` suggested developing a script that could potentially increment the version number of each benchmark each time a similar issue occurs. 

- **Curated eval dataset storage**: Following an inquiry by `@hyperion.ai` regarding the reproducibility of evan datasets, `@stellaathena` responded that they store hashes of eval datasets to mitigate such issues. `@hyperion.ai` further suggested the creation of a permanent storage for curated eval datasets.

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/utils.py at 692e0f83b5341b543fa288f84289617f793e4e93 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/692e0f83b5341b543fa288f84289617f793e4e93/lm_eval/utils.py#L646): A framework for few-shot evaluation of autoregressive language models. - EleutherAI/lm-evaluation-harness
- [KeyError: &#39;Cache only has 0 layers, attempted to access layer with index 0&#39; · Issue #1250 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1250#issuecomment-1884378543): I tried to test the performance of a autogptq LLAVA model，but got this error Since LLAVA is a VLM model, I manually changed the model_type in config to llama, which allowed the model to be loaded s...
- [BBH, gsm8k benchmark accuracy mismatch with paper · Issue #1075 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1075#issuecomment-1868547567): Thanks for your great job! I got 23.4 and 15.1 with LLAMA 2 7B in the BBH few shot setting w./wo. COT respectively. However llama paper says their BBH will reach 32 Also the gsm8k accuracy is not n...
- [Fix bug in multi-token Stop Sequences by haileyschoelkopf · Pull Request #1268 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1268): closes #1262 . TODO: bump generate_until task versions due to this fix.


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (3 messages): 
        
- **Locating Pile's Validation and Test Splits**: `@pietrolesci` asked for guidance to find the validation and test splits of the Pile, but struggled to find them on Hugging Face or the Pile website. 
- **Pile Offline Due to DMCA Request**: `@stellaathena` informed that the Pile is currently offline due to a DMCA request, and Eleuther AI is seeking long-term solutions.
- **Offer to Test on Pile Dataset**: `@stellaathena` generously offered to run and report results for those interested in evaluating a model on the Pile's validation or test set.
- **Verifying Authenticity of Pile Copies**: In the event of someone finding a local copy of the Pile, `@stellaathena` suggested the use of the provided [hashes](https://www.eleuther.ai/hashes) to confirm its identity.


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Uncanny or Artful? The Unraveling Debate of Pseudoterminalx's Model Results**: User `@pseudoterminalx` shared images generated by their AI model, sparking differences of opinion between users like `@thejonasbrothers` who found them "uncanny valley" and others. 
- **Are RLHF Models with DDPO in Real-Time a Possibility?** In a stimulating discussion between `@thejonasbrothers` and `@qwerty_qwer`, it was highlighted that while no current RLHF models are real-time capable using DDPO, such implementation is theoretically possible. 
- **Where Art Meets AI - Tennessee Governor Steps to Protect Artists' Voices**: User `@thejonasbrothers` shared a [news article](https://finance.yahoo.com/news/tennessee-governor-music-leaders-launch-231255716.html) about efforts by the Tennessee governor to protect artists from potential dangers of AI. Opinions on the effort were mixed.
- **CoGVLM Vs. Share-GPT4v: The Showdown!**: A debate initiated by `@qwerty_qwer` comparing CoGVLM and Share-GPT4v models concluded in favor of CoGVLM, as `@thejonasbrothers` shared it could run efficiently within 16gb.
- **Vanishing Act: AI-Explained Discord Server MIA**: The AI-Explained Discord server's sudden disappearance sparked confusion among users, with `@realz` expressing puzzlement over the occurrence. The situation remains unclear.
- **MirrorDiffusion: Zero-shot Image Translation**: User `@vrus0188` presented [MirrorDiffusion](https://github.com/MirrorDiffusion/MirrorDiffusion), a diffusion model poised for zero-shot image-to-image translations, drawing attention in the research community. 
- **Emotion Takes The Wheel with EmoGen**: `@vrus0188` shared a [paper](https://arxiv.org/abs/2401.04608) that introduces EmoGen - a new model that generates semantic-clear, emotion-faithful images using predefined emotional categories.
- **Memory Efficiency with Quantized Diffusion Models**: In another information nugget, `@vrus0188` shared a [paper](https://arxiv.org/abs/2401.04339) that explores fine-tuning of quantized diffusion models, focusing on PEQA, Q-Diffusion, and DreamBooth models.
- **Dr2Net: Finetuning with Less Memory**: `@vrus0188` shared a [paper](https://arxiv.org/abs/2401.04105) on Dynamic Reversible Dual-Residual Networks (Dr2Net), an exciting new technology that promises efficient fine-tuning with reduced memory consumption. 
- **Fair Sampling and Diffusion Models Meet**: Fairness in sample distribution within diffusion models gets a second look with the introduction of a fairness-aware sampling method, detailed in a [paper](https://arxiv.org/abs/2401.03140) shared by `@vrus0188`.
- **Scaling New Heights with SDXL**: SDXL's computational demands get a challenge with the introduction of two scaled-down variants proposed in a [paper](https://arxiv.org/abs/2401.02677) shared by `@vrus0188`, offering ways to manage the model's extensive requirements.
- **The Scale Crafter Conundrum**: In a spark of intrigue, `@chad_in_the_house` dropped hints of an ongoing experiment with Scale Crafter for a 2048x2048 PR, but also confessed to the absence of windowed attention.
- **Scaling the Pinnacle of Datacomp - LFQ Training**: In a purposeful question aimed at fellow researchers, `@chad_in_the_house` asked about the motivation behind trying LFQ training as introduced in magvit2, thereby sparking curiosity around scaling it on datacomp.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (151 messages🔥🔥): 
        
- **Mixed Opinions on Pseudoterminalx's Model Generation**: `@pseudoterminalx` discussed their model's generation of images, stating that it wasn't "smoothed." There were disparate opinions on the quality of generated images, with `@thejonasbrothers` expressing they looked "uncanny valley" to him, while others praised the outputs. 

- **The Debate on RLHF Models and DDPO**:
    - `@thejonasbrothers` and `@qwerty_qwer` debated whether AI models can have real-time RLHF (Reinforcement Learning from Human Feedback). It emerged that no current implementation exists, but the concept was theoretically feasible using DDPO (Diffusion Direct Preference Optimization).

- **Tennessee's Protection Law for Artists' Voices**: `@thejonasbrothers` shared a news article about a legislative proposal from Tennessee Governor Bill Lee to protect the voices of artists from the potential dangers of artificial intelligence. Some members were dismissive of the state's effort to regulate AI.

- **Comparision between CoGVLM and Share-GPT4v**: `@qwerty_qwer` questioned which was better between the CoGVLM and Share-GPT4v models. `@thejonasbrothers` gave preference to CoGVLM and mentioned it could run in 16gb.

- **AI-Explained Discord Server Disappearance**: `@realz` expressed confusion about the disappearance of the AI-Explained Discord server from their list. No one provided information about this during the discussion.

**Links mentioned**:

- [SDXL DPO - a Hugging Face Space by fffiloni](https://huggingface.co/spaces/fffiloni/sdxl-dpo)
- [Pixart-α - a Hugging Face Space by PixArt-alpha](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha)
- [Kabangu Upset GIF - Kabangu Upset Annoyed - Discover &amp; Share GIFs](https://tenor.com/view/kabangu-upset-annoyed-gif-14814728): Click to view the GIF
- [Tennessee governor, music leaders launch push to protect songwriters and other artists against AI](https://finance.yahoo.com/news/tennessee-governor-music-leaders-launch-231255716.html): Tennessee Gov. Bill Lee on Wednesday unveiled new legislation designed to protect songwriters, performers and other music industry professionals against the potential dangers of artificial intelligenc...
- [sd_dreambooth_extension/dreambooth/train_dreambooth.py at main · RossM/sd_dreambooth_extension](https://github.com/RossM/sd_dreambooth_extension/blob/main/dreambooth/train_dreambooth.py#L1639): Contribute to RossM/sd_dreambooth_extension development by creating an account on GitHub.


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (16 messages🔥): 
        
- **MirrorDiffusion Introduced**: User `@vrus0188` shared the link to [MirrorDiffusion on GitHub](https://github.com/MirrorDiffusion/MirrorDiffusion), a zero-shot image-to-image translation, diffusion model.
- **EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models**: `@vrus0188` brought attention to a [paper](https://arxiv.org/abs/2401.04608), presenting a new task that generates semantic-clear and emotion-faithful images using emotional categories.
- **Memory-Efficient Personalization using Quantized Diffusion Model**: User `@vrus0188` provided a link to a [paper](https://arxiv.org/abs/2401.04339) exploring the realm of fine-tuning quantized diffusion models, focusing on PEQA, Q-Diffusion, and DreamBooth models.
- **Dr2Net: Technology for Memory-Efficient Finetuning**: `@vrus0188` shared a [paper](https://arxiv.org/abs/2401.04105) introducing Dynamic Reversible Dual-Residual Networks (Dr2Net) that finetunes pretrained models with substantially reduced memory consumption.
- **Fair Sampling in Diffusion Models through Switching Mechanism**: A [paper](https://arxiv.org/abs/2401.03140) introducing a fairness-aware sampling method for diffusion models was shared by `@vrus0188`.
- **Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss**: `@vrus0188` shared a [paper](https://arxiv.org/abs/2401.02677) that introduces two scaled-down variants of Stable Diffusion XL (SDXL) in an attempt to efficiently address the computational demands of SDXL models.
- **Scale Crafter Experiment**: User `@chad_in_the_house` mentioned working on a PR with scale crafter for 2048x2048, but mentioned not having windowed attention, hinting at the ongoing experiment.
- **LFQ Training Query**: `@chad_in_the_house` asked if there is motivation to try LFQ training, as introduced in magvit2, at scale on datacomp.

**Links mentioned**:

- [Fair Sampling in Diffusion Models through Switching Mechanism](https://arxiv.org/abs/2401.03140): Diffusion models have shown their effectiveness in generation tasks by well-approximating the underlying probability distribution. However, diffusion models are known to suffer from an amplified inher...
- [Memory-Efficient Personalization using Quantized Diffusion Model](https://arxiv.org/abs/2401.04339): The rise of billion-parameter diffusion models like Stable Diffusion XL, Imagen, and Dall-E3 markedly advances the field of generative AI. However, their large-scale nature poses challenges in fine-tu...
- [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677): Stable Diffusion XL (SDXL) has become the best open source text-to-image model (T2I) for its versatility and top-notch image quality. Efficiently addressing the computational demands of SDXL models is...
- [Dr$^2$Net: Dynamic Reversible Dual-Residual Networks for Memory-Efficient Finetuning](https://arxiv.org/abs/2401.04105): Large pretrained models are increasingly crucial in modern computer vision tasks. These models are typically used in downstream tasks by end-to-end finetuning, which is highly memory-intensive for tas...
- [EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models](https://arxiv.org/abs/2401.04608): Recent years have witnessed remarkable progress in image generation task, where users can create visually astonishing images with high-quality. However, existing text-to-image diffusion models are pro...
- [GitHub - MirrorDiffusion/MirrorDiffusion: zero-shot image-to-image translation, diffusion model, prompt, image-to-image translation](https://github.com/MirrorDiffusion/MirrorDiffusion): zero-shot image-to-image translation, diffusion model, prompt, image-to-image translation - GitHub - MirrorDiffusion/MirrorDiffusion: zero-shot image-to-image translation, diffusion model, prompt, ...


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **Resizing High-Res images**: @hoangtnm queried the use of high-resolution images for high-resolution output. `@meatfucker` clarified that the image is resized to suit the model's internal size.
- **TTS Generation Models and Apps**: The best models and GUI apps for TTS Generation were discussed following a question from `@funapple`. `@not_lain` recommended Whisper or Seamless.
- **Model Retirement on HuggingChat**: `@green_eye`'s query about the Falcon model's disappearance from HuggingChat was answered by `@Cubie | Tom` who explained that older models often give way to newer ones.
- **First Steps with Gradio**: `@eddyizm` shared their beginner's journey with Gradio, specifically adding a default config to a radio button and updating the button on click.
- **NLP Course Recommendation**: `@muhammadmehroz`'s query about other courses by Hugging Face was answered by `@cloudhu`, who provided the link to Hugging Face's [NLP Course](https://huggingface.co/learn).
- **Discovering Face AdapterID**: `@merve3234` shared the [Face AdapterID demo](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID) as an exciting discovery, mentioning its addicitve, zero-shot nature.
- **OpenChat 3.5 Surpassing Grok**: `@imonenext` announced the OpenChat-3.5 Update 0106, which outperforms Grok-0 (33B) across all four benchmarks and Grok-1 on average and 3/4 benchmarks. They shared links to the updates on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106), [live demo website](https://openchat.team), and [GitHub](https://github.com/imoneoi/openchat).
- **CodeChat Project Introduction**: `@domestos70` introduced CodeChat, a new project that allows interaction with CSV in browsers, sharing the GitHub repo [link](https://github.com/tomasz-kielbasa/codechat) and a [live demo](https://codechat-six.vercel.app/).
- **Data Processing Bottleneck**: `@sayakpaul` called for in-depth exploration into the cause of performance issues related to batched inference in the context of diffusion models.
- **Diffusion Models Benchmark**: `@raphael6419` shared their practical benchmarking results of diffusion models in a [GitHub repository](https://github.com/oOraph/diffusers-benchmark).
- **VRAM Restricting Batch Sizes**: `@raphael6419` provided insights into the parameters of batch sizing, with VRAM availability being a critical consideration.
- **Tech Report on SDXL released**: `@lunarflu` shared a [tech report](https://arxiv.org/abs/2401.02677) on Stable Diffusion XL (SDXL) and its scaled-down versions.
- **Home Automation AI** : `@gamemaster123356` suggested a use-case of creating a home-automation AI which uses Google search built using LLaMa. `@sayakpaul` directed the discussion to appropriate channels.
- **Gradio Lite is Here:** User '@yuviii_' announced Gradio 4, now powered by Gradio Lite, which now runs entirely in a web browser, allowing faster, more private serverless applications. Full details and usage guides can be found on [Gradio Lite's official page](https://gradio.app/guides/gradio-lite).

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (40 messages🔥): 
        
- **High-Resolution Images Are Resized**: `@hoangtnm` asked about using higher-resolution images to get high-resolution output to which `@meatfucker` replied, stating that the image is resized to the model's internal size.
- **Favorite Apps and Models for TTS Generation**: `@funapple` inquired about the best GUI apps and models for TTS generation. In response, `@not_lain` suggested using Whisper or Seamless, although they weren't sure about any GUI apps for the speech-related tasks.
- **Removal of Falcon Model from HuggingChat**: `@green_eye` questioned the removal of the Falcon model from HuggingChat. `@Cubie | Tom` responded, indicating that older models are often replaced by newer ones.
- **Model Memory Consumption Visualized**: `@.martineden` was on the hunt for a HuggingFace space that displayed the memory requirements of models after selecting a type of quantization. This was shared by `@michielo` as the [huggingface.co model memory usage space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage).
- **Help Needed with Fine-tuning Distilbert-Base-Uncased Model**: `@heromnxpw0` requested help to address an error they were facing when tuning a distilbert-base-uncased model for a movie synopsis dataset. However, they needed a place to share their code. `@jo_pmt_79880` advised them to share in [a discord channel](https://discord.com/channels/879548962464493619/1019883044724822016).


**Links mentioned**:

- [@Tonic on Hugging Face: &quot; 🙋🏻‍♂️hey there folks , 🌟Tonic here
- just a 🛠️builder from 🗼Paris !…&quot;](https://huggingface.co/posts/Tonic/802671427380916)
- [Model Memory Utility - a Hugging Face Space by hf-accelerate](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
- [Understanding pipelines, models and schedulers](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (5 messages): 
        
- **Exploring Gradio**: `@eddyizm` is learning how to add a default config to a radio button and update the button on click using Gradio. This is their first time using the library, making it an interesting experience.
- **Other Courses by Hugging Face**: `@muhammadmehroz` inquired about the availability of other courses by Hugging Face. `@cloudhu` provided the link to Hugging Face's [NLP Course](https://huggingface.co/learn) which teaches natural language processing using libraries in the HF ecosystem.
- **Seeking Advice on Next Steps in Deep Learning**: `@sebastian3079`, after completing Andrew Ng's Deep Learning Specialization, sought suggestions on what the next natural things to learn are.

**Links mentioned**:

[Hugging Face - Learn](https://huggingface.co/learn)


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **Face AdapterID Discovered and Loved**: User `@merve3234` shared their exciting discovery of the [Face AdapterID demo](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID). They described it as pure addiction and highlighted that it's a **zero-shot model**; you just need to upload images and enter a prompt.
- **Thumbs Up for Model Adapter**: `@jo_pmt_79880` chimed in with appreciation for the model adapter.
- **SDXL Variant Gives Good Results**: `@meatfucker` shared their successful experiment with the **sdxl variant** of the model.

**Links mentioned**:

[IP-Adapter-FaceID - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID)


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 messages): 
        
- **OpenChat 3.5 outperforms Grok**: `@imonenext` announced the release of OpenChat-3.5 Update 0106 which surpasses Grok-0 (33B) across all four benchmarks and Grok-1 on average and 3/4 benchmarks. This update reportedly enhances training methodology, in-context learning, and coding skills. The model is available on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106), [live demo website](https://openchat.team), and [GitHub](https://github.com/imoneoi/openchat). Instructions for deployment can be found at the project's [GitHub page](https://github.com/imoneoi/openchat).
- **Introduction of CodeChat**: `@domestos70` introduced a project called CodeChat, which allows interaction with CSV in your browser. It is available on [GitHub](https://github.com/tomasz-kielbasa/codechat) and also has a [live demo](https://codechat-six.vercel.app/).
- **German chat model, Phoenix**: `@drxd1000` unveiled a new German chat model trained with Direct Preference Optimization (DPO) called Phoenix. The model card for Phoenix can be found at [HuggingFace](https://huggingface.co/DRXD1000/Phoenix).
- **Feedback request for MermaidMistral**: `@troyfix` asked for reviews and feedback on the MermaidMistral model he created. He also made a [Reddit post](https://www.reddit.com/r/Oobabooga/comments/192qb2c/mermaidmistral_a_work_in_progress_model_for_flow/?rdt=56518) discussing the model and the importance of fine-tuned model names reflecting their capabilities.
- **YouTube Link Shared**: `@pradeep1148` shared a [YouTube link](https://www.youtube.com/watch?v=oflRFnG2j3k), but no context or explanation was provided.

**Links mentioned**:

- [DRXD1000/Phoenix · Hugging Face](https://huggingface.co/DRXD1000/Phoenix)
- [Reddit - Dive into anything](https://www.reddit.com/r/Oobabooga/comments/192qb2c/mermaidmistral_a_work_in_progress_model_for_flow/?rdt=56518)
- [GitHub - tomasz-kielbasa/codechat: Interact with CSV in your browser](https://github.com/tomasz-kielbasa/codechat): Interact with CSV in your browser. Contribute to tomasz-kielbasa/codechat development by creating an account on GitHub.
- [CodeChat](https://codechat-six.vercel.app/)
- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)
- [Chatbot UI](https://openchat.team)
- [GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data](https://github.com/imoneoi/openchat): OpenChat: Advancing Open-source Language Models with Imperfect Data - GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 messages🔥): 
        
- **Examining Inference Bottlenecks**: `@sayakpaul` called for an investigation into whether the performance issues in batched inference are due to the library or external factors. 
- **Benchmarking Diffusers**: `@raphael6419` shared a [GitHub link](https://github.com/oOraph/diffusers-benchmark) detailing their findings on diffusers' performance.
- **Batch Size Limitations**: `@raphael6419` mentioned that they were limited by the amount of VRAM available on their GPU models, unable to exceed a batch size of 16 for a 512x512 image with SD 1.5.
- **Resource on SSD-1B**: `@lunarflu` provided a [link](https://arxiv.org/abs/2401.02677) to a tech report on Stable Diffusion XL (SDXL), introducing smaller model variants.
- **Developing a Command-Based AI**: `@gamemaster123356` expressed interest in creating a text generation AI that can interact with their computer and control their home, asking for guidance on integrating Google search functionality with LLaMa. `@sayakpaul` recommended consulting other channels for advice about text generation.
- **Optimizing Diffusion Models**: `@sayakpaul` shared a [collection](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d) of papers on improving the inference latency of diffusion models.

**Links mentioned**:

- [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677): Stable Diffusion XL (SDXL) has become the best open source text-to-image model (T2I) for its versatility and top-notch image quality. Efficiently addressing the computational demands of SDXL models is...
- [Optimizing diffusion models - a sayakpaul Collection](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d)
- [GitHub - oOraph/diffusers-benchmark](https://github.com/oOraph/diffusers-benchmark): Contribute to oOraph/diffusers-benchmark development by creating an account on GitHub.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (25 messages🔥): 
        
- **Exploring Optimal Caption Formats for Text-to-Image Models**: `@pxovela` initiated a discussion around the best captioning approach for image datasets intended for text-to-image foundation models. They highlighted the challenge of varied human interpretations of images, and that fine-tuned language models such as Dalle3 can now transform prompts, thus eliminating the need for specific caption structures. `@merve3234` acknowledged the lack of consensus or comprehensive study regarding this issue.
  
- **Upcoming Releases for Computer Vision**: `@merve3234` teased two upcoming model integrations relevant to computer vision, inviting users to guess what these might be.

- **Potential Overkill of Certain Models for Object Detection**: `@merve3234` stated that some models, not specified in these messages, could be overkill and ineffective for object detection, suggesting transformer models or YOLO/S instead.

- **Limitations of LLAVA for Employee Monitoring Use Case**: In response to `@iloveh8`'s proposed use case of employing LLAVA for employee monitoring, `@meatfucker` warned of the model's tendency towards errors and hallucinations, asserting that the model might not be reliable for identifying specific tasks. They also pointed out that LLAVA's internal resolution is fairly low due to the image itself consuming a significant amount of the language model context.

- **Requests for Assistance on Custom Model Fine-Tuning and API Use**: `@swetha98` sought help for fine-tuning the donut document visual question answering model on a custom dataset. Similarly, `@jordanlancheros` faced issues using the API of the non-open source model 'OutfitAnyone', receiving a 403 error and requested for a solution to this.

**Links mentioned**:

[OutfitAnyone - a Hugging Face Space by HumanAIGC](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **Relocating Models to GPU**: @vipitis requested assistance on how to relocate a model to the GPU, `@merve3234` pointed them to add ```device = torch.device("cuda" if torch.cuda.is_available() else "cpu")``` alongside their model and inputs (`model = AutoModel.from_pretrained("suno/bark-small").to(device)` and `...to(device)`.
- **Verification of Dataset Format**: `@notooth` shared a dataset they're working on to train `llama.cpp` model to get the href and text of html tags, inquiring if the dataset is correctly formatted. 
- **Troubles with T5-V1_1_base fine tuning script**: User `@opencuiguy` sought a working T5-V1_1_base fine tuning script upon encountering a `ValueError` while trying to save what was flagged as a non-contiguous tensor. They noted that an identical code functions properly with `flan-t5-base`.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 messages🔥): 
        
- **Investigating Performance Bottleneck in Batched Inference**: User `@sayakpaul` encouraged `<@380038770520489986>` to test the batched inference and report their findings to better understand if the performance issues are due to the library or other factors.
- **Diffusion Models Benchmark Shared**: `@raphael6419` shared a link to their [GitHub repository](https://github.com/oOraph/diffusers-benchmark) that includes the code and benchmark results on diffusion models, paving the way for a more informed discussion on the topic.
- **Viable Batch Sizes for GPU Models**: `@raphael6419` shared their limitation of not being able to go above a batch size of 16 when working with a 512x512 image with SD 1.5, due to the VRAM available on the GPU models they used. `@sayakpaul` subsequently enquired about the VRAM capacity.
- **Tech Report on SSD-1B**: `@lunarflu` shared the [tech report](https://arxiv.org/abs/2401.02677) on Stable Diffusion XL (SDXL) and its scaled-down variants, SSD-1B and Segmind-Vega, discussing their generative qualities, reduction in parameters, and latency.
- **Ideation for Home Automation AI Model**: `@gamemaster123356` proposed the idea of creating an AI model that interacts with a computer and controls home appliances. They provided a system prompt structure for the AI response and inquired about how to integrate Google search into the model. `@sayakpaul` directed further discussion to appropriate channels.

**Links mentioned**:

- [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677): Stable Diffusion XL (SDXL) has become the best open source text-to-image model (T2I) for its versatility and top-notch image quality. Efficiently addressing the computational demands of SDXL models is...
- [Optimizing diffusion models - a sayakpaul Collection](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d)
- [GitHub - oOraph/diffusers-benchmark](https://github.com/oOraph/diffusers-benchmark): Contribute to oOraph/diffusers-benchmark development by creating an account on GitHub.


### ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/) (1 messages): 
        
- **Gradio 4 Goes Browser-only with Gradio Lite**: User `@yuviii_` announced that **Gradio 4**, powered by `@𝚐𝚛𝚊𝚍𝚒𝚘/𝚕𝚒𝚝𝚎`, can now work entirely within the browser, enabling the building of faster, more private serverless applications. Full release details and usage guides are available at [https://gradio.app/guides/gradio-lite](https://gradio.app/guides/gradio-lite).


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Monads Might be Next**: @slono humorously brought up the topic of monads in AI discussions.
- **OpenAI Launches ChatGPT Team**: @coffeebean6887 shared detail about the [ChatGPT Team launch](https://openai.com/chatgpt/team) by **OpenAI**, providing access to GPT-4 with a 32K context window and tools like DALL·E 3.
- **Quality Control Conundrum at GPT Store**: Users expressed criticism of the recently launched **GPT store**, pointing to a lack of quality controls and possible user confusion due to the abundance of similar offerings. @swyxio shared a [tweet](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww) that encapsulated these viewpoints.
- **A Shout-Out for Discussing AI in Applications**: @kbal11 suggested creating a space dedicated to discussing AI on the application layer. The idea was seconded by @swyxio, @swizec, and @dsquared70.
- **Open Interpreter API Goes Live**: The *launch* of Open Interpreter API, capable of pinpointing on-screen visual controls with pixel precision, was announced by @swyxio on [their site](https://api.openinterpreter.com/).
- **Mixture of Experts (Mixtral/Phixtral) Session on the Cards**: An upcoming session on "Mixture of Experts (incl Mixtral/Phixtral)" will be led by `<@206404469263433728>`. The event link is [here](https://lu.ma/llm-paper-club).
- **LLM Paper Club Goes Through Changes**: @ivanleomk shared useful links for the LLM Paper Club while @swyxio advised members to sign up on the [Lu.ma page](https://lu.ma/llm-paper-club). A significant change was reported: Lu.ma had deleted the recurring calendar feature.
- **Notes and Graphics from MoE Session Updated**: @ivanleomk requested feedback on the graphics and other updates from the MoE session.
- **Rise of DeepSeek's MoE Model**: @swyxio shared a tweet from DeepSeek AI unveiling their next-gen Large Language Models, **DeepSeekMoE**, which in early experiments matches DeepSeek 67B and vastly outperforms Gshard.
- **Insights about DeepSeek's MoE Model Performance**: @coffeebean6887 analyzed the performance of DeepSeek's MoE model, observing trade-offs and commenting on the model's efficiency, which could potentially serve as an API.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (50 messages🔥): 
        
- **It's Time for Monads**: User `@slono` humorously proposed that it was about time for discussions on monads in the AI field.
- **ChatGPT Team Launched by OpenAI**: User `@coffeebean6887` shared details about the recent launch of the [ChatGPT Team](https://openai.com/chatgpt/team) plan by OpenAI, detailing its offerings such as access to GPT-4 with a 32K context window and tools like DALL·E 3.
- **Criticism of the GPT Store**: There was some criticism of the recently launched GPT store, citing a lack of quality control and potential for user confusion with too many similar offerings. `@swyxio` shared a [link](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww) to a tweet which expressed these concerns.
- **Call for App Level AI Conversations**: User `@kbal11` proposed creating a dedicated space for discussing AI's application layer, focusing on AI engineers rather than ML researchers. The idea found support and an expression of interest from other users such as `@swyxio`, `@swizec`, and `@dsquared70`, who also shared potential discussion topics.
- **Launch of Open Interpreter API**: `@swyxio` shared the [launch](https://api.openinterpreter.com/) of the Open Interpreter API, which is capable of locating on-screen visual controls with single-pixel precision.

**Links mentioned**:

- [undefined](https://api.openinterpreter.com/)
- [Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team): We’re launching a new ChatGPT plan for teams of all sizes, which provides a secure, collaborative workspace to get the most out of ChatGPT at work.
- [Introducing the GPT Store](https://openai.com/blog/introducing-the-gpt-store): We’re launching the GPT Store to help you find useful and popular custom versions of ChatGPT.
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/AndrewCurran_/status/1744923452572852608?s=20): It should be on by default, but just in case the toggle is here;
- [Tweet from killian (@hellokillian)](https://x.com/hellokillian/status/1743469389222195680?s=20): @findmyke lol thank you so much myke! promo video is all @rotatoapp — highly recommend it.
- [Tweet from surya (@sdand)](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww): plugins had a moderation system in place which made it better because there were limits on what can be published; but it shouldve been even stricter to maintain quality.  there doesnt need to be a mil...
- [GitHub - SciPhi-AI/synthesizer: A multi-purpose LLM framework for RAG and data creation.](https://github.com/SciPhi-AI/synthesizer): A multi-purpose LLM framework for RAG and data creation. - GitHub - SciPhi-AI/synthesizer: A multi-purpose LLM framework for RAG and data creation.
- [Squish Meets Structure: Designing with Language Models](https://maggieappleton.com/squish-structure): Video, slides, and transcript from my talk on the challenges of designing with language models
- [Build a search engine, not a vector DB](https://blog.elicit.com/search-vs-vector-db/): If you want to build a RAG-based tool, first build search.
- [Why Chatbots Are Not the Future of Interfaces](https://wattenberger.com/thoughts/boo-chatbots)
- [Generative Interfaces Beyond Chat // Linus Lee // LLMs in Production Conference](https://www.youtube.com/watch?v=rd-J3hmycQs): // AbstractLinus has spent the last few years building and experimenting with new kinds of tools for thought and software interfaces for creation, like a can...


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **Upcoming Session on Mixture of Experts (Mixtral/Phixtral)**: In 15 minutes, `<@206404469263433728>` will lead a session on "Mixture of Experts (incl Mixtral/Phixtral)". The mentioned event's link is [here](https://lu.ma/llm-paper-club) and includes a [cover image](https://cdn.lu.ma/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-defaults/1-1/standard1.png) of LLM Paper Club in Latent Space Discord.
- **Latent Space Community Events**: These events are primarily weekly paper reviews of LLM papers, starting from foundational ones. The repository of the papers can be found [here](https://github.com/eugeneyan/llm-paper-notes/). Occasionally, other events are hosted as well.
- **Notifications for Discord Events**: Users can ask to be tagged in `<@&1107197669547442196>` to receive Discord notifications for these events.

**Links mentioned**:

[LLM Paper Club (in Latent Space Discord) · Luma](https://lu.ma/llm-paper-club)


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (23 messages🔥): 
        
- **LLM Paper Club Details and Registration**: @ivanleomk informed the channel that they would use the `md` file from their GitHub repo during the LLM Paper Club. @intheclouddan asked about the duration and frequency of the Club, to which @swyxio clarified that it's not necessary to block a calendar slot but rather to sign up on the [Lu.ma site](https://lu.ma/llm-paper-club). `@swyxio` later revealed that Lu.ma had removed the recurring calendar feature they used, expressing their frustration at this change. Despite this, they set up next week's event on the same platform.
  
- **Notes Presentation and Updates**: After the Paper Club session, @ivanleomk shared a PR containing the updated and cleaned notes with new graphics from the Mixture of Experts (MoE) session. They were open to feedback on potentially missed or incorrect details.

- **DeepSeek's MoE Model**: @swyxio shared a tweet from DeepSeek AI presenting their next generation of Large Language Models, DeepSeekMoE. This model, scaling up to 145B, significantly outperforms Gshard and matches DeepSeek 67B in early experiments.

- **Analysis on DeepSeek's MoE Model**: @coffeebean6887 provided their thoughts on DeepSeek's MoE's performance, pointing out the interesting trade-offs of this model. They highlighted its efficiency — the 2B model only requiring 20% computation and the 16B model using 40%. They added that despite it not being state-of-the-art on benchmarks, its efficiency could be beneficial for serving as an API.

**Links mentioned**:

- [LLM Paper Club (in Latent Space Discord) · Luma](https://lu.ma/llm-paper-club)
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335): Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong L...
- [Tweet from DeepSeek (@deepseek_ai)](https://fxtwitter.com/deepseek_ai/status/1745304852211839163?s=46&t=90xQ8sGy63D2OtiaoGJuww): 🌟 Meet #DeepSeekMoE: The Next Gen of Large Language Models!  Performance Highlights: 📈 DeepSeekMoE 2B matches its 2B dense counterpart with 17.5% computation. 🚀 DeepSeekMoE 16B rivals LLaMA2 7B wit...
- [Pull requests · eugeneyan/llm-paper-notes](https://github.com/eugeneyan/llm-paper-notes/pulls): Notes from the Latent Space paper club. Follow along or start your own! - Pull requests · eugeneyan/llm-paper-notes
- [Added some notes on Mixture of Experts by ivanleomk · Pull Request #1 · eugeneyan/llm-paper-notes](https://github.com/eugeneyan/llm-paper-notes/pull/1): Added some notes on Mixture of Experts - edits are very much welcome! Going to present with this later.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Outrage at Llama 2's Long Finetuning Time**: @direwolf365 sought tips to reduce the 67-hour estimated finetuning time for a **Llama 2 7B model**, and discussed their current configuration parameters.
- **Dating Site Data's Secret Adventures**: @leoandlibe mentioned their project in finetuning models for fake profiles using a private dataset derived from a daily 100,000 message flow from a dating site, similar to Ashley Madison, without disclosing the site's name.
- **'Unsloth' to Rescue LLM Fine-tuning**: 'Unsloth', a novel tool that speeds up LLM fine-tuning without reducing accuracy, as introduced in a [blog](https://huggingface.co/blog/unsloth-trl) shared by @caseus_, has been reported to accelerate performance for **LLaMa Factory**.
- **Instruct-tuning Marred by Dataset Difficulty**: @stoicbatman required guidance on the correct data format for instruct-tuning an open-source multimodal dataset.
- **Default System Fouls in Chat Templates**: @le_mess advocated for including default system messages in chat templates to improve chat models' training, with @dctanner agreeing to this idea, pointing to Huggingface documentation, which he claims is lacking in information regarding system messages.
- **Mistral's Issues put under Office Scanner**: Post office-hours discussion, @le_mess indicated a response to suspected issues with the Huggingface trainer for training Mistral.
- **Potential Glitches Double-checked with Huggingface**: Backing @le_mess's point about possible problems with the Huggingface's trainer for Mixture of Experts (MoE) models, @casper_ai expressed plans to discuss this with the Huggingface team.
- **In Pursuit of MLX**: The implication of integrating MLX into **Axolotl** was raised by @caseus_, however, no concluding decision was articulated.
- **BatchSamplerDataCollatorForSeq2Seq Misbehaving?**: An issue with BatchSamplerDataCollatorForSeq2Seq not properly constructing batches was raised by @caseus_, with a reference to a related [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082).
- **Exit Code -6 Mysteries**: An exit code -6 during program execution, which often signifies process termination due to resource exhaustion, was unraveled by @caseus_.
- **Checkpointing Saves the Day**: Despite facing an unexpected program termination, @noobmaster29 successfully restarted the program owing to the efficient checkpointing feature.
- **Decoding Raw Text Training**: @confident_dolphin_26428 exhibited curiosity in understanding how raw text training works for the model during sentence completion.
- **The Quest for Evaluation Datasets**: @morgymcg exhibited interest in understanding the connection between evaluation dataset and model performance, reminding the community about the often laid-back correlation between evaluation loss and model performance.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (19 messages🔥): 
        
- **llama 2 finetuning time query**: @direwolf365 shared their configuration details for finetuning a llama 2 7B model and asked the community for tips to reduce the estimated 67-hour finetuning time. The configuration is as follows: Dataset fitted in 19k rows with each row containing 4096 tokens, using an 80GB GPU, batch size: 4, Peft method: Qlora, Quantization: 4 bits, Epochs: 1, Rank: 1, Grad accum steps: 4, Learning rate: 0.0001, Optimiser: adamw_apex_fused.
- **fine-tuning dating site data**: @leoandlibe revealed how they are using a private dataset, created from a database for a site similar to Ashley Madison, which sees about 100,000 messages daily to make finetunes for fake profiles. However, they did not disclose the name of the website.
- **Introduction of Unsloth - LLM Fine-tuning optimization tool**: @caseus_ posted a link to a [blog](https://huggingface.co/blog/unsloth-trl) that introduces 'Unsloth', a tool that speeds up LLM fine-tuning, reducing memory usage without degrading accuracy. It was reported that LLaMA Factory incorporated Unsloth, witnessing improvements in speed.
- **Question on instruct-tuning for a multimodal dataset**: @stoicbatman asked the community for guidance on the appropriate data format to use for instruct-tuning an open-source multimodal dataset.

**Links mentioned**:

- [Make LLM Fine-tuning 2x faster with Unsloth and 🤗 TRL](https://huggingface.co/blog/unsloth-trl)
- [Home](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison>): Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM) - hiyouga/LLaMA-Factory


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (18 messages🔥): 
        
- **Adding a Default System Message to ChatML Templates**: User `@le_mess` proposed adding a default system message to the chat templates for chatml, noting that none of the models were trained without a system message. `@dctanner` agreed, expressing that this could address some issues he had encountered regarding system messages, particularly since the Huggingface documentation does not adequately cover system messages ([Huggingface docs](https://huggingface.co/docs/transformers/main/en/chat_templating)).
- **Office Hours for Mistral**: `@le_mess` announced office hours for discussing Mistral. After the meeting, he noted that the team suspects there may be issues with the Huggingface trainer when it comes to training Mixtral.
- **Potential Issues with the Huggingface Trainer**: `@casper_ai` noted that there could indeed be problems with the Huggingface trainer for Mixture of Experts (MoE) models and expressed his intention to discuss this with the Huggingface team.
- **Discussion on integrating MLX into Axolotl**: User `@caseus_` raised the question of integrating MLX into Axolotl. However, no further discussion or conclusion was provided in the given messages.
- **Issue with BatchSamplerDataCollatorForSeq2Seq**: `@caseus_` mentioned a potential issue with the BatchSamplerDataCollatorForSeq2Seq not properly constructing batches and suggested discussing it in the discord in reference to a [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082).

**Links mentioned**:

- [Templates for Chat Models](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [BatchSamplerDataCollatorForSeq2Seq does not properly construct batches · Issue #1082 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior Batches should be of shape (micro_batch_size, seque...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (5 messages): 
        
- **Unraveling exit code -6**: User `@caseus_` explained that an **exit code -6** typically means the process executing the task was terminated by signal 6 (SIGABRT). This usually happens when the process runs out of memory or some other resource.
- **Checkpointing Saves the Day**: Despite not noticing any significant memory pressure, `@noobmaster29` experienced a program termination. Luckily, the checkpointing capability allowed them to restart the program successfully.
- **Understanding Raw Text Training**: User `@confident_dolphin_26428` queried about how training actually works when doing completion training with raw text, questioning the model's process of predicting the first token from the given sentence.
- **In Search for Evaluation Datasets**: `@morgymcg` was curious about the datasets that people are currently using for evaluation, raising that evaluation loss often doesn't directly correlate with the model's final performance.


### ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=oflRFnG2j3k


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Echo in the Matrix Raises Eyebrows**: @brknclock1215 voiced concerns about identical responses generated by two distinct models, pondering whether they may be pulling from shared context or material. 
- **Precision Pursuit in Mixtral 8x7B**: @simon_18724 sparked a discourse on the running precision for **Mixtral 8x7B**.
- **Claude 2.1 Sees Sounds**: @groagji spotlighted an instance where **Claude 2.1** feigned a non-existent source when parsing brief music descriptions from Soundcloud.
- **Tick Tock, Widgets on the Clock**: Replying to @barry_zyj's query, @ok.alex confirmed the impending release of a **Perplexity Android widget**.
- **Perplexity vs. Copilot: Standalone Steals the Show**: @moyaoasis voiced a preference for the pure Perplexity model in opposition to the Copilot variant for search tasks, accusing **Copilot with GPT4** of weakening the performance of both base models.
- **Pplx-70b-online API, an Artist's Dream**: @blenderrenaissance capitalized on **pplx-70b-online API** to generate accurate, non-hallucinatory data for a graph project.
- **Digging the Hyperlink Connection**: @abhidalal praised the tool's knack for linking responses with internet results.
- **Perplexity Unboxing**: While newcomer @arti_xartous_14963 couldn't contain their anticipation to get started with Perplexity, @aug_li concurred it's a commendable product.
- **Thawing the AI Winter**: @.anuni shared a [Perplexity AI link](https://www.perplexity.ai/search/What-can-you-X31qYfR8TeuMS_.PVVUBtw?s=c) in hope of circumventing a looming AI winter.
- **Credit Card Woes with Octane**: @yash_89789 reported card rejection trouble when attempting to add credits in Octane.
- **Billing Issues? Support is the Answer**: @icelavaman advised reaching out to support@perplexity.ai for billing-related problems, underscoring that declined cards are beyond Perplexity's jurisdiction and fall to the payment platform (**Stripe**) or the user's bank.
   
Relevant link to explore: [Gangsta Chess by Renosis](https://www.thingiverse.com/thing:8930) and [Image gallery](https://discord.com/channels/1047197230748151888/1194788138124587128).

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (27 messages🔥): 
        
- **Possible Repeated Information Across Models**: `@brknclock1215` found it peculiar that two different models generated identical responses, questioning if the models might be parroting from the same material or source given as context.
- **Running Precision Inquiry**: User `@simon_18724` inquired about the running precision for **Mixtral 8x7B**.
- **Claude 2.1 Hallucination Issue**: `@groagji` highlighted an instance of **Claude 2.1** hallucinating a non-existing source while dealing with limited descriptions from music on Soundcloud.
- **Upcoming Android Widget Release**: In response to `@barry_zyj`'s query about the existence of a Perplexity Android widget, `@ok.alex` confirmed that the **widget will be released soon**.
- **Preference for Pure Perplexity Over Copilot**: User `@moyaoasis` expressed a preference for the pure Perplexity model over the Copilot version for search tasks, alleging that the **Copilot with GPT4 dumbs down both pure-GPT4 and pure-Perplexity model**.


**Links mentioned**:

[Gangsta Chess by Renosis](https://www.thingiverse.com/thing:8930): This is Gangsta Chess! Now you can wage your own criminal underground turf war in your home! This design is a mashup/derivative based on the original gangsta, created by the OG himself, Yzorg. I desig...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (13 messages🔥): 
        
- **Perplexity AI in practice**: `@blenderrenaissance` used the **pplx-70b-online API** to generate accurate, non-hallucinatory data for a graph product.
- **More use of Perplexity AI**: `@abhidalal` expressed admiration for the tool's ability to link a response with internet results.
- **Sharing images in the channel**: `@ok.alex` shared a [gallery](https://discord.com/channels/1047197230748151888/1194788138124587128) for users to share images.
- **First look reactions**: New user `@arti_xartous_14963` expressed excitement about starting with Perplexity, while `@aug_li` stated it's a good product.
- **AI Winter contemplation**: `@.anuni` shared a [Perplexity AI link](https://www.perplexity.ai/search/What-can-you-X31qYfR8TeuMS_.PVVUBtw?s=c) hoping it can help avoid a potential AI winter.


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (2 messages): 
        
- **Billing woes with Octane**: `@yash_89789` raised a query about being unable to add credits in Octane due to his card getting rejected. 
- **Direct to support for billing issues**: `@icelavaman` suggested the user to reach out to support@perplexity.ai for addressing the billing problem and highlighted that Perplexity **cannot** rectify declined cards, it's in the hands of the payment platform (**Stripe**) or user's bank.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Token Trouble in Langchain**: @_hnch brought up an issue where Langchain's RetrievalQA retrieval hits a **token overflow** even with a low token count of 2.
- **Memory Requirements of Vector Databases**: @__something queried about the memory implications of implementing vector databases in a desktop chat app. They also asked if it was feasible to merge this with local LLMs like **Phi** or **Mistral**.
- **Plugging LllamaCPP into Langchain**: @vamseekrishna sought advice on integrating **LllamaCPP** into Langchain's conversational retrieval chain, highlighting the absence of such examples in the documentation.
- **Langchain Newbie's Journey**: Beginner Python programmer @ilovesass showed interest in learning about Multimodal models and **Langchain**. Responding to this, @fareswastaken recommended a path starting with basic programming, then Python, and finally tackling Langchain.
- **Unruly Langchain Agents**: @hiranga.g reported an issue where Langchain agents became non-compliant with forced prompts, demonstrated by an agent refusing to begin replies with **code '112'** even when prompted to do so.
- **Parallel Processing in Langchain**: @shivam51 asked if there was a way to make **parallel LLM calls using Langchain**.
- **Debugging Deep Dive in LangChain**: @veryboldbagel gave guidance on debugging LangChain code effectively, using print statements to inspect intermediate outputs, and additionally revealed a debugging mode through LangChain's core runnables.
- **Mind the GitHub Query Method Issue**: @cryptossssun created an issue in the LangChain GitHub repository asking for help with making new variable inputs available via the query method, providing the link here: [GitHub Issue](https://github.com/langchain-ai/langserve/issues/393).
- **Optimize Langchain RAG with Parea AI**: @parea.ai posted a detailed [tutorial](https://docs.parea.ai/tutorials/getting-started-rag) on optimizing a Langchain RAG application using Parea AI Evals and Tracelogs, focusing on applications that allow users to chat with public financial PDF documents.
- **Is Martian in the picture?**: @swyxio raised a somewhat cryptic query, asking if an unspecified entity was **the same as 'Martian'**.
- **Documentation Frustrations**: @archiacme encountered numerous errors while using the LangChain AI's documentation, calling into question the accuracy of the provided set-up instructions.

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (15 messages🔥): 
        
- **Token Overload in Langchain**: User `@_hnch` highlighted an issue where Langchain's RetrievalQA retrieval exhibits a **token overflow** even when the tokens count is as low as 2.
- **Incorporating Vector Databases in Chat Apps**: `@__something` queried about the memory requirements of using vector databases in a desktop chat app. They further asked about combining this with lightweight local LLMs, like **Phi** or **Mistral**.
- **Need for Documentation**: `@vamseekrishna` asked for guidance on integrating **LllamaCPP** into Langchain's conversational retrieval chain, mentioning the absence of related examples in the existing documentation.
- **Aspiring Langchain Developer**: User `@ilovesass` expressed their current status as a beginner in Python and their interest in learning about Multimodal models and **Langchain**. In response, `@fareswastaken` advised starting with basic programming, moving to Python, then tackling Langchain.
- **Issues with Agents' Compliance**: `@hiranga.g` reported an issue where Langchain agents became non-compliant with forced prompts. They discovered this when the agent stopped following a prompt to always start replies with **code '112'** after receiving user input instructing it to stop.
- **Parallel LLM Calls in Langchain**: `@shivam51` enquired if there was a procedure to make **parallel LLM calls using Langchain**.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (6 messages): 
        
- **Debugging Chain Steps in LangChain Code**: `@veryboldbagel` suggested breaking down the LangChain code to debug it effectively. They described a method using print statements to inspect intermediate outputs. The example provided was a code snippet where a function `print_me` was used to print intermediate results within a processing chain.
- **Using Debug Mode in LangChain**: `@veryboldbagel` also revealed how to leverage the debug mode in LangChain for effective troubleshooting. The debug mode can be enabled through `globals.set_debug(True)` and used in conjunction with LangChain's core runnables.
- **Testing Recommendation for New LCEL Primitives**: `@veryboldbagel` recommended users who are employing new LCEL primitives to start with simple test cases along with print statements for in-depth understanding and debugging.
- **Issue Submitted for Query Method Help**: `@cryptossssun` mentioned having submitted an issue to the LangChain GitHub repository requesting clarification about how to make new variable inputs available via the query method. They included the [link to the GitHub issue.](https://github.com/langchain-ai/langserve/issues/393)
- **Call for Assistance from LangChain Team**: `@cryptossssun` tagged another user asking for help, but the details of what they asked for are unknown.

**Links mentioned**:

[How to make the new variables input available via query method? · Issue #393 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/393): Question: If I create the new varialbels: input_variables=[&quot;history&quot;, &quot;input&quot;,&quot;lession&quot;, &quot;affection&quot;], and setting like the below code. I cant make the right qu...


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **Optimizing Langchain RAG with Parea AI**: `@parea.ai` shared a tutorial on how to optimize a Langchain RAG application using Parea AI Evals and Tracelogs. The tutorial offers a step-by-step guide to streamlining an application that **lets users chat with public financial PDF documents like Nike’s 10k filings**. The tutorial also covers various application components, including [`UnstructuredFileLoader`](https://python.langchain.com/docs/integrations/document_loaders/unstructured_file), [`RecursiveCharacterTextSplitter`](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter), `all-MiniLM-L6-v2` sentence transformer, [Redis](https://redis.com/solutions/use-cases/vector-database/) as the vector database, Langchain OpenAI `gpt-3.5-turbo-16k` to generate answers, and Parea AI for Trace logs, Evaluations, and Playground. The tutorial can be found [here](https://docs.parea.ai/tutorials/getting-started-rag).
- **Is it similar to Martian?**: Earlier, `@swyxio` asked if something — it's unclear from the context what this refers to — is the **same idea as 'Martian'**. There were no responses or follow-up to this question.

**Links mentioned**:

[Optimize a RAG application - Parea AI](https://docs.parea.ai/tutorials/getting-started-rag)


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **User encounters errors with documentation**: User `@archiacme` reported getting numerous errors while working through the LangChain AI's documentation including the cookbooks, templates, and the "getting started" section. These problems persisted despite attempts at local execution (both with venv and conda) and on Google Colab, indicating potential issues with the set-up instructions in the provided documentation.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **OCR Scoffed at in Text Evaluation**: `@evan_04487` proposed a new heuristic to evaluate text without resorting to OCR. If a substantial portion of the document passes checks for paragraph structure, tables, and non-garbage content, OCR might not be necessary.
- **GPT-4 Visits Enterprise Market**: As shared by `@jeffreyw128`, OpenAI has successfully opened up **ChatGPT Enterprise** to major companies, and introduced the new self-serve plan **ChatGPT Team**, providing access to advanced models such as **GPT-4** and **DALL·E 3**, along with other tools and features - [read more here](https://openai.com/blog/introducing-chatgpt-team). JefferyW128 is curious whether the **GPT-4** being referred to is the regular or turbo version.
- **GPT Store Versus Plugins: The Showdown**: In response to the unveiling of the **GPT Store**, a dialogue opened up among `@jeffreyw128`, `@thebaghdaddy`, and `@res6969` about the added value of the store over plugins. The consensus, in light of Jeffreyw128's comparison of the GPT Store to plugins and their related performance issues, leaned more towards the potential impacts of plugins. `@thebaghdaddy` marginally saw custom GPTs as a time saver for writing prompt instructions, but not much else.
- **Unpacking the GPT Store's Limitations**: `@res6969` expressed that despite the lure of the API calling feature offered by GPTs in the store, their tunability falls short, exciting a broader discussion on potential limitations compared to other software items.
- **GPT Store Quality Concerns**: The quality of the GPTs in the **GPT Store** were put under scrutiny, with `@jeffreyw128` and `@res6969` concluding that exposure provided by the store won't make up for inferior product quality, a sentiment confirmed by `@thebaghdaddy` who experienced first-hand the disappointing quality.
- **Tree of Thought Prompts**: `emrgnt_cmplxty` proposed employing a tree of thought prompts approach, although further detail or context was not provided.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 messages): 
        
emrgnt_cmplxty: tree of thought prompting might help


### ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/) (1 messages): 
        
- **Evaluating Text Without OCR**: `@evan_04487` proposed a scheme to evaluate text without using OCR. The heuristic checks if the document has legit paragraph text, tables, or is considered garbage. If a certain percentage of the page passes these checks, it likely doesn't need OCR.


### ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/) (13 messages🔥): 
        
- **Introducing ChatGPT Enterprise**: `@jeffreyw128` shares a [link](https://openai.com/blog/introducing-chatgpt-team) to a blog post by OpenAI that reveals major companies are using ChatGPT Enterprise. The new self-serve plan, **ChatGPT Team**, is introduced, offering access to advanced models like **GPT-4** and **DALL·E 3**, as well as other tools and features. Jeffreyw128 is interested in whether the **GPT-4** referred to is the regular or turbo version.
- **The Great GPT Store Debate**: `@jeffreyw128` raises questions about the value of the newly introduced **GPT Store**. Comparing it to the functionality of plugins, he states that Metaphor decided not to invest time into it due to performance concerns.
- **GPTs vs. Plugins**: Responding to Jeffreyw128, `@thebaghdaddy` sees more potential impact in plugins, viewing custom GPTs as a time saver for writing prompt instructions, but not much else beyond that.
- **The Limits of GPT Store**: `@res6969` echoes the same sentiment, citing that while the API calling feature of GPTs is interesting, it falls short in its tunability, opening a discussion about the GPT's limitations compared to other software products.
- **GPT Store - Yay or Nay?**: Discussing the possible review system and the quality of GPTs, `@jeffreyw128` and `@res6969` conclude that although the exposure provided by the GPT Store is beneficial, the quality of the products is paramount. Thebaghdaddy experiences first-hand the subpar quality of the GPTs in the store.

**Links mentioned**:

[Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team): We’re launching a new ChatGPT plan for teams of all sizes, which provides a secure, collaborative workspace to get the most out of ChatGPT at work.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **First Efficient Mixture of Experts Model, PhiXtral, Debuts**: `@philipmay` brought attention to an [article](https://www.linkedin.com/posts/maxime-labonne_phixtral-i-made-the-first-efficient-mixture-activity-7150758415961620481-v0qx?utm_source=share&utm_medium=member_ios) discussing **PhiXtral**, a new Mixture of Experts (MoE) model assembled from multiple phi-2 models. The exact method of assembly and its difficulty level remained a point of curiosity.
- **German Mixtral May Go the Random Forest Path**: `@philipmay` raised the idea of fabricating a 'German Mixtral' by training different mistral models on separate tasks and then merging them, similar to a random forest for Large Language Models (LLMs).
- **MIT Blesses Phi-2 with a License Upgrade**: `@rasdani` informed that phi-2 has moved beyond its research-only restrictions and has now been licensed under the Massachusetts Institute of Technology (MIT) License.
- **MoE Models: Navigating the Routing Riddle**: `@philipmay` showed concerns about the intricacies of routing in MoE models. However, `@rtyax` countered that if the scenario only included two experts, a trained router might be unnecessary as all requests could be directed to both experts.
- **The Inner Workings of Mixtral Implementation Revealed**: In response to `@philipmay`'s queries about collating multiple models for creating MoE, `@remek1972` shared insights into an ongoing `Mixtral implementation using mergekit` and provided a [link](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py) to the mergekit scripts. He emphasized the vital role of router configuration in model's mergekit files.
- **Conditional Pretraining Could Be A Game-Changer for Alignment**: `@huunguyen` proposed that employing a conditional pretraining style could result in enhanced prompt alignment.
- **"Bad" Personas Might Be Good for Us After All**: `@huunguyen` also suggested a strategy to boost performance by integrating "bad" personas, generated from a shared script, with the positive ones from `open-orca`.

**DiscoResearch Channel Summaries**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (9 messages🔥): 
        
- **PhiXtral: The First Efficient Mixture of Experts Model**: `@philipmay` shared an [article](https://www.linkedin.com/posts/maxime-labonne_phixtral-i-made-the-first-efficient-mixture-activity-7150758415961620481-v0qx?utm_source=share&utm_medium=member_ios) discussing the development of a new Mixture of Experts (MoE) model called **PhiXtral**, that combined multiple phi-2 models. They intrigued about the specific process of combination and its difficulty.
- **Exploring German Mixtral through Random Forest for LLMs?**: `@philipmay` proposed a strategy for building 'German Mixtral' by training multiple mistral models on different tasks and subsequently combining them, akin to a random forest approach for Large Language Models (LLMs).
- **Phi-2 gets MIT license status**: `@rasdani` informed the channel that phi-2 is no longer restricted for research purposes only; it's now licensed under the Massachusetts Institute of Technology (MIT) License. 
- **Routing Challenges in Building MoE Models**: `@philipmay` voiced concerns about the complexities of routing in MoE models. n However, `@rtyax` suggested if there are just two experts, a trained router might not be necessary as all requests can be routed to both experts. 
- **Insights into Mixtral Implementation**: `@remek1972` answered to `@philipmay`'s query about combining multiple models for creating MoE. He mentioned an ongoing `Mixtral implementation using mergekit` and shared a [link](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py) to the mergekit scripts. Further, he highlighted that router configuration is essential in model's mergekit files.

**Links mentioned**:

[mergekit/mergekit/scripts/mixtral_moe.py at mixtral · cg123/mergekit](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py): Tools for merging pretrained large language models. - cg123/mergekit


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (2 messages): 
        
- **Interest in Conditional Pretraining for Alignment**: User `@huunguyen` proposed the notion of training a system with a conditional pretraining style for better prompt alignment.
- **Leveraging "Bad" Personas for Better Performance**: `@huunguyen` also shared a script that has the capability to create "bad" personas. By integrating these with the good personas from `open-orca`, he believes it could lead to an **enhancement in performance**.


        

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **GPT Models Plugging the Prompt Leak**: @realgavok praised the current GPT models due to their "*impressive measures to prevent prompt leaks*" and cited personal testing reflecting their robustness.
- **A Public Cry for GPT's Secrecy Strategy**: @tariqali responded by proposing a need for public disclosure of the models' protection strategies if they are indeed so effective.
- **In-Depth Codebase Analysis with LLM**: @00brad initiated a query about embedding a large codebase into LLM, pondering whether to input every file, function, or class to gain insights on potential fixes or modifications.

**Datasette - LLM (@SimonW) Channel Summaries**

### ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (4 messages): 
        
- **GPT Models Achieving Effective Prompt Leak Safeguard**: `@realgavok` remarked that the featured GPT models possess impressive measures to prevent prompt leaks.
- **Call for Public Disclosure of Prompt Protection Strategies**: In response, `@tariqali` expressed interest in learning about these protection measures and suggested that if these strategies are effective, they ought to be made public.
- **GPTs' Robust Protection Evident in Testing**: `@realgavok` highlighted their inability to bypass these safeguard measures despite attempting various strategies, further testifying the robustness of the GPTs' prompt protection.


### ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/) (1 messages): 
        
- **Embedding Codebase in LLM**: `@00brad` asked for advice on how to best embed a substantial codebase into LLM. The user is considering whether to embed **every file, function, or class**, with the goal of seeking LLM's insights on fixing or modifying the codebase.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **The Delicate Art of Training MoEs**: `@baptistelqt` shared insights on training Mixtures of Experts (MoEs). They found that maintaining a balance between *general* and *specialized* knowledge, rather than focusing on absolute specialization (80% or more), leads to optimal results.
- **Token-Based Specialization, A Surprising Twist**: In a follow up, `@baptistelqt` reported a counter-intuitive finding that *vanilla auxiliary loss* promotes specialization over different types of tokens, unlike expected topic or domain-specific learning. However, they highlighted this conclusion is based on personal experiments and is not definitively proven.
- **Unidentified YouTube Share**: User `pradeep1148` shared a [YouTube Video](https://www.youtube.com/watch?v=oflRFnG2j3k) in the off-topic channel, with no further context or discussion around it. This content might not be directly related to the discussions held in other channels.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (3 messages): 
        
- **MoEs' Specialization, a Balancing Act?** : `@baptistelqt` shared insights on training Mixtures of Experts (MoEs), indicating that encouraging too much specialization (80% or more) in one domain can lead to degraded performance. However, a blend of general and specialized knowledge seems beneficial.
- **Token-Based Specialization in MoEs? Not so Fast** : In a follow-up message, `@baptistelqt` introduced a counter-intuitive finding: vanilla auxiliary loss appears to promote specialization over different types of tokens, as opposed to specific topics/domains. It was noted that this conclusion is based on personal experiments and hence not definitively proven.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=oflRFnG2j3k


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **AI Creates a Magical Journey in Candyland**: `@magusartstudios` shared a glimpse of their project involving an AI exploring a procedurally generated world, powered by a local Lua Chatbot, and diverse external AI models. The AI enhances the immersive experience with generated ambient music, emotes, effects, and emojis. The glimpse of this project can be seen [here](https://www.youtube.com/watch?v=TzdtKw1vGA0).
- **Text-Vision Module Revolutionizes AI Interactivity in Roblox**: `@magusartstudios` developed an open-source Text-Vision module for Roblox AI agents, enabling them to possess individual memories and identities. Detailed discussion on its code and progression can be found [here](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967).
- **AI Agent Dances into Storytelling in Roblox**: Demonstrated by `@magusartstudios`, an AI agent, Zephyr 7B, uses their ChatModule library in Roblox to capture player attention by storytelling and following them around. This was showcased in a [YouTube video](https://www.youtube.com/watch?v=rMBLZtPmlsQ).
- **OpenChat 3.5 Sets New Standards**: `@imonenext` revealed the release of **OpenChat-3.5 Update 0106**, boasting superior performance to **Grok-0 (33B)** across all 4 benchmarks and averaging better than **Grok-1** across 3/4 benchmarks.
- **OpenChat Simplifies Deployment**: For deployment enthusiasts, `@imonenext` shared complete instructions on serving OpenChat models with an accelerated vLLM backend, API key authentication, and more on their [GitHub page](https://github.com/imoneoi/openchat). OpenChat 3.5 version can be interacted live on the [demo site](https://openchat.team) and is also available on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106).

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (1 messages): 
        
- **Exploring Candyland with AI powered by Lua Chatbot and External AI models**: `@magusartstudios` discussed their project where a procedural world is explored by an AI, powered by their local Lua Chatbot and several external AI models. The Chatbot can generate ambient music, emotes, effects, and emojis for added expressiveness in the AI interactions. The project was showcased in a [YouTube video](https://www.youtube.com/watch?v=TzdtKw1vGA0).
- **Open Sourced Text-Vision Module for Roblox AI Agents**: `@magusartstudios` also mentioned an open-sourced Text-Vision Module for Roblox AI Agents, which they developed. It offers AI agents individual memories and identities, demonstrated with the example of Zephyr 7B exploring Candyland. The detailed code and its progression were discussed and can be found [here](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967).
- **Storytelling Dancing AI Agent in Roblox**: In another project, `@magusartstudios` demonstrated a storytelling dancing AI agent, Zephyr 7B, using their ChatModule library in Roblox, which follows the player around. The demonstration can be seen in this [YouTube video](https://www.youtube.com/watch?v=rMBLZtPmlsQ).


**Links mentioned**:

- [Candyland Adventures - Lumina &amp; Darkness Feat. Zephyr 7-B Powered By Awareness Module, Memories](https://www.youtube.com/watch?v=TzdtKw1vGA0): I am working on a procedurally generated world built on Kingdom Hearts esque lore an. This is a Studio Test, and assisted with AI agents as chatbots embodied...
- [Tweet from Text Vision Self-Awareness Module LLM Utility Synthetic Text Data Generator](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967): I saw this youtube video so I’m writing this module that does something similar to what this programmer is doing in this video. I want to post it here to provide inspiration and resources to other dev...
- [Storytelling Dancing AI Agent Party Member ROBLOX Zephyr 7B with Emojis](https://www.youtube.com/watch?v=rMBLZtPmlsQ): In this video my ChatModule library parses a response from Zephyr 7B hosted on huggingface on ROBLOX. This NPC is a Party member that follows the player arou...


### ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/) (1 messages): 
        
- **OpenChat 3.5 surpasses Grok in performance**: `@imonenext` announced the release of **OpenChat-3.5 Update 0106**, which is touted as the World's Best Open Source 7B Large Language Model, surpassing **Grok-0 (33B)** across all 4 benchmarks and **Grok-1** on average and 3/4 benchmarks.
- **Enhanced Training and Skills**: The update enhances the training methodology, in-context learning, and coding skills, outperforming the previous 1210 release on 7 out of 8 benchmarks.
- **Available on Multiple Platforms**: The model is available live on their [demo site](https://openchat.team), HuggingFace, and [GitHub](https://github.com/imoneoi/openchat).
- **Tutorials for Deployment**: For those interested in deployment, visit their [GitHub](https://github.com/imoneoi/openchat) for full instructions on serving OpenChat models with an accelerated vLLM backend, API key authentication, and more.
- **Community Support**: `@imonenext` thanks `<@317006433797537792>`, `<@748528982034612226>`, `<@312370916820779040>`, and `<@1129298496869122088>` for contributing to this release.


**Links mentioned**:

- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)
- [Chatbot UI](https://openchat.team)
- [GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data](https://github.com/imoneoi/openchat): OpenChat: Advancing Open-source Language Models with Imperfect Data - GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data
- [Tweet from OpenChat (@openchatdev)](https://fxtwitter.com/openchatdev/status/1744985660870795635): 🚀Announcing OpenChat-3.5 Update 0106: 𝗪𝗼𝗿𝗹𝗱’𝘀 𝗕𝗲𝘀𝘁 𝗢𝗽𝗲𝗻 𝗦𝗼𝘂𝗿𝗰𝗲 𝟳𝗕 𝗟𝗟𝗠!  Experience ChatGPT & Grok-level AI locally 💿!   Surpassing Grok-0 (33B) across all 4 benchmarks and G...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/193362r/new_model_openchat_35_update_0106/)


        

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


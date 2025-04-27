---
id: d0f32680-11a3-4948-96e6-fe80c7b8e327
title: '12/30/2023: Mega List of all LLMs'
date: '2023-12-31T10:23:31.628480Z'
original_slug: ainews-12302023-mega-list-of-all-llms
description: >-
  **Stella Biderman**'s tracking list of **LLMs** is highlighted, with resources
  shared for browsing. The **Nous Research AI** Discord discussed the **Local
  Attention Flax** module focusing on computational complexity, debating linear
  vs quadratic complexity and proposing chunking as a solution. Benchmark logs
  for various LLMs including **Deita v1.0** with its **SFT+DPO** training method
  were shared. Discussions covered model merging, graded modal types, function
  calling in AI models, and data contamination issues in **Mixtral**. Community
  insights were sought on **Amazon Titan Text Express** and **Amazon Titan Text
  Lite** LLMs, including a unique training strategy involving bad datasets.
  Several GitHub repositories and projects like **DRUGS**, **MathPile**,
  **CL-FoMo**, and **SplaTAM** were referenced for performance and data quality
  evaluations.
companies:
  - nous-research
  - hugging-face
  - amazon
  - mistral-ai
models:
  - deita-v1.0
  - mixtral
  - amazon-titan-text-express
  - amazon-titan-text-lite
topics:
  - local-attention
  - computational-complexity
  - benchmarking
  - model-merging
  - graded-modal-types
  - function-calling
  - data-contamination
  - training-methods
people:
  - stella-biderman
  - euclaise
  - joey00072
---


<!-- buttondown-editor-mode: plaintext -->Stella Biderman often mentions her tracking list of LLMs - it came up again today in the Eleuther discord. good to browse: https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0

(gist form: https://gist.github.com/veekaybee/f8e589fea42ba7131e4ca0a0f280c0a4?utm_source=ainews&utm_medium=email)

 ![image.png](https://assets.buttondown.email/images/26befbe3-ecd2-4f3f-9d66-6fc0b949fe37.png?w=960&fit=max) 

also, notable image AI activity in Huggingface-land

https://www.youtube.com/watch?v=ApcJ1UyLQB8&feature=youtu.be

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Detailed examination of the **Local Attention Flax** module with a focus on computational complexity. Relevant discussions included a linear vs quadratic complexity debate, confusion over code implementation, and posited solutions such as chunking data. Two GitHub repositories were shared for reference ([repo1](https://github.com/lucidrains/local-attention-flax/blob/main/local_attention_flax/local_attention_flax.py), [repo2](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8)).
- Conversations surrounding various topics such as using AI in board games, launching MVP startups, critique of Lex Fridman's interview style, income generation via social media, and a click-worthy [YouTube video](https://www.youtube.com/watch?v=cs1TDTOby58) critically examining RAG's retrieval functionality in OpenAI's Assistants API.
- Sharing of **benchmark logs for different LLMs** including Deita v1.0 with reference to a specific **Large Language Model (LLM) training method**, Deita SFT+DPO ([log link](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/deita-v1.0-Mistral-7B.md)).
- Sharing and discussion of various projects and tools like [DRUGS](https://github.com/EGjoni/DRUGS), [MathPile](https://arxiv.org/abs/2312.17120), [Deita](https://github.com/hkust-nlp/deita), [CL-FoMo](https://sites.google.com/view/irinalab/blog/continued-pretraining-blog), and [SplaTAM](https://spla-tam.github.io/). Key points included benefits, data quality considerations, and evaluations of performance efficiency.
- Extensive dialogue about the implications of merging models with different architectures, the potential use of **graded modal types**, training combined models, best AI models for function calling, and data contamination issues in Mixtral ([GitHub link to logs](https://github.com/uukuguy/multi_loras), [HuggingFace link to model merge](https://huggingface.co/uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b)).
- Community insights requested for Amazon's new LLMs, **Amazon Titan Text Express and Amazon Titan Text Lite**. A unique training strategy involving bad dataset utilization was proposed and discussed along with the search for a catalogue of ChatGPT missteps ([Amazon Titan Text release link](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/)).

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (7 messages): 
        
- **Local Attention Computation**: `@euclaise` referred to a clever masking and einsum method, and linked to a [GitHub repository](https://github.com/lucidrains/local-attention-flax/blob/main/local_attention_flax/local_attention_flax.py) for a Local Attention Flax module.
- **Complexity Query**: `@joey00072` questioned the operation complexity of the method, suggesting it was quadratic (`n^2`) rather than linear (`nxw`). `@euclaise` confirmed that it should be linear (`nxw`).
- **Confusion Over Code**: `@joey00072` expressed confusion over a particular [code segment](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8) which appears to show a cubic (`n^3`) operation.
- **Suggested Solution for Local Attention**: `@euclaise` suggested a potential solution of chunking the data and applying attention over the chunks.


**Links mentioned**:

- [local-attention-flax/local_attention_flax/local_attention_flax.py at main · lucidrains/local-attention-flax](https://github.com/lucidrains/local-attention-flax/blob/main/local_attention_flax/local_attention_flax.py): Local Attention - Flax module for Jax. Contribute ...
- [local-attention-flax/local_attention_flax/local_attention_flax.py at e68fbe1ee01416648d15f55a4b908e2b69c54570 · lucidrains/local-attention-flax](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8): Local Attention - Flax module for Jax. Contribute ...


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (26 messages🔥): 
        
- **AI Project Via Valkyrie**: `@mnt_schred` discussed their project that uses [Valkyrie](https://npbruce.github.io/valkyrie/) to create AI-generated scenarios for the board game Mansions of Madness. They pondered which Nous model would be the best storyteller, considering Trismegistus.
- **Discussions on Launching MVP Startup**: `@fullstack6209` queried about the future implications of launching an MVP startup with an "e/acc" discount. `@teknium` humorously suggested it may lead to regret.
- **Criticism of Lex Fridman's Interview Style**: `@fullstack6209` critically evaluated Lex Fridman's interviewing skills, describing them as poor and lacking in insight and context. This sentiment was endorsed by `@teknium`.
- **Discussion on Social Media Influencers**: `@gabriel_syme` expressed amazement at how individuals can earn significant income through social media posts.
- **Exploration of AI YouTube Content**: `@fullstack6209` recommended a [YouTube video](https://www.youtube.com/watch?v=cs1TDTOby58) which critically examines RAG's retrieval functionality in OpenAI's Assistants API. `@gabriel_syme` concurred, citing personal experience with RAG's issues in real-world application deployment.

**Links mentioned**:

- [Valkyrie GM for Fantasy Flight Board Games](https://npbruce.github.io/valkyrie/)
- [RAG&#39;s Collapse: Uncovering Deep Flaws in LLM External Knowledge Retrieval](https://www.youtube.com/watch?v=cs1TDTOby58): The retrieval functionality of the new Assistants ...


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (2 messages): 
        
- **Deita v1.0 Mistral 7B Benchmark Logs**: User `@teknium` shared a [GitHub link](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/deita-v1.0-Mistral-7B.md) to **benchmark logs for different LLMs** including **Deita v1.0** with **Mistral 7B**.
- **Model Training Methods**: User `@teknium` mentioned **Deita SFT+DPO** without further elaboration, possibly referring to a specific **Large Language Model (LLM) training method**.

**Links mentioned**:

[LLM-Benchmark-Logs/benchmark-logs/deita-v1.0-Mistral-7B.md at main · teknium1/LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/deita-v1.0-Mistral-7B.md): Just a bunch of benchmark logs for different LLMs....


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (39 messages🔥): 
        
- **DRUGS Project**: `@gabriel_syme` shared an exciting project called [DRUGS](https://github.com/EGjoni/DRUGS) which aids in handling finicky sampling parameters.  
- **MathPile Corpus**: `@giftedgummybee` linked to a math-centric corpus called [MathPile](https://arxiv.org/abs/2312.17120), emphasizing the data quality over quantity, even in the pre-training phase.
- **Deita Project Discussion**: `@.beowulfbr` shared [Deita](https://github.com/hkust-nlp/deita), a Data-Efficient Instruction Tuning for Alignment. However, `@teknium` revealed that the benchmarks degraded every benchmark as compared to the base model except mt bench. `@ldj` compared it to Capybara and mentioned that it seemed to be less cleaned and smaller.
- **Continual Learning of Foundation Models**: `@giftedgummybee` shared an update on [CL-FoMo](https://sites.google.com/view/irinalab/blog/continued-pretraining-blog), a suite of open-source LLMs comprising four 410M and four 9.6B models. They were trained on Pile, SlimPajama (SP), Mix Pile+SP, and Continual (Pile, SP).
- **SplaTAM**: `@spirobel` introduced [SplaTAM](https://spla-tam.github.io/), a tool for precise camera tracking and high-fidelity reconstruction in challenging real-world scenarios, and pointed out that a more user-friendly version is under development.

**Links mentioned**:

- [LLM-Benchmark-Logs/benchmark-logs/deita-v1.0-Mistral-7B.md at main · teknium1/LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/deita-v1.0-Mistral-7B.md): Just a bunch of benchmark logs for different LLMs....
- [GitHub - hkust-nlp/deita: Deita: Data-Efficient Instruction Tuning for Alignment](https://github.com/hkust-nlp/deita): Deita: Data-Efficient Instruction Tuning for Align...
- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://spla-tam.github.io/)
- [Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math](https://arxiv.org/abs/2312.17120): High-quality, large-scale corpora are the cornerst...
- [CERC-AAI Lab - Continued Pretraining Blog](https://sites.google.com/view/irinalab/blog/continued-pretraining-blog): Continual Learning of Foundation Models:CL-FoMo S...
- [Interviewing Tri Dao and Michael Poli of Together AI on the future of LLM architectures](https://youtu.be/OFFHiJzPpCQ?si=uk2dTVrYmLHBlCyn): The introduction to this post can be found here: h...
- [GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!](https://github.com/EGjoni/DRUGS?tab=readme-ov-file): Stop messing around with finicky sampling paramete...


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (231 messages🔥🔥): 
        
- **Model Merging Discussion**: Users `.beowulfbr`, `ldj`, and `giftedgummybee` had a detailed conversation about merging models with different architectures like **Llama2** and **Mistral**. Discussions touched upon how merging models can yield surprisingly strong results, with `@ldj` sharing a link to a successful merge of numerous models with distinct prompt formats on [HuggingFace](https://huggingface.co/uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b). They also discussed the implications of the merge size, and how some processes tend to create larger models.
- **Potential of Graded Modal Types**: User `.beowulfbr` proposed the idea of using **graded modal types** to track where objects are located on CPU and GPU, theorizing it could potentially improve performance substantially.
- **Discussions on Chatbot Training**: `@gabriel_syme` prompted a discussion about training merged models, with responses indicating this has already been done, but isn't commonly done. `@giftedgummybee` shared their current focus, which involves fine-tuning a Mixtral -> Mistral model with wiki + slimorca.
- **AI Model Suggestions**: User `@dogehus` inquired about strong AI models for function calling. Several users, including `@mihai4256` and `@ldj`, provided suggestions, including **NexusRaven V2** and **Nous-Hermes-2**. 
- **Mixtral Model Metamath Contaminat**: `@nonameusr` pointed out that the **Metamath** dataset used in **Mixtral** is contaminated.

**Links mentioned**:

- [README.md · TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T at main](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/README.md)
- [NobodyExistsOnTheInternet/mergedallmixtralexpert · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/mergedallmixtralexpert)
- [NobodyExistsOnTheInternet/unmixed-mixtral · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/unmixed-mixtral)
- [uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b · Hugging Face](https://huggingface.co/uukuguy/speechless-llama2-hermes-orca-platypus-wizardlm-13b)
- [🤗 Transformers](https://huggingface.co/docs/transformers/index)
- [Tweet from Rohan Paul (@rohanpaul_ai)](https://fxtwitter.com/rohanpaul_ai/status/1741044633495326861): Run Mixtral-8x7B models in Free colab or smallish ...
- [NobodyExistsOnTheInternet/wikidedupedfiltered · Datasets at Hugging Face](https://huggingface.co/datasets/NobodyExistsOnTheInternet/wikidedupedfiltered)
- [GitHub - uukuguy/multi_loras: Load multiple LoRA modules simultaneously and automatically switch the appropriate combination of LoRA modules to generate the best answer based on user queries.](https://github.com/uukuguy/multi_loras): Load multiple LoRA modules simultaneously and auto...
- [Tweet from Nexusflow (@NexusflowX)](https://twitter.com/NexusflowX/status/1732041385455624256?t=tqs6W80qRinFlGq7aHk4ug&s=19): 🚀Calling all developers of copilots and AI agents...
- [GitHub - asahi417/lm-question-generation: Multilingual/multidomain question generation datasets, models, and python library for question generation.](https://github.com/asahi417/lm-question-generation): Multilingual/multidomain question generation datas...
- [llama.cpp/examples/finetune/finetune.cpp at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/finetune/finetune.cpp): Port of Facebook&#39;s LLaMA model in C/C++. Contr...
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui/): A Gradio web UI for Large Language Models. Support...
- [Mixtral Experts are initialized from Mistral 7b - Low Rank conversion possible? · Issue #4611 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/4611): We have evidence that Mixtral&#39;s Experts were i...
- [TinyLlama Pretraining Report](https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-04-23-38-15---Vmlldzo1MzA4MzIw?accessToken=5eu2sndit2mo6eqls8h38sklcgfwt660ek1f2czlgtqjv2c6tida47qm1oty8ik9): See  https://whimsical-aphid-86d.notion.site/Relea...


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (22 messages🔥): 
        
- **Discussion on Amazon Titan Text Express and Amazon Titan Text Lite**: User `@spaceman777` sought community insights about Amazon's new large language models (LLM), Amazon Titan Text Express and Amazon Titan Text Lite. Despite its release on [Nov 29, 2023](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/), user finds no publicly available benchmarks, leading to speculation about Amazon's low-key approach to AI releases. 
- **DL Model Training Strategy**: `@max_paperclips` introduced an idea of creating a deliberately bad dataset and finetuning a model on it to subtract the delta from the base model, then applying a well-curated dataset for further finetuning. This concept sparked a discussion with `@teknium` and `@giftedgummybee`, comparing this process to reversing a LoRA model.
- **Seeking Repository of ChatGPT Failures and Bloopers**: User `@max_paperclips` was curious about the existence of any list showcasing typical errors made by ChatGPT. `@giftedgummybee` responded that no such definitive list existed but suggested the possibility of using the LLAMA tool.

**Links mentioned**:

[Amazon Titan Text models—Express and Lite—now generally available in Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/)


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Overfitting in Models Talk**: Various users discussed their concerns about the potential for models to overfit on certain data sets, leading to potential copyright infringement issues. Specifically, `@thejonasbrothers` mentioned that MJ was likely to have trained their model on entire 4k movies.
    - `@astropulse` emphasized the importance of considering the potential for extracting original artist details from the output of a fine-tuned model.
    - `@pseudoterminalx` discussed an approach of limiting the exposure of a dataset to the model to a single epoch to temper the issue of overfitting. 
    - User `@SegmentationFault` added that issues may arise if models reproduce copyrighted text or images nearly verbatim, as discussed in relation to a [New York Times lawsuit vs OpenAI](https://garymarcus.substack.com/p/things-are-about-to-get-a-lot-worse).

- **Model Size and Performance**:
    - `@.undeleted` criticized the trend towards developing inefficient, oversized models that not only create legal trouble but waste resources.
    - `@thejonasbrothers` maintained that smaller models eliminate overfitting and train faster.
    - The users agreed that adding more parameters is not an ideal alternative to longer training.

- **Copyright and Legal Issues**:
    - There was a lengthy discussion on the legal ambiguity surrounding the use of copyrighted materials in AI model training. 
    - `@SegmentationFault` mentioned that infringement is handled case-by-case, based on the degree of similarity between the AI-produced content and the original copyrighted materials.
    - `@clock.work_` added that any form of profiting from proprietary outputs could lead to legal troubles.
    - The application of these legal standards could affect both AI companies such as Midjourney and the development of open-source models.

- **Proprietary vs Open Models**:
    - The users discussed the implications of monetizing outputs from proprietary models and the potential issues confronting open-source AI development. 
    - `@SegmentationFault` stressed a preference for open models as fair use and expressed concerns about the implications of legal actions against proprietary models extending to open models.

- **MJ's Video Model**: `@SegmentationFault` highlighted that Midjourney was training a video model, suggesting that if the model begins producing identical video clips from movies, it could lead to serious copyright infringement issues.

**Links mentioned**:

- [Things are about to get a lot worse for Generative AI](https://garymarcus.substack.com/p/things-are-about-to-get-a-lot-worse): A full of spectrum of infringment
- [TheBloke/dolphin-2_6-phi-2-GGUF · Hugging Face](https://huggingface.co/TheBloke/dolphin-2_6-phi-2-GGUF)
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/18ryu8a/forbes_rob_toews_of_radical_ventures_predicts/)

        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- Concerns and discussions centered around *GPT-4 and ChatGPT's performance, limitations, potential misuse, and response times*, especially within the paid premium version. Issues such as artificial usage limitations, slow response time, the AI behavior, and problems with human verification were cited by various users across the guild. 
- *Technical issues* encountered by users while interfacing with GPT-4 and ChatGPT were prevalent; these included issues with running dolphin mixtral locally, publishing GPT's errors, text file extraction, and continuous human verification.
- Users explored the potential for using *custom-built GPT models* to perform specific tasks such as enhanced creativity or structured thinking, as indicated in the GPT-4-discussions. 
- A series of inquiries about using *langchain's memory* to enhance or tune prompts, and recurse prompts to match a desired output length were prevalent in API-discussions and Prompt-engineering.
- A discussion on potential *changes in consumption models* for unlimited GPT and ChatGPT use and potential effects, such as increased scams due to misuse, was held.
- Several conversations emphasized the *need for responsible use*, compliance with OpenAI's guidelines, and potential consequences, with policies such as OpenAI's usage policies and guild conduct being highlighted.
- The upcoming *launch of the GPT store* in early 2024 was revealed in the GPT-4-discussions. 

**Links mentioned**:
- [Usage policies](https://openai.com/policies/usage-policies)
- [status.openai.com](https://status.openai.com)
- [GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails)

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (14 messages🔥): 
        
- **Running Dolphin-Mixtral Locally**: `@strongestmanintheworld` posed a question about running dolphin mixtral locally, but no responses were provided. 
- **Use of ChatGPT's App**: Discussion about the use of **ChatGPT's app** vs their website, with `@jayswtf` expressing a preference for the website over the app. `@prajwal_345` noticed this as well. 
- **ChatGPT Assistant Response Time**: `@aviassaga` raised issue of ChatGPT's overly long response times, sometimes waiting up to 40 seconds for a response. 
- **Discussion on Bing's Tone**: `@arevaxach` expressed frustration over Bing's sassy and annoying demeanor. `@jaicraft` suggested that things might improve with GPT-4 turbo in Copilot, hinting it might act more like ChatGPT. `@Rock` however, seemed to prefer Bing due to its personality and better coding skills.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (120 messages🔥🔥): 
        
- **ChatGPT and Usage Limitations**: There were discussions about the limitations of ChatGPT, particularly the restriction on number of messages. `@colt.python` brought up the issue of usage cost per hour, and it was clarified by `@smilebeda` that the API has no such restriction, but the online app does. `@infec.` expressed dissatisfaction with the paid premium version still being subjected to these usage limitations.
- **Concerns about Custom ChatGPTs**: `@infec.` talked about using custom ChatGPTs as work aides, expressing disappointment when they encountered usage limits despite paying for the service.
- **Potential for Unlimited Usage**: `@lemtoad` speculated on how much users would be willing to pay for unlimited ChatGPT use. The conversation touched on potential risks, such as increased scams and misuse. `@.cymer` noted that while power users would relish unlimited usage, this could lead to misuse.
- **Potential Issues with ChatGPT Answering Questions**: Users offered their experiences with ChatGPT seemingly avoiding answering direct questions or 'scamming' users out of their daily message allowance. `@.cymer` and `@kaio268` both shared frustrations with this aspect. 
- **Creating Chatbots with GPT-4**: User `@abhijeet0343` asked for advice regarding inconsistencies in the responses from a GPT-4-based chatbot they developed, which used langchain for embeddings and stored them in Azure AI search. Suggestions from `@poofeh_` and `@kaveen` included making system prompts more assertive or giving specific examples, and employing guardrails to handle the issues of LLMs having difficulty counting things.

**Links mentioned**:

[GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails): Adding guardrails to large language models. Contri...


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (76 messages🔥🔥): 
        
- **GPT-4 Performance**: User `@꧁༒☬Jawad☬༒꧂` expressed frustration over the deteriorating performance of *GPT-4*, especially regarding its inability to *browse the web* and fetch the required data. 

- **Human Verification Loop**: Users `@rodesca` and `@realmcmonkey` encountered a repetitive *human verification loop* that prevented them from logging in. User `@dystopia78` suggested contacting support and checking the website status using [status.openai.com](https://status.openai.com).

- **ChatGPT Limitations**: User `@m54321` described the inconvenience caused by limitations placed on the number of messages in a chat, which necessitates starting a new chat and retraining the model. User `@laerun` suggested using a *custom GPT* and creating focused data chapters to improve efficiency.

- **Persistent Verification**: User `@ekot_0420` complained about being constantly verified by *ChatGPT* after asking each question.

- **User Quota Exceeded**:  User `@not_richard_nixon` reported getting a *"User quota exceeded"* error when attempting to upload an image to *GPT-4*'s chat.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (15 messages🔥): 
        
- **Generating Extreme Imagery with GPT-4/DALL-E**: User `@gabigorgithijs` asked for ways to generate more 'extreme' content using DALL-E despite difficulties in generating simple items. User `@satanhashtag` clarified that public personalities cannot be used in such applications.
- **Publishing GPTs Errors**: User `@jranil` reported an issue on the difficulty of publishing their latest GPT models, either experiencing an error message ("Error saving") or no response from the page.
- **Text File Extraction Issue**: `@writingjen` sought help with an issue extracting text files in GPT-4.
- **Exploring Capabilities of Custom GPTs**: `@happyg` initiated a discussion on the potential capabilities of custom-built GPT models that perform tasks not usually handled by the default GPT. Examples provided included models designed for structured thinking, brainstorming, or enhanced creativity.
- **GPT Message Limit Concerns**: `@writingjen` expressed frustration over hitting a message limit after creating a few messages, despite abstaining from using advanced features like Dall-e. `@solbus` clarified that it was a rolling cap, fully resetting only after 3 hours of non-use. Further activities within the three-hour window eat into the limit balance.
- **Launch of GPT Store**: In response to `@kd6`'s query on the launch of the GPT store, `@solbus` provided information that the intended launch was set for early 2024.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 messages): 
        
- **Prompt Length Control**: `@barium4104` raised the question whether the only way to control the length of a prompt response is through prompt recursion.
- **Enhancing and Tuning Prompts**: `@prochatkiller` asked about the possibility of enhancing and tuning prompts, and whether the use of langchain memory could assist in this matter.
- **Increasing 'Extreme' Output**: `@gabigorgithijs` inquired for ways to make ChatGPT-4 and DALL-E generate more 'extreme' results, as simple generation was proving to be a challenge.
- **Usage Policies Clarification**: `@eskcanta` responded to `@gabigorgithijs`'s request with a reminder to check OpenAI's [usage policies](https://openai.com/policies/usage-policies), indicating that certain types of extreme content might be disallowed and discussing them could risk account access. They also mentioned a specific OpenAI Discord channel `<#1107255707314704505>` for further discussions, as long as everything was within the rules.

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 messages): 
        
- **Prompt Response Length**: `@barium4104` questioned if the only way to achieve a specific length in a prompt response is through prompt recursion.
- **Enhancing and Tuning Prompts**: `@prochatkiller` asked if there are ways to enhance and tune prompts and if using langchain memory would help with the task.
- **Making GPT-4/DALL-E More Extreme**: `@gabigorgithijs` expressed difficulty in generating even simple things with GPT-4/DALL-E and wanted to know how to make the AI generate more 'extreme' things.
- **GPT-4/DALL-E Usage Policies**: `@eskcanta` responded to `@gabigorgithijs's` query by emphasizing the importance of moral and legal use of OpenAI's models. They pointed to OpenAI's usage policies, warning about consequences for violations, while offering to help achieve goals within the bounds of the rules. They provided a [link to OpenAI's Usage Policies](https://openai.com/policies/usage-policies).

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Active discussion around **Mixture of Experts (MoE) and Feed-Forward Networks (FFNs)**; `@stefangliga` clarified that MoE only replaced some FFNs.
- Ongoing exploration of various models to train using available rigs; `@nruaif` advised using the [Official Capybara dataset](https://huggingface.co/datasets/LDJnr/Capybara) with the YAYI 2 model and `@faldore` confirmed that **Axolotl** works with **TinyLlama-1.1b**.
- Conversations about **Axolotl's compatibility**, sourcing datasets for continued pretraining, and implementing RLHF fine-tuning; suggested use of [LASER](https://github.com/pratyushasharma/laser) for improving Large Language Models and standardizing RAG prompt formatting.
- Training challenges encountered and resolved, including YAYI 2 training issues fixed by downloading the model manually and using Zero2 to save 51GB when training Mixtral.
- Discussion on **temperature setting** during RLHF showcase odd outputs while tweaking the value.
- Updates on community projects like `@faldore` training **Dolphin** on TinyLlama-1.1b dataset.
- Notable community guidance on handling DPO implementation in the main branch of Axolotl and how to fine-tune on preference-rated datasets, along with a call for data filtering due to bad data in certain datasets.
- Shared resources for multi-chat conversations, useful datasets, and logger tools that run on Axolotl, along with repositories and tools for ML preprocessing.
- Hardware-specific conversations on the need for certain rigs like 2 or 4x A100 80gb for running yayi 30b.
- Data-related practices emphasized include the avoidance of GPT4 generated data and the inclusion of non-English datasets; the use of FastText [link](https://fasttext.cc/) recommended for non-English data filtering.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (87 messages🔥🔥): 
        
- **MoE vs FFN discussion**: `@caseus_` had a query about the interchangeability of Mixture of Experts (MoE) and Feed-Forward Network (FFN). `@stefangliga` clarified that only some FFNs were replaced, likely to save parameters.

- **Training Models**: `@le_mess` asked for ideas on what model to train on his available 4x A100's. `@nruaif` suggested using the Official Capybara dataset and the YAYI 2 model, providing the [link](https://huggingface.co/wenge-research/yayi2-30b) to the dataset and specifying that it needed to be reformatted for use with Axolotl. `@le_mess` stated they would train the model if the data was reformatted to a suitable format. 

- **YAYI 2 Training Issues**: While training, `@le_mess` ran into a `AttributeError: 'YayiTokenizer' object has no attribute 'sp_model'`error. Despite attempting to fix it using a PR found on GitHub, the error persisted. Eventually, the model was downloaded and fixed manually, which seemed to work.

- **Microtext Experiment**: `@faldore` noted that he was training Dolphin on TinyLlama-1.1b dataset. `@caseus_` later mentioned plans to train on sheared Mistral in the next week.

- **Training Progress**: `@le_mess` made progress with yayi2 training and shared the [link](https://wandb.ai/mhenrichsen/yayi2?workspace=user-mhenrichsen) to the WandB runs.

**Note:** The conversations are ongoing and the discussion topics could be better summarized with more context from future messages.


**Links mentioned**:

- [wenge-research/yayi2-30b · fix AttributeError: &#39;YayiTokenizer&#39; object has no attribute &#39;sp_model&#39;](https://huggingface.co/wenge-research/yayi2-30b/discussions/5/files)
- [LDJnr/Capybara · Datasets at Hugging Face](https://huggingface.co/datasets/LDJnr/Capybara)
- [wenge-research/yayi2-30b · Hugging Face](https://huggingface.co/wenge-research/yayi2-30b)
- [axolotl/examples/yi-34B-chat at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples/yi-34B-chat): Go ahead and axolotl questions. Contribute to Open...
- [axolotl/examples/yayi2-30b/qlora.yml at yayi2 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/yayi2/examples/yayi2-30b/qlora.yml): Go ahead and axolotl questions. Contribute to Open...
- [mhenrichsen](https://wandb.ai/mhenrichsen/yayi2?workspace=user-mhenrichsen): Weights & Biases, developer tools for machine lear...
- [nRuaif/Kimiko_v3-v0.1 · Datasets at Hugging Face](https://huggingface.co/datasets/nRuaif/Kimiko_v3-v0.1)


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (23 messages🔥): 
        
- **Compatibility of Axolotl With TinyLlama-1.1b**: `@faldore` confirmed that **Axolotl** works with **TinyLlama-1.1b** with no modifications needed.
- **Discussion On Checkpoint Size When Training Mixtral**: `@nruaif` shared that **Zero2** checkpoint will save **51GB** when training **Mixtral**.
- **Share of Research Paper About Language Model Hallucinations**: `@faldore` shared a research paper on how to teach a language model to refuse when it is uncertain of the answer -> [Research Paper](https://arxiv.org/abs/2311.09677).
- **Introduction to LASER**: `@faldore` introduced **LASER** (LAyer-SElective Rank reduction), which is a technique for improving the performance of Large Language Models (LLMs) by removing higher-order components of their weight matrices after training. This method reportedly requires no additional parameters or data and can significantly boost predictive performance -> [Learn More](https://pratyushasharma.github.io/laser/) | [GitHub Repo](https://github.com/pratyushasharma/laser).
- **Training DPO Models in Axolotl**: `@sumo43` guided `@faldore` on how to train DPO models in **Axolotl** by sharing the link to the branch where they trained their models -> [Axolotl Branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/rl-trainer). He also shared an example config -> [Example Config](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B-v2/blob/main/configs/axolotl.yml).
- **Need for Availability of 2 or 4x A100 80gb**: `@le_mess` expressed a need for 2 or 4x **A100 80gb** to run yayi **30b** as fft. He stated that running it on 4x **A100 40gb** with zero3 was not feasible.

**Links mentioned**:

- [configs/axolotl.yml · openaccess-ai-collective/DPOpenHermes-7B-v2 at main](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B-v2/blob/main/configs/axolotl.yml)
- [R-Tuning: Teaching Large Language Models to Refuse Unknown Questions](https://arxiv.org/abs/2311.09677): Large language models (LLMs) have revolutionized n...
- [The Truth Is In There: Improving  Reasoning in Language Models with Layer-Selective Rank Reduction](https://pratyushasharma.github.io/laser/)
- [GitHub - pratyushasharma/laser: The Truth Is In There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://github.com/pratyushasharma/laser): The Truth Is In There: Improving Reasoning in Lang...
- [GitHub - OpenAccess-AI-Collective/axolotl at rl-trainer](https://github.com/OpenAccess-AI-Collective/axolotl/tree/rl-trainer): Go ahead and axolotl questions. Contribute to Open...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (13 messages🔥): 
        
- **Using the RL-trainer Branch in Axolotl**: `@tank02` is trying to figure out how to create prompt formats like chatml for Axolotl to use in a run using the RL-trainer branch. They are not sure about the format `Intel/orca_dpo_pairs` would use within Axolotl and how to ensure that any dataset they use is properly formatted for Axolotl. They shared a prompt format example at [DPOpenHermes-7B Config](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B/raw/main/configs/dpo.yml).

- **Importing Axolotl into Jupyter Notebook**: `@wgpubs` is having trouble importing Axolotl into a Jupyter notebook after pip installing the library. They are seeking a way to generate random examples based on an Axolotl configuration to verify prompts and their tokenized representations.

- **Conversion of DPO Dataset Prompts into ChatML**: `@caseus_` explains that the existing transforms in Axolotl convert the existing prompt from the DPO dataset into a chatml input. They convert the chosen and rejected tokens to only include the eos token, as that's all that needs to be generated by the model.

- **Training of 8-Bit LoRA with Mixtral**: `@caseus_` asked if anyone has been able to train a regular 8-bit LoRA with Mixtral. `@nruaif` confirmed having done so, but mentioned that without deepspeed it runs out of memory at a 16k context, and that the peak VRAM use for a 2k context was around 70gb.

- **Question About Batch Size and Learning Rate**: `@semantic_zone` is curious about the reasons for a smaller batch size with a bigger model and asks if there's a rule of thumb for changing learning rate based on batch size. They wonder if they should adjust their learning rate when they double their `gradient_accumulation_steps`.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (9 messages🔥): 
        
- **Data Quality Concerns**: `@faldore` warned users to filter a certain dataset because **it contains lots of "bad data"**, such as empty questions, empty responses, and refusals.
- **Preferred Dataset**: `@xzuyn` suggested using a dataset from **HuggingFace**, which is binarized using preference ratings and cleaned. This dataset, found at [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), is recommended when fine-tuning on UltraFeedback.
- **Tool Inquiry**: `@noobmaster29` asked if anyone had experience with [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured), an open-source library for building custom preprocessing pipelines for machine learning.

**Links mentioned**:

- [argilla/ultrafeedback-binarized-preferences-cleaned · Datasets at Hugging Face](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
- [GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.](https://github.com/Unstructured-IO/unstructured): Open source libraries and APIs to build custom pre...


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (24 messages🔥): 
        
- **RAG Fine-tuning Data Discussion**: `@_jp1_` expressed dissatisfaction with a dataset used in a paper. They mentioned his team's work on generating fine-tuning data for RAG (in different formats). They suggested an open-source release of the dataset if there is general interest and called for standardization of rag/agent call prompt formatting among open llms.
- **Multi-Chat Convo Resources**: `@faldore` provided various resources in response to requests for multi-chat conversation data. He shared a link to the Samantha dataset on HuggingFace and recommended Jon Durbin's Airoboros framework. He suggested using autogen to generate conversations and provided a link to a logger tool.
- **DPO Implementation**: `@jaredquek` asked about the implementation of DPO in the main branch, and `@caseus_` responded that it will be available soon and provided a link to the relevant pull request. He stated that DPO can be activated by setting `rl: true` in the configuration.
- **Temperature Parameter Setting**: `@dangfutures` shared an experience of tweaking the temperature setting for a model, resulting in odd model outputs.
- `@faldore` also shared multiple model links named "Samantha" on HuggingFace and discussed a bit about AI models believing in their own sentience.

**Links mentioned**:

- [configs/dpo.yml · openaccess-ai-collective/DPOpenHermes-7B at main](https://huggingface.co/openaccess-ai-collective/DPOpenHermes-7B/blob/main/configs/dpo.yml?)
- [src/index.ts · cognitivecomputations/samantha-data at main](https://huggingface.co/datasets/cognitivecomputations/samantha-data/blob/main/src/index.ts)
- [Meet Samantha](https://erichartford.com/meet-samantha): https://huggingface.co/ehartford/Samantha-1.11-70b...
- [[WIP] RL/DPO by winglian · Pull Request #935 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935)
- [autogen/notebook/agentchat_groupchat_RAG.ipynb at f39c3a7355fed3472dce61f30ac49c9375983157 · microsoft/autogen](https://github.com/microsoft/autogen/blob/f39c3a7355fed3472dce61f30ac49c9375983157/notebook/agentchat_groupchat_RAG.ipynb): Enable Next-Gen Large Language Model Applications....
- [oailogger.js](https://gist.github.com/ehartford/ef5d23bf9a43b9a2467f9c1285815f68): GitHub Gist: instantly share code, notes, and snip...


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (28 messages🔥): 
        
- **Continued Pretraining Datasets**: `@caseus_` shared links to datasets such as _SlimPajama-627b_, _OpenWebMath_, _The Stack_, and _peS2o_ and asked for recommendations for more to be used in further pretraining. Links to Hugging Face subsets are shared [here](https://huggingface.co/datasets/cerebras/SlimPajama-627B), [here](https://huggingface.co/datasets/open-web-math/open-web-math), [here](https://huggingface.co/datasets/bigcode/starcoderdata), and [here](https://huggingface.co/datasets/allenai/peS2o).
- **Input on Additional Pretraining Datasets**: In response, `@nruaif` suggested using textbook data and provided links to smaller datasets such as _tiny-textbooks_, _tiny-codes_, and _tiny-orca-textbooks_ located [here](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks), [here](https://huggingface.co/datasets/nampdn-ai/tiny-codes), and [here](https://huggingface.co/datasets/nampdn-ai/tiny-orca-textbooks).
- **Avoiding GPT4 Generated Data**: `@dctanner` and `@caseus_` agreed to avoid using data generated by GPT4 models to prevent impacts from OpenAI terms during the continued pretraining.
- **Mixtral Concerns & Support**: `@nruaif` proposed the idea of embarking on Mixtral, however, `@caseus_` raised that it's necessary to address the existing Mixtral training bugs first before adding more to the mix. They expressed the anticipation of seeing an 8x3B Mixtral.
- **Inclusion of Non-English Datasets**: `@nruaif` and `@xzuyn` proposed using non-English datasets, like _yayi2_pretrain_data_, and _CulturaX_ found [here](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data) and [here](https://huggingface.co/datasets/uonlp/CulturaX), with the suggestion of filtering for the English texts where possible. `@nruaif` suggested using FastText to filter out non-English data. FastText is available [here](https://fasttext.cc/).

**Links mentioned**:

- [allenai/peS2o · Datasets at Hugging Face](https://huggingface.co/datasets/allenai/peS2o)
- [fastText](https://fasttext.cc/): Library for efficient text classification and repr...
- [nampdn-ai/tiny-textbooks · Datasets at Hugging Face](https://huggingface.co/datasets/nampdn-ai/tiny-textbooks)
- [nampdn-ai/tiny-codes · Datasets at Hugging Face](https://huggingface.co/datasets/nampdn-ai/tiny-codes)
- [nampdn-ai/tiny-orca-textbooks · Datasets at Hugging Face](https://huggingface.co/datasets/nampdn-ai/tiny-orca-textbooks)
- [uonlp/CulturaX · Datasets at Hugging Face](https://huggingface.co/datasets/uonlp/CulturaX)
- [wenge-research/yayi2_pretrain_data · Datasets at Hugging Face](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data)
- [cerebras/SlimPajama-627B · Datasets at Hugging Face](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- [open-web-math/open-web-math · Datasets at Hugging Face](https://huggingface.co/datasets/open-web-math/open-web-math)
- [bigcode/starcoderdata · Datasets at Hugging Face](https://huggingface.co/datasets/bigcode/starcoderdata)


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- Warm welcome to new members `@mahimairaja` and `@oganBA` who are keen on contributing to the community.
- In-depth technical discussion about suitable **architectures for a Robotics project** with a focus on Multi-Head Attention (**MHA**) on input vectors and seq2seq **LSTM**s with attention.
- Relevant suggestions and resources provided towards identifying datasets for pretraining Language Models such as The Pile, RedPajamas, (m)C4, S2ORC, and the Stack. 
- Sharing of [Detailed Listings of Language Models](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit) in a comprehensive spreadsheet and a follow-on conversation about creating a **public database** of such models.
- Deep dive into the *Math-Shepherd* research paper and its associated challenges, particularly focusing on reward assignment, model result verification, and concerns about misleading comparisons.
- Various practical elements discussed related to model resilience to **noise injection**, quantization bias and robustness of pretrained models, with a special mention of the concept of **dither**.
- Query about **GPT-NeoX training speed** compared to other repos like x-transformers, with clarification on its superior speed in large multi-node systems.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (35 messages🔥): 
        
- **An Introduction to the Community**: `@mahimairaja` and `@oganBA` introduced themselves to the community, expressing their interests and looking forward to contributing to the domain.
- **Seeking Suggestions on Architectures for Robotics Project**: `@marbleous` asked for suggestions on architectures that allow Multi-Head Attention (**MHA**) on input vectors along with a hidden state to track previous observations for a robotics project. `@catboy_slim_` suggested to look into the work by rwkv and `@thatspysaspy` mentioned seq2seq **LSTM**s with attention.
- **Discussion on Listing Datasets used for Pretraining LLMs**: In response to `@sk5544`'s query about available lists of datasets used for pretraining language models, `@stellaathena` mentioned The Pile, RedPajamas, (m)C4, S2ORC, and the Stack as the major compilation datasets.
- **Detailed Listings of Language Models**: `@stellaathena` shared [a link](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit?usp=sharing) to a detailed spreadsheet listing various language models along with their attributes. A [second sheet](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit) was shared when `@sentialx` asked about models that used GLU activation functions.
- **Discussion on Creating a Public Database of Language Models**: The discussion evolved into exploring ways to create a public database of language models. `@stellaathena` and `@veekaybee` discussed approaches, including creating a markdown file, a small react app, or using a platform like Airtable for the public to update and filter. A key requirement was for the platform to allow for reviewable community contributions.

**Links mentioned**:

- [directory_of_llms.md](https://gist.github.com/veekaybee/f8e589fea42ba7131e4ca0a0f280c0a4): GitHub Gist: instantly share code, notes, and snip...
- [Common LLM Settings](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit): All Settings  Model Name,Dataset,Tokenizer,Trainin...
- [Directory of LLMs](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit?usp=sharing)): Pretrained LLMs  Model,Date,Parameters,Organizaton...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (48 messages🔥): 
        
- **Math-Shepherd Discussion**: `@gabriel_syme` shared a research paper on Math-Shepherd, a process-oriented math reward model that assigns a reward score to each step of math problem solutions. The model showed improved performance, especially for Mistral-7B. However, `@the_sphinx` pointed out that the results might be misleading as they typically sample multiple generations and use a verifier to pick one, thus boosting performance significantly.

- **Necessary Verifier in Practice**: `@gabriel_syme` and `@the_sphinx` agreed on the necessity of a verifier in practical applications. However, the latter suggested a more honest evaluation of the actual gains achieved from the verifier. A potential issue could be self-consistency in theorem-proving settings.

- **Noise Injection and Model Resilience**: `@kharr.xyz` hinted at the need for careful noise injection in both training and inference to avoid the model going off the rails with a bad set of activations. Pretrained models without dropout are less resilient to noise. The range of noise resilience can be determined by observing the performance of quantized model versions. 

- **Misleading Comparisons**: There was a general agreement (mainly from `@the_alt_man`) about the potential misleadingness of comparisons in research papers and how they might overshadow genuinely interesting research.

- **Dither Concept**: `_inox` and `@uwu1468548483828484` had a discussion about the concept of dither, which involves adding noise to deal with quantization bias, especially in heavy quantization situations.  


**Links mentioned**:

- [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935): In this paper, we present an innovative process-or...
- [Tweet from Lewis Tunstall (@_lewtun)](https://fxtwitter.com/_lewtun/status/1740722475149975774): Very cool to see scalable oversight working for ma...


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (2 messages): 
        
- **GPT-NeoX Training Speed**: User `@ad8e` asked if **GPT-NeoX** is expected to train faster than miscellaneous repos like x-transformers, assuming equal neural network architecture. `@stellaathena` responded that GPT-NeoX would train faster if on a **large multi-node system**, or would not be slower if the other systems were highly efficient.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- There was a significant **discussion on diffusions**. User `@joelkoch` sparked a conversation about creating new models and whether smaller models can be used for testing. `@sayakpaul` highlighted the need for a model with standard depth for experimentation and described scaling experiments as intricate processes. `@abhishekchoraria` faced challenges with training Mistral 7B on a custom dataset, receiving an error due to the token indices sequence length.
    - Link: [Introducing Würstchen: Fast Diffusion for Image Generation](https://huggingface.co/blog/wuerstchen)
- **End-to-End FP8 Training Implementation** and **Machine Specifications** were primary topics in the *#today-im-learning* channel, with `@neuralink` sharing their implementation progress and their work on a H100 machine.
- In the *#general* channel, the discussion revolved around **ByteLevelBPETokenizer**, **Fine-tuning LLMs**, *resources for beginners*, **DeepSpeed ZeRO3 and LoRA compatibility**, **unsplash Embeddings**, **access issues with the HuggingFace site**, **multi experts LLMs**, the use of **Intel Xeon with AMX**, and **WSL jobs interruption**.
    - Link: [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt)
    - Link: [💥 Fast State-of-the-Art Tokenizers optimized for ...](https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py)
- The *#cool-finds* channel featured **new HuggingFace Spaces**, namely **Text Diffuser 2, DiffMorpher & SDXL Auto FaceSwap**, which have been collectively showcased in a YouTube video. The possibility of using an LLM for shell scripts was also discussed.
    - Link: [Generate AI Images with Text - Text Diffuser 2, DiffMorpher &amp; SDXL Auto FaceSwap!](https://youtu.be/ApcJ1UyLQB8)
- *#i-made-this* saw updates on the **NexusRaven2** function calling model and a call for help to complete code in a shared Colab notebook. `@vashi2396` also shared a demo of the in-progress code.
    - Link: [Nexus🐦‍⬛Raven - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/NexusRaven2)
    - Link: [Google Colaboratory](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)
    - Link: [LinkedIn post - Vashisth Malik](https://www.linkedin.com/posts/vashisth-malik_googleai-gemini-aichatbots-activity-7143976408422187008-huTV?utm_source=share&utm_medium=member_android)
- Finally, the *#reading-group* and *#NLP* channels contained queries about the **Mamba paper** and **multilingual pre-trained models**, while also emphasizing the importance of **avoiding bias in results**.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (46 messages🔥): 
        
- Discussion around **ByteLevelBPETokenizer** and loading it from a `.json` file: `@exponentialxp` asked how to load their saved tokenizer configuration. `@vipitis` and `@hynek.kydlicek` provided multiple suggestions with `@hynek.kydlicek`'s solution of using the `Tokenizer.from_file` method reportedly working.
- **Fine-tuning LLMs**: `@skyward2989` asked about making their fine-tuned language model stop generating tokens. `@vipitis` answered by suggesting defining stop tokens or use a different stopping criteria.
- Asking for **LLMs beginner resources**: `@maestro5786` asked for resources on how to train an open source language model, `@skyward2989` recommended HuggingFace's transformers documentation and course.
- **DeepSpeed ZeRO3 and LoRA compatibility**: `@galcoh.` asked whether there's a way to enable DeepSpeed ZeRO3 with LoRA (PEFT), asking about a presumed issue of having no tensors in the model and the optimizer using all model size.
- Query on **unsplash Embeddings**: `@nagaraj4896` asked about the embeddings of Unsplash-25k-photos.
- Issue with **HuggingFace site access**: `@weyaxi` reported an issue with accessing the HuggingFace site and `@xbafs` suggested disabling VPN if any is being used. Similarly, `@SilentWraith` reported an issue of the site not redirecting properly.
- `@typoilu` asked for explanations or documentation about **multi experts LLMs**.
- `@vipitis` expressed an interest in **Intel Xeon with AMX** and `@zorian_93363` appreciated the choice between AMD and Intel for years.
- Concern about **WSL jobs interruption**: `@__nord` questioned the disturbance in running training jobs in Windows Subsystem for Linux (WSL) that get interrupted after the PC is idle for a while, even with sleep mode disabled.

**Links mentioned**:

- [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt)
- [tokenizers/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py at main · huggingface/tokenizers](https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py): 💥 Fast State-of-the-Art Tokenizers optimized for ...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (10 messages🔥): 
        
- **End-to-End FP8 Training Implementation**: `@neuralink` shared that they have implemented 17% of **end-to-end FP8 training in 3D parallelism** (excluding FP8 kernels) and 24% of **DoReMi** over the past three days.
- **Machine Specifications**: `@neuralink` disclosed that they have been working on a **H100 machine**. `@lawls.net` expressed their desire to contribute their resources (Apple M3 Max 48Gb) to the open source community.
- **Implementation from Scratch**: In response to `@gag123`'s question, `@neuralink` confirmed that they implemented all components **from scratch**, apart from CUDA kernels.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (6 messages): 
        
- `@cognitivetech` shared an interest in shell scriptable Language Learning Model (LLM) and also mentioned a [link](https://justine.lol/oneliners/) claiming: *The Mistral 7b instruct llamafiles are good for summarizing HTML URLs if you pipe the output of the links command, which is a command-line web browser*.
- `@devspot` introduced few **new HuggingFace Spaces** this week featuring **Text Diffuser 2, DiffMorpher & SDXL Auto FaceSwap** in a [YouTube video](https://youtu.be/ApcJ1UyLQB8). The video details the functionality of each space.
  - **Text Diffuser 2**: A new model that integrates words into generated images.
  - **Inpainting Version**: An enhancement of the Text Diffuser 2, this allows users to integrate text into certain areas of an existing image.
  - **DiffMorpher**: This feature allows for the smooth transformation of one image into another.
  - **SDXL Auto FaceSwap**: This feature generates images by swapping faces. The speaker demonstrates an example with Mona Lisa's face swapped onto a pilot female.

**Links mentioned**:

[Generate AI Images with Text - Text Diffuser 2, DiffMorpher &amp; SDXL Auto FaceSwap!](https://youtu.be/ApcJ1UyLQB8): A brief video about some of the trending huggingfa...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 messages): 
        
- **Sharing of NexusRaven2 on Community Box**: User `@tonic_1` shared **NexusRaven2**- a function calling model, on the community box. They indicated it's for demo purposes and plan to improve it over time. They shared a link to the project ([link](https://huggingface.co/spaces/Tonic1/NexusRaven2)).
  
- **Request For Code Completion**: User `@vashi2396` shared a link to a colab notebook ([link](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)) and requested that any willing volunteer can help in completing the code mentioned in the notebook which is a 'work in progress'.
 
- **Demo of Progressed Code**: `@vashi2396` also shared a demo of the in-progress code through a LinkedIn post ([link](https://www.linkedin.com/posts/vashisth-malik_googleai-gemini-aichatbots-activity-7143976408422187008-huTV?utm_source=share&utm_medium=member_android)).

**Links mentioned**:

- [Nexus🐦‍⬛Raven - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/NexusRaven2)
- [Google Colaboratory](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 messages): 
        
- **Understanding Mamba paper**: `@_hazler` brought up a query about a diagram in the **Mamba paper**, expressing confusion about the presence of a Conv (Convolutional Neural Network) layer since Mamba is generally known as a purely recurrent model.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (4 messages): 
        
- **New Models Creation**: User `@joelkoch` inquired about the practical approach to creating new models, highlighting that the diffusion model **Würstchen** featured in a Hugging Face [blog post](https://huggingface.co/blog/wuerstchen) harnesses a unique architecture. He further queried about the potential use of smaller models for quick iteration and validation of the approach.
- **Training Models on Custom Datasets**: `@abhishekchoraria` experienced an issue in training **Mistral 7B** on a custom dataset using autotrain, reporting an error stating "token indices sequence length is greater than the maximum sequence length". He sought guidance on changing the sequence length in auto-train.
- `@sayakpaul` responded to `@joelkoch`, opining that small models might not yield useful findings. He emphasized the necessity for a model with standard depth for experimentation, describing scaling experiments as highly intricate.


**Links mentioned**:

[Introducing Würstchen: Fast Diffusion for Image Generation](https://huggingface.co/blog/wuerstchen)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **Avoiding Bias in Results**: User `@vipitis` highlighted the importance of holding out a section of data for testing to avoid overfitting. They also suggested using **k-fold cross-validation** as another method to circumvent bias in results.
- **Bilingual or Multilingual Pre-Trained Models**: User `@horosin` asked for research or guidance on the topic of bilingual or multilingual pre-trained models. `@vipitis` mentioned that most work in this field is being done on **English-Chinese** models.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (4 messages): 
        
- **New Models Creation and Experimentation**: `@joelkoch` asked about the iteration process in developing new models like Würstchen and whether smaller models can be used for quicker testing. `@sayakpaul` responded that historically, super small models don't provide clear insights, hence the need for standard depth models, leading to scaling experiments becoming complex activities. [Würstchen blog](https://huggingface.co/blog/wuerstchen)
- **Issue with Training Mistral 7B**: `@abhishekchoraria` is encountering an error while using autotrain to train a custom dataset on Mistral 7B. The error is related to tokens indices exceeding the maximum sequence length and they're seeking help to change the sequence length in autotrain.

**Links mentioned**:

[Introducing Würstchen: Fast Diffusion for Image Generation](https://huggingface.co/blog/wuerstchen)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Extensive discussions emerged around the potential and capabilities of **small models like Mistral-Tiny**; `.tanuj.` defended the feasibility to perform complex tasks offline on local machines, making it cost-effective and versatile. Models' capabilities for stringing together tasks like **GoogleSearch, EvaluateSources, CreateEssay, DeployWebsite** were hypothesized, heralding new potential for abstract reasoning.
- The **JSON output and Tokenization of Chinese characters** were topics of conversation; `@sublimatorniq` suggested asking the model to output JSON in TypeScript interface format, `@poltronsuperstar` noted that Chinese characters are often 2 tokens due to Unicode, and `.tanuj.` offered assistance in understanding Mistral's tokenization of Chinese characters.
- The deployment channel focused on machine performance for **running tasks on CPU, comparison of LPDDR5 RAM speed, and achieving similar performance to LLM on Apple Silicon GPU**.
- The showcase channel featured a use case demonstration by `@.tanuj.` of Mistral-Tiny for **mathematical operations and task orchestration** as well as `.gue22` sharing insights on Mistral-8x7B variant with helpful resource links.
- On the la-plateforme channel, testing Mistral with **French text** was inquired about, feedback on **Mistral-Medium tuning** was shared, and confusion about the term **Mixtral/Mistral** was highlighted. A synthetic dataset's planning was also mentioned.

**Selected Quotes and Direct Mentions**

`.tanuj.`: "*If you can get good reasoning from a small model, you can get pretty powerful agents made in real time by a user, and be as powerful as you'd like them to be! It can be a solution like one prompt -> building a full web app and deploying it, no user input needed in between.*" 

`@theledgerluminary`: "*But applying a similar architectural pattern to a large model could achieve better results. Really the only thing I see smaller models being beneficial for are real-time communication. If the overall goal is a large “long-running” task, it seems like a waste of time to only use a small model.*"

`@poltronsuperstar` on potential question posed to AGI: "What's your first question to an AGI?"

**Links**

[Google Colaboratory](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb)

[How to fine tune Mixtral 8x7B Mistral Ai Mixture of Experts (MoE) AI model](https://www.geeky-gadgets.com/fine-tune-mixtral-8x7b)


**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (35 messages🔥): 
        
- **Prompting Strategies Challenge from `.tanuj.`**: `.tanuj.` proposed a challenge to design a prompt or craft chat history that allows the CalculatorGPT to accurately solve various mathematical expressions using arithmetic operators, provided complete steps for reaching the intended answer, using the `mistral-tiny` model with the Mistral API endpoint alongside automated re-prompting. 
- **Debate about Applicability of Small Models**: `.tanuj.` defended the feasibility of using a smaller model like `mistral-tiny` for more complex task solving through intelligent, automated re-prompting and function calling. He suggested the possibility of performing complex tasks on a local machine offline, which can make the approach more cost-effective and versatile. `@theledgerluminary` doubted the capabilities of smaller models compared to larger ones, and suggested the use of fine-tuned models specialized for different tasks, though `.tanuj.` argued for the practicality and simplicity of the "agent" over fine tuning.
- **JSON Output Suggestions**: `@sublimatorniq` suggested asking the model to output JSON in the format of a TypeScript interface at the end of the prompt.
- **Affirmations of Small Model Potentials**: Both `@poltronsuperstar` and `.tanuj.` praised the potential of the Mistral tiny model for task orchestration.

Relevant quotes include: 

`.tanuj.`: "*If you can get good reasoning from a small model, you can get pretty powerful agents made in real time by a user, and be as powerful as you'd like them to be! It can be a solution like one prompt -> building a full web app and deploying it, no user input needed in between.*" 

`@theledgerluminary`: "*But applying a similar architectural pattern to a large model could achieve better results. Really the only thing I see smaller models being beneficial for are real time communication. If the overall goal is a large “long running” task, it seems like a waste of time to only use a small model.*"

Relevant Links:

- None were discussed.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (3 messages): 
        
- **Running on CPU and iGPU**: `@ethux` suggested that in certain situations, a task will just run on the CPU since VRAM is faster, but there's no reason to run it on the iGPU.
- **RAM Speed Comparison**: `@hharryr` did a comparison of the speed of LPDDR5 RAM for the new ultra CPU, which is close to 78~80GB/s, similar to the bandwidth of RAM for the M1 / pro chip.
- **LLM Performance on Apple Silicon GPU**: `@hharryr` pondered if comparable performance to LLM running well on Apple Silicon GPU could be achieved with a machine using a new ultra processor.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (12 messages🔥): 
        
- **Use Case of Mistral-Tiny for Mathematical Operations**: `@.tanuj.` presented that >Mistral-Tiny< can be used for calculations like "Evaluate (10**2*(7-2*1)+1)" by setting up a chat between the user and the model involving the computation steps. 
- He mentioned, "*there were in-between steps that were automated,*" and "*only appends to the official chat history when it's a valid, 100% callable function*," indicating that models can perform tasks. 
- `@.tanuj.` suggested a future scenario where **Mistral-Tiny could have functions like GoogleSearch, EvaluateSources, CreateEssay, DeployWebsite**, thus showing the model's potential for abstract reasoning.
- `@poltronsuperstar` saw potential in this approach, stating "*Seems important that an agent can chain functions in a step by step way*". Despite this being a toy problem, it was regarded as having possible real-life applications.
- Referring to the **Mistral-8x7B** variant, `.gue22` shared that it runs on *ancient, free Colab Nvidia T4 w/ 16GB of VRAM* or any *local Nvidia 16GB GPU + 11GB RAM*. He shared links to a [Google Colab notebook](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb), an associated [YouTube video](https://www.youtube.com/watch?v=ZyFlySElG1U), and the related paper [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/pdf/2312.17238.pdf) for more details.

**Links mentioned**:

[Google Colaboratory](https://colab.research.google.com/github/dvmazur/mixtral-offloading/blob/master/notebooks/demo.ipynb)


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (4 messages): 
        
- **Tokenization of Chinese Characters**: `@poltronsuperstar` noted that "**Chinese chars are often 2 tokens** because unicode". 
- **MistralAI Library for Understanding Token Usage**: `@.tanuj.` suggested that using the **MistralAI library in Python** could help in understanding token usage, as the response object includes details about tokens used in the prompt and completion, and the total for the API call. 
- **Tokenizing Chinese Characters in Mistral**: `@.tanuj.` also offered to help anyone interested in understanding how Mistral tokenizes Chinese characters, as he was curious about the process himself. They would just need to DM him. 
- **First Question to an AGI**: `@poltronsuperstar` asked the chat for ideas on what their first question to an **Artificial General Intelligence (AGI)** would be.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (8 messages🔥): 
        
- **Testing Mistral with French Text**: User `@simply34` raised a question about whether anyone has tested the **Mistral embeddings model with French text**, and how it performs when compared to open source multilingual models like multilingual-e5-large. No responses provided as of now.

- **Discussion on Mixtral/Mistral Confusion**: `@everymans.ai` brought up a confusion about whether it's **Mixtral or Mistral** and the functioning of the Mixture of Experts (MoE) AI model, sharing a related [article](https://www.geeky-gadgets.com/fine-tune-mixtral-8x7b). `@dv8s` speculated that "Mix" could be a play on words relating to Mixture of Experts.

- **Feedback on Mistral-Medium Tuning**: `@jaredquek` shared feedback on **Mistral-Medium tuning**, indicating that the model often outputs unnecessary explanations which he believes is a waste of tokens and money. He suggests this is a result of the model not correctly following instructions and could require further tuning.

- **Planning for Synthetic Dataset Generation**: User `@.superintendent` is contemplating when to generate a synthetic dataset, hoping to avoid contributing to high traffic times.
  


**Links mentioned**:

[How to fine tune Mixtral 8x7B Mistral Ai Mixture of Experts (MoE) AI model](https://www.geeky-gadgets.com/fine-tune-mixtral-8x7b): When it comes to enhancing the capabilities of the...


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- A conversation held predominantly by `@philipmay` and `@thewindmom` regarding *German language semantic embedding models* and their different applications, with the [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) touted as the best open-source model for German, and Cohere V3 denoted as the overall best. `@philipmay` also shared his [Colab notebook](https://colab.research.google.com/drive/1J1-6d5AJ1uEbqAi0ViYaGEVHP7Ulv4CS?usp=sharing) for evaluating German semantic embeddings.
- The group addressed the nuances of *Question/Answer (Q/A) retrieval models* versus semantic models and established the lack of a dedicated open-source finder for Q/A retrieval in German. Suggestions included *the Cohere V3 multilingual model*, and *e5 large multilingual by Microsoft*.
- The topic of *Retrieval-Augmented Generation (RAG) on a German Knowledge Corpus* came up, and while not a dedicated model for this, the aforementioned models were suggested due to their semantic capabilities.
- `@philipmay` shared his experiences training the [deutsche-telekom/gbert-large-paraphrase-cosine](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine) and [deutsche-telekom/gbert-large-paraphrase-euclidean](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean) models, stating they are well-suited for training with SetFit.
- `@_jp1_` drew attention to a research paper, [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685), looking into *automatic data selection strategies for alignment with instruction tuning*.
- Discussions around issues concerning the *DPO optimized Mixtral model* were held, with `@philipmay` and `@bjoernp` discussing the problems with router balancing and potential solutions, such as exploring alternatives like [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI#MoE), [stanford-futuredata/megablocks](https://github.com/stanford-futuredata/megablocks), and [laekov/fastmoe](https://github.com/laekov/fastmoe). There were also discussions about the location and absence of actual training code on GitHub.

**DiscoResearch Channel Summaries**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (1 messages): 
        
- **Paper on Alignment and Instruction Tuning**: `@_jp1_` shared a link to a research paper which examines automatic data selection strategies for alignment with instruction tuning. The paper also proposes a novel technique for enhanced data measurement. The work is said to be similar to ongoing endeavors in the discord community. [_Link to the paper_](https://arxiv.org/abs/2312.15685)

**Links mentioned**:

[What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685): Instruction tuning is a standard technique employe...


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (8 messages🔥): 
        
- **DPO Optimized Mixtral Model from Argilla**:
    - `@philipmay` mentioned the DPO optimized **Mixtral model** released by Argilla on huggingface.co with additional **training code** on GitHub. Links provided are: [Notux 8x7B-v1 on Hugging Face](https://huggingface.co/argilla/notux-8x7b-v1), [GitHub - argilla-io/notus](https://github.com/argilla-io/notus).
- **Issues Regarding the Router Balancing**: `@bjoernp` pointed out that the DPO optimized **Mixtral model** is equally affected by the issues regarding the **router balancing** due to its reliance on the transformers mixtral implementation. 
- **Lack of Actual Training Code on GitHub**: `@philipmay` observed that while the model card of **Notux 8x7B-v1** links to a GitHub project, the actual training code seems omitted, with only the older Notus code available.
- **Location of the Training Code**: `@philipmay` discovered the actual training code, which resided in a different GitHub subtree, at [argilla-io/notus](https://github.com/argilla-io/notus/tree/mixtral-fine-tune/vx/fine-tune), but had not yet been merged.
- **Alternative MoE Training Tools**: `@philipmay` proposed considering alternative MoE training tools like [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI#MoE), [stanford-futuredata/megablocks](https://github.com/stanford-futuredata/megablocks), and [laekov/fastmoe](https://github.com/laekov/fastmoe), which could potentially bypass the **router balancing** issues. `@bjoernp` responded that contributions were underway in making the auxiliary-loss implementation in transformers equivalent to that of megablocks and that working directly with megablocks might be a viable but complex option.

**Links mentioned**:

- [notus/vx/fine-tune at mixtral-fine-tune · argilla-io/notus](https://github.com/argilla-io/notus/tree/mixtral-fine-tune/vx/fine-tune): Notus is a collection of fine-tuned LLMs using SFT...
- [argilla/notux-8x7b-v1 · Hugging Face](https://huggingface.co/argilla/notux-8x7b-v1)
- [GitHub - argilla-io/notus: Notus is a collection of fine-tuned LLMs using SFT, DPO, SFT+DPO, and/or any other RLHF techniques, while always keeping a data-first approach](https://github.com/argilla-io/notus): Notus is a collection of fine-tuned LLMs using SFT...
- [GitHub - hpcaitech/ColossalAI: Making large AI models cheaper, faster and more accessible](https://github.com/hpcaitech/ColossalAI#MoE): Making large AI models cheaper, faster and more ac...
- [GitHub - stanford-futuredata/megablocks](https://github.com/stanford-futuredata/megablocks): Contribute to stanford-futuredata/megablocks devel...
- [GitHub - laekov/fastmoe: A fast MoE impl for PyTorch](https://github.com/laekov/fastmoe): A fast MoE impl for PyTorch. Contribute to laekov/...


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (18 messages🔥): 
        
- **Experience with Embedding Models**: In a discussion with `@thewindmom`, `@philipmay` shared his experience with embedding models, especially German ones. He made clear distinctions between *semantic embedding models* and *embedding model for Q/A retrieval*, explaining that questions and potential answers are not necessarily semantically similar.
- **Best Semantic Embedding Models**: `@philipmay` recommended the [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) as the best open-source German semantic embedding model, while the best overall was the *new Cohere V3 embedding model*. He also pointed out that [ADA-2 embedding](https://github.com/UKPLab/sentence-transformers/issues/1897#issuecomment-1693359539) was not well-suited for German text. 
- **Use of German BERT**: He also explained how models he trained, [deutsche-telekom/gbert-large-paraphrase-cosine](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine) and [deutsche-telekom/gbert-large-paraphrase-euclidean](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean), whilst not as efficient in semantic embedding as the paraphrase model he mentioned above, are very well suited as basic models for training with SetFit.
- **RAG on German Knowledge Corpus**: In response to `@thewindmom`'s query about the best model for doing RAG on a German knowledge corpus, `@philipmay` noted the lack of a dedicated open-source Q/A retrieval model for German and recommended *the Cohere V3 multilingual model*. However, `@aiui` suggested *e5 large multilingual by Microsoft* as the best model based on practical experience.
- **Benchmarking and Evaluation**: `@philipmay` shared a link to a [Colab Notebook](https://colab.research.google.com/drive/1J1-6d5AJ1uEbqAi0ViYaGEVHP7Ulv4CS?usp=sharing) that he created for evaluating German semantic embeddings and `@rasdani` described a potential benchmark for context retrieval based on deepset/germanquad.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1J1-6d5AJ1uEbqAi0ViYaGEVHP7Ulv4CS?usp=sharing)
- [sentence-transformers/paraphrase-multilingual-mpnet-base-v2 · Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [deutsche-telekom/gbert-large-paraphrase-cosine · Hugging Face](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine)
- [deutsche-telekom/gbert-large-paraphrase-euclidean · Hugging Face](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean)
- [Question: OpenAI ada-002 embedding · Issue #1897 · UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers/issues/1897#issuecomment-1693359539): Hi @nreimers , your blog about OpenAI embeddings i...


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Langchain's main components**: In the context of Langchain, `@atefyamin` outlines the two main components which are the **chains** and **agents**. The chain is a *"sequence of calls to components like models, document retrievers, or other chains"* while the agent *"is responsible for making decisions and taking actions based on inputs and reasoning"*. 
- **Agents vs Tools**: There was a discussion around the roles and functions of an agent and tools in Langchain with `@shivam51` and `@atefyamin`. Shivam was unsure about when tools are used instead of agents, but Atefyamin clarified that agents use tools to carry out their tasks. The discussion also explored if tools could be passed to chains.
- **Implementing ConversationBufferMemory**: `@atefyamin` asked for help implementing ConversationBufferMemory using an integration, sharing some of their code included firebase but read functionality seemed to lack. 
- **Output Templates in Langchain**: `@repha0709` asked for assistance in creating output templates in Langchain so as to achieve a specific format of responses. `@seththunder` suggested using prompt templates might aid in achieving this, although `@3h0480` cautioned that prompts might not guarantee 100% compliance to the desired template.
- **Langchain Examples on GitHub**: `@rajib2189` shared a [GitHub link](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py) to examples on how to use Langchain.

**Links mentioned**:

[langchain_examples/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py at main · rajib76/langchain_examples](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py): This repo consists of examples to use langchain. C...

        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Invitation to Live Podcast**: User `@teknium` [invited](https://twitter.com/altryne) `@208954685384687617` to join a live Twitter Spaces podcast. However, the invitation was politely declined by the recipient due to their preference for written English over spoken.
- **Inquiry about Podcast**: `@ikaridev` questioned where they could listen to the podcast. `@teknium` provided the [link](https://twitter.com/altryne) to his Twitter for accessing the podcast, which occurs every Thursday at 8AM PST.
- **AI as a Language Translator**: In relation to the declined invitation due to language barriers, `@rusch` shared a [link](https://venturebeat.com/ai/meta-ai-unveils-seamless-translator-for-real-time-communication-across-languages/) to an AI that translates languages in real-time.
- **Evaluation of Open Chat 3.5**: `@axel_ls` shared their experience with training, fine-tuning, and testing Open Chat 3.5. They stated that, although it's not bad, it falls short when compared to GPT 3.5 for coding tasks. Also, they observed that **fine-tuning didn't improve performance much**, but rather led to overfitting.

**Links mentioned**:

[Tweet from undefined](https://twitter.com/altryne)

        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Issues Encountered in Azure Integration with OpenAI:** Users in the #general channel express multiple difficulties including the setup process `@pantsforbirds`, complexities in managing different API limits across regions `@robotums`, managing different models/regions and their respective resource limits `@0xmmo`, and devops and security concerns `@pantsforbirds` once more.
- In the #offtopic channel, user `@joshcho_` expressed interest in *'uploading a VITS model (text-to-speech) and making it available through an API'*, seeking advice on model uploading for API creation.
- Discussion in the #prompting channel revolved around a newly released project, [TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main), by `@ayenem`, which aims to improve prompt alignment with a model's tokenizer. Feedback on this novel project was requested, with `@pantsforbirds` already commending it as *"really cool"*.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (4 messages): 
        
- **Azure Account for OpenAI Setup Challenges**: User `@pantsforbirds` expressed difficulties in setting up an exclusive Azure account for OpenAI, citing the setup process as a deterrence.
- **Region-Specific API Limit Issues**: `@robotums` highlighted the complexities of managing different API limits given by different regions, necessitating the management of multiple OpenAI objects for each model/deployment.
- **Model and Resource Limitations Per Region**: `@0xmmo` mentioned the additional challenge of different models per region each with its own resource limits. Furthermore, they addressed the issue of needing different API keys per resource leading to a massive number of environment variables to manage.
- **Concerns Over Integration and Security Setup**: `@pantsforbirds` also voiced concerns over the heavy devops work required to integrate OpenAI with their existing system and the added complexities of setting up the security.


### ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/) (2 messages): 
        
- **Model Uploading for API Creation**: User `@joshcho_` enquired if anyone has uploaded models to replicate to create APIs, expressing interest in uploading a **VITS model** (text-to-speech) and making it available through an API.


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/) (3 messages): 
        
- **TokenHealer Release**: User `@ayenem` released a new project called [TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main), an implementation that trims and regrows prompts to align more accurately with a model's tokenizer, improving both completion and robustness to trailing white space and punctuation.
- **Feedback on Release**: `@ayenem` has welcomed feedback on the project, stating a lack of experience in releasing projects. User `@pantsforbirds` has commended the project, stating it looks "really cool".

**Links mentioned**:

[GitHub - Ayenem/TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main): Contribute to Ayenem/TokenHealer development by cr...


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- A query about **server configuration** for combining 4x A100 and 4x L40S to run on the same server led to a discussion about the creation of an app to convert enterprise unstructured data into datasets for **fine tuning LLMs** in the `#ai-general-chat` channel. User `@aristokratic.eth` explored the idea and `@fanahova` encouraged him to look into similar existing solutions.
- The `#ai-event-announcements` channel featured an update on the release of a recent podcast episode by `@latentspacepod`, highlighting top startups from NeurIPS 2023, including companies led by `@jefrankle`, `@lqiao`, `@amanrsanger`, `@AravSrinivas`, `@WilliamBryk`, `@jeremyphoward`, Joel Hestness, `@ProfJasonCorso`, Brandon Duderstadt, `@lantiga`, and `@JayAlammar`. Links to the podcast were provided - [Podcast Tweet](https://fxtwitter.com/latentspacepod/status/1741160693582504275?s=46&t=90xQ8sGy63D2OtiaoGJuww) and [Podcast Page](https://www.latent.space/p/neurips-2023-startups).
- Noteworthy AI research papers of 2023 were proposed by `@eugeneyan` in the `#llm-paper-club` channel for the reading group's consideration, mainly with a focus on large language models. A [link](https://open.substack.com/pub/sebastianraschka/p/10-ai-research-papers-2023) to the selection of these papers was provided.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (4 messages): 
        
- **Possibility of Server Configuration**: User `@aristokratic.eth` inquired the feasibility of having **4x A100 and 4x L40S** on the same server.
- **Building an App for Unstructured Data**: `@aristokratic.eth` is considering the development of an application that could convert enterprise unstructured data into datasets for **fine tuning LLMs**. He asked for the community's thoughts on the product-market fit for this idea.
- `@fanahova` suggested `@aristokratic.eth` to research similar applications, indicating that such a solution might already exist in the market. 
- Consequently, `@aristokratic.eth` asked for references to such existing solutions for further examination.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **NeurIPS 2023 Recap — Top Startups**: `@swyxio` announced the release of the latest pod from `@latentspacepod`, which covers NeurIPS 2023's top startups. Notable participants include:
     - `@jefrankle`: Chief Scientist, MosaicML
     - `@lqiao`: CEO, Fireworks AI
     - `@amanrsanger`: CEO, Anysphere (Cursor)
     - `@AravSrinivas`: CEO, Perplexity
     - `@WilliamBryk`: CEO, Metaphor
     - `@jeremyphoward`: CEO, AnswerAI
     - Joel Hestness: Principal Scientist, `@CerebrasSystems`
     - `@ProfJasonCorso`: CEO, Voxel51
     - Brandon Duderstadt: CEO, `@nomic_ai` (GPT4All)
     - `@lantiga`: CTO, Lightning.ai
     - `@JayAlammar`: Engineering Fellow, Cohere
 The podcast can be accessed via the provided links: [Podcast Tweet](https://fxtwitter.com/latentspacepod/status/1741160693582504275?s=46&t=90xQ8sGy63D2OtiaoGJuww) and [Podcast Page](https://www.latent.space/p/neurips-2023-startups).

**Links mentioned**:

[Tweet from Latent Space Podcast (@latentspacepod)](https://fxtwitter.com/latentspacepod/status/1741160693582504275?s=46&t=90xQ8sGy63D2OtiaoGJuww): 🆕 NeurIPS 2023 Recap — Top Startups!  https://www...


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages): 
        
- **AI Research Paper Recommendations**: `@eugeneyan` shared a [link](https://open.substack.com/pub/sebastianraschka/p/10-ai-research-papers-2023) to a list of 10 noteworthy AI research papers from 2023. He suggested these papers for the reading group and pointed out that their focus is mainly on large language models. His selection criteria for these papers were based on his personal enjoyment or their impact in the field.

**Links mentioned**:

[Ten Noteworthy AI Research Papers of 2023](https://open.substack.com/pub/sebastianraschka/p/10-ai-research-papers-2023): This year has felt distinctly different. I&#x27;ve...


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

caviterginsoy: https://arxiv.org/abs/2305.11243
        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **CheXNet Model Deployment Issue**: User `@taher_3` is facing difficulties with deploying a pretrained chexnet model from [CheXNet-Keras](https://github.com/brucechou1983/CheXNet-Keras). They are encountering a problem where every loaded model produces the same prediction as the first image for all subsequent images. They are seeking help from anyone that has faced a similar issue.
        
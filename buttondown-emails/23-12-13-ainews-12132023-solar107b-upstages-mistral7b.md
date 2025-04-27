---
id: 345ace40-a282-40b7-9d43-ac97244d93fd
title: 12/13/2023 SOLAR10.7B upstages Mistral7B?
date: '2023-12-13T23:29:29.946888Z'
type: archival
original_slug: ainews-ai-discords-12132023-6438
description: >-
  **Upstage** released the **SOLAR-10.7B** model featuring the novel Depth
  Up-Scaling technique built on **Llama2** architecture and integrating
  **Mistral-7b** weights. The **Nous community** finds it promising but not
  exceptional. Additionally, weights for the **Phi-2** base model were released,
  trained on 1.4 trillion tokens including synthetic data from **GPT-3** and
  filtered websites by **GPT-4**, using 96 A100 GPUs over 14 days. On
  **OpenAI**'s Discord, users discussed challenges with various **GPT** models
  including incoherent outputs, API usage limits, and issues with **GPT-4 Vision
  API**. Conversations also covered the definitions and expectations of **AGI**
  and **ASI**, concerns about OpenAI's partnership with Axel Springer, and
  comparisons involving **Gemini AI** integrated into Bard, with mentions of
  Microsoft's prompting techniques to match Gemini Ultra performance.
companies:
  - upstage
  - mistral-ai
  - nous-research
  - openai
  - microsoft
models:
  - solar-10.7b
  - llama-2
  - mistral-7b
  - phi-2
  - gpt-3
  - gpt-4
  - gemini-ultra
topics:
  - depth-upscaling
  - model-pretraining
  - synthetic-data
  - gpu-training
  - api-usage
  - gpt-4-vision
  - agi
  - asi
  - model-integration
  - prompting
people: []
---


<!-- buttondown-editor-mode: plaintext -->Upstage's 10.7B [model was released](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0):

> We developed the Depth Up-Scaling technique. Built on the Llama2 architecture, SOLAR-10.7B incorporates the innovative Upstage Depth Up-Scaling. We then integrated Mistral 7B weights into the upscaled layers, and finally, continued pre-training for the entire model.

 ![image.png](https://assets.buttondown.email/images/08b70925-b7c1-4057-92ff-8cb8273bff32.png?w=960&fit=max) 

The Nous community thinks it's good but not great.

In other news, weights for [the Phi-2 base model](https://news.ycombinator.com/item?id=38634490) were released - it's 1.4T tokens of Phi 1.5 + 250B worth of new GPT3-created synthetic texts and GPT4-filtered websites trained over 96 A100s for 14 days.

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Challenges, solutions, and discussions on OpenAI's GPT Models**: Users across the guilds reported difficulties and proposed solutions with the workings of various GPT models, mentioning problems like incoherent output from GPTs, GPT's refusal to check knowledge files before answering, and errors with GPT-4 Vision API. These discussions also covered specific issues like GPT's inability to play Hangman effectively, generate diagrams, and create quality content, where higher performance is seen after creating an outline first and enhancing it later.
- **OpenAI API Usage Concerns and Clarifications**: Dialogue was focused on understanding the limitations of the AI, including its inability to generate visual content, issues related to its usage cap, and problems generating less useful bullet point responses. Comparisons were made between GPT Assistants and Custom GPTs, even as users explored different use case possibilities of OpenAI's APIs, like uploading bulk PDFs and cross-referencing food ingredients for dietary restrictions. Notably, the challenge of integrating the API in playing the Hangman game was highlighted, with some users providing successful examples and the limitations of slower Python access.
- **Account Issues and Functionality Problem Reports**: There were numerous discussions about account-related problems (account downgrades, deletions, login issues), and users were advised to contact official OpenAI support. They deliberated on a variety of issues like loss of chat data, inability to load certain conversations, browser-specific troubles affecting GPT models, and the reality of GPT's message limits as 40 messages per hour.
- **Collaborative Understanding of AGI and ASI**: Users in the guild deepened their understanding of Artificial General Intelligence (AGI) and Artificial Super Intelligence (ASI), discussing rising expectations and their potential implications.
- **Responses to OpenAI Business Strategies**: Users voiced concerns over OpenAI's partnership with Axel Springer due to fears of potential bias and the ethical implications of such a partnership. Changes in pricing and access to GPT Plus also sparked conversations across the guild.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (55 messages🔥🔥): 
        
- **ChatGPT and PDF Reading**: `@vantagesp` recounted an issue where the AI seemed uncertain about its ability to read and summarize PDFs, despite the AI's capable functionality.
- **Discussions about AGI and ASI**: `@【ｐｅｎｕｌｔｉｍａｔｅ】` sparked a conversation about the definitions and expectations of Artificial General Intelligence (AGI) and Artificial Super Intelligence (ASI), asserting that people's expectations of AGI continue to rise.
- **Jukebox by OpenAI**: `@【ｐｅｎｕｌｔｉｍａｔｅ】` shared [a link to OpenAI's project called Jukebox](https://openai.com/research/jukebox), a music generation model, expressing hope that a more updated or enhanced version of such a model would be developed.
- **Issues with GPT-4 and ChatGPT**: Users `@Saitama` and `@kyoei` reported issues with GPT-4, including mediocre response quality and an error in the input stream. User `@slickog` also mentioned a problem with the incoherent output ("word spaghetti") from GPTs.
- **Usage of Gemini AI**: Users `@iron_hope_shop`, `@lugui`, `@solbus`, and `@jason.scott` discussed about Gemini, a chat model integrated into Bard. Microsoft's capability of inducing GPT-4 to perform at the level of Gemini Ultra with proper prompting was also noted by `@【ｐｅｎｕｌｔｉｍａｔｅ】`.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (270 messages🔥🔥): 
        
- **Integration Issues with GPT-4 Vision**: Several users have been discussing difficulties in utilizing the GPT-4 Vision API. For instance, `@moonkingyt` and `@gingerai` both reported issues, with `@gingerai` particularly mentioning that errors often occur when uploading images. `@lugui` and `@rjkmelb` provided some guidance and troubleshooting tips [bug report thread](https://discord.com/channels/974519864045756446/1183909233436131358).

- **Discussions around Limitations of GPT**: User `@mischievouscow` initiated a discussion regarding the limitations of ChatGPT, specifically related to its usage cap and issues with generating less useful bullet point responses. The conversation proceeded with `@dabonemm` comparing GPT's learning process to the "monkey typewriter experiments."

- **ChatGPT Plus Access Issues and Notifications**: Several users reported issues and queries related to accessing and procuring ChatGPT Plus. For instance, `@isrich` and `@themehrankhan` raised queries related to price changes and access to GPT Plus, while `@openheroes` and `@miixms` announced that ChatGPT re-enabled subscriptions.

- **Concerns over OpenAI Partnership with Axel Springer**: Users `@zawango`, `@jacobresch`, `@loschess` and others expressed disappointment and concerns about OpenAI's partnership with Axel Springer, citing potential bias and ethical implications of partnering with a news outlet that has faced controversy.

- **Various Use Cases and Opinions of ChatGPT**: `@textbook1987` shared positive feedback on using ChatGPT for drafting a professional-sounding letter to a doctor. `@loschess` noted limitations in the AI's ability to code complex projects and had to hire a developer to finish a project. `@thepitviper` criticized prompt limits interfering with user experience, which could push users towards alternatives.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (168 messages🔥🔥): 
        
- **Account Issues**: Users including `@vexlocity`, `@Kevlar`, `.draixon`, `norvzi`, `@thedebator`, and `.sweetycuteminty` discussed various account related issues such as login problems, account downgrades and account deletions. They were advised to reach out to the official OpenAI support for help. 
- **Problems with GPT Functionality**: Users like `@woodenrobot`, `@david_36209`, `@vadimp_37142`, `@8u773r`, `@astoundingamelia`, `@bayeslearner`, and `@happydiver_79` reported and discussed issues regarding the functionality of their models. Problems included GPT refusing to check knowledge files before answering, stoppages in the use of external APIs/Actions in custom GPTs, slow performance, models not interacting properly with images, and inability to access voice chat in OpenAI GPT. 
- **Use Case Discussions**: `@samuleshuges` sought advice on a project involving "uploading bulk pdf's and then chatting with them", while `@core4129` asked about the feasibility of using the API for cross referencing food ingredients for diet and allergy restrictions. `@skrrt8227` was interested in using OpenAI to dictate notes into Notion. 
- **Browser-Related Issues**: There were conversations involving `@Mdiana94` and `@jordz5`, regarding browser-specific problems affecting the functioning of GPT models. Clearing cache and changing the browser was suggested as a potential solution. 
- **Data Loss Issues**: Users like `@singularity3100`, `@Mdiana94`, and `@Victorperez4405` raised concerns about losing their chat data or inability to load certain conversations. Besides suggestions to log out and log back in, users were advised to contact official support.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (21 messages🔥): 
        
- **Reaching OpenAI staff for support**: User `@eskcanta` explained that this Discord is mostly community members. They mentioned some OpenAI officials with gold names who could be contacted. There's a link to a post by `@Michael`, an OpenAI member, clarifying the process of reaching to a human through their help portal. *"[It should] lead to a person, says they intend to make it easier and faster to reach where you can message a human."* [Link to Discord post](https://discord.com/channels/974519864045756446/1047565374645870743/1181345605688250469)
- **Modifying conversation titles of GPTs**: `@lduperval` asked if it's possible to affect the title a GPT gives to a conversation. They were interested in ways to identify the GPT that created it.
- **GPT grading essays**: User `@bloodgore` expressed difficulties getting their GPT to scan and respond based on an uploaded document (a rubric). They are trying to use GPT for grading essays but are facing issues with the model hallucinating its own rubrics despite the correct one being in its knowledge base.
- **Uploading files for GPT reference**: `@solbus` and `@mysticmarks1` suggested referencing the specific filename in the user's request to get the GPT to analyze the document, emphasizing that the context limit might not allow for complete document evaluation.
- **Message Limits and Upgrade**: Users `@elpapichulo1308, @bloodgore, @satanhashtag, @solbus` and `@loschess` discussed the message limit rules. The limit seems to be 40 messages per hour, contrary to some users perception that it was per day. `@solbus` pointed out that "days" was a translation error. `@loschess` brought up that upgrading was supposed to result in "No wait times".


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (45 messages🔥): 
        
- **How to Enhance AI Outputs**: User `@cat.hemlock` advised to instruct the AI to create an outline or brainstorm the article first, and then amplify and proofread its generated content to achieve superior results.
- **Issues with AI and Hangman**: `@eskcanta` shared a link to a conversation where the AI failed to correctly play the game of Hangman, even with clear instructions and examples.
- **AI Successfully Runs Python Game for Hangman**: Responding to `@eskcanta`, `@thepitviper` shared a link to a successful instance of AI generating a Python game for Hangman, though it works slowly due to Python access.
- **Seeking Effective Marketing Advice from GPT**: `@dnp_` solicited advice for getting more specific and actionable output from the AI, such as using certain methods or powerwords in marketing campaigns. `@bambooshoots` suggested instructing the AI that it's an expert in a specific field providing responses to a highly knowledgeable audience.
- **Visualizing Elements of Viral Content**: `@dnp_` also expressed interest in creating diagrams to visually represent elements necessary for creating viral content, though struggled with the AI's inability to generate visual content like Venn diagrams. `@exhort_one` confirmed that GPT-3 is text-based and cannot create visuals unless combined with other tools like DALL-E. `@bambooshoots` provided a prompt for generating visualizations using Python, though `@dnp_` noted limitations with break down that involves more than 3 elements.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (45 messages🔥): 
        
- **Creating Quality Content with OpenAI APIs**: `@cat.hemlock` shared an approach to creating quality AI content by first brainstorming an outline and then generating content around it, emphasizing the importance of refining and proofreading after content is generated.
- **Challenges with Hangman Game**: `@eskcanta` pointed out that OpenAI ChatGPT struggles with playing the Hangman game, even with a known word, and known order of letters to pick. However, `@thepitviper` shared a Python game they generated where Hangman worked well, bringing up the point that access to the python tool makes the algorithm slow.
- **OpenAI API and Content Creation**: `@dnp_` asked for advice on using ChatGPT to get specific answers such as methods, frameworks, and power words for a marketing campaign. `@bambooshoots` suggested making the assumption that the AI is an expert in a particular field and tailoring responses to match the expertise level of the audience. `@dnp_` also expressed interest in creating data visualizations with the model, to which `@exhort_one` responded that GPT is text-based and cannot create visual images or diagrams.
- **Comparison of GPT Assistants and Custom GPTs**: `@dnp_` asked about the comparison between GPT Assistants and Custom GPTs, but no clear answer was provided.
- `@bambooshoots` provided an advanced script for creating Python diagrams with further explanation, but `@dnp_` mentioned the limitation of Venn diagrams not being able to support more than 3 circles.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Discussion on **server rack preferences** and potential **performance issues** when running high-load applications, such as running an LLM on GPU VRAM, Python maximising the CPU, and multi-screening YouTube videos; options for **reducing graphics load** and the unique multitasking abilities of Intel's cores were discussed.

- Benchmarked performance comparison between the **SOLAR 10.7B** model and **Mistral 7B** on several tests, showing improvements with SOLAR 10.7B; `@euclaise` suggested benchmarking the **DeciLM-7B** model next and shared a [link to the model](https://huggingface.co/Deci/DeciLM-7B).

- Introduction of and discussion around the **SOLAR-10.7B model**, its features and potential inaccuracies in its performance claims; a debate about the quality and quantity of data used for training AI models; suggestions for using **quality heuristics**, semantic de-duplication and SSL Prototype filtering for mass data transformations.

- Advancements in **RWKV** discussed, with highlight on increased performance and reduced VRAM usage; dialogue around the **long prompt length limitation** in LLM Modelling; continued debate on GPT's AGI capabilities and performance; memory usage concerns with **DeBERTa 1B** model; discussion on open-source communities, potential copyright conflicts and the implications of licensing restrictions.

- Recommendations for models including **Mamba, Mistral0.2, Mixtral, and openHermes** to benchmark performance on a task related to conversation snippets; comparison of running speed between **stable diffusion** and a **3 billion parameter LLM**; multiple users sharing resources and models; inquiry about running **Mistral-7B-v0.1** on a 2080 Ti graphics card with recommendations for quantized versions and model offloading provided; discussion about running small models on **mobile devices**.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (9 messages🔥): 
        
- **Server Rack Preferences**: `@fullstack6209` and `@erichallahan` expressed a preference for a **stealth 4U server rack** and **air-cooling** instead of water-cooling for their tech setup. `@fullstack6209` mentioned readiness to pay a high price for this specific setup.
- **Graphics Performance Concerns**: `@everyoneisgross` asked about expected glitches when running an LLM on GPU VRAM, having Python maxing the CPU, and multi-screening YouTube videos. `@coffeebean6887` predicted potential **lagging, freezing, and out of memory errors** under these conditions.
- **Lowering Graphics Load**: `@coffeebean6887` suggested running the system **headless**, disconnecting extra monitors, and using lower monitor resolution to reduce graphics load and improve system performance. 
- **Multitasking with Intel's Cores**: `@airpods69` mentioned successful multitasking using Intel's **efficient cores** for web-browsing and **performance cores** for model inference, without experiencing lag.
- **API Limitations**: `@fullstack6209` remarked on the absence of **stop or log bias features** in a particular API.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (19 messages🔥): 
        
- **Performance Comparison of SOLAR 10.7B and Mistral 7B**: `@teknium` shared the benchmark results of the **SOLAR 10.7B** model in comparison to **Mistral 7B** on several tests. Key observations included:
- **SOLAR 10.7B** scored around 39% in AGIEval, a significant improvement over Mistral 7B's 30.65%.
- In the GPT4All benchmark, **SOLAR 10.7B** achieved a score of 72%, slightly ahead of Mistral 7B's 71.16%.
- **SOLAR 10.7B** performed only slightly better than Mistral 7B in the TruthfulQA benchmark, scoring 45% as compared to Mistral 7B's 42.5%.
- **BigBench Results**: The BigBench test was also conducted for the **SOLAR 10.7B model**. According to the results shared by `@teknium`, the model had an average score of 38.66%.
- **Perceptions of the SOLAR 10.7B Model**: Some users, like `@artificialguybr` and `@teknium`, mentioned that while **SOLAR 10.7B** showed good performance, it didn't stand out significantly compared to other models of similar complexity.
- **DeciLM-7B**: Following the performance summaries of other models, `@euclaise` suggested benchmarking the **DeciLM-7B** model next, and shared a [link to the model](https://huggingface.co/Deci/DeciLM-7B), indicating its high performance on the Open LLM Leaderboard.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (103 messages🔥🔥): 
        
- **Introduction of SOLAR-10.7B Model**: `Metaldragon01` shared a link to the new 10.7 billion parameter model, **SOLAR-10.7B**, which claimed to be better than **Mixtral** using a new method. `teknium` promised to bench test this model against the claim and share the results. 

- **Discussion on SOLAR-10.7B Model Performance**: The discussion evolved around the performance of the SOLAR-10.7B model. Several members, `n8programs` and `teknium`, expressed skepticism based on initial test results, questioning if the claims made in the model card were accurate. `Carsonpoole` mentioned that SOLAR-10.7B is only a pre-trained model and requires fine-tuning for specific tasks. 

- **Updates on Phi-2 Weights**: `metaldragon01` informed that Phi 2 weight has been made available and `giftedgummybee` hinted that the weights would soon be available on Hugging Face (HF). Discussions on the usability and compatibility of Phi-2 with HF followed. 

- **Debate on Data Quality and Amount for Model Training**: A detailed discussion, led by `georgejrjrjr` and `crainmaker`, revolved around the choice and amount of data used for training AI models. Topics covered include the idea of minimizing data requirements instead of maximizing them, the question of repeated epochs vs. new data, and the concept of synthetic data being used as padding. 

- **Data Generation, Transformation and Filtering**: `atgctg` questioned `georgejrjrjr` about data transformation on a large scale and the latter responded with suggestions on filtering junk text with quality heuristics, semantic de-duplication, and SSL Prototype filtering.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (382 messages🔥🔥): 
        
- **RWKV Discussion and Updates**: Users `@vatsadev`, `@gabriel_syme`, `.benxh` and `@giftedgummybee` discussed the advancements in RWKV, noting the rapid development and improvements in each version. User `@vatsadev` highlighted the increase in performance and reduced VRAM usage in RWKV as compared to transformers. In addition, `@gabriel_syme` shared his experience with training an Architext model on RWKV 2. Relevant link: [Architext Model](https://link.to.examples)
  
- **Long Prompt Length in LLM Modelling**: Users `@ldj`, `@gitfedgummybee`, and `@carsonpoole` discussed the limitation of 6k tokens per second at a batch size of 256 for Mistral 7b on A100, noting this would likely exceed memory capacity if batch size was set to 1. 

- **Debate on GPT's AGI Capabilities**: Users `@fullstack6209`, `@nruaif`, `.benxh` and `@crainmaker` had a discussion around GPT's AGI capabilities, suggesting that GPTs might benefit from more few-shot prompting. `@kenshin9000`'s twitter post suggests that understanding of "propositional logic" and "concept" can significantly enhance GPT4's performance. 

- **DeBERTa 1B Discussion**: `@euclaise` and `@coffeebean6887` discussed the high memory usage of the DeBERTa 1B model, noting that it tends to take more memory than larger models. 

- **Open Source Communities and Model Sharing**: `@yikesawjeez`, `@atgctg`, `@.benxh`, `@tsunemoto`, `@beowulfbr`, `@nruaif` and others discussed the sharing of models on Hugging Face, arguing over licensing agreements, the potential for copyright conflicts, and the pros and cons of model sharing within the community. They also touched upon Microsoft's Phi-2 release and the implications of its licensing restrictions. Relevant link: [Phi-2](https://huggingface.co/microsoft/phi-2)


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (72 messages🔥🔥): 
        
- **Benchmarking Models for Conversation Snippets**: `@brace1` requested recommendations for open-source models to benchmark performance on a task related to extracting and understanding text from conversation snippets. `@night_w0lf` suggested several models, including **Mamba, Mistral0.2, Mixtral, and openHermes**. They also recommended looking at the models’ ELO on the [chatbot arena leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard). `@n8programs` cautioned that these leaderboards are benchmark-based and suggested putting more weight on a model's ELO.

- **Speed Comparison Between Models**: `@n8programs` discussed the disparity in speed between **stable diffusion** and a **3 billion parameter LLM**. Despite being similarly sized, stable diffusion can only complete one iteration per second, while the latter can process 80+ tokens/second on the same hardware.

- **Resource and Model Sharing**: Various users shared resources and models throughout the discussion. For instance, `@tsunemoto` posted a [link](https://huggingface.co/tsunemoto/mlx_mistral_7b) to the **MLX Mistral weights**. `@coffeebean6887` discussed their experience with Apple's model conversion script and shared a [Github link](https://github.com/ml-explore/mlx-examples/tree/main/mixtral) with examples to guide others through the process. `@.benxh` suggested using the open source ChatGPT UI, located in [this Github repo](https://github.com/mckaywrigley/chatbot-ui). 

- **Quantization and Model Running**: `@Fynn` inquired about running Mistral-7B-v0.1 on a 2080 Ti graphics card with 10GB of VRAM through Hugging Face Transformers, asking for advice on using a quantized version or model offloading. `@.beowulfbr` suggested a quantized version of Mistral from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF). They also shared a [Google Colab notebook](https://colab.research.google.com/github/adithya-s-k/CompanionLLM/blob/main/Mistral_7B_qLora_Finetuning.ipynb) on fine-tuning Mistral-7B using Transformers, which contains a section on inference. 

- **LLMs on Mobile Devices**: `@gezegen` and `@bevvy` asked about running small models on iPhone and Android, respectively. `@tsunemoto` suggested using the LLM Farm TestFlight app to run quantized GGUF models on iPhone, while `@night_w0lf` mentioned [mlc-llm](https://llm.mlc.ai/) as a solution for deploying models natively on a range of hardware backends and platforms, including Android and iOS.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- Focus on **OpenAI's new chat version**, trained on the **Prometheus dataset**. Participants wondered whether this had contributed to the reported improved performance, despite the dataset containing no code-related examples. Reference to OpenChat's detailed description on [HuggingFace](https://huggingface.co/openchat/openchat-3.5-1210#%F0%9F%A23%91%E2%9A%96%EF%B8%8F-experimental-evaluator--feedback-capabilities).
- Discussions about **Mixture of Experts** (MoE) indicate that orthogonality between experts might result in better performance. The concept of **lower order rank approximation** of individual experts was also brought up. Other topics included dynamic allocation of experts based on a set minimum threshold and custom MoE routing.
- The **QuIP# method** was discussed. This is a weights-only quantization method that allows a model to achieve near fp16 performance using 2 bits per weight and operate on an 11G GPU. Find more details on [GitHub](https://github.com/Cornell-RelaxML/quip-sharp) and the model in question on [HuggingFace](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit).
- Conversation around **OpenSource German embedding models** with reference to Deutsche Telekom's models and Jina's forthcoming German embedding model. Also mentioned were upcoming models based on **LeoLM Mistral 7B, LeoLM Llama2 70B, and Mixtral 8x7B**.
- Debates on the **effectiveness of EQ Bench**, in light of potential subjectivity in emotional evaluations. Agreement that subjectivity in emotional intelligence measurement presents a challenge.
- Debate about **Mixtral's performance** despite reports that the model seems "smarter" than other 7B models. Several models were tested during the evaluation, and the possibility was proposed that a base 7B model may limit certain types of cognition, which might also affect role-playing and creative writing tasks.


**DiscoResearch Channel Summaries**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (3 messages): 
        
- **OpenChat Version Training on Prometheus Data**: User `_jp1_` mentioned that the new OpenChat version was trained on the Prometheus dataset and asked if this inclusion improved or deteriorated the overall evaluation performance. `_jp1_` provided a link to the OpenChat version's details on the Huggingface website: [OpenChat Details](https://huggingface.co/openchat/openchat-3.5-1210#%F0%9F%A7%91%E2%9A%96%EF%B8%8F-experimental-evaluator--feedback-capabilities).
- **Improved HumanEval Performance**: User `le_mess` responded to `_jp1_`'s query stating that there has been an **insane improvement on humaneval** after the incorporation of the Prometheus dataset.
- **Effect of Prometheus Dataset and C-RLFT Training**: User `_jp1_` expressed curiosity about why the Prometheus dataset, which contains no code-related examples, improved performance. `_jp1_` also specified that the custom C-RLFT training appears to be working effectively, with potential for future enhancement.


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (160 messages🔥🔥): 
        
- **Mixture of Experts and Router Performance**: `@fernando.fernandes` shared a conjecture about mixtures of experts (MoE) performing better due to orthogonality of experts leading to higher rankings and thus more efficient storage and retrieval of information. Mentioned the concept of "lower order rank approximation" of individual experts being challenging due to assumed high matrix ranks [discussion](https://discord.com/channels/702969561295732908/825923693617668156/836645400899592223).

- **Quantized-Based QuIP Method for GPUs**: `@2012456373` shared an AI model from Hugging Face that uses a 2-bit per weight, weights-only quantization method (QuIP#) that can run on a 11GB GPU. This method is said to achieve near fp16 performance [model link](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit) and [source code](https://github.com/Cornell-RelaxML/quip-sharp).

- **Dynamic Allocation of Experts**: `@kalomaze` and `@nagaraj_arvind` discussed the possibility of dynamically allocating experts based on a set minimum threshold for each layer/token. It led to a discussion on modifying and rebuilding `llama.cpp` to cater for such a parameter [discussion](https://discord.com/channels/702969561295732908/825923693617668156/836645400899592223).

- **Custom MoE Routing**: `@kalomaze` developed a custom version of `llama.cpp` that allows the number of routed experts to be specified in the experts.txt file created by the script [source code](https://github.com/kalomaze/koboldcpp/releases/tag/custom-routing-test).

- **Discussion on Experts and MoE Routing System**: Experimenting on the routing mechanism and the number of experts selected for processing in a token was a predominant discussion. Posts from Reddit and a Discord server (TheBloke discord) were referenced [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/18h5p1v/mixtral_still_works_well_when_using_1_expert_per/) and [Discord server invite](https://discord.gg/974amsUB).


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (18 messages🔥): 
        
- **Quantize base QuIP# method**: `@2012456373` shared information on a [model](https://huggingface.co/Minami-su/Yi_34B_Chat_2bit) that uses the **QuIP#** method, a weights-only quantization method that achieves near fp16 performance using only 2 bits per weight. The model can run on an **11G mem GPU** and more details about the QuIP# method can be found on [GitHub](https://github.com/Cornell-RelaxML/quip-sharp).
- **STK Training Code for Mixtral**: `@e.vik` is seeking help with **stk training code for mixtral**. They are close to getting it working but are experiencing memory access errors.
- **German Embedding Models**: `@thewindmom` and `_jp1_` discussed various **open-source German embedding models** they've tested. `_jp1_` also mentioned an upcoming **DiscoLM German v2** model planned to be released during the Christmas period which would use new finetuning datasets. `@rasdani` recommended Deutsche Telekom's `gbert-large-paraphrase-euclidean` and `gbert-large-paraphrase-cosine` models, while `@flozi00` announced Jina's intent to release a German embedding model.
- **New Models under Leolm and Mixtral**: `_jp1_` shared plans for creating models based on **LeoLM Mistral 7B**, **LeoLM Llama2 70B**, and **Mixtral 8x7B**. However, they clarified that all models may not be ready/released immediately.
- **Comparison of Mixtral and Qwen-72b**: `@aslawliet` inquired about the comparison between Mixtral and Qwen-72b, while `@cybertimon` expressed anticipation for the release of the Mixtral 8x7B model.


### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (6 messages): 
        
- **Concerns about Sharing Code on GitHub**: User `@rtyax` expressed hesitation about sharing their code on GitHub due to potential privacy issues. They, however, considered creating a new GitHub account to sidestep this concern.
- **EQ Bench and Subjectivity in Emotion Evaluation**: `@_jp1_` pointed out that the EQ Bench could be influenced by subjectivity in emotions assessment and suggested conducting a comparative study of expert evaluations. `@calytrix` agreed that emotional intelligence measurement's inherent subjectivity is a challenge, but still believes the EQ Bench can effectively discriminate between different responses.
- **Questions about Mixtral's Performance**: `@_jp1_` also questioned why **Mixtral's** performance wasn't as high as expected. `@calytrix` responded that they used the specified tokenisation and tested several models including: DiscoResearch/mixtral-7b-8expert, mattshumer/mistral-8x7b-chat, mistralai/Mixtral-8x7B-Instruct-v0.1, migtissera/Synthia-MoE-v3-Mixtral-8x7B. They hypothesized that having a base 7b model might limit certain types of cognition and expect this might also impact role playing & creative writing tasks.
- **Subjective Perceptions of Model Performance**: `@technotech` remarked that despite the lower performance scores, in their subjective opinion, the **Mixtral model** seems “much smarter” than other 7B models.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Announcement of the **availability of Microsoft Phi-2 model weights** for research purposes [Microsoft's Official Phi-2 Repo](https://huggingface.co/microsoft/phi-2). Comparison of Phi-2 and GPT4 was also discussed.
- Conversation about diverse **instruction datasets**, with topics on how unstructured datasets are structured.
- Mention of an idea to establish an **AI model benchmark**; fun discussion on the concept and a hypothetical dataset for benchmarking purposes.
- Update announcement for **Mixtral (FFR issue fix)**, facilitating FFT operations on specific hardware configurations.
- Discussion of a **credit card issue** and attempts to find a solution.

---

- Interest in enabling evaluation of **finetuned models** against a test dataset for comparison with the base model.
- Exchange about **loss spikes** and methods to stabilize them, such as adjustments in learning rate.
- Conversation about **memory constraints** when training large models and potential solutions like the use of offload.
- Dialogue on an issue with **multi-GPU training**; shared strategies to counteract this problem, including a mentioned Pull Request on Github [Pull Request on Github](https://github.com/huggingface/transformers/pull/27929).
- Shared observations about the performances of **Mixtral model** versus Mistral model, including a useful Pull Request for better DeepSpeed loading [Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/950).

---

- Discussion around an issue with **Flash Attention installation** and transformers' attention implementation, involving changes made by HuggingFace [HuggingFace Changes](https://github.com/huggingface/transformers/pull/26572#discussion_r1400917933).
- Tactical attempts to **fix the training error**, like modifying `sample_packing` to `false` in configuration files.
- Fix deployed to tackle the aforementioned **training error**; verification of its success.
- Exploration of potential reasons behind a sudden **loss spike in pretraining**.

---

- Shared user experience with a **PDF parsing tool** that was not fully satisfactory.
- Shared and discussed a [tweet by Jon Durbin](https://fxtwitter.com/jon_durbin/status/1734714789669056532).
- Willingness expressed to experiment with **different solutions for better PDF parsing**.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (71 messages🔥🔥): 
        
- **Phi-2 Weights Availability**: User `@caseus_` shared that the **Microsoft Phi-2** weights are now available on Hugging Face for research purposes following a discussion about the model's availability on Azure, sparked by `@noobmaster29`. The model is reportedly not as effective as GPT4 but comparable in certain aspects ([Microsoft's Official Phi-2 Repo](https://huggingface.co/microsoft/phi-2)).
- **Instruction Datasets Discussion**: Users `@dreamgen` and `_dampf` discussed various instruction datasets like OpenOrca, Platypus, Airoboros 3.1, and OpenHermes 2.5. Queries were raised about how such datasets are curated and how unstructured datasets are made structured.
- **Potential Model Benchmark**: There was a humorous discussion about creating a benchmark for AI models. `@faldore` suggested naming it Dolphin Benchmark, while `@le_mess` offered the idea to generate a dataset using one of `@faldore`'s models. They joked about training on the test set and comparing results of various prompting strategies.
- **Mixtral FFT Update**: `@caseus_` announced an update for Mixtral that fixes a Fast Fourier Transform (FFT) issue enabling FFT operation on certain hardware configurations.
- **Credit Card Issue**: User `@nruaif` reported a problem with their credit card getting rejected while trying to access some services. They also sent a friend request to another user on the chat for assistance.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (82 messages🔥🔥): 
        
- **Evaluation of Finetuned Models**: `@z7ye` expressed interest in being able to evaluate finetuned models against a test dataset for comparison with the base model. They included an example of potential commands they'd like to run.
- **Loss Spikes**: `@casper_ai` experienced sudden loss spikes when training, but managed to stabilize this by adjusting the learning rate to 0.0005.
- **Memory Constraints**: A discussion arose about memory constraints when training large models. `@casper_ai` mentioned they were unable to fit a model on 8x A100 using FFT due to its memory-consuming nature. `@nruaif` suggested the use of offload, assuming the VM has sufficient RAM.
- **Issue with Model Saving**: Several members, including `@casper_ai` and `@caseus_`, discussed an issue when trying to save a model during multi-GPU training. `@caseus_` shared a link to a [Pull Request on Github](https://github.com/huggingface/transformers/pull/27929) that addresses the problem.
- **Mixtral Model Performance**: When comparing the Mixtral and Mistral models, `@casper_ai` observed that the Mixtral model seemed to have more stable loss during finetuning. `@caseus_` also shared a [Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/950) which aims to improve DeepSpeed loading for the Mixtral model.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (14 messages🔥): 
        
- **Flash Attention Installation and Transformers Attention Implementation**: `@jovial_lynx_74856` had trouble with training due to a change in `transformers` attention implementation. `@caseus_` found that the patch may not be working as intended due to this change and believes the [implementation may be falling back to SDPA](https://github.com/huggingface/transformers/pull/26572#discussion_r1400917933).
- **Changes in Transformers Attention Implementation**: `@caseus_` pointed out the recent [changes made by huggingface in transformers](https://github.com/huggingface/transformers/pull/26572#discussion_r1400917933) attention implementation that could be causing the problem.
- **Attempt to Fix the Training Error**: `@jovial_lynx_74856` managed to run the training without having it packed by modifying `sample_packing` to `false` in `openllama-3b/lora.yml`.
- **Updated Branch for Training Error**: `@caseus_` pushed another update to fix the issue. `@jovial_lynx_74856` confirmed that the training runs without any issue now.
- **Loss Spike in Pretraining**: `@jinwon_k` brought up an unidentified issue about experiencing a loss spike during pretraining and asked for known reasons behind it.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (6 messages): 
        
- **Discussion on PDF Parsing Tool**: User `@visuallyadequate` discussed their experience with a PDF parsing tool. They stated that the tool doesn't work with some of their PDFs and mentioned that it is not open-source. They also reported that the API rejects some files without explanation, the tool works slowly if it does work, and it garbles the formatting. 
- **Tweet Share from Jon Durbin**: `@lightningralf` shared a [tweet from Jon Durbin](https://fxtwitter.com/jon_durbin/status/1734714789669056532) for discussion.
- **Further Discussion on PDF Parsing**: Responding to `@lightningralf`, `@visuallyadequate` expressed a willingness to keep trying different solutions for better PDF parsing.


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Discussion on **API Access Invitations** and **Mistral Model Performance**, with users expressing feedback and inquiries on topics like model evaluation and access procedure.
- Various technical considerations on **Mistral vs Qwen Models**, **privacy concerns**, **data sovereignty**, and **hosting and pricing of Mistral API** were brought up.
    - The pricing can be found on the [pay-as-you-go platform](https://docs.mistral.ai/platform/pricing/).
- Detailed conversations on hyperparameters for **Mistral Models**, **API Output Size**, **Discrepancies Between API and HF Model Outputs**, and various responses were made surrounding these areas.
    - Announcement of a new channel, `<#1184444810279522374>` for API and dashboard-related questions.
- Several deployment related topics were explored, with inquiries about **endpoint open source**, **FastChat Tokenization**, and **recommended settings for Mixtral**.
- In reference to implementation, discussion consisted of rectifying **wrong HF repo for Mistral v0.2** along with sharing the correct one, [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
    - Links provided to updated documentation for [Mistral-7B-v0.2 Instruct files](https://files.mistral-7b-v0-2.mistral.ai/Mistral-7B-v0.2-Instruct.tar) and [Mixtral-8x7B-v0.1-Instruct](https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar).
- Query by user `@casper_ai` on the **finetuning learning rate of Mistral instruct**.
- Showcased work and experiments included the release of a new [fine-tune](https://twitter.com/jon_durbin/status/1734970280391344288) of **Mistral-7b**, termed "Bagel," sharing of GitHub repository for [Chat Completion Streaming Starter](https://github.com/mattlgroff/chat-completion-streaming-starter).
- Random questions and humor in regards to querying **LLMS with Random Questions** and comparing **Mistral to QMoE Mixtral**.
- Within the `la-plateforme` channel, several topics were featured including **API Usage**, **Performance issues**, **Grammar Integration**, **Model Feedback and Playground**, **Rate Limit**, **Billing**, and **Bug Reports**.
    - Creation of a Mistral interfacing playground [Github link](https://github.com/Xmaster6y/mistral-playground) and web link (https://mistral-playground.azurewebsites.net/).

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (74 messages🔥🔥): 
        
- **API Access Invitations**: Users such as `@vortacs` and `@jaredquek` have inquired about the speed of invites for API access. `@aikitoria` signed up a couple of hours after the announcement and got access earlier, indicating a relatively quick process.
- **Mistral Model Performance**: `@someone13574` discussed about the possibility of dynamic expert count routing in **Mistral** and asked for evaluations for different numbers of active experts per token at inference time.
- **Hosting and Pricing of Mistral API**: The hosting and pricing of the **Mistral API** was the main topic of discussion. It was confirmed that the API is hosted in Sweden on Azure. For international customers, the billing is in Euros and will incur some currency conversion fees. For US customers, they will be billed directly in USD in the future. The pricing can be found on their [pay-as-you-go platform](https://docs.mistral.ai/platform/pricing/).
- **Privacy Concerns and Data Sovereignty**: Users expressed concerns regarding the Patriot Act and the hosting of data by US cloud providers. `@lerela` from **Mistral** assured that it's an important topic and they are working on providing better guarantees to their customers. The models can also be deployed on-premises for enterprise customers.
- **Mistral vs Qwen Models**: `@aslawliet` and `@cybertimon` discussed the comparison between **Mistral** and **Qwen** models with the conclusion being **Mistral** being faster, easier to run and being better compared to **Qwen 7b** and **14b**. The comparison with **Qwen 72b** wasn't exact due to lack of in-depth testing, but users lean towards **Mistral** for its stated benefits.


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (21 messages🔥): 
        
- **Hyperparameters for Mistral Models**: User `@tim9422` expressed concern about the lack of information on hyperparameters used for Mistral model training, seeking guidance for full-finetuning.
- **API Output Size**: `@vince_06258` discussed limitations with API outputs, finding that independent of prompt length, the generated responses are consistently brief. `@aikitoria` suggested adding a system message at the start to generate longer responses, but confirmed that there isn't any parameter for "desired length".
- **Discrepancies Between API and HF Model Outputs**: `@thomas_pernet` had issues reconciling API results with outputs from transformer's model from Hugging Face, ending up with significantly different results. `@daain` pointed out that the API, in contrast to base models, uses the instruct model. `@thomas_pernet` managed to reproduce the API results using the correct instruct model.
- **New Channel for API Inquiries**: `@lerela` announced the creation of a new channel, `<#1184444810279522374>` dedicated to API and dashboard-related questions, noting that the API utilizes instruct-only models.
- **Discussion on Mixtral and SOLAR 10.7B Models**: Users `@_dampf`, `@_bluewisp`, and `@vijen.` initiated a conversation about Mixtral and SOLAR 10.7B models. Discussion points included whether Mixtral was trained on new English data, a comparison of Mixtral's performance with the newly released SOLAR 10.7B-Instruct-v1.0 model, and inquiries on the specific use cases for Mixtral.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (6 messages): 
        
- **Endpoint Open Source Inquiry**: `@yvaine_16425` asked if the endpoint is also open-source.
- **FastChat Tokenization Issue**: `@lionelchg` noted a tokenization caveat of FastChat (used in vLLM) and asked if this issue is also present in the TensorRT-LLM deployment or if NVIDIA correctly sends tokens. `@tlacroix_` clarified that tensorrt-llm is just token-in/token-out and suggested following Triton Inference Server tutorials for setting up pipelines for tokenization/detokenization.
- **Recommended Settings for Mixtral**: `@sa_code` asked for recommended settings for top-p and temperature for Mixtral.
- **Queries about RPS Cap on API**: `@kml1087` is considering deploying to production, but first wanted to know if there's a cap on the requests per second (RPS) for the Mistral's API.


### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (8 messages🔥): 
        
- **Wrong HF Repo for Mistral v0.2**: User `@titaux12` pointed out that the [Mistral AI Platform page](https://mistral.ai/news/la-plateforme/) had an incorrect link to the HuggingFace repo for **Mistral v0.2**. The incorrect link pointed to [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), but it should point to [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). The error was in the section `Generative endpoints > Mistral-tiny`.
- **Error Correction**: User `@lerela` confirmed the mistake and thanked `@titaux12` for pointing it out. The issue was subsequently fixed.
- **Mistral-7B-v0.2-Instruct Docs Update**: `@tlacroix_` shared a [link to the Mistral-7B-v0.2 Instruct files](https://files.mistral-7b-v0-2.mistral.ai/Mistral-7B-v0.2-Instruct.tar) indicating that the link would be added to the documentation. `@vgoklani` appreciated the update.
- **Request for Mixtral-8x7B-Instruct-v0.1**: `@vgoklani` requested for a non-HuggingFace version of `Mixtral-8x7B-Instruct-v0.1`. `@tlacroix_` responded jovially and promised to work on it.
- **Reference Implementation Preference**: `@vgoklani` expressed a preference for the reference implementation, citing cleaner code and efficient performance with FA2 (Fast Attention Assemble), particularly when fused with implementations from Tri Dao for both the RMS_NORM and Rotary Embeddings. `@vgoklani` also mentioned working on flash-decoding and a custom AWQ implementation for the reference model.
- **Link to Mixtral-8x7B-v0.1-Instruct**: `@tlacroix_` provided the [requested link for Mixtral-8x7B-v0.1-Instruct](https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar) and confirmed this would be added to the documentation.


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
- **Finetuning of Mistral Instruct**: User `@casper_ai` raised a question asking for information about the **learning rate** used to finetune the **Mistral instruct**. Further discussion or response to the question wasn't capture in the provided message history.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (8 messages🔥): 
        
- **Running Mistral on Macbook**: `@sb_eastwind` mentioned that he was running a 3bit version of **Mistral** on his 32gb Macbook with llamacpp and LM Studio, sharing a [Twitter post](https://twitter.com/sbeastwindy/status/1734635206127063343?t=SilgkZDR885cHc4Cvu0A2Q&s=19) that illustrates the setup.
- **Increasing Memory Allocation**: In response to `@sb_eastwind`, `@daain` suggested that more memory could be allocated to llamacpp on MacOS to run the 4bit version of Mistral, pointing to a [Discord message](https://discord.com/channels/1144547040454508606/1144547040928481394/1184227995465027614) for details.
- **Difference between Q3 and Q4**: `@sb_eastwind` asked about the difference between `q3_k_m` and `Q4`, to which `@daain` stated they don't know as they haven't tried `Q3`.
- **Chat Completion Streaming Starter**: User `@flyinparkinglot` shared a GitHub repository for [Chat Completion Streaming Starter](https://github.com/mattlgroff/chat-completion-streaming-starter), which allows users to toggle between **OpenAI GPT-4** and **Mistral Medium** in a React app.
- **Release of Bagel, Mistral-7b Fine-Tune**: `@jondurbin` announced the release of a new [fine-tune](https://twitter.com/jon_durbin/status/1734970280391344288) of **Mistral-7b**, termed "Bagel", which comes with features such as dataset merging, benchmark decontamination, multiple prompt formats, NEFTune, and DPO.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (3 messages): 
        
- In **Querying LLMS with Random Questions**, `@basedblue` suggested compiling a list of random questions for the Language Model Learning System (LLMS).
  
- **Querying Mistral Equivalent to Querying QMoE Mixtral**: The user `@sb_eastwind` humorously questioned whether querying the Mistral AI is similar to querying the QMoE Mixtral.
  
- **Can Generated Text from Mistral AI be Used for Commercial Datasets**: `@nikeox` proposed a question for intellectual property lawyers, AI ethics experts and data privacy specialists, regarding the commercial use of generated text from Mistral AI. The user broke down their enquiry into a 2-part expert plan and requested confirmation from others in the channel.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (52 messages🔥): 
        
- **API Usage and Performance issues**: user `@_jp1_` inquired about tracking usage, and `@qwertio` identified a typo in the [documentaion](https://docs.mistral.ai/api#operation/createChatCompletion). User `@flyinparkinglot` expressed interest in Function Calling feature while `@sa_code` and `@coco.py` agreed. `@svilup` experienced API access issues which turned out to be a URL rendering error.
  
- **Grammar Integration and Model Usage**: `@nikeox` requested information on the possibility of integrating grammars such as [Llama.CPP](https://github.com/ggerganov/llama.cpp/tree/master/grammars). `@tlacroix_` confirmed that function calling and grammars are on the Mistral roadmap, and further queried `@nikeox` on the specifics of their grammar usage. Conversation about using a system role in the chat completions API, and how to include it was discussed by `@lionelchg`, `@tlacroix_` and `@nikeox`.

- **Mistral Model Feedback and Playground**: `@delip.rao` shared a positive feedback on `mistral-medium` model for handling complex coding tasks linking to [Twitter Post]https://x.com/deliprao/status/1734997263024329157?s=20). User `@xmaster6y` created a playground to interface with the Mistral API and shared [Github link](https://github.com/Xmaster6y/mistral-playground) and web link (https://mistral-playground.azurewebsites.net/)


- **Rate Limit, Billing, and Model Embeddings**: Questions about rate limits were raised from `@_jp1_` and `@yusufhilmi_`, and `@tlacroix_` explained them as approx 1.5M tokens per minute and 150M per month. `@akshay_1` inquired about the embedding model where `@alexsablay` recommended context upto 512. Additional inquiries on the batch limit for embedding requests were raised by `@lars6581` and `@tlacroix_` expecting answers for precise rate limits for APIs.

- **Bug Reports**:`@nikeox` reported an inconsistency between the real and example API response provided the API documentation. 'oleksandr_now' reported issues about API responding over time and requested a Billing API to monitor their usage.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Several updates and releases were discussed in the realm of AI and machine learning models including **Transformers**, **Mixtral**, **multimodal LLMs**, **Seamless M4T v2 models**, **Phi 1 and 1.5 from Microsoft**, and **AMD GPU support** for Llama-inspired architectures; all part of the open-source updates mentioned by `@osanseviero`. Additionally, the **Spaces Analytics** got a shoutout along with announcements regarding **LoRAs inference in the API** and models tagged as text-to-3D. A new course by Codecademy focused on Hugging Face was also shared ([Intro to Hugging Face Course](https://www.codecademy.com/learn/intro-to-hugging-face) & [Gaussian Splatting](https://twitter.com/dylan_ebert_/status/1732115430737629319)).
- The General channel had various discussions, notably multiple plans and fine-tuned versions for **Deepsex-34b model**, brain simulations aiming for artificial intelligence, potential text classification using HuggingFace models. There was a mention of an Australian supercomputer **DeepSouth** for simulating human brain. LSTM discussions were also held among the members regarding time series prediction.
- The 'Today I'm Learning' channel had discussions mainly around **Mamba** architecture, **RNN**, **LSTM**, and **GRUs**. User `@merve3234` provided guidance to `@nerdimo` in understanding these RNN architectures ([Mamba Paper](https://arxiv.org/abs/2312.00752.pdf)).
- Astounding new language models have caught attention in the 'Cool Finds' channel including the **MOE-8x7B** release by Mistral AI, and **tabular deep learning** for bias. Security issues of ML models were also shared with an interesting read on membership inference attacks. A diffusion model for depth estimation called **Marigold** was revealed and a free AI code-along series from DataCamp mentioning a session on "Building NLP Applications with Hugging Face" ([DataCamp NLP session](https://www.datacamp.com/code-along/building-nlp-applications-hugging-face)). A comprehensive article on precision medicine was also shared.
- The 'I Made This' channel primarily focused on individual achievements. `@rwitz_` published a fine-tuned model **Go Bruins V2**, and `@thestingerx` compiled a project **RVC** which offers audio conversion and TTS capabilities. Bringing in the festive cheer, `@andysingal` used Stable Diffusion to create Christmas vibes. An **online artspace** by `@Metavers Artspace` was introduced along with a quantization blog post by `@merve3234` on LLMs.
- The 'Reading Group' channel is opened with a proposed discussion topic **Distilling the Knowledge in a Neural Network**.
- GPU upgrade possibility was brought up in the 'Diffusion Discussions' channel.
- In the 'Computer Vision' channel, model recommendations are sought for **recognizing 2D construction drawings**.
- The NLP channel saw proposals on **combining graph theory or finite state automata theory with models**, looking for solutions to distribute model checkpoints between GPUs, seeking advice on models for **resume parsing**, and dealing with issues on custom datasets for machine translation.

**HuggingFace Discord Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 messages): 
        
- **Transformers Release**: User `@osanseviero` announced several updates including compatibility of Mixtral with Flash Attention, introduction of multimodal LLMs (Bak)LlaVa, Seamless M4T v2 models that offer multiple translation capabilities, forecasting models like PatchTST and PatchTSMixer, Phi 1 and 1.5 from Microsoft, AMD GPU support and Attention Sinks for Llama-inspired architectures. Further details can be found on the official [GitHub release page](https://github.com/huggingface/transformers/releases/tag/v4.36.0).
- **Open Source**: Some significant updates include Small Distil Whisper which is 10x smaller and 5x faster with similar accuracy as the larger Whisper v2; Llama Guard, a 7B model for content moderation to classify prompts and responses; and Optimum-nvidia, which allows achieving faster latency and throughput with a 1-line code change. Additionally, there are updates on models in Apple’s new MLX framework available on the HuggingFace platform.
- **Product Updates**: Spaces Analytics is now available in settings pages of Spaces. LoRAs inference in the API has vastly improved. Models tagged as text-to-3D can now be easily found on the Hub. You can learn more about LoRAs inference on [Huggingface's blog page](https://huggingface.co/blog/lora-adapters-dynamic-loading?source=twitter).
- **Learning Resources**: Codecademy launched an Intro to Hugging Face course. Here is the [link to the course](https://www.codecademy.com/learn/intro-to-hugging-face). Also mentioned is an intriguing introduction to Gaussian Splatting, available [here](https://twitter.com/dylan_ebert_/status/1732115430737629319).


### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (57 messages🔥🔥): 
        
- **Deepsex-34b Model Collaborations**: User `@databoosee_55130` shared a set of fine-tuned versions of the **Deepsex-34b** model by different contributors including `[TheBloke](https://huggingface.co/TheBloke/deepsex-34b-GGUF)`, `[waldie](https://huggingface.co/waldie/deepsex-34b-4bpw-h6-exl2)`, and others from the HuggingFace community. He announced plans to create a new model based on the "Seven Deadly Sins" series.

- **Brain Simulation and AI**: A discussion on brain simulations and artificial intelligence was initiated by `@databoosee_55130` stating that simulating the human brain is extremely complex and current artificial neural networks only accomplish a fraction of the human brain's complexity. `@ahmad3794` added that hardware implementations mimicking real neurons could yield more efficient simulations.

- **Supercomputer for Brain Simulations**: `@stroggoz` shared information on the **DeepSouth** supercomputer, capable of simulating human brain synapses, being developed by Australia's International Center for Neuromorphic Systems, scheduled for launch in 2024. 

- **Inquiries on Text Classification and HuggingFace Support**: Users `@jeffry4754` and `@fireche` requested advice on text document classification, specifically using existing models for category prediction. `@cakiki` suggested reaching out to experts using the [HuggingFace support form](https://huggingface.co/support).

- **Debate on LSTM Neural Network**: `@cursed_goose` received feedback from the group on implementing LSTM cells for time series prediction. He shared a basic implementation in Rust and asked for guidance on creating an LSTM layer and making predictions. `@vipitis` suggested doing unsupervised training or clustering, or trying a zero-shot task with large models.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (4 messages): 
        
- Discussion on **Mamba** by `@merve3234`: Introduced **Mamba**, an architecture that surpasses Transformers in both performance and efficiency, sharing a link to the Mamba paper on Arxiv ([Mamba Paper](https://arxiv.org/abs/2312.00752.pdf)).
- Learning about **Recurrent Neural Network (RNN) Architectures** by `@nerdimo`: Starting to learn about **GRUs and LSTMs**, specifically to tackle the vanishing gradient problem faced by standard RNNs.
- `@merve3234`'s Advice on Understanding **RNN, LSTM, and GRUs**: Noted the initial complexity in understanding LSTMs and GRUs compared to standard RNNs.
- `@nerdimo` shared their learning experience, noting the difficulties in understanding the intricate logic gate mechanisms of **LSTMs and GRUs**.


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (7 messages): 
        
- **Mistral AI's Latest Language Model**: `@tokey72420` shared a link about Mistral AI's recent breakthrough in language models with the release of MOE-8x7B, [Check here for more](https://www.marktechpost.com/2023/12/12/mistral-ai-unveils-breakthrough-in-language-models-with-moe-8x7b-release/).
- **Tabular Deep Learning**: `@d3aries` recommended a paper titled "An inductive bias for tabular deep learning", available on [Amazon.science](https://www.amazon.science/publications/an-inductive-bias-for-tabular-deep-learning).
- **Machine Learning Security**: `@shivank2108` shared a link to a research paper on the security of machine learning and membership inference attacks, accessible on [arXiv](https://arxiv.org/abs/2311.15373).
- **New Depth Estimation Model, Marigold**: `@merve3234` mentioned Marigold, a diffusion model for depth estimation, an alternative to dense prediction transformers, available on [huggingface.co](https://huggingface.co/spaces/toshas/marigold).
- **AI Code-Along Series**: `@datarhys` shared about a free 9-part AI code-along series from DataCamp. The series includes a session on "Building NLP Applications with Hugging Face" and is freely accessible at [Datacamp NLP session](https://www.datacamp.com/code-along/building-nlp-applications-hugging-face).
- **Precision Medicine and Its Role in Medical Practices**: `@jeffry4754` shared an article on how precision medicine, driven by technologies like genomics and proteomics, has the potential to change medical practices, accessible on [link.springer.com](https://link.springer.com/article/10.1186/s12859-020-03836-4).


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 messages): 
        
- **Go Bruins V2 - A Fine-tuned Language Model**: `@rwitz_` shared a [link](https://huggingface.co/rwitz/go-bruins-v2) to their fine-tuned language model, Go Bruins V2.
- **RVC Project**: `@thestingerx` discussed their RVC project that includes audio conversion, various TTS capabilities, and more. They mentioned the challenge of running on CPU and shared a [link](https://huggingface.co/spaces/TheStinger/Ilaria_RVC) to the project.
- **Christmas Vibes using Stable Diffusion**: `@andysingal` shared a [YouTube video](https://youtu.be/iqv2Xn0UVnA?si=BRbw85Z5U6PNZl0a) highlighting the application of Stable Diffusion to create Christmas vibes.
- **Metavers Artspace**: `@Metavers Artspace` shared a [link](https://oncyber.io/0xccc) to an online artspace.
- **Quantization of LLMs Blog Post**: `@merve3234` wrote a [blog post](https://huggingface.co/blog/merve/quantization) on the quantization of LLMs, excluding AWQ. They encouraged other users to write blog posts on Hugging Face.
- `@nerdimo` expressed interest in examining `@merve3234`'s blog post on quantization in their free time.


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (5 messages): 
        
- **Channel Introduction**: User `@murtazanazir` indicates interest in discussing and understanding more about a particular topic. `@merve3234` suggests that the discussion be held on the channel to allow others to join.
- **Discussion Topic**: `@murtazanazir` suggests a topic for discussion: **Distilling the Knowledge in a Neural Network**.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **GPU Upgrade Discussion**: `@pseudoterminalx` mentioned that, even though it was possible to upgrade the GPU in early 2023, it used **350w for 2 seconds per iteration**. `@knobels69` responded that it **might be time for them to upgrade their GPU**.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **2D Construction Data Recognition**: User `@fireche` inquired about a model that can recognize **2D construction drawings** based on image classification, which `@merve3234` clarified.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **Merging Graph Theory or Finite State Automata With Models**: User `@stroggoz` proposed the idea of an open-source library that combines graph theory or finite state automata theory with models. The decision of which model to use would be based on the predictions/scores.
- **Parallelism for Distributing Model Checkpoints Between GPUs**: In response to `@acidgrim's` enquiry about splitting a model between two distinct systems, `@merve3234` referred them to the concept of parallelism as explained on the HuggingFace's documentation on [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism) and [Transformers' Documentation on Performance Optimization for Training](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism).
- **Model for Resume Parsing**: `@surya2796` sought information on creating a model for resume parsing. No further conversation or suggestions followed this query.
- **Issues with Custom Dataset for Machine Translation**: `@dpalmz` faced a `ValueError` while trying to train with a custom dataset for machine translation. `@nerdimo` offered to assist, having encountered the same problem in the past.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **GPU Upgrade Discussion**: User `@pseudoterminalx` suggested that it is not practical to use a certain feature that they did not specify due to high power usage, saying "*most likely the simplest answer is No and even if you could (it was briefly possible in early 2023) it used 350w for 2 seconds per iteration*". In response, `@knobels69` showed interest in upgrading their **GPU**.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Azure Search Service with LangChain**: `@hyder7605` is seeking assistance with integrating multi-query retrieval or fusion RAG capabilities with Azure Cognitive Search, utilizing LangChain. He aims to define filters and search params during document retrieval and incorporate advanced features like hybrid and semantic search.
- **ReRanking model with Node.js**: `@daii3696` is looking for recommendations on incorporating the ReRanking model into a Node.js RAG application. Noted that the Cohere Re-Ranking model is currently only accessible for LangChain in Python.
- **GPT-3.5 Language Response**: `@b0otable` is discussing the difficulty in getting gpt-3.5 to respond in a desired language especially with longer and mixed language context. Current prompt works about 85% of the time.
- **LangChain Error**: `@infinityexists.` is experiencing an error while running any model with LangChain: "Not implemented error: text/html; charset =utf -8 output type is not implemented yet".
- **gRPC Integration with LangChain**: `@manel_aloui` is looking for information on how to integrate gRPC with LangChain. Asked the community if anyone has found a solution.
- **Document Discrepancy**: `@vuhoanganh1704` noticed discrepancies between the LangChain JavaScript and Python documentation, causing confusion.
- **LANGSMITH Sudden Non-Response**: `@thejimmycz` brought up an issue where LANGSMITH isn't saving any calls. `@seththunder` verifies this to be a problem as well.
- **Stream Output with Next.js**: `@menny9762` is asking for advice on streaming the output using Next.js and LangChain. `@seththunder` suggested a callback function called StreamCallBack and to set `stream = true` as a parameter in the language model. 
- **Reranking with Open Source, Multilingual Support**: `@legendary_pony_33278` is interested in open-source reranking techniques for RAG applications that support multiple languages, including German. Noted that most tutorials use the Cohere Reranker Model.
- **Fine-tuning Mixtral-8x7B with LangChain**: `@zaesar` is inquiring about how to fine-tune the Mixtral-8x7B model with LangChain.
- **LangChain Materials**: `@sniperwlf.` is searching for learning materials for LangChain, similar to O'Reilly books. `@arborealdaniel_81024` commented that the project is too young and unstable for a book yet.
- **Indexing Solutions with Sources on RAG**: `@hucki_rawen.io` asked about indexing solutions with sources used in RAG. Questioned about the desired outcome from this process.
        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Video presentation, **"Ben Evans's 2023 AI Presentation"**, was shared. [YouTube link by @stealthgnome](https://www.youtube.com/watch?v=xNBiPd2H9J0)
- Introduction of the **AI News service**, a tool that summarizes discussions across AI Discord servers, with an option to sign up for the service launch. [@swyxio's post](https://buttondown.email/ainews/archive/ainews-12122023-towards-langchain-01/)
- Discussions on **Langchain's rearchitecture** prompted by the post on the subject in the AI News service. [Langchain rearchitecture post](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/?utm_source=ainews&utm_medium=e)
- Sharing of the first **Mistral Medium report**, and subsequent **successful experimentation** with Mistral-Medium by @kevmodrome for UI components creation. [Twitter link by @swyxio](https://fxtwitter.com/skirano/status/1734612606055338383?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- A talk on **LLM development** beyond scaling was attended by @swyxio, with a link to the discussion shared. [Twitter link by @swyxio](https://fxtwitter.com/srush_nlp/status/1732931246915719606?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- Commentary on an unspecified subject by **Sama** was shared, albeit described as not being particularly insightful. [Twitter link by @swyxio](https://fxtwitter.com/tsarnick/status/1734849976667443285?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- Active organization of paper reviews and related discussions:
- An invitation to a chatroom discussion about the **Q-Transformer paper** was extended. [Paper link](https://qtransformer.github.io/)
- Encouragement to sign up for weekly reminders for the recurring **LLM Paper Club** events. [Event signup link](https://lu.ma/llm-paper-club)
- In the **LLM Paper Club** channel:
- Announcement of a presentation on the **Q-Transformer paper** by @cakecrusher, with a link to a copilot on the topic. [Copilot link](https://chat.openai.com/g/g-Aquz1gSDY-q-transformer)
- User queries and clarifications on technical issues and topics for upcoming discussions.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (9 messages🔥): 
        
- **Ben Evans's 2023 AI Presentation**: `@stealthgnome` shared a [link to a YouTube video](https://www.youtube.com/watch?v=xNBiPd2H9J0) of Ben Evans's 2023 AI presentation. 
- **AI News Service Announcement**: `@swyxio` introduced the [AI News service](https://buttondown.email/ainews/archive/ainews-12122023-towards-langchain-01/), an MVP service that summarizes discussions across AI Discord servers, and shared a sign-up link for the upcoming service launch.

- **Langchain Rearchitecture**: The AI News service included a link to a [post about the Langchain rearchitecture](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/?utm_source=ainews&utm_medium=e).  

- **First Mistral Medium Report**: `@swyxio` shared the first Mistral Medium report's [Twitter link](https://fxtwitter.com/skirano/status/1734612606055338383?s=46&t=90xQ8sGy63D2OtiaoGJuww). 

- **Experimentation with Mistral-Medium**: `@kevmodrome` mentioned their successful experimentation with Mistral-Medium in generating UI components.
- **Discussion on LLM Development**: `@swyxio` attended a talk about LLM development beyond scaling and shared the [Twitter link](https://fxtwitter.com/srush_nlp/status/1732931246915719606?s=46&t=90xQ8sGy63D2OtiaoGJuww) to the discussion.
- **Sama Commentary**: `@swyxio` shared a [Twitter link](https://fxtwitter.com/tsarnick/status/1734849976667443285?s=46&t=90xQ8sGy63D2OtiaoGJuww) to a commentary by Sama, pointing out it doesn't say much.


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **Q-Transformer Paper Discussion**: `@swyxio` invited the server members to join a chatroom led by `<@383468192535937026>`, discussing the **Q-transformer** paper. The paper, titled "*Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions*", is authored by multiple researchers, including *Yevgen Chebotar, Quan Vuong, Alex Irpan, Karol Hausman,* etc. The paper can be accessed at [https://qtransformer.github.io/](https://qtransformer.github.io/) and is intended for open discussion.
- **LLM Paper Club**: `@swyxio` also encourages signups for *weekly reminders* for the **LLM Paper Club**, a recurring event that conducts weekly paper reviews, breaking down and discussing various *LLM papers*. Interested members can register for this event at [https://lu.ma/llm-paper-club](https://lu.ma/llm-paper-club). Hosted by *Kevin Ball, Eugene Yan & swyx*, it encourages a read-through of the chosen paper prior to the discussion.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (7 messages): 
        
- **Q-Transformer Presentation Plans**: `@cakecrusher` announced that they will be presenting the Q-Transformers paper and shared a link to their *"copilot"* [Q-Transformer on OpenAI Chat](https://chat.openai.com/g/g-Aquz1gSDY-q-transformer).
- **Access Issue**: User `@swyx.io` expressed having trouble with something, although it's unclear what exactly was the problem.
- **Confusion About Paper Topic**: `@__pi_pi__` expressed confusion about the paper being discussed in the next meeting. This confusion was cleared up by `@slono` providing a [direct link to the thread](https://discord.com/channels/822583790773862470/1184332179661127831) discussing the Q-transformers paper.
- **Next Week's Paper**: `@coffeebean6887` inquired about the paper to be discussed in the following week. There was no response to this query in the provided messages.


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- Interest in **Gemini Pro API** usage and latency stats on Google's Gemini expressed by `dongdong0755` and `rabiat`, respectively, indicating a need for more information on the topic.
- Queries regarding GPT-4 and GPT Tasks evidenced by `.psychickoala`'s request for updates or examples and `@dhruv1`'s query about GPT tasks in the *#resources* channel.
- Conversations around fine-tuning AI, including `@robertchung` seeking to understand the term "nested replies" and `@robhaisfield` discussing using GPT-4 via TypeChat for a specific Email interface.
- Sharing of valuable resources, including `@robotums` sharing a blog post titled ["Phi-2: The Surprising Power of Small Language Models"](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) and `@nosa_.` providing [a helpful link](https://fxtwitter.com/tomas_hk/status/1734664304924721245) in the *#resources* channel, potentially relating to GPT.
- Discussions surrounding evaluation metrics for Language Learning Model (LLM) apps. `@joschkabraun` shared a [blog post](https://docs.parea.ai/blog/eval-metrics-for-llm-apps-in-prod) on the topic and underlined the thoughts of Bryan Bischof, Head of AI at Hex, on using defensive code parsing in **GitHub Copilot**.
- Unfinished inquiry by `jeffreyw128` on uncovering a better understanding of an unspecified subject matter under the *#prompting* channel.

**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 messages): 
        
dongdong0755: Anyone trying out gemini pro api?


### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (1 messages): 
        
.psychickoala: Any update here? Anyone have examples of this?


### ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/) (2 messages): 
        
- **Nested Replies Discussion**: `@robertchung` asked for clarification about the term "nested replies", questioning if it refers to the composition of 1st inbound -> other replies.
- **Fine-tuning AI for Email Interface**: `@robhaisfield` talked about using GPT-4 via TypeChat to produce a specific `Email` interface. He also speculated that a fine-tuned GPT-3.5 or Mistral might be able to do it as effectively, but without fine-tuning, he believed that GPT-4 would be the most viable option.


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 
        
- **Phi-2 Small Language Models**: User `@robotums` shared a blog post from **Microsoft Research** about the potential of small language models. The blog post, titled ["Phi-2: The Surprising Power of Small Language Models"](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/), includes contributions from various researchers such as Marah Abdin, Jyoti Aneja, and Sebastien Bubeck among others.


### ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/) (2 messages): 
        
- **Discussing GPT Tasks**: User `@dhruv1` asked an unspecified user about the kind of task they are running with GPT.
- **Resource Link Shared**: `@nosa_.` shared [a link](https://fxtwitter.com/tomas_hk/status/1734664304924721245) that they found useful, potentially related to the GPT discussion.


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages): 
        
rabiat: Are there any latency stats on googles gemini?


### ▷ #[eval](https://discord.com/channels/1168579740391710851/1168986849784635553/) (1 messages): 
        
- **Evaluation Metrics for LLM Apps**: `@joschkabraun` shared a [blog post](https://docs.parea.ai/blog/eval-metrics-for-llm-apps-in-prod) on evaluation metrics (including code), which don't rely on ground truth data and are suitable for those who want to evaluate live traffic or offline experiments with their LLM app without ground truth data.
- **Quality Control and Evaluation in LLM Apps**: Insights reported from Bryan Bischof (Head of AI at Hex) point towards the usage of defensive code parsing in the **GitHub Copilot**, amounting to thousands of lines to catch undesired model behaviors. Bryan mentioned that this defensive coding can be created with evaluation metrics, emphasizing their importance in quality control and evaluation for building production-grade LLM applications.


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/) (1 messages): 
        
jeffreyw128: Anyone find a good way to understand


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Discussion concerning the **recent version updates** featuring a comparison between the updated **v0.2** and the differences amongst **OpenHermes 2.5 and 2**, as pointed out by `@lightningralf` and `@gabriel_syme`.
- Mention of a potential **new base model** with extensive pretraining, as indicated by `@gabriel_syme`.
- Dialogue on the **Upstage's Depth Upscaling technique** employed on a franken llama-Mistral to attain a size of **10.7B parameters with added pretraining**. The use of this method led to superior results compared to Mixtral, as claimed by `@entropi`. The model, named [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0), fine-tuned for single-turn conversation, was shared on Hugging Face.
- Engaging conversation around **phi 2** about any exciting applications, instigated by `@joshxt`.
- Sharing of the **Phi-2 model summary** by `@entropi`, with a [link provided](https://huggingface.co/microsoft/phi-2) to the Hugging Face model card of Phi-2, a transformer model. The same sources used for training [Phi-1.5](https://huggingface.co/microsoft/phi-1.5) were used for Phi-2, along with some added NLP synthetic texts and filtered websites.
- Mentioned recent availability of **Phi-2 weights** as noted by `@entropi`.

**Alignment Lab AI Channel Summaries**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (4 messages): 
        
- **Version Discussion**: `@lightningralf` and `@gabriel_syme` discussed about the latest version, with `@lightningralf` noting it's **v0.2** which is newer. `@gabriel_syme` compared this with the difference between **openhermes 2.5** and **2**.
- **Model Pretraining**: `@gabriel_syme` mentioned the expectation of a **new base model** with more pretraining.


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 messages): 
        
- **Upstage Depth Upscaling**: User `@entropi` shared about **Upstage's Depth Upscaling** on a franken llama-mistral to achieve a size of 10.7B params with continued pretraining. They claimed to beat Mixtral with this approach and shared the link to their model named [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) on Hugging Face. They pointed out it is a fine-tuned version for single-turn conversation.


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (3 messages): 
        
- **Discussion on Phi-2**: `@joshxt` asked if anyone has done anything interesting with **phi 2**.
- **Phi-2 Model Summary**: `@entropi` shared a [link](https://huggingface.co/microsoft/phi-2) to the **Hugging Face model card of Phi-2**, a transformer model with 2.7 billion parameters. The model was trained using the same data sources as [Phi-1.5](https://huggingface.co/microsoft/phi-1.5), augmented with a new data source of various NLP synthetic texts and filtered websites.
- **Availability of Phi-2 Weights**: `@entropi` mentioned that the **weights for Phi-2** had just been made available.


        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- Proposal by `@tonic_1` about the creation of **Langchain openagents using Gemini**.
- Feedback by `@tonic_1` on the recent **Google Gemini walkthrough**, describing it as disturbed due to the presenter's excessive coughing. Several interruptions and switches to elevator music were noted.
- User `@juanreds`'s update on **Java SDK acquisition** explaining their absence from a meeting while mentioning that they are still working on getting the Java SDK.

**AI Engineer Foundation Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/) (2 messages): 
        
- **Creation of Langchain Openagents Using Gemini**: User `@tonic_1` made a proposal for creating Langchain openagents using **Gemini**.
- **Google Gemini Walkthrough Issues**: `@tonic_1` described the recent Google Gemini walkthrough as disturbed by the presenter's excessive coughing. This led to several interruptions and switches to elevator music.


### ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/) (1 messages): 
        
- **Java SDK Acquisition**: User `@juanreds` apologized for missing the previous meeting, mentioning that they are **still working on getting the Java SDK**.


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord Summary

- Announcement of **Feature and Model Management with Featureform & MLFlow Webinar** scheduled for *December 19th at 8 AM PT* by `@marie_59545`. This session aims to assist various roles like Data Scientists and ML Engineers, among others in understanding how to leverage Featureform's data handling and MLflow's model lifecycle management tools. Registrations can be made through [this link](https://buff.ly/3TvlQh4).
- User `@misturrsam` interested in **online courses focusing on ML model deployment**, with a particular emphasis on Microsoft Azure, Google Cloud, and Amazon AWS platforms, seeking community recommendations.

**MLOps @Chipro Channel Summaries**

### ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/) (1 messages): 
        
- **Feature and Model Management with Featureform & MLFlow Webinar**: `@marie_59545` announced an upcoming webinar on *December 19th at 8 AM PT*, hosted by Simba Khadder. The session will demonstrate how to enhance machine-learning workflows using **Featureform's** efficient data handling and **MLflow's** model lifecycle management tools. The webinar is aimed at *Data Scientists, Data Engineers, ML Engineers, and MLOps/Platform Engineers*. The event is free and attendees can sign up at [this link](https://buff.ly/3TvlQh4).
- **About Featureform and MLflow**: Featureform is an open-source feature store used for managing and deploying ML feature pipelines, while MLflow is a system for managing the end-to-end ML model lifecycle. Both tools together create a robust environment for machine learning projects.


### ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/) (1 messages): 
        
- **ML Model Deployment Courses**: User `@misturrsam` requested recommendations for **good online courses focusing on ML model deployment** for specific platforms: **Microsoft Azure, Google Cloud, and Amazon AWS**.


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Benchmark Comparison of Phi and Zephyr**: `@albfresco` inquired if there is a **benchmark comparison of the new Phi and Zephyr** models, both of which are claimed to be strong 3B models with extraordinarily high benchmarks.
        

---
The Ontocord (MDEL discord) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
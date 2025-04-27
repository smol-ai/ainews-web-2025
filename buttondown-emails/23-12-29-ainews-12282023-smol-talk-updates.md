---
id: 42e09d30-0605-4da8-921b-0e4cf2fd2c43
title: '12/28/2023: Smol Talk updates'
date: '2023-12-29T10:32:18.263166Z'
original_slug: ainews-12282023-smol-talk-updates
description: >-
  **Nous Research AI Discord** discussions covered **AI placement charts**,
  **ChatGPT's** issues with **Latex math format** compatibility in **Obsidian**,
  and performance metrics of the **TinyLlama 1.1B** model on various benchmarks
  like **truthfulqa_mc** and **arc_challenge**. Users shared resources including
  the **MathPile** math-centric corpus and debated decentralized computation for
  models like **Mixtral**. Technical topics included **model conversion to
  AWQ**, **token counting**, and **finetuning** strategies. The community also
  discussed the **Obsidian** model, **TinyGPT-V** multimodal model release, and
  philosophical aspects of AI sentience and vulnerabilities.
companies:
  - nous-research
  - openai
  - tyrannosaurus
models:
  - tinyllama-1.1b
  - mixtral
  - tinygpt-v
topics:
  - benchmarking
  - latex
  - model-conversion
  - tokenization
  - finetuning
  - multimodality
  - decentralized-computation
  - philosophy-of-ai
  - model-vulnerabilities
people:
  - teknium
  - nonameusr
  - ldj
  - night_w0lf
  - max_paperclips
  - beowulfbr
  - error.pdf
  - skadeskoten
  - metaldragon01
---


<!-- buttondown-editor-mode: plaintext -->Not much news today so a great time to improve the summary quality - we've improved the scraper's ability to reliably grab metadata from twitter for the summarizer to consume. And now we also spit out all links for easy browsing. Should be more readable/usable as a link surfacer now!

 ![image.png](https://assets.buttondown.email/images/7e43e094-1a14-4c38-828f-3c7775b29a9b.png?w=960&fit=max) 


[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Discussion on AI Placement Charts**: In-depth discussion regarding placements on an AI attitudes chart. Users asked about specific individual placements and acknowledged the limitations of such presentation style.
- **ChatGPT's Compatibility with Latex Math Format**: Issues arisen about ChatGPT's inability to produce Latex math format compatible with Obsidian software and potential solutions suggested.
- **TinyLlama 1.1B Performance**: Detailed performance metrics of the TinyLlama 1.1B model across several tasks shared by `@teknium`.
- **AI Related Resources and Projects**: Sharing of various AI related resources including the new math-centric corpus MathPile, methods for building knowledge graphs, links to Open Large Language Model related GitHub repositories, and discussions on potential technological implementations.
- **Complex Discussion on AI-capabilities**: Users discussed the feasibility of decentralised computation for running large language models like Mixtral, the philosophical aspects of sentience and consciousness in AI, and potential exploitations of Large Language Models vulnerabilities.
- **Technical Queries about Large Language Models**: Queries about model conversion to AWQ, token counting methods for open-source models without importing entire libraries, and finetuning strategies for text classification enhancement.
- **Project Obsidian Discussions**: Focused talks on running and analysis of the Obsidian model from Nous, the process of vision model trainings, release of TinyGPT-V - a smaller multimodal model from Tyrannosaurus, and the community-oriented nature of open source projects.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (18 messages🔥): 
        
- **Placement on AI Chart**: `@nonameusr` asked `@ldj` about his placement on a certain chart representing attitudes toward AI. `@ldj` mentioned that he considered his placement a bit too south and west than he would personally identify. However, he also acknowledged the practical limitations of such a graphical arrangement and emphasized that the quadrant was ultimately what mattered. He identified himself as optimistic about human-level AI within our lifetime and optimistic about the advancement of civilization in the next 50 years.
- **Identifying Individuals on AI Chart**: `@night_w0lf` inquired about the placement of an individual named Tek, which `@nonameusr` clarified as being on the bottom left. `@max_paperclips` expressed confusion about an individual named Pico appearing in all four corners of the chart.
- **ChatGPT and Latex Math Format**: `.beowulfbr` shared concerns about ChatGPT not generating a Latex math format compatible with Obsidian and asked for a solution. `@nonameusr` suggested asking ChatGPT to respond in latex, but `.beowulfbr` responded that the suggestion partially solved the issue. 
- **Gary Marcus' Opinions on AI**: `@Error.PDF` inquired about whether Gary Marcus, an individual on the same AI chart, represented the perspective of AGI never being achieved, to which his own response was negative.


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (4 messages): 
        
- **Performance of TinyLlama 1.1B on different tasks**: User `@teknium` shared results of running `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` on different tasks. Some key results include:
    - On `truthfulqa_mc`, achieves mc1 score of `0.2203` and mc2 score of `0.3759`.
    - On `arc_challenge`, achieves an accuracy of `0.2782` and normalized accuracy of `0.3012`. The average score across this set of tasks is **52.99**.
    - On `agieval_aqua_rat`, achieves an accuracy of `0.1575` and normalized accuracy of `0.1693`. The average score across these evaluation tasks is **21.05**.
    - On `bigbench_causal_judgement`, achieves a multiple_choice_grade of `0.5053`. The average score across these bigbench tasks is **31.95**.
- No links or blogposts were shared or discussed in these channel messages.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (12 messages🔥): 
        
- **Common Working Model Discussion**: `@skadeskoten` mentioned the potential for creating ASICs or FPGA's once there is a common working model.
- **Generative AI for Math - MathPile**: `@metaldragon01` shared a link to a [Twitter thread](https://fxtwitter.com/arankomatsuzaki/status/1740564961032556942) by `@arankomatsuzaki` about a new math-centric corpus named MathPile. He also provided links to the project [website](https://gair-nlp.github.io/MathPile/), [GitHub repository](https://github.com/GAIR-NLP/MathPile/), and [abstract](https://arxiv.org/abs/2312.17120). `@gabriel_syme` inquired if it's a pretraining dataset.
- **Tinyllama Checkpoints Benchmarking**: `@teknium` managed the last three checkpoints of Tinyllama.
- **Modded Minecraft Discussion**: `@teknium` asked if `@1084792750001618965` plays modded Minecraft, to which `@max_paperclips` responded affirmatively but noted that he rarely has time for games.
- **Building Knowledge Graph with Instructor Project**: `@fullstack6209` shared a [Gist](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34) regarding building knowledge graph data with guidance and a project named Instructor. He stated that it takes about 30 minutes to digest a book while maxing out a 2080ti/3090 with "vllm".
      
Links mentioned:

- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://fxtwitter.com/arankomatsuzaki/status/1740564961032556942): Generative AI for Math: MathPile - Presents a diverse and high-quality math-centric corpus comprising about 9.5B tokens
- [asdf.py](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34): GitHub Gist: instantly share code, notes, and snippets.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (170 messages🔥🔥): 
        
- **Discussion on Mixtral Model's Capacity and Performance**: There's a detailed exchange among `@skadeskoten`, `@giftedgummybee`, and `@n8programs` about the feasibility of using decentralised computation for running the **Mixtral** model. They discuss its complex structure and the difficulties posed by latency and resource orchestration when considering a decentralised architecture. They conclude that current infrastructures don't favor decentralised computation for large language models like Mixtral. 
- **Cognitive Abilities and Sentience in AI**: `@teknium` shares a detailed**Hermes 2** response on the subject of consciousness, sentience, and qualia in Artificial Intelligence. The AI comments on the philosophical challenges these concepts present and the potential scientific methodologies to research them, stating that current understanding does not support the claim that AI can possess these attributes similar to living beings. 
- **Using Obsidian Model**: `@vic49.` inquires about running the **Obsidian** model. `@orabazes` suggests using the GGUF quant and llama.cpp for backend operation, referring to the original repository for gradio.
- **Finetuning Llama Variant for Text Classification**: `@shivdinho` is looking for suitable datasets for finetuning a version of **Llama** to enhance its text classification abilities.
- **Testing LLM Vulnerabilities**: `@shinchan5137` created a platform to test vulnerabilities in Large Language Models and has thus far been able to perform Exfiltration, Jailbreaking, and Prompt Hijacking. Additional vulnerabilities are being explored.
      
Links mentioned:

- [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://proceedings.mlr.press/v119/such20a.html): This paper investigates the intriguing question of whether we can create learning algorithms that automatically generate training data, learning environments, and curricula in order to help AI agen...
- [Chat with Open Large Language Models](https://chat.lmsys.org/)
- [Resurrect.ing](https://resurrect.ing)
- [Tweet from Emad (@EMostaque)](https://fxtwitter.com/EMostaque/status/1740306310204440677): I do wonder how the law/society will deal with embodied robots & their constant training
- [Tweet from Together AI (@togethercompute)](https://fxtwitter.com/togethercompute/status/1740586773296885767): @eating_entropy @Teknium1 @zamdoteth Should be up tomorrow!
- [Tweet from anton (@abacaj)](https://fxtwitter.com/abacaj/status/1740432829979459903): Tried chatglm3-6b-32k for the first time... and it&#39;s actually kind of good? I ran humaneval on it and it scored 60%. It has near perfect recall on 32k context (context image from reddit)
- [GitHub - NousResearch/Obsidian: Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️](https://github.com/NousResearch/Obsidian)


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (10 messages🔥): 
        
- **Conversion to AWQ**: `@fullstack6209` queried about converting a model to AWQ. `@beowulfbr` referenced page [here](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) that offers instructions on converting a model to AWQ using the AutoAWQ tool, though stressed that the current support in vLLM for AWQ is still under-optimized. `@casper_ai` further advised to draw examples from [here](https://github.com/casper-hansen/AutoAWQ/tree/main/examples) for a better understanding.

- **Token Counting of Open-Source Models**: `@fullstack6209` inquired if there exists a method for token counting for open-source models that doesn't require importing the entire transformers library. `@kcaverly` suggested that it might be possible to use tokenizers only, providing a relevant link to the [Hugging Face tokenizers](https://github.com/huggingface/tokenizers). `@vic49` affirmed the method, stating that they personally use tokens for count tokens. `@orangetin` offered an easy way to count tokens using the tokenizers library without importing the entire transformers library, providing an illustrative Python example. `@fullstack6209` expressed gratitude for the assistance and admitted that they had initially thought it would need a specialty library and that the process was slow.
      
Links mentioned:

- [AutoAWQ &#8212; vLLM](https://docs.vllm.ai/en/latest/quantization/auto_awq.html): 
,- [AutoAWQ/examples at main · casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ/tree/main/examples): AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference. - casper-hansen/AutoAWQ


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (82 messages🔥🔥): 
        
- **Obsidian Model Run Attempt and Help Request**: User `@vic49` initially asked for help with running the Obsidian model, expressing frustration over promised scripts that hadn't been delivered. User `@teknium`, the cofounder of Nous, tried to assist by sharing links to GitHub repositories and suggested pieces of code needed to inference Obsidian, although admitting uncertainty in how to successfully do so. He explained that Obsidian doesn't work with transformers directly, and that a custom class from Llava was used instead.

- **Community Guidance and Real Expectations**: User `@gabriel_syme` reminded `@vic49` about the volunteer basis of open source projects and the possibility of real-life events delaying promised updates, to which `@vic49` expressed disappointment over the broken promise.

- **Obsidian Model Analysis**: `@vic49` analysed the Obsidian model, stating that it relies on the Transformers library in part and includes custom classes. However, they felt they couldn't build a script without more guidance or a simple example, likening it to CogVLM's approach.

- **Introduction of TinyGPT-V**: `@qnguyen3` introduced `TinyGPT-V`, a smaller model built by Tyrannosaurus that is designed for multimodal use. He also referred to MobileVLM, a mobile-oriented multimodal vision language model.

- **Vision Model Training**: A discussion about vision model training unfolded. `@teknium` asked why vision models required multiple stages of training and why images couldn't be encoded as tokens into the regular SFT stage of an LLM. `@qnguyen3` argued that vision encoders usually lower the quality of image representation, hence the need for multiple stages of training. `@coffeebean6887` added that vision encoders were smaller, pretrained models and that training normally happened in two stages, with the end goal of mapping vision embeddings to the text embedding space.
      
Links mentioned:

- [Google Colaboratory](https://colab.research.google.com/drive/1C1FkoeZYBv3dZELaKgxahoZzWPfz0En8?usp=sharing): 
- [Tyrannosaurus/TinyGPT-V · Hugging Face](https://huggingface.co/Tyrannosaurus/TinyGPT-V): 
- [Paper page - MobileVLM : A Fast, Reproducible and Strong Vision Language Assistant
  for Mobile Devices](https://huggingface.co/papers/2312.16886): 
- [Paper page - TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones](https://huggingface.co/papers/2312.16862): 
- [GitHub - NousResearch/Obsidian: Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️](https://github.com/NousResearch/Obsidian): Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️  - GitHub - NousResearch/Obsidian: Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️
- [Obsidian/llava/serve/cli.py at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/llava/serve/cli.py)
- [GitHub - qnguyen3/hermes-llava](https://github.com/qnguyen3/hermes-llava): Contribute to qnguyen3/hermes-llava development by creating an account on GitHub.


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Data privacy and AI training discussions**: Users voiced concerns over individual privacy and discussed the possibility of opting out of AI training; 'Chat History and Training' option mentioned as a key control mechanism.
- **Career directions with AI**: Conversation centered around preparing for AI's impact on job markets, with opinions leaning towards deep specialization and empathetic entrepreneurship; possible broad impacts of AI on education emphasized.
- **AI-based tools for image generation**: Request for AI similar to Dalle for designing projects led to recommendations like Bing, albeit with a daily limit.
- **Comparisons between AI models**: Mixtral's potential to outperform GPT triggered various responses, with some pointing to the need for more potent models and others implying Mixtral's capacity for tasks typically handled by GPT-3.5.
- **ChatGPT Plus as prerequisite for custom GPT**: A user’s inability to create a custom GPT despite being a ChatGPT user revealed the likelihood of needing a ChatGPT Plus subscription for custom GPT.
- **Issues with OpenAI's User Interface**: Complaints about degradation in the quality of GPT-4's outputs, for instance with SQL queries, post-UI change and the necessity of reminding AI to exclude comments in the coding process were highlighted.
- **Reference to GPT's Knowledge File**: Suggestion made that GPT may default to its own training data and, thus, it could be beneficial to explicitly instruct GPT to check the knowledge file.
- **Frequent Human Verification Steps**: Reports of recurring verifications and puzzle tests during AI interaction; clarification needed over whether this is standard or a bug.
- **Interaction Restrictions in ChatGPT**: Complaints regarding ChatGPT's refusal to analyze personal data, specifically bank statements, led to speculation about limitations to prevent misuse of sensitive data.
- **Challenges with Large File Handling**: Suggestions to break larger files into smaller chunks to overcome 'Error in input stream' issues with chat GPT.
- **AI-assisted image prompt creation**: Question about transforming ChatGPT to an image prompt writing artist received a response with a [link to a custom GPT function](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer).
- **Payment issues and solutions**: Questions regarding payment methods for OpenAI services confirmed credit card as the only accepted form. An enquiry about continuing bills post account ban was directed to [OpenAI Support](https://help.openai.com/).

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (60 messages🔥🔥): 
        
- **Data Privacy in AI Training**: User `@the_boss7044` expressed a desire for the ability to opt-out individual conversations from being used in AI training. User `@tariqali` clarified that any conversations with ChatGPT will not be used in training if the "Chat History and Training" option is turned off.
- **Career Prospects in the Era of AI**: User `@apex_algorithms` initiated a discussion on how to position oneself in the face of advancing AI. The conversation revolved around two strategies: a) specialization in subject areas that require deep understanding and b) empathetic entrepreneurship. Several users, including `@dydzio` and `@bambooshoots`, provided varying perspectives on the impact of AI on jobs, particularly in software development. `@michael_6138_97508` suggested that presuming hard-set abilities and limitations for AI could be a mistake and that AI might pave the way for profound change in sectors like education. `@.dooz` added that AI programming might be easier to manage than AI software development due to the specific requirements of the latter.
- **AI and Image Generation**: User `@vix6262` inquired for a free AI to use as an image generator like dalle for designing projects. Suggestions such as Bing were provided by `@satanhashtag`, who also noted a limit per day.
- **Comparisons Among AI Models**: User `@dojan1` query if mixtral will soon be better than GPT was met with mixed responses. `@jaicraft` stated more substantial models are needed while `@exx1` suggested that mixtral could execute some of the tasks usually done by GPT-3.5.
- **Decreasing Usefulness of GPT for Code**: User `@Pandor` expressed dissatisfaction with the decreasing effectiveness of GPT in providing proper code, even contemplating cancellation of the subscription. Discussions were made about the possibility of using older models and other platforms.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (56 messages🔥🔥): 
        
- **Payment Method Discussion**: User `@grimgrinner` raised a question regarding the availability of making cash payments for OpenAI services. `@satanhashtag` and `@lugui` clarified that OpenAI only accepts payment through credit cards.
- **DALL-E Application**: A discussion was held on the possibility of using DALL-E in Discord, initiated by `@ryobdqui`. `@lugui` suggested that it's possible to develop a Discord bot that uses the OpenAI API to generate images, and mentioned there are public ones that could be downloaded and set up.
- **VPN & Puzzle Test**: `@busybenss` asked if VPNs could trigger a puzzle test for every message. The answer by `@satanhashtag` and `@lugui` was affirmative.
- **GPT-4 Accuracy and Message Limit Issue**: `@august8319` noticed a decrease in the accuracy of GPT-4 and an introduction of a message limit. `@mrhorse`, `@jaicraft`, and `@【ｐｅｎｕｌｔｉｍａｔｅ】` discussed that the changes are due to a switch to the GPT-4 Turbo model and implemented message caps to prevent server overload.
- **Technical Issue with Custom GPT**: `@odiseo3468` experienced an "Error searching knowledge" message when interacting with their custom GPT.

Link mentioned: [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1039968564699992106): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (30 messages🔥): 
        
- **Issues with Custom GPT Creation and Access**: User `@andreimotin` faced an issue related to creating a custom GPT. Upon accessing the relevant page on the OpenAI website, they received a "You do not currently have access to this feature" message. `@elektronisade` prompted whether `@andreimotin` had ChatGPT Plus, to which the response was negative, indicating that **ChatGPT Plus might be required to access this feature**.

- **ChatGPT Plus Payment Issue**: User `@.caan` reported that although their ChatGPT account was banned, they had still been billed for ChatGPT Plus. `@satanhashtag` suggested that only the official [OpenAI support page](https://help.openai.com/) could assist with payment issues.

- **ChatGPT Refusing to Analyze Personal Data**: User `@zoemdoef` complained about ChatGPT's refusal to help analyze their bank statements, potentially due to content restrictions imposed to prevent misuse of sensitive data like medical prescriptions and reports, as observed by `@elektronisade`.

- **Issues with GPT-4 Responses and Query Quality**: User `@PythonGuy` cited a decrease in the quality of responses from GPT-4, particularly after the UI change. Specifically, they mentioned the AI's tendency to insert incomplete components in SQL queries. The issue was confirmed by `@elektronisade` who advised reminding the AI to avoid comments and produce complete code solutions.

- **ChatGPT Issues for Users**: Users `@Delzi` and `@PythonGuy` faced an issue where their chat stopped working and encountered errors while sending messages. They had to repeatedly close the app or refresh the browser for a temporary workaround.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (28 messages🔥): 
        
- **GPT Knowledge File Reference**: `@solbus` suggested that when instructing a GPT to reference a knowledge file, it could be beneficial to specifically request the GPT to check the knowledge file. The reason being, GPT *may often defer to its own training data*. An example provided was: "When a user asks you about a screwdriver, search knowledge file 'screwdrivers.txt' for information relevant to their query".
- **Issue with Human Verification Steps**: `@vova5963` raised an issue regarding the frequent occurrence of human verification steps during the use of OpenAI's Chatbot. The issue was intermittently happening, with sometimes 5 consecutive verification prompts. There was confusion regarding these verification prompts, as others in the channel hadn't experienced it or had mistaken it for two-factor authentication (2FA).
- **Problem with Citation in Azure OpenAI**: `@shico_ai` brought up a problem where the citation in Azure OpenAI referred to the link of the file stored on Azure Blob Storage for their video data. They would prefer the citation to be the link of the video itself.
- **Code Repetition Bug**: `@happyg` reported a bug where the GPT they were building would repeatedly try to run the code `search('<query>')`, which was fixed by indicating the GPT not to use the Code Interpreter.
- **Slow GPT Performance**: `@dreamer100` and `@lodosd` noted that GPT handling of small text files (~20-30K) was very slow, comparing it to using a 20-year-old computer. They suggested that OpenAI was tweaking its speed to save computing power.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (14 messages🔥): 
        
- **Working with Large PDF Files**: User `@greenyellowblue` sought advice on how to use chat GPT with a large PDF file (60mb and 450 pages), which was leading to an "Error in input stream". `@madame_architect` suggested breaking the file into smaller chunks and renaming them sequentially (e.g., "Rulebook Vol 1", "Vol 2", etc.).
- **Converting ChatGPT into an Image Prompt Writing Artist**: `@errorsource` asked if there was a way to convert ChatGPT into an image prompt writing artist, to which `@madame_architect` provided a link to a custom GPT function for this purpose ([Custom GPT](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer)).
- **Message Limitations and Accuracy of GPT-4**: `@august8319` queried about GPT-4 seemingly becoming less accurate and the inception of a limit to 40 messages within 3 hours, which `@beanz_and_rice` confirmed has been in place for few months.
- **Generating Semi-Randomized Character Portraits**: `@ponderous_` asked for tips on generating a variety of character portraits with ChatGPT as the generated characters were appearing very similar and looked like models. No reply was given during this conversation period.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (14 messages🔥): 
        
- **Working with Large .PDF Files**: User `@greenyellowblue` experienced issues while attempting to work with a large .PDF file (60mb and 450 pages) which consistently resulted in an "Error in input stream". `@madame_architect` suggested breaking the file into smaller chunks, which seemed to resolve the issue.
- **Transforming ChatGPT into an Image Prompt Writing Artist**: In response to `@errorsource's` question about converting ChatGPT into an image prompt writing artist to get more diverse prompts, `@madame_architect` provided a link to a Custom GPT solution - i.e., [Midjourney Prompt Engineer](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer).
- **Limitations and Performance of GPT-4**: User `@august8319` raised concerns about GPT-4's perceived decrease in accuracy, as well as the message cap of 40 messages in 3 hours. `@beanz_and_rice` confirmed that the message cap had been implemented for a few months and suggested that the perceived decrease in accuracy might be a reflection of the model becoming "lazier".
- **Generating Semi-randomized Character Portraits**: `@ponderous_` is seeking advice on generating semi-randomized character portraits, aiming to avoid making characters look too similar or too much like models. There were no offered solutions in the discussed messages.
- **Chat Structure Review**: User `@bambooshoots` received advice from `@beanz_and_rice` to work on the structure of their message. A link was shared, but there was no direct quote or further discussion regarding the nature of the structure issue.


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Extensive discussions around **Language Model (LM) selection and performance**, with focus on memory requirements for different GPU setups and model sizes. Notably, users shared advice on running larger models on less powerful machines and compared model performance. Topics included differentiating between models, running multiple local servers, code integrations and handling model errors.
- Users shared **experiences and solutions to technical challenges** while using different LMs such as Dolphin 2.6 Mistral 7B GGUF and mixtral 8x7B dolphin 2.6. Embedding models using the LM Studio and running multiple instances were also discussed.
- In the subject of **integrations**, users discussed possibilities of running different language models on the LM Studio explorer in server mode, indicating potential integration scenarios.
- Hardware-related conversations revolved around matching larger language models with high-end **hardware setups**, using next-gen CPUs from AMD and Intel, and ways to run larger models via GPU with LM Studio.
- Relevant **beta releases issues were addressed**, in this case the Search Pane crash issue in the LM Studio application, which was solved by reinstallation.
- Key resources shared include [Silly Tavern Documentation](https://docs.sillytavern.app/usage/api-connections/openai/#proxy), [TheBloke's GPT-3 Models](https://huggingface.co/TheBloke), and [BBC-Esq's ChromaDB-Plugin-for-LM-Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio) among others.

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (119 messages🔥🔥): 
        
- **User Hardware and Model Selection**: `@qwerty_qwer` asked for advice on which language model to use with his 16GB GPU. `@starscreen.` inquired about the largest language model they could run on a machine with 64GB of memory, `@heyitsyorkie` suggested that `starscreen.` could possibly run 7b or 13b models on their setup. 
- **LM Studio and Silly Tavern Integration**: `@american_pride` shared an idea to integrate LM Studio into another platform named Silly Tavern. `@dagbs` shared a [documentation link](https://docs.sillytavern.app/usage/api-connections/openai/#proxy) covering API connections in Silly Tavern that could potentially enable this integration.
- **Models on Linux**: `@binepilo` had an issue trying to run phi-2 models on Linux. `@psipiai` suggested updating LM Studio to version 0.2.10 to solve this issue.
- **Differentiating Between Models**: `@mr.kittyhawk` and `@dagbs` suggested the use of tags to differentiate models by their use cases and capabilities. They suggested tags for model sizes, genres, usage scenarios, and hardware compatibility, as well as on broad categories such as `Programming`, `Narrative Writing`, `Character AI`. `@yagilb` added these suggested tags to the discussion forum.
- **Running Multiple Local Servers in LM Studio**: `@loganyang` queried the possibility to run multiple local servers in LM Studio. `@dagbs` clarified that only one model can be loaded at a time in LM Studio, while `@psipiai` suggested running multiple instances of LM Studio.
- **Embedding Models**: `@loganyang` also queried whether LM Studio supports language model embeddings. The community members didn't have a direct answer and indicated the feature may not be implemented yet.
- **Model Error**: `@pminev` was facing an issue running a specific model on LM Studio. `@dagbs` suggested it may be a GPU-related issue and recommended testing the model with GPU offloading disabled.
- **Extending LM Studio**: `@pminev` inquired about the possibility of adding functionality for the language model to call user's own APIs in LM Studio. `@dagbs` provided pointers to places in the Discord server where feature requests can be made.
      
Links mentioned:

- [Chat Completions | docs.ST.app](https://docs.sillytavern.app/usage/api-connections/openai/#proxy): Chat completion APIs include OpenAI, Claude, and PaLM. WindowAI & OpenRouter allows connection to these as well.
- [cognitivecomputations/dolphin-2.5-mixtral-8x7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b/tree/main)


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (22 messages🔥): 
        
- **Model Discussion and Performance**: User `@dagbs` shared the [**Dolphin 2.6 Mistral 7B GGUF** model](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF) indicating its unbiased/uncensored nature. User `@wordweaver` evaluated it and found all of the mistral variants to have adequate censorship, specifically mentioning **perlthoughts/mistral-instruct-v0.2-2x7b-moe-q4_k_m.gguf** model performed well, and denied the need of **Mixtral 8x** as it was too slow.
- **Model Hardware Requirements**: `@a1vx` discussed about the slowness of the **mixtral 8x7B dolphin 2.6** model on his machine. `@dagbs` clarified that it requires about 64GB of RAM and if not loaded in VRAM, it could slow down due to RAM to VRAM data transfers.
- **Other Models and Queries**: `@kujila` introduced the [**MixtralOrochi8X7B GGUF** model](https://huggingface.co/TheBloke/MixtralOrochi8x7B-GGUF) but `@dagbs` queried its goals and trainings as they are not listed in the shared link.
- User `@unskilless` inquired about the CodeNinja platform, its performance and configuration details.
- User `@dedded___` attempted to merge two models and faced quantization error, though details of the error were not provided.
      
Links mentioned:

- [TheBloke/dolphin-2.6-mistral-7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF): 
- [TheBloke/MixtralOrochi8x7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/MixtralOrochi8x7B-GGUF):


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (8 messages🔥): 
        
- **Running Local Server for Embedding Model in LM Studio**: User `@loganyang` inquired if there was a way to run a local server for embedding models in the LM Studio, as it would open up various integration possibilities. `@vic49.` confirmed that LM Studio does have a server mode. 
- `@loganyang` further voiced that on searching he wasn't able to find any embedding models available on [Hugging Face](https://huggingface.co). 
- **Solution for Embedding Models in LM Studio**: `@vic49.` suggested the use of [his unofficial plugin](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio) to use embedding models. It creates a ChromaDB vector database to work with LM Studio running in server mode.
- `@dagbs` engaged in a casual conversation asking about `@vic49.`'s arrival in the AMD with Windows world. User `@vic49.` ended the conversation by bidding good night.
      
Links mentioned:

- [sentence-transformers/all-mpnet-base-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2): 
- [GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio): Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode! - GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to wor...


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (39 messages🔥): 
        
- **Discussion on Loading Larger Models with Limited Hardware**: `@senpaira6969` brought up the topic of running a 70B model on a relatively less powerful laptop using a tool named air_llm. `@alphalogic_` added to the discussion, seeking advice on how they could leverage their higher-end hardware to run larger language models with LM Studio, expressing concerns about the difficulty of matching Language Models to individual hardware setups ([Link for further reading](https://discord.com/channels/1110598183144399058/1153759714082033735/1189065014292791376)).
- **Quantization on Various Hardware**: Users like `@pefortin` and `@totallybored` shared their experiences about running models of different sizes on various hardware configurations. `@pefortin` specifically gave a general guideline for what to expect when running 7B to 70B models on hardware configuration shared by `@alphalogic_`.
- **Potential of Next-Gen Intel CPUs**: `@funapple` raised a query about the potential of next-gen Intel CPUs with integrated NPUs in running larger models like 30B+ using only CPUs and RAM. `@totallybored` shared optimism for the utilization of AI CPUs for overall computation but emphasized waiting for more testing on these potentials.
- **Trouble Running Models on GPU with LM Studio**: `@alphalogic_` raised concerns about issues in loading the models on their device's GPU using LM Studio, despite correctly recognizing the Nvidia driver. `@totallybored` provided help, advising setting `n_gpu_layers` to `-1`, offloading the operations fully to the VRAM of the GPU, while suggesting a use of CPU with 8 threads. `@rem_l` also shared a similar problem and a method to solve it using the `n_gpu_layers` option and changing the number of CPU threads.
- **Direct Coding with Models**: `@rem_l` expressed their satisfaction using LM Studio directly with Visual Studio Code, claiming that it was extremely useful in generating code, bug hunting, debugging and creating complex search and replace regular expressions. They also mentioned using a few extensions:`Continue` and `GPT-Pilot`, to aid in their coding process.
- **Running Larger Models on M.2 slot**: `@pefortin` asked about connecting a 3090 GPU to an m.2 slot via an adapter/riser, raising concerns about the potential data transfer speeds for model loading and unloading. `@totallybored` confirmed that any bus slowdown would indeed impact the performance.
- **Future of AI CPUs**: `@totallybored` discussed the promising future of AI CPUs, especially the new generations planned by AMD and Intel. However, they pointed out that the current models are still laptop-exclusive. `@funapple` showed interest in next-gen CPUs for their potential in running larger models without needing VRAM.
      
Links mentioned:

- [Continue](https://continue.dev/): 
- [GitHub - Pythagora-io/gpt-pilot: Dev tool that writes scalable apps from scratch while the developer oversees the implementation](https://github.com/Pythagora-io/gpt-pilot): Dev tool that writes scalable apps from scratch while the developer oversees the implementation


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (4 messages): 
        
- **Search Pane Crash Issue**: User `@kujila` reported that the search pane was crashing back to the main screen after an input from a few days prior. They were advised by `@yagilb` to **redownload and reinstall** the app from the website to fix the issue. Following the suggestion fixed the issue, as confirmed by `@kujila`.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Discussions on training issues with **Mixtral** and **Mistral** models, including the negative impact of increasing **top_k**, an unaligned base model generating random text, handling fluctuating loss, and experiencing CUDA out-of-memory errors. Direct quote: "*Increasing top_k might activate experts on tokens they didn't encounter during training, potentially degrading performance.*" `[general]`
- Inquiries about utilizing similar datasets for Preference-Based DPO, the success of **Mixtral's** quantitative analysis, and instruction tuning with LoRa adapters. `[general][general-help]`
- An issue involving **tokenizing strategy** and **sequence length** leading to CUDA memory errors, with a proposed solution of using `pad_to_sequence_len: true` - and further exploration mentioned for this problem. `[axolotl-dev]`
- A suggestion to add a separate prompt format to the YAML was raised for specific datasets and to be more explicit in dataset configurations, with reference to chat templates. An issue with `sample_packing` causing unexpected VRAM spikes, with a confirmation that training worked fine when `sample_packing` was disabled. `[axolotl-dev]`
- Questions regarding full weight fine-tuning of Mistral 7B, with clarifications that it is equivalent to pretraining, and the significant VRAM requirement (160GB) was highlighted. A query about the necessity for both forward and backward passes to use flash attention for proper functioning also discussed. `[general-help]`
- The difference in performance between the `Mixtral-8x7B-v0.1` and `mistral-7b-v0.1` models was reported, indicating an improved train and eval loss when using `mistral-7b-v0.1`, however no speculation as to the reason was offered. `[general-help]`
- Discussion around synthetic data generation and using the llama index for the ragdata set in the 'rlhf' channel, although detailed conversation or context was not provided for these topics. `[rlhf]`

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (16 messages🔥): 
        
- **Effect of Increasing top_k**: `@stefangliga` indicated warnings about increasing top_k in training. Increasing top_k might activate experts on tokens they didn't encounter during training, potentially **degrading performance**.

- **Using Same Dataset for Preference-Based DPO**: `@dangfutures` asked if the same dataset is supposed to be used for Preference-Based DPO.

- **Quantization of mixtral**: `@dangfutures` inquired about the success of quantitative analysis of **Mixtral**.

- **Random Text Generation by Unaligned Base Model**: `@noobmaster29` experienced an unaligned base model generating random text irrespective of the input, using a simple phrase like 'hello' as their test case.

- **Training on Spicyboro Dataset**: `@dangfutures` disclosed training their **Mistrals** on the Spicyboro dataset, humorously implying that the model has learned potentially sensitive information from the dataset.

- **Concerns about Evaluating Loss and Fluctuating Loss**: `@mihai4256` shared his experience of encountering a fluctuating loss and a constant decrease in eval_loss throughout the training of his **LoRa model**. `_jp1_` advised increasing the batch size and reducing the learning rate to handle fluctuations, noting that it's okay as long as the **eval_loss is decreasing**. `_jp1_` also advocated for the immediate use of **wandb** for better understanding model performance.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (21 messages🔥): 
        
- **Tokenizing strategy and sequence length issue**: User `@_jp1_` reported an issue related to **tokenizing strategy** and **sequence length** during the training process. The users observed that sequences exceeding the defined `sequence_len` and not automatically truncated lead to CUDA out of memory errors. The issue was noticed when changing datasets in a stable environment.
- **Discussion on length filter**: From the code perspective, `@caseus_` reminded of the existing `drop_long filter` which should filter out sequences exceeding the defined length for all datasets. However, `_jp1_` mentioned that despite this, they still faced the issue and assumed that deleting sequences manually could have affected the packing, which led to the apparent disappearance of the issue.
- **Proposed changes in prompt format**: User `@caseus_` put forth the idea of adding a separate prompt format to the yaml. This could be specific to a dataset and default to formatting using that. The use of chat templates was also proposed to be more explicit in the dataset configurations.
- **Investigation of sample_packing issue**: `_jp1_` reported facing unexpected VRAM spikes during specific conditions when `sample_packing` was enabled. A discussion ensued with `@nanobitz` about potential causes, but no firm conclusions were reached. `_jp1_` confirmed the issue didn't occur with `sample_packing` disabled.
- **Exploration of potential solutions**: Among the discussed potential solutions was using `pad_to_sequence_len: true` instead of `sample_packing`. `_jp1_` confirmed that training works fine with `pad_to_sequence_len` enabled. 
- **Further investigation**: The conversation ended with `_jp1_` expressing the intention to further explore this issue and`@nanobitz` offering support on this matter.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (36 messages🔥): 
        
- **OOM Errors during Training**: `@_jp1_` was dealing with an [OOM error](https://discord.com/channels/891714134014103552/912454685702639708/922369707724046336) during the training of a 7B Mistral model with an 8k sequence length on a new dataset. The errors seemed to occur at random steps and appeared linked to the training data since the model worked fine with identical configuration but a different dataset. `@caseus_` suggested trying to write out to a file the data each step from the data loader for debugging purposes. However, `_jp1_` noted that the error disappears without using sample packing.
- **Discussion on Full Weight Finetuning**: `@athenawisdoms` inquired about the VRAM requirements for a full weight fine-tuning of Mistral 7B and whether full weight fine-tuning is the same as continued pretraining. `@noobmaster29` clarified that full tuning is pretraining and all the Axolotl full tune configurations are in bf16. The discussion also highlighted the large VRAM needs (160GB) for pretraining a 7B model, implying the need for high-end GPUs like the A100. 
- **Pretraining and Applying LoRa Adapters**: `@athenawisdoms` also asked whether one can apply the original `Llama-2-13B-Chat-fp16` LoRa adapter to a pretrained `Llama-2-7B-fp16` model and still achieve the same instruction tuning.
- **Flash Attention Query**: `@xzuyn` asked whether both the forward and backward passes need to utilize flash attention for it to work properly. `@caseus_` confirmed that it has to be used for both since the backward pass relies on the computation graph from the forward pass.
- **Mixtral vs Mistral Performance**: `@semantic_zone` reported that changing the base model from `Mixtral-8x7B-v0.1` to `mistral-7b-v0.1` significantly improved train and eval loss on their large classification dataset. They requested speculation on why this might be the case.
      
Links mentioned:

- [axolotl/examples/mistral/config.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/config.yml): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [GitHub - hiyouga/LLaMA-Factory: Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM)](https://github.com/hiyouga/LLaMA-Factory): Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM)


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (1 messages): 
        
emperor: generally people use both


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (3 messages): 
        
- **Data Generator Script Inquiry**: `@dangfutures` inquired if anyone has a script for synthetic data generation. No responses or further details were provided.
- **Mention of Self Rag**: `@dangfutures` mentioned "self rag", however the context was unclear, and no further discussion followed.
- **Use of Llama Index for Ragdata Set**: `@dangfutures` reported that they are using the llama index for the ragdata set and the rag fine-tune example. No links or additional information were given.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Discussion on **HuggingFace's new channels** for daily model updates and community training of a Hinglish model, alongside several **open-source updates**, including the release of Diffusers v0.25 and Transformers.js v2.13 with new features and models. 
- **Model performance enhancements** were noted, with Whisper being around 40% faster due to the integration of native SDPA and Torch backend for STFT; Mixtral Instruct was updated with 4-bit serialization.
- The guild had an active exploration of **Weights and Biases (wandb)** as a tool for machine learning project tracking, the differences between **GGUF and GPTQ model formats**, and challenges with **website loading and setting up custom training loops**.
- A notable discussion was centered on the management of AI paper content, with an emphasis on speed reading as a suggested strategy and a caution against using LLMs for summarization due to potential accuracy issues.
- Learning projects were shared in the today-im-learning channel, including building a **GPT model** from scratch and setting up a **vLLM server inside a Docker container**.
- Active conversation in the NLP channel revolved around Mystral embedding retrieval performance, a comparison of encoder-decoder and decoder-only architectural models, **model capacity**, the completion of a **data annotation tool**, and issues with loading a GPTQ model with Langchain.
- Diffusion related topics of discussion involved the Pokemon dataset, converting diffusers to single safetensors, an introduction to the SHAP-E model, using depth-conditioned controlnet weights, and questions about training methods for faces and textual inversion.
  
Resources shared:

- [Diffusers v0.25 release](https://github.com/huggingface/diffusers/releases/tag/v0.25.0)
- [Transformers.js v2.13 update tweet](https://twitter.com/xenovacom/status/1740037798650859755?s=46)
- [and more...](https://huggingface.co/blog/hwaseem04/drag-gan)

**HuggingFace Discord Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 messages): 
        
- **New Channels**: HuggingFace has introduced two new channels. <#1184447251746127913> will share a new model every day with relevant information and code, and <#1189605147068858408> is a community effort to train a Hinglish model.
- **Open Source Updates**: Diffusers v0.25 has been released with various features including aMUSEd, a lightweight and fast text-to-image model [link to release](https://github.com/huggingface/diffusers/releases/tag/v0.25.0). Transformers.js v2.13 is out with many updates including SegFormet and VITS which are available to use in the browser [link to tweet](https://twitter.com/xenovacom/status/1740037798650859755?s=46). 
- **Model Performance Enhancements**: Whisper is now approximately 40% faster due to the integration of native SDPA (Scaled Dot Product Attention) and Torch backend for STFT (Short-Term Fourier Transform) as detailed in this [tweet](https://twitter.com/reach_vb/status/1739358185994047524). Mixtral Instruct has been updated with 4-bit serialization [link to model](https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit).
- **Community Updates**: The MLX community has uploaded pre-converted MLX models to the Hub, and HuggingFace has released Fellow Highlights for the Winter Edition [link to highlights](https://huggingface2.notion.site/Hugging-Face-Fellows-Highlights-Winter-Edition-b26c2c7d3f9143ec88d98ec43b98af29).
- **Reading Recommendations**: Several blog posts have been shared including "Speculative Decoding for 2x Faster Whisper Inference" [link to blog](https://huggingface.co/blog/whisper-speculative-decoding), "Build an AI Chatbot to Run Code and Tweak plots" [link to blog](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat), and "Combating Evaluation Data Contamination in LLMs: Strategies for High-Quality Finetuning and Model Merging" [link to blog](https://huggingface.co/blog/rishiraj/merge-models-without-contamination).
      
Links mentioned:

- [Release v0.25.0: aMUSEd, 3x faster SDXL, interruptable pipelines · huggingface/diffusers](https://github.com/huggingface/diffusers/releases/tag/v0.25.0): aMUSEd is a lightweight text to image model based off of the MUSE architecture. aMUSEd is particularly useful in applications that require a lightweight and fast model, such as generating m...
- [Tweet from Xenova (@xenovacom)](https://twitter.com/xenovacom/status/1740037798650859755?s=46): 🤗 Transformers.js v2.13 - Holiday update! ☃️ In this version, we added:
1. SegFormer for semantic segmentation and image classification.
2. VITS for multilingual text-to-speech (&gt;1000 languages).
3. CLIPSeg for zero-shot image segmentation.
4. Table Transformer for table extraction.
5. DiT for document image classification.
6. SigLIP for zero-shot image classification.
7. RoFormer for masked language modelling, sequence classification, token classification, and question answering.
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://twitter.com/reach_vb/status/1737905584089846108): Common Voice 16 by @mozilla is out on the Hub! 🔥This brings a total 30,328 hours of audio spread across 120 languages! Out of the total 30K hours of audio 19.5K is validated! ✨
- [Tweet from dylan (@dylan_ebert_)](https://twitter.com/dylan_ebert_/status/1736857719620161895): 🚀 Announcing gsplat.js - a JavaScript Gaussian Splatting Library - Update 1.0
- [Tweet from younes (@younesbelkada)](https://twitter.com/younesbelkada/status/1739244971905966380): Following up from the great work from community that enabled bitsandbytes 4-bit serialization, I pushed Mixtral-Instruct-bnb-4bit on @huggingface for anyone that wants to easily load the model
https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://twitter.com/reach_vb/status/1739358185994047524): We made Whisper even faster. ~40% faster!! 🔥
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://huggingface2.notion.site/Hugging-Face-Fellows-Highlights-Winter-Edition-b26c2c7d3f9143ec88d98ec43b98af29)
- [Tweet from Awni Hannun (@awnihannun)](https://twitter.com/awnihannun/status/1737510739987120248): The crew at Hugging Face 🤗 made a bunch of pre-converted MLX models! Llama, Phi-2, Mistral, Mixtral (and instruct and code variations where available)!
- [Speculative Decoding for 2x Faster Whisper Inference](https://huggingface.co/blog/whisper-speculative-decoding)
- [Build an AI Chatbot to Run Code and Tweak plots](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat)
- [Combating Evaluation Data Contamination in LLMs: Strategies for High-Quality Finetuning and Model Merging](https://huggingface.co/blog/rishiraj/merge-models-without-contamination)
- [Drag GAN - Interactive Point-based Manipulation on the Generative Image Manifold](https://huggingface.co/blog/hwaseem04/drag-gan)


### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (42 messages🔥): 
        
- **Weights and Biases (wandb)**: In response to `@bennoo_`'s question, `@krebzonide` and `@natika1` explained that wandb is an online tracking tool for machine learning projects, useful for the evaluation of model performance. 

- **Difference between Model Formats**: `@jiha` initiated a discussion on different model formats, to which `_sabine_wren_` provided a detailed comparison between **GGUF** and **GPTQ**. GGUF is used with LLAMA models and prioritizes CPU compatibility, while GPTQ is optimized for efficiency on GPUs and leverages quantization methods.

- **Issues with Loading Platforms**: Users `@rk_man` and `@zhangjie_shanghai` discussed difficulties they were experiencing in loading a platform, with `@zhangjie_shanghai` suggesting using Chrome as a resolution to this problem.

- **Creating a Custom Training Loop**: `@goodtimes5241` sought advice on the process of fine-tuning a stable diffusion model for an audio generation project, specifically asking for resources on creating a training loop and saving weights for use in a stable diffusion pipeline.

- **Discussion on Reading AI Papers**: `@tonyaichamp` initiated a conversation on how to manage the vast amount of AI papers being published and asked for recommendations on platforms to discover, read, and summarize such content. `.naptastic` suggested speed reading and pointed out that LLMs (language model) should not be used to summarize papers due to potential accuracy issues.


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (2 messages): 
        
- **Building GPT from Scratch**: `@gag123` shared information about their ongoing project of learning to build **GPT** from scratch as per tutorials provided by **AndrejKarpathy**. They reported struggling with achieving the expected results, with constant fluctuation in loss and non-meaningful output produced by their model. For reference or assistance, they provided the link to their [repository](https://github.com/gagangayari/my-gpt) and the **AndrejKarpathy**'s [tutorial series](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- **Setting Up vLLM Server Inside Docker Container**: `@warmonger9626` discussed their successful setup of a **default vLLM server** running inside a Docker container within WSL on a Windows machine, utilizing their local CUDA GPU.


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **Mystral Embedding Retrieval Performance**: `@nickprock` raised a question regarding the higher retrieval score of **Mystral** embeddings on MTEB, which was not reflected on the leaderboard. `@merve3234` suggested that perhaps they hadn't submitted their results.
- **Comparison of Encoder-Decoder and Decoder-Only Architectural Models**: `@hieunguyen1053` inquired about the feasibility and potential performance of an encoder-decoder model like T5 with 1.6B parameters and pretraining on a 600GB dataset. They also wondered about how such a model might perform against a decoder-only model with equivalent parameters when fine-tuned with instruction data. `@merve3234` observed that differences between such models might be more dependent on dataset inputs.
- **Discussion on Model Capacity**: `@hieunguyen1053` expressed skepticism about the ability of a 1.6B parameter model to memorize a 600GB dataset, citing their tests on end-user products. The user highlighted a lack of clarity on whether retrieval processes were employed.
- **Data Annotation Tool**: `@stroggoz` announced the completion of their data annotation tool for named entity recognition but expressed difficulties in making the tool operable on other computers.
- **Issues with Loading GPTQ Model with Langchain**: `@.sgp` posted a query on how to load a GPTQ model with langchain. They shared their Python code and the error encountered, a RuntimeError linked to an attribute 'object' issue in numpy. The error message referred the user to the numpy 1.20 release notes on deprecations.

      
Links mentioned:

- [NumPy 1.20.0 Release Notes &#8212; NumPy v2.0.dev0 Manual](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations```):


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (9 messages🔥): 
        
- **Pokemon dataset suggestion**: `@sayakpaul` provided a link to a sample image-caption dataset available on HuggingFace: [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).
- **Converting diffusers to single safetensors**: `@sayakpaul` suggested using the script from the [diffusers GitHub repo](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) to convert diffusers to single safetensors for seamless VEGA application.
- **SHAP-E introduction**: `@sayakpaul` directed to the [SHAP-E model](https://huggingface.co/openai/shap-e) on HuggingFace, a diffusion process capable of generating 3D images from a text prompt.
- **Depth-conditioned controlnet weights**: `@sayakpaul` also shared a resource for [controlnet weights](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid) trained on stabilityai/stable-diffusion-xl-base-1.0 with depth conditioning.
- **Questions about training methods for faces and textual inversion**: `@matthewcroughan` raised questions about the effectiveness of textual inversion for faces, and asked if there is a summary available for different training methods.
      
Links mentioned:

- [openai/shap-e · Hugging Face](https://huggingface.co/openai/shap-e): 
- [diffusers/controlnet-depth-sdxl-1.0-mid · Hugging Face](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid): 
- [lambdalabs/pokemon-blip-captions · Datasets at Hugging Face](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions):


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- Bright discussions have taken place regarding the **limits and capabilities of prompt templates** across different Mistral models. One user noted that their custom template produced comparable results between small and medium models, with a bias towards medium. Referenced quote: "*...custom prompt template gives comparable results between small and medium models...*"
- Users shared project resources such as the **AirLLM project** for running 70B models on 4GB VRAM, albeit with performance concerns. This information included the **[AirLLM GitHub repository](https://github.com/lyogavin/Anima/tree/main/air_llm)**.
- Emphasized the need for **Free Chat UIs compatible with the Mistral API**. User `@fayiron` recommended the **sillytavern** which recently added support for the Mistral API.
- Open-source project **Microchain** was highlighted, now featuring Mistral support, improved token-usage metrics, and optimized agents. The project's **[GitHub repository](https://github.com/galatolofederico/microchain)** was shared.
- There was a question regarding the feasibility of **running the 7B model on 12GB of VRAM**. The recommended workaround was to use a lower resolution model like **openchat 3.5**, led by an available [link to the model](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main).
- Questions about **model overfitting and comparison** between base instruct models and fine-tuned models emerged. Alternatives like **Goliath 120B** and **OpenHermes 2.5 Mistral 7B** were discussed amidst tool recommendations like **[text-generation-webui](https://github.com/oobabooga/text-generation-webui)** for local model running.
- Users queried the advantage of using the [**MistralAI's Python Client**](https://github.com/mistralai/client-python) over the [**OpenAI Python package**](https://github.com/openai/openai-python) when working with Mistral API, with the consensus being the suitability of either depending on specific needs.
- **Mistral Medium performance feedback** on coding tasks also took place, with one user expressing satisfaction but wishing for speed optimizations. It was noted "*...Mixtral Medium averaged at 15 tokens/s and peaked at 45 tokens/s.*"

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (19 messages🔥): 
        
- **Limits and Capabilities of Prompt Template**: `@The Ledger Luminary` sought to define the term **"complex"** in relation to prompt templates used with different Mistral models. They stated that their custom prompt template gives comparable results between small and medium models, with a slight preference toward the medium model due to its higher quality output.

- **AirLLM for 70B Models**: `@unskilless` introduced a GitHub repository for **AirLLM**, a project that aims to run 70B models with 4GB VRAM ([link to repository](https://github.com/lyogavin/Anima/tree/main/air_llm)). However, `@peasantry ⚒` found the application to be slow and expressed concerns about the potential risk of **malware**.

- **Free Chat UIs for Mistral API**: `@mka79` asked for recommendations for free Chat UIs compatible with the Mistral API. `@fayiron` suggested **sillytavern**, which recently added support for the Mistral API.

- **Improving Microchain Open-Source Project**: `@.tanuj.` has been enhancing the open-source project **Microchain** by adding Mistral support, better metrics for token usage and optimizing agents to carry out tasks as intended ([link to repository](https://github.com/galatolofederico/microchain)).

- **Suggestions for System Setup**: In response to `@aldt440`'s query about recommended system setups, `@orabazes` proposed using multiple **3090's** for value. However, for inference with the 7B Mistral model, they mentioned that **minimal system requirements** would be sufficient.


Link mentioned: [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm): 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (8 messages🔥): 
        
- **Running 7b model on limited VRAM**: `@azetisme` inquired if the **7b model** can be run on 12GB of VRAM, even though 16GB are typically required.
- Consensus was that running the full 7b model with these specifications would not be possible, with `@ethux` suggesting using a lower resolution model such as **openchat 3.5**. A link to this model was provided: [`openchat-3.5-1210-GGUF`](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main) for potential use in terms of text-generation, transformers, GGUF, mistral, open chat, and C-RLFT.
- `@azetisme` thanked `@ethux` for the advice and decided to explore the suggested option.

Link mentioned: [TheBloke/openchat-3.5-1210-GGUF at main](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main):


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
kushagra4761: any guide for mistral 7b fintune on multi gpu


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (21 messages🔥): 
        
- **API Call Formatting**: User `@bam4d` clarified how to call the API with correct `{ "role": "user", "content": "message"}` formatting and mentioned how code would need the start/stop/instruct tokens when integrating with the raw downloaded instruct model such as Mistral-7B-Instruct-v0.1.

- **Model Overfitting**: User `.gue22` raised questions on whether the model could be overfitting. Several other users suggested exploring different versions of models, such as [Goliath 120B](https://huggingface.co/alpindale/goliath-120b) and OpenHermes 2.5 version of [Mistral 7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) to improve their results.

- **Correct Prompting Format**: User `@fayiron` highlighted the importance of using the correct prompt format and shared an example. This could have influenced `.gue22`'s results with Mistral and help mitigate the problem of garbage output early on in their machine learning experiment.

- **Model Comparisons**: An interesting discussion ensued between `.gue22` and `.tanuj.` on comparing base instruct models with fine-tuned models. `.tanuj.` suggested OpenHermes-2.5-Mistral-7B with 4-bit quantization running as a good benchmark for comparison.

- **Local Model Running**: `@fayiron` recommended the use of the tool, [text-generation-webui](https://github.com/oobabooga/text-generation-webui), for running models locally, a suggestion that `.gue22` showed interest in exploring.
      
Links mentioned:

- [mistralai/mistral-7b-instruct-v0.1 – Run with an API on Replicate](https://replicate.com/mistralai/mistral-7b-instruct-v0.1): 
- [GitHub - rbgo404/OpenHermes-2.5-Mistral-7B: OpenHermes 2.5 Mistral 7B is an advanced version of the OpenHermes 2 model, This enhancement has led to improvements in several non-code benchmarks such as TruthfulQA, AGIEval, and the GPT4All suite.](https://github.com/rbgo404/OpenHermes-2.5-Mistral-7B): OpenHermes 2.5 Mistral 7B is an advanced version of the OpenHermes 2 model, This enhancement has led to improvements in several non-code benchmarks such as TruthfulQA, AGIEval, and the GPT4All suit...


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **Mistral API Client Libraries Discussion**: User `@jakobdylanc` asked about the purpose and advantages of using [MistralAI's Python Client](https://github.com/mistralai/client-python) as opposed to the [OpenAI Python package](https://github.com/openai/openai-python) when working with Mistral API. User `@lerela` clarified that while the Mistral client is more lightweight and primarily focuses on completions, the OpenAI client supports many additional features. Hence, it's fine to use the OpenAI Python package if an application is already using it.

- **Feedback on Mixtral Medium**: User `@casper_ai` expressed appreciation for the performance of Mixtral Medium in coding tasks, highlighting improved performance compared to Mixtral Small and GPT 3.5. However, `@casper_ai` wished for speed optimizations as Mixtral Medium averaged at 15 tokens/s and peaked at 45 tokens/s.
      
Links mentioned:

- [GitHub - mistralai/client-python: Python client library for Mistral AI platform](https://github.com/mistralai/client-python): Python client library for Mistral AI platform.


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain 0.1 Documentation Update**: User `hwchase17` initiated an important discussion on updates to the LangChain documentation, inviting the community to give their feedback. The document updates include new additions to Quickstart guide, Conceptual Guides, etc., along with new pages for LCEL, LLMs, ChatModels, Agents, and the future updates are planned for the Use Cases docs. Feedback is requested on components of the API docs which need improvement and suggestions are sought for further additions. Community engagement around this update was accessed via [GitHub link](https://github.com/langchain-ai/langchain/pull/15281) and [Vercel Preview](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction).
- **LangChain Usage and Dataset**: Group discussions revolved around LangChain dataset suggestions, GPU utilization, and optimization, understanding LangChain's use for a large number of parameters, and usage of embeddings in LangChain. Notably, `rahuldey8431` provided a guide to use FireworksEmbeddings in LangChain which was complemented by a linked [Embedding model tutorial](https://python.langchain.com/docs/modules/data_connection/text_embedding/).
- **Crypto Scams Notice**: A pertinent issue raised by `justignorethesharks` focused on the presence of cryptocurrency scams in the community which was degrading trust and called for necessary action.
- **Community Hiring Announcements**: There were several instances of job postings by `tanny1208`, `dougdotcon`, with the same link to a discord invite for a web3job. This was consistently observed across '#langserve', '#langchain-templates', and '#share-your-work' channels, suggesting a prevalent hiring drive in the community.
- **Shared Work and Resources**: The '#share-your-work' channel also featured shared projects including an interactive app built using the Agent Framework and shared by `aabbhishekk0804`, with the link to the app [here](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant), and a comprehensive resource guide for creating a bedrock agent shared by `rajib2189`, accessible [here](https://youtu.be/OcMXPFZ5gbs).
- One inconsequential interaction was noted by `rameshvoodi` in the '#tutorials' channel with no relevant information provided.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **LangChain 0.1 Documentation Update**: `hwchase17` has announced an update to the **LangChain** documentation, seeking community feedback. The updates are accessible via [this link](https://github.com/langchain-ai/langchain/pull/15281) and [Vercel Preview](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction)
- **Improvements to Documentation**: Additions have been made to the Quickstart guide, Conceptual Guides and "When to use" tables for output parsers, agents, and advanced retrieval methods. Outdated pages have been removed as part of the update.
- **Forthcoming Updates**: Future updates will include how-to guides for creating custom LLM, ChatModel, Retriever, VectorStore, Agent, and Tool. Updates to the Use Cases docs are also planned.
- **Feedback Request from Community**: The team seeks feedback on which parts of the API docs need improvement and which integration pages need to be updated. They've also added dedicated pages for LCEL, LLMs, ChatModels, Agents, and are looking for suggestions for other additions.
- **Prompt Engineering Technique Updates**: As part of the updates, OSS models have been added to the quickstart guide. The team acknowledges that working with local models is still challenging.

      
Links mentioned:

- [[documentation] documentation revamp by hwchase17 · Pull Request #15281 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/15281): needs new versions of langchain-core and langchain
- [Introduction | 🦜️🔗 Langchain](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction): LangChain is a framework for developing applications powered by language models. It enables applications that:


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (19 messages🔥): 
        
- **LangChain Dataset and GPU Utilization**: `@asterix3651` asked for dataset suggestions that contain both reference summary and llm summary. `@breathesmall` raised an issue about GPU utilization when using langchain with ollama.
- **Cryptocurrency Scams Notice**: `@justignorethesharks` urged the community, mentioning a few users, to clear out the crypto scam content from all channels. He pointed out that there seems to be a considerable amount of mistrust in the community due to the lack of activity and communication from the teams. 
- **LangChain Use for Large Number of Parameters**: `@refik0727` inquired about the required RAM and GPU for using LangChain to build a LocalGPT or own LLM with parameters 3B and above.
- **Use of Embeddings in LangChain with Fireworks**: `@3h0480` asked about how to use embeddings in langchain with fireworks. In response, `@rahuldey8431` provided a detailed guide on how to incorporate FireworksEmbeddings in LangChain, even though `@3h0480` later faced an execution error.
- **InMemoryCache Location Query**: `@seththunder` asked about the exact location where InMemoryCache() is saved and how to create a unique cache for every user.
      
Links mentioned:

- [Text embedding models | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/text_embedding/): Head to Integrations for documentation on built-in integrations with text embedding model providers.


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (4 messages): 
        
- **Hiring Opportunities**: `@tanny1208` and `@dougdotcon` both posted the same unidentified hiring message with a link to a [Discord invite](https://discord.com/invite/web3job) targeting a web3job.
- **Questions and Inactive Members**: `@cryptossssun` asked if there's anyone available to answer questions, tagging two unseen users for attention.


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (5 messages): 
        
- `@tanny1208` and `@dougdotcon` announced a hiring opportunity with a link to Discord [Web3job](https://discord.com/invite/web3job).

- `@aabbhishekk0804` shared a deployed app on Huggingface space. This application answers document-related queries and also processes queries requiring Search APIs. The application was built using the Agent Framework. Check it out [here](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant).

- `@rajib2189` provided a useful resource on creating a bedrock agent with an action group, sharing a link to a [YouTube tutorial](https://youtu.be/OcMXPFZ5gbs) and a [Medium blog post](https://medium.com/@rajib76.gcp/aws-bedrock-agent-part-4-action).
      
Links mentioned:

- [ChatPdfAgent - a Hugging Face Space by Aabbhishekk](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant): 
- [AWS Bedrock Agents | Action Groups](https://youtu.be/OcMXPFZ5gbs): In this recording, I show how to associate and action group to a Bedrock Agent.Medium blog : https://medium.com/@rajib76.gcp/aws-bedrock-agent-part-4-action-...

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion on the **efficiency of LLMs** in the #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) channel. *@slono* shared their experience of a significant productivity boost with LLMs, prompting some humorous speculation about their identity by *@guardiang*. *@swizec* showed interest in the productivity metrics used by *@slono*.

- Conversation on the #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) channel revolved around Apache-2.0 licensed datasets on Hugging Face. *@swyxio* shared links to the **Evol Instruct Paper and Instruction Tuning Datasets** [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b) and [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4). The terms **"ragtuning"** and **"finetuning datasets and paper"** were mentioned by *@swyxio* and *@eugeneyan* respectively but without any additional context.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (4 messages): 
        
- **Efficiency of LLMs**: `@slono` shared their experience that **LLMs have been a 10-20x productivity boost** for them in comparison to their day job duties, causing `@guardiang` to jokingly question if `@slono` is an AI.
- **Measuring Productivity with LLMs**: `@swizec` showed interest in the metrics `@slono` mentioned, inquiring about how these productivity measurements are attained.


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (3 messages): 
        
- **Evol Instruct Paper and Instruction Tuning Datasets**: `@swyxio` shared two links to datasets on Hugging Face, both of which had their licenses changed to Apache-2.0. The specific dataset links shared were [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b) and [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4).
- `@swyxio` mentioned the term **"ragtuning"** but did not provide any additional context or information around it.
- `@eugeneyan` mentioned **"finetuning datasets and paper"** but did not provide any relevant links or supplementary information.

      
Links mentioned:

- [Update README.md · theblackcat102/evol-codealpaca-v1 at 01d1e3c](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b): 
- [Change license to Apache-2.0 · ise-uiuc/Magicoder-Evol-Instruct-110K at b0079be](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4):


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Setup for LORA or Full Finetuning of Mixtral**: User `@huevosabio` asked for advice on the preferred setup, including any **template code**, **GPU provider**, etc., for **LORA** or complete **finetuning of Mixtral**. They stated their expectation that **guides would be available now**, considering they have not done any proper training recently.
        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Summarizing PDF books and interpreting texts from images**: `@codermickey` asked if there are any recommended tools, prompts, or plugins to summarize PDF books. The user also inquired about methods for reading and interpreting text from images like diagrams and charts as part of summarization.
        

---
The Alignment Lab AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

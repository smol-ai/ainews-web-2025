---
id: fd48fb4a-2ba8-4a30-8320-96820a4f8e54
title: '[AINews] 12/31/2023: Happy New Year'
date: '2024-01-01T05:33:14.937445Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-12312023-happy-new-year
---

<!-- buttondown-editor-mode: plaintext --> ![image.png](https://assets.buttondown.email/images/d7934abf-71d8-428f-9121-c0d717339821.png?w=960&fit=max) 


[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Variations in **Dolphin and Mistral models**, with a spotlight on optimizing hardware and software configurations. Users compared Dolphin Mistral 7b versions and exchanged insights on bias and censorship in AI models. Meanwhile, challenges were flagged with Mixtral deployment in local machines and suggestions were provided to download models from HuggingFace. The community also debated on how GPU use, particularly vRAM, can influence processing speed.

     - *"the difference between Dolphin and other models is mainly in their fine-tuning"* `@dagbs`
     - *"for such hardware specifications, sticking with 7B q4 models would be the best practice."* `@heyitsyorkie`

- Users explored **Understand Emotional Intelligence** of AI models and endeavored to invoke personalities in AI with extended, colorful prompts.
     - [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)

- Community expressed the **smooth usability of LM Studio**, with discussions around software upgrades, hardware utilization, and UI appreciations. Lively debate also occurred around ChatGPT enhancements, API updates, and Autogen integration.
     - [GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT](https://github.com/billmei/every-chatgpt-gui)
     - [How to Train Your Own Large Language Models](https://www.youtube.com/watch?v=5qlLJrv_q-Q)

- Detailed dialogues on **hardware developments**, including building budget AI compute servers and the realities of costly, high-specification equipment.
     - *"Is the extra CPU/RAM speed of an R730 vs R720 worth the additional cost given that they're planning to use 64GB VRAM"* `@leviticus_slow`
     - [AMD ComposeTM](https://www.amd.com/en/products/accelerators/amd-compose)

- Issues and solutions revolving around **integrations with ChromaDB and Autogen**, where users elucidated the nuances of various integration options, managed downloading issues, and addressed operational disruptions.
     - *"I suggested downloading the updated requirements.txt and replace_pdf.py from the "src" folder on the GitHub repository to resolve any issues"* `@vic49.` 

- Exchanges on **New Year celebrations** marked the communal interactions in the guild.
     - [Oprah Winfrey GIF - Oprah Winfrey - Discover &amp; Share GIFs](https://tenor.com/view/oprah-winfrey-gif-26673456)

**LM Studio Channel Summaries**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (164 messages🔥🔥): 
        
- **Change of default models folder in LM Studio**: `@musixela` provides suggestions on changing the default models folder location in LM Studio. They suggest either using webui to download models and then connect to the folder via LM Studio, or creating shortcuts in the webui top directory.
- **Issues with Mixtral on local machines**: 
    - `@dagbs` mentions that the `mixtral` model presents operational challenges on local hardware due to its large size. An alternative smaller model, `Mistral`, is suggested which does not exhibit the same sizing issues.
    - `@xyrezz.sol` raises an issue with a `Dolphin Mixtral 2.5 8x7b Q4_K_M` model running slowly on his machine with 16gb CPU RAM and 6 GB VRAM. It is recommended by `@heyitsyorkie` that for such hardware specifications, sticking with 7B q4 models would be the best practice.
- **Discussion on hardware limitations**:
    - In the discussion initiated by `@.gregly` regarding hardware upgrades to his computer, it's concluded that the key to increased processing speed lies in expanding the vRAM of the computer’s GPU rather than upgrading the CPU.
    - `@dagbs`, `@miashusband`, and `@fabguy` discuss VRAM limits in various GPU models ranging from consumer cards limited to 24GB VRAM up to professional accelerators featuring up to 188GB VRAM.
- **Downloading models from HuggingFace using proxies**: 
    - `@cmpleo.` discusses the issue of being unable to access and download models from HuggingFace using LM Studio in China, even through a v2rayNG proxy. `@fabguy` suggests a workaround by downloading models directly from HuggingFace and then placing them manually into the LM Studio models folder.
    - Despite the workaround, `@heyitsyorkie` suggests that the issue might arise from HuggingFace being blocked in China, which might not be circumvented with a VPN when using LM Studio.
- **New Year's celebrations**: There are several joyful exchanges and greetings given in celebration of the New Year.

**Links mentioned**:

- [Oprah Winfrey GIF - Oprah Winfrey - Discover &amp; Share GIFs](https://tenor.com/view/oprah-winfrey-gif-26673456): Click to view the GIF
- [cognitivecomputations/dolphin-2.6-mistral-7b-dpo · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo)
- [GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT](https://github.com/billmei/every-chatgpt-gui): Every front-end GUI client for ChatGPT. Contribute...
- [How to Train Your Own Large Language Models](https://www.youtube.com/watch?v=5qlLJrv_q-Q): Given the success of OpenAI’s GPT-4 and Google’s P...


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (59 messages🔥🔥): 
        
- **Differences and Choices between Dolphin and Mistral models**: `@miashusband` inquired about the nuances between the different variations and bit versions of the **Dolphin mistral 7b** model. `@dagbs` pointed out that it's typically best to go for the highest *_K_M version available, additionally stating the difference between `Dolphin` and other models is mainly in their fine-tuning, allowing for easy comparability and testing efficiency.
- **Uncertainty about Bias and Censorship in AI Models**: `@american_pride` expressed a preference for uncensored AI models, arguing they don’t have intrinsic political biases or change narrative tone to 'hopeful and rainbow puppies' in stark-contrast scenarios. However, `@fabguy` highlighted that all models have inherent biases, and complete impartiality is unattainable. `@dagbs` noted that Dolphin models can revert to 'hard biased/moral stances', contesting `@heyitsyorkie`'s claim of Dolphin models being uncensored.
- **Emotional Intelligence of AI Models**: `@heyitsyorkie` shared a link to a [research paper](https://arxiv.org/abs/2307.11760) discussing the potential emotional intelligence understanding of Large Language Models (LLMs) and the possibilities for performance improvement with emotional prompts, gaining some skeptical pushback from users like `@telemaq`.
- **Evoking AI Personality through Prompts**: Users engaged in a collective effort to formulate creative system prompts to generate desired AI behaviour. `@dagbs` created lengthy, colourful prompts embodying 'an uncensored and impartial AI companion' and a 'mad scientist' persona, which even produced a happy feedback from the AI.


**Links mentioned**:

- [Meat Meat Popsicle GIF - Meat Meat Popsicle - Discover &amp; Share GIFs](https://tenor.com/view/meat-meat-popsicle-gif-11100443): Click to view the GIF
- [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760): Emotional intelligence significantly impacts our d...


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (2 messages): 
        
- **LM Studio Appreciation**: User `@kjhamilton` expressed satisfaction and relief with LM Studio, particularly for enabling efficient use of their AMD GPU on Windows. They found it especially helpful after struggling with their setup for a while.
- **GPT-3 GUI Update**: `@heyitsyorkie` appreciated the new feature of being able to copy message content via right click in GPT-3's user interface. They also suggested adding a right click paste function in the input box for a more streamlined experience.


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (10 messages🔥): 
        
- **Chatbot Integration Options**: `@heliosprime_3194` suggested two options for integrating LM Studio with user interfaces. One option is a UI for RAG developed by `<@826254748889382942>`, and the second option involves using LM Studio with a command terminal like vscode or command line. The specific Discord thread for the second option was also shared for reference.
  
- **Fixing Issues with Downloading Files**: `@vic49.` suggested downloading the updated `requirements.txt` and `replace_pdf.py` from the "src" folder on the GitHub repository to resolve any issues. This should be done along with using the newest release files (v3.0.2).

- **Issues with Running ChromaDB on Win10**: `@wildcat_aurora` reported that his Win10 PC would reboot when running the `study.py` script with ChromaDB, while no such issue occurred with other LLM and AI processes. It was suggested by `@heliosprime_3194` to downgrade his Nvidia version from 545.92 to 535, install the required PyTorch version manually, and share the Conda list for troubleshooting.

- **Solution Found for Rebooting Issue & Feedback Regarding Data Extraction**: After manually installing PyTorch, `@wildcat_aurora` was able to avoid PC reboots, implying that incorrect PyTorch version might be the cause. He also observed that certain models from LM studio, such as Zephyr and Mixtral 2.6, were not extracting as much data from the database as expected.

- **Suggestions to Improve Data Extraction**: `@heliosprime_3194` suggested using a more advanced embedding model and modifying the chunk file sizes in the `study.py` script. He also mentioned changing the preset in the config.json file in LM Studio, to craft prompts that can help recheck information, which could address the less than optimal data extraction experienced by `@wildcat_aurora`.


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (50 messages🔥): 
        
- **Building Budget AI Compute Server**: `@leviticus_slow` is planning to build a budget AI compute server using 2 Nvidia Tesla cards in a PowerEdge. They asked whether the extra CPU/RAM speed of an R730 vs R720 is worth the additional cost given that they're planning to use 64GB VRAM. 
- **Impact of GPU on Processing Speed**: `@zhynem` noticed about double the tokens per second speed when enabling the Apple Metal (GPU) option on their machine/model/context. They and `@totallybored` discussed the potential impact of the quant size, specifically using *lmcocktail phi 2* with the Q8_0 quant.
- **Context Size Impact on Processing Time**: `@Pierre-jean Lainé` questioned why a larger context size leads to a longer time delay before processing, regardless of the actual prompt size.
- **GPU Utilization on Windows**: `@madan.pandit` sought assistance determining if their GPU is being utilized, as their Windows performance monitor showed no GPU usage. `@fabguy` asked about their n_gpu_layers setting and whether dedicated vRAM utilization changes when loading/ejecting a model in LMStudio. 
- **Discussion on Mixtral and Alternative LLMs**: User `@heyitsyorkie` advised that 8GB GPU plus *Mixtral Q8* would be problematic and recommended *OpenHermes 2.5 Mistral 7b* for `@madan.pandit`'s hardware. `@pefortin` and `@heyitsyorkie` confirmed returning to *Openhermes mistral* as a consistently good choice.
- **Expensive Hardware**: `@dagbs` shared a link to a powerful AMD acceleration platform with 1.5TB HBM3, prompting a discussion about its high cost and potential uses. Users speculate that businesses in R&D, developer assistance, medical research, and AI might invest in such hardware.


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (29 messages🔥): 
        
- **OpenAI Version and Autogen Compatibility Issue**: `@heliosprime_3194` suggested upgrading to **OpenAI 1.3.1** to resolve an error message received with older versions each time the OpenAI API updates. `@tyler8893` experienced the same issue, even after downgrading from OpenAI 1.3.7 to 1.3.1, and planned to further investigate in a new conda environment. `@heliosprime_3194` offered to share their **conda** list if it could be helpful.

- **OpenAI Authentication Error**: `@totallybored` and `@ftl24` faced an `AuthenticationError` with the API key, which was later clarified by `@dagbs` and `@tyler8893`. They explained that a string value, even if "null", must be provided for the **"api_key"** parameter to resolve the issue.

- **Issues with Function Calls and LM Studio**: `@tyler8893` expressed difficulty with function calls using LM studio. They mentioned **functions work fine with GPT**, but not with **LM Studio**. They speculated the issue could be addressed in a future update.

- **Updates to Autogen and memGPT**: `@tyler8893` and `@dagbs` discussed the challenge of keeping up-to-date with changes and updates to **Autogen** and **memGPT**. They noted changes could occur every other week and that the OpenAI API lacked a standardization like PEP, causing rules to be "free-flowing".


### ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 
        
rouw3n: <@1164606940098334843> use oogaboga webui problem solved


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- *Debate on Bingsly vs GPT-4*: `@Rock` finds **Bingsly** more effective in comparison to GPT-4 for coding and initiating conversations, whereas `@arevaxach` holds an opposing view citing Bingsly's tendency to lie and its unsatisfactory interaction quality. 
- *Discussion on Assistant API*: Users, including `@lugui`, suggested that streaming is a better option for faster data retrieval due to the time-consuming nature of non-streaming use of the assistant API.
- *Chat about Sam Altman*: The discussion was brought up by `@iiimandalorianiii` who views Sam Altman as ambitious, possibly monopolizing Language Models, but still expressed support for him.
- *Interest in AI technologies and AI-generated songs*: `@sarcasm83` enquired about AI technologies and shared examples of AI-generated songs: [Kurt Cobain singing "Gotye - Somebody that I used to know"](https://www.youtube.com/watch?v=212i9-aqMGY) and [Chester Bennington singing "Bring Me The Horizon - Sleepwalking"](https://www.youtube.com/watch?v=SiwGwjy0olg).
- Problems with **ChatGPT consistency, speed, crashes, functionalities**, as well as overstepping bounds with NSFW content, were discussed with various strategies suggested to address these issues, including adjusting system prompts, using guardrails, checking the network connection, managing GPTs carefully, and regulating content to comply with Terms of Services.
- *Addressing Technical Challenges with GPTs*: Users debated around issues such as difficulty guiding a Turbo35 model (`@kyper`), trouble with counting in large language models, and managing slow responses. Eventually, potential solutions put forward include using pseudocode, understanding API's lack of context retention, crafting well-structured sentences, and backing up data regularly to prevent loss.
- *Compliance with Policies*: `@eskcanta` urged users to comply with OpenAI's [usage policies](https://openai.com/policies/usage-policies), warning of potential account suspension or termination for discussions on disallowed content. 
- *Focus on Prompt Engineering*: `@iiimandalorianiii` noted the novelty in the term "prompt engineering" being treated as an established job. They pointed out that optimized prompts' understanding is still in the early stages, primarily pursued by a few enthusiastic individuals.
- In respect to the **limits of messages and API Costs**, participants discussed the cost implications for bumping beyond the limit of 40 messages per hour when using ChatGPT. There is consensus, albeit implied, on the beneficial aspects of learning to code over solely relying on AI. The use of OpenAI's Copilot in complement to GPT-4 was also touched on.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (27 messages🔥): 
        
- **Bingsly and GPT-4 Comparison**: `@Rock` stated they find **Bingsly** more useful than GPT-4 in starting a conversation and coding, while `@arevaxach` disagreed, indicating that Bingsly has a toxic personality, is prone to lying and generally doesn't provide satisfactory interaction. 
- **Assistant API Discussion**: A discussion occurred regarding the assistant API, where `@lugui` explained that for non-streaming use, users need to wait for the full generation process to complete, which can be time-consuming. For this reason, streaming was suggested as an option to retrieve data as it is generated.
- **Sam Altman Discussion**: `@iiimandalorianiii` brought up the topic of Sam Altman. While there was minimal engagement from others on this topic, they perceive Sam as ambitious and business-minded, potentially monopolizing Language Models, but still supportive of him.
- **AI Enthusiasm and Technologies**: `@sarcasm83` enquired if there are channels dedicated to discussions around various AI technologies including AI-generated songs. They provided [Kurt Cobain singing "Gotye - Somebody that I used to know"](https://www.youtube.com/watch?v=212i9-aqMGY), and [Chester Bennington singing "Bring Me The Horizon - Sleepwalking"](https://www.youtube.com/watch?v=SiwGwjy0olg) as examples.


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (142 messages🔥🔥): 
        
- **Inconsistency in GPT4 Chatbot Response**: `@abhijeet0343` shared a challenge with their chatbot developed using GPT4. The bot exhibits response inconsistency when data is in PDF format, sometimes returning fewer bullet points in the answer than expected. Several solutions were proposed including being assertive in the system prompt, using guardrails, or implementing a code interpreter to count bullet points.

- **Discussions on AI Counting Capability**: There was a conversation regarding the capability of AI and large language models (LLM) in counting. Some users believe that AI, in general, can count but LLMs have specific issues with it.

- **New Year's Celebration**: Many users took the opportunity to wish everyone a happy New Year.

- **Technical Issues with Chat GPT**: Users (`@Rosyskull`, `@quanta1933`, `@mrcrack_`, `@millymox`) reported having issues with Chat GPT, including it being slow, hanging, and providing a drop in quality outputs. `@Darthgustav` pointed out that it could be network-related between the users and the GPT servers.

- **Concerns about the Limit of Messages and API Costs**: `@slimified` expressed concerns about the limit of 40 messages per hour when using ChatGPT for application development and sought for ways to get past this limit. `@darthgustav` suggested using API calls but highlighted the potential cost. Conversations ensued on the value and cost-effectiveness of learning to code versus using AI as a development assistive tool. Furthermore, some users discussed the use of OpenAI's Copilot in conjunction with GPT-4.

**Links mentioned**:

[GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails): Adding guardrails to large language models. Contri...


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (66 messages🔥🔥): 
        
- **Issues with ChatGPT Verification and Slow Responses**: User `@ekot_0420` reported issues with ChatGPT taking too long to verify if a user is human. `@rutrruns` also mentioned experiencing slow responses leading to crashes.

- **Recovering Lost GPTs**: `@georgip` reported a GPT disappearing with the error "GPT inaccessible or not found". `@darthgustav.` suggested starting work on a new version of the GPT and waiting for possible recovery, while regularly backing up all data to prevent data loss in the future.

- **Impact of Rushed Work on GPTs**: `@darthgustav.` advised being careful and slow with updates, considering the autosave feature of the GPTs. `@mysticmarks1` and `@darthgustav.` warned against hasty decisions, especially when deleting conversations.

- **Issues with GPT Responses and TOS (Terms of Service) Violations**: User `@idkwhyigotdeleted` reported getting flagged due to an unpredicted NSFW response GPT generated to a prompt about eggplants. Users including `@gamerg.` and `@satanhashtag` advised going through chat history and editing/deleting any content that might cause a flag.

- **General Technical Problems**: Users including `@lowkeyhighbrow` and `@1984.dystopia` reported unspecified technical issues with GPT-4 and GPT responses respectively. `@not_richard_nixon` reported getting a "User quota exceeded" error when trying to upload an image to GPT-4's chat on various browsers. `@misterfyre` mentioned being unable to add a new payment method.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (19 messages🔥): 
        
- **Working with Turbo35 Model and User Confirmations**: User `@kyper` was trying to guide a turbo35 model to call functions but with a requirement for user confirmation. They struggled to get it to work consistent and sought advice on possible ways for resolution.
- **Pseudocode Suggestion**: `@darthgustav.` suggested trying pseudocode, highlighting that **GPT-3.5 Turbo** deals well with it. However, this suggestion didn't resolve `@kyper`'s problem.
- **API and Context Limitation**: `@darthgustav.` noted that the API does not retain context, which might have been the cause for the problem `@kyper` was facing.
- **Successful Solution**: Ultimately, `@kyper` resolved the issue by storing a "semi-started" function call and then added a "confirm_function" function that takes a true/false and a function-call id as parameters. They have a full client with context stored in a db to achieve the desired behaviour.
- **Discussion on Language Use**: There was a discussion about language use with `@darthgustav.` stating that *a well-crafted sentence is the best pseudocode, but such sentences are rare*. `@iiimandalorianiii` responded humorously, suggesting the sentence quality may depend on individual writing style.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 messages): 
        
- **OpenAI Usage Policies**: `@eskcanta` highlights the importance of adhering to OpenAI's recently updated [usage policies](https://openai.com/policies/usage-policies), especially in relation to disallowed content. Any discussion of disallowed content can lead to account suspension or termination. They also pointed to a reference in channel <#1107255707314704505> for additional context. 
- **Prompt Engineering as a Term and Job**: `@iiimandalorianiii` finds it amusing that the term "prompt engineering" is being used like it's an established job, considering the fact that those at the forefront are few online individuals. They, however, acknowledge a gap in understanding of optimized prompts, validating the importance of the term.

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 messages): 
        
- **OpenAI Usage Policies**: `@eskcanta` shared a link to [OpenAI's updated usage policies](https://openai.com/policies/usage-policies) and emphasized the importance of compliance to these policies. They cautioned that discussions around disallowed content could result in account suspension or termination.
- **Prompt Engineering Discussion**: User `@iiimandalorianiii` made observations on the use of the term "prompt engineering," noting that the concept is not yet well-established and is mostly being driven by a handful of dedicated individuals who are investing large amounts of time into it. They also recognized a knowledge gap in the understanding of optimized prompts.

**Links mentioned**:

[Usage policies](https://openai.com/policies/usage-policies)


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- Engaging discussions on understanding **Local Attention Parameters** in Jax-flax with a focus on better parameterization and a suggestion for chunking the data for cross-chunk interaction. Direct code reference - [source-code link](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8).
- Various off-topic discussions, including a user sharing their experience deploying a **RAG application**, another starting a **non-AI ethics review board**. Mention of admiration for the **Open LLM Leaderboard** and announcement of a potential open-source project for a **framework** developed for *multi GPU fine-tuning, batch inference/serving*, and further optimizations.
- Sharing of **interesting links** ranging from articulating the use of minhash similarity filtering, alignment phrases filtering, foreign languages filtering, filtering out URLs in projects, to a mixture of artificial intelligence developers' interview and project. Recommended articles include [Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T), [SplaTAM](https://spla-tam.github.io/) and Alibaba's [DreaMoving](https://fxtwitter.com/heyBarsee/status/1741106778849300900).
- Discussion around **Hot-swappable LoRa** that allows for model finetunes to be quickly switched via API, insights around **Mistral-based Mixtral Experts** with resource sharing, project showcase of **TinyLlama Project** aiming to pretrain a 1.1 billion parameter LLaMA model on 3 trillion tokens for compactness and applicability with LLaMA-based open-source projects.
- Inquisitive discussions in the **Ask-about-LLMs** channel around Amazon's new large language models, **Titan Text Express and Titan Text Lite**. Unconventional idea proposed for improving model performance, interest shown for known failures of **ChatGPT**, exploration for improving performance of English trained **LLMs** on the Czech language, and queries about suitable base models for HF's Auto Train feature.

**Nous Research AI Channel Summaries**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (4 messages): 
        
- **Understanding of Local Attention Parameters**: `@euclaise` initially suggested a different parameterization (`nxw`) for the local attention function [(source code)](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8). In response, `@joey00072` expressed confusion over the parameters, expecting the shape to be `(nh, T, T)`. `@euclaise` conceded his explanation might be unclear.
- **Suggestion on Chunking Data**: For a more practical approach, `@euclaise` suggested chunking the data and adding the past chunk with a mask for cross-chunk interaction. The local attention function can then be `vmap`'d over the chunks.

**Links mentioned**:

[local-attention-flax/local_attention_flax/local_attention_flax.py at e68fbe1ee01416648d15f55a4b908e2b69c54570 · lucidrains/local-attention-flax](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8): Local Attention - Flax module for Jax. Contribute ...


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (28 messages🔥): 
        
- **RAG Application Deployment**: User `@gabriel_syme` shared an experience about deploying a RAG application to 7k people calling it a "disaster", which they learned the hard way.
- **Non-AI Ethics Review Board**: User `@fullstack6209` announced that they're starting a non-artificial intelligence ethics review board aimed to issue ethics guidelines for real beings.
- **Open LLM Leaderboard**: User `@Error.PDF` expressed their admiration for the Open LLM Leaderboard.
- **Framework for Multi GPU, Fine Tuning and more**: User `@carsonpoole` shared about a framework they developed, which includes features such as multi GPU fine tuning, merging models, batch inference/serving, converting dense models to LoRAs, exporting loras to dense weights and much more. They also mentioned considering open sourcing (OSSing) it. Their intent was applauded by `@giftedgummybee`.
- **Inference Optimizations**:`@carsonpoole` discussed that the framework uses customized CUDA graphs through PyTorch for inference, achieving around 500 tokens per second with mistral on a single A100 unit, with a batch size of 32. They also shared their benchmark result (585 precise), and mentioned the potential for further optimizations.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (27 messages🔥): 
        
- **Minhash Similarity Filtering**: `@ldj` discussed the use of minhash similarity filtering alongside alignment phrases filtering, foreign languages filtering, filtering out URLs, and ANSI escapes in projects. They plan to mention these steps in a forthcoming paper and/or in the Amplify-Instruct Repo.
- **Interview with Tri Dao and Michael Poli**: `@ldj` highlights Tri Dao's discussion about the differences between Striped Hyena and Mamba and his future plans. The [interview](https://youtu.be/OFFHiJzPpCQ?si=uk2dTVrYmLHBlCyn) is available on YouTube.
- **Tiny Llama**:` @yobibyte` shared a link to the completed [Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) project and raises the question about the possibility of a similar approach for Tiny Hermes or Tiny Capybara.
- **Real-time SLAM and 3D Gaussians**: `@spirobel` proposes an alternative to Gaussian creation via training. They shared a link to [SplaTAM](https://spla-tam.github.io/), a real-time method for 3D Gaussians in SLAM.
- **DreaMoving by Alibaba**: `@nonameusr` shared a [tweet](https://fxtwitter.com/heyBarsee/status/1741106778849300900) about Alibaba's release of DreaMoving, a technology for animating using a single image or text prompts.

**Links mentioned**:

- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://spla-tam.github.io/)
- [Tweet from Eric Hartford (@erhartford)](https://fxtwitter.com/erhartford/status/1741651883108999295?): https://huggingface.co/cognitivecomputations/yayi2...
- [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [Interviewing Tri Dao and Michael Poli of Together AI on the future of LLM architectures](https://youtu.be/OFFHiJzPpCQ?si=uk2dTVrYmLHBlCyn): The introduction to this post can be found here: h...
- [Artificial Intelligence | 60 Minutes Full Episodes](https://www.youtube.com/watch?v=aZ5EsdnpLMI): From January 2019, Scott Pelley&#39;s interview wi...
- [Tweet from Barsee 🐶 (@heyBarsee)](https://fxtwitter.com/heyBarsee/status/1741106778849300900): It&#39;s been 24 hours since Alibaba released Drea...


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (1 messages): 
        
- **New Year Greetings**: User `@teknium` wished everyone a Happy New Year using various emojis.


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (129 messages🔥🔥): 
        
- **Hot-swappable LoRa**: `@fullstack6209` discussed the impending mainstream popularity of hot-swappable LoRa, which allows for model finetunes to be quickly switched via API. They referenced a company, Openpipe, that is claiming to beat GPT-4 on specific tasks using this technique. `@ldj` and `@spirobel` questioned its advantages over quickly swapping different LLM finetunes. `@spirobel` pointed out that this technique allows for batched inference of multiple PEFT loras at the same time.
- **Mistral-based Mixtral Experts**: `@spirobel` shared a [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/4611) that revealed that Mixtralx8, a model comprising of eight experts, was made using Mistral 7b as a common ancestor. They intrigued the group with the idea of extracting the differences between the models as PEFT adapters, to which `@giftedgummybee` responded that this has been done before.
- **TinyLlama Project**: `@giftedgummybee` shared a project aiming to pretrain a 1.1 billion parameter LLaMA model on 3 trillion tokens. This model, termed TinyLlama, aims to mesh compactness with the ability to be used in conjunction with open-source projects built upon LLaMA. More details of the project can be found [here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/README.md). 
- **LASER Interventions on Pallas-0.5**: `@mihai4256` presented initial findings from a project in which LASER, using torch.svd_lowrank, is applied to various layers of a model with the hope of improvement. Initial findings were not indicative of strong improvement in terms of accuracy or speed, but did show slight potential for memory and disk space savings.
- **Hydra MOE Project**: `@night_w0lf` queried about the status of the Hydra MOE project which seems stalled, to which `@teknium` suggested they could ask the project participants directly for any updates. 


**Links mentioned**:

- [README.md · TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T at main](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/README.md)
- [Mihaiii/Pallas-0.5-LASER-0.1 · Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1)
- [llama.cpp/examples/finetune/finetune.cpp at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/finetune/finetune.cpp): Port of Facebook&#39;s LLaMA model in C/C++. Contr...
- [GitHub - uukuguy/multi_loras: Load multiple LoRA modules simultaneously and automatically switch the appropriate combination of LoRA modules to generate the best answer based on user queries.](https://github.com/uukuguy/multi_loras): Load multiple LoRA modules simultaneously and auto...
- [Mixtral Experts are initialized from Mistral 7b - Low Rank conversion possible? · Issue #4611 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/4611): We have evidence that Mixtral&#39;s Experts were i...
- [TinyLlama Pretraining Report](https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-04-23-38-15---Vmlldzo1MzA4MzIw?accessToken=5eu2sndit2mo6eqls8h38sklcgfwt660ek1f2czlgtqjv2c6tida47qm1oty8ik9): See  https://whimsical-aphid-86d.notion.site/Relea...


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (31 messages🔥): 
        
- **Amazon Titan Text Express and Lite**: `@spaceman777` shared a link about Amazon's new large language models, **Titan Text Express and Titan Text Lite** and sought anyone's experiences or benchmarks for these models. He also noted that Amazon doesn't make a hype about their AI-related releases and implies that they backdate their releases.([Amazon Bedrock link](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/))
- **Improvement Strategy for Finetuning Models**: `@max_paperclips` suggested a potentially novel approach for improving model performance - finetuning a model on a bad dataset and subtracting the delta from the base model to delete the bad paths, followed by applying a good delta. Initial responses from `@teknium` and `@giftedgummybee` seemed unsure about the potential efficacy of this plan, with `@gifedgummybee` suggesting a similar principle in the form of a reversible LoRA (Learning Rate Annealing).
- **List of ChatGPT Failures**: `@max_paperclips` inquired about the existence of a list of ChatGPT failures, to which `@giftedgummybee` replied with a negative and suggest using Llama, while `@tokenbender` suggested that the task was too broad.
- **Improvements of English trained LLMs on Czech**: `@hynek.kydlicek` sought advice on improving performance of English trained LLMs on Czech language, suggesting two specific strategies and `@teknium` confirmed that someone (`@282315082749444097`) had tried this before. 
- **Training LLM with HF Auto Train**: `@agcobra1` wanted to know if DeciLM-7B was the best base model to use with Hugging Face's Auto Train feature, or if Mistral is a better option.

**Links mentioned**:

[Amazon Titan Text models—Express and Lite—now generally available in Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- An ongoing discussion about the utility of **smaller models** versus more complex architectures, including the challenges of fine-tuning and hybrid approaches for designing enterprise solutions. Notable discussions included the usage of LLM agents and model customization. [GitHub link to microchain example](https://github.com/TanGentleman/microchain)
- Discussions about **fine-tuning** methods and tutorials, including the sharing of a Tamil tuned model and a Mistral 7B Instruct fine-tuning guide. Notable advice involved substituting datasets for specific language tasks and using PEFT tutorials for those with limited VRAM.
    - [General fine-tuning tutorial](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model)
    - [PEFT Tutorial](https://huggingface.co/blog/peft)
- In the showcase channel, notable topics included the **chaining of LLM outputs** with defined function calls, feedback on **Mistral Instruct 7B v0.2 Q8**, and discussions about app architecture checking methods for Apple Silicon Macs. [Hugging Face link to Hermes model](https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF)
- Interesting dialogues in the random channel involved **tokenization of Chinese characters**, community discussions on AGI's first question, an open letter to OpenAI by **VERSES** calling for a new route to AGI, and debates on the implications of VERSES' approach.
    - [VERSES' blog post](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough)
- Insights from the la-plateforme channel revolved around issues with Mistral-Medium for **DPO dataset creation**, instruction-following discrepancies, structuring model outputs with **GPT-4 32k 0613**, and debates about the effects of **JSON instruction** on AI reasoning capabilities. Discussions also point towards synthetic dataset generation.
    - [Prompt engineering reference](https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Engineering.md)

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (39 messages🔥): 
        
- **Discussion on the Utility of Smaller Models**: `@theledgerluminary` questioned the value of using a complex architecture implementation with smaller models like the Mistral 7B, especially if the end-goal is to create an enterprise solution. This initiated a lively discussion and `@.tanuj.` argued for the benefits of being able to chain steps to solve complex problems, even offline, and using different models for various tasks.

- **Fine-Tuning Small Models vs Using Base Models for Tasks**: `@theledgerluminary` suggested that fine-tuning a community of specific small models, including one for orchestration, could yield great results, but using a base model for large tasks seemed less adequate. `@.tanuj.` countered, stating that the act of fine-tuning models may be more challenging than creating a reasoning "agent" that utilizes the Locally Linear Model (LLM) queries to solve tasks.

- **Hybrid Approach for Enterprise Solution Design**: The idea of taking a hybrid approach to design was proposed by `@superseethat`. The approach includes developing an "agent swarm architecture" with specialization in mind and then fine-tuning one specialization at a time.

- **Views on LLM Agents**: User comments varied on the utility of LLM agents. `@jessicant.` brought up the point that LLM fine-tuning could potentially improve program reliability, especially for tasks requiring multi-turn conversations. However, `@sublimatorniq` expressed doubts about the feasibility of GPT-4 agents beyond toy applications.

- **Customization of the Agent Framework**: `@.tanuj.` discussed the benefits of customizing the agent framework to be robust to any type of model, allowing for consistent chaining of requests on any model that respects a contract. This user also provided an [example](https://github.com/TanGentleman/microchain) of function calling-based LLM agents. The limitations of Transparent, Freeform, and English Instructions were also discussed, showing preference for more manually fine-tuned control.


**Links mentioned**:

[GitHub - TanGentleman/microchain: function calling-based LLM agents](https://github.com/TanGentleman/microchain): function calling-based LLM agents. Contribute to T...


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
- **Tamil Tuned Model**: User `@colmevans` shared a [model tuned for Tamil](https://huggingface.co/abhinand/tamil-llama-7b-instruct-v0.1), though with no guarantee on its quality.
- **Mistral 7B Instruct Fine-tuning Guide**: User `@colmevans` also provided a [general fine-tuning tutorial](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model) for the Mistral 7B Instruct model advocating its benefits and usability for various tasks, including coding.
- **Tamil Dataset for Fine-tuning**: For individuals intending to use this method for Tamil language tasks, he suggested simply substituting the dataset outlined in the tutorial for a Tamil dataset.
- **PEFT Tutorial**: User `@colmevans` recommended the [PEFT tutorial](https://huggingface.co/blog/peft), especially for those with limited VRAM. This tutorial covers parameter efficient fine-tuning of billion-scale models on low-resource hardware.


**Links mentioned**:

- [A Beginner&#x27;s Guide to Fine-Tuning Mistral 7B Instruct Model](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model): Fine-tuning a state-of-the-art language model like...
- [Parameter-Efficient Fine-Tuning using 🤗 PEFT](https://huggingface.co/blog/peft)


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (22 messages🔥): 
        
- **Discussion on Chaining LLM outputs with Defined Function Call**: User `@.tanuj.` proposed the idea of an LLM that can chain function calls in a step-by-step manner, providing detailed outputs. The discussion involved conceptualizing a "ResearchGPT" model that had features/functions such as GoogleSearch, EvaluateSources, CreateEssay, DeployWebsite, etc. `@poltronsuperstar` acknowledged its potential real-life applications.
- **Using "Instruct" over "Base" Model for Querying**: `@.gue22` gave feedback that **Mistral Instruct 7B v0.2 Q8** yielded better answers to the user's queries over its base model. `@.gue22` also shared a detailed way to determine if an app is written for x86 or ARM architecture on Apple Silicon Macs, which was generated by the instructed model. `@.tanuj.` suggested filling more of the instruct model's 32K window and providing examples for better results.
- **Recommendations on Other Models**: `@fayiron` advised `@.gue22` to try **Mixtral**, **Qwen**, or a **Yi finetune** (e.g., nous-hermes 2 yi 34b) due to their setup. After this suggestion, `.gue22` started a download for the Nous Hermes 2 Yi 34B model from Hugging Face for further evaluation.
- **Discussion on App Architecture Checking Methods for Apple Silicon Macs**: `@.tanuj.` mentioned a quicker way to check whether an application is built for Apple Silicon - by looking at the processes running in the Activity Monitor and checking for a tag saying "Apple".

**Links mentioned**:

[TheBloke/Nous-Hermes-2-Yi-34B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF)


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (13 messages🔥): 
        
- **Tokenization of Chinese Characters**: `@poltronsuperstar` remarked that Chinese characters often use two tokens due to Unicode encoding. Alternatively, `@.tanuj.` suggested using the `MistralAI` library in Python which includes token usage in the response object, or direct messaging for assistance with tokenizing Chinese characters.

- **Community Discussion - AGI's First Query**: `@poltronsuperstar` initiated a discussion asking other members what they would first ask an AGI. Responses varied, with `@sublimatorniq`'s question addressing AI consciousness and `@kdawgdfw` suggesting a leisurely topic: “So, what do you do for fun? Any hobbies?”

- **Open Letter to OpenAI from VERSES** : `@poltronsuperstar` shared a [blog post](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough) by VERSES. In an open letter to OpenAI, VERSES appealed to assist their path to AGI development, indicating concerns over the current mainstream path relying on deep learning and large language models.

- **Implications of VERSES' Approach**: Reactions to the blog post were mixed. `@daain` commented that the idea sounded intuitively good, but its efficient implementation is yet to be seen. They also pointed out the invocation of an OpenAI clause about assisting competing AGI developments as a clever PR move, and shared a [link](https://www.wired.com/story/karl-friston-free-energy-principle-artificial-intelligence/) to similar ideas in the past. In response, `@poltronsuperstar` mentioned that without a demo, such claims don't hold much value.


**Links mentioned**:

[The Science and Standards Behind the Breakthrough](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough): Letter from the CEO


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (12 messages🔥): 
        
- **Issues with Mistral-Medium for DPO dataset creation**: `@jaredquek` reported that while generating a DPO dataset, he found **Mistral-Medium** kept providing unnecessary explanations for inferior responses, which contradicts its intent of omitting such explanations. [Prompt engineering](https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Engineering.md) attempts to correct this issue were not successful.
- **Poor Instruction Following**: `@alimsss` hinted that the model's competency in following instructions is underwhelming, though a few shots can partially fix this behavior.
- **Attempt to Structure Model Outputs**: `@casper_ai` discussed a technique of generating a specific output structure from models, which can later be parsed with regex. He also suggested that **GPT-4 32k 0613** is efficient in producing such structured outputs.
- **Effects of JSON instruction on AI Reasoning Capabilities**: `@jaredquek` and `@casper_ai` had a discussion about whether instructing models in JSON format limits their reasoning capabilities. `@casper_ai` argued that using JSON may limit the models considering JSON might comprise only a small portion of their pretraining data. 
- **Synthetic Dataset Generation**: `.superintendent` is considering generating a synthetic dataset and looking for a time with low demand to avoid worsening the current high traffic.


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **DeepSpeed ZeRO3 Usage with LoRA (PEFT)**: `@galcoh.` queried about DeepSpeed ZeRO3's compatibility with LoRA (PEFT), highlighting issues with an optimizer during use with Accelerator.
- **Embeddings for Unsplash-25k-Photos-Embeddings.pkl**: The user `@nagaraj4896` requested details about image embeddings of `unsplash-25k-photos-embeddings.pkl`.
- **HuggingFace Website Registration Error 418**: Persistent registration and login errors reported by `@xratox` and `@muhammad.shakeel`. `@vipitis` suggested emailing HuggingFace for resolution.
- **Explanations on Multi-expert Long Language Models (LLM)**: `@typoilu` asked for resources on multi-expert LLMs, and `__nord` provided a Google Research Blog post, [Mixture-of-Experts with Expert Choice Routing](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1).
- **Inference Endpoint Creation Issues**: `@dragonburp` reported difficulties in creating an inference endpoint and requested help.
- **Personal Implementation with CUDA Kernels**: `@gag123` asked if `@neuralink` undertook all the implementation themselves, with the exception of CUDA kernels, to which `@neuralink` confirmed and mentioned the ongoing progress.
- **Sharing HuggingFace AI Community Projects**: A variety of user-created projects were shared, including `@vashi2396`'s [work-in-progress code](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA), freecs's [ArtificialThinkerSet](https://huggingface.co/freecs/ArtificialThinker-Phi2), and `@andysingal`'s [amazon-sentiment-dataset](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset).
- **Operation of the Reading-Group Channel**: New member `@.lzack` inquired about the channel's conduct, whether there were specific readings or if the purpose was sharing readings of interest.
- **Discussion Placement in Diffusion-Discussions Channel**: `@sayakpaul` reinforced that questions about Mixtral should not be posted in channels dedicated to diffusion models.
- **Pose Estimation Model and Gradient Calculations**: In the computer-vision channel, `@_dashwood_` expressed aspirations for using a pose estimation model to derive key points in a specific JSON format, and `@lokesh1826` required insights on how to extract gradients from a complete picture rather than individual patches during image classification, and also how to collect the output and gradients from a specific layer of a Vision Transformer (ViT) model.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (59 messages🔥🔥): 
        
- **Loading Large Models and Applying DeepSpeed ZeRO3**: `@galcoh.` queried whether it is possible to enable DeepSpeed ZeRO3 with LoRA (PEFT) and indicated having issues with the optimizer when using Accelerator (`get_peft_model` is failing).
- **Image Embedding in Unsplash-25k-Photos-Embeddings.pkl**: `@nagaraj4896` sought information regarding image embedding of `unsplash-25k-photos-embeddings.pkl`. No response was given within the transcript.
- **Issues with HuggingFace Website Registration**: Several users (`@xratox`, `@muhammad.shakeel`) reported a recurring "Error 418" when trying to register or log in on the HuggingFace website, and requested assistance from various members. The issue remained unresolved with `@vipitis` suggesting to email HuggingFace and wait for a response.
- **Discussion on Multi-expert Long Language Models (LLM)**: `@typoilu` asked for explanations or documentation on how multi-expert LLMs work, with `__nord` providing a link to a Google Research Blog post detailing the Mixture-of-experts model.
- **Inference Endpoint Creation Issues**: `@dragonburp` expressed having difficulties creating an inference endpoint, stating an error found in log files. Assistance was sought but no solution was provided within the transcript.


**Links mentioned**:

- [AnimateDiff - a Hugging Face Space by guoyww](https://huggingface.co/spaces/guoyww/AnimateDiff)
- [rabbit — Waitlist](https://www.rabbit.tech/waitlist?utm_source=discord&utm_medium=discord&utm_campaign=waitlist): Jan 09 at 10am PT
- [Textual Inversion](https://huggingface.co/docs/diffusers/training/text_inversion)
- [Mixture-of-Experts with Expert Choice Routing &#8211; Google Research Blog](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1)
- [9 days until the pixels reveal.](https://www.youtube.com/watch?v=mw8O-nS75hM): Join the waitlist to see the launch of rabbit’s fi...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **Implementation Discussion**: User `@gag123` asked if the user `@neuralink` implemented everything from scratch. To this, `@neuralink` confirmed that they **implemented everything themselves**, except for the **CUDA kernels**.
- `@neuralink` also mentioned that their work is **still in progress**.


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (8 messages🔥): 
        
- **HuggingFace AI community projects**:
    - `@vashi2396` shared a [work-in-progress code](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA) on Google Colab and invited volunteers to try out and complete it. He also provided a [LinkedIn demo](https://www.linkedin.com/posts/vashisth-malik_googleai-gemini-aichatbots-activity-7143976408422187008-huTV) of the code.
    - `@gr.freecs.org` introduced freecs's [ArtificialThinkerSet](https://huggingface.co/freecs/ArtificialThinker-Phi2) which emphasizes on 'Reasoning' for fine-tuning AI Language Models. He invited users to test the model and encouraged feedback. The model is based on the paper [Reasoning Is All You Need](https://freecs.org/blog/Reasoning_Is_All_You_Need).
    - `@andysingal` added a new [amazon-sentiment-dataset](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset) on HuggingFace datasets and shared the link in this channel.


**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)
- [freecs/ArtificialThinker-Phi2 · Hugging Face](https://huggingface.co/freecs/ArtificialThinker-Phi2)
- [Reasoning Is All You Need](https://freecs.org/blog/Reasoning_Is_All_You_Need)
- [Andyrasika/amazon-sentiment-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset)


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 messages): 
        
- **Introduction and Clarification**: New member `@.lzack` joined the `#reading-group` channel and inquired about how the channel operates. They asked if there are assignments for specific books/papers to read or if the purpose is to share interesting findings from miscellaneous reading materials.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- User `@sayakpaul` reminded everyone that **Mixtral related questions** should not be discussed in channels dedicated to diffusion models. No links or further details were provided.
- User `@chokipro` shared a Discord server link. The link and its context or relevance wasn't clear.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **Pose Estimation Model for Key Point Extraction**: `@_dashwood_` sought advice on the usage of a pose estimation model to get key points in a specific JSON format. He mentioned attempts with openpose but was unable to find a suitable solution for 2D images and Python code implementation.
- **Gradient Calculations for Image Classification**: `@lokesh1826` inquired about obtaining the gradients of an image during backpropagation using the HuggingFace transformers package. He presented his code and expressed a concern about receiving the gradients of patches instead of the complete image.
- **Extract Outputs and Gradients from nth layer of Model**: `@lokesh1826` requested help in extracting the output and gradient from a particular layer of a Vision Transformer (ViT) model, specifically, wanting to obtain the query, key and value vectors of each encoder layer in ViT.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **Mixtral Discussion Placement**: `@sayakpaul` clarified that questions related to **Mixtral** should not be posted in channels devoted to diffusion models.
- **Unspecified Discord Link**: `@chokipro` posted a [link](https://discord.com/channels/879548962464493619/1190992567366602752) without any context or description.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- '['AutoGemini' tool](https://huggingface.co/datasets/seungduk/autogemini) presented by user `@seungduk` enables collaborative editing of text datasets via the Gemini Pro API. Future ambitions from users around the training of the **TinyLlama model** to develop personal assistants lead to bouts of excitement and curiosity.
- Discussion around training **Yayi 30b** with FFT and the encountered issues. Suggestions for offloading were made by `@nruaif` and `@nanobitz`. Clarifications regarding the **DPO** support in Axolotl and its related documentation issues were also mentioned.
- Multiple queries relating to **ChatML input transformation**, **LoRA training with Mixtral**, **Batch size and learning rate**, **Qlora DSZ 3 compatibility**, and **memory requirements for DPO** were addressed within the community.
- Dataset discussions where user `@zeroshotkevin` requested for a Q/A dataset for a "hello, world" fine-tuning experiment. It was recommended to use the dataset available in the example file and the [mhenrichsen/alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) dataset.
- Debate on the comparison between **DPO** vs **PPO** stirred by user `@swyxio`. It was opined that DPO models generally outperform PPO models in various benchmarks but other approaches like **OpenChat** also perform well.
- Discussions in the `#shearedmistral` channel revolving around aversion to GPT-generated data to bypass OpenAI's terms, filtering datasets based on language using resources like [fastText](https://fasttext.cc/), considering larger context length in the samples, and introduction of numerous datasets including [peS2o](https://huggingface.co/datasets/allenai/peS2o), [yayi2_pretrain_data](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data), [MathPile](https://huggingface.co/datasets/GAIR/MathPile), and [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) for use in future studies.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (5 messages): 
        
- **Dataset Transformation Tool**: User `@seungduk` shared a link to a tool called **AutoGemini**, designed to facilitate the collaborative editing of text datasets via the Gemini Pro API. This tool allows for community contributions to project datasets, providing features like query rate management, job reservation expiry, dataset flexibility, and a community leaderboard. Tool is accessible at the [Hugging Face repository](https://huggingface.co/datasets/seungduk/autogemini).
- **TinyLlama Model Discussion**: User `@le_mess` expressed excitement about the **TinyLlama model**, highlighting its ability to train 8 billion tokens in about 48 hours. Future plans include creating a personal assistant that can run on various platforms. This message sparked interest and further questions from users `@tank02.` and `@nanobitz`.

**Links mentioned**:

[seungduk/autogemini · Datasets at Hugging Face](https://huggingface.co/datasets/seungduk/autogemini)


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (20 messages🔥): 
        
- **Attempting to run Yayi 30b with FFT**: `@le_mess` mentioned they were unable to fit **Yayi 30b** on 4x A100 40gb with zero3 and is looking for a 2 or 4x A100 80gb solution.
- **Suggestions for Offloading**: Both `@nruaif` and `@nanobitz` suggested trying offloading, with `@nanobitz` providing a specific code snippet showing how to offload to the CPU.
- **Failure with CPU Offloading**: Upon implementing the CPU offloading feature in the configuration, `@le_mess` encountered a failure, as evidenced by a posted traceback.
- **Configuration Adjustments**: `@tank02` inquired if `@le_mess` made any configuration alterations other than adjusting the model and datasets used. `@sumo43` responded no changes were made.
- **Support for DPO**: `@mrfakename_` inquired about **DPO** support in Axolotl, with `@nanobitz` confirming an open branch on GitHub accommodating this feature. The documentation is reportedly a work in progress.


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (15 messages🔥): 
        
- **ChatML Input Transformation**: User `@caseus_` clarified that the existing transforms in the DPO dataset essentially convert existing prompt into ChatML inputs and reformats the chosen/rejected responses to incorporate the end of sentence (eos) token. 
- **LoRA Training with Mixtral**: `@caseus_` asked if anyone has been able to train an 8-bit LoRA with Mixtral. `@nruaif` responded that it hits an Out Of Memory (OOM) error at 16k context, even on an impressive A100 80gb. The peak Virtual RAM (VRAM) usage at 2k context was reported to be 70gb.
- **Batch Size and Learning Rate**: User `@semantic_zone` asked about the relationship between batch size, learning rate and model size. They queried if batch size needs to be smaller for larger models strictly due to memory constraints, and sought a rule of thumb for adjusting learning rate relative to batch size.
- **Qlora DSZ 3 Compatibility**: `@tank02.` inquired if Qlora supports DSZ 3, to which `@le_mess` responded that they heard it should but didn't try it. Meanwhile, `@casper_ai` mentioned there are some issues with it.
- **Memory Requirements for DPO**: `@tank02.` asked about memory requirements for DPO, especially while using a 3b model with qlora on a 24gb card, which led to an OOM error. `@nanobitz` responded that the user needs to come into account the fact that the model is loaded twice, and suggested adjusting the optimization and batch size.


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **Request for Q/A Dataset**: In this channel, user `@zeroshotkevin` requested for a simple question/answer dataset to fine-tune a model similar to **Mistral 7B** with the aim of getting a discernible difference from the original model. This is targeted at performing a fine-tuning "hello, world" experiment with **Axolotl**.
- **Dataset Recommendation** : User `@nruaif` recommended the utilization of the dataset available in the example file and also shared the [link](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) to **mhenrichsen/alpaca_2k_test** dataset hosted on HuggingFace containing dialogues such as giving tips for staying healthy.

**Links mentioned**:

[mhenrichsen/alpaca_2k_test · Datasets at Hugging Face](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test)


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (3 messages): 
        
- **DPO vs PPO**: User `@swyxio` asked if there was a consensus about **DPO** being comparable or better than **PPO**. `@_jp1_` expressed that there may not be a consensus, but mentioned that **DPO models** perform well and top various benchmarks. They compared this with PPO models, which according to them, were never competitive. However, they also highlighted the performance of other approaches like **OpenChat**.


### ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/) (1 messages): 
        
dangfutures: Ugh got deleted


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (23 messages🔥): 
        
- **Avoiding GPT-Generated Data**: `@dctanner` expressed a desire to avoid using GPT-generated data so as not to be burdened by OpenAI's terms during pretraining.
- **Continued Training on Mistral**: A discussion occurred among `@nruaif` and `@caseus_` regarding the transition to training on **Mixtral** after dealing with potential bugs, and expressing a need to focus on continued training with **Mistral**. They both agreed that losing experts during training is a concern since they are token-wise experts.
- **Data Filtering and Handling**: `@nruaif` and `@caseus_` discussed about the need to filter specific datasets based on language, especially removing non-English subset. `@nruaif` recommended using [fastText](https://fasttext.cc/), an open-source library for learning text representations and text classifiers, for filtering non-English content.
- **Consideration for Bigger Context Length**: `@caseus_` suggested a preference towards bigger context length in the samples. However, the final decision depends on affirmation from the team members.
- **Datasets Suggestions**: A couple of datasets were mentioned for consideration, including [peS2o](https://huggingface.co/datasets/allenai/pes2o), [yayi2_pretrain_data](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data), [MathPile](https://huggingface.co/datasets/GAIR/MathPile), and [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX), with `@xzuyn` mentioning that CulturaX includes 6.3 Trillion tokens across 167 languages.

**Links mentioned**:

- [fastText](https://fasttext.cc/): Library for efficient text classification and repr...
- [allenai/peS2o · Datasets at Hugging Face](https://huggingface.co/datasets/allenai/peS2o)
- [uonlp/CulturaX · Datasets at Hugging Face](https://huggingface.co/datasets/uonlp/CulturaX)
- [wenge-research/yayi2_pretrain_data · Datasets at Hugging Face](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data)


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Discussion on **LangChain's functionality,** with examples of structuring output, passing multiple inputs to a prompt, and a shared [GitHub repository](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py) by `@rajib2189` for additional practical applications.
- There was interest in LangChain's compatibility with other platforms, with `@sarrah_1` inquiring about integrating LangChain with a Laravel Project and availability of a specific PHP library, and `@evolutionstepper` concerning its utility in running everything asynchronously in FastAPI. A possible asynchronous implementation via tokio framework was also suggested.
- Clarification requested on the difference between OpenAI Functions and OpenAI Tools Agents, `@toasted_shibe` explained that Tools Agent allows for parallel function calling providing a link to the [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call).
- `@alexk1919` asked about the relevance of LangChain for creating a sequence of prompts that integrate results from the previous prompt.
- `cheerful_moose_30860` encountered an error while importing sentence-transformers.

**LangChain AI Channel Summaries**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
cheerful_moose_30860: Error importing sentence-transformers


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (33 messages🔥): 
        
- **LangChain Output Structure**: `@seththunder` pointed out that the LangChain output parser can be used to format output in a specific structure.
- **LangChain Examples**: `@rajib2189` provided an example of how to pass multiple inputs to a prompt in LangChain and shared a [GitHub link](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py) for the same.
- **LangChain Integration with Laravel**: `@sarrah_1` inquired about the possibility of integrating LangChain with a Laravel project and if there's a specific PHP library available for this.
- **Asynchronous Implementation in FastAPI with LangChain**: `@evolutionstepper` expressed concerns about whether LangChain can handle running everything asynchronously in FastAPI. `@quantumqueenxox` confirmed it's possible and mentioned they have code to make processes asynchronous. `@evolutionstepper` also showed interest in a langchain built on top of the tokio framework.
- **Difference between OpenAI Functions and OpenAI Tools Agents**: `@keenborder` asked for clarification on the difference between OpenAI Functions and OpenAI Tools Agents, where `@toasted_shibe` explained that the tools agent calls the new tools API endpoint, allowing for parallel function calling and referred to the [OpenAI docs](https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call) for further info.
- **Use of LangChain for a sequence of prompts**: `@alexk1919` questioned whether LangChain is the right tool for creating a sequence of prompts that leverage the results from the previous prompt.

**Links mentioned**:

[langchain_examples/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py at main · rajib76/langchain_examples](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py): This repo consists of examples to use langchain. C...


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- Discussed the **unavailability of the LAION dataset** for research due to potential legal issues. Users expressed concerns and provided alternative solutions like creating own datasets from [CommonCrawl](https://github.com/rom1504/cc2dataset). They also suggested a thorough cleanup of existing dataset, including the removal of invalid content and broken links.
    - "*The dataset is currently under review to remove all NSFW content, especially child porn-related content...*"
- Debated on dataset modifications, handling discarded content, and the necessity to rebase after PR. The conversation continued on the difficulties of dealing with those who have an old copy of a dataset and the challenge to keep it clean and up-to-date.
    - "*...this could rebase after making the PR. But it would not be effective for users who already have the old dataset.*"
- Noted a **delays in the DREAM project** due to computer issues.
- Discussed the potential **leak of GPT-4 details** shared in a [blog post](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a). However, users expressed skepticism due to lack of solid evidence supporting the leak.
    - "*...there's no solid evidence supporting the accuracy of the blog post or any other speculation about GPT-4.*"
- Announced the release of a new model called "**Anytext Text ControlNet**" and shared a [link](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary) to its summary.
- Positive appraisal for [Modelscope](https://modelscope.cn/) shared by the user `@puffy310`.
    - "*...[Modelscope] it's "kinda good", although not quite as good as Hugging Face.*"
- Provided in-depth explanation on the structural differences between **ChatGPT**, **SD**, and **SDXL** models in terms of their architecture, inputting output and training methods.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (25 messages🔥): 
        
- **Publishing of LAION Dataset**: User `@ggez` asked about the expected time when the LAION dataset will be published again. `@chad_in_the_house` responded that the dataset is currently under review to remove all NSFW content, especially child porn-related content, due to legal concerns.
- **Alternative Dataset Source**: `@thejonasbrothers` suggested creating one's own dataset from [CommonCrawl data](https://github.com/rom1504/cc2dataset). They discussed the issues regarding the current LAION dataset and predicted the actions LAION might need to take, such as a complete rebuild of the dataset from more recent CommonCrawl data while ensuring the absence of objectionable materials.
- **Dataset Modification**: A conversation around modifying the dataset in response to changing content legality ensued. `@progamergov` suggested that LAION could rebase after making the PR, to which `@nodja` countered that this would not be effective for users who already have the old dataset. They further discussed the issue of dataset owners needing to filter their old copies.
- **Dataset Cleanup and Link Rot**: `@nodja` also suggested a cleanup of the dataset, including the removal of 404 links and unmatched image hashes, assuming a ~10% loss of the dataset by now. `@progamergov` agreed, mentioning the significant link rot already experienced in the LAION 5B dataset.
- **DREAM Project Delay**: Finally, `@xylthixlm` noted a delay in their work on the DREAM project due to computer issues, projecting the pause to last around a week.


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (7 messages): 
        
- **GPT-4 Details Leaked**: `@vrus0188` shared a [blog post](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a) supposedly revealing details about GPT-4, including its model architecture, training infrastructure, parameter count, training data composition, and more. The source of the leak is Yam Peleg, who shared the details, which were initially placed behind a paywall by Semi-Analysis, on Twitter for free.
- `@metal63` expressed skepticism, noting that there's no solid evidence supporting the accuracy of the blog post or any other speculation about GPT-4.
- **Anytext Text ControlNet Release**: `@thejonasbrothers` announced the release of a new model called "Anytext Text ControlNet" and shared a [link](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary) to its summary. 
- **Modelscope Review**: `@puffy310` commented positively about [Modelscope](https://modelscope.cn/), stating that it's "kinda good", although not quite as good as Hugging Face.

**Links mentioned**:

- [GPT4- All Details Leaked](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a): The details about the best LLM model trainning and...
- [AnyText多语言视觉文字生成与编辑模型](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary)


### ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/) (1 messages): 
        
- **Discussion on Chatbot Architecture**: User `@JH` provided an in-depth explanation of the architectural differences between **ChatGPT**, **SD**, and **SDXL** models. According to them, ChatGPT primarily uses a casual decoding transformer that performs inference based on next token prediction tasks. On the other end, SD models primarily use a convolutional U-Net architecture, inputting output embeddings from Clip L for **SD v1** and from Clip L + openClip G for **SDXL**. The U-Net architecture incorporates cross attention layers and self attention layers, trained via variational lower bound loss and a noise prediction loss. Lastly, `@JH` deems it is reasonable to expect these different architectures to learn concepts differently due to their distinct objectives.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- Detailed discussion on the use and effect of the **GEGLU** activation function in transformer models, with various strategies suggested to reduce parameter overhead. A coding example from the [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95) implementation was shared for tangible reference.
- New member `@mouldysoul` inquired about resources to improve their understanding of flow-based models and their potential relationship with optimal transport theory.
- Diverse discussions in the research channel, with topics spanning from **PPO-based adapter training** to the **novel modifications to transformer architecture**, with links to [trlX paper](https://aclanthology.org/2023.emnlp-main.530/) and [abstract of a research paper](https://arxiv.org/abs/2311.02265). Discussed insights on **ELC-BERT architecture** and its significance in model training.
- In the **interpretability** channel, discussions revolved around approaches to automated interpretability, edge attribution patching, and the current trend toward integrating high-level causal variables into subspace discovery. A [research paper](https://arxiv.org/abs/2310.10348) on edge attribution patching was shared. Interest in **MoE Models and their interpretability** was expressed, and a link to [Mixtral's repository](https://github.com/dvmazur/mixtral-offloading) was shared as a means to run the model on consumer-grade platforms.
- A reminder by `catboy_slim_` regarding the deprecation cycle of Python 3.8 on the `gpt-neox-dev` platform.

**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (8 messages🔥): 
        
- **Discussion on the Use of GEGLU**: `@sentialx` asked about the advantage of using the **GEGLU** activation function in transformer models, pointing out that it adds more parameters and offers negligible performance improvement. `@ad8e` claimed that the use of GEGLU shrinks dimensions, thus keeping parameter counts the same. In response, `@sentialx` mentioned that when using GEGLU, the transformer's FFN intermediate linear layer requires a two-fold increase in output dimensionality.
- **Decreasing Parameter Overhead of GEGLU Models**: `@bob80333` explained that it's a common strategy to reduce the intermediate size in models employing **GEGLU** (or its variants) so that they maintain parameter equivalence, citing *llama's use of an 8/3 multiplier* instead of the standard 4x multiplier in its FFN layer to offset the use of swiglu.
- **Clarification on Model Size with Respect to GEGLU**: `@maxmatical` clarified that the hidden size for the transformer's FFN layer would be `16/3` when applying `swiglu` after implementing `llama's strategy of an 8/3 multiplier`. They provided the [NVIDIA/Megatron-LM implementation](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95) as a code reference.
- **Introduction of New Member `@mouldysoul`**: `@mouldysoul`, a professional involved in deploying AI models and an aspiring machine learning researcher, introduced themselves to the community.
- **Inquiry into Flow-Based Models**: `@mouldysoul` requested guidance and resources to better understand flow-based models, emphasizing their interest in understanding the models' bijective mappings, faster sampling capabilities than diffusion models, better interpolation, and their potential relation with optimal transport theory.

**Links mentioned**:

[Megatron-LM/megatron/model/transformer.py at 2bc6cd307a11423928c675f741e79e03df23e721 · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95): Ongoing research training transformer models at sc...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (13 messages🔥): 
        
- **PEFT Techniques and Adapters**: `@cormerod` queried if adapters can be trained using PPO in consideration with PEFT techniques for case-by-case output improvements in a 7b parameter model. `@stellaathena` affirmed it, and `@maxmatical` mentioned such a feature can be utilized in trl, deepspeed chat, and other libraries.
- **trlX Paper Reference**: `@stellaathena` drew attention to the [trlX paper](https://aclanthology.org/2023.emnlp-main.530/) that speaks about PEFT techniques and other relative features like layer freezing. [GitHub repo of the trlX project](https://github.com/CarperAI/trlx).
- **Discussion on Modification of Transformer Architecture**: `@digthatdata` shared the [abstract of a research paper](https://arxiv.org/abs/2311.02265) that proposes a novel transformer architecture modification for efficient pretraining of language models. `@kharr.xyz` remarked that such a modification is favourable for models smaller than 100M params and insignificant as the scale increases. `@ad8e` dismissed the BabyLM competition referred to in the paper as not being very competitive.
- **Insights on ELC-BERT Architecture**: `@ad8e` provided insights on the importance of ELC-BERT architecture, considering the last layer attending to the first one. `@kharr.xyz` debated that these patterns change over the course of training and advised not to put too much weight on these figures. Following the discussion, `@ad8e` inferred that the last layer attending the first layer might turn from a tiny task to a larger one with more training data. `@kharr.xyz` confirmed this.
- **Robustness to Noise**: `@eron_gj` shared experience on the robustness of architecture to noise, stating that even rotating the k/v/a vectors on average up to 30 degrees for half of the layers doesn't hamper the coherence of the outputs.

**Links mentioned**:

[Not all layers are equally as important: Every Layer Counts BERT](https://arxiv.org/abs/2311.02265): This paper introduces a novel modification of the ...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (6 messages): 
        
- **Automated Interpretability Work Post ACDC**: `@dashiell_s` inquired about notable advancements in automated interpretability after ACDC and noted the existence of an ACDC repository. `@t_kwa` identified edge attribution patching and token position analysis via Joseph Miller's implementation of edge subnetwork probing as progress in the field. They also mentioned ongoing work to integrate high-level causal variables into subspace discovery. 
- **ACDC Repository Usability**: In terms of the ACDC repository, `@t_kwa` pointed out that although it's not straightforward to utilize due to the need for script conversion from FAR AI's Kubernetes setup, the demo notebook can still be run smoothly.
- **Edge Attribution Patching Efficiency**: `@neelnanda` referenced a paper supervised by `<@349859906570027010>` that demonstrates the superiority of edge attribution patching over ACDC in terms of speed and circuit output retrieval. The paper can be accessed [here](https://arxiv.org/abs/2310.10348).
- **Interest in MoE Models plus Interpretability**:  `@sk5544` expressed curiosity about work being done on the intersection of interpreterability and Mixture of Experts (MoE) models. They noted the high compute intensity of even small MoE models as a hindrance for academic experimentation.
- **Running MoE Models on Consumer-grade Platforms**: In response, `@stellaathena` suggested running Mixtral, an MoE model, on Google Collab, and provided a link to its [repository](https://github.com/dvmazur/mixtral-offloading).

**Links mentioned**:

- [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348): Automated interpretability research has recently a...
- [GitHub - dvmazur/mixtral-offloading: Run Mixtral-8x7B models in Colab or consumer desktops](https://github.com/dvmazur/mixtral-offloading): Run Mixtral-8x7B models in Colab or consumer deskt...


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (1 messages): 
        
catboy_slim_: python 3.8 gets deprecated next year or this year depending on your current time zone


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Benchmark for Context Retrieval**: `@rasdani` expressed an interest in creating a **benchmark for context retrieval**, based on deepset/germanquad. The plan is to select 60 question/context pairs from the test set, pairing half with irrelevant contexts, and using 0 and 1 as ground truth for cosine similarity. The aim of the benchmark is to compare different embedding models and calculate pairwise correlations.
- **Dot Product vs Cosine Similarity**: `@philipmay` advised that the **dot product is more effective than cosine similarity** when using semantic embeddings for questions and passages. This tip was originally provided to them by Nils Reimers, who they consider an expert in embeddings.
- **Metrics for Retrieval Systems**: In response to a conversation about retrieval system metrics, `@philipmay` stated that MRR@10 is often used, while `@hammadkhan` noted that the MTEB leaderboard uses NDCG@10, which assesses the quality of retrieval based on relevance and position within the top 10 items.  
- **Data Sets for Multiple Positive Contexts**: `@rasdani` asked for recommendations of contextual QA datasets with multiple positive contexts in German, as they are planning to use MRR@10 for their benchmark due to having only one positive reference context per question in germanquad.
- **New Year Greetings**: `@bjoernp` and `@thewindmom` greeted the members of the discord server and expressed their anticipation for future developments.
        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Organising an Event in Milano**: User `@alexio.c` proposed the idea of organizing an event in Milano, Italy. `@fanahova` responded positively, suggesting to put the word out in other local groups.
- **AI Platforms Discussion**: `@aristokratic.eth` sought suggestions for AI platforms. `@fanahova` recommended **Unstructured.io** as it has the most funding.
        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- Users in the guild exchange **New Year greetings** and well-wishes, fostering a sense of community and camaraderie.
- In celebration of the new year, a [New Year 2022 GIF](https://tenor.com/view/new-year-2022-gif-24334949) was shared by user `@cryptossssun` across both oo and oo2 channels, adding a festive and joyful tone to the discussions.
- Details about the shared GIF were also provided, noting a file size of **1303KB**, a duration of **1.200 sec**, and dimensions of **498x331**, indicating an attention to detail and possible relevance to discussions on digital media resolutions and formats.

**Alignment Lab AI Channel Summaries**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (4 messages): 
        
- **Happy New Year Wishes**: Users `@cryptossssun`, `@teknium`, and `@neverendingtoast` shared their **New Year greetings** and well-wishes to the community in Alignment Lab's oo Discord channel.
- **New Year Gif**: `@cryptossssun` also shared a [New Year Gif](https://tenor.com/view/new-year-2022-gif-24334949) to celebrate the start of 2022.

**Links mentioned**:

[New Year GIF - New Year 2022 - Discover &amp; Share GIFs](https://tenor.com/view/new-year-2022-gif-24334949): Click to view the GIF


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/) (2 messages): 
        
- User `@cryptossssun` shared a **[New Year 2022 GIF](https://tenor.com/view/new-year-2022-gif-24334949)**, wishing everyone a Happy New Year and success in their endeavors.
- The GIF details include a file size of **1303KB**, a duration of **1.200 sec**, and dimensions of **498x331**. The GIF was created on **1/1/2022**.

**Links mentioned**:

[New Year GIF - New Year 2022 - Discover &amp; Share GIFs](https://tenor.com/view/new-year-2022-gif-24334949): Click to view the GIF


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

Only 1 channel had activity, so no need to summarize...

teknium: lol it's exactly what he asked for 😄
        
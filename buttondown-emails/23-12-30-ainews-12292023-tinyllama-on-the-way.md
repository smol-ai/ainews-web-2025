---
id: 8607f821-a9e1-4ebe-8456-8b3f9bc0f439
title: '12/29/2023: TinyLlama on the way'
date: '2023-12-30T11:06:56.557602Z'
original_slug: ainews-12292023-tinyllama-on-the-way
description: >-
  The **Nous/Axolotl community** is pretraining a **1.1B model on 3 trillion
  tokens**, showing promising results on **HellaSwag** for a small 1B model. The
  **LM Studio Discord** discussions focus on **GPU-related issues**, **Discord
  bot integration with OpenAI API**, **model compatibility and use-cases**, and
  **hardware limitations** affecting large context sizes. Community members also
  discuss **server hosting for embeddings and LLM models**, **embedding API
  integration**, and the **Databerry Project** for custom LLM agents. Additional
  topics include fixes for a **gibberish problem** in beta releases and
  troubleshooting **Autogen installation and operation**.
companies:
  - nous
  - axolotl
  - openai
  - lmstudio
  - hugging-face
models:
  - tinyllama-1.1b
topics:
  - gpu-optimization
  - discord-bots
  - model-compatibility
  - embedding-models
  - server-hosting
  - context-windows
  - model-performance
  - autogen
  - fine-tuning
people:
  - pminev
  - dagbs
  - thelefthandofurza
  - fabguy
  - rofliex
  - vic49
  - gregly
---


<!-- buttondown-editor-mode: plaintext -->The Nous/Axolotl community is currently [pretraining a 1.1B model on 3 trillion tokens](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T). 59 HellaSwag very promising for a smol 1B model.

 ![image.png](https://assets.buttondown.email/images/3ec767c4-1a38-4f44-8973-a9acc1b31beb.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/e4e0cdd0-6098-4910-b5b3-f57a3c49b711.png?w=960&fit=max) 

[TOC] 

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- Extensive discussions were held on various **GPU-related problems**, ranging from model operational issues to compatibility concerns. For instance, `@pminev` required assistance with GPU-related issues and on configuring the model for other functions and `@dagbs` guided them towards the Inference Server in LM Studio.
- An ongoing conversation on incorporating a **Discord bot** with OpenAI API was witnessed. `@thelefthandofurza` shared a [Github](https://github.com/openai/gpt-discord-bot) link aiding users in tweaking the bot's existing code as per their needs.
- The community also interacted about specific **LM Studio use-cases and compatibility**, discussing character prompts in roleplaying contexts and integrating a GitHub wiki with the LLM assistant for more contextual responses. 
- In terms of **hardware**, the topic revolved around the limitations that users faced in utilizing various models due to GPU restrictions. Possible solutions for running models with large context sizes were also speculated, with `@fabguy` remarking, "*Large context sizes slow down processing and they eat up RAM/vRAM like crazy.*".
- Discussion on **server hosting** for embedding and LLM models was initiated by `@rofliex` with helpful input provided by `@vic49`. Also discussed was integrating with the embeddings API and the use of [Databerry Project](https://github.com/gmpetrov/databerry) for building custom LLM Agents. 
- Community members proposed updates on the **discord channels**, requesting set up of a dedicated category for model development, and advocated for a visible leaderboard model section with data from trusted external sources. They also expressed the need to exercise caution in accepting submissions due to polluted training data issues.
- User `@.gregly` shared a temp fix for a **gibberish problem** in the 0.2.10 (Windows) version in the beta releases discussion. 
- The **Autogen** topics revolved around installation troubles, understanding error messages, usage confusion about Docker, and Autogen’s operation nature. Clarity was provided by users who conducted detailed explanations about how Autogen functions under different conditions.

**LM Studio Channel Summaries**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (102 messages🔥🔥): 
        
- **GPU Related Issues and Queries**: `@pminev` faced issues with the model operation and received suggestions to check GPU-related issues from `@dagbs`. `@pminev` was also interested in configuring the model for other functions similar to OpenAI, and `@dagbs` pointed him towards the Inference Server in LM Studio.
- **Discord Bot Implementation Discussions**: Users `@Trip` and `@rewire` showed an interest in finding a Discord bot that works well with the OpenAI API. `@thelefthandofurza` shared a [Github link](https://github.com/openai/gpt-discord-bot) for a discord bot, noting that users may have to tweak the existing code.
- **LM Studio Use-Case and Compatibility Discussions**: `@olofp` suggested opening a specific channel for discussing use-cases of LM studio. `@vanthryn` asked for the best practices regarding character prompts in the context of using an LLM to roleplay. `@professorakram` sought advice for integrating a GitHub wiki with the LLM assistant for context in responses and `@dagbs` suggested using autogen.
- **System Compatibility Queries**: `@katanasoul91` and `@basedking` had issues related to their system compatibility with certain models. `@fabguy`, `@yagilb`, and `@dagbs` offered their advice and guidance.
- **Model Performance Queries**: `@jiha` and `@rocketraccoon6074` were looking for models that align with their specific hardware capabilities and requirements. Suggestions and guidance were offered by `@fabguy`, `@dagbs`, and others.

**Links mentioned**:

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)
- [GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio): Plugin that creates a ChromaDB vector database to ...
- [GitHub - openai/gpt-discord-bot: Example Discord bot written in Python that uses the completions API to have conversations with the `text-davinci-003` model, and the moderations API to filter the messages.](https://github.com/openai/gpt-discord-bot): Example Discord bot written in Python that uses th...


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (6 messages): 
        
- **Confusion on Model Training Purpose**: User `@dagbs` expressed confusion over the datasets used and the end-goal of a certain model. They stated, "*I'm so confused by the end-goal of the Model and what it was trained for.*"
- **Hardware Limitations**: User `@dagbs` also noted the model's size (`8x7b`) prevents them from running it due to hardware limitations.
- **Excitement Over Potential of SOLARC-MOE-10.7Bx4 Model**: `@jiha` shared their enthusiasm about the potential power of the untested `SOLARC-MOE-10.7Bx4` model, providing its link at the [coverage](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF). They also expressed a desire to see it tested but lamented not having the necessary hardware.
- **Speed and Memory Challenges with Large Context Sizes**: `@fabguy` warned about performance issues with large context sizes, stating "*Large context sizes slow down processing and they eat up RAM/vRAM like crazy.*". They suggested a RAG setup could be beneficial.
- **Questions on MoE Model Processing Speeds**: `@a1vx` raised a query about the processing speed of MoE models, seeking to understand how an expert FFN router from a `7b` model could be stacked eight times.

**Links mentioned**:

[TheBloke/SOLARC-MOE-10.7Bx4-GGUF · Hugging Face](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF)


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (9 messages🔥): 
        
- **Model Development Channel Request**: User `@dagbs` proposed setting up a specific channel category for model development, including subcategories such as general, pretraining, datasets, finetuning, and quantization, to foster collaboration within the LM Studio community.
- **Leader Board Model Section Suggestion**: `@pandora_box_open` recommended adding a leaderboard model section visible to all. The data for this could be fetched from external sources like HuggingFace and they linked to [OpenCompass](https://opencompass.org.cn/leaderboard-llm) as an example.
- User `@fabguy` affirmed the idea of a leaderboard but also cautioned that currently no submissions are being accepted due to issues with polluted training data.
- `@pandora_box_open` responded by suggesting the possibility of having a section for reviewers using LMstudio for rankings, which could serve as promotion for LM Studio while benefiting the community.

**Links mentioned**:

[OpenCompass](https://opencompass.org.cn/leaderboard-llm)


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (13 messages🔥): 
        
- **Hosting Servers for Embedding and LLM Models**: `@rofliex` asked about the possibility of hosting the server on 1234 and LM Studio on 1234 for embeding + LLM model. `@vic49` clarified that his program connects to LM Studio running in server mode and additional server mode isn't required by his program. `@rofliex` expressed appreciation for this solution.
- **Using Embeddings API**: `@rofliex` enquired about the need to clear suffix/preffix textboxes in LM Studio server panel configuration for utilizing embedings api and whether this requirement was only specific to `@vic49`'s chat implementation. In response, `@vic49` suggested disabling "automatic prompt formatting" in LM Studio, selecting a prompt, then updating settings in his program. He also advised removing anything in the prefix/suffix boxes in LM Studio.
- **Attempt to Run Databerry Project**: `@rofliex` mentioned attempting to run the [Databerry Project](https://github.com/gmpetrov/databerry), a no-code platform for building custom LLM Agents, and expressed the need for the correct embedding api for qdrant.
- **Feeding Multiple Folder to LLM**: `@andrew.lost` raised the question of whether it's possible to feed an LLM a folder containing multiple sub-folders and files for reading and scanning. This query remained unanswered as per the message log.

**Links mentioned**:

[GitHub - gmpetrov/databerry: The no-code platform for building custom LLM Agents](https://github.com/gmpetrov/databerry): The no-code platform for building custom LLM Agent...


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (12 messages🔥): 
        
- **Using Models on GPU**: `@taigasasori_94251` asked how to make models run on a 4090 GPU as only CPU load was shown. `@dagbs` suggested setting the GPU parameter to `-1` or a positive number, while `@fabguy` noted that the application UI doesn't show GPU utilization. Later, `@pefortin` advised to check the GPU offloading box in the UI and monitor vRAM usage using system tools.
- **Efficiency of Model on GPUs**: `@pefortin` shared their experience with dolphin mixtral Q5 on a combo of 3090 and 3060ti using PCIe x1 to x16 riser. They observed an increase in tokens per second from 6 to 10-11 and planned to test with old 10xx and 20xx series GPUs.
- **Issue on AMD GPU**: `@LokedTMX` experienced issues of GPU non-utilization while off-loading to an RX 6950xt AMD GPU. `@yagilb` acknowledged it as a known issue concerning AMD GPUs and provided a [link](https://discord.com/channels/1110598183144399058/1190037562866290699/1190037562866290699) for updates as well as invited comments to potentially test a beta build when available.


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (2 messages): 
        
- **Gibberish Problem in 0.2.10 (Windows)**: `@.gregly` noticed that switching truncation strategies and regenerating seems to temporarily fix the gibberish problem in version 0.2.10 (Windows), although the issue reoccurs on the next generation. This feedback was identified as useful by `@yagilb`.


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (21 messages🔥): 
        
- **Installation and usage of AutoGen**: User `@ddhmksoi` shared their struggle with setting up AutoGen. They followed the steps including downloading the latest zip from git, running install.py, and using pip install for autogen. However, they encountered issues while running certain Autogen scripts.
  
- **Understanding Error Messages in AutoGen**: `@ddhmksoi` encountered an error message that refers to issues with `autogen.oai.completion` and dependencies on `openai<1` and `diskcache` which raised concerns.

- **Use of Docker with AutoGen**: `@ddhmksoi` expressed confusion over Docker's involvement in the AutoGen process. They installed Docker as recommended but did not observe an active instance within the Docker application.

- **How AutoGen Works**: User `@dagbs` provided insight into how Autogen works. They pointed out that Autogen's behavior not only heavily depends on the model used, but also on the prompt given. Autogen may terminate prematurely if it determines the task is completed. To prevent early termination, `@dagbs` suggested adding a `system_message` inside of `UserProxyAgent()` to guide the model on task completion status. 

- **Location of AutoGen Files**: `@ddhmksoi` inquired about the location where Autogen files are saved after execution. `@dagbs` clarified that Autogen does not save any files, as it is a Python script that's meant for direct interaction.


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- User `@pradeep1148` sparked a discussion on AI Models by inquiring about the differences between transformer architectures in **llama2** and **mistral** on the #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) channel. A man contacting an astronaut on the ISS using a homemade antenna was also discussed through a [Twitter link](https://fxtwitter.com/historyinmemes/status/1740878634184061295) shared by `@teknium`.

- In the #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) channel, `@teknium` shared benchmarks displaying varied performance across multiple tasks and data sets for different versions of `TinyLlama`.  

- The #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) channel had discussions on running benchmarks for `Tinyllama` checkpoints, gaming in modded Minecraft, building a knowledge graph with Instructor and Pydantic for response_type, and a comprehensive comparison and ranking of various 7B models, including dolphin-2.6-mistral-7b, dolphin-2.6-mixtral-8x7b, Marcoroni-7B-v3, and mistral-ft-optimized-1218.

- `@teknium` ignited a thoughtful conversation on AI consciousness in the #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) channel while discussing the Hermes 2 AI bot's view on consciousness, sentience, and qualia in AI. The potential effects of doubling the model layers in AI models and the desire to train a *tiny-llama semantic chunker* using GPT4 data, were other areas of interest.

- The #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) channel had interesting exchanges about the simplicity of tokenization and the unveiling of a new AI model called [NeuralMix-2x7b](https://huggingface.co/mlabonne/NeuralMix-2x7b), a Mixture of Experts (MoE) created using [mergekit](https://github.com/cg123/mergekit).

- `@vic49` sparked a discussion about a query with a script execution in the #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) channel. The discussion evolved into project integration topics and code suggestions for smooth operations.

**Nous Research AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (3 messages): 
        
- **User Shared Links**: 
    - `@pradeep1148` shared a [YouTube link](https://www.youtube.com/watch?v=fpwCfoicHRM) without any context.
    - `@teknium` shared a [Twitter link](https://fxtwitter.com/historyinmemes/status/1740878634184061295) describing a man contacting an astronaut on the International Space Station using a homemade antenna.
- **Discussion on AI Models**: `@pradeep1148` asked about the differences between transformer architectures in **llama2** and **mistral**.

**Links mentioned**:

[Tweet from Historic Vids (@historyinmemes)](https://fxtwitter.com/historyinmemes/status/1740878634184061295): This guy contacted an astronaut on the ISS using a...


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (8 messages🔥): 
        
- **Training TinyLlama**: `teknium` shared benchmarks for different versions of `TinyLlama`, specifically intermediate steps with various batch sizes.
    - **TinyLlama-1.1B-intermediate-step-1431k-3T**: Demonstrated varied performance across different tasks. Achieved an average accuracy of 52.99% on a set of tasks including ARC, Boolq, HellaSwag, and others.  Achieved an average accuracy of 21.05% on a set of tasks from AgeIVal (AQA, LogiQA, LSAT AR, etc.). Performed at 31.95% average on a set of tasks from BigBench. "TruthfulQA MC" performance is reported with mc1 and mc2 metrics.
    - **TinyLlama-1.1B-intermediate-step-1195k-token-2.5T**: Showed slightly inconsistent performance compared to the previous model. Obtained an average of 53.84% on a set similar to the first batch, but then dropped to 21.45% on a set from AgeIVal. In the BigBench tasks, it achieved an average performance of  31.73%. Similar to the previous model, "TruthfulQA MC" performance is reported.


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (10 messages🔥): 
        
- **Benching Tinyllama Checkpoints**: `@teknium` mentioned running benchmarks for the last three checkpoints of Tinyllama.
- **Gaming Discussion**: A discussion about playing modded Minecraft was initiated by `@teknium` casually inquiring if `@1084792750001618965` indulges in it. The conversation ended up including `@max_paperclips`, who mentioned that they do play the game occasionally.
- **Knowledge Graph Building**: `@fullstack6209` shared a [GitHub Gist link](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34) about their project on building knowledge graph data with guidance using Instructor and Pydantic for response_type. They mentioned the process taking about 30 minutes on a 2080ti/3090 setup with VLLM.
- **AI Model Comparison**: `@metaldragon01` shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/18u122l/llm_comparisontest_ranking_updated_with_10_new/) offering a comprehensive comparison and ranking of various 7B models, including dolphin-2.6-mistral-7b, dolphin-2.6-mixtral-8x7b, Marcoroni-7B-v3, and mistral-ft-optimized-1218. The Nous Capybara model was mentioned favorably. 
- **Model Offloading**: `@gabriel_syme` shared a [GitHub link](https://github.com/dvmazur/mixtral-offloading) about running Mixtral-8x7B models in Colab or consumer desktops.

**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18u122l/llm_comparisontest_ranking_updated_with_10_new/)
- [GitHub - dvmazur/mixtral-offloading: Run Mixtral-8x7B models in Colab or consumer desktops](https://github.com/dvmazur/mixtral-offloading): Run Mixtral-8x7B models in Colab or consumer deskt...
- [asdf.py](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34): GitHub Gist: instantly share code, notes, and snip...


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (75 messages🔥🔥): 
        
- **Speculation on AI Consciousness**: `@teknium` shared a Hermes 2 AI bot's output contemplating *consciousness, sentience, and qualia* in artificial intelligence. The bot highlighted the abstract and poorly understood nature of these concepts, and how they might manifest within AI. Despite some forms of AI exhibiting characteristics of human sentience and consciousness, the AI concluded that current understanding and technology don't support the assertion that AI possesses the conscious, sentient, or qualitative attributes of living beings.
- **Discussion on Scaling AI Models**: `@.wooser` speculated about doubling the model layers and the effects it could have on the model's performance. They questioned if such actions would provide a fourfold increase in performance efficiency.
- **Mistral's Ranking**: `@mihai4256` mentioned that Mistral's strongest model now ranks similarly to 7b models, which shows a different trend against other benchmarks. They were still investigating the reason for this trend.
- **Semantic Chunking on GPT4**: `@gabriel_syme` expressed interest in training a *tiny-llama semantic chunker* using GPT4 data. Their approach would involve taking 4k token text inputs provided to GPT4 and splitting the output into 1-10 sentence chunks based on semantic context.
- **Impressed with Mergekit**: `@mihai4256` remarked on how shocked they were to discover that Mergekit has a mixtral branch. Expectedly, users are looking forward to seeing how well it performs.

**Links mentioned**:

- [nRuaif/IWasDointCrystalMethOnTheKitchenButThenMomWalkedIn-NeuralHermesStripedCapybara-Mistral-11B-SLERP · Hugging Face](https://huggingface.co/nRuaif/IWasDointCrystalMethOnTheKitchenButThenMomWalkedIn-NeuralHermesStripedCapybara-Mistral-11B-SLERP)
- [mlabonne/Beyonder-4x7b · Hugging Face](https://huggingface.co/mlabonne/Beyonder-4x7b)
- [Tweet from lmsys.org (@lmsysorg)](https://twitter.com/lmsysorg/status/1740792947711570084?t=kiDS--15lesIEPcjc4Qq_g&s=19): @MistralAI&#39;s strongest model, Mistral-Medium, ...
- [RAG](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/rag#transformers.RagTokenizer)
- [Tweet from Delip Rao e/σ (@deliprao)](https://x.com/deliprao/status/1740610760219168883?s=20): AI researcher tuning hyperparameters of their LLM


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (6 messages): 
        
- **Tokenization Explained**: `@wooser` commented on the simplicity of tokenization, stating that it's **computationally super easy** and involves chopping up a text file using items from a dictionary. 
- **New AI Model - NeuralMix-2x7b**: `@jason.today` shared a new AI model called [NeuralMix-2x7b](https://huggingface.co/mlabonne/NeuralMix-2x7b), a **Mixture of Experts (MoE)** created using [mergekit](https://github.com/cg123/mergekit) (mixtral branch). It's composed of the following base models: [OpenPipe/mistral-ft-optimized-1218](https://huggingface.co/OpenPipe/mistral-ft-optimized-1218) and [mlabonne/NeuralHermes-2.5-Mistral-7B](https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B).
- **Unexpected Language Output**: `@fullstack6209` reported that NeuralMix-2x7b started speaking Russian for an undisclosed reason.

**Links mentioned**:

[mlabonne/NeuralMix-2x7b · Hugging Face](https://huggingface.co/mlabonne/NeuralMix-2x7b)


### ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/) (1 messages): 
        
dogehus: show me how I can operate the lasted neo cortex we have available pleas


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (9 messages🔥): 
        
- **Code Execution Discussion**: User `@vic49` raised an issue about a problem with a script execution, presuming it to be a **command line argument creator**.
- `@qnguyen3` confirmed that **LMStudio** uses the same code for testing purposes.
- **Project Integration Topic**: `@qnguyen3` shared insights on people integrating **Obsidian** into their app. They suggested that if there were any issues, they should have been reported on both **HF and GitHub**.
- `@vic49` specified that his reference is related to using the **native format** of Obsidian, not the GGUF version used by LMStudio.
- **Code Suggestion**: `@qnguyen3` proposed a command to try: `python llava/serve/cli.py --image-file your_image.jpg`.


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- Discussion about **loss reduction techniques** for ultimate model performance and **usage of Wandb** for better visualization.
    - "*Increase the batch size and reduce the learning rate to reduce the fluctuation in training loss*." - `_jp1_`  
    
- Conversation on **estimating model performance** with a new tokenizer, with the mention of using the **16/32 rank alpha combination** for training, and checking performance through task completions.

- Focus on **Airship Axolotl training**: the discussion concerning a VRAM spike issue with `sample_packing`, suggestion of adding chat templates to `tokenizer_config.json` for traits like `chatml`, `vicuna`, and `llama2 chat`.
    - [VRAM spike issue discussion](https://discord.com/channels/1104757954588196865/1104758010959634503/1189936010092626022)

- A call for managing Axolotl installation issues via a *reproducible pip / conda environment*, and considering `mamba` as a *dependency*.
    - [Issue comment related to Mamba dependency](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1019#issuecomment-1872353545)

- A milestone by the **TinyLlama project**: pretraining a [1.1B model on 3 trillion tokens](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T).

- Observations on the performance difference between *Mixtral* and *Mistral* models, and complications arising from *EOS token conflicts* when merging instruct models and story models.
    - [Classifier tutorial link](https://github.com/mistralai/mistral-src/blob/main/tutorials/classifier.ipynb)

- Insight into the **ultrachat_200k** dataset, with questions on how to use it for training, understanding the `train_gen` format, and confirming the usage of `train_sft` split and binarized dataset based on specific recipes.
    - [Dataset card link for ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
    - [Dataset card link for ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
    - [Link to Zephyr-7B-β recipe](https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta)


**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (5 messages): 
        
- **Loss Reduction Techniques**: `@_jp1_` recommended increasing the batch size and reducing the learning rate to **reduce the fluctuation in training loss**. He added that fluctuations in training loss are not concerning as long as the evaluation loss is decreasing.
- **Usage of Wandb**: `@_jp1_` emphasized the importance of using **wandb** for learning and tracking model performance, stating that its use only requires setting an environment variable and adding a line in the axolotl configuration.
- **Evaluation of Model Performance**: `@noobmaster29` questioned the possible ways to estimate if the model is working well with a new tokenizer. He also mentioned that a loss of around 2 seems decent for text completion. 
- **Rank and Alpha Combination**: In reply to `@noobmaster29`, `@nanobitz` suggested using the **16/32 rank alpha combination** for training the model. He also added that testing the model can simply be done through some completions.


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (45 messages🔥): 
        
- **VRAM spike issue with sample_packing**: Users `@_jp1_` and `@nanobitz` discussed a [VRAM spike issue](https://discord.com/channels/1104757954588196865/1104758010959634503/1189936010092626022), which appears when using `sample_packing` during training. This issue seems to be specific to certain datasets. While a definite solution has not been found, the issue doesn't occur when `sample_packing` is switched off. 

- **Adding chat templates to the tokenizer_config.json**: `@le_mess` raised the topic of adding chat templates to `tokenizer_config.json`, asking users what chat templates, other than `chatml`, should be included. `@caseus_` suggested including `llama2-chat`, `chatml`, `vicuna`, and `alpaca-instruct`.

- **Reproducible pip / conda environment for Axolotl**: `@nanobitz` raised the need for a reproducible pip / conda environment for Axolotl due to several installation issues observed. `@le_mess` suggested using the `pip freeze > requirements.txt` command, and `@xyzzyrz` noted complications with supporting multiple versions of torch for CI processes that build docker images.

- **Mamba Dependency for Axolotl**: Discussion occurred regarding making `Mamba` a required dependency for Axolotl to avoid related issues. `@nanobitz` mentioned an [issue comment](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1019#issuecomment-1872353545) related to this, to which `@caseus_` responded affirmatively.

- **TinyLlama Project milestone**: `@faldore` reported that the TinyLlama project had reached the milestone of pretraining a [1.1B Llama model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) on 3 trillion tokens.

**Links mentioned**:

- [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [Adds chat templates by mhenrichsen · Pull Request #1022 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1022): Adds Chat templates for easier inferencing chat mo...
- [fix: warn user to install mamba_ssm package by NanoCode012 · Pull Request #1019 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1019#issuecomment-1872353545): Fixes #975 . Warns when user does not have package...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (15 messages🔥): 
        
- **Performance Difference Between Mixtral and Mistral**: `@semantic_zone` has noticed a significant decrease in training and evaluation loss when switching the model from *Mixtral* to *Mistral* on their large dataset. `@_jp1_` suggested that the learning rate could be adjusted as it should probably be much smaller with a smaller batchsize and without sample packing. `@_jp1_` also provided a link to a [classifier tutorial](https://github.com/mistralai/mistral-src/blob/main/tutorials/classifier.ipynb) that suggested using a linear predictor on embeddings instead of next token prediction.
- **ChatML Models and EOS Token Conflicts**: `@henk717` shared that merging *instruct models* with *story models* previously led to *EOS token conflicts*, causing problems with models that specified `

**Links mentioned**:

- [Intel/neural-chat-7b-v3-1 · Prompt Template?](https://huggingface.co/Intel/neural-chat-7b-v3-1/discussions/1#655533a4bc6ff300d447f85d)
- [mistral-src/tutorials/classifier.ipynb at main · mistralai/mistral-src](https://github.com/mistralai/mistral-src/blob/main/tutorials/classifier.ipynb): Reference implementation of Mistral AI 7B v0.1 mod...


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **Training with ultrachat_200k dataset**: User `@noobmaster29` inquired about methods to use the **ultrachat_200k** dataset for training and whether there is an axolotl data template for this purpose or whether manual configuration is needed in the dataset. He provided a [dataset card link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) for more information on **ultrachat_200k**.
- **train_gen split in ultrachat_200k**: `@noobmaster29` also sought clarity on the format of the `train_gen` split in the **ultrachat_200k** dataset, as he didn't see a chosen/rejected pair in this split compared to what is present in the `ultrafeedback_binarized` dataset, sharing its [dataset card link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) for reference.
- **Use of train_sft split and binarized dataset**: According to `@noobmaster29`, based on the alignment handbook's recipts on **Zephyr-7B-β**, only the `train_sft` split from the **ultrachat_200k** dataset was used for **sft**, and the binarized dataset was used for **dpo**. He provided a [link to the recipe](https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta) for reference. 
- **ultrachat prompt strategy merging**: User `@caseus_` responded that they had just merged an ultrachat prompt strategy.

**Links mentioned**:

- [HuggingFaceH4/ultrachat_200k · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [HuggingFaceH4/ultrafeedback_binarized · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [alignment-handbook/recipes/zephyr-7b-beta at main · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta): Robust recipes for to align language models with h...


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Language Learning Models (LLMs) and Function Choice Optimization**: Users explored how to optimize LLMs function choice through prompt modifications. User `.tanuj.` shared this method of defining a function call using system messages and context generation from prompt in [their code](https://github.com/TanGentleman/microchain/blob/main/microchain/engine/engine.py).
- **Interests and Discussions on Different Mistral Models**: Chatters expressed anticipation for the **Mistral Medium** feature, and shared experiences of **Mistral-Medium**'s performance. Also, question regarding using wasmedge to run **Mistral** and the possibility of fine-tuning **Mistral** on non-English languages were made.
- **Deployment and Hardware-related Inquiries**: User discussions included potential model choices based on hardware capabilities, including suggestions such as the [openchat 3.5 model](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main) for systems with hardware limitations. Query about deploying a chatbot on an integrated GPU was also raised.
- **Mistral Model Discussions and Showcasing**: Users shared experiments with different models, including **Mistral-7b** and **Mistral-7B-Instruct-v0.2**. Inquiries about MoE model's capabilities and a predictive future of resolving performance issues by the end of January were discussed. Users shared the [HuggingFace blog post](https://huggingface.co/blog/moe) and an [AI research survey video](https://www.youtube.com/watch?v=fpwCfoicHRM) for a deeper understanding of MoE.
- **Model Limitations and their Impact**: User `@gilford3641` sought advice on a local GPT model that supports a high number of input tokens to facilitate large text processing. A suggestion was given for [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main), however, it didn't fit `@gilford3641`'s needs.
- **Platform and Model-related Discussions**: Users compared **Mistral AI client** vs **OpenAI client**; speculated performance of **mistral-tiny API**; anticipated the introduction of 2FA on **Mistral.ai**; faced issues with rate limits on Mistral and experienced issues with **mistral-medium model**. Identified issues with LLMs performance with Huggingface libraries were discussed. `@daain` suggested a [GitHub project](https://github.com/fleet-ai/context) to generate answers or code in Python using a LLM.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (9 messages🔥): 
        
- **Prompting Method and LLMs Abstract APIs**: `@theledgerluminary` inquired for examples on how to prompt Language Learning Models (LLMs) to optimize function choice. They suggested that LLMs API's are abstract and function ideally by adjusting the prompt templates. `@.tanuj.` responded by sharing a high-level overview of his approach, emphasizing creating system messages for tasks, generating context with few-shot examples, allowing the agent to iterate series of steps and ensuring there is no unintentional side effect. They [shared a link to their code](https://github.com/TanGentleman/microchain/blob/main/microchain/engine/engine.py) for processing the LLM's output into a clearly defined function call.
- **Interest in Mistral Medium**: `@lee0099` and `@meyelo` expressed anticipation for **Mistral Medium** feature.
- **Mistral Medium Performance and Local Usage**: `@gilford3641` shared their experience that **Mistral-Medium** appears to perform better than **Mistral-8x7b**. They further inquired if it's possible to run Mistral-Medium locally, to which `@lee0099` responded that it hasn't been released yet.
- **Warm Welcome to New Members**: `@akasik7243` and `@aircactus500` announced their arrival and expressed their excitement to be part of the chat.

**Links mentioned**:

[microchain/microchain/engine/engine.py at main · TanGentleman/microchain](https://github.com/TanGentleman/microchain/blob/main/microchain/engine/engine.py): function calling-based LLM agents. Contribute to T...


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (1 messages): 
        
ved_ikke: Anybody using wasmedge to run mistral?


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (6 messages): 
        
- **Chatbot Deployment with Specific Hardware**: `@ethux` advised that deploying a 4 or 3bit GGUF model would be possible with a GPU that has 4GB VRAM and an additional 8GB Shared VRAM from the PC. 
- **Alternative Model Recommendation**: `@ethux` suggested to consider the [openchat 3.5 model](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main) for those with hardware limitations. `@azetisme` expressed intent to look into this suggestion. 
- **Integrated GPU Inquiry**: `@hharryr` queried about the possibilities of deploying a chatbot on an integrated GPU, particularly the one on R7 7840H, which is paired with 32GB of RAM. This question remained unanswered in the given data.

**Links mentioned**:

[TheBloke/openchat-3.5-1210-GGUF at main](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main)


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (3 messages): 
        
- **Fine-tuning Mistral on Non-English Languages**: `@deatheater006` enquires about the process of fine-tuning Mistral on a non-English language. Specifically, they express interest in the Tamil language. `@pieswap` engages with the query, seeking specific details on the intended language.


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (10 messages🔥): 
        
- **Mistral-7b Performance**: `@.gue22` shared their experience with the [Mistral-7b model running on Nvidia A40 (Large) GPU hardware](https://replicate.com/mistralai/mistral-7b-instruct-v0.1), noting subpar performance, lack of generalization, and the benefit of Google search speed over the chatbot's responses. 
- **Local Model Execution**: `@fayiron` recommended running models locally using [text-generation-webui](https://github.com/oobabooga/text-generation-webui) for improved performance and control, especially on a Debian desktop.
- **Potential of MoE Models**: In response to `@.gue22`, `@daain` elaborated on the transition to Mixture of Experts (MoE) models for robust performance at lower computational costs, citing the [HuggingFace blog post](https://huggingface.co/blog/moe) on the subject.
- **Future Projections**: `@daain` also predicted that by the end of January, the remaining performance issues with MoE models will be resolved, enabling a mid-range model to run locally at small model speeds and opening up new use cases.
- **Model Education Resource**: For further understanding of MoE, `@pradeep1148` and  `@.gue22` shared a YouTube link to an [AI research survey video](https://www.youtube.com/watch?v=fpwCfoicHRM). The video covers the impact of Mixture of Experts (MoE), multimodal learning, and Artificial General Intelligence (AGI) on generative AI.

**Links mentioned**:

- [Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [mistralai/mistral-7b-instruct-v0.1 – Run with an API on Replicate](https://replicate.com/mistralai/mistral-7b-instruct-v0.1)
- [From Google Gemini to OpenAI Q*: A Survey of Reshaping the Generative AI Research Landscape](https://www.youtube.com/watch?v=Z8VUhK1OGfk.): This survey examines the impact of Mixture of Expe...


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (4 messages): 
        
- **Discussion on GPT Model that Supports Large Input Tokens**: User `@gilford3641` was seeking suggestions on a local GPT model that allows a high number of input tokens. They aim to enter a large text (up to 10k tokens), which the model will paragraph conditionally. Their trials with several models on **Gpt4All** resulted unsuccessful, citing the apps' lack of support for long inputs. 
- **Suggestion from bam4d**: In response, `@bam4d` recommended [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main), a model that is tuned for a 32k context window. 
- **gilford3641's Previous Experience with Mistral Medium Model**: Despite acknowledging the suggestion, `@gilford3641` stated that they had previously tested this model, which was unable to fully process their input. They provided more detail about their experiment, mentioning that their attempt involved an input of over 7k Simplified Chinese characters (with 1 character equal to 1 token). The model processed about 60% of the text and did not yield more output. They questioned whether emotional medium's token count relies on ASCII or a 2-byte encoding, suggesting this as a potential reason for its failure to process 7k * 2 tokens.

**Links mentioned**:

[mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main)


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (17 messages🔥): 
        
- **Mistral AI Client vs OpenAI Client**: `@lerela` clarified that the **Mistral AI client** primarily focuses on completions, whereas the **OpenAI client** incorporates numerous OpenAI-exclusive features. They emphasized that while Mistral's client is more streamlined, the openai Python package is still a viable choice if an application is already utilizing it.

- **Performance of mistral-tiny API**: `@sublimatorniq` posed a query around the performance of the **mistral-tiny API**, hypothesizing it might be sharper than the mistral-7b weights they downloaded to operate locally, potentially due to having the quant v. The speculation was concurred by `.superintendent`, attributing the sharpness to the API running at fp16 and the local operation running a quant.

- **Two-Factor Authentication on Mistral.ai**: `@ved_ikke` inquired about the anticipated introduction of two-factor authentication (2FA) when logging into **Mistral.ai**. There was no answer at this time.

- **Rate Limits on Mistral**: Michaelwechner discussed their experience with the rate limits, noting that a single user received an average of 34 responses per minute with an average response time of 1.76 seconds. The user ran into an issue when they increased to two concurrent users and began receiving a "request rate limit exceeded" message after a few queries. They linked to the [Mistral pricing and rate limits](https://docs.mistral.ai/platform/pricing/) document for further reference.

- **Limited Language Model Performance with Huggingface Libraries**: `@casper_ai` voiced an observation that most large language models (LLMs), including **Mistral medium**, seem quite weak while working with Huggingface libraries—they tend to hallucinate arguments to simple functions. They also expressed hope for future optimization. In response, `@daain` suggested a project on [GitHub](https://github.com/fleet-ai/context) that uses a vector database embedding the ~1200 most popular Python libraries, which can be used in conjunction with an API or a local LLM for generating answers or code.

- **Mistral-medium Model Issues**: `@pw3456` reported experiencing issues with the **mistral-medium model** no longer following chat protocols and replying on behalf of both parties. `@sublimatorniq` reported not seeing this issue currently.

**Links mentioned**:

- [Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/): Pay-as-you-go
- [GitHub - fleet-ai/context: A CLI tool &amp; API over the top 1221 Python libraries.](https://github.com/fleet-ai/context): A CLI tool &amp; API over the top 1221 Python libr...


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- Conversation around **safe script sharing practices**, with a recommendation to leverage platforms like GitHub or Hugging Face Hub instead of .zip files. 
    - "*avoid sharing .zip files*" - cakiki
- Various **resources for AI job searches** were discussed, with specific mention of [https://www.aimljobs.fyi/](https://www.aimljobs.fyi/) and solicitation for additional platforms for AI related positions.
- Queries and interest around **large language models (LLMs) and gradient-free methods**, specifically evolutionary algorithms, for model training.
- A user displayed interest upon **MoE model SOLARC-MOE-10.7Bx4** and its potential performance, sharing the model's [Hugging Face link](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF).
- A call for advice and resources for **fine-tuning the Blenderbot model** and understanding the dataset format.
    - "*fine-tuning the Blenderbot model*" - tchi_tchi_
- Learning exploration of **model soups** and **LightGBM** as shared by a user on the guild.
    - "*model soups and LightGBM*" - onceabeginner
- A recommendation to explore [Trending Papers](https://trendingpapers.com/papers?o=pagerank_growth&pd=Since%20beginning&cc=Cited%20and%20uncited%20papers&c=All%20categories), a resource that ranks the **top trending papers in computer science**.
- Announcement and launch of **MindMirror**, an AI-based app providing audio transcriptions and sentiment analysis, and the **Bunkoer Library** was introduced as a new open-source Python library aimed at enhancing data security for LLM tasks with [Github link](https://github.com/Bunkoer/bunkoer).
- Sharing of the **Canarim-Bert-Nheengatu project**, a BERT model pre-trained for the Nheengatu language with the link [here](https://huggingface.co/dominguesm/canarim-bert-nheengatu).
    - "*Canarim-Bert-Nheengatu Project*" - dominguesm
- Discussion on **personalization techniques in Diffusers**, and a [document](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation) was shared showcasing techniques to control the generation of diffusion models.
- A discussion on the **InternVL** model, ViT-6B and comparison to Google's ViT-22B, alongside an interest in the **Sloot Digital Coding System** and shared [Wikipedia link](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System).
- Addressing concerns on **validation of legal data** and limitations set by using a typical train-test split methodology.
- Insight on the performance of **Korean SOLAR-LLM-10.5B** and its comparison to Mixtral7*8B.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (16 messages🔥): 
        
- **Sharing scripts**: `@cakiki` advised to **avoid sharing .zip files** in the server, recommending GitHub or Hugging Face Hub instead.
- **AI job search**:  `@Priyansh Rastogi` asked for resources on finding AI related jobs, mentioning they currently use [https://www.aimljobs.fyi/](https://www.aimljobs.fyi/) but are seeking other platforms.
- **LLMs and gradient-free methods**: `@_hazler` inquiring about any known research related to **training LLMs with gradient-free methods** such as evolutionary algorithms.
- **MoE model SOLARC-MOE-10.7Bx4**: `@jiha` drew attention to the SOLARC-MOE-10.7Bx4 model, expressing interest in its potential performance. They shared the model's [Hugging Face link](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF).
- **Fine-tuning Blenderbot**: `@tchi_tchi_` asked for assistance with **fine-tuning the Blenderbot model** and needed help understanding the dataset format.

**Links mentioned**:

- [AI, ML, Data Science Jobs](https://www.aimljobs.fyi/)
- [TheBloke/SOLARC-MOE-10.7Bx4-GGUF · Hugging Face](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF)


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (1 messages): 
        
- **Model Soups and LightGBM**: User `@onceabeginner` shared that they are currently learning about **model soups** (averaging weights of models) and **LightGBM** (gradient boosting decision tree).


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (2 messages): 
        
- **Trending Papers in Computer Science**: `@cyruscao` shared a link to [Trending Papers](https://trendingpapers.com/papers?o=pagerank_growth&pd=Since%20beginning&cc=Cited%20and%20uncited%20papers&c=All%20categories), a resource that ranks the top trending papers in computer science. The site added 616 new papers in the last three days. `@horosin` followed up by asking about `@cyruscao`'s specific area of interest (architecture).

**Links mentioned**:

[Trending Papers](https://trendingpapers.com/papers?o=pagerank_growth&pd=Since%20beginning&cc=Cited%20and%20uncited%20papers&c=All%20categories)


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (6 messages): 
        
- **MindMirror App Announcement**: `@ddchiken` announced the demo of a new app, **MindMirror**, which is an AI audio transcription tool designed to analyze thoughts and emotions throughout conversations. Currently providing basic sentiment analysis, the app's future plans are to include summarization, action items, and other insights. Designed with privacy in mind, it does not perform data synchronization or transmit users' audio off-device. The app is free, works on mobile and desktop via web browser, and does not require an account. `@ddchiken` encourages everyone to use it and provide feedback. [MindMirror App link](https://mindmirror.onrender.com/).
- **Canarim-Bert-Nheengatu Project Sharing**: `@dominguesm` shared their project, a BERT model pre-trained for the Nheengatu language—an indigenous language spoken in Brazil. The project was particularly time-consuming due to the extensive data collection required, sourcing primarily from books dating to the 1800s and 1900s. `@dominguesm` says the model could be useful for future NLP tasks aimed at developing resources for the Nheengatu language. [Project link](https://huggingface.co/dominguesm/canarim-bert-nheengatu).
- User `.naptastic` asked whether the dataset for Canarim-Bert-Nheengatu is available. `@dominguesm` replied that it's not available yet but will be soon.
- **Introduction of Bunkoer Library**: `@jossai88` introduced a new open-source Python library, **Bunkoer**, aimed at enhancing data security in LLM tasks. Capabilities include data anonymization—specifically for CSV and PDF files, Streamlit integration for a user-friendly interface, and contextual anonymization for local data security. The library is actively developed with plans for further expansion, encouraging contributions. For detailed information, they shared the link to the [GitHub repository](https://github.com/Bunkoer/bunkoer).

**Links mentioned**:

- [dominguesm/canarim-bert-nheengatu · Hugging Face](https://huggingface.co/dominguesm/canarim-bert-nheengatu)
- [Star Trek Star Trek Tos GIF - Star Trek Star Trek Tos Scotty - Discover &amp; Share GIFs](https://tenor.com/view/star-trek-star-trek-tos-scotty-electrocute-electricity-gif-16908385): Click to view the GIF
- [GitHub - Bunkoer/bunkoer: This the bunkoer library, for secure your data on all your llm task](https://github.com/Bunkoer/bunkoer): This the bunkoer library, for secure your data on ...
- [MindMirror](https://mindmirror.onrender.com/)


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 
        
- **Personalization Techniques in Diffusers**: `@sayakpaul` discusses the techniques to control outputs generated by diffusion models, which is an active research topic in the community. He mentions that subtle changes in inputs can drastically change outputs in diffusion models. He also shares a HuggingFace [document](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation) presenting some of the techniques diffusers support to control generation of diffusion models. The goal is to map changes in input accurately to changes in output, influence qualities of generated images beyond semantic preservation, and generate outputs with good quality that adhere to a particular style or be realistic.

**Links mentioned**:

[Controlled generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation)


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (6 messages): 
        
- **InternVL Discussion**: `@chklrd` shared a link to the model card of InternVL-Chat-ViT-6B-Vicuna-13B, which was developed by [OpenGVLab](https://github.com/OpenGVLab/InternVL). InternVL scales up the Vision Transformer (ViT) to 6 billion parameters and aligns it with Language Model. It has achieved 32 state-of-the-art performances on tasks such as visual perception, cross-modal retrieval, and multimodal dialogue. [([Project Link](https://arxiv.org/abs/2312.14238))].
- `@nielsr_` pointed out that InternVL presents an open-source alternative to Google's ViT-22B.
- **Sloot Digital Coding System**: `@tomgale_` brought up the topic of the Sloot Digital Coding System, a data sharing technique that allegedly could store a complete digital movie file in 8 kilobytes of data. He noted that he has all the sources and proofs based on scientific method and observation and is seeking help for the algebra side of the project. He shared a link to a [Wikipedia article](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System) about the system.

**Links mentioned**:

- [OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B · Hugging Face](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B)
- [Sloot Digital Coding System - Wikipedia](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (2 messages): 
        
- **Validation of Legal Data**: `@shilz0145` expressed concerns about how to perform validation on a chunk of legal data and the limitations of using a train-test split in this scenario.
- **Performance of Korean SOLAR-LLM-10.5B**: `@harsh_xx_tec_87517` pointed out the impressive performance of **Korean SOLAR-LLM-10.5B** on HuggingFace leaderboard, noting that it almost matches the performance of **Mixtral7*8B** and inquired about the difference in these models.


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 
        
- **Controlling Diffusion Models Output**: User `@sayakpaul` shared a link to a document on the HuggingFace site discussing how to control outputs generated by diffusion models, an active research topic. The document explores ways to preserve semantics in the inputs for consistent outputs, and techniques supported by `diffusers` to regulate the generation of diffusion models. Find the document [here](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation).

**Links mentioned**:

[Controlled generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/controlling_generation)


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- Discussion regarding tools and processes in **LangChain**, with users sharing inquiries and solutions. Key topics include per user retrieval with Chroma, creating more advanced RAG in Node, adjusting `RecursiveCharacterTextSplitter`, and using the URL document loader from LangChain for FAQ generation. Various questions about API options, Firebase support in Python, and MongoDB Atlas Vector Search were also raised. User `@3h0480` specifically highlighted an issue regarding the transfer of information between generations in LangChain ([source](https://github.com/langchain-ai/langchain/issues/15247)). Other users sought clarification on terminology within LangChain.
- Exploration of **data security** with Large Language Models, with `@jossai88` initiating a discussion about safely handling sensitive data with models like ChatGPT 4, Llama 2, or Mistral AI, highlighting the parallels with launching a Docker container.
- Announcement of a new software release, **Bunkoer v0.0.3**, designed to anonymize PDFs and CSV files, aiming to enhance data protection in AI applications. `@jossai88` invited contributions to the Bunkoer repository on Git, though no specific link was provided.
- Sharing of a **tutorial link** in the #tutorials channel with no context ([source](https://youtu.be/Z50BFFrmMbc?si=rsn4AbIcbzmU6GgJ)).
- Brief mention in the #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) channel by user `@a404.eth` suggesting the holiday season as a reason for some unspecified situation.
  
**Links mentioned**:

- [Per-User Retrieval | 🦜️🔗 Langchain](https://python.langchain.com/docs/use_cases/question_answering/per_user)
- [neuralmagic/bge-large-en-v1.5-quant · Hugging Face](https://huggingface.co/neuralmagic/bge-large-en-v1.5-quant)
- [DOC: langchain LCEL - transfer of information between generations · Issue #15247 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/15247)
- [Tutorial Video](https://youtu.be/Z50BFFrmMbc?si=rsn4AbIcbzmU6GgJ)

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (12 messages🔥): 
        
- **Query regarding per user retrieval using Chroma**: `@pauln07` asked how they can implement per user retrieval using Chroma like they did with pinecone based on this [tutorial](https://python.langchain.com/docs/use_cases/question_answering/per_user).
- **Creating more advanced RAG in Node**: `@andremik` sought advice on whether to use a fastapi server with LangChain in Python or create it in Node. The user mentioned wanting to use features like query expansion, hybrid search against supabase or pinecone, and cohere reranking.
- **Tool names must be alphanumeric**: `@a404.eth` mentioned that tool names in LangChain need to be alphanumeric.
- **Adjusting `RecursiveCharacterTextSplitter`**: `@nas0875` asked how they can adjust the `RecursiveCharacterTextSplitter` so that full stops appear at the end of the chunks rather than at the start.
- **Using URL document loader from LangChain**: `@kvn2000` suggested using the URL document loader from LangChain to generate FAQs. This process involves passing the loaded URL content to LLM with a prompt and then using output schemas and parsers for output formatting.
- **Query regarding MongoDB Atlas Vector Search**: `@vaironman` inquired if anyone has experience with MongoDB Atlas Vector Search.
- **Choosing between different HuggingFaceEmbeddings or APIs**: `@mr.dronie` asked for advice on choosing between `TaylorAI/bge-micro-v2`, `neuralmagic/bge-large-en-v1.5-quant`, `together.ai`, or `perplexity` API for better models and faster inference, sharing the link of the [HuggingFace neuralmagic model](https://huggingface.co/neuralmagic/bge-large-en-v1.5-quant).
- **Support for Firebase in Python**: `@atefyamin` asked if Firebase is supported for memory in Python, noting that there's a JavaScript implementation.
- **Issue with langchain**: `@3h0480` asked for help regarding an issue they encountered involving the transfer of information between generations in LangChain, linking a related [GitHub issue](https://github.com/langchain-ai/langchain/issues/15247) for reference.
- **Definition of agent and chain**: `@shivam51` asked for clarification regarding the difference between an agent and a chain in LangChain.

**Links mentioned**:

- [neuralmagic/bge-large-en-v1.5-quant · Hugging Face](https://huggingface.co/neuralmagic/bge-large-en-v1.5-quant)
- [Per-User Retrieval | 🦜️🔗 Langchain](https://python.langchain.com/docs/use_cases/question_answering/per_user): When building a retrieval app, you often have to b...
- [DOC: langchain LCEL - transfer of information between generations · Issue #15247 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/15247): Issue with current documentation: I do not underst...


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
a404.eth: Maybe b/c it's the week between xmas and NYE?


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Data Security with Large Language Models**: User `@jossai88` raised a discussion about the importance of data security when processing sensitive data with advanced Large Language Models like ChatGPT 4, Llama 2, or Mistral AI. They emphasized the necessity for this issue to be addressed with the same caution as launching a Docker container.
- **Bunkoer v0.0.3**: `@jossai88` also presented their latest release, Bunkoer v0.0.3, which is designed to anonymize PDFs and CSV files. This update aims to provide advanced data protection features for secure and reliable AI applications.
- **Call for Contribution to Bunkoer Repo on Git**: The user invited community members to contribute to the Bunkoer repository on Git, especially if they are using tools like LangChain, LlamaIndex, Pinecone, FAISS, Auto-GPT, llamacpp, or OpenAI. No specific link was provided to the Bunkoer Git repository.


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
datasciencebasics: https://youtu.be/Z50BFFrmMbc?si=rsn4AbIcbzmU6GgJ


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **German Embedding/Retrieval Models Discussion**: User `@thewindmom` initiated a discussion on German embedding/retrieval model progress. Several other users expressed interest in the topic, including `@_jp1_` and `@rasdani`. Some specific models were mentioned such as `Colbertv2`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, and `deutsche-telekom/gbert-large-paraphrase` models. [Link to German BERT large paraphrase cosine](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine) and [German BERT large paraphrase euclidean](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean) were shared by `@rasdani`.
- **Vision Models Inquiry**: User `@lightvector_` asked about the current progress of vision models.  `@rasdani` responded, suggesting the user check out the ThursdAI podcast which recently covered multimodal models. [Link to podcast](https://overcast.fm/+BCi78S557I).
- **DPO Dataset/Tokenization Issue**: `_jp1_` expressed difficulty with the expected format for the dataset for DPO according to [TRL's documentation](https://huggingface.co/docs/trl/dpo_trainer#expected-dataset-format) and HuggingFace's [Alignment Handbook](https://github.com/huggingface/alignment-handbook/blob/61a11a5c7d66179ed0a930b0dd12e532fce701dd/src/alignment/data.py#L58). A comparison was also made regarding the format used in the [Ultrafeedback Binarized dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized). A problem with tokenization with ChatML and datasets with a system prompt was identified and subsequently fixed by user. They suggested to create a PR, as they saw a potential bug in the existing DPO pipeline.

**Links mentioned**:

- [📅 ThursdAI - Dec 28 - a BUNCH of new multimodal OSS, OpenAI getting sued by NYT, and our next year predictions &mdash; ThursdAI - The top AI news from the past week &mdash; Overcast](https://overcast.fm/+BCi78S557I)
- [deutsche-telekom/gbert-large-paraphrase-cosine · Hugging Face](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-cosine)
- [deutsche-telekom/gbert-large-paraphrase-euclidean · Hugging Face](https://huggingface.co/deutsche-telekom/gbert-large-paraphrase-euclidean)
- [sentence-transformers/distiluse-base-multilingual-cased-v2 · Hugging Face](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
- [DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer#expected-dataset-format))
- [HuggingFaceH4/ultrafeedback_binarized · Datasets at Hugging Face](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [alignment-handbook/src/alignment/data.py at 61a11a5c7d66179ed0a930b0dd12e532fce701dd · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/blob/61a11a5c7d66179ed0a930b0dd12e532fce701dd/src/alignment/data.py#L58): Robust recipes for to align language models with h...

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- Discussion on **Optimizing Code Metrics with LLM**: User `@slono` highlighted the impact of LLM on his coding habits, allowing for more efficient refactoring and quality tool development, with an example of a [tool for managing mass ticket deletion on Zendesk](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/zendesk).
- Exchange on **LLM Codemods and TypeScript's Role in Refactoring**: `@swizec` is enthusiastic about "LLM codemods" for large-scale refactoring and added value of TypeScript in catching basic mistakes.
- Conversation on the **Future of AI Field**: `@swyxio` shared a [Tweet with a list of open questions](https://fxtwitter.com/jxmnop/status/1740804797777797296?s=46&t=90xQ8sGy63D2OtiaoGJuww) for the AI industry in 2024, including potential breakthroughs, architecture, data privacy, and unchecked AI behaviour.
- **Podcasts and the AI Community**: `@swyxio` acknowledged an [appreciative Tweet](https://x.com/jackmccloy/status/1740812238217502917?s=46&t=90xQ8sGy63D2OtiaoGJuww) featuring a quote endorsing the approach of improving software without adding complexity, shared by `@JackMcCloy` in `@latentspacepod`.
- Announcement of the **Last Podcast of 2023**: In `#ai-event-announcements`, `@swyxio` shared a preview [link](https://www.latent.space/p/f05ffdf0-2563-4b9e-b9a7-96a3660d4780) to the podcast, set to be the last one for the year 2023.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (11 messages🔥): 
        
- **Optimizing Code Metrics with LLM**: User `@slono` highlighted the impact of LLM *(the Latent Language Model)* on his coding habits, noting that it allowed for a paradigm shift in how programming problems can be approached. This includes more efficient refactoring, and the creation of better quality tools that offer extensive assistance in his refactoring processes. An example he gave was the [tool he developed in 3 hours to manage mass ticket deletion on Zendesk](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/zendesk).
- **LLM Codemods and TypeScript's Role in Refactoring**: `@swizec` expressed enthusiasm about the concept of "LLM codemods" for large-scale refactoring and added that the ability of TypeScript to catch basic mistakes further facilitates the refactoring process.
- **Questions for the AI Field in 2024**: `@swyxio` shared a [Tweet with a list of open questions](https://fxtwitter.com/jxmnop/status/1740804797777797296?s=46&t=90xQ8sGy63D2OtiaoGJuww) for the AI industry in 2024 by `@jxmnop`. The questions cover potential breakthroughs, architecture, optimal parameters, data privacy, unchecked AI behaviour, and future learning models.
- **Podcast Shoutout**: `@swyxio` acknowledged an [appreciative Tweet](https://x.com/jackmccloy/status/1740812238217502917?s=46&t=90xQ8sGy63D2OtiaoGJuww) from `@JackMcCloy` that featured a quote from George Hotz on the `@latentspacepod`, endorsing the approach of improving software without adding complexity.

**Links mentioned**:

- [Tweet from jack morris (@jxmnop)](https://fxtwitter.com/jxmnop/status/1740804797777797296?s=46&t=90xQ8sGy63D2OtiaoGJuww): people keep saying AI is moving so fast. some days...
- [go-go-labs/cmd/apps/zendesk at main · go-go-golems/go-go-labs](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/zendesk): GO GO EXPERIMENTAL LAB. Contribute to go-go-golems...
- [Tweet from Jack McCloy (@JackMcCloy)](https://x.com/jackmccloy/status/1740812238217502917?s=46&t=90xQ8sGy63D2OtiaoGJuww): &#34;You can always make your software do more. Th...


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
swyxio: preview of last pod of 2023 https://www.latent.space/p/f05ffdf0-2563-4b9e-b9a7-96a3660d4780


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- A question was posed about comparisons being made between **children's reasoning abilities and Language Model's (LLMs) reasoning abilities** instigated by `@leuyann` in the General channel.
- In regards to **ChatGPT**, a user named `@hdartem` initiated a discussion about the use of **Nougat** for inputting data for paper reviews in the Papers channel.
- A possibility of collaborations was mentioned and flagged by `@hdartem` in the Off-Topic channel, citing potential overlapping work others might be doing.
- Shared resource in the Off-Topic channel from `@pradeep1148` with a link to a [YouTube video](https://www.youtube.com/watch?v=fpwCfoicHRM) discussing a new quantization technique called **Half-Quadratic Quantization (HQQ)**.
- Lastly, user `lightvector_` asked about any updates on vision in OSS in the Bakklava-1 channel.

**Skunkworks AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (1 messages): 
        
- **Comparing children's and LLM's reasoning abilities**: User `@leuyann` initiated a discussion asking if anyone has read any insights or research about comparing **children's reasoning abilities and Language Model's (LLMs) reasoning abilities**.


### ▷ #[papers](https://discord.com/channels/1131084849432768614/1131305311714672710/) (1 messages): 
        
- **Use of Nougat for ChatGPT**: `@hdartem` discussed using a tool called **Nougat** to input information onto **ChatGPT** for potential paper reviews and asked for clarification on the types of papers that were of interest.


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (2 messages): 
        
- **Potential Collaborations**: `@hdartem` mentioned that some people might already be working on an unspecified project, suggesting the need to identify these individuals.
- **Resource Sharing**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=fpwCfoicHRM) titled: "Half-Quadratic Quantization of LLM's (colab)", discussing a new quantization technique called Half-Quadratic Quantization (**HQQ**).

**Links mentioned**:

[Half-Quadratic Quantization of LLM&#39;s (colab)](https://www.youtube.com/watch?v=fpwCfoicHRM): In this article, we propose a new quantization tec...


### ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 
        
lightvector_: any updates on vision in oss?


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Configuring Azure OpenAI Service**: User `@0xmmo` expressed frustration with the process of configuring Azure's OpenAI service, likening it to wanting to "pierce my eyelids with dull rusty nails". They ended the vent shortly after, not adding more details.
        

---
The Alignment Lab AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The MLOps @Chipro Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The Perplexity AI Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---
The YAIG (a16z Infra) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
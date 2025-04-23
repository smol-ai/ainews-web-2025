---
id: 5aa32ff4-41c3-4ddf-8f02-7d76d46ca933
title: '[AINews] 1/3/2024: RIP Coqui'
date: '2024-01-04T06:56:46.257833Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-132024-rip-coqui
---

<!-- buttondown-editor-mode: plaintext -->> Meta: More tuning since yesterday. We've tuned down the repetitive OpenAI bug reports, and also tweaked the prompts for better summarization.

Coqui, one of the leading open source text to speech options surviving from the Mozilla ML group, [shut down today](https://twitter.com/_josh_meyer_/status/1742522906041635166). The announcement tweet is beautiful and heartfelt.

---

**Table of Contents**

[TOC] 


## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **"Fast as lightning" sdxl questions its own speed**: As pointed out by User `@aifartist`, the performance claims of **sdxl** being `3X faster` are dependent on specific techniques like use of `torch.compile` and removal of `fp16` and `attention`, casting doubt on the role of **diffusers 0.25** features in this performance improvement.
- **"Sharing is Caring" extends to HuggingFace user tokens as well**: According to `@osanseviero`, a *HuggingFace user token* can indeed be utilized on multiple running machines, although using distinct tokens is suggested for safer operations.
- **Learning Loss Minimization (LLM) Leaderboards play hide and seek**: `@lee0099`'s initial query about LLM leaderboard non-functioning was mooted as the leaderboard was later found to be working fine.
- **Creating Transformers from square one**: `@torres8552` shared a [Kaggle notebook](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook) providing a deep dive on constructing Transformer architecture for language-translation tasks from scratch using PyTorch.
- **Shoes, Sandals, and Boots strut on the Image Dataset ramp**: `@andysingal` introduced an image [dataset](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages) containing 15k images of shoes, sandals, and boots, promoting its use for multiclass classification with deep neural networks.
- **Web-Crawling mysteries of Common Crawl revealed**: `@exponentialxp`'s curiosity about the working of Common Crawl was satisfied by `@cakiki`'s explanation of the process involving powerful computers, a URL list and a 'spider' software for web crawling and indexing. An invitation for further exploration of the Common Crawl was extended via a link to the [Common Crawl codebase](https://github.com/commoncrawl) on GitHub.

**HuggingFace Discord Channel Summaries**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (85 messages🔥🔥): 
        
- **Speedy sdxl under scrutiny**: User `@aifartist` expressed skepticism regarding some *performance claims* related to **sdxl**, such as being `3X faster`. They noted that these claims appeared to depend heavily on methods not specific to **diffusers 0.25**, such as using `torch.compile` and removing `fp16` and `attention`. They requested clarification on which features specific to **diffusers 0.25** actually improved performance.
- **HuggingFace user token on multiple machines**: `@dizzyme` inquired whether a *HuggingFace user token* can be employed on two or more running machines. `@osanseviero` confirmed that it could, but suggested that using distinct tokens might generally be safer.
- **Python command issue on Arch Linux**: User `@gez_gin` encountered an issue on Arch Linux where the terminal reported `from` as an unknown command. `@cakiki` pointed out that `from` is a Python keyword, and suggested to `@gez_gin` that they run Python first to obtain a Python REPL.
- **Learning Loss Minimization (LLM) Leaderboards Glitch**: `@lee0099` queried about issues with the LLM leaderboards stating that it was not functioning. Later, they updated that the problem seemed to have been resolved.
- **Confusion over MoE Frankenmodels**: `@kquant` sought assistance with their entries to the open LLM leaderboard. They'd submitted two entries – one mistakenly labeled as an adapter – and requested help from admins to remove the incorrect entries and keep only the correct, 'original' entry. They had not slept for several days and apologized for any inconvenience their errors might've caused.

**Links mentioned**:

- [Diffusers Gallery - a Hugging Face Space by huggingface-projects](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)
- [solarc-moe-10.7bx4.Q6_K.gguf · TheBloke/SOLARC-MOE-10.7Bx4-GGUF at main](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF/blob/main/solarc-moe-10.7bx4.Q6_K.gguf)
- [Kquant03/CognitiveFusion-4x7B-bf16-MoE · Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE)
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Meet](https://meet.google.com/dmn-wvxn-wpr): Real-time meetings by Google. Using your browser, ...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (8 messages🔥): 
        
- **@neuralink progresses on end-to-end FP8 training**: Stated they implemented 19% of end-to-end FP8 training, indicative of some interesting progress in their work with 3D parallelism.
- **@duplaja discovers SpeechT5 nuances while optimizing**: Shared updates on their work with SpeechT5, focusing on creating a custom handler and troubleshooting issues with numerals and larger strings pagination. They found using multiple instances on the lower AWS GPU T4 more cost-effective and shared their working handler.py [here](https://huggingface.co/Dupaja/speecht5_tts/blob/main/handler.py).
- **@farlin9000 relearns ML basics via Luis Serrano**: Shared a [YouTube video](https://www.youtube.com/watch?v=BR9h47Jtqyw) by Luis Serrano on Deep Learning with Neural Networks as they review ML basics. Farlin9000 initially encountered confusion on activation functions and probabilities but later understood the accounts of truth classification.


**Links mentioned**:

[A friendly introduction to Deep Learning and Neural Networks](https://www.youtube.com/watch?v=BR9h47Jtqyw): A friendly introduction to neural networks and dee...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (23 messages🔥): 
        
- **Exploring Transformers from Scratch**: User `@torres8552` shared a [Kaggle notebook](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook) on exploring and building the Transformer architecture for language-translation tasks from scratch using PyTorch, trained on the OpusBook dataset.
- **Shoe vs Sandal vs Boot Image Dataset**: `@andysingal` introduced a new image [dataset](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages) which contains 15,000 images of shoes, sandals and boots. Ideal for performing multiclass classification with deep neural networks like CNNs.
- **Illustration with resnet-50 on Shoe vs Sandal vs Boot image Dataset**: `@andysingal` presented a [notebook](https://github.com/andysingal/PyTorch-ML/blob/main/notebooks/resnet-50.ipynb) using resnet-50 on Shoe vs Sandal vs Boot image dataset.
- **Introduction of Augmentoolkit**: `@heralax` created Augmentoolkit, a fully-local [dataset generation tool](https://github.com/e-p-armstrong/augmentoolkit) powered by LLM. It turns plaintext into multi-turn conversations that can finetune instruct-tuned models.
- **Using Augmentoolkit on Different Datasets**: `@andysingal` expressed interest in applying Augmentoolkit on an Instruction-based dataset like the one at [Kaggle](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset). `@heralax` explained that it could be done by modifying a couple of cells in the notebook, but the code would differ based on the dataset structure.


**Links mentioned**:

- [Transformer From Scratch With PyTorch&#x1F525;](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook): Explore and run machine learning code with Kaggle ...
- [Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset): Can you use NLP to answer these questions?
- [llama_index/examples/paul_graham_essay/data/paul_graham_essay.txt at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/examples/paul_graham_essay/data/paul_graham_essay.txt): LlamaIndex (formerly GPT Index) is a data framewor...
- [Andyrasika/ShoeSandalBootimages · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)
- [PyTorch-ML/notebooks/resnet-50.ipynb at main · andysingal/PyTorch-ML](https://github.com/andysingal/PyTorch-ML/blob/main/notebooks/resnet-50.ipynb): Contribute to andysingal/PyTorch-ML development by...


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (6 messages): 
        
- **Live Participation vs Async Discussions**: `@swyxio` inquired about the format of discussions, needing a heads up for live events. `@lunarflu` clarified that the discussions are **usually async and text-only**, due to the global nature of the community. 
- **Blogpost Discussion Suggestion**: `@lunarflu` suggested having **discussions under each blogpost**, similar to the format for papers, acknowledging that this feature is currently unavailable.
- **Weekly Event for Paper Discussion**: Following the discussion format query, `@lunarflu` proposed creating a **weekly event for discussing papers**, including start times and range.
- **Call-to-action for Personal Presentations**: `@lunarflu` encouraged members to put together presentations for discussions, offering to create server-wide events once a date is provided.
- **Presentation Schedule Confirmation**: In response to `@lunarflu`'s call, `@dhruvdh` committed to preparing a presentation by **Friday**.


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (5 messages): 
        
- **Concerns About Opening Images in Datasets**: `@xcykim_56659` asked about how to open the image content in datasets and get the image data from an ImageFolder PIL object for a pretrained CVT model. Later, `@xcykim_56659` resolved their own inquiry and reported success. 
- **FPS Computation Query in Object Detection Leaderboard**: `@anasuna` expressed doubts about the frames per second (fps) computation on the [Object Detection Leaderboard](https://huggingface.co/spaces/hf-vision/object_detection_leaderboard), indicating the numbers seemed too low.
- **Training CV Model on Continuous Values**: `@tony_assi` expressed interest in resources for training a Computer Vision (CV) model using images paired with continuous numerical values, rather than discrete labels.

**Links mentioned**:

[Open Object Detection Leaderboard - a Hugging Face Space by hf-vision](https://huggingface.co/spaces/hf-vision/object_detection_leaderboard)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **Common Crawl's web indexing explained**: `@exponentialxp` asked how web data is collected for Common Crawl, and `@cakiki` explained that the process involves **powerful computers, a list of URLs, and software referred to as a 'spider'** for the crawling and indexing of these sites, similar to functions performed by search engines like Google and Bing.
- **Invitation to explore Common Crawl's codebase**: `@cakiki` provided a link to the [Common Crawl codebase](https://github.com/commoncrawl) on GitHub for `@exponentialxp` to explore if they're interested.

**Links mentioned**:

[Common Crawl Foundation](https://github.com/commoncrawl): Common Crawl provides an archive of webpages going...


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Prompts Tango with Mistral-7B**: `@cognitivetech` ponders over [two](https://discord.com/channels/1234/1212/12121) [ways](https://discord.com/channels/1234/1212/12121) of system prompts using **Mistral-7b**, with speed and quality consistency as looming challenges.
- **Deciphering Ooba's Enigma**: `@cognitivetech` shared a [template from Ooba](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml) but found it confusing.
- **Taking the AI Lab Home**: `@quantumpioneer.` queried hardware prerequisites for a local AI lab setup for [experiments](https://discord.com/channels/1234/1212/12121).
- **Hit or ReTrain**: `@maxdipper` probed ways to lean on a previously trained uncensored model for additional training as a cost-effective alternative to retraining from scratch.
- **Data Mining with Mixtral/Mistral**: `@unknownperson2156` sought feedback on user experiences using Mixtral or Mistral to extract predefined question data using LLMs.
- **Big Dreams with Mistral 8x7B**: `@mysterious2078` was on the hunt for documents or papers about the **Mistral 8x7B model**.
- **Unshackling the Local Runway**: `@michaelwechner` shared success running Mistral 7B locally on Mac M1 and in the cloud using [Ollama](https://github.com/jmorganca/ollama) and [Scaleway](https://www.scaleway.com/en/mac-mini-m2-pro/).
- **Tackling Virtual Limitations**: `@Idellarus` detailed his struggle to run a model on a restricted virtual desktop environment, as confirmed practical by `@duck`.
- **vLLM vs TGI, A Mixtral Story**: `@andersruge` inquired about the ramifications of vLLM and TGI on performance metrics, answered succinctly by `@casper_ai`.
- **Nano-Chatbots for All**: `@daain` gave a rundown of options for deploying real-time chatbots on limited resources, including APIs and smaller models like Phi-2 or TinyLlama-1.1B-Chat-v1.0.
- **GPU Hunting Season**: `@comcyber_12802` asked for GPU specs for finetuning Mistral 7B and got RTX 3090 recommended by `@le_mess` with a training time approximation of 1 hour.
- **Mistral, The Open Source Mystery**: `@darshansharma_` clarified that **Mistral** is indeed open source, with `@refik0727` validating the fact.
- **AGI's Imminent Arrival?**: `@poltronsuperstar` sparked the challenge by predicting the advent of AGI in weeks to month with the observe-built-nurture system marking the era of no-code AI but clarified the *"absolute genius"* of the eventual model.
- **The Quest to Define AGI**: User `@.tanuj.` invited the community to share their interpretations of **Artificial General Intelligence (AGI)**; indeed a challenge worth undertaking.

**Mistral Channel Summaries**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (61 messages🔥🔥): 
        
- **Exploring System Prompts with Mistral-7b**: `@cognitivetech` sought advice on system prompts using Mistral-7b, experimenting between two formats with varying success [`#1`](https://discord.com/channels/1234/1212/12121) and [`#2`](https://discord.com/channels/1234/1212/12121). Speed and quality consistency seemed to be issues when modifying the prompts. 
- **Template from Ooba for Implementing Prompts**: `@cognitivetech` shared [Ooba's template for implementing prompts](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml), though found it confusing [`#1`](https://discord.com/channels/@cognitivetech/1212/12121). 
- **Hardware for Local AI Experiments**: `@quantumpioneer.` inquired about hardware specifications and power requirements for a PC setup, intended for running local AI experiments [`#1`](https://discord.com/channels/1234/1212/12121). 
- **Additional Training after Uncensored Model:** `@maxdipper` asked if there would be cheaper way to add additional content training on top of an uncensored model, comparing it to training an uncensored model from scratch [`#1`](https://discord.com/channels/1234/1212/12121). 
- **Lead Collection with Mixtral or Mistral**: `@unknownperson2156` asked for user experiences using Mixtral or Mistral, specifically for data or information collection, i.e., predefined question data as a conversation with Long Language Models (LLM) [`#1`](https://discord.com/channels/1234/1212/12121).

**Links mentioned**:

- [mistralai (Mistral AI_)](https://huggingface.co/mistralai)
- [app.py · openskyml/mixtral-46.7b-chat at main](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py)
- [‎Riff Runner: Heavy Metal](https://apps.apple.com/us/app/riff-runner-heavy-metal/id6468704254): ‎Unleash the power of heavy metal in Riff Runner, ...
- [Riff Runner Metal (Pre-Release - Apps on Google Play](https://play.google.com/store/apps/details?id=app.titangen.games.ga008b)


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (2 messages): 
        
- **Interest in Edge Computing**: `@kagevazquez` expressed enthusiasm towards edge computing, stating, "*Nope but edge computing sounds awesome*".
- **Inquiry about Mistral 8x7B Documentation**: `@mysterious2078` sought for any available documents or papers about the **Mistral 8x7B model**.


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (34 messages🔥): 
        
- **Running Large Language Models Locally**: `@michaelwechner` shared his experience using [Ollama](https://github.com/jmorganca/ollama) to run Mistral 7B locally on a Mac M1 and in the cloud through [Scaleway](https://www.scaleway.com/en/mac-mini-m2-pro/) using an Apple Mac mini M2 Pro. The discussion also extended to whether Ollama and other similar tools are wrappers of llama.cpp. 
- **Deployment Restrictions on Virtual Desktops**: `@kartik.07` discussed his challenge of running a model locally on a virtual desktop, where he couldn't install new software or third-party tools. `@duck` confirmed that running inference would require some type of software, which might not be possible with such restrictions.
- **Comparing vLLM and TGI for Mixtral**: On `@andersruge`'s query about performance benchmarks between vLLM and TGI, `@casper_ai` highlighted that vLLM is generally faster as it prioritizes optimization, whereas TGI is mainly focused on reducing time to first token. 
- **Scaling Down for Real-Time Chatbot Applications**: `@daain` suggested options for deploying real-time chatbots with limited resources, such as using APIs, choosing smaller models like Phi-2 or TinyLlama-1.1B-Chat-v1.0, or utilizing NVidia Jetson Nano.

**Links mentioned**:

- [GitHub - jmorganca/ollama: Get up and running with Llama 2 and other large language models locally](https://github.com/jmorganca/ollama): Get up and running with Llama 2 and other large la...
- [Run Mistral 7B using Ollama on a Mac M2 16GB at Scaleway](https://medium.com/@michael.wechner/run-mistral-7b-using-ollama-on-a-mac-m2-16gb-at-scaleway-d640a4bd2158): I recently installed Mistral 7B using Ollama on my...


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (5 messages): 
        
- **Request for GPU Recommendation**: `@comcyber_12802` asked for the minimum GPU requirements for finetuning Mistral 7B for a dataset of about 5000 question-answer pairs. `@le_mess` suggested using an RTX 3090, mentioning it could train the stated dataset in approximately 1 hour, and offered assistance via private messages.
- **Taking Time to Learn**: Post GPU recommendation, `@comcyber_12802` stated intentions of investing more time in better understanding the agents like RAG, QLoRA, Axolotl, Peft before proceeding, appreciating `@le_mess`'s assistance.
- **Unrelated Conversation**: `@akshay_1` commented on an unspecified source by saying it's equivalent to telling someone to "google it," to which `@duck` apologized if it came off as offensive.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (13 messages🔥): 
        
- **Poltronsuperstar's Take on No-Code AGI Platform**: User `@poltronsuperstar` suggested a **no-code platform powered by Language Learning Models (LLMs)** with multiple types of agents; a generalist agent to overarch various specialist agents. The focus becomes having smart high-level decision-making, not singularly on implementation.
- **Inter-Agent Communication & Contextual Data Storage**: `@poltronsuperstar` elucidated that agents should **communicate directly and via shared context**. Files were suggested as ideal tools for storing high varying data, emphasizing the efficiency of a filesystem, facets, and history in a slightly repurposed git repo.
- **AGI Around the Corner?**: In a daring prediction, `@poltronsuperstar` predicted the advent of **Artificial General Intelligence (AGI) in a matter of weeks to months**. Citing GPT-4 level LLMs as a possible ceiling, the timeline was admitted to be somewhat intuition-dependent.
- **AGI: Simple but Genius**: While AGI is predicted to be rather simple to explain (akin to GAN), `@poltronsuperstar` disclaimed that the explanation's simplicity doesn't take away from the eventual model being "*absolute genius*".
- **Defining AGI**: User `@.tanuj.` posed an important question: "*How do y’all define AGI?*", seeking to understand the variety of definitions held by the chat community.


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (6 messages): 
        
- **Question about Mistral's Open Source status**: `@darshansharma_` asked if **Mistral** is open source, to which `@refik0727` confirmed that it is.
- **Open Discussion Initiated**: `@lerela` encouraged open question asking on the channel.
- **Request for MISTRAL_API_KEY**: `@carloszela` mentioned that he is adding a java library into langchain4j for mistral-ai and sought a **MISTRAL_API_KEY** demo. 
- **Medium's Performance Inquiry**: `@_definitely_not_sam_` asked if other users have also experienced slow performance on Medium, but no responses were noted.


        

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **LAION's Child-Pornography Contamination Dilemma**: `@chad_in_the_house` brought up a [Stanford paper](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s) uncovering child porn in LAION datasets that provoked an imperative discussion regarding responsibility and dataset sanitation. Further debate on disclosure norms by `@progamergov`, `@.undeleted`, and `@peacekeeper8310` raised possible anti-FOSS AI agenda and corporate regulatory capture motivations.
- **Decoding the LAION Conundrum**: Amid rising concerns over LAION's controversial datasets, `@thejonasbrothers` and `@chad_in_the_house` discussed possible mitigation approaches, therapy dilemma between total eradication and acceptable-degree reduction, and the issue's influence on legality perceptions of crawling and storing likely contaminated data.
- **Dissecting SISR's Noise Challenge**: `@vrus0188` pointed to a [research paper](https://arxiv.org/abs/2312.17526) outlining how intrinsic early-training noise of deep-learning-based Single Image Super-Resolution (SISR) complicates obtaining optimal results.
- **Innovations for Refined Image Generation**: HandRefiner and ElasticDiffusion, shared by `@vrus0188`, introduce strategies for refining malformed digital-hand renderings and training-free arbitrary-size image generation, respectively. URLs: [HandRefiner](https://github.com/wenquanlu/HandRefiner) and [ElasticDiffusion](https://github.com/MoayedHajiAli/ElasticDiffusion-official).
- **Advancements in Boundary Modelling and Document Reasoning**: `@thejonasbrothers` highlighted a [differentiable model](https://arxiv.org/abs/2401.00935) that uses boundary attention to excel in modelling image boundaries and a new [DocLLM approach](https://arxiv.org/abs/2401.00908) that merges bounding-box information with spatial layout structure to refine document understanding.
- **Robotics Inspired by Curiosity**: `@vrus0188` highlighted a [YouTube video](https://www.youtube.com/watch?v=Nnpm-rJfFjQ) showcasing how robots can be developed to embody the element of curiosity.

**LAION Channel Summaries**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (110 messages🔥🔥): 
        
- **LAION in Hot Waters Due to Unsavory Content**: `@chad_in_the_house` discussed a recent [Stanford paper](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s) that discovered child pornography in LAION datasets, forcing LAION to take them down. The community expressed concern over this issue and discussed alternatives like using Common Crawl.
  
- **Debate Over Responsible Disclosure and Impact**: Users `@progamergov`, `@.undeleted`, and `@peacekeeper8310` evaluated the Stanford researchers' approach, with some stating that revealing the issue without allowing LAION to mitigate it first could be construed as reckless and not aligned with responsible disclosure norms in the security world. In addition, they pointed out the possibility of an anti-FOSS AI agenda and corpos looking for regulatory capture.

- **Rethinking the Strategy – More Due Diligence?**: `@thejonasbrothers` and `@chad_in_the_house` debated potential solutions to the problem, acknowledging the mutable nature of illegal images and the impossibility of a 100% uncontaminated dataset. They argued for a middle ground approach - potentially legalizing datasets if due diligence has been made to remove Not-Safe-For-Work (NSFW) content.

- **Intricacies of Content Responsibility**: User `@thejonasbrothers` pointed out that the responsibility must ultimately lie with those hosting illicit content, not with LAION for containing potentially 'harmful strings'. Yet, the ongoing dilemma raises questions about the legality of crawling, saving, and possibly distributing potentially contaminated data.

- **Hard Questions on Purging Troublesome Data**: In light of the recent issues with LAION databases, users `@chad_in_the_house` and `@thejonasbrothers` navigate the complexities of removing all problematic content. They concede that total eradication might be impossible, but reducing it to an acceptable degree could become the next best move. However, the paper exposing the issue in LAION datasets could inadvertently provide a roadmap for locating illicit content on the internet, complicating the matter further.

**Links mentioned**:

- [Electronic Tip Form | FBI](https://tips.fbi.gov/home)
- [nvidia/parakeet-rnnt-1.1b · Hugging Face](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
- [Another Hit Piece on Open-Source AI](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s): Stanford researchers find problematic content in L...


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (8 messages🔥): 
        
- **Noise Obstacles in Image Super-Resolution Optimisation**:
    - `@vrus0188` introduced a [research paper](https://arxiv.org/abs/2312.17526) highlighting the challenges posed by inherent noise during early training steps in deep-learning-based **Single Image Super-Resolution (SISR)**. The study underscores the need for further scrutiny into the ill-posed nature of SISR processes.
    
- **HandRefiner aims to improve Image Generation**:
    - A GitHub repository named [HandRefiner](https://github.com/wenquanlu/HandRefiner) was shared by `@vrus0188`. This project presents a method—Diffusion-based Conditional Inpainting—for refining malformed hands in generated images.
  
- **ElasticDiffusion offers Training-free Image Generation**:
    - `@vrus0188` introduces [ElasticDiffusion](https://github.com/MoayedHajiAli/ElasticDiffusion-official) from GitHub repository, offering a novel **PyTorch implementation** for training-free arbitrary size image generation.
    
- **Differentiable Model Architecture to Improve Image Boundaries**:
    - `@thejonasbrothers` toted a [study](https://arxiv.org/abs/2401.00935) exhibiting a **differentiable model** employing boundary attention, which can exceptionally model boundaries while offering superior resistance to noise, sub-pixel precision and the adaptability to handle images at their native resolutions.
    
- **DocLLM: Innovative Approach to Visual Document Reasoning**:
    - `@thejonasbrothers` discusses a [paper](https://arxiv.org/abs/2401.00908) that presents **DocLLM**—a lightweight extension to traditional Large Language Models (LLM)—delegates attention only to bounding box information to integrate the spatial layout structure, thus, sidestepping expensive image encoders. Furthermore, it tailors a pre-training objective that helps to infill text segments. The poster also provided a direct [quote](https://discord.com/channels/782201995011817493/874797969064538123/912727896369774618) from the paper. 

- **Robot Development Inspired by Curiosity**:
    - A [YouTube video](https://www.youtube.com/watch?v=Nnpm-rJfFjQ) entitled "This Curious Robot Should Be Impossible!" was flagged by `@vrus0188`.

**Links mentioned**:

- [Boundary Attention: Learning to Find Faint Boundaries at Any Resolution](https://arxiv.org/abs/2401.00935): We present a differentiable model that explicitly ...
- [Noise-free Optimization in Early Training Steps for Image Super-Resolution](https://arxiv.org/abs/2312.17526): Recent deep-learning-based single image super-reso...
- [DocLLM: A layout-aware generative language model for multimodal document understanding](https://arxiv.org/abs/2401.00908): Enterprise documents such as forms, invoices, rece...
- [This Curious Robot Should Be Impossible!](https://www.youtube.com/watch?v=Nnpm-rJfFjQ): ❤️ Check out Weights &amp; Biases and sign up for ...
- [GitHub - wenquanlu/HandRefiner](https://github.com/wenquanlu/HandRefiner): Contribute to wenquanlu/HandRefiner development by...
- [GitHub - MoayedHajiAli/ElasticDiffusion-official: The official Pytorch Implementation for ElasticDiffusion: Training-free Arbitrary Size Image Generation](https://github.com/MoayedHajiAli/ElasticDiffusion-official): The official Pytorch Implementation for ElasticDif...


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Fine-tuning Limbo**: `@l_teto_l` queried if **fine-tuning LLAMMA 2** with Manticore datasets could yield better results, sparking an engaging discussion with multiple users chiming in with insights and linked resources.
- **Bug Hunt for Mixtral**: `@bratao` shared [a bug report](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942) pointing out some issues with **Mixtral finetuning**. Despite this, they observed that Mixtral instruct performed better even after applying suggested fixes.
- **Adventures in Attribution**: `@yamashi` initiated a debate on pinpointing the most influential tokens to an output, recommending backpropagation and input gradient analysis. Various users suggested tools like **ooba**.
- **Benchmark Bash**: `@yamashi` criticized benchmark tests like **medmcqa** and **pubmedqa** for incomplete words and skewed distributions, leading to discussions about better evaluation methods.
- **Bounty Hunting with Triton Kernels**: `@caseus_` announced a [$2400 bounty](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038) for improved speed and memory efficiency for Triton kernels for FFT.
- **Balancing the Act of Learning Rates**: `@nafnlaus00` discussed optimal learning rates, evaluative loss, and training loss, highlighting their impact on model performance and emphasizing maintaining balanced ratios.
- **The Dropout Debate**: `@nafnlaus00` shared their insights on ascertaining the most effective dropout rates and the ongoing processes of metaparameter tuning.
- **Hyperparam Magic with Axolotl**: `@giftedgummybee` piqued interest by mentioning the use of autohyperparam tuning in Axolotl.
- **Skipping Workflows for Merging Multiple PRs**: `@caseus_` proposed using `[skip ci]` tags for merging multiple PRs in succession to reduce workflow runs, pointing to [GitHub documentation](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs) for the concept.
- **Unraveling Grouped GEMM and Grouped Experts**: `@caseus_` and `@casper_ai` deep-dived into the link between Grouped GEMM and grouped experts, also sharing a comparative [GitHub link](https://github.com/imoneoi/openchat/compare/master...moe).
- **Tackling Non-English Fine-tuning**: `@muhammad_ichsan` discussed challenges in fine-tuning **Mistral** for non-English languages (Indonesian), prompting advice from members like `@nanobitz`on tokenizer enlargement and text instruction.
- **Navigating Large Model Trainings on Multiple GPUs**: `@b_ryan0` sought strategies for training large models (like codellama 34b) across multiple GPUs. `@noobmaster29` suggested a solution using `zero3` and micro-batching.
- **Solving Non-GPU Development for Axolotl**: `@kcaverly` enquired about a feasible non-GPU development setup for Axolotl's CLI, leading `@noobmaster29` to suggest affordable rental options on runpod.
- **Boosting Non-English Performance**: `@noobmaster29` shared [an academic paper](https://arxiv.org/pdf/2401.01055.pdf) for improving non-English performance in models like Mistral.
- **Praying for Shearing Mistral Code**: `@dangfutures` requested sharing of the shearing mistral code once it's figured out.
- **Quest for Quantifying Token Effect**: `@nosa_.` recommended testing whether increasing token quantity could improve the capabilities of **Sheared-LLaMA** using extensive datasets like **SlimPajama**.
- **Legal Compass for Non-Copyright Content Use**: `@dctanner` instigated a discussion regarding the use of non-license restricted content to avoid any legal consequences, especially after recent copyright cases.
- **Casting Doubt on Bluemoon Quality**: `@xzuyn` warned against the sole use of **bluemoon** due to lower content quality and advocated for an assorted book dataset within copyright limits.

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (42 messages🔥): 
        
- **Fine-tuning Dilemma**: `@l_teto_l` asked if **finetuning LLAMMA 2** with the datasets used for Manticore would yield great results. This sparked a discussion where various users chimed in with their insights and shared relevant links.
- **Mixtral Finetune Bugs**: `@bratao` shared [a bug report about Mixtral finetuning](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942) but added that the **Mixtral instruct** still performed better even after applying certain fixes.
- **Tokens Contribution Analysis**: `@yamashi` sparked an interesting conversation about figuring out which tokens contribute most to the output, suggesting backpropagation and looking for the gradient for each token in the input. Other users like `@nanobitz` mentioned tools like **ooba** which might provide this feature.
- **Criticisms on Benchmarks**: `@yamashi` expressed frustration at the apparent shortcomings of benchmarks like **medmcqa** and **pubmedqa**, stating that they sometimes didn't provide complete words and often had skewed distribution, prompting a need for closer assessment.
- **Bounty for Optimizing Triton Kernels**: `@caseus_` made an announcement about a [$2400 bounty for optimized Triton kernels for FFT](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038), looking for improvements on speed and memory efficiency.


**Links mentioned**:

- [CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models](https://arxiv.org/abs/2312.04350): The ability to perform causal reasoning is widely ...
- [Question · Issue #6 · pratyushasharma/laser](https://github.com/pratyushasharma/laser/issues/6#issuecomment-1874828714): Hi, Thanks for releasing this code. Does this code...
- [ Incorrect implementation of auxiliary loss  · Issue #28255 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942): System Info transformers version: 4.37.0.dev0 Plat...
- [[BOUNTY] Optimized Triton Kernels for full fine tunes · Issue #1038 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038): 🔖 Feature description We&#39;ve seen marketing fr...
- [HellaSwag or HellaBad? 36% of this popular LLM benchmark contains errors](https://www.surgehq.ai/blog/hellaswag-or-hellabad-36-of-this-popular-llm-benchmark-contains-errors): We analyzed HellaSwag, a popular LLM benchmark, an...
- [Fix load balancing loss func for mixtral by liangxuZhang · Pull Request #28256 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28256): What does this PR do?   Fixes #28255 Before submit...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (10 messages🔥): 
        
- **Balancing Learning Rates and Loss Ratios**: `@nafnlaus00` discussed the relationship between learning rates (LR), evaluative loss, and training loss, advising to watch for their ratios as they affect model performance. They noted: "*Depends on your LR. Watch the ratio between eval loss and train loss, aka how focused it is on memorizing the training data.*" They also mentioned that the ideal divergence between evaluative and training loss should not exceed 5-10%.
- **Determining Ideal Dropout Rates**: `@nafnlaus00` shared insights on optimal dropout rates stating, "*I had been using 0.25 dropout but I think lower is probably better.  But I think higher than 0.07 is probably best.*" They acknowledged still being in the process of metaparameter tuning to find the best dropout and LR for their case.
- **Autohyperparam Tuning in Axolotl**: `@giftedgummybee` made a comment about using autohyperparam tuning in Axolotl, provoking curiosity among the community members.
- **Skipping Workflow Runs while Merging Multiple PRs**: `@caseus_` suggested using `[skip ci]` tags while merging multiple PRs in a row to reduce workflow runs. They shared a link ([Skipping workflow runs - GitHub Docs](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs)) for the same from GitHub docs.
- **Grouped Experts and MOE**: `@caseus_` and `@casper_ai` discussed the relationship between Grouped GEMM and grouped experts, with the latter saying, "*Grouped GEMM = grouped experts as far as I can see*". `@caseus_` also highlighted a comparison link ([Comparing master...moe · imoneoi/openchat](https://github.com/imoneoi/openchat/compare/master...moe)) on GitHub to further exemplify.

**Links mentioned**:

- [Skipping workflow runs - GitHub Docs](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs)
- [Comparing master...moe · imoneoi/openchat](https://github.com/imoneoi/openchat/compare/master...moe): OpenChat: Advancing Open-source Language Models wi...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (50 messages🔥): 
        
- **Struggling with Non-English Fine-tuning**: User `@muhammad_ichsan` expressed difficulty with fine-tuning **Mistral** on Indonesian Wikipedia dataset, citing stagnant training loss. `@nanobitz` advised him to increase tokens in the tokenizer, feed the model a lot of tokens, and then instruction tune. `@noobmaster29` also suggested mixing in English during the Full-Fine-Tuning (FFT), given `@muhammad_ichsan`'s report of catastrophic forgetting with English queries. [Link to Wikpedia Dataset](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id)
- **Mistral Vicuna1.1 Formatting**: `@le_mess` shared a chat template they created for **Vicuna1.1**, with `@nanobitz` suggesting to add `\n` when making it single line. 
- **Training Large Models Across GPUS**: `@b_ryan0` inquired about a recipe for training large models like codellama 34b across multiple GPUs, and `@noobmaster29` provided a solution using `zero3` and micro-batching.
- **Non-GPU Development for Axolotl**: `@kcaverly` asked about a GPU-poor development setup for the CLI of Axolotl, to which `@noobmaster29` suggested renting on runpod for affordability.
- **Improving Non-English Performance**: `@noobmaster29` shared an academic paper (https://arxiv.org/pdf/2401.01055.pdf) that might be helpful for those seeking to improve non-English language performance of models like Mistral. 


**Links mentioned**:

[wikimedia/wikipedia at main](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id)


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (7 messages): 
        
- **Request for Shearing Mistral Code**: `@dangfutures` requested that the code for shearing mistral be shared once figured out.
- **Hypothesis on Token Quantity**: `@nosa_.` suggested that it would be interesting to test the hypothesis that increasing token investment could further improve the capability of **Sheared-LLaMA**.
- **Debate on Data Adequacy**: In the context of testing the above hypothesis, `@nosa_.` and `@xzuyn` agreed that **SlimPajama** might offer a large enough set of data for testing.
- **Discussion on Non-Copyright Content Use**: `@dctanner` raised concerns about using non-license restricted content for continued pre-training to avoid potential legal issues, particularly considering the recent developments in the NYTimes case.
- **Quality Concerns about Bluemoon Dataset**: `@xzuyn` advised against the singular use of **bluemoon** due to possible content quality issues and recommended gathering a book dataset that wouldn't pose any copyright challenges.


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Hungry for a Spanish Interface**: User `@juaniespeche` voiced the need for a **Spanish UI** for Perplexity, pointing out that the AI can already respond accurately in Spanish.
- **Perplexity Pricing Puzzles**: `@archient` requested clarification on **Perplexity's token pricing** when utilizing multiple models. `@icelavaman` and `@ok.alex` clarified that Perplexity operates under a **prepaid credits system**, with the total cost being the cumulative amount for each model based on processed tokens.
- **Craving Direct Model Communication**: `@saltrockr` queried about the possibility of interacting with models directly without internet searches. `@reflext` suggested using Perplexity's **writing mode** for this purpose.
- **Unexpected Hiccups in Trial Period Payment**: `@ava12138` and `@boredkarma` conversed about difficulties in validating payments for the 7-day Perplexity Pro trial, observing inconsistencies in the acceptance of different card types.
- **Striking UI Similarities between Phind and Perplexity**: `@neuralspace` and `@reflext` discussed the noticeable similarities between the UIs of Phind and Perplexity. `@reflext` argued that such resemblances are inevitable given the central search-bar design convention.
- **Gratitude for Perplexity AI's Help**: `@hei_veno` gave positive feedback about how Perplexity AI has significantly aided in developing training content, although the specifics couldn't be shared due to confidentiality. `@aontoni` and `@whiterickruben` also shared their experiences with Perplexity AI assisting in a university project and an exam prep, respectively.
- **Showcasing Perplexity AI's Profile through an Article and a Video**: `@nayka3473` provided a link to an article they wrote about Perplexity and other AI chat platforms, as well as a [YouTube video](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh) titled: "Ranking top AI Chat Platforms: Phind, ChatGPT, Claude, Gemini Pro, Poe and more!".
- **Pondering Perplexity App's Roles**: `@archient` posed an interesting question about the correlation between a profile in the Perplexity app and a system role in the API.
- **Call for a Solar 10.7b Model**: `@arcinarci` suggested the inclusion of a "*solar 10.7b model*" in the Perplexity spectrum.

**Perplexity AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (65 messages🔥🔥): 
        
- **Spanish User Interface Needed**: User `@juaniespeche` expressed a desire for a **Spanish interface** in Perplexity, noting that the AI already responds effectively in Spanish. 
- **API Pricing Clarification**: `@archient` inquired about **Perplexity's token pricing** when using multiple models. `@icelavaman` explained that the total cost would be a sum of the costs for each model based on tokens processed. Further inquiries about usage billing led `@icelavaman` and `@ok.alex` to clarify that Perplexity operates via a **prepaid credits system**.
- **Direct Conversations with Models**: `@saltrockr` asked for a way to query models directly without internet searches involved. `@reflext` suggested the use of the **writing mode** in Perplexity.
- **Payment Issues for Trial Period**: `@ava12138` and `@boredkarma` discussed issues in payment validation methods for the 7-day trial of Perplexity Pro, noting inconsistencies in which cards are accepted. 
- **UI Similarities between Phind and Perplexity**: `@neuralspace` and `@reflext` discussed the similarities between the user interfaces of Phind and Perplexity. `@reflext` stated that such similarities are inevitable given the central search-bar design type.

**Links mentioned**:

- [Perplexity - AI Companion](https://chrome.google.com/webstore/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo): Ask anything while you browse
- [Perplexity - AI Search](https://chrome.google.com/webstore/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol): Upgrade your default search engine
- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started)
- [Perplexity - AI Search](https://chromewebstore.google.com/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol?pli=1): Upgrade your default search engine


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (5 messages): 
        
- **User Feedback on Perplexity AI**: User `@hei_veno` mentioned that Perplexity AI helped a lot in developing training content, although the detailed information could not be shared due to work-related confidence.
- **Resource Recommendation**: `@aontoni` shared a [link](https://www.perplexity.ai/search/is-it-recommended-4cG8AoJaSnWId74QGXo7Cg?s=u) that they found helpful, but didn't specify further details.
- **Perplexity AI Assists with MS Access**: `@aontoni` later stated how Perplexity AI helped them understand the relationship between a form and a query in MS access for a university project.
- **Perplexity AI useful for Exam Help**: User `@whiterickruben` mentioned that Perplexity AI helped them assist a friend with an upcoming exam.
- **Article on AI Chat Platforms Including Perplexity**: `@nayka3473` wrote an article about Perplexity and other AI chat platforms, which they shared via this [link](https://medium.com/towards-artificial-intelligence/aichatplatforms-7be703c1f21d?sk=98b1f2335efa58013585aa64c2ebc29a). They also shared a [YouTube video](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh) titled: "Ranking top AI Chat Platforms: Phind, ChatGPT, Claude, Gemini Pro, Poe and more!" and asked for feedback.

**Links mentioned**:

- [The Rise of AI: comprehensive list of top AI Chat Platforms](https://medium.com/towards-artificial-intelligence/aichatplatforms-7be703c1f21d?sk=98b1f2335efa58013585aa64c2ebc29a): Top AI Chat Platforms of 2023
- [Ranking top AI Chat Platforms: Phind, ChatGPT, Claude, Gemini Pro, Poe and more!](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh): Discover our top-ranked AI Chat Platforms of 2023,...


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (2 messages): 
        
- **Question about Perplexity App's Profile vs API System Role**: `@archient` asked, "*Is the profile in the perplexity app the same as a system role in the API?*".
- **Request for Solar 10.7b Model**: `@arcinarci` inquired about the possibility of having a "*solar 10.7b model*".


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Missing img2img Functionality in ChatGPT**: `_typedef` inquired about an img2img model, `@solbus` clarified that **ChatGPT does not currently support direct img2img functionality**. However, DALL·E developers in an AMA hinted at future "image references" potentially introducing img2img feature. [AMA Link](https://discord.com/channels/974519864045756446/1173674915153592320/1174040158111273070)
- **Ease of API Integration With Actions**: `@iamhere6321` complimented the **ease of use and effectiveness of Actions** in connecting to an external API. In contradiction, `@niko3757` preferred more flexibility and the ability to create new threads.
- **Concerns with Decreasing Gpt4 Efficiency**: Seeing a decline in gpt4's efficiency, `@caesarrzk` asked for recommendations to improve this, `@my5042` suggested using custom gpt and "you are chatgpt" instruction for better output.
- **ChatGPT Performance and Signup Issues**: `@wolf.lover` expressed issues with **ChatGPT's lagging and errors**, `@zeromaid` faced **issues during the signup process**.
- **GPT4 Factual Accuracy Concerns**: `@wesego` raised concerns about GPT4's factual accuracy when generating text from an attached document, and `@niko3757` suggested using interconnected APIs or CI.
- **Teaching Immutable Syntax to ChatGPT**: `@facebreaker.` inquired on how to teach ChatGPT an **immutable fixed syntax or structure** for more specific and reproducible responses.
- **File Review with GPT's Assistance**: `@jferrari_75079` asked for assistance with a project where GPT reviews / summarizes files' content, and provides recommendations on action (delete, archive, or save).
- **CreatingTheLatest Investment Articles Without Advisories**: `@komal0887` asked for help refining a prompt for generating articles with only the latest investment information, specifically without any advice or evaluative sentences. They were using the **gpt-3.5-turbo-instruct model** for this task.
- **Chatbots Mimicking Conversation Styles**: `@emaanios` inquired about chatbots that could mimic a provided conversation style for their language generation bot research.

**OpenAI Channel Summaries**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (13 messages🔥): 
        
- **No Direct img2img Feature in ChatGPT Yet**: `@_typedef` inquired if the model for txt2img is the same for img2img. `@solbus` clarified that currently, **ChatGPT does not have a direct img2img functionality**. It recognizes an uploaded image (img2txt), which can then be used to generate a similar image in a subsequent txt2img step. However, Solbus referenced an AMA where **DALL·E developers hinted at potential future "image references"**, which could introduce some form of img2img feature. The [AMA's link](https://discord.com/channels/974519864045756446/1173674915153592320/1174040158111273070) was shared but may require archive access to view. 
- **Image to Image - A General Query**: `@_typedef` later clarified that their previous question about img2img functionality was general and not specifically related to OpenAI.
- **An URL without context**: `@jaicraft` shared a [URL](https://g.co/bard/share/cfec5f03f662) without any preceding or succeeding context.
- **Digital Exhaustion**: User `@mad_cat__` expressed fatigue and found it hard to navigate discord rooms. However, they also mentioned their excitement about their work.

**Links mentioned**:

[‎Steve Jobs Unveils Siri Chat](https://g.co/bard/share/cfec5f03f662): Created with Bard.


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (20 messages🔥): 
        
- **OpenAI Actions Ease of Use**: `@iamhere6321` complimented the **ease of configuration** and effectiveness of using **Actions** to connect to an **external API**, calling it a promising approach. `@niko3757` shared an alternate perspective, preferring assistants that have more flexibility and can create new threads. 
- **Signup Issues Encountered**: User `@zeromaid` reported issues with the signup process on the platform, receiving a message that **"Signup is currently unavailable, please try again later."** They reiterated the problem, indicating they were unable to sign up. 
- **ChatGPT Performance Issues**: `@wolf.lover` reported **performance issues with ChatGPT**, indicating it had become laggy and was causing errors in Firefox. They expressed concern about needing to switch chats despite having spent a significant amount of time on the current one.
- **Advantages of Using Assistants**: In a discussion with `@iamhere6321`, `@niko3757` listed several advantages of using Assistants over custom GPTs. These include unlimited actions, the ability to package multiple actions into one, triggering new threads, and increased knowledge embedding into the model among other perks. Despite highlighting these advantages, `@niko3757` also noted that these features come with a cost.
- **Seeking Assistance with GPT4 Accuracy**: `@wesego` asked if anyone had success getting GPT4 to write text while accurately adhering to factual information in an attached document. They noted discrepancies in the AI's generated story and the factual accuracy based on their experience. `@niko3757` suggested moving away from CustomGPT and trying interconnected APIs, potentially also involving Continuous Integration (CI).
- **Challenges with Imposing Fixed Syntax and Structure**: `@facebreaker.` sought guidance on how to teach **ChatGPT an immutable fixed syntax/structure**. They experienced problems with changing syntax and quality reduction over time, and hoped to make the model's responses reproducible and specific to their needs. 
- **Issues After Switching User-Agent**: `@vova5963` joked about being blocked by Mouser after frequently switching their **User-Agent**, noting that this allowed them to watch YouTube without being blocked.


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (12 messages🔥): 
        
- **Refining Prompts for Article Generation**: User `@komal0887` asks for help refining a prompt that generates articles based on text extracted from different URLs. The generated articles should only contain the latest information and not include investment advice, call-to-action, or evaluative sentences. The user uses **gpt-3.5-turbo-instruct model**.

- **Issues with Lazy gpt4**: `@caesarzzk` expressed concerns about gpt4 seeming to get lazier over time, omitting output code or analysis when possible and sometimes even struggling with comprehension. `@my5042` suggests using instructions like "you are chatgpt" in custom gpt for better result.

- **Building an Accurate Story**: `@wesego` asked for guidance on how to write an accurate story. 

- **Questions on System Prompts**: `@itsnp` asked if they could pose their queries about **system prompts** in the channel.

- **Chatbot Mimicking Conversation Style**: `@emaanios` inquired if any chatbot exists that can mimic the style of conversation from provided chat logs for their research in language generation bots. 

- **Help Requested with File Management using GPT**: `@jferrari_75079` asked for assistance with a project in which GPT would examine each file, subfolder, and image and advise on whether to delete, archive, or save it. The task also includes GPT providing a short summary of the file's content. The user reported that their earlier attempts resulted in GPT making decisions based on superficial aspects like the file's last modified date.


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (12 messages🔥): 
        
- **Refining Investment Article Prompts**: `@komal0887` expressed need for assistance with refining the prompt given to the `gpt-3.5-turbo-instruct model` for generating articles from text extracted from different URLs related to financial updates. They want the output to contain only latest information and not advise or evaluative sentences.
- **Increasing Efficiency of Gpt4**: `@caesarzzk` noted Gpt4 being increasingly less efficient and asked for recommendations to improve this predicament. `@my5042` suggested using custom gpt and adding the instruction "you are chatgpt" for a better output.
- **Recursive Checker for Conciseness and Thoroughness**: In response to an undefined problem, `@madame_architect` proposed a solution involving a recursive checker skill to ensure the right balance between comprehensiveness and conciseness in writing.
- **Chatbots Mimicking Conversation Styles**: `@emaanios` asked about chatbots specifically designed to mimic conversation styles based on provided chat logs and `@beanz_and_rice` confirmed their existence.
- **Help with GPT Reviewing Files**: `@jferrari_75079` sought help with GPT thoroughly examining files to decide whether to delete, archive, or save them based on their content. They also wanted GPT to provide a short summary of each file's content. It was noted that GPT was previously basing its decisions on superficial aspects like file's last modified date.


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **The DPO is All About Distribution**: @gabriel_syme drew focus to how **Differential Privacy Offsetting (DPO)** is more related to distribution than samples.
- **The Lion Roars in Optimizing**: @marthinwurer shed light on the functions of the **lion optimizer**, emphasizing that it doesn't allow large loss spikes due to its fixed change in weights every step.
- **Image Captioner Hunt**: @frazermc is on the look-out for a nimble **image captioner** to run through 500k images, indicating a preference for non-LM-augmented options. He shared an [Awesome-Multimodal-Large-Language-Models repository](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) for reference.
- **Caught in the Mixture of Experts**: @michaelmelons queried if anyone experimented with **Mixture of Experts (MoE)** with experts of varying parameter sizes including simple and complex architecture experts.
- **Transformers Learn Algorithms and the Collaboration Proposal**: @stellaathena proposed a collaboration around a study named [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028) and the mysteries of compositional capabilities of transformers.
- **Pythia-70m Stumbles** : @micpie reported a drastic underperformance by the **Pythia-70m** model in a benchmark test, noting an accuracy drop to **0.002**. The insightful @hailey_schoelkopf proposed that floating point precision with *fp16* auto dtype could be behind this and adjusting to `float32` could rectify the issue.


**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (18 messages🔥): 
        
- **Lion Optimizer Prevents Large Loss Spikes**: `@marthinwurer` observed the practical benefits of using the **lion optimizer**, specifically that there are no large loss spikes, as the weights only change a fixed amount each step, not a multiple of the gradient.
- **LLM Flipping Response Logic**: `@sk5544` sought community input regarding a paper or research that might explain why a **Large Language Model (LLM)** flips its response when asked "Are you sure?".
- **Seeking Efficient Image Captioner**: `@frazermc` shared about needing an **image captioner** to process 500k images, ideally not an LM augmented one. They shared a [GitHub repository on Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) for reference. 
- **Efficient Sequence Shifting in Huggingface Datasets**: `@.the_alt_man` shared code based on using **Huggingface datasets** to _shift sequence_, but noted that the overhead of `torch -> list -> jax.Array` is too heavy and asked if there's a better way to accomplish this preprocessing natively in Huggingface. 
- **Running lm-evaluation-harness in Google Colab**: `@lee0099` asked if it's possible to run **lm-evaluation-harness** in Google Colab, `@hailey_schoelkopf` confirmed it's possible and shared a [guideline on GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb) on how to do so.
- **Implementation of a Data-Controlled Forget Gate in LSTM**: `@sentialx` queried on how to implement a **data-controlled forget gate** in an LSTM, `@wonkothesensible` suggested looking at **rwkv** for inspiration.
- **Praise for Pythia LLM Analysis**: `@swyxio` acknowledged and highlighted the work done by the **Pythia team**, sharing a [Twitter thread by @rasbt](https://fxtwitter.com/rasbt/status/1734920232173539796) that extols Pythia's comprehensive analysis of Large Language Models.


**Links mentioned**:

- [lm-evaluation-harness/examples/lm-eval-overview.ipynb at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb): A framework for few-shot evaluation of autoregress...
- [Tweet from Sebastian Raschka (@rasbt)](https://fxtwitter.com/rasbt/status/1734920232173539796): I am reviewing my favorite papers of the year, and...
- [GitHub - BradyFU/Awesome-Multimodal-Large-Language-Models at Evaluation](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation): :sparkles::sparkles:Latest Papers and Datasets on ...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (17 messages🔥): 
        
- **DPO Distribution Focus**: `@gabriel_syme` noted that the connection with Differential Privacy Offsetting (DPO) focuses more on **distribution** rather than samples.
- **Theorem 5.4 Discussion**: `@salmon_lemon` expressed confusion regarding *Theorem 5.4*. `@sumo43` provided some insights, suggesting that by successfully optimizing the generator, its output would become similar to the data, and explained lambda as a learning rate parameter.
- **Concept Erasure for Image Models**: `@voxs` inquired if anyone has done **concept erasure for image models** and later said they found some relevant resources.
- **Mobile ALOHA Imitation Learning System**: `@ai_waifu` posted a link to [Mobile ALOHA](https://mobile-aloha.github.io/resources/mobile-aloha.pdf), a low-cost, whole-body teleoperation system developed for imitating mobile manipulation tasks in robotics. `@thatspysaspy` admired the demo and queried about its robustness, while `@ai_waifu` discussed cost-efficiency and claimed that mass production could bring down the cost significantly.
- **Mixture of Experts with Variable Parameter Sizes**: `@michaelmelons` asked if anyone had attempted **MoE (Mixture of Experts)** at scale with experts of varying parameter sizes, including simple and more complex architecture experts.

**Links mentioned**:

[Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation](https://mobile-aloha.github.io/): by Zipeng Fu*, Tony Z. Zhao* and Chelsea Finn at S...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (2 messages): 
        
- **Collaboration Proposal on Transformer Algorithms**: User `@stellaathena` discussed the possibility of a collaboration with the lead author of [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028). They explored the topic of **compositionality** and **information theoretical complexity of tasks as expressed in RASP(-L)**, and expressed interest in understanding why transformers don't achieve perfect generalization. 
- **Positive Response to Collaboration**: User `@dashiell_s` expressed interest in joining the proposed collaboration.


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (15 messages🔥): 
        
- **Pythia-70m drastically underperforms in tests**: User `@micpie` noticed that the **Pythia-70m** model underperformed on a benchmark test, resulting in an accuracy of **0.002** instead of the previous result of **0.609** [See message](https://discord.com/channels/824728003853975583/967159777337462845/969868847195512896).
- **Floating point precision might be the issue**: `@hailey_schoelkopf` suggested that the issue could be due to the model running in fp16 using the auto dtype in HF. By adjusting the dtype to `float32`, the test returned more reasonable results [See message](https://discord.com/channels/824728003853975583/967159777337462845/969868919384489032).
- **More Pythia models affected**: The issue seemed specific to the v1 **Pythia** models and more prevalent in smaller models. According to `@hailey_schoelkopf`, enabling torch autocast could potentially help [See message](https://discord.com/channels/824728003853975583/967159777337462845/969868974884458564).
- **Difficulty loading local datasets**: `@micpie` experienced a problem loading local datasets with JSON format. `@hailey_schoelkopf` suggested using `dataset_path: json` and `dataset_kwargs: { data_dir: /path/to/benchmark_0-2 }` as a temporary solution, but noted that they will make changes to restore the original functionality [See message](https://discord.com/channels/824728003853975583/967159777337462845/969872582121836656). 
- **Pending changes to restore original functionality**: Despite the suggested workaround for loading local datasets, `@micpie` chosen to wait for changes to be implemented so they won't have to adjust their approximately 400 config files [See message](https://discord.com/channels/824728003853975583/967159777337462845/969874635933319178).


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Healing Tokens with Ayenem's Project**: `@ayenem` [unveiled a project named TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main), which trims and regrows prompts to harmonize with a model's tokenizer. This increases model completion and its resilience to trailing whitespaces/punctuation. More context on the issue TokenHealer addresses can be found in [this article](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69).
- **API Barrier for MidJourney**: `@kevmodrome` sought to know if **MidJourney** can be used via APIs other than Discord. `@jevonm` clarified that it is currently Discord-exclusive.
- **Seeking An AI for Audio Analysis**: `@zf0` was curious about a chat model capable of audio analysis instead of mere video frames. `@swyxio` suggested exploring riffusion style approaches or Meta's Seamless models.
- **Coqui's Closure Echoes in the AI Community**: `@swyxio` disseminated the [news of Coqui's shutdown](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww). Coqui was an open-source speech technology organization.
- **GPT-4 Summarizes AI/ML Papers**: `@intheclouddan` spotlighted a [tool on emergentmind.com](https://www.emergentmind.com/) that employs GPT-4 to summarize AI/ML papers.
- **InsightPilot to be Discussed at LLM Paper Club**: `@swyxio` and `@eugeneyan` announced a discussion on **InsightPilot** at the [upcoming LLM Paper Club](https://lu.ma/llm-paper-club). InsightPilot is an LLM-power automated data exploration system.
- **Mixture of Experts (MoEs) on the Horizon**: For the upcoming week, the LLM Paper Club, as informed by `@swyxio`, will discuss a paper on 'Mixture of Experts', a buzzing topic in the open AI community. The link to the blog post is [here](https://huggingface.co/blog/moe).
- **Noting Down the LLM Paper Club**: `@swyxio` emphasized the need for note-taking during the paper club sessions and invited suggestions for discord notetaking bot tools.

**Latent Space Channel Summaries**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (17 messages🔥): 
        
- **TokenHealer released by Ayenem**: User `@ayenem` [introduced TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main), a project that trims and regrows prompts to align with a model's tokenizer. This improves model completion and its robustness to trailing whitespaces/punctuation. A [related blog post](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69) was also shared to provide more context on the problem TokenHealer is solving.
- **MidJourney platform query**: User `@kevmodrome` asked if **MidJourney** could be used via any API apart from Discord. `@jevonm` replied that currently it is only accessible via Discord.
- **Query about chat model for audio analysis**: `@zf0` inquired about a chat model that can analyze audio instead of just video frames. `@swyxio` suggested looking into a "riffusion style approach" or Meta's Seamless models.
- **Shutdown of Coqui announced**: `@swyxio` shared the [news of Coqui's shutdown](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww), an open-source speech technology organization. 
- **New tool for summarizing AI/ML papers**: `@intheclouddan` brought attention to a [tool on emergentmind.com](https://www.emergentmind.com/) that uses GPT-4 to summarize AI/ML papers.


**Links mentioned**:

- [Tweet from Josh Meyer 🐸💬 (@_josh_meyer_)](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww): Coqui is shutting down.  It&#39;s sad news to star...
- [Tweet from Sam (@Sam_Awrabi)](https://fxtwitter.com/Sam_Awrabi/status/1742324900034150646?s=20): 1. AI funding mostly sits in the model layer for n...
- [AI/ML Research, Explained | Emergent Mind](https://www.emergentmind.com/): Stay informed about important new AI/ML arXiv rese...
- [GitHub - Ayenem/TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main): Contribute to Ayenem/TokenHealer development by cr...
- [The Art of Prompt Design: Prompt Boundaries and Token Healing](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69): Learn how standard greedy tokenization introduces ...


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **InsightPilot discussion with a leading force**: `<@187636841988620288>` will guide a discussion on **InsightPilot** (copilots for data analysis) [here](https://lu.ma/llm-paper-club).
- **LLM Paper Club**: This event is a weekly paper review of LLM papers, with focus on **big ideas**, their **relevance**, and any **open-ended questions** after reading.
- **No Upcoming Sessions Yet**: The series presently has no upcoming sessions but advises regular check-back for updated schedules.
- **Matrix for Paper Selection**: The paper for review is decided a week ahead, with details shared in the `#llm-paper-club` channel.
- **Tag In for Discord Notifications**: Users are encouraged to request to be tagged in `<@&1107197669547442196>` for discord notifications related to the meet-up.


**Links mentioned**:

[LLM Paper Club (now in Discord) · Luma](https://lu.ma/llm-paper-club): A weekly paper review of LLM papers, starting from...


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (14 messages🔥): 
        
- **InsightPilot: LLM-Empowered Automated Data Exploration System**: `@swyxio` shared details of today's paper on InsightPilot, an LLM-based, automated data exploration system designed to streamline the data exploration process. The paper can be found at [this link](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/).
- **Join the InsightPilot Discussion**: `@eugeneyan` invites members to join the discussion about the LLM to analyze data through [this Discord link](https://discord.gg/zuZp95ya).
- **Next in Line: Mixture of Experts (MoEs)**: `@swyxio` provides the link for next week's paper on 'Mixture of Experts', a hot topic in the open AI community. The link to the blog post is [here](https://huggingface.co/blog/moe).
- **Future Paper Consideration: Self-Play Fine-Tuning (SPIN)**: `@swizec` suggests considering a paper on Self-Play Fine-Tuning (SPIN) for a future discussion. The proposed paper can be found at [this link](https://arxiv.org/abs/2401.01335).
- **Note-Taking for the Paper Club**: `@swyxio` expressed a need for good note-taking during the paper club sessions and is seeking suggestions for discord notetaking bot tools.

**Links mentioned**:

- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335): Harnessing the power of human-annotated data throu...
- [Join the /dev/invest + Latent Space Discord Server!](https://discord.gg/zuZp95ya): Check out the /dev/invest + Latent Space community...
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [InsightPilot: An LLM-Empowered Automated Data Exploration System - Microsoft Research](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/): Exploring data is crucial in data analysis, as it ...


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **LLM eyed to rephrase Anki cards**: @thebaghdaddy shared a curious interest in using **LLM** to rephrase Anki cards for better *information generalizability* in the `#collaboration` channel.
- **Exploring a Multi-agent System for smoother storytelling**: @yikesawjeez proposed setting up a multi-agent system, including an 'orchestrator', a 'state manager' and a small model trained in play-by-post material, to **manage narrative creation**.
- **Steering Plot with 'Objectives'**: @yikesawjeez further suggested the inclusion of an 'Objectives' section checked by a 'DM' in the system, that can help *steer the plot* in the intended direction.
- **Aim to break the AI Narrative Loop**: @yikesawjeez pinpointed a common issue with AI content generation -- *repetitive narrative loops*. Solution suggested: Altering both player's and model's texts to disrupt the loop.
- **Long-context Models to assist narrative management**: @yikesawjeez believes that long-context models managing narratives could potentially benefit from plot unfolding examples for precise *few-shot directions*.
- **Search + Search RAG API out for beta-testing**: @emrgnt_cmplxty in the `#rag` channel announced the release of a new Search + Search RAG API, inviting eager contributors for a *beta test* and user application feedback. This model is also **open-sourced**.
- **Community interest in New API**: @yikesawjeez showed keenness to check out this new API and requested a **link**.


**LLM Perf Enthusiasts AI Channel Summaries**

### ▷ #[collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018/) (5 messages): 
        
- **Using LLM to Rephrase Content**: `@thebaghdaddy` expressed interest in utilizing **LLM** to rephrase Anki cards with an objective to improve information generalizability.
- **Multi-agent System for Narrative Creation**: `@yikesawjeez` detailed their idea to operate a multi-agent system to manage narrative creation. The system proposed included, an 'orchestrator', a 'state manager' and a small model trained in play-by-post material, collaborating to **compress narrative information into manageable sections**. 
- **Objective-driven Narrative Management**: `@yikesawjeez` also mentioned the possibility of having an extra 'Objectives' section checked by a 'DM' to steer the plot in a specific direction.
- **Avoiding Narrative Loops**: `@yikesawjeez` underscored the challenge of navigating AI-generated narrative loops, where similar responses can trigger repetitive text. They suggested modifying player's messages and the model's to break the loop.
- **Long-context Models for Narrative Management**: `@yikesawjeez` proposed that long-context models managing narrative can benefit from examples of how a plot might unfold, facilitating targeted few-shot directions.


### ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/) (3 messages): 
        
- **New Search + Search RAG API for Beta Testing**: `@emrgnt_cmplxty` announced the release of a new Search + Search RAG API and asked the community if they could do a quick **beta test** and provide feedback, specifically if it would be useful for their applications.
- **Open Source Model**: `@emrgnt_cmplxty` mentioned that the model behind this newly introduced API is **open sourced**.
- **Request for Link to New API**: User `@yikesawjeez` showed interest and asked for a **link** to this new API.


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **GPT-4 Turbo vs GPT-4 Comparison**: User `@philipmay` asked for a judgement on the performance of **GPT-4 turbo (gpt-4-1106-preview)** compared to regular **GPT-4**.
- **Turbo Excels in Conversations**: `_jp1_` noted that **GPT-4 Turbo** may even be better than **GPT-4** for "convenience prompts" or normal dialogues and tasks involving long contexts, based on personal impressions. 
- **Turbo Struggles with Complex Tasks**: However, `_jp1_` also mentioned that **GPT-4 Turbo** seems to underperform when faced with *complex instructions*, such as a series of custom tasks in a specific order.
- **Coding Contexts Prove Challenging**: `@mister_poodle` expressed that in the context of coding, **GPT-4 Turbo** often struggles to implement the full code, even when explicitly instructed; this issue is less frequent with **GPT-4** unless when dealing with long context lengths. 
- **Overall Performance of GPT-4**: `@mister_poodle` observed a perceived degradation in the performance of both **GPT-4 Turbo** and **GPT-4** since their respective launches.

        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Warm Welcome to a Physics Pundit**: New member `@ddt1909` aka **Daniel** shared his experience in **ML/Computer Vision** and his current project on information extraction using LLMs for enterprises, influenced by a podcast recommendation to join the server. 
- **Phi-Tuning Falls Flat**: `@benxh` described having had **mostly negative experiences** with phi-tuning, warning the community about the struggles with this model adjustment parameter.
- **Hugging Face Models: Less Than a Hug More of a Thud**: `@benxh` found that the **fine-tuned models available on Hugging Face are lackluster**, indicating potential unidentified issues, creating a deeper conversation about quality control and expectations around pre-trained models.

**Alignment Lab AI Channel Summaries**

### ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/) (1 messages): 
        
- **New member introduction**: `@ddt1909` introduced himself as **Daniel**, who has a physics background and has been working in **ML/Computer Vision** since 2017. He's currently building an information extraction product based on LLMs for the enterprise. His decision to join the server was influenced by `@660097403046723594`’s recommendation on a podcast.


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (3 messages): 
        
- **Negative Experiences with Phi-Tuning**: User `@benxh` expressed dissatisfaction with the phi-tuning, as they've had **mostly negative experiences**.
- **Lackluster Fine-Tuned Models on Hugging Face**: `@benxh` also points out that **the fine-tuned models present on Hugging Face are lackluster** and there seems to be an unidentified issue with them.


        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

Only 1 channel had activity, so no need to summarize...

- **Finding Resources on Analytical Databases**: User `@pranay01` expressed an interest in learning about *state of the art in analytical databases/large scale analytical systems* and asked for suggestions on whom to follow, noting their appreciation for user `<@1016864328189759488>`.
- **Resource Recommendation from Expert**: User `@andypavlo` pointed `@pranay01` to an upcoming course on this exact topic and provided a [link to the course's page](https://15721.courses.cs.cmu.edu/spring2024/).
- **Accessibility for Non-CMU Folks**: `@pranay01` followed up by asking if there was a previous version of this course that they could access, and whether non-Carnegie Mellon University students could enroll in these courses.

**Links mentioned**:

[CMU 15-445 :: Advanced Database Systems (Spring 2024)](https://15721.courses.cs.cmu.edu/spring2024/): Carnegie Mellon University

        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **NEJM Image Challenge Dataset Now Accessible**: `onuralp.` shared the NEJM Image Challenge dataset in [GitHub](https://github.com/cx0/nejm-image-challenge), noting there's no need for data cleaning for users with existing models. Plans to share **gpt4v results** this week were hinted at, with any suggestions for model tweaks or other amendments welcomed.

**Skunkworks AI Channel Summaries**

### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=O6RPmtuGKMM


### ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 
        
- **NEJM Image Challenge dataset shared**: `onuralp.` made the dataset for the NEJM Image Challenge available on [GitHub](https://github.com/cx0/nejm-image-challenge), and mentioned that there is no need for data cleaning for users who already have their model in place. He also mentioned his plans to upload the **gpt4v results** this week, and welcomed any suggestions for model changes or other modifications.

**Links mentioned**:

[GitHub - cx0/nejm-image-challenge: NEJM Image Challenge dataset and experiments](https://github.com/cx0/nejm-image-challenge): NEJM Image Challenge dataset and experiments. Cont...


        

---
The Datasette/LLM (@SimonW) Discord has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
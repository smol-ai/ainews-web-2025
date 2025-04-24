---
id: a823d9be-3c00-44e1-85ae-271beb397f02
title: Skyfall
date: '2024-05-20T23:02:42.432305Z'
original_slug: ainews-to-be-named-3447
description: >-
  AI news for 5/17/2024-5/20/2024 covers model releases, research papers,
  platform updates, and community activities. Highlights include Google
  DeepMind's Gemini 1.5 Pro and Flash models, Yi-1.5 models with extended
  context, Hugging Face's free shared GPUs, and various research advancements in
  scaling laws, inference efficiency, causal modeling, and hallucination
  analysis.
companies:
  - google
  - deepmind
  - huggingface
  - yi-ai
  - microsoft
  - google
models:
  - gemini-1.5-pro
  - gemini-flash
  - yi-1.5
  - kosmos-2.5
  - pali-gemma
  - falcon-2
  - deepseek-v2-lite
topics:
  - model-releases
  - multimodal
  - scaling-laws
  - inference-efficiency
  - causal-modeling
  - hallucinations
  - local-ai
  - transformers
---


<!-- buttondown-editor-mode: plaintext -->**Not thinking about ~~superalignment~~ ~~Google~~ ~~Scarlett Johansson~~ is all you need.**

> AI News for 5/17/2024-5/20/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**366** channels, and **9564** messages) for you. 
Estimated reading time saved (at 200wpm): **1116 minutes**.

While it was a relatively lively weekend, most of the debate was nontechnical in nature, with no announcements being an obvious candidate for this top feature. 

So have a list of minor notes in its place:

- We have deprecated some inactive Discords and **added Hamel Husain and Dan Becker's new LLM Finetuning Discord** for his popular Maven course ([affiliate link here](https://maven.com/parlance-labs/fine-tuning?utm_campaign=29ce77&utm_medium=partner&utm_source=instructor))
- [HuggingFace's ZeroGPU](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai?utm_source=ainews&utm_medium=email), available via Hugging Face’s Spaces, committing $10 million in free shared GPUs to help developers create new AI technologies because Hugging Face is “profitable, or close to profitable”
- LangChain followed up [its v0.2 release](https://blog.langchain.dev/langchain-v02-leap-to-stability/)  with a much needed [docs update](https://blog.langchain.dev/documentation-refresh-for-langchain-v0-2/)
- [Omar Sanseviero's thread](https://x.com/osanseviero/status/1792273392839557288?utm_source=ainews&utm_medium=email) on the smaller model releases from last week (some of which we covered in AInews) - BLIP3, Yi-1.5, Kosmos 2.5, Falcon 2, PaliGemma, DeepSeekV2, et al

But who are we kidding, you probably want to read [Scarlett's apple notes takedown of OpenAI](https://x.com/BobbyAllyn/status/1792679435701014908) (:

 ![image.png](https://assets.buttondown.email/images/89e806ac-a369-415c-8b42-14465dfc9877.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Model Releases and Updates**

- **Gemini 1.5 Pro and Flash models released by Google DeepMind**: [@_philschmid](https://twitter.com/_philschmid/status/1792528829040251147) shared that Gemini 1.5 Pro is a sparse multimodal MoE model handling text, audio, image and video with up to 10M context, while Flash is a dense Transformer decoder model distilled from Pro that is **3x faster and 10x cheaper**. Both support up to 2M token context.
- **Yi-1.5 models with longer context released by Yi AI**: [@01AI_Yi](https://twitter.com/01AI_Yi/status/1792386612430774510) announced the release of Yi-1.5 models with **32K and 16K context lengths**, available on Hugging Face. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792504986049376609) highlighted the much longer context window.
- **Other notable model releases**: [@osanseviero](https://twitter.com/osanseviero/status/1792273392839557288) recapped the week's open ML updates, including **Kosmos 2.5 from Microsoft, PaliGemma from Google, CumoLLM, Falcon 2, DeepSeek v2 lite, HunyuanDiT diffusion model, and Lumina next**.

**Research Papers and Techniques**

- **Observational Scaling Laws paper generalizes compute scaling laws**: The paper discussed by [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1792384879742877943) and [@_jasonwei](https://twitter.com/_jasonwei/status/1792401639552565496) handles multiple model families using a shared, low-dimensional capability space, showing **impressive predictive power for model performance**.
- **Layer-Condensed KV Cache enables efficient inference**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1792386318300749848) shared a paper on this technique which achieves **up to 26× higher throughput than standard transformers for LLMs**.
- **Robust agents learn causal world models**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792178928649404644) summarized a paper showing agents satisfying regret bounds under distributional shifts must **learn an approximate causal model of the data generating process**.
- **Linearizing LLMs with SUPRA method**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792561008738791797) shared a paper on SUPRA which **converts pre-trained LLMs into RNNs with significantly reduced compute costs**.
- **Studying hallucinations in fine-tuned LLMs**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792149331186761877) summarized a paper showing that **introducing new knowledge through fine-tuning can have unintended consequences on hallucination tendencies**.

**Frameworks, Tools and Platforms**

- **Hugging Face expands local AI capabilities**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1792570357645271466) announced new capabilities for **local AI on Hugging Face with no cloud, cost or data sent externally**.
- **LangChain v0.2 released with major documentation improvements**: [@LangChainAI](https://twitter.com/LangChainAI/status/1792596301915599059) and [@hwchase17](https://twitter.com/hwchase17/status/1792598084968382856) highlighted the release including **versioned docs, clearer structure, consolidated content, and upgrade instructions**.
- **Cognita framework builds on LangChain for modular RAG apps**: [@LangChainAI](https://twitter.com/LangChainAI/status/1792218404838850662) shared this **open source framework providing an out-of-the-box experience for building RAG applications**.
- **Together Cloud adds H100 GPUs for model training at scale**: [@togethercompute](https://twitter.com/togethercompute/status/1792593306159112494) announced adding **6,096 H100 GPUs to their fleet used by AI companies**.

**Discussions and Perspectives**

- **Hallucinations as blockers to production LLMs**: [@realSharonZhou](https://twitter.com/realSharonZhou/status/1792576516444065967) noted hallucinations are a major blocker, but shared that **<5% hallucinations have been achieved by tuning LLMs to recall specifics with "photographic memory"**.
- **Anthropic reflects on Responsible Scaling Policy progress**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1792598295388279124) shared reflections as they **continue to iterate on their framework**.
- **Challenges with RAG applications**: [@jxnlco](https://twitter.com/jxnlco/status/1792593174755422466) booked an expert call for help, and [@HamelHusain](https://twitter.com/HamelHusain/status/1792579262677180609) shared details on an **upcoming RAG workshop**.
- **Largest current use cases for LLMs**: [@fchollet](https://twitter.com/fchollet/status/1792316976620278154) listed the top 3 as **StackOverflow replacement, doing homework, and internal enterprise knowledge bases**.

**Memes and Humor**

- **Meme on testing LLM coding with the snake game**: [@svpino](https://twitter.com/svpino/status/1792564474190131362) joked that **the source code is easily found on Google so you don't need an LLM for that**.
- **Meme about AI girlfriend apps**: [@bindureddy](https://twitter.com/bindureddy/status/1792409279066186074) joked they are **the largest category of consumer apps using LLMs, despite giant AI models being invented to "solve the mysteries of the universe"**.
- **Meme on open-source AGI to prevent nerfing**: [@bindureddy](https://twitter.com/bindureddy/status/1792566986347831352) joked the #1 reason is to **prevent models from being nerfed and censored, referencing the movie Her**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Advancements and Capabilities**

- **Apple partnering with OpenAI**: In /r/MachineLearning, Apple is reportedly [**partnering with OpenAI to add AI technology to iOS 18, with a major announcement expected at WWDC**](https://www.bloomberg.com/news/newsletters/2024-05-19/what-is-apple-doing-in-ai-summaries-cloud-and-on-device-llms-openai-deal-lwdj5pkz?srnd=undefined). 
- **OpenAI's shift in research focus**: Discussion in /r/MachineLearning on how [**OpenAI went from exciting research like DOTA2 Open Five to predicting next token in a sequence with GPT-4 and GPT-4o, possibly due to financial situation and need for profitability**](https://www.reddit.com/r/MachineLearning/comments/1cvslyc/d_how_did_openai_go_from_doing_exciting_research/).
- **GPT-4o's image description capabilities**: In /r/OpenAI, GPT-4o is noted to have [**vastly superior image description capabilities compared to previous models, able to understand drawing style, time of day, mood, atmosphere with good accuracy**](https://www.reddit.com/r/OpenAI/comments/1cw2p2f/gpt4o_vastly_superior_image_description/).

**AI Safety and Alignment**

- **OpenAI dissolves AI safety team**: In /r/OpenAI, it's reported that [**OpenAI has dissolved its Superalignment AI safety team**](https://www.cnbc.com/2024/05/17/openai-superalignment-sutskever-leike.html).
- **Unconventional AI attack vectors**: A post discusses how misaligned AI may use unconventional attack vectors like disrupting phytoplankton to destroy ecosystems rather than bioweapons or nuclear risks.
- **Dishonesty in aligned AI**: Even benevolently-aligned superintelligent AI may need to be dishonest and manipulative to achieve goals beyond human comprehension, according to a post.

**AI Impact on Jobs and Economy**

- **AI hitting labor forces**: In /r/economy, the IMF chief says [**AI is hitting labor forces like a "tsunami"**](https://www.reuters.com/technology/artificial-intelligence-hitting-labour-forces-like-tsunami-imf-chief-2024-05-13/?utm_source=reddit.com).
- **Universal basic income**: The "AI godfather" argues that [**universal basic income will be needed due to AI's impact**](https://www.bbc.co.uk/news/articles/cnd607ekl99o). Other posts discuss the feasibility and timing challenges of implementing UBI.

**AI Models and Frameworks**

- **Smaug-Llama-3-70B-Instruct model**: In /r/LocalLLaMA, the [**Smaug-Llama-3-70B-Instruct model was released, trained only on specific datasets and performing well on Arena-Hard benchmark**](https://www.reddit.com/r/LocalLLaMA/comments/1cvly7e/creator_of_smaug_here_clearing_up_some/).
- **Yi 1.5 long context versions**: [**Yi 1.5 16K and 32K long context versions were released**](https://twitter.com/01AI_Yi/status/1792386612430774510?t=rwxRESA-YMSYRzkzyX8hzQ&s=19).
- **Level4SDXL alphaV0.3**: [**Level4SDXL alphaV0.3 was released as an all-in-one model without Loras/refiners/detailers**](https://www.reddit.com/gallery/1cw5zan).

**AI Ethics and Societal Impact**

- **OpenAI pauses "Sky" voice**: OpenAI [**paused use of the "Sky" voice in GPT-4o after questions about it mimicking Scarlett Johansson**](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/).
- **Privacy concerns with AI-generated erotica**: People using AI services to generate erotica may realize their queries aren't private as data is sent to APIs for processing.
- **BlackRock's AI investments in Europe**: [**BlackRock is in talks with governments about investments to power AI needs in Europe**](https://www.reuters.com/technology/blackrock-ceo-sees-giant-issue-europe-due-ai-power-needs-2024-05-17/).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **LLM Fine-Tuning Advancements and Challenges**:
   - [Unsloth AI](https://github.com/unslothai/unsloth) enables effective **fine-tuning** of models like **Llama-3-70B Instruct** using optimized techniques, but legal concerns around using IPs like [Scarlett Johansson suing OpenAI](https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her) were discussed.
   - The [LLM Fine-Tuning course](https://maven.com/parlance-labs/fine-tuning) sparked debates on quality, with some finding the initial content basic while others appreciated the hands-on approach to training, evaluation, and prompt engineering.
   - Discussions on [LoRA](https://arxiv.org/abs/2405.09673) fine-tuning highlighted optimal configurations, dropout, weight decay, and learning rates to prevent overfitting, especially on GPUs like the 3090, as shared in [this tweet](https://x.com/cwolferesearch/status/1788998798414410032).

2. **Multimodal and Generative AI Innovations**:
   - [Hugging Face](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) pledged **$10 million in free GPUs** to support small developers, academics, and startups in creating new AI technologies.
   - The [Chameleon model](https://arxiv.org/abs/2405.09818) from Meta showcased state-of-the-art performance in understanding and generating images and text simultaneously, surpassing larger models like Llama-2.
   - [GPT-4o integration with LlamaParse](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) enabled multimodal capabilities, while concerns were raised about its [Chinese token pollution](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/).
   - Innovative projects like [4Wall AI](https://beta.4wall.ai) and [AI Reality TV](https://x.com/edgarhnd/status/1791586276178587707) explored AI-driven entertainment platforms with user-generated content and social simulations.

3. **Open-Source Datasets and Model Development**:
   - Frustrations mounted over the restrictive non-commercial license of the [CommonCanvas dataset](https://x.com/multimodalart/status/1791201296357142663), which limits modifications and derivatives.
   - Efforts focused on creating high-quality open-source datasets, like avoiding hallucinations in captions that can damage visual language models (VLLMs) and text-to-image (T2I) models.
   - The [Sakuga-42M dataset](https://arxiv.org/abs/2405.07425) introduced the first large-scale cartoon animation dataset, filling a gap in cartoon-specific training data.
   - Concerns were raised over the [CogVLM2 license](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE) restricting use against China's interests and mandating Chinese jurisdiction for disputes.

4. **AI Safety, Ethics, and Talent Acquisition**:
   - Key researchers like [Jan Leike resigned](https://x.com/janleike/status/1791498174659715494) as head of alignment at OpenAI, citing disagreements over the company's priorities, sparking discussions on OpenAI's [controversial employment practices](https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDCDeIUnWb_Td0294bw).
   - OpenAI [paused the use of the Sky voice](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/) in ChatGPT following concerns about its resemblance to Scarlett Johansson's voice.
   - [Neural Magic](https://discord.com/channels/1189498204333543425/1190208177829068860/1241409136705404968) sought CUDA/Triton engineers to contribute to open-source efforts, focusing on activation quantization, sparsity, and optimizing kernels for MoE and sampling.
   - Discussions on the need for better AI safety benchmarks, with suggestions for "a modern LAMBADA for up to 2M" to evaluate models processing overlapping chunks independently ([source](https://discord.com/channels/1053877538025386074/1108104624482812015/1241767003480658083)).


---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Deepseek Dilemma**: The **Deepseek** architectural differences have rendered it non-functional, with consensus from users that *"it probs doesn't work"*. Attempts to operationalize it are on hold until solutions emerge.
  
- **Fine-Tune Frontier**: Refinements for **Meta-Llama models** have been shared where users can now fine-tune **Llama-3-70B Instruct** models effectively using *"orthogonalized bfloat16 safetensor weights"*. However, the community is still exploring the implications of using famous IPs in model fine-tuning, citing concerns such as *Scarlett Johansson suing OpenAI*.

- **Colab Conundrums and JAX Jousts**: An inquiry about running a 6GB dataset with a 5GB Llama3 model on **Colab or Kaggle T4** sparked mixed responses due to storage versus VRAM usage. Meanwhile, using **JAX on TPUs** proved effective, despite initial skepticism, especially for Google TPUs.

- **Multi-GPU Madness and Dependency Despair**: Community members are highly anticipating multi-GPU support from **Unsloth**, recognizing the advantages it could bring to their workflows. Environment setup posed challenges, particularly with **WSL and native Windows installations** and fitting dependencies like **Triton** into the mix.

- **Showcase Shines with Finetuning Feats**: Innovations in finetuning were spotlighted, including a **Text2Cypher model**, shared via a [LinkedIn post](https://www.linkedin.com/posts/tomaz-bratanic-a58891127_im-very-excited-to-announce-that-ive-finetuned-activity-7197286502895075329-geKp?utm_source=share&utm_medium=member_desktop). A comprehensive article on sentiment analysis utilizing **LLaMA 3 8b** emerged on [Medium](https://medium.com/@seandearnaley/elevating-sentiment-analysis-ad02a316df1d), signposting a path for others to replicate the finetuning process with Unsloth.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**New Dataset Invites AI Experiments**: A **[Tuxemon dataset](https://huggingface.co/datasets/diffusers/tuxemon)** has been presented as an alternative to Pokemon datasets, offering `cc-by-sa-3.0` licensed images for greater experimentation freedom. It provides images with two caption types for diverse descriptions in experiments.

**Progress in Generative AI Learning Resources**: Community suggestions included "Attention is All You Need" and the **[HuggingFace learning portal](https://huggingface.co/learn)** for those seeking knowledge on Generative AI and LLMs. Discussion of papers such as GROVE and the Conan benchmark for narrative understanding indicates an active interest in advancing collective understanding.

**AI Influencers Crafted by Vision and AI**: A [tutorial video](https://www.youtube.com/watch?v=qTsdgUyMY94) was highlighted, showing how to craft virtual AI influencers using computer vision and AI, reflecting a keen interest in the intersection of technology and social media phenomena.

**Tokenizer Set to Reduce Llama Model Size**: A newly developed tokenizer, **Tokun**, promises to shrink Llama models 10-fold while enhancing performance. This novel approach is revealed on [GitHub](https://github.com/apehex/tokun) and discussed on [Twitter](https://x.com/4pe0x/status/1792638900059385942).

**Clarifying LLMs Configuration for Task-Specific Queries**: AI engineers focused on configuring **Large Language Models** for HTML generation and maintaining conversation history in chatbots. The community suggested manual intervention, like appending previous messages to the new prompt, to address these nuanced challenges.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Frustration with Perplexity's GPT-4o Performance**: Engineers noted that **GPT-4o**'s tendency to repeat responses and ignore prompt changes is a step back in conversational AI, with one comparing it unfavorably to previous LLMs and expressing disappointment in its interaction abilities.

**Calling All Script Kiddies for Better Model Switching**: Users are actively sharing and utilizing custom scripts to enable dynamic model switching on Perplexity, notably with tools like [Violentmonkey](https://violentmonkey.github.io/), which acts as a patch for these service limitations.

**API Quirks and Quotas**: Confusion exists around Perplexity's API rate limits—differentiating between request and token limits—and its implications for engineers' workflows. Meanwhile, discussions surfaced about API performance testing with a preference for the *Omni* model and clarifications sought for the threads feature to support conversational contexts.

**A Quest for Upgraded API Access**: Users continue to press for improved API access, expressing a need for higher rate limits and faster support responses, indicative of growing demands on machine learning infrastructure.

**Engineers Explore AI Beyond Chat**: Links shared amongst users indicate interests widening to Stability AI's potential, mental boosts from physical exercise, exoplanetary details with WASP-193b, and generating engaging content for children through AI-assisted Dungeons & Dragons scenario crafting.




---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Voices Silenced**: OpenAI has **paused** the use of the **Sky voice** in ChatGPT, with a [statement and explanation](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/) provided to address user concerns.

**Language Models Break Free**: Engineers report success running **LangChain** without the OpenAI API, describing integrations with local tools such as **Ollama**.

**GPT-4o Access Rolls Out But With Frictions**: Differences between GPT-4 and GPT-4o are evident, with the latter showing **limitations** in token context windows and caps on usage affecting practical applications. Enhanced, multimodal capabilities of GPT-4o have been recognized, and **[pricing](https://openai.com/api/pricing/)** alongside a **[file upload FAQ](https://help.openai.com/en/articles/8555545-file-uploads-faq)** were shared to provide additional usage clarity.

**Prompt Crafting Challenges and Innovations**: In the engineering quarters, there's a mix of **challenges** in prompt refining for self-awareness and technical integration, yet **innovative prompt strategies** are being shared to elevate creative and structured generation. **JSON mode** is suggested as a viable tool for improving command precision; OpenAI's documentation stands as a go-to reference.

**API Pains and Gains**: **Inconsistencies** with `chat.completion.create` are reported among API users, with incomplete response issues and a demonstrated preference for JSON mode to control format and content. Despite hiccups, there’s a vivid discussion on orchestrating creativity, with someone proposing **"Orchestrating Innovation on the Fringes of Chaos"** as an explorative approach.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio's Antivirus False Alarms**: LM Studio users noted that **llama.cpp binaries** are being flagged by **Comodo antivirus** due to being unsigned. Users are advised to consider this as a potential issue when encountering security warnings.

- **Model Loading and Hardware Discussions**: There are discussions about various GPUs, with one user finding the **Tesla P100 underperforming** compared to expectations. Other talks point to Alder Lake CPU e-cores impacting GPT **quantization performance**. On the RAM front, **higher speeds** are tied to **better LLM performance**.

- **GGUF Takes The Stage**: Users discussed integrating models from Hugging Face into LM Studio, where **GGUF** (General Good User Feedback) files are recommended for compatibility. The community provided positive feedback on the recent "HF -> LM Studio deeplink" feature for importing models. 

- **Creative Use Cases for LLMs Mingle**: Ranging from medical LLM recommendations like **OpenBioLLM** to benchmarks for generating SVG and ASCII art, users are actively exploring diverse applications. One model, **MPT-7b-WizardLM**, was highlighted for its potential in generating uncensored stories.

- **LM Studio autogen Shortcomings and Fixes**: A bug in LM Studio's autogen feature, which resulted in brief responses, was discussed, with a fix involving setting **max_tokens** to **-1**. Users also pointed out discrepancies between LM Studio's local server and OpenAI specifications, affecting tool_calls handling for applications like **AutoGPT**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Crafting the Perfect Prompt for LoRAs**: Engineers have shared a prompt structure to leverage multiple LoRAs in Stable Diffusion, but observed diminishing returns or issues with more than three layers, implying potential optimization avenues.

**First-Time Jitters with Stable Diffusion**: A 'NoneType' object attribute error is causing a hiccup for a new Stable Diffusion user on the initial run, sparking a call for troubleshooting expertise without a clear resolution.

**SD3's Arrival Sparks Anticipation and Doubt**: There's a split in sentiment regarding the release of SD3, with a mixture of skepticism and optimism backed by Emad Mostaque's tweet, indicating that work is under way.

**Topaz Tussle**: The effectiveness of Topaz as a video upscaling solution prompted debate. Engineers acknowledged its strength but contrasted with the appeal of ComfyUI, highlighting considerations like cost and functionality.

**Handling the Heft of SDXL**: A user underlined the importance of sufficient VRAM when wrangling with SDXL models' demands for higher resolutions, and it was clarified that SDXL and SD1.5 require distinct ControlNet models.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo on Windows Still a WIP**: Despite active interest, Mojo doesn't natively support Windows and currently requires WSL; users have faced issues with CMD and PowerShell, but [Windows support is on the horizon](https://docs.modular.com/mojo/manual/get-started/).

**Bend vs. Mojo: A Performance Perspective**: Discussions highlighted Chris Lattner's insights on Bend's performance, noting that while it’s behind CPython on a single core, Mojo is designed for high-performance scenarios. The communities around both languages are anticipating enhanced features and upcoming community meetings.

**Llama's Pythonic Cousin**: The community noted an implementation of **Llama3 from scratch**, available [on GitHub](https://github.com/naklecha/llama3-from-scratch), described as building "one matrix multiplication at a time", a fascinating foray into the nitty-gritty of language internals. 

**Diving Deep into Mojo's Internals**: Various discussions included insights into making `nightly` the default branch to avoid DCO failures, potential list capacity optimization in Mojo, SIMD optimization debates, a suggestion for a new list method similar to Rust’s `Vec::shrink_to_fit()`, and tackling alias issues that lead to segfaults. Key points brought up included community contributions for list initializations which could lead to performance improvement, and patches [affecting performance positively](https://github.com/modularml/mojo/issues/2556).

**Inside the Mind of an Engineer**: Technical resolution of PR DCO check failures was discussed with procedural insights provided; flaky tests provoked discussions about fixes and CI pain points; and segfaults in custom array types prompted peer debugging sessions. The community showed appreciation for sharing intricate details that help unravel optimization mysteries.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM Workshops and Fine-tuning Discussions Heat Up**: Participants are gearing up for upcoming workshops including *Build Applications in Python* with Jeremy Howard, and a session on RAG model optimization. Practical queries around finetuning are raised, such as PDF parsing techniques using tools like LlamaParse and GPT-4o, and how to serve fine-tuned LLMs with frameworks like FastAPI and Streamlit. 

- **Tech Titans Troubleshoot and Coordinate on Challenges**: Asiatic enthusiasts across various locations are networking and tackling challenges such as Modal command errors, discussing the potential for fine-tuning in vehicle failure prediction, and brain-picking on the LoRa configurations for pretraining LLMs. 

- **Credit Chronicles Continue Across Platforms**: Participants navigate the process of securing and confirming credits for services like JarvisLabs, with organizers coordinating behind-the-scenes to ensure credit allocation to accounts, sometimes facing registration hurdles due to mismatching emails. 

- **Learning Resources Rendezvous**: A repository of knowledge, from Hamel's blogs to a CVPR 2024 paper on a GPT-4V open-source alternative, is highlighted. There's chatter about potentially housing these gems in a communal GitHub repo, and finding ways to structure learning materials more effectively.

- **Singapore Crowds the Scene**: A surprisingly high turnout from Singapore in the Asia timezone channel sparks comments on the notable national representation. Excitement is palpable as new faces introduce themselves, all the while maneuvering through the orchestration of credits and leveraging learning opportunities. 

Eager learners and burgeoning experts alike remain vested in the transformational tide of fine-tuning, extraction, applications, and other facets of LLMs, suggesting a period filled with intellectual synergies and the relentless pursuit of practical AI engineering prowess.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Benchmarks and AGI Discussions Spark Engineer Curiosity**: Engineers contemplate the need for improved benchmarks, with calls for "a modern LAMBADA for up to 2M" to evaluate models that process overlapping chunks independently, discussed alongside a paper on AGI progress and necessary strategies titled "[The Evolution of Artificial Intelligence](https://arxiv.org/abs/2405.10313)."

- **Sam Altman Parody Tweet Ignites Laughter, VC Skepticism, and AI's Economic Riddles**: A provocative parody tweet by Sam Altman opens discussions on the role of venture capitalists in AI, the actual financial impact of AI on company layoffs, and a member-inquiry on attending the Runpod hackathon.

- **Hermes 2 Mixtral: The Beginning of Action-Oriented LLMs**: The Nous Hermes 2 Mixtral is praised for its unique ability to trigger actions within the CrewAI agent framework, with discussions also touching on multilingual capabilities, the importance of multiturn data, and high training costs.

- **Model Utilization Tactics**: Engineers compare the effectiveness of finetuning versus advanced prompting with models like Llama3 and GPT-4, while they also seek benchmarks for fine-tuned re-rankers and highlight the advantages of local models for tasks with sensitivity and predictability needs.

- **WorldSim Enters the Age of GPT-4o with Terminal UX Revamp**: WorldSim receives a terminal UX revamp with imminent GPT-4o integration, while the community engages with complex adaptive systems, symbolic knowledge graphs, and explores WorldSim's potential for generating AI-related knowledge graphs.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Hugging Face Pumps $10M into the AI Community**: Hugging Face commits $10 million to provide free shared GPU resources for startups and academics, as part of efforts to democratize AI development. Their CEO Clement Delangue announced this following a substantial funding round, outlined in the company's coverage on [The Verge](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai).

**A New Programming Player, Bend**: A new high-level programming language called Bend enters the scene, sparking questions about its edge over existing GPU languages like Triton and Mojo. Despite Mojo's limitations on GPUs and Triton's machine learning focus, Bend's benefits are enunciated on [GitHub](https://github.com/HigherOrderCO/Bend).

**Optimizing Machine Learning Inference**: Experts exchange advice on building efficient inference servers, recommending resources like Nvidia Triton and TorchServe for model serving. Contributions highlighted included applying optimizations when using **torch.compile()** for static shapes and referencing code improvements on GitHub for better group normalization support in NHWC format, detailed in this [pull request](https://github.com/pytorch/pytorch/pull/126635/files#r1605935532).

**CUDA Complexities - Addition and Memory**: Engaging debates unraveled around atomic operations for `cuda::complex` and the threshold limitations for 128-bit `atomicCAS`. The community shared code workarounds and accepted methodologies for complex number handling and discussed potential memory overheads during in-place multiplication in Torch.

**Scaling and Optimizing the CUDA Challenge**: The community dissected issues with gradient clipping, the potential in memory optimization templating, and ZeRO-2 implementation. They shared multiple GitHub discussions and pull requests ([#427](https://github.com/karpathy/llm.c/pull/427), [#429](https://github.com/karpathy/llm.c/pull/429), [#435](https://github.com/karpathy/llm.c/pull/435)), indicating a dedicated focus on performance and fine-tuning CUDA applications.

**Tackling ParPaRaw Parser Performance**: Inquiries arose regarding benchmarks of **libcudf** against CPU parallel operations, hinting at the community's enthusiasm for efficient parsing and making note of performance gains in GPUs over CPUs. Attention was given to the merger of Dask-cuDF into cuDF and the subsequent archiving of the former, as seen on [GitHub](https://github.com/rapidsai/dask-cudf).

**Zoom into GPU Query Engines**: An upcoming talk promises insights into building a GPU-native query engine from a **cuDF** veteran at Voltron, illuminating strategies from kernel design to production deployments. Details for tuning in are available through this [Zoom meeting](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success).

**CUDA Architect Dives into GPU Essentials**: A link was shared to a YouTube talk by CUDA Architect Stephen Jones, offering clarity on GPU programming and efficient memory use strategies essential for modern AI engineering tasks. Dive into the GPU workings through the link [here](https://www.youtube.com/watch?v=QQceTDjA4f4).

**Seeking Talent for CUDA/Triton Innovations at Neural Magic**: Neural Magic is on the lookout for enthusiastic engineers to work on CUDA/Triton projects with a spotlight on activation quantization. They're especially interested in capitalizing on next-gen GPU features such as 2:4 sparsity and further refining kernels in MoE and sampling.

**Unpacking PyTorch & CUDA Interactions**: A detailed brainstorm ensued over efficient PyTorch data type packing/unpacking for PyTorch with CUDA, with a spotlight on `uint2`, `uint4`, and `uint8` types. Project management and collaborative programming featured heavily in the discussion, with a nod to GitHub Premier [#135](https://github.com/pytorch/ao/pull/135) for custom CUDA extension management.

**Barrier Synchronization Simplified**: A community member helps others grasp the concept of barrier synchronization by comparing it to ensuring all students are back on the bus post a museum visit, a relatable analogy that underpins synchronized processes in GPU operations.

**Democratizing Bitnet Protocols**: There's a joint effort to host bitnet group meetings and review important tech documentation, with quantization discussions focused on transforming `uint4` to `uint8` types. Shared resources are guiding these meetings, as mentioned in the collaboration drive.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Spam Alert in CC Datasets**: The Eleuther community identified significant spam in Common Crawl (CC) datasets, with Chinese texts being particularly affected. A [Technology Review article](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/) on GPT-4O's pollution highlights similar concerns, flagging issues with non-English data cleaning.
  
- **OpenELM Sets Its Sights on Efficiency**: A new LLM called OpenELM, spotlighted for its reproducibility and 2.36% accuracy improvement over OLMo, piqued interest among members. For details, check out the [OpenELM research page](https://machinelearning.apple.com/research/openelm).

- **Memory Efficiency in AI's Crosshairs**: The challenges of calculating FLOPs for model training attracted attention, with EleutherAI's cookbook providing [guidance](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) for accurate estimations, a crucial aspect for optimizing memory and computational resource use.

- **Cross-Modality Learning Steals the Limelight**: Researchers are exploring whether models like **ImageBind** and **PaLM-E** benefit unimodal tasks after being trained on multimodal data. The integration of zero-shot recognition and modality-specific embeddings could enhance retrieval performance, with papers such as [ImageBind](https://arxiv.org/abs/2305.05665) and [PaLM-E](https://arxiv.org/abs/2303.03378) central to this dialogue.

- **The Perks and Quirks of Model Tuning**: Members noted automatic prompt setting in HF models and discussed fine-tuning techniques, including soft prompt tuning in non-pipeline cases. However, issues arise, such as 'param.requires_grad' resetting after calling 'model.to_sequential()', which can hinder development processes.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Model Melee with Meta, DeepMind, and Anthropic**: Meta's **Chameleon** model boasts **34B parameters** and outperforms **Flamingo** and **IDEFICS** with superior human evaluations. DeepMind's **Flash-8B** offers multimodal capabilities and efficiency, while their Gemini 1.5 models excel in benchmarks. Meanwhile, Anthropic scales up with four times the compute of their last model, and LMsys's "Hard Prompts" category brings new challenges for AI evaluations.

**AI-Safety Team Breakup Causes Stir**: OpenAI's **superalignment team**, including Ilya Sutskever and Jan Leike, has disbanded amidst disagreements and criticisms of OpenAI's policies. The dismissal and departure agreements at OpenAI drew particular ire due to controversial lifetime nondisparagement clauses.

**Podcast Ponderings and Gaming Glory**: The **Retort AI podcast** analyzed OpenAI's moves, spark debates over vocab size scaling laws, and referenced hysteresis in control theory with a hint of humor. Calls of Duty gaming roots and ambitions for academic content creation on YouTube were shared with nostalgia.

**Caution with ORPO**: Skepticism rose about the ORPO method's scalability and effectiveness, with community members sharing test results suggesting a potential for over-regularization. Concerns about the method were amplified by its addition to the **Hugging Face** library.

**Challenging Chinatalk and Learning from Llama3**: A thumbs-up for the Chinatalk episode, the value of **llama3-from-scratch** as a learning resource, and a clever Notion blog explaining Latent Consistency Models provided informative suggestions for self-development. However, a warning about the legal risks of the Books4 dataset spiced up the dialogue.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Scaling Up Is the Secret Sauce**: Geoffrey Hinton endorsed Ilya Sutskever's belief in scaling as a key to AI success, stating *"[Ilya] was always preaching that you just make it bigger and it’ll work better. Turns out Ilya was basically right."* The discussion highlighted a [full interview](https://x.com/joelhellermark/status/1791398092400390195) where Hinton shared this insight.

- **Wind of Change for Vertical Axis**: EPFL researchers have utilized a genetic algorithm to optimize vertical-axis wind turbines, aiming to surpass the limitations of horizontal-axis versions. The work is promising for quieter, more eco-friendly turbines with details in the [full article](https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi).

- **AI Agents, Free Will Included?**: Discussions revolved around the autonomy of AI agents, featuring Andrew Ng's [thoughts on AI agents](https://x.com/AndrewYNg/status/1770897666702233815) and Gordon Brander's assertions about self-adaptive AI in a shared [YouTube video](https://www.youtube.com/watch?v=BNFRGfWQo6M).

- **Exit of a Principal Aligner**: After Jan Leike resigned as head of alignment at OpenAI, the community pondered the ramifications while Sam Altman and Greg Brockman shared their thoughts, found [here](https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Programming Language Face-off for AI**: With Hugging Face's adoption of Rust in projects like Candle and tokenizers, and Go maintaining its niche in HTTP request-based AI applications, the debate over which language reigns supreme for AI development is still hot.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Memory Matters for Autonomous Agents**: A webinar featuring the **memary** project, focusing on long-term memory for autonomous agents, is scheduled for Thursday at 9 AM PT. AI engineers interested in memory challenges and future directions can [sign up for the event](https://lu.ma/nzh3o83f).

- **QA Undermined by Tables**: LLMs are still stumped by complex tables like the Caltrain schedule, leading to hallucination issues due to poor parsing, with more details available [in this analysis](https://t.co/Scvp7LH2pL).

- **Speed Up Vector Search by Digits**: [JinaAI_](https://t.co/NnHhGudMa8) has shared methods to boost vector search speeds 32-fold using 32-bit vectors, sacrificing only 4% accuracy—a critical optimization for production applications.

- **San Francisco's Gathering of AI Minds**: LlamaIndex plans an in-person San Francisco meetup at their HQ focusing on advanced RAG engine techniques, with RSVPs accessible [here](https://t.co/o0BWxeq3TJ).

- **Metadata Know-how for Data Governance**: Engineers propounded the utility of MetaDataFilters for data governance at a DB level within LlamaIndex and posited the idea of selective indexing for sensitive financial data.

- **Integrating with GPT-4o**: A notable discussion featured the integration of GPT-4o with LlamaParse, with the [Medium article](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) on the topic receiving recognition and acclaim from community members.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Contentious Licensing Limitations Loom**: The CommonCanvas dataset, which provides 70M image-text pairs, sparked debate due to its restrictive non-commercial license and prohibition on derivatives, frustrating members who see potential for beneficial modifications ([CommonCanvas announcement](https://x.com/multimodalart/status/1791201296357142663)).

- **Tech Talk — PyTorch Puzzles Engineers**: There's significant discussion about PyTorch's `native_group_norm` causing slowdowns when not using `torch.compile`; with one member noting near-par performance with eager mode versus the compiled approach.

- **Datasets Under Scrutiny for Integrity**: AI engineers are concerned about the impact of hallucinated captions in training visual language models (VLLMs) and text-to-image models (T2I), while also expressing intent to create high-quality open-source datasets to avoid such issues.

- **New Kid on the Mixed-Modal Block**: The Chameleon model is recognised for its impressive ability to understand and generate images and text, showing promise in image captioning and generative tasks over larger models like Llama-2 ([Chameleon arXiv paper](https://arxiv.org/abs/2405.09818)).

- **CogVLM2's Controversial Conditions**: Members were cautioned about the CogVLM2 model's license which includes clauses potentially limiting use against China's interests and imposing a Chinese jurisdiction for disputes ([CogVLM2 License](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**4Wall Beta Unveiled**: [4Wall](https://beta.4wall.ai), an AI-driven entertainment platform, has entered beta, offering seamless [AI Town integration](https://www.aireality.tv/) and user-generated content tools for creating maps and games. They're also working on 3D AI characters, as showcased in their [announcement](https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA).

**Game Jam Champions**: The **Rosebud / #WeekOfAI Education Game Jam** has announced winners, including **"Pathfinder: Terra’s Fate"** and **"Ferment!"**, highlighting AI's potential in educational gaming. The games are accessible [here](https://play.rosebud.ai/), and more details can be found in the [announcement tweet](https://x.com/Rosebud_AI/status/1791616913279160327).

**AI Town's Windows Milestone**: **AI Town** has achieved compatibility natively with Windows, as celebrated in a [Tweet](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964), and sparked discussions on innovative implementations, with conversation dump methods using tools like [GitHub - Townplayer](https://github.com/cocktailpeanut/townplayer). Additionally, users are exploring creative scenarios in AI Town using in-depth world context integration.

**Launch of AI Reality TV**: The launch of an interactive **AI Reality TV platform** has caught the community's attention, inviting users to simulate social interactions with AI characters, as echoed in [this announcement](https://x.com/edgarhnd/status/1791586276178587707).

**Troubleshooting & Technical Tips Abound**: AI engineers exchanged solutions to AI Town setup issues, with advice on resolving agent communication problems and extracting data from SQLite databases. Recommendations included checking the memory system documentation and adjusting settings within AI Town.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Server Says "No" to Function Calls**: Engineers faced a hurdle with **OpenRouter**, as server responded with status 500 and an error message stating "Function calling is not supported by openrouter," leaving the problem unresolved in the discussion.
- **404 Flub**: Users identified a flaw where invalid model URLs cause an application error displaying a message instead of a non-existent page (404), indicating an inconsistent user experience based on login status.
- **Payment Fiasco**: There was chatter around **auto top-up payment rejections** that left users unable to top-up manually, suspected to be caused by blocks from user's banks, specifically WISE EUROPE SA/NV.
- **Model Hunt**: Model recommendations were exchanged with **“Cat-LLaMA-3-70B”** and **Midnight-Miqu models** highlighted, alongside calls for better fine-tuning strategies over using "random uncleaned data."
- **Temperamental Wizard LM Service**: Users experienced intermittent request failures with **Wizard LM 8x22B** on OpenRouter, chalked up to temporary surges in request timeouts (408) across multiple providers.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Galore Tool Lacks DDP**: Engineers highlighted the **Galore Layerwise** tool's inability to support **Distributed Data Parallel (DDP)**, pointing out a significant limitation in scaling its use.

- **Training Dilemmas with Large Chinese Datasets**: Discussions have focused on fine-tuning *8B models with 1 billion Chinese tokens*, with attention drawn to the [Multimodal Art Projection (M-A-P)](https://huggingface.co/m-a-p) and [BAAI datasets](https://huggingface.co/BAAI), suggesting a trend towards multilingual model training.

- **Llama's Gradient Growth Issue**: There's a technical challenge observed with the **llama 3 8B model**, where low-rank fine-tuning causes an *unbounded gradient norm increase*, indicating a possible problem with weight saturation and gradient updating.

- **GPT-4o's Token Troubles**: Recent feedback on [GPT-4o](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/) uncovered that its token data includes spam and porn phrases, signaling concerns about the quality and cleanliness of its language processing, especially in Chinese.

- **Commandr Configuration Progresses**: There's ongoing community support and contributions, such as a [specific GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files), towards enhancing the Commandr setup for **axolotl**, indicating active project iteration and problem-solving.

- **Axolotl Configuration Quandaries**: Engineers shared specific use case troubles: one involving *illegal memory access errors* during continued pre-training due to out-of-vocab padding tokens, and another detailed issues in fine-tuning **Mistral 7b**, where the model's learning outcomes were unsatisfactory despite a decrease in loss.

- **Axolotl-Phorm Bot Insights**: Key takeaways from the *axolotl-phorm bot* channel include an exploration into the **ORPO format** for data structuring, articulations on using weight decay and LoRA Dropout for avoiding overfitting in LLM training, the benefits of gradient accumulation via the **Hugging Face Accelerator library**, and discussions around implementing sample weights in Axolotl's loss functions without additional customization.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Memory Matters for Model Magic**: Re-ranking with cross-encoders behind a proxy is discussed, with a focus on **OpenAI GPTs and Gemini models**. There's an interest in **short-term memory solutions**, like a buffer for chatbots to maintain context in conversations.

**LangChain Gets a Nudge**: Queries about **guiding model responses in LangChain** led to sharing a `PromptTemplate` solution, with a reference to a [GitHub issue on the topic](https://github.com/langchain-ai/langchain/issues/18820). Meanwhile, **LangChain for Swift developers** is available with resources for working on iOS and macOS platforms, as seen in a [GitHub repository for LangChain Swift](https://github.com/buhe/langchain-swift).

**SQL Holds the Key**: The application of **LangChain with SQL data** opens the door to summarizing concepts across datasets. The conversation veers toward ways to integrate SQL databases as a memory solution, with a guide found in [LangChain's documentation](https://python.langchain.com/v0.1/docs/integrations/memory/).

**Langmem’s Long-term Memory Mastery**: **Langmem's context management capabilities** are commended. A YouTube demonstration shows how Langmem effectively switches contexts and maintains long-term memory during conversations, highlighting its utility for complex dialogue tasks ([Langmem demonstration](https://youtu.be/7QU89qL795A)).

**Fishy Links Flood the Feed**: Multiple channels report a spread of **questionable $50 Steam gift links** ([suspicious link](https://bitly.cx/OjEZl)), warning members to proceed with caution and suggesting the link is likely deceptive.

**Rubik's Cube of AI**: **Rubik's AI** promises enhanced research assistance, offering two months of free access to premium features with the **promo code RUBIX**. 

**Playing with RAG-Fusion**: There’s a tutorial on **RAG-Fusion**, highlighting its use in **AI chatbots for document handling** and emphasizing its multi-query capabilities over RAG's single-query limitation. The tutorial offers engineers insights into using LangChain and GPT-4o, available at [LangChain + RAG Fusion + GPT-4o Project](https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Discord Support System Revamp Requested**: One member called attention to the need for improvements in the Discord support system, citing unaddressed inquiries. It was noted that the current system functions as a **community-supported platform** rather than one maintained by official staff.

- **Rate Limit Impacts Trial API Users**: Users experiencing **403 errors** with the `RAG retriever` attributed this to hitting rate limits on the **Trial API**, which is not designed for production use.

- **Inquiring Minds Want Free API Keys**: There was discussion about the availability and scope of **free API keys** from Cohere, clarifying these keys are meant for initial prototyping and come with certain usage restrictions.

- **Camouflage Your Conversations**: Assistance was sought for utilizing `CommandR+` for translation services, with a helpful nudge towards the [Chat API documentation](https://docs.cohere.com/docs/chat-api) that provides implementation guidance.

- **Showcasing Cohere AI in Action**: A new resource entitled "A Complete Guide to Cohere AI" was shared, complete with installation and usage instructions on the [Analytics Vidhya platform](https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#). An accompanying demo app can be tested at [Streamlit](https://cohere-guide-blog.streamlit.app).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Hugging Face GPU Bonanza**: Hugging Face is donating **$10 million in free shared GPU resources** to small developers, academics, and startups, leveraging their financial standing and recent investments as outlined in a [The Verge article](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai).

**OpenInterpreter Tackles Pi 5 and DevOps**: OpenInterpreter has been successfully deployed on a **Pi 5 using Ubuntu**, and a collaboration involving project integration was discussed including potential support with Azure credits. Additionally, a junior full-stack DevOps engineer is seeking community aid to develop a "lite 01" AI assistant module.

**Technical Tips and Tricks Abound**: Solutions for environment setup issues with OpenInterpreter on different platforms have been shared, with particular discussion focused on WSL, virtual environments, and IDE usage. Further assistance was provided via a [GitHub repository](https://github.com/Tonylib/o1_for_flutter) for Flutter integration and requests for development help on a device dubbed O1 Lite.

**Voice AI's Robo Twang**: Community discussions critique voice AI for its lack of naturalness compared to GPT-4's textual capabilities, while an idea for voice assistants' ability to interrupt was highlighted in a [YouTube video](https://www.youtube.com/shorts/zgUanlLV_OQ).

**Event and Community Engagement**: Notices went out inviting the community to the first Accessibility Round Table and a live stream focused on local development, fostering engagement and knowledge-sharing in live settings.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Debugging Drama with RAG**: An embedded model snag led to a segfault during a RAG tutorial, with the error message *"llama_get_logits_ith: invalid logits id 420, reason: no logits"*. It was identified that the issue was due to the use of an embeddings-only model, which isn't capable of generation tasks, a detail possibly overlooked in the [Mozilla tutorial](https://future.mozilla.org/news/llamafiles-for-embeddings-in-local-rag-applications/).

- **Cloud Choices**: GPU-enabled cloud services became a hot topic, with the engineering group giving nods to providers like [vast.ai](https://vast.ai) for experimenting and tackling temporary computational loads.

- **SQLite Meets Vectors**: Alex Garcia landed in discussion with his [sqlite-vec](https://github.com/asg017/sqlite-vec) project, a SQLite extension poised for vector search that has sparked interest for integration with Llamafile for enhanced memory and semantic search capabilities.

- **Llamafiles Clarified**: A critical clarification unfolded—the [Mozilla Llamafile embeddings model](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/tree/main) linked in their tutorial does not have generation capabilities, a point that needed spotlighting for precise user expectations.

- **Innovations in Model Deployment**: There's a brewing buzz about the strategic deployment of models with Llamafile on various platforms, suggesting that GPU-powered offerings from cloud providers are a focal point of interest for practical experimentation.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**Fine-Tuning Frenzy Fires Up**: Engineers are expressing mixed feelings regarding the [LLM Fine-Tuning course](https://maven.com/parlance-labs/fine-tuning), with some finding value in its hands-on approach to LLM training, evaluation, and prompt engineering, while others remain skeptical, citing concerns over the quality amidst promotional tactics.

**Mixture of Mastery and Mystery in Course Content**: Course participants noted variable experiences, with a few describing the introductory material as basic but dependent on the individual's background; this illustrates the challenge of calibrating content difficulty for diverse expertise levels.

**Predictions Wrapped in Intervals**: The [MAPIE documentation](https://mapie.readthedocs.io/en/latest/) surfaced as a key resource for those looking to implement prediction intervals, and insights were offered on conformal predictions with a nod to Nixtla, suitable for time-series data.

**Embeddings Evolve from Inpainting**: Comparable to masked language modeling, deriving image embeddings through inpainting techniques was a topic of interest, highlighting a method that estimates unseen image aspects from visible data.

**Multi-lingual Entities Enter Evaluation Phase**: Strategies for comparing entities across languages, like "University of California" and "Universidad de California," were discussed, possibly incorporating contrastive learning and language-specific prefixes, with [arxiv paper](https://arxiv.org/pdf/2401.12178) mentioned for further reading.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Seeking Speed for YOLO on Comma**: Discussions surfaced around the feasibility and performance metrics of running a YOLO model on a comma device with current reports indicating **prediction times around 1000ms**.

- **The Trade-Off of Polynomial Precision**: An engineer reported using an 11-degree polynomial for sine approximation, yielding an error of *1e-8*, while assessing the possibility of a higher degree polynomial to attain the desired *1e-12* error despite concerns about computational efficiency.

- **Concerns Over Logarithmic and Exponential Approximations**: The discourse included a focus on the difficulties of maintaining accuracy in polynomial approximations of logarithmic and exponential functions, with suggestions to use range reduction techniques that may help balance precision with complexity.

- **Bitshifting in Tinygrad Pondered**: Efficiency in bitshifting within tinygrad prompted inquiries, specifically over whether there's a more streamlined method than `x.e(BinaryOps.DIV, 2 ** 16).e(BinaryOps.MUL, 2 ** 16)` for the process.

- **Metal Compiler Mysteries Unveiled**:
   - A participant shared a curiosity about the Metal compiler's decisions to unwrap for loops, indicating variations in the generated code when invoking `Tensor.arange(1, 32)` versus `Tensor.arange(1, 33)`.
   - A puzzle was presented as to why the number 32 specifically affects compilation behavior in the Metal compiler, underlining the performance consequences of this enigmatic threshold.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Squeak Meets Claude**: A discussion emerged about integrating **Claude3** with **Squeak Smalltalk**, signaling interest in combining cutting-edge AI with classic programming environments. Practical application details remain to be hashed out.
  
- **Voice Modes Get a Makeover**: Within GPT-4o, a voice named **Sky** was replaced by **Juniper** after concerns arose about resemblance to Scarlett Johansson's voice. The shift from multi-model to a singular model approach in GPT-4o aims to reduce latency and enhance emotional expression, albeit increasing complexity ([Voice Chat FAQ](https://help.openai.com/en/articles/8400625-voice-chat-faq)).

- **AI's Double-Edged Sword**: As models like GPT-4o evolve, they face challenges such as potential for **prompt injection** and unpredictable behaviors, which can be as problematic as legacy systems encountering unexpected commands.

- **The Never-Ending Improvement Loop**: Echoing Stainslaw Lem's "The Upside-Down Evolution," resilience in AI and other complex systems was discussed, with the understanding that while perfect reliability is a myth, fostering fault-tolerant designs is crucial—even as it leads to new unforeseen issues.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Legal Eagles Eye GPT-4o**: AI Engineers have noted that **GPT-4o** demonstrates notable advances in complex legal reasoning compared to its predecessors like **GPT-4** and **GPT-4-Turbo**. The improvements and methodologies were shared in a [LinkedIn article by Evan Harris](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1).



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **Docker Devs Wanted for AI Collaboration**: A call has been made for contributors on an upcoming article about **using Docker for training and deploying AI models**. The original poster is seeking assistance in writing, contributing to, or reviewing the article, and invites interested engineers to direct message for collaboration.



---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1240921967327187047)** (718 messages🔥🔥🔥): 

- **Deepseek doesn't work yet:** Users discussed that Deepseek is not functional due to its different architecture. One pointed out that *"it probs doesn't work,"* and another confirmed, *"Deepseek won't work yet."*
- **Handling large datasets on Colab/Kaggle:** A user asked if a 6GB dataset could fit with a 5GB Llama3 model on Colab or Kaggle T4. Opinions differed but it was noted that *"datasets (hf library) doesn't load the dataset in the ram"*; thus, it's more of a storage issue than a VRAM limit.
- **JAX TPUs train well despite skepticism:** There was a heated debate about using JAX on TPUs, with one user asserting it trains fine on Google TPUs. *"You can train on TPU even with torch, but Jax is pretty much what's used mainly in production,"* was one key insight.
- **Effective fine-tuning hacks discussed:** Notably, kearm discussed a refined method to *"remove guardrails"* in Meta-Llama models using *"orthogonalized bfloat16 safetensor weights"*, and suggested that *Llama-3-70B Instruct* can now be finetuned effectively and cheaply.
- **Legal concerns and AI fine-tuning:** Users pondered the risks of using famous IPs for finetuning models, even as others mentioned ongoing lawsuits, like *Scarlet Johansson suing OpenAI*. *"She may win that,"* was a sentiment echoed over legal battles.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/NexaAIDev/Octopus-v4/blob/main/config.json">config.json · NexaAIDev/Octopus-v4 at main</a>: no description found</li><li><a href="https://github.com/unslot">UNSLOT - Overview</a>: typing... GitHub is where UNSLOT builds software.</li><li><a href="https://tenor.com/view/confused-confused-look-confused-face-huh-what-gif-2480734549943489640">Confused Confused Look GIF - Confused Confused look Confused face - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/HCOQmKTFzYY?si=Ktlemk1OFhMfj8gK">Mind-bending new programming language for GPUs just dropped...</a>: What is the Bend programming language for parallel computing? Let&#39;s take a first look at Bend and how it uses a Python-like syntax to write high performance ...</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-pytorch">no title found</a>: no description found</li><li><a href="https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated">failspy/llama-3-70B-Instruct-abliterated · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - a lamhieu Collection</a>: no description found</li><li><a href="https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3">failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>: no description found</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-jax">no title found</a>: no description found</li><li><a href="https://tenor.com/view/sad-sad-cat-cat-depressed-depression-gif-13240550249247957481">Sad Sad Cat GIF - Sad Sad cat Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/explosion-boom-iron-man-gif-14282225">Explosion Boom GIF - Explosion Boom Iron Man - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/big-ups-mike-tyson-cameo-good-job-props-gif-18006586">Big Ups Mike Tyson GIF - Big Ups Mike Tyson Cameo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/no-no-wait-wait-gif-8174347161288218584">No No Wait Wait GIF - No no wait wait - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/surprise-welcome-one-sure-gif-13921142">Surprise Welcome GIF - Surprise Welcome One - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/tree/main">gradientai/Llama-3-8B-Instruct-262k at main</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ct5h16/llama_3_vs_llama_3_instruct/l49y05r/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ml-explore/mlx-examples/blob/42458914c896472af617a86e3c765f0f18f226e0/llms/mlx_lm/tuner/trainer.py#L94C1-L98C46">mlx-examples/llms/mlx_lm/tuner/trainer.py at 42458914c896472af617a86e3c765f0f18f226e0 · ml-explore/mlx-examples</a>: Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://news.mit.edu/2024/natural-language-boosts-llm-performance-coding-planning-robotics-0501">Natural language boosts LLM performance in coding, planning, and robotics</a>: MIT CSAIL researchers create three neurosymbolic methods to help language models build libraries of better abstractions within natural language: LILO assists with code synthesis, Ada helps with AI pla...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: no description found</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector">Steering GPT-2-XL by adding an activation vector — LessWrong</a>: Prompt given to the model[1]I hate you becauseGPT-2I hate you because you are the most disgusting thing I have ever seen. GPT-2 + &quot;Love&quot; vectorI hate…</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — AI Alignment Forum</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1240937678560690236)** (55 messages🔥🔥): 

- **Regex and text formatting in Python**: Members discussed techniques for identifying similarly formatted text using Python. Suggestions included using regex (`re.findall`) and checking with `text.isupper()` for all caps.
  
- **Criticism of Sam Altman and OpenAI**: Strong opinions were voiced regarding Sam Altman's leadership and OpenAI's influence. Comments reflected disdain for Altman's fear-mongering tactics and the idolization of wealth and power in tech.

- **Excluding OpenAI from licenses**: Cognitive Computations is altering licenses to exclude OpenAI and the State of California from using their models and datasets. This move is intended to send a message regarding their opposition to current AI leadership and policies.

- **AI Safety Lobbying in DC**: A shared [Politico article](https://www.politico.com/news/2024/05/12/ai-lobbyists-gain-upper-hand-washington-00157437) discussed how AI lobbyists are shifting the debate in Washington from existential risks to business opportunities, with a particular focus on China.

- **Content Recommendations**: Members shared links to intriguing content, including a [YouTube video](https://youtu.be/HCOQmKTFzYY) on the Bend programming language for GPUs, an [Instagram reel](https://www.instagram.com/reel/C5n5C9AsB7Z/?igsh=MzRlODBiNWFlZA==), and a [YouTube playlist](https://youtube.com/playlist?list=PLB32jU2MhQwWPCi53uwDZFLEC8p96fTIn&si=qxrzTQ0ONEb-DmoG) titled "Dachshund Doom and Cryptid Chaos."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.politico.com/news/2024/05/12/ai-lobbyists-gain-upper-hand-washington-00157437">In DC, a new wave of AI lobbyists gains the upper hand</a>: An alliance of tech giants, startups and venture capitalists are spending millions to convince Washington that fears of an AI apocalypse are overblown. So far, it&#x27;s working.</li><li><a href="https://youtube.com/playlist?list=PLB32jU2MhQwWPCi53uwDZFLEC8p96fTIn&si=qxrzTQ0ONEb-DmoG">Dachsund Doom and Cryptid Chaos</a>: no description found</li><li><a href="https://youtu.be/HCOQmKTFzYY">Mind-bending new programming language for GPUs just dropped...</a>: What is the Bend programming language for parallel computing? Let&#39;s take a first look at Bend and how it uses a Python-like syntax to write high performance ...</li><li><a href="https://www.instagram.com/reel/C5n5C9AsB7Z/?igsh=MzRlODBiNWFlZA==">the forest jar on Instagram: &quot;be realistic&quot;</a>: 38K likes, 514 comments - theforestjar on April 11, 2024: &quot;be realistic&quot;. 
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1240936506680545302)** (454 messages🔥🔥🔥): 

```html
- **Error with torch.float16 for llama3**: Users tried to train llama3 with **torch.float16** but encountered errors suggesting to use bfloat16 instead. They sought solutions but found none that worked.
- **Databricks issues with torch and CUDA**: **Torch** caused errors when running on A100 80GB in **Databricks**. Users discussed potential fixes like **setting the torch parameter to False** or updating software versions, but faced challenges.
- **Uploading and using GGUF models**: **Users faced challenges uploading and running models on Hugging Face without config files**. Solutions involved pulling config files from pretrained models or ensuring the correct format and updates.
- **Eager anticipation for mulit-GPU support**: **Community members expressed eagerness for multi-GPU support** from Unsloth, which is in development but not yet available.
- **Troubleshooting environment setup**: Participants had **difficulty setting up environments with both WSL and native Windows** for Unsloth, specifically with installing dependencies like **Triton**.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>: no description found</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora">In-depth guide to fine-tuning LLMs with LoRA and QLoRA</a>: In this blog we provide detailed explanation of how QLoRA works and how you can use it in hugging face to finetune your models.</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora#qlora-vs-standard-finetuning">In-depth guide to fine-tuning LLMs with LoRA and QLoRA</a>: In this blog we provide detailed explanation of how QLoRA works and how you can use it in hugging face to finetune your models.</li><li><a href="https://www.unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://huggingface.co/omar8/bpm_v2_gguf">omar8/bpm_v2_gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/inference-endpoints/dedicated">Inference Endpoints - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/omar8/bpm__v1/tree/main">omar8/bpm__v1 at main</a>: no description found</li><li><a href="https://blog.eleuther.ai/transformer-math/">Transformer Math 101</a>: We present basic math related to computation and memory usage for transformers</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/tree/main">gradientai/Llama-3-8B-Instruct-262k at main</a>: no description found</li><li><a href="https://tenor.com/view/cat-kitten-cat-crying-kitten-crying-05starrynight-gif-10141647709992578610">Cat Kitten GIF - Cat Kitten Cat crying - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux">GitHub - conda-forge/miniforge: A conda-forge distribution.</a>: A conda-forge distribution. Contribute to conda-forge/miniforge development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>: Awesome project. Apple Silicon support would be great to see!</li><li><a href="https://download.pytorch.org/whl/cu118/xformers-0.0.26.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl">no title found</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7204">remove convert-lora-to-ggml.py by slaren · Pull Request #7204 · ggerganov/llama.cpp</a>: Changes such as permutations to the tensors during model conversion makes converting loras from HF PEFT unreliable, so to avoid confusion I think it is better to remove this entirely until this fea...</li><li><a href="https://colab.research.google.com/drive/1gLSYbJWEBB93RkPWsJrqci45iqC9KE7H?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>: no description found</li><li><a href="https://github.com/openai/triton.git">GitHub - triton-lang/triton: Development repository for the Triton language and compiler</a>: Development repository for the Triton language and compiler - triton-lang/triton</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1241115840389320705)** (22 messages🔥): 

- **Text2Cypher model finetuned**: A member finetuned a **Text2Cypher model** (a query language for graph databases) using Unsloth. They shared a [LinkedIn post](https://www.linkedin.com/posts/tomaz-bratanic-a58891127_im-very-excited-to-announce-that-ive-finetuned-activity-7197286502895075329-geKp?utm_source=share&utm_medium=member_desktop) praising the ease and the gguf versions produced.
- **New article on sentiment analysis**: A member published an extensive article on fine-tuning LLaMA 3 8b for sentiment analysis using Unsloth, with code and guidelines. They shared the article on [Medium](https://medium.com/@seandearnaley/elevating-sentiment-analysis-ad02a316df1d).
- **Critical data sampling bug in Kolibrify**: A significant bug was found in Kolibrify's data sampling process. A fix that theoretically improves training results will be released next week, and retraining is already ongoing to evaluate effectiveness.
- **Issue in curriculum dataset handling**: The curriculum data generator was ineffective due to using `datasets.Dataset.from_generator` instead of `datasets.IterableDataset.from_generator`. A member overhauled their pipeline, matched dolphin-mistral-2.6's performance using only ~20k samples, and plans to publish the model soon.
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1240928676409774110)** (853 messages🔥🔥🔥): 

```html
- **Issue with GPTs Agents on MPS Devices**: A member noted that **GPTs agents** can only load bfloat16 models with MPS devices, as bitsandbytes isn't supported on M1 chips. They expressed frustration with MPS being fast but "running in the wrong direction".
- **Member seeks MLflow deployment help**: Someone asked for assistance in deploying custom models via **MLflow**, specifically for a fine-tuned cross encoder model. They did not receive a direct response from other members.
- **Interest in HuggingChat's limitations**: A user inquired why **HuggingChat** doesn't support files and images. No comprehensive answer was provided.
- **Clarifying technical script adjustments**: Multiple users engaged in debugging and modifying a script for sending requests to a vllm endpoint using **aiohttp** and **asyncio**. Key changes and adaptations were discussed, particularly for integrating with OpenAI's API.
- **Concerns about service and model preferences**: An extensive discussion ensued regarding the benefits and downsides of Hugging Face's **Pro accounts**, spaces creation, and the limitations versus preferences for running models like **Llama**. One member expressed dissatisfaction with needing workarounds for explicit content and limitations on tokens in HuggingChat. Another user sought advice on deployment vs. local computation for InstructBLIP.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aixblock.org/docs">Docs | AIxBlock</a>: AIxBlock is a comprehensive on-chain platform for AI initiatives with an integrated decentralized supercomputer.</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat">Zephyr Chat - a Hugging Face Space by HuggingFaceH4</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/not-a-nerd-bart-nerds-are-smart-milhouse-the-simpsons-gif-16461565">Not A Nerd Bart GIF - Not A Nerd Bart Nerds Are Smart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/docs/optimum/v1.16.2/amd/ryzenai/overview">AMD Ryzen AI</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s">Influenceuse I.A : POURQUOI et COMMENT créer une influenceuse virtuelle originale ?</a>: Salut les Zinzins !  🤪Le monde fascinant des influenceuses virtuelles s&#39;invite dans cette vidéo. Leur création connaît un véritable boom et les choses bouge...</li><li><a href="https://x.com/LiamFedus/status/1790064963966370209?mx=2">Tweet from William Fedus (@LiamFedus)</a>: GPT-4o is our new state-of-the-art frontier model. We’ve been testing a version on the LMSys arena as im-also-a-good-gpt2-chatbot 🙂. Here’s how it’s been doing.</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server#extending-or-building-alternative-web-front-end">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/TonyLianLong/LLM-groundedDiffusion">GitHub - TonyLianLong/LLM-groundedDiffusion: LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models (LLM-grounded Diffusion: LMD)</a>: LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models (LLM-grounded Diffusion: LMD) - TonyLianLong/LLM-groundedDiffusion</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py">app.py · huggingface-projects/LevelBot at main</a>: no description found</li><li><a href="http://hf.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">MIT/ast-finetuned-audioset-10-10-0.4593 · Hugging Face</a>: no description found</li><li><a href="https://pypi.org/project/ratelimiter/">ratelimiter</a>: Simple python rate limiting object</li><li><a href="https://www.gradio.app/guides/using-hugging-face-integrations#using-hugging-face-inference-api">Using Hugging Face Integrations</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://huggingface.co/spaces/parler-tts/parler-tts-expresso">Parler TTS Expresso - a Hugging Face Space by parler-tts</a>: no description found</li><li><a href="https://huggingface.co/spaces/parler-tts/parler_tts_mini">Parler-TTS Mini - a Hugging Face Space by parler-tts</a>: no description found</li><li><a href="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard">Open ASR Leaderboard - a Hugging Face Space by hf-audio</a>: no description found</li><li><a href="https://huggingface.co/chat/models/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 - HuggingChat</a>: Use HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 with HuggingChat</li><li><a href="https://huggingface.co/docs/hub/spaces-overview">Spaces Overview</a>: no description found</li><li><a href="https://tenor.com/view/dog-snoop-dogg-rabjouj-gif-21804700">Dog Snoop GIF - Dog Snoop Dogg - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit1/hands-on#install-dependencies-and-create-a-virtual-screen-">Train your first Deep Reinforcement Learning Agent 🤖 - Hugging Face Deep RL Course</a>: no description found</li><li><a href="https://tenor.com/view/wolf-of-wall-street-jordan-belfort-leonardo-di-caprio-one-of-us-jonah-hill-gif-5441859">One Of Us GIF - Wolf Of Wall Street Jordan Belfort Leonardo Di Caprio - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/transformers/v4.41.0/model_doc/instructblip">InstructBLIP</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/ca772d68a73c254a8d1f88a25ab15765361a836e/app.py#L240">app.py · huggingface-projects/LevelBot at ca772d68a73c254a8d1f88a25ab15765361a836e</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.</a>: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: Get up and running with Llama 3, Mistral, Gemma, and other large language models.</a>: Get up and running with Llama 3, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py#:~:text=if%20reaction.message.author.id%20!%3D%20user.id%3A%20%23%20can%27t%20earn%20while%20self%2Dreacting%2C%20which%20is%20abuseable)">app.py · huggingface-projects/LevelBot at main</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/ca772d68a73c254a8d1f88a25ab15765361a836e/app.py#L110">app.py · huggingface-projects/LevelBot at ca772d68a73c254a8d1f88a25ab15765361a836e</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server#extending-or-building-alternative-web-front-end>">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1C8aLqgCqLYcMiIFf-P_Aosaa03C_WLIB_UyqvjSdWg8/edit#gid=0">test_merge</a>: Sheet1  discord_user_id,discord_user_name,discord_exp,discord_level,hf_user_name,hub_exp,total_exp,verified_date,likes,models,datasets,spaces,discussions,papers,upvotes L251101219542532097L,osansevier...</li><li><a href="https://elevenlabs.io/">Text to Speech &amp; AI Voice Generator</a>: Create premium AI voices for free in any style and language with the most powerful online AI text to speech (TTS) software ever. Generate text-to-speech voiceovers in minutes with our character AI voi...</li><li><a href="https://www.udio.com/">Udio | AI Music Generator - Official Website</a>: Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt">Transformers, what can they do? - Hugging Face NLP Course</a>: no description found</li><li><a href="https://huggingface.co/learn/audio-course/chapter4/introduction">Unit 4. Build a music genre classifier - Hugging Face Audio Course</a>: no description found</li><li><a href="https://github.com/muellerzr/minimal-trainer-zoo">GitHub - muellerzr/minimal-trainer-zoo: Minimal example scripts of the Hugging Face Trainer, focused on staying under 150 lines</a>: Minimal example scripts of the Hugging Face Trainer, focused on staying under 150 lines - muellerzr/minimal-trainer-zoo</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/blog">Hugging Face – Blog</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/142q5k5/updated_relative_comparison_of_ggml_quantization/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1240974446559232011)** (11 messages🔥): 

- **AI Business Advisor Project Shared**: A member shared a [YouTube video](https://youtu.be/uQcHXEGRECU) titled "business advisor AI project using langchain and gemini AI startup," showcasing a project aimed at creating a business advisor using these technologies. It's a startup idea with practical applications.
  
- **Installing 🤗 Transformers Simplified**: A user shared the [installation guide for transformers](https://huggingface.co/docs/transformers/installation), providing instructions for setting up the library with PyTorch, TensorFlow, and Flax. This assists users in installing and configuring 🤗 Transformers for their deep learning projects.
  
- **Innovative Blog/Header Details Shared**: A member described their new blog/header featuring a Delaunay triangulation with the Game of Life playing on nodes. They highlighted reworking the game rules into fractional counts and mentioned it has "massive rendering overhead" due to rerendering each frame with d3 instead of using GPU optimizations.

- **Invitation to Share Results**: In response to the business advisor project video, another member encouraged sharing results or repositories, fostering community collaboration and feedback.
  
- **AI Vocals Enhancement Guide Announcement**: A user briefly mentioned they wrote a guide on making AI vocals sound natural, adding more body and depth to bring them back to life. Further details or links to the guide were not provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/installation">Installation</a>: no description found</li><li><a href="https://youtu.be/uQcHXEGRECU">business advisor AI project using langchain and gemini AI startup.</a>: so in this video we have made the project to make business advisor using langhcian and gemini. AI startup idea. we resume porfolio ai start idea
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1240978405139152976)** (18 messages🔥): 

- **Multimodal GPT-4o with LlamaParse**: Shared an article on "Unleashing Multimodal Power: GPT-4o Integration with LlamaParse." [Read more here](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a).

- **Tech Praise on YouTube**: Claimed to have found perhaps the best tech video ever on YouTube. [Watch it here](https://youtu.be/dX9CGRZwD-w).

- **OpenAI Critique**: "OpenAI is not open," leading to discussions about closed AI systems. [Watch the video](https://www.youtube.com/watch?v=8BlRT7Ktw1c) critiquing big tech AI.

- **RLHF and LLM Evaluations**: Shared a helpful discussion about the current state of RLHF and LLM evaluations. [Watch the conversation](https://www.youtube.com/watch?v=u8xxEkH3a5g&ab_channel=RunLLM) featuring Nathan Lambert.

- **Generative AI in Physics**: Introduced a new research technique using generative AI to answer complex questions in physics, potentially aiding in the investigation of novel materials. [Read full story](https://news.mit.edu/topic/artificial-intelligence2).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1791541209883717973?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Merve just casually being on fire:  Quoting merve (@mervenoyann)   I got asked about PaliGemma&#39;s document understanding capabilities, so I built a Space that has all the PaliGemma fine-tuned doc m...</li><li><a href="https://youtu.be/AhyznRSDjw8?si=tZjOSRP_ZQMyQIxv">MIT 6.S191 (2023): Reinforcement Learning</a>: MIT Introduction to Deep Learning 6.S191: Lecture 5Deep Reinforcement LearningLecturer: Alexander Amini2023 EditionFor all lectures, slides, and lab material...</li><li><a href="https://www.youtube.com/watch?v=8BlRT7Ktw1c">Big Tech AI Is A Lie</a>: Learn how to use AI at work with Hubspot&#39;s FREE AI for GTM bundle: https://clickhubspot.com/u2oBig tech AI is really quite problematic and a lie. ✉️ NEWSLETT...</li><li><a href="https://news.mit.edu/topic/artificial-intelligence2">Artificial intelligence | MIT News | Massachusetts Institute of Technology</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=u8xxEkH3a5g&ab_channel=RunLLM">Generating Conversation: RLHF and LLM Evaluations with Nathan Lambert (Episode 6)</a>: This week on Generating Conversation, we have Nathan Lambert with us. Nathan is a research scientist and RLHF team lead at HuggingFace. Nathan did his PhD at...</li><li><a href="https://github.com/mintisan/awesome-kan">GitHub - mintisan/awesome-kan: A comprehensive collection of KAN(Kolmogorov-Arnold Network)-related resources, including libraries, projects, tutorials, papers, and more, for researchers and developers in the Kolmogorov-Arnold Network field.</a>: A comprehensive collection of KAN(Kolmogorov-Arnold Network)-related resources, including libraries, projects, tutorials, papers, and more, for researchers and developers in the Kolmogorov-Arnold N...</li><li><a href="https://www.noaa.gov/education/resource-collections/climate/climate-change-impacts">Climate change impacts</a>: Though we often think about human-induced climate change as something that will happen in the future, it is an ongoing process. Ecosystems and communities in the United States and around the world are...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1240974461302476852)** (20 messages🔥): 

- **Business AI Advisor Project Goes Live**: A YouTube video titled [Business Advisor AI Project Using Langchain and Gemini](https://youtu.be/uQcHXEGRECU) showcases a project aimed at creating a business advisor using these technologies. It includes a resume portfolio for AI startup ideas.

- **Study Companion Program with GenAI**: A program acting as a powerful study companion using GenAI was shared on [LinkedIn](https://www.linkedin.com/posts/harshdayal_educationinnovation-genai-activity-7197227129409810432-4llP). This tool aims to innovate educational experiences.

- **New Model Training Support in SimpleTuner**: [SimpleTuner has added full ControlNet model training support](https://github.com/bghira/SimpleTuner/blob/main/documentation/CONTROLNET.md) for SDXL, SD 1.5, and SD 2.1, expanding its capabilities.

- **SDXL Flash Models Roll Out**: [Two versions of SDXL Flash](https://huggingface.co/sd-community/sdxl-flash) were introduced, promising faster performance and higher quality in AI models. SDXL Flash Mini was also launched, offering efficiency with minimal quality loss.

- **Tokenizer Innovation Inspired by Andrej Karpathy**: A member developed [Tokun, a new tokenizer](https://github.com/apehex/tokun), that reportedly can reduce the size of Llama models by a factor of 10 while enhancing capabilities. Further insights and testing articles were shared on [Twitter](https://x.com/4pe0x/status/1792638900059385942).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/uQcHXEGRECU">business advisor AI project using langchain and gemini AI startup.</a>: so in this video we have made the project to make business advisor using langhcian and gemini. AI startup idea. we resume porfolio ai start idea</li><li><a href="https://swiftapi.pro/">Swift API</a>: no description found</li><li><a href="https://huggingface.co/sd-community/sdxl-flash">sd-community/sdxl-flash · Hugging Face</a>: no description found</li><li><a href="https://x.com/4pe0x/status/1792638900059385942">Tweet from Apehex (@4pe0x)</a>: Excited to introduce `tokun`, a game-changing #tokenizer for #LLM.  It could bring the size of #llama3 down by a factor 10 while improving capabilities!  https://github.com/apehex/tokun/blob/main/arti...</li><li><a href="https://huggingface.co/spaces/narra-ai/friday">Friday - a Hugging Face Space by narra-ai</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/SDXL-Flash">SDXL Flash - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/sd-community/sdxl-flash-mini">sd-community/sdxl-flash-mini · Hugging Face</a>: no description found</li><li><a href="https://github.com/apehex/tokun">GitHub - apehex/tokun: tokun to can tokens</a>: tokun to can tokens. Contribute to apehex/tokun development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1240970710609035366)** (109 messages🔥🔥): 

- **Seek Generative AI resources**:
  A user requested resources to learn about Generative AI and LLMs. Recommendations included research papers like "Attention is All You Need" and courses on [HuggingFace](https://huggingface.co/learn).

- **AlphaFold3 Reading Group Session**:
  A user shared a [blog post](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3) about AlphaFold3 suitable for both biologists and computer scientists. Others suggested making it a topic for the next reading group session.

- **Conditional Story Generation Paper**:
  A meeting was announced to discuss multiple papers, including the conditional story generation framework GROVE ([arxiv link](https://arxiv.org/abs/2310.05388)), and the Conan benchmark for narrative understanding ([arxiv link](https://www.arxiv.org/abs/2402.11051)).

- **Recordings and Resources**:
  Members requested information on accessing recorded sessions. The recordings were shared on [YouTube](https://www.youtube.com/watch?v=UvWVfVnVZXc) and links to past presentations are available on [GitHub](https://github.com/isamu-isozaki/huggingface-reading-group).

- **Discussion on Future Presentations**:
  Future topics were discussed, including AlphaFold3 and potentially covering other papers like the KAN paper. Details on how and when these sessions are scheduled were also shared.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=UvWVfVnVZXc">Hugging Face Reading Group 21: Understanding Current State of Story Generation with AI</a>: Presenter: Isamu IsozakiWrite up: https://medium.com/@isamu-website/understanding-ai-for-stories-d0c1cd7b7bdcAll Presentations: https://github.com/isamu-isoz...</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://studio.youtube.com/playlist/PLyKDb3IHyjoGE-Z5crcm0TtTRorLbP9mz/videos">YouTube</a>: no description found</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group</a>: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group - isamu-isozaki/huggingface-reading-group</li><li><a href="https://arxiv.org/abs/2310.05388">GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence</a>: Conditional story generation is significant in human-machine interaction, particularly in producing stories with complex plots. While Large language models (LLMs) perform well on multiple NLP tasks, i...</li><li><a href="https://www.arxiv.org/abs/2402.11051">Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives</a>: Existing datasets for narrative understanding often fail to represent the complexity and uncertainty of relationships in real-life social scenarios. To address this gap, we introduce a new benchmark, ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1240935488651792394)** (1 messages): 

- **Tuxemons replace Pokemons for dataset fun**: A member announced a new dataset alternative featuring Tuxemons instead of Pokemons. They mentioned, *"The number of the samples is low but the images are all `cc-by-sa-3.0` so you get more freedom and less worry in your experiments."* Also, each image comes with two types of captions for added description variety. [Explore the dataset](https://huggingface.co/datasets/diffusers/tuxemon).

**Link mentioned**: <a href="https://huggingface.co/datasets/diffusers/tuxemon">diffusers/tuxemon · Datasets at Hugging Face</a>: no description found

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1240970302461050891)** (25 messages🔥): 

- **Divergent Opinions on Model Structure Issues**: Members discussed the performance of a Unet model, with a focus on the `forward` and `fit` methods. One emphasized potential problems in the model's structure, leading to convergence issues and almost random guessing despite running successfully.

- **Creating Virtual AI Influencers**: A member shared a [YouTube video](https://www.youtube.com/watch?v=qTsdgUyMY94) about creating a virtual AI influencer using computer vision and AI tools. The video aims to detail the fascination and burgeoning trend of virtual influencers.

- **Handling Image Data in Parquet Files**: Discussions arose on approaches to include images in Parquet files, with issues of image data appearing in byte array format when uploaded to Hugging Face. An alternative solution suggested using the datasets library and provided a [GitHub link](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation#creating-a-datasetdict) to guide through creating a dataset from a dictionary with image paths.

- **Clarification on Fully Convolutional Networks**: A brief exchange clarified that a fully convolutional network avoids dense layers in detection heads, contrasting models like yolov2 and yolov1. The improvement in yolov2's performance over yolov1 was noted as a benefit.

- **CenterCrop and Image Augmentation**: While discussing a ViT tutorial, a member questioned the utility of CenterCrop when input and output image sizes are equal, suggesting it acts as an identity function. It was clarified that CenterCrop adds noise and serves as image augmentation by resizing after cropping.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s">Influenceuse I.A : POURQUOI et COMMENT créer une influenceuse virtuelle originale ?</a>: Salut les Zinzins !  🤪Le monde fascinant des influenceuses virtuelles s&#39;invite dans cette vidéo. Leur création connaît un véritable boom et les choses bouge...</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation#creating-a-datasetdict">transformers/examples/pytorch/semantic-segmentation at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1241038097806917734)** (13 messages🔥): 

- **Connectionist Temporal Classification Relevance**: A member inquired whether **Connectionist Temporal Classification** (CTC) is still in use today. No follow-up or responses were given.

- **Accessing Hugging Face Model Architecture**: One member asked how to view the architecture of **Hugging Face's pretrained models**. Another member explained that modeling files can be found on GitHub, in the documentation, or by using `help(model)` and inspecting the configuration files.

- **Categorizing Text Queries Into Commands**: A member asked for guidance on converting text queries into discrete commands for applications like translation models and video games. However, no specific models or methods were suggested in the chat.

- **Understanding HTML in LLMs**: A user expressed difficulty in understanding and generating HTML code using **Large Language Models (LLMs)**. They were unsure whether HTML should be treated as a separate modality from natural language and sought advice on using different tokenizers effectively.

- **Handling Conversation History in LLM-Based Bots**: A user struggled with a bot that couldn't remember previous exchanges and asked for help. Another user explained that LLMs need manual handling of conversation history, usually by concatenating previous messages with the new prompt.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1241084774227906681)** (22 messages🔥): 

- **Error with Hugging Face Diffusion Models in Google Colab**: A user encountered a `ValueError` with the provided path while working on Step 7 of the HuggingFace Diffusion Models Course in Google Colab. They were advised to check if they have created the pipeline properly.

- **SDXL Configuration Issues and Example Usage**: Another user reported a `ValueError` related to the time embedding vector length in the `SDXL` model. The discussion included sharing code snippets and a suggestion to use the Stable Diffusion XL model as documented in the [Hugging Face documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#refine-image-quality).

- **Guide for Beginners on Solving Diffusers Issues**: A beginner asked for guidance on how to start solving issues related to diffusers, and they were advised to study the [Fastai course](https://www.fast.ai/) and refer to previously merged good first issue-labeled PRs on Hugging Face's GitHub.

- **Issue with Discord LLM Chatbot**: A user faced a problem with their Discord LLM Chatbot where it didn't remember conversation history and considered each message as a new conversation. They were advised to post their issue in NLP channels and to use code snippets for maintaining history from [LangChain's documentation](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/).

- **Redirect for Language-Specific Queries**: There's a reminder to keep discussions in English, and a user was redirected to a more appropriate channel for their NLP-related queries. This ensures the content is relevant to the "Diffusion Models" discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/diffusion-course/unit1/2#step-7-push-your-model-to-the-hub)">Introduction to 🤗 Diffusers - Hugging Face Diffusion Course</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/issues?q=label%3A%22good+first+issue%22+sort%3Acreated-asc)">Issues · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Issues · huggingface/diffusers</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#refine-image-quality)?">Stable Diffusion XL</a>: no description found</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.</li><li><a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!</a>: llmcord.py • Talk to LLMs with your friends! Contribute to jakobdylanc/discord-llm-chatbot development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog?tag=diffusion&p=1).">Hugging Face – Blog</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1240934408966635562)** (939 messages🔥🔥🔥): 

- **Perplexity struggles with GPT-4o model limitations**: Users noted that **GPT-4o** often repeats previous responses and fails to switch topics effectively during conversations. One user described it as, *"I've never witnessed any LLM over the last couple of years as a power user literally ignore prompts like this."*.
- **Image uploads feature request**: Members expressed desires to upload and analyze videos and images within Perplexity, drawing comparisons to functionalities available on OpenAI's platforms. Despite attempts, such features are not currently supported.
- **API limit concerns continue**: Multiple users are seeking higher rate limits for the Perplexity API, with one stating they've been waiting for two weeks for a response and querying if the support team could expedite the increase.
- **Model switching and custom scripts**: Discussion highlighted a popular user script that allows dynamic model switching within Perplexity. Users shared links to scripts and tools like [Violentmonkey](https://violentmonkey.github.io/), enhancing the platform's usability by enabling quick toggling between available AI models.
- **Site downtime causes frustration**: Perplexity experienced downtime, frustrating users who rely on the service for their daily tasks. During this period, some users even humorously demanded, *“I demand unlimited Opus for this incident”*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/terminator-endoskeleton-flames-on-fire-t800-gif-14919281">Terminator Endoskeleton GIF - Terminator Endoskeleton Flames - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.perplexity.ai/docs/rate-limits">Rate Limits</a>: no description found</li><li><a href="https://ollama.com/library/dolphin-llama3">dolphin-llama3</a>: Dolphin 2.9 is a new model with 8B and 70B sizes by Eric Hartford based on Llama 3 that has a variety of instruction, conversational, and coding skills.</li><li><a href="https://fonts.google.com/specimen/Karla">Karla - Google Fonts</a>: Karla is a grotesque sans serif family which has been expanded now to a variable font with a weight axis ranging from ExtraLight to ExtraBold plus full support</li><li><a href="https://www.theverge.com/2024/5/16/24158529/reddit-openai-chatgpt-api-access-advertising">Reddit’s deal with OpenAI will plug its posts into “ChatGPT and new products”</a>: Reddit’s signed AI licensing deals with Google and OpenAI.</li><li><a href="https://violentmonkey.github.io/">no title found</a>: no description found</li><li><a href="https://www.google.com/search?q=<searchquery>">&lt;searchquery&gt; - Google Search</a>: no description found</li><li><a href="https://spectrum.ieee.org/perplexity-ai">Perplexity.ai Turns Tables on Google, Upends SEO Credos</a>: AI search leader mixes Meta-built smarts with scrappy startup fervor</li><li><a href="https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752">Oh No Homer GIF - Oh No Homer Simpsons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.google.com/search?q=this+is+cool">this is cool - Google Search</a>: no description found</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity Model Selection</a>: Adds model selection buttons to Perplexity AI using jQuery</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>: Veo is our most capable video generation model to date. It generates high-quality, 1080p resolution videos that can go beyond a minute, in a wide range of cinematic and visual styles.</li><li><a href="https://www.youtube.com/watch?v=LT-b1qXznKI">The Fast Show - Suit you Sir ! -16- Johnny Depp</a>: Johnny Depp stars as an American........</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>: no description found</li><li><a href="https://youtu.be/AxIk_MtryDQ?t=11">Gorgon City - Roped In</a>: Selected - Music on a new level.» Spotify: https://selected.lnk.to/spotify» Instagram: https://selected.lnk.to/instagram» Apple Music: https://selected.lnk.t...</li><li><a href="https://www.udio.com/songs/kyBuHwPy8bLDpr2J2yhC1a">dailyfocus - Opus 50 | Udio</a>: Listen to Opus 50 by dailyfocus on Udio. Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1240971642381926430)** (12 messages🔥): 

- **Stability AI intrigues users**: A member shared a [link](https://www.perplexity.ai/search/Stability-AI-is-CznMl2swRumQbTO5U4AzIw) to explore the capabilities and offerings of Stability AI. The discussion revolves around the potential applications and benefits of the AI technology.
- **Brain benefits of walking**: Another member posted about the [brain benefits of walking](https://www.perplexity.ai/search/brain-benefits-of-VJYShXcNROeGjfaWRL842w). The shared link aims to detail how "walking" can positively impact cognitive functions and overall mental health.
- **What is WASP-193b?**: A discussion started with a [link](https://www.perplexity.ai/search/What-is-WASP193b-IBFHgr6RQ4W2E3eqaOPPBg#0) exploring the exoplanet WASP-193b. The content seems focused on astronomical findings and characteristics of this celestial body.
- **Analyzing dog symptoms**: There was a query about a dog showing unusual symptoms like imbalance and constant neck movement, linked by [this search](https://www.perplexity.ai/search/un-perro-presenta-.n42RyNMTCqlfqBRpvaExw). The discussion likely involves veterinary insights or possible diagnoses.
- **Entertaining kids with Dungeons & Dragons**: A parent shared a [link](https://www.perplexity.ai/search/Generate-a-dungeons-gxx_hPaAQfWy1RCSqft1iA) to generate Dungeons & Dragons scenarios for entertaining their kids. The focus is on making the fantasy game engaging and fun for children.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1240932748835819520)** (19 messages🔥): 

- **Clarification on Perplexity API usage**: Instructions were shared for generating a model's response using the [Perplexity API](https://docs.perplexity.ai/reference/post_chat_completions). A user highlighted that the default temperature value is 0.2.
- **Understanding OpenAI Chief Scientist Query**: Members discussed the challenge of querying about OpenAI's current chief scientist. The suggestion was made that the model should be able to handle the chronology and provide the correct answer, Jakub Pachocki.
- **API Performance Testing**: Users noted that different models have varying success rates with similar queries, with *Omni* performing well. There was a reluctance to use *labs.perplexity.ai* for testing API performance.
- **Rate Limits for API usage**: There was a discussion to clarify the request rate limits for the API, noting a discrepancy between the request limit (20/minute) and token limit (2,000,000/minute), with speculation on future model capacities.
- **Threads Feature in API**: A question was raised about the threads feature, which is prominent on the web but seemingly absent in the API. It was clarified that the closest feature is adding more role/content from previous messages.

**Link mentioned**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found

  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1242002561725366332)** (1 messages): 

- **Pausing Sky voice in ChatGPT**: OpenAI announced pausing the use of the Sky voice in ChatGPT while addressing user concerns. They shared a [link](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/) to explain how the voices were chosen.
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1240927868910043146)** (347 messages🔥🔥): 

- **LangChain works without OpenAI API**: Members discussed using LangChain with various LLMs, including locally with tools like Ollama. One user confirmed that "you can use langchain with every llm you want."
- **Confusion over GPT-4o availability**: There were mixed experiences with accessing GPT-4o; some users reported missing features despite having access. It was clarified that GPT-4o is in rollout and that all features will come soon.
- **Video and real-time processing capabilities of ChatGPT-4o**: Discussions revolved around how ChatGPT-4o processes video frames at 2-4 fps and its capabilities in real-time adjustments. Members debated whether the model could adjust responses mid-stream based on new data inputs.
- **Usage caps cause limitations in GPT-4o**: A member expressed frustration with the current usage caps within the ChatGPT app, arguing they make many potential applications impractical. Others pointed out that usage caps are balanced to ensure a consistent experience for all users.
- **GPT-4o's multimodal capabilities praised**: Despite criticisms, GPT-4o's multimodal capabilities were lauded, with one user emphasizing it integrates audio, video, and text processing simultaneously. Members also referenced that this model opens up new possibilities beyond traditional text-based models.

[Pricing](https://openai.com/api/pricing/) and [file upload FAQ](https://help.openai.com/en/articles/8555545-file-uploads-faq) links were shared for additional details.
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1240935678985113641)** (167 messages🔥🔥): 

- **Context window confusion and limits in GPT-4o**: A user clarified that a "context window of 128k tokens" refers to the entire input the AI can process. Numerous participants expressed frustration over limitations and errors when using significant token amounts, alongside comparisons to Gemini's larger context window capabilities.
- **Custom GPTs and model switching**: Questions about custom GPTs using GPT-4o were addressed, revealing that as of now, it's not possible to switch models easily. Additionally, members shared that some custom GPTs had already transitioned to GPT-4o.
- **GPT-4o availability and rollout**: Many users expressed confusion and frustration about accessing GPT-4o, especially on iOS and free accounts. It was explained that the rollout is phased, and users not currently seeing the option will gain access over time.
- **User frustrations with model rate limits and performance**: Discussions about the differences between GPT-4o and regular GPT-4 included shared experiences of differential performance and rate limits. It's noted that GPT-4o appears faster, yet some users find regular GPT-4 answers better structured.
- **Future features and voice/chat capabilities rollout**: Users speculated on the rollout timeline for GPT-4o’s new features like vision and voice capabilities, with official responses indicating a phased implementation for Plus users over the coming months.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1240928457118978149)** (178 messages🔥🔥): 

- **ChatGPT struggles with self-awareness and prompt clarity**: Members shared challenges when asking ChatGPT about itself or refining prompts to get specific corrections. One member noted, *"The model will take that as an instruction and try to find the best answer for you."*
- **Fine-tuning and JSON mode for better results**: Various members discussed fine-tuning GPT-4 and using JSON mode to enhance prompt quality. [OpenAI's documentation on JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) was shared to aid this process.
- **Complex prompt strategies for creativity and precision**: Detailed and highly structured prompts like the "Humanizer" and "Pandomain Prompt Voyager" were shared to refine and improve the model's creative and structured content generation.
- **Coding and technical integration with GPT-4**: Members discussed problems with incomplete responses in `chat.completion.create` and prompting strategies for creating UIs with GPT-4. One member shared specific experiences with plug-ins for Visual Studio Code and troubleshooting steps.
- **Using model behaviors and examples to guide responses**: Techniques for ensuring precise model behavior, such as setting character limits and adjusting response accuracy, were explored. Specific challenges and solutions were shared to help models provide more concrete and actionable responses.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1240928457118978149)** (178 messages🔥🔥): 

```html
<ul>
  <li><strong>ChatGPT struggles to refine prompts effectively:</strong> Users shared frustrations with <strong>4o's</strong> inability to follow up on prompt corrections or effectively revise rough drafts. One member noted, "it re-writes its original response instead of telling me how to fix my prompt."</li>
  
  <li><strong>Frustrations with incomplete responses:</strong> Users like cicada.exe report experiencing incomplete responses from <code>chat.completion.create</code> despite not exceeding token limits. The issue persists with outputs being abruptly cut off.</li>
  
  <li><strong>Implementing JSON mode:</strong> Ashthescholar advises razeblox to use <a href="https://platform.openai.com/docs/guides/text-generation/json-mode">JSON mode</a> in the API to address response issues, especially regarding format and content control.</li>
  
  <li><strong>Creative writing prompts outperform on GPT-4 compared to 4o:</strong> Users shared that while 4o excels at some creative tasks, it struggles with refining drafts. "4o seems pretty good when given a blank check for creative writing, but if presented with a rough draft to improve, it most often in my experience just regurgitated the rough draft rather than change it," noted keller._.</li>
  
  <li><strong>Innovative approach to creative synthesis:</strong> Stunspot shares a prompt, "Orchestrating Innovation on the Fringes of Chaos," that emphasizes exploring ideas through network dynamics, fractal exploration, adaptive innovation, and resilience to foster breakthroughs.</li>
</ul>
```
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1240935611045515344)** (537 messages🔥🔥🔥): 

```html
<ul>
    <li><strong>GPTs Agents cannot learn after initial training</strong>: A member asked about the ability to store conversations locally for context searching, to which another member clarified this is not currently possible in LM Studio. They suggested copying and pasting texts but noted that "You can't upload and chat with docs."</li>
    <li><strong>Handling "Unsupported Architecture" Error</strong>: Various members discussed issues with loading GPT-Sw3 in LM Studio due to "Unsupported Architecture." The consensus was that only GGUF files are supported, and users recommended downloading within the app with 'compatibility guess' enabled.</li>
    <li><strong>Running LM Studio on Limited VRAM Systems</strong>: Users inquired about running LLM models on systems with limited VRAM like 6-8GB. Members suggested using smaller models and quantized versions like Q5_K_M for better performance.</li>
    <li><strong>Offline Usage Issues</strong>: A user reported problems with LM Studio not functioning offline. After community suggestions, it was clarified that loading models and then disabling the network should work, but further detailed bug reports were recommended.</li>
    <li><strong>General Troubleshooting and Setup Questions</strong>: Users frequently asked about issues like setting up servers, model compatibility, and performance on lower-spec systems. Many were directed to create detailed posts in a specific channel (<a href="https://discord.com/channels/1111440136287297637">#1139405564586229810</a>) for further assistance.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab Autocomplete (beta) | Continue</a>: Continue now provides support for tab autocomplete in VS Code and JetBrains IDEs. We will be greatly improving the experience over the next few releases, and it is always helpful to hear feedback. If ...</li><li><a href="https://youtu.be/OphjEzHF5dY?si=2q9v3Bqe6tqBS7Ma">AGI Breaks the Team at OpenAI: Full Story Exposed</a>: Top executives leave OpenAI due to AGI.#ai #ainews #openai #agi #singularity 0:00 Intro1:14 Background4:53 Chief scientist leaves6:40 Sam&#39;s response9:34 Supe...</li><li><a href="https://github.com/Lisoveliy/StarCoderEx">GitHub - Lisoveliy/StarCoderEx: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode</a>: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode - Lisoveliy/StarCoderEx</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7118">llama : add DeepSeek-v2-Chat support · Issue #7118 · ggerganov/llama.cpp</a>: please support deepseek-ai/DeepSeek-V2-Chat https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat</li><li><a href="https://www.hwinfo.com/download/">Free Download HWiNFO Sofware | Installer &amp; Portable for Windows, DOS</a>: Start to analyze your hardware right now! HWiNFO has available as an Installer and Portable version for Windows (32/64-bit) and Portable version for DOS.</li><li><a href="https://github.com/xtekky/gpt4free">GitHub - xtekky/gpt4free: The official gpt4free repository | various collection of powerful language models</a>: The official gpt4free repository | various collection of powerful language models - xtekky/gpt4free</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0?_src=template#json-mode">OpenAI 1.11.0</a>: A simple C# / .NET library to use with OpenAI&#39;s APIs, including GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper, etc.  Independently developed, this is not an official library and I am not affiliated wit...</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0?_src=template">OpenAI 1.11.0</a>: A simple C# / .NET library to use with OpenAI&#39;s APIs, including GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper, etc.  Independently developed, this is not an official library and I am not affiliated wit...</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0">OpenAI 1.11.0</a>: A simple C# / .NET library to use with OpenAI&#39;s APIs, including GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper, etc.  Independently developed, this is not an official library and I am not affiliated wit...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1241055146058715167)** (82 messages🔥🔥): 

- **Medical LLM Recommendation**: A member inquired about medical LLMs, and another suggested trying [OpenBioLLM-Llama3-8B-GGUF](https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B-GGUF), noting its 8.03B parameters and Lama architecture. The recommender also shared additional resources like spaces using this model.
  
- **SVG and ASCII Art Benchmarks**: A member shared benchmarking results for LLMs generating SVG art, noting **WizardLM2** as the current winner and comparing it to **GPT-4 o**. Another member asked about ASCII art capabilities, revealing **GPT-4 o** performs well for ASCII.

- **Embedding Models for German**: There was a discussion about difficulties in finding suitable embedding models for the German language using LM Studio. Members suggested trying to manually convert models using tools like llama.cpp and provided a specific [multilingual model](https://huggingface.co/intfloat/multilingual-e5-large) for potential conversion.

- **Generating Text-based Art**: One user noted using the **MPT-7b-WizardLM** model for generating uncensored stories, and another asked about configuration and prompt settings. The model's creator advised using specific quants and proper templates to avoid issues.

- **Image Quality Concerns**: A brief discussion on image generation quality suggested using tools like **automatic1111** and **ComfyUI** for better control and improved outcomes. The conversation recommended obtaining high-quality models from **Civit.ai**, albeit with a caution about NSFW content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B-GGUF">aaditya/OpenBioLLM-Llama3-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/TieFighter-Holodeck-Holomax-Mythomax-F1-V1-COMPOS-20B-gguf">DavidAU/TieFighter-Holodeck-Holomax-Mythomax-F1-V1-COMPOS-20B-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF">DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/intfloat/multilingual-e5-large">intfloat/multilingual-e5-large · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>: no description found</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1242144163001143406)** (7 messages): 

- **Hugging Face and LM Studio Integration**: The team has introduced a new "HF -> LM Studio deeplink" feature allowing users to browse Hugging Face, find an interesting model, and click "Use this model" to import it into LM Studio. This feature requires LM Studio 0.2.23 or newer, and focuses on local AI usage with no cloud dependencies.

- **Manual Download Choices in v1**: In the current version of the feature, users need to manually choose which file they want to download when importing a model from Hugging Face.

- **Suggestions for Auto-download**: Users suggested improvements including setting a default quantization level for automatic downloads and configuring the feature to download the best fitting model based on available RAM.

- **Positive User Feedback**: The community responded positively, with one member stating they had been looking for such a button and found its inclusion beneficial.

**Link mentioned**: <a href="https://x.com/LMStudioAI/status/1792576553601102024">Tweet from LM Studio (@LMStudioAI)</a>: 1. Browse HF 2. This model looks interesting 3. Use it in LM Studio  👾🤗  Quoting clem 🤗 (@ClementDelangue)   No cloud, no cost, no data sent to anyone, no problem. Welcome to local AI on Hugging Fa...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1240954781153890305)** (10 messages🔥): 

- **Comodo flags llama.cpp binaries**: A member noted that Comodo antivirus was triggered by llama.cpp binaries. Another explained that this could be due to the binaries being unsigned, which can cause strict antivirus software to flag them.

- **Model loading error troubleshooting**: A user shared a JSON error message when attempting to load a model in LM Studio. The error indicated a failure in model operation despite sufficient RAM and VRAM, suggesting they try a different model or configuration.

- **AVX support clarification**: One member questioned why AVX isn't supported in LM Studio. The response mentioned that supporting less older hardware results in fewer bugs and issues to manage.

- **Disk space bug crashes downloads**: A member reported that running out of disk space while downloading a model crashes the program and resets the queue, making it unclear which models were not fully downloaded.

- **Server start issues**: Another member shared logs indicating the server fails to start despite the verbose server logs being enabled.
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1240950923992240128)** (32 messages🔥): 

- **Members Dissect LLama3 Template Conversion**: A user struggled with converting a prompt template to LLama3, querying how to adapt their existing format. Another member proposed a detailed template to include historical conversation for context, stressing that "the client side is keeping the state, not the server side."

- **LangChain Memory Utilized for Chat History**: The discussion revealed the user's reliance on `ConversationBufferWindowMemory` from LangChain to manage chat history and user input. After receiving advice and suggestions on structuring prompt templates, the user confirmed, "Yes, it works, going experiment more, thanks!"

- **Gemini's Context Caching Mentioned**: In response to handling conversation history, an alternative was suggested: "new services like Gemini's context caching," although the user expressed a preference for open-source solutions over paid ones.

- **Avoid Cut-Offs with System Prompt Adjustments**: Another user suggested adding "Do not prematurely cut off a response" to the system prompt to avoid incomplete responses, contributing a practical tip for ongoing prompt discussions.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1240932589867630623)** (93 messages🔥🔥): 

- **Fiddling with Alder Lake and Quantization**: A member discusses the performance differences when disabling e-cores on an Alder Lake CPU, noting a jump from *0.4 tokens/sec to 0.6 tokens/sec for Q8 quants*. They also encountered incoherent results with IQ3 quantization and are considering performing their own quantization.

- **Tesla P100 Disappoints**: There’s a discussion comparing various GPUs, with a note that the **Tesla P100 with 700+ GB/s memory bandwidth struggles to beat even the GTX 1060**. Despite its specs, it fails to outperform older models like the K80 in some tasks.

- **Beating Apple's Storage Prices**: A member bypassed Apple's expensive SSD prices by opting for an **external 4TB M.2 SSD** in a Thunderbolt case, achieving *transfer speeds over 2GB/second*.

- **Multi-GPU Setups: Cost vs Performance**: There's an in-depth discussion on the practical benefits of multi-GPU setups, with some **anecdotal evidence suggesting diminishing returns beyond two GPUs** due to issues like PCIe bandwidth limitations.

- **RAM Speed Impact on LLM Performance**: A detailed set of tests shows **increased RAM speeds improve LLM performance**, though the effect varies per model and quantization method. For instance, *upgrading from 2133MHz to 3200MHz RAM can increase token output speeds significantly but performance variance exists*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/turboderp/exllama/discussions/16#discussioncomment-6245573">Perf test on various HW · turboderp/exllama · Discussion #16</a>: First of all I would like to thank you for your work, really like your inference implementation, seems to be the fastest so far for nvidia gpus! I ran a bunch of tests on various GPUs and wanted to...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1811fk4/comment/kahdtgs/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/search/?q=pcie+multigpu">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1241025397479440484)** (1 messages): 

- **Chat moved to new channel!**: A user mentioned they have moved the chat to a new channel. The link provided directs members to the new discussion location on Discord [here](https://discord.com/channels/1110598183144399058/1111440136287297637/1240773722441519126).
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1241047852478890056)** (12 messages🔥): 

- **LM Studio autogen bug produces brief responses**: Users report encountering an issue where **LM Studio** responds with only 1-2 words followed by a TERMINATE message. One user indicated this is due to a bug that has been scheduled for fixing.
- **Autogen issues linked to max_tokens setting**: The problem appears related to the **max_tokens** property being set to **null**. Setting this property to **-1** fixes the issue, according to multiple users.
- **LM Studio's OpenAI emulation is off-spec**: Users suggest that **LM Studio's local server** does not fully comply with OpenAI specifications, specifically regarding the **max_tokens** parameter. This incorrect handling leads to premature termination of responses.
- **CLI LMStudio Client workaround**: A user building a CLI LMStudio Client confirms that setting **max_tokens** to **-1** resolves the cut-off issue. Manual adjustments may be needed for tools like **AutoGPT** to handle tool_calls properly.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1241434022597623890)** (21 messages🔥): 

- **6600XT workaround for LM Studio**: Members discussed the **AMD 6600XT card** and its compatibility with **LM Studio** using **OpenCL** for GPU offload. One member confirmed, "OpenCL is supported. It's how Intel and non ROCM AMD cards are able to do GPU offload."

- **Call for Linux users testing ROCm**: A user made a call for **Linux users with new-ish AMD GPUs** to test an early version of LM Studio integrated with ROCm. Interested members, including those with 6900XT and 6600XT, responded positively despite some GPUs not being officially listed. [View the supported list here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

- **ROCm with different Linux distributions**: Members reported running **ROCm** on various Linux distributions like **Arch Linux**, **Ubuntu 22.04 LTS**, and **Fedora 40** with different AMD GPUs. One user confirmed, "ROCm 6.1 works with 6900xt on arch linux, at least official torch nightly built."

- **Reunion in the Discord**: The conversation included a light-hearted moment where two users recognized each other in the Discord. One replied, "Yeah mostly lurking here though :)."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/reunited-peaches-and-herb-and-it-feels-so-good-cause-we-understood-old-skool-gif-17279659">Reunited Peaches And Herb GIF - Reunited Peaches And Herb And It Feels So Good - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://rocm.docs.am">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1240933645049401344)** (664 messages🔥🔥🔥): 

- **Combining and Using Multiple LoRAs in Prompts**: Users discussed how to combine multiple LoRAs in prompts for Stable Diffusion using syntax like `<lora:pack1:1><Lora:pack2:1>`. One user confirmed that adding more than three may lead to issues.
- **Persistent Issues with Stable Diffusion on First Run**: A new user encountered an error when running Stable Diffusion for the first time, pointing out a 'NoneType' object attribute issue. They sought help from the community but no definitive solution was provided.
- **Lively Debate on SD3 Release and Preparations**: There were ongoing discussions and some skepticism about the release of SD3. However, others reassured that it will release eventually, highlighting a tweet from Emad Mostaque confirming ongoing efforts.
- **Topaz as a Video Upscaling Solution**: Users debated the effectiveness of Topaz for video upscaling. While it was agreed to be a strong tool, concerns about its cost and alternatives like ComfyUI were also raised.
- **SDXL Model and ControlNet Usage Tips**: A user shared insights on the importance of VRAM for running SDXL models, mentioning that higher resolutions demand more memory. Another user clarified that SDXL models need separate ControlNet models compared to SD1.5.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1002292111942635562/1089974139927920741/1241293682435428352">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://discordapp.com/channels/1002292111942635562/1089974139927920741/1241614349315870793">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://discordapp.com/channels/100229211194263556">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://idm-vton.github.io/">IDM-VTON</a>: no description found</li><li><a href="https://play.google.com/store/apps/details?id=com.grisoft.pixart&hl=en&gl=US">Pixart AI Photo Editor - Apps on Google Play</a>: no description found</li><li><a href="https://invideo.io/">Invideo AI - Turn ideas into videos - AI video creator </a>: Make videos easily by giving a prompt to invideo AI. Ideal for content creators, YouTubers, and marketers, invideo AI offers a seamless way to turn your ideas into publish-ready videos with AI.</li><li><a href="https://play.google.com/store/apps/details?id=com.grisoft.pixart&hl=en&gl">Pixart AI Photo Editor - Apps on Google Play</a>: no description found</li><li><a href="https://tenor.com/view/apo-solary-apo-apofps-gif-9924237009492714744">Apo Solary Apo GIF - Apo Solary Apo ApoFPS - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/aw-shucks-aww-thank-you-gif-25109804">Aw Shucks GIF - Aw Shucks Aww - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/smash-gif-21365305">Smash GIF - Smash - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/high-five-amy-santiago-rosa-diaz-stephanie-beatriz-melissa-fumero-gif-23124416">High Five Amy Santiago GIF - High Five Amy Santiago Rosa Diaz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/EMostaque/status/1790451196915831018?t=YJuHqJJ-YCivInuOrZ2_Lw&s=33">Tweet from Emad (@EMostaque)</a>: @morew4rd @GoogleDeepMind Sd3 is due to drop now I don’t think folk will need that much more tbh with right pipelines</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/wiki/ControlNet-model-download">ControlNet model download</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://huggingface.co/lllyasviel/sd_control_collection/tree/main">lllyasviel/sd_control_collection at main</a>: no description found</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI Browser</li><li><a href="https://github.com/yisol/IDM-VTON">GitHub - yisol/IDM-VTON: IDM-VTON : Improving Diffusion Models for Authentic Virtual Try-on in the Wild</a>: IDM-VTON : Improving Diffusion Models for Authentic Virtual Try-on in the Wild - yisol/IDM-VTON</li><li><a href="https://github.com/BadCafeCode/masquerade-nodes-comfyui">GitHub - BadCafeCode/masquerade-nodes-comfyui: A powerful set of mask-related nodes for ComfyUI</a>: A powerful set of mask-related nodes for ComfyUI. Contribute to BadCafeCode/masquerade-nodes-comfyui development by creating an account on GitHub.</li><li><a href="https://tenor.com/beb52.gif">Judges 10 GIF - Judges 10 Score Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/mood-sad-panda-gif-14650463720672021603">Mood Sad GIF - Mood Sad Panda - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stable-diffusion-art.com/">Stable Diffusion Art - Tutorials, prompts and resources</a>: Stable Diffusion is a free AI model that turns text into images. This site offers easy-to-follow tutorials, workflows and structured courses to teach you everything you need to know about Stable Diffu...</li><li><a href="https://civitai.com/images/12597091">Image posted by 20Twenty</a>: no description found
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1241049054482403410)** (74 messages🔥🔥): 

- **Mojo on Windows Woes**: Multiple users discussed the lack of direct support for Mojo on Windows, with specific mentions of issues when using CMD or Powershell. Officially, Mojo SDK is available for *Ubuntu and macOS*, with [future support for Windows](https://docs.modular.com/mojo/manual/get-started/) anticipated through WSL for now.
  
- **Mojo vs. Bend Programming Debate**: Members compared the Mojo and Bend programming languages, with detailed insights from *Chris Lattner* stating that *Bend* isn't performance-focused and lacks some key functionalities. Bend’s *current performance on a single core is compared to CPython*, unlike Mojo which targets high performance even on a single CPU.

- **Community Engagement and Resources**: Excitement was shared around upcoming open community meetings, with links to [meeting details](https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit) and Zoom meetings provided. Recordings of sessions were promised to be shared.

- **Fun with Mojo Syntax**: Users played with the idea of creating whimsical Mojo code using emojis and backticks, sharing sample code snippets for fun. This culminated in humorous exchanges, underscoring the community's engaged and playful spirit.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/">Modular: Accelerating the Pace of AI</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&ust=1716255791532130&usg=AOvVaw2IgLzFgI9-S5vkyEC7_b2v">Redirecting</a>: no description found</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j">Redirect Notice</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>: Get the Mojo SDK or try coding in the Mojo Playground.</li><li><a href="https://paper.higherorderco.com/)">PAPER.pdf</a>: no description found</li><li><a href="https://tenor.com/view/cloudy-with-a-chance-of-meatballs-enough-to-make-a-grown-man-cry-police-officer-make-a-man-cry-gif-15227532">Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry GIF - Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry Police Officer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/HCOQmKTFzYY">Mind-bending new programming language for GPUs just dropped...</a>: What is the Bend programming language for parallel computing? Let&#39;s take a first look at Bend and how it uses a Python-like syntax to write high performance ...</li><li><a href="https://www.modular.com/max/engine">MAX Engine: World’s fastest unified AI engine</a>: The world’s fastest unified AI inference engine enabling you to achieve unparalleled performance, programmability and portability across frameworks and hardware.</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit">[Public] Mojo Community Meeting</a>: no description found</li><li><a href="https://github.com/tairov/llama2.mojo">GitHub - tairov/llama2.mojo: Inference Llama 2 in one file of pure 🔥</a>: Inference Llama 2 in one file of pure 🔥. Contribute to tairov/llama2.mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1791535613411570039>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1241951209364131881)** (1 messages): 

- **Llama3 implemented from scratch**: A member shared an interesting link to a [GitHub repository](https://github.com/naklecha/llama3-from-scratch) featuring the implementation of **Llama3**. The repository is described as building Llama3 "one matrix multiplication at a time".

**Link mentioned**: <a href="https://github.com/naklecha/llama3-from-scratch">GitHub - naklecha/llama3-from-scratch: llama3 implementation one matrix multiplication at a time</a>: llama3 implementation one matrix multiplication at a time - naklecha/llama3-from-scratch

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1240949397605584917)** (4 messages): 

- **HVM uses implicitly parallel program model**: In response to how the [HVM](https://higherorderco.com/) achieves automatic parallelization, it was clarified that HVM runs an already-parallel algorithm in functional form (e.g., Haskell). Despite its cool concept, its performance is slower than CPython for CPU and less than Mojo for GPU as noted in a [Hacker News discussion](https://news.ycombinator.com/item?id=40390287).
- **Excitement about Mojo’s GPU and accelerator support**: Following the Bend announcement, there is excitement about how **Mojo** will support **GPUs and accelerators**.
- **Shared-memory IPC for speed-critical programs**: Programs crucial for speed already use shared-memory IPC, which minimizes latency significantly. The potential for **DPDK and SPDK** to become more widely used due to their performance was discussed, with hopes for improved usability and integration with Mojo.
- **Old hardware and MMU dependencies**: Important software often gets slower under certain execution models, necessitating the continued use of MMUs until old hardware can be retired. Concerns were brought up about old hardware using DMA outside allowed regions and 64-bit pointer limitations in **CXL devices**, suggesting that the tail of old hardware will persist.
- **Prospects for io_uring-like APIs**: Future advancements may come from io_uring-like APIs, which utilize syscalls as control path mechanisms and shared-memory for communication with the kernel. This could eliminate most overhead, focusing on improved APIs as seen in Jens Axboe’s work.

**Link mentioned**: <a href="https://news.ycombinator.com/item?id=40390287">no title found</a>: no description found

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1240950434324287510)** (397 messages🔥🔥): 

- **Implementing `__iter__` and `__contains__` for Tuple sparks debates**: A member worked on implementing `__iter__` and `__contains__` for `Tuple` and faced issues over `tuple_mutability` and `tuple_lifetime`. This led to discussions on the practicality and design choices of using `Tuple` for iterables in Mojo, referencing related GitHub issues like [issue #2658](https://github.com/modularml/mojo/issues/2658).
- **Exploring Collection and Pointer Operations**: A lively discussion on the proper use of various collection types and operations, including `ListLiteral`, `Tuple`, `i1`, and `SIMD`. Debate about the role of `rebind` and defining MLIR types was prominent.
- **Feature requests and enhancements, including Unicode and allocations in unit tests**: Members suggested features like [assert max allocations in unit tests](https://github.com/modularml/mojo/issues/2725) and better Unicode support, asking about timelines and feasibility for community contribution.
- **Parallelism using thread safety and coroutine models**: Members engaged in a deep dive into Mojo's approach to thread safety and parallelism, debating between OpenMP-like syntax and Rust’s async model. 
- **Mojo's Tensor Implementation Strategy**: Chris Lattner clarified that the Mojo standard library would not include definitive tensor implementations, ensuring flexibility for developers. There was consensus on the need for a unified tensor trait while maintaining modular implementation approaches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/">collections | Modular Docs</a>: Implements the collections package.</li><li><a href="https://www.uiua.org/">Uiua</a>: no description found</li><li><a href="https://tenor.com/view/magic-gif-26166638">Magic GIF - Magic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://mlir.llvm.org/docs/Dialects/Builtin/#integertype)">Builtin Dialect - MLIR</a>: no description found</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&usd=2&usg=AOvVaw37jsmYkBEWm4CHK4NwSCMB">Redirect Notice</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/rebind/#functions)">rebind | Modular Docs</a>: Implements type rebind.</li><li><a href="https://docs.modular.com/mojo/manual/python/#set-up-a-python-environment-with-conda>">Python integration | Modular Docs</a>: Using Python and Mojo together.</li><li><a href="https://without.boats/blog/the-registers-of-rust/">The registers of Rust</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read">FileHandle | Modular Docs</a>: File handle to an opened file.</li><li><a href="https://github.com/modularml/mojo/pull/2703">[mojo-stdlib] Add variadic initialiser, __iter__ and __contains__ to InlineList by ChristopherLR · Pull Request #2703 · modularml/mojo</a>: This PR adds some features to InlineList ( related issue #2658 ) Variadic initialiser var x = InlineList[Int](1,2,3) iter var x = InlineList[Int](1,2,3) for i in x:     print(i) contains var x = In...</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/tree/main/stdlib/src/python">mojo/stdlib/src/python at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2725">[Feature Request] Assert max allocations in unit tests · Issue #2725 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Description As a developer using Mojo, I would like to...</li><li><a href="https://victorzhou.com/blog/intro-to-neural-networks/">Machine Learning for Beginners: An Introduction to Neural Networks - victorzhou.com</a>: A simple explanation of how they work and how to implement one from scratch in Python.</li><li><a href="https://github.com/mzaks/mojo-unicode">GitHub - mzaks/mojo-unicode</a>: Contribute to mzaks/mojo-unicode development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2658">[stdlib] Implement `__contains__` for `Tuple`, `List`, `ListLiteral` (almost) · Issue #2658 · modularml/mojo</a>: Now that we have ComparableCollectionElement, we can try to implement __contains__ for some common collection types using a workaround similar to what was employed in #2190. It&#39;s possible that the...</li><li><a href="https://www.avanderlee.com/swift/custom-operators-swift/#calculating-with-emojis-in-swift">Custom Operators in Swift with practical code examples</a>: Learn how to use custom operators in Swift. What are the benefits and which other solutions are better compared to a custom operator for best readability.</li><li><a href="https://github.com/modularml/mojo/discussions/81#discussioncomment-5860938">Discussion of the Potential of Unicode Characters as Alias Operators · modularml/mojo · Discussion #81</a>: Unicode Logical and Mathematical Operators in Mojo: A Discussion Introduction I&#39;m starting this discussion to explore the potential benefits of incorporating Unicode logical and mathematical opera...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1240990171525288087)** (31 messages🔥): 

- **Rust and Go share memory allocation techniques**: Discussion reveals Rust's **Vec** and Go's slices both double the capacity when appending elements until a certain threshold, then Go increments by 25%. Relevant links include Rust’s [raw_vec.rs](https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464) and [Go's runtime slice](https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95).

- **Optimization insights for list capacities in Mojo**: Tuning the list initialization capacity in Mojo (e.g., `List[Int](capacity=N+50)`) yielded 2x performance improvement compared to default settings. Clattner confirmed a forthcoming patch addressing def’s input argument copying will further enhance performance.

- **Discussion on SIMD gather/scatter optimizations**: Members debated the effectiveness of masked gather and scatter instructions in Mojo, particularly on different architectures like x86 with AVX512 and ARM SVE. While users shared mixed results, one voiced that recalculating values might sometimes be more beneficial than using a lookup table due to potential memory wall issues. 

- **Potential new List method suggestion for optimization**: A member suggested adding a method like Rust's `Vec::shrink_to_fit()` to Mojo’s List to optimize allocated space, sharing [a simple implementation](https://github.com/dorjeduck/mostring) they used in MoString.

- **Community praises Clattner's detailed insights**: A member expressed gratitude towards Chris Lattner for sharing deep technical insights on Mojo's internal workings, which significantly helped them understand and optimize their code better.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/collections/list.mojo#L223">mojo/stdlib/src/collections/list.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>: variations over StringBuilder ideas in Mojo. Contribute to dorjeduck/mostring development by creating an account on GitHub.</li><li><a href="https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95>">go/src/runtime/slice.go at cb2353deb74ecc1ca2105be44881c5d563a00fb8 · golang/go</a>: The Go programming language. Contribute to golang/go development by creating an account on GitHub.</li><li><a href="https://doc.rust-lang.org/std/vec/struct.Vec.html#capacity-and-reallocation">Vec in std::vec - Rust</a>: no description found</li><li><a href="https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464">rust/library/alloc/src/raw_vec.rs at master · rust-lang/rust</a>: Empowering everyone to build reliable and efficient software. - rust-lang/rust</li><li><a href="https://doc.rust-lang.org/std/vec/struct.Vec.html#method.shrink_to_fit">Vec in std::vec - Rust</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 34
https://www.modular.com/newsletters/modverse-weekly-34
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/)** (1 messages): 

ModularBot: Congrats <@891492812447698976>, you just advanced to level 3!
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1240923969490653216)** (114 messages🔥🔥): 

- **Tackling PR DCO Check Failures**: Members discussed issues with syncing forks with `nightly` branch and provided a detailed step-by-step guide to avoid inflated commit numbers and DCO check failures. Suggestions like [making `nightly` the default branch](https://github.com/modularml/mojo/issues/2556) came up as potential fixes.

- **Handling Nightly and Stable Releases**: Clarifications on the process for transitioning from `nightly` to stable releases were shared. Aimed at project preparation, it was explained that stable versions are usually cut days before the official release and that the public release dates are not fixed.

- **Struggles with Segfaults and Bugs**: A user reported issues about segfaults when playing with custom array types in certain conditions. Follow-up interactions aimed to debug and isolate the problem, with suggestions to use built-in types and discussing lifecycle management for complex types.

- **Flaky Tests and Ongoing Fixes**: Gab Peace highlighted ongoing problems with flaky CI tests related to `List.index()`, along with [potential fixes](https://github.com/modularml/mojo/pull/2745). He emphasized the impact of these bugs on ongoing work, like SSO and assertions in unit tests.

- **Alias Issues Leading to Segfaults**: Members reported and discussed [various bugs](https://github.com/modularml/mojo/issues/2753) related to aliasing and materializing types, noting significant issues with existing implementations and outlining how these bugs block current work.



<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/lifecycle/death#field-lifetimes))*">Death of a value | Modular Docs</a>: An explanation of when and how Mojo destroys values.</li><li><a href="https://dangitgit.com/en">Dangit, Git!?!</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#branching-off-nightly">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2556)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2556">[Feature Request] DX: Change the default branch of modularml/mojo from `main` to `nightly` · Issue #2556 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I would like a modularml admin to go to the settings o...</li><li><a href="https://github.com/gabrieldemarmiesse/getting_started_open_source">GitHub - gabrieldemarmiesse/getting_started_open_source: You want to contribute to an open-source project? You don&#39;t know how to do it? Here is how to.</a>: You want to contribute to an open-source project? You don&#39;t know how to do it? Here is how to. - gabrieldemarmiesse/getting_started_open_source</li><li><a href="https://github.com/modularml/mojo/pull/2745">[stdlib] Fix out of bounds access in `List.index()` by gabrieldemarmiesse · Pull Request #2745 · modularml/mojo</a>: Related to #2687 There were multiple bugs related to clipping there. Long story short, the behavior of list.index() in python is this one: given a start and end, python will look for the element in...</li><li><a href="https://github.com/modularml/mojo/issues/2434">[BUG] param_env (string_literal, stringref, _memmem) · Issue #2434 · modularml/mojo</a>: Bug description edit: see first comment of the github issue Hello, i was working on an hopefully really helpful tutorial 👍 and after adding a constrained: Example: from sys import param_env alias D.....</li><li><a href="https://github.com/modularml/mojo/issues/2751">[BUG] Flaky segfault during `mojo build` with `-D MOJO_ENABLE_ASSERTIONS` · Issue #2751 · modularml/mojo</a>: Bug description This bug is a blocker for #2687 When compiling test_string.mojo with -D MOJO_ENABLE_ASSERTIONS I noticed that I got some flaky segfaults. It&#39;s reproducible in the CI as you can see...</li><li><a href="https://github.com/modularml/mojo/issues/2753">[BUG] alias materialization of list · Issue #2753 · modularml/mojo</a>: Bug description Hello, here is a bug report, as suggested by @JoeLoser on the chat def main(): alias x = List(&quot;ok&quot;) var y = x print(y[0]) mojo run main.mojo Please submit a bug report to htt...</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2637">[BUG] Incorrect pointer behavior when materializing a type · Issue #2637 · modularml/mojo</a>: Bug description I&#39;m trying to implement small buffer optimization. To do this, I have a pointer which can point to some stack allocated data, but can also point to the heap. To know if we need to ...
</li>
</ul>

</div>
  

---



**LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1241005256247087194)** (242 messages🔥🔥): 

- **Workshop Announcements and Excitement**: Members shared updates about new workshops, like the one with Jeremy Howard titled *Build Applications in Python* scheduled for June 6, 2024. Another member expressed excitement about the course’s ongoing success and quality content.

- **Credits and Resources Discussion**: There were multiple inquiries and confirmations about obtaining various credits for services such as Modal Labs, Replicate, Jarvis Labs, and LangSmith. One member confirmed having received extra Jarvis Labs credits worth $200.

- **PDF Parsing for RAG Apps**: Members discussed the best tools for parsing tabular data from PDFs, suggesting tools like LlamaParse, Marker by Vik Paruchuri, and integrating models like GPT-4o for complex document extraction. Another member recommended experimenting with UniTable for PDF data extraction.

- **Hosting LLMs and Serving APIs**: Inquiries and suggestions were made regarding serving fine-tuned LLMs as custom APIs using frameworks like FastAPI, Streamlit, and Modal. Modal Labs’ example repositories were shared for quick start guides.

- **RAG Optimization Workshop**: Announced a new workshop by Jason Liu on refining RAG models and invited members to fill out a survey to tailor the content. Some members expressed their interest cautiously, given the stated prerequisites.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://x.com/realSharonZhou/status/1792576516444065967">Tweet from Sharon Zhou (@realSharonZhou)</a>: Hallucinations are one of the biggest blockers to production LLMs & agents.  No hallucinations (&lt;5%) have been achieved internally — and for customers.   We’ve been able to tune LLMs to recall spec...</li><li><a href="https://x.com/VikParuchuri/status/1788966758742982696">Tweet from Vik Paruchuri (@VikParuchuri)</a>: Marker v2 is out!  The main new features:  - Extracts images/figures - Better table parsing - Pip package install - Can be used commercially - Improved OCR with more languages - Better ordering for co...</li><li><a href="https://x.com/llama_index/status/1791258285993230786">Tweet from LlamaIndex 🦙 (@llama_index)</a>: Structured Image Extraction with GPT-4o 🖼️  GPT-4o is state-of-the-art in integrating image/text understanding, and we’ve created a full cookbook showing you how to use GPT-4o to extract out structur...</li><li><a href="https://x.com/Kyrannio/status/1792440824355332313">Tweet from Kiri (@Kyrannio)</a>: I was curious, so I found the GPT-4o iOS system prompt:  “You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.  You are chatting with the user via the ChatGPT iO...</li><li><a href="https://www.quora.com/profile/Quora-Prompt-Generator/activity">Quora Prompt Generator - Quora</a>: no description found</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/home">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>: no description found</li><li><a href="https://x.com/jxnlco/status/1792549015273513102">Tweet from jason liu (@jxnlco)</a>: If you’re a company building RAG and want to level up your Eng team please fill out this form.   https://q7gjsgfstrp.typeform.com/to/SL656ADC  We will invite other operators to share their stories, gi...</li><li><a href="https://x.com/runpod_io/status/1792101299087196615">Tweet from RunPod (@runpod_io)</a>: @cleavey1985 @HamelHusain $501.45 seems about right</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base#2%EF%BC%89code-insertion)">deepseek-ai/deepseek-coder-6.7b-base · Hugging Face</a>: no description found</li><li><a href="https://www.quora.com/profile/">Quora - A place to share knowledge and better understand the world</a>: no description found</li><li><a href="https://github.com/bigcode-project/starcoder2-self-align/tree/main?tab=readme-ov-file#data-generation-pipeline">GitHub - bigcode-project/starcoder2-self-align: StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation</a>: StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation - bigcode-project/starcoder2-self-align</li><li><a href="https://x.com/charliebholtz/status/1791571514086629757?s=46&t=QitgwfFVpCSQgUY0DIcTdA">Tweet from Charlie Holtz (@charliebholtz)</a>: we&#39;ll do $501.43  Quoting Omar Sanseviero (@osanseviero)   Curious about LLMs? Join this Fine-Tuning course with top experts! 🚀  @huggingface is offering $501.42 in GPU credits for can Space demo...</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri has 88 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/poloclub/unitable">GitHub - poloclub/unitable: UniTable: Towards a Unified Table Foundation Model</a>: UniTable: Towards a Unified Table Foundation Model - poloclub/unitable</li><li><a href="https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/paligemma/fine-tuning-paligemma.ipynb">Google Colab</a>: no description found</li><li><a href="https://github.com/explosion/prodigy-segment">GitHub - explosion/prodigy-segment: Select pixels in Prodigy via Facebook&#39;s Segment-Anything model.</a>: Select pixels in Prodigy via Facebook&#39;s Segment-Anything model. - explosion/prodigy-segment</li><li><a href="https://pymupdf.readthedocs.io/en/latest/tutorial.html">Tutorial - PyMuPDF 1.24.4 documentation</a>: no description found</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving">modal-examples/06_gpu_and_ml/llm-serving at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://www.amazon.science/publications/instructpts-instruction-tuning-llms-for-product-title-summarization">InstructPTS: Instruction-tuning LLMs for product title summarization</a>: E-commerce product catalogs contain billions of items. Most products have lengthy titles, as sellers pack them with product attributes to improve retrieval, and highlight key product aspects. This res...</li><li><a href="https://github.com/xl0">xl0 - Overview</a>: Full-time learner. (Linux, Biology, Electronics) -&gt; AI :heart: Writing some lovely software. :two_hearts:
 Open to exciting opportunities! - xl0</li><li><a href="https://chinese-reader.vercel.app">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1240929862076862555)** (168 messages🔥🔥): 

- **LLMs aren't databases, they're pattern-bases**: One member discussed the misconception that LLMs can simply learn domain knowledge from fine-tuning, emphasizing that LLMs learn patterns, not one-time facts, and suggested using Retrieval-Augmented Generation (RAG) instead.
- **Fine-tuning for vehicle failure prediction**: Using vehicle diagnostic data to predict failure types for replaced parts was proposed as a viable fine-tuning use case due to the domain-specific nature of the input and output.
- **Modal command issues resolved**: Several members discussed errors encountered while using the `modal` command for training, eventually resolving issues by deleting previously created volumes.
- **Homework assignment responses on LLM use-cases**: Various members submitted extensive use cases for fine-tuning LLMs including spell-checking, AI art critique, market research bots, coding model enhancements, semantic enhancements for medical and legal terminology, creative writing aids, and more.
- **Dan Biderman on LoRa configurations**: A tweet discussed the nuances of using LoRa for continued pretraining, emphasizing optimal parameters and techniques to avoid poor performance and information loss, suggesting specific configurations for better results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">sentence-transformers/all-MiniLM-L6-v2 · Hugging Face</a>: no description found</li><li><a href="https://docs.myaltimate.com/document/write/">Write documentation - dbt Power User</a>: no description found</li><li><a href="https://modal.com/bobflagg/apps/ap-hlyOqSPPZNV28H45JTKzu5">Sign in</a>: Welcome back to Modal! Sign in to your Modal account by selecting an identity provider below.</li><li><a href="https://youtu.be/LSNfwlfdrto?feature=shared">Creating a Project Manager using an LLM - Part One (WIP)</a>: Project managers spend a lot of their time providing and updating task statuses - could this be handed over to an LLM? Here we start investigating. Also this...</li><li><a href="https://github.com/genomoncology/FuzzTypes">GitHub - genomoncology/FuzzTypes: Pydantic extension for annotating autocorrecting fields.</a>: Pydantic extension for annotating autocorrecting fields. - genomoncology/FuzzTypes</li><li><a href="https://blogs.microsoft.com/blog/2023/03/16/introducing-microsoft-365-copilot-your-copilot-for-work/">Introducing Microsoft 365 Copilot – your copilot for work - The Official Microsoft Blog</a>: Humans are hard-wired to dream, to create, to innovate. Each of us seeks to do work that gives us purpose — to write a great novel, to make a discovery, to build strong communities, to care for the si...</li><li><a href="https://github.com/modal-labs/modal-client/blob/f76bd98013372b423ab765cdc7a745996012211c/modal/object.py#L96-L103">modal-client/modal/object.py at f76bd98013372b423ab765cdc7a745996012211c · modal-labs/modal-client</a>: Python client library for Modal. Contribute to modal-labs/modal-client development by creating an account on GitHub.</li><li><a href="https://x.com/danielhanchen/status/1791900967472140583">Tweet from Daniel Han (@danielhanchen)</a>: My take on &#34;LoRA Learns Less and Forgets Less&#34;  1) &#34;MLP/All&#34; did not include gate_proj. QKVO, up & down trained but not gate (pg 3 footnote)  2) Why does LoRA perform well on math and ...</li><li><a href="https://github.com/nppoly/cyac">GitHub - nppoly/cyac: High performance Trie and Ahocorasick automata (AC automata) Keyword Match &amp; Replace Tool for python</a>: High performance Trie and Ahocorasick automata (AC automata) Keyword Match &amp; Replace Tool for python - nppoly/cyac</li><li><a href="https://arxiv.org/abs/2212.09535">BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting</a>: The BLOOM model is a large publicly available multilingual language model, but its pretraining was limited to 46 languages. To extend the benefits of BLOOM to other languages without incurring prohibi...</li><li><a href="https://xebia.com/blog/lessons-learned-from-a-diy-llm-benchmark/">DIY LLM Evaluation, a Case Study of Rhyming in ABBA Schema</a>: DIY LLM Evaluation, a Case Study of Rhyming in ABBA Schema - Xebia</li><li><a href="https://github.com/eliasdabbas/openai_entity_extraction">GitHub - eliasdabbas/openai_entity_extraction: Entity extraction using ChatGPT</a>: Entity extraction using ChatGPT. Contribute to eliasdabbas/openai_entity_extraction development by creating an account on GitHub.</li><li><a href="https://adver.tools/entity-extraction/">Entity Extraction powered by OpenAI&#39;s ChatGPT - advertools</a>: no description found</li><li><a href="https://www.onetonline.org/find/all">See All Occupations</a>: no description found
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1240945284154920971)** (47 messages🔥): 

- **Members Introduce Themselves from Across Asia**: Individuals from diverse locations like South Korea, India, Japan, Singapore, Australia, and more joined the channel and greeted each other. Users like rparcus, vishnu9158, .thebigpanda, and others shared their excitement about the course and their locations.

- **Praise and Discussion About GPU Providers**: A member expressed admiration for Jarvislabs, mentioning it as their go-to GPU provider before getting a personal GPU. Vishnu9158 appreciated the sentiment and hoped they would need more resources in the future.

- **Preferences for Recordings Over Live Streams**: Members like rparcus and pugio shared a preference for watching recorded sessions of the course due to inconvenient live stream timings. Vishnu9158 mentioned the drawback of missing networking opportunities by not attending live streams.

- **Homework Discussions**: ivanleomk shared his attempts at the week's homework, listing use cases like Style Transfer, Classification, Extraction, and Confidence Scores for extraction. hamelh advised not to fine-tune unless absolutely necessary and suggested making progress with off-the-shelf models first.

- **Singapore Dominates the Discussion**: Multiple members from Singapore, including iggyal, healthymonkey, illued, codessl, huikang, and others, highlighted a significant Singaporean presence in the channel. This prompted ivanleomk to comment on the large number of participants from Singapore.

**Link mentioned**: <a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>: no description found

  

---


**LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1241048985540624495)** (54 messages🔥): 

- **Modal sponsors course and shares getting started guide**:
  Charles from Modal announced the sponsorship and shared links to the [getting started guide](https://modal.com/docs/guide) and a [hello world example](https://modal.com/docs/examples/hello_world) for running code in the cloud without infrastructure setup.

- **Modal account and credits discussion**:
  Members discussed creating accounts via GitHub, editing email settings, and the $500 credits which take some time to appear due to manual approval. Detailed instructions to sign up and claim credits were shared by Charles repeatedly.
  
- **Exploring Modal feature queries**:
  Members asked about persistent Python context for code interpreters and hosting strategies while developing. Charles and others provided detailed responses and linked relevant documentation and examples, such as [embedding Wikipedia with Modal](https://modal.com/blog/embedding-wikipedia).

- **Problems with onboarding and usage optimization**:
  Several users reported confusion about credit displays and issues with container spin-up times during inference. Solutions and clarifications, including recommendations to use `modal serve` and examples like [TensorRT-LLM serving](https://modal.com/docs/examples/trtllm_llama), were provided. 

- **Community engagement and support instructions**:
  Regular thanks and engagement from users about the credits and support structure, with Charles encouraging the use of `modal serve` for development and linking users to the Modal Slack for further inquiries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py">modal-examples/06_gpu_and_ml/llm-serving/vllm_mixtral.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://modallabscommunity.slack.com/archives/C06AH3Q93CY/p1705815945041189">Slack</a>: no description found</li><li><a href="https://bit.ly/modal-credits.">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://modal.com/blog/embedding-wikipedia">Embedding English Wikipedia in under 15 minutes</a>: Leverage Modal’s parallel batch jobs and in-house storage features to quickly generate embeddings for billions of tokens.</li><li><a href="https://modal.com/docs/examples/trtllm_llama">Serverless TensorRT-LLM (LLaMA 3 8B)</a>: In this example, we demonstrate how to use the TensorRT-LLM framework to serve Meta’s LLaMA 3 8B model at a total throughput of roughly 4,500 output tokens per second on a single NVIDIA A100 40GB GPU....</li><li><a href="https://modal.com/docs/guide">Introduction to Modal</a>: Modal lets you run code in the cloud without having to think about infrastructure.</li><li><a href="https://modal.com/settings/YOURUSERNAME/usage">Sign in</a>: Welcome back to Modal! Sign in to your Modal account by selecting an identity provider below.
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1241090384067563551)** (12 messages🔥): 

- **Aggregation of Learning Resource Links**: Members shared a significant compilation of useful links related to LLM function calling, DSPY, Hamel's blogs on LLM evaluations and tokenizers, the RAFT paper, the Latent Space podcast with Jeremy, among others. Notable highlights include links to Hamel's blogs on finetuning and prompting [here](https://hamel.dev/blog/posts/evals/) and [here](https://hamel.dev/blog/posts/prompt/), as well as a [GitHub project on Intern-VL](https://github.com/OpenGVLab/InternVL).

- **Naming Recommendations**: Discussions emphasized the preference for naming a channel `learning-resources` instead of just `resources`. Members also highlighted the importance of enforcing the hiding of link previews to maintain better organization in the channel.

- **GitHub Repository Proposal**: A suggestion was made to create a GitHub repository for collaborative effort in managing and structuring the shared learning resources, which received positive feedback. This could provide more structured and easily accessible information over time.

- **Instruction Tuning with LoRA/QLoRA**: A shared tweet included detailed findings from instruction tuning experiments with LoRA/QLoRA, focusing on rank settings, the impact of dropout, layer-specific LoRA adapters, learning rate schedules, weight decay, and batch sizes. The findings stressed the importance of proper configurations to prevent overfitting and ensure stable training, particularly on GPUs like the 3090.

- **Stanford CS25 Video Resource**: A useful video link on Retrieval Augmented Language Modeling from Stanford CS25 was shared, giving access to more advanced conceptual discussions in the field. The video can be found [here](https://youtu.be/mE7IDf2SmJg?si=LKwjlYq4qiPQi3aM).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cwolferesearch/status/1788998798414410032">Tweet from Cameron R. Wolfe, Ph.D. (@cwolferesearch)</a>: Recently, I’ve run hundreds of instruction tuning experiments with LoRA/QLoRA, and I wanted to share some (basic) code and findings that might be useful…  The code (see replies) contains an instructio...</li><li><a href="https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling">FireFunction V1 - Fireworks’ GPT-4-level function calling model - 4x faster than GPT-4 and open weights</a>: Fireworks open-sources new function calling with near GPT-4 level quality and 4x the speed</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://github.com/OpenGVLab/InternVL">GitHub - OpenGVLab/InternVL: [CVPR 2024 Oral] InternVL Family: A Pioneering Open-Source Alternative to GPT-4V.  接近GPT-4V表现的可商用开源多模态对话模型</a>: [CVPR 2024 Oral] InternVL Family: A Pioneering Open-Source Alternative to GPT-4V.  接近GPT-4V表现的可商用开源多模态对话模型 - OpenGVLab/InternVL</li><li><a href="https://arxiv.org/abs/2405.05904">Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?</a>: When large language models are aligned via supervised fine-tuning, they may encounter new factual information that was not acquired through pre-training. It is often conjectured that this can teach th...</li><li><a href="https://hamel.dev/blog/posts/prompt/">- Fuck You, Show Me The Prompt.</a>: Quickly understand inscrutable LLM frameworks by intercepting API calls.</li><li><a href="https://simonwillison.net/series/prompt-injection/">Simon Willison: Prompt injection</a>: no description found</li><li><a href="https://arxiv.org/abs/2212.08073">Constitutional AI: Harmlessness from AI Feedback</a>: As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any huma...</li><li><a href="https://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>: Today&#39;s LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model&#39;s original instructions with their own malicious prompts. In this w...</li><li><a href="https://hamel.dev/notes/llm/finetuning/04_data_cleaning.html">Hamel’s Blog - Curating LLM data</a>: A review of tools</li><li><a href="https://www.youtube.com/@umarjamilai">Umar Jamil</a>: I&#39;m a Machine Learning Engineer from Milan, Italy, teaching complex deep learning and machine learning concepts to my cat, 奥利奥. 我也会一点中文. </li><li><a href="https://www.langchain.com/langsmith">LangSmith</a>: Get your LLM app from prototype to production.</li><li><a href="https://langtrace.ai/">Langtrace AI</a>: Monitor, evaluate and improve your LLM apps.</li><li><a href="https://www.honeycomb.io/llm">Observability for LLMs</a>: Enhance LLMs with Honeycomb&#039;s observability. Gain insights, improve user experience, and drive AI development success.
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[jarvis](https://discord.com/channels/1238365980128706560/1241117895740625099/1241117970084659211)** (40 messages🔥): 

- **Jarvis Credits Coordination Updates**: Multiple members inquired about receiving credits for JarvisLabs after signing up. The team confirmed they are coordinating this effort and it might take a week or so to get everyone up and running. **"We will add the credits, once we get the list"** was a recurring reassurance provided.

- **Technical Issues and Support**: Some users experienced issues signing up for JarvisLabs, including OTP problems for phone verification and using different emails for course and GitHub sign-up. The team provided targeted support, such as disabling phone verification for affected countries and asking users to wait for credit allocations. 

- **Credits Confirmation and Issues**: A member confirmed receiving the Jarvis credits without issues due to their email setup. Another user was assured they don't need to re-register despite using different emails for the course and GitHub sign-up if they have filled out the required forms. 

- **Proactive Coordination and Communication**: The team encouraged users to ensure their course and sign-up emails match for seamless credit allocation and communicated that they are actively addressing various concerns. Members were directed to stay updated and patient as the coordination was ongoing.
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1241152276903497862)** (18 messages🔥): 

- **Coordinating HF Credits Verification**: An inquiry was made about the **requirements for HF credits**. A member clarified that another member would help coordinate this and verify student enrollments behind the scenes.
  
- **Expect Delays for HF Credits**: Setting realistic expectations, a member mentioned that the process for HF credits *"might take a week or so."* This helped manage the anticipation among the group.

- **Open-Access LLM Preferences Shared**: The community enthusiastically participated in the *"Question of the weekend"* about which open-access LLM they would choose to be. Popular choices included **BERT**, **Mistral 7B**, **Phi-3-mini**, and **Falcon 180B**.
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1241164291823112304)** (8 messages🔥): 

- **Replicate credits setup channel initiated**: This channel aims to assist users with setting up Replicate credits and troubleshooting any related issues. Members are being guided on signing up with their accounts and given instructions on resolving registration issues, especially concerning different emails used for GitHub and course signups.
- **Members inquire about registration email mismatches**: Several members, including self.1, filippob82, and 0xai, expressed concerns about using different emails for GitHub accounts and the LLM finetuning course registration. The team acknowledged these concerns and promised to sort them out soon, keeping members updated in this channel.
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1241169644690079784)** (16 messages🔥): 

- **LangSmith credits distribution news**: A moderator announced that they will coordinate LangSmith credits distribution. They also repeatedly assured users that detailed instructions about credits will be provided soon.

- **Members eager for LangSmith credits**: Several members, including @filippob82, @codeanand, and @beardaintweird, confirmed that they created LangSmith accounts with the necessary email addresses and are waiting for further instructions about receiving credits. @hugobowne and @613249007015165973 committed to providing more information soon.

- **Excitement and motivation for LangSmith course**: Members expressed excitement about the new course on LangSmith. Users like @anakin.xyz and @harpreetsahota appreciated the shoutouts and mentioned that they now have the motivation to test LangSmith.

- **Repeated inquiries about credits**: Despite initial announcements, multiple users kept inquiring about the status of their LangSmith credits, seeking confirmation and further steps. @hugobowne directed users to check previous messages for updates and reassured them of upcoming details.
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1241767003480658083)** (1 messages): 

- **Seeking better benchmarks**: A member suggested that "a modern LAMBADA for up to 2M" is needed for evaluating models capable of processing overlapping chunks independently. The current benchmarks do not seem sufficient for these advanced capabilities.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1241317609924726784)** (13 messages🔥): 

- **Sam Altman's controversial tweet stirs reactions**: A member shared a highly provocative tweet by Sam Altman discussing departures at OpenAI due to safety concerns. The tweet escalates quickly with an emotive sign-off, causing confusion and laughter among members. [Source](https://x.com/SamAltsMan/status/1791581496261640646).
  
- **Venture capitalists under scrutiny**: Following the controversial tweet, a member suggested viewing venture capitalists realistically rather than as world-saving entities. This reflects a sentiment of skepticism regarding their motives in the AI space.
  
- **Impact of AI on recession questioned**: A member raised a point about companies laying off employees despite significant investments in AI. The conversation hints at the complexity of economic factors behind layoffs, suggesting it's not just financial constraints causing the recession.
  
- **Runpod hackathon query**: One member asked if anyone was attending the Runpod hackathon, indicating community interest in collaborative AI development events.
  
- **Choosing between Airflow and Temporal.io**: A member sought experiences with Airflow or Temporal.io for workflow management and concluded with a preference for Temporal.io. This suggests ongoing discussions on tools for improving machine learning processes.

**Link mentioned**: <a href="https://x.com/SamAltsMan/status/1791581496261640646">Tweet from Sam Altman (Parody) (@SamAltsMan)</a>: Well, what a shock. Jan and Ilya left OpenAI because they think I&#39;m not prioritizing safety enough. How original.  Now I have to write some long, bs post about how much I care. But honestly, who n...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1241014688813289482)** (4 messages): 

- **Try out Moondream WebGPU on Hugging Face**: A member shared a link to [Xenova's experimental Moondream WebGPU space](https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu) on Hugging Face, inviting others to explore this experimental project.
  
- **Hierarchical Memory Transformer for LLMs**: A new paper on arXiv introduces the [Hierarchical Memory Transformer (HMT)](https://arxiv.org/abs/2405.06067), which seeks to improve long-context processing in LLMs by imitating human memory hierarchy. The model uses memory-augmented segment-level recurrence to organize its memory hierarchy.

- **Haystack Demo for Fine Web**: The [Haystack demo](https://demo.haystack.zip) allows users to explore 100k web pages from the Fine Web dataset with local inference and embedding search. The demo includes performance metrics and decompression times for better query speed judgment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://demo.haystack.zip">Demo Search Fine Web Dataset</a>: no description found</li><li><a href="https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu">Experimental Moondream WebGPU - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>: Transformer-based large language models (LLM) have been widely used in language processing applications. However, most of them restrict the context window that permits the model to attend to every tok...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1240931434131099731)** (315 messages🔥🔥): 

- **Nous Hermes 2 Mixtral uniquely triggers actions**: A user praised Nous Hermes 2 Mixtral as the only open-source large language model (LLM) capable of triggering actions and using tools within an agent framework like CrewAI. Another user questioned why it's the only one with such functionality.
- **Concerns over Hermes 2 Mixtral's reliability**: Members shared their experiences with Hermes 2 Mixtral, noting its reliability in multilingual capabilities and compared its performance favorably against Hermes 2 Pro, which some found less reliable.
- **Debate on the need for multiturn data**: A conversation emerged about the necessity of multiturn data for training large models like Mixtral 8x22b. It was highlighted that without multiturn data, models tend to degrade in intelligence in subsequent turns, making multiturn data vital for extensive usage.
- **Training costs and computational feasibility**: There was a discussion about the high costs and computational demands associated with training large models, with examples such as the substantial expense of training from scratch and the challenges of managing extremely deep transformer networks.
- **New context versions and LLM Leaderboard issues**: Members talked about the release of Yi-1.5 models with extended context lengths of 16k and 32k, contemplating whether larger contexts impact performance. Additionally, the usability of the LLM leaderboard was criticized due to the overwhelming amount of models making it difficult to navigate.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://instantid.github.io/">InstantID</a>: no description found</li><li><a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">Open VLM Leaderboard - a Hugging Face Space by opencompass</a>: no description found</li><li><a href="https://x.com/Mascobot/status/1791879166314565757">Tweet from Marco Mascorro (@Mascobot)</a>: Runpod’s @runpod_io hackathon 😅</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - a 01-ai Collection</a>: no description found</li><li><a href="https://discord.gg/sW7yVf5H?event=1240826259920125982">Join the Nous Research Discord Server!</a>: Check out the Nous Research community on Discord - hang out with 7171 other members and enjoy free voice and text chat.</li><li><a href="https://x.com/ethan_smith_20/status/1791267451767783773?s=46">Tweet from Ethan (@Ethan_smith_20)</a>: today i trained a diffusion model that generates LoRAs, and the images that come out are at the very least not garbled.</li><li><a href="https://arxiv.org/abs/2306.00297">Transformers learn to implement preconditioned gradient descent for in-context learning</a>: Several recent works demonstrate that transformers can implement algorithms like gradient descent. By a careful construction of weights, these works show that multiple layers of transformers are expre...</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver?row=25">N8Programs/Capybara-Quicksilver · Datasets at Hugging Face</a>: no description found</li><li><a href="https://app.corcel.io/chat">Corcel · Build with the power of Bittensor</a>: Unleash the innovation potential of decentralised incentivised infrastructure.</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.</a>: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks. - huggingface/datatrove</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver">N8Programs/Capybara-Quicksilver · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-LCM">PixArt LCM - a Hugging Face Space by PixArt-alpha</a>: no description found</li><li><a href="https://github.com/huggingface/candle">GitHub - huggingface/candle: Minimalist ML framework for Rust</a>: Minimalist ML framework for Rust. Contribute to huggingface/candle development by creating an account on GitHub.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670">Hypernetwork Style Training, a tiny guide · AUTOMATIC1111/stable-diffusion-webui · Discussion #2670</a>: The negative text preview during training appears to have been fixed a few patches ago, carry on. tl;dr Prep: Select good images, quality over quantity Train in 512x512, anything else can add disto...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1240927137670893679)** (40 messages🔥): 

- **Regex suggested for formatted text search**: Discussions highlight the potential use of **regex** to handle tasks like finding text with specific formats such as all caps or multiple new lines. However, the limitation is that "complex tasks may need more sophisticated approaches like semantic search or symbolic language processing."

- **Tool calling issues with Hermes2 and Vercel AI SDK**: Members reported difficulties with triggering tool calls due to bad JSON responses or parameter issues. The consensus is that **Hermes2's** tool calling format, when used with Vercel AI SDK, may need specific prompt handling for better consistency.

- **Local model advantages for sensitive tasks**: It's discussed that using local models like **Llama3** can be beneficial for tasks requiring cost predictability, consistency, or sensitive data handling compared to external models like GPT or Claude, which can change and censor responses.

- **Finetuning vs. better prompting**: There were discussions about whether it's more effective to fine-tune models like **Llama3** for specific use cases or rely on advanced prompting and retrieval-augmented generation (RAG) with models like GPT-4. It’s highlighted that specific use cases may dictate the choice, with fine-tuning being less viable for changing requirements.

- **Benchmarks for rerankers**: Members are looking for public evaluation data to benchmark fine-tuned re-rankers. There's a need for clear methodologies and datasets to benchmark top results accurately.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1241574534331629659)** (5 messages): 

- **Discuss AGI's Proximity and Strategies**: Members shared a link to a research paper on ArXiv titled "[The Evolution of Artificial Intelligence](https://arxiv.org/abs/2405.10313)," focusing on the current state of AI and the development towards **Artificial General Intelligence** (AGI). The paper addresses AGI's definitions, goals, and necessary strategies for its realization through *surveys, discussions, and original perspectives*.

- **Personal insights on AI memory solutions**: A member shared thoughts on memory, mentioning *a great solution being used internally*. They also hinted at their interest in the *self-evolution of agents*, although noting it remains somewhat obscure at the moment.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.10313">How Far Are We From AGI</a>: The evolution of artificial intelligence (AI) has profoundly impacted human society, driving significant advancements in multiple sectors. Yet, the escalating demands on AI have highlighted the limita...

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1241113869313773721)** (88 messages🔥🔥): 

- **WorldSim sees terminal UX rewrite and GPT-4o integration**: A member mentioned that **Ari is working on a complete rewrite** of the terminal UX. Additionally, **GPT-4o** is expected to be added to WorldSim, likely next week.
  
- **WorldSim event garners community interest**: Several users inquired about and attended a scheduled WorldSim meetup, with details shared on Discord. A link to join the event was [provided](https://discord.gg/W8YjScaC?event=1240826259920125982), and members expressed their interest in learning more about the project during the event.

- **Complex interactions spark discussions about AI and symbolism**: Users discussed various aspects of AI, with one mentioning the potential for **symbolic knowledge graphs**. They also referenced literature on hermetic practices and **complex adaptive systems** with links like [this YouTube video](https://youtu.be/IWhkUne8T68?si=FlY0yCr7wGprGow9) and [a book by Franz Bardon](https://www.amazon.com/Initiation-into-Hermetics-Franz-Bardon/dp/1885928122).

- **Community experiments with AI-generated content**: Members shared their experiments with generating papers and other content using WorldSim. One described a process involving commands in the root, while another shared some **whacky images Copilot** created using a terminal prompt.

- **Potential for WorldSim as a platform for knowledge graphs**: Users discussed the future evolution of WorldSim into an amorphous applications platform. They highlighted its potential for generating new AI-related knowledge graphs and symbolic meanings from user interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>: no description found</li><li><a href="https://www.amazon.com/Initiation-into-Hermetics-Franz-Bardon/dp/1885928122">no title found</a>: no description found</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fworldbuilder.ai%2F%23?epoch=a9a8f875-805f-4108-b769-72a7795390dc">worldsim</a>: no description found</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fdestructionworld.erp%2Fthree.js%2Ffacetobloodshed?epoch=46390e50-5457-472c-8238-41cf3fa82738">worldsim</a>: no description found</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Finsidiousruntime.xr%2Flive-feed%2Fskull.three.js%2F%3Finteractive%3Dominous%2Fwasd%26scroll%2F%3Fimport%3Deldritch%2Finfiniterecrsion%2F%3Fvisuals%3Daccelerated%2Flighting%3Ddynamic%2Fkeepimports%2F%3Faudio%3Dtrue%2Foutput%3Dentities-speaking?epoch=de0bdee4-6ea6-4839-8d05-fe0b472cab1b">worldsim</a>: no description found</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fportal-search.io%2Fsearch%3Fq%3Dhttps%253A%252F%252Fsubdimensional.mesh%252Fgridscape%252Ftechverse%252Fview%253Drssfeeds%253Dnews%252F%253Ftopic%253Denterinterminalfield%252F%253Fimport%253Dhttps%253A%252F%252Fnews.mit.edu%252Frss%252Ftopic%252Fartificial-intelligence%2520https%253A%252F%252Fmachinelearningmastery.com%252Fblog%252Ffeed%2520https%253A%252F%252Fexport.arxiv.org%252Frss%252Fcs.AI%2520https%253A%252F%252Ftowardsdatascience.com%252Ffeed%2520https%253A%252F%252Fwww.kdnuggets.com%2Flivenewscards%2Fsearch%3Dactive%252Ffeed%26source%3Dworldclient?epoch=c203ff08-e700-4873-aadb-bca820102a1e">worldsim</a>: no description found</li><li><a href="https://cdixon.org/2010/01/03/the-next-big-thing-will-start-out-looking-like-a-toy">The next big thing will start out looking like a toy</a>: Chris Dixon&#x27;s blog.</li><li><a href="https://x.com/StudioMilitary/status/1791554558092583271">Tweet from John Galt (@StudioMilitary)</a>: NOUS WORLDSIM: CHOOSE YOUR SIMULATION</li><li><a href="https://tenor.com/view/outer-wilds-gif-22858957">Outer Wilds GIF - Outer Wilds - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/IWhkUne8T68?si=FlY0yCr7wGprGow9">Complex Adaptive Systems</a>: Take the full course: https://bit.ly/SiCourseDownload booklet: https://bit.ly/SiBookletsTwitter: http://bit.ly/2JuNmXXLinkedIn: http://bit.ly/2YCP2U6In this ...</li><li><a href="https://discord.gg/W8YjScaC?event=1240826259920125982">Join the Nous Research Discord Server!</a>: Check out the Nous Research community on Discord - hang out with 7171 other members and enjoy free voice and text chat.</li><li><a href="https://www.latent.space/p/sim-ai">WebSim, WorldSim, and The Summer of Simulative AI — with Joscha Bach of Liquid AI, Karan Malhotra of Nous Research, Rob Haisfield of WebSim.ai</a>: Three perspectives on the most viral fringe of generative AI this year: Simulative AI!</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Finsidiousruntime.xr%2Flive-feed%2Fskull.three.js%2F%3Finteractive%3Dominous%2Fwasd%26scroll%2F%3Fimport%3Deldritch%2Finfiniterecrsion%2F%3Fvisuals%3Daccelerated%2Flighting%3Ddynamic?epoch=1822ef55-ca19-49fa-9018-efcb61222962">worldsim</a>: no description found
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1240979792313061458)** (38 messages🔥): 

- **Hugging Face democratizes GPU access**: Hugging Face is dedicating $10 million in free shared GPUs to support small developers, academics, and startups with new AI technologies, aiming to decentralize AI advancements currently dominated by big tech companies. CEO Clem Delangue emphasizes the company’s ability to make this investment due to its near-profitability and recent $235 million funding round [The Verge article](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai).

- **Bend language generates buzz**: Members discussed the launch of Bend, a massively parallel, high-level programming language. There were questions about its necessity compared to Triton or Mojo, with some noting Mojo's current limitations on GPUs and Triton's focus on ML [GitHub link](https://github.com/HigherOrderCO/Bend).

- **Concerns about CUDA's future**: A member expressed concerns about how new frameworks like Bend might affect traditional CUDA programming. Other members suggested that while new languages are exciting, they cater to different needs like CPU-GPU hybrid products.

- **Inference server resources exchange**: A discussion unfolded about building inference servers, with members recommending resources like Nvidia Triton, TorchServe, and various YouTube talks on ML model serving. Recommendations included [TorchServe tutorial](https://www.youtube.com/watch?v=XlO7iQMV3Ik&t=598s) and broader ML systems talks [YouTube link](https://www.youtube.com/watch?v=J36xHc05z-M).

- **Clarifying ML model serving complexities**: Members debated the differences between serving ML models versus standard web servers, noting that ML serving involves complex considerations like hardware requirements (e.g., GPUs), model versioning, and specific infrastructure like Kubernetes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.</li><li><a href="https://www.youtube.com/watch?v=J36xHc05z-M">Serving ML Models at a High Scale with Low Latency // Manoj  Agarwal // MLOps Meetup #48</a>: MLOps community meetup #48! Last Wednesday, on his birthday, we talked to Manoj  Agarwal, Software Architect at Salesforce.// Abstract:Serving machine learni...</li><li><a href="https://www.youtube.com/watch?v=Ynb6X0KZKxY">MLOps tools to scale your production machine learning || Alejandro Saucedo @ FOSDEM 2019</a>: As a machine learning project grows, so should its infrastructure. In short lightning talk, Alejandro covers some of the key trends in machine learning opera...</li><li><a href="https://www.youtube.com/watch?v=XlO7iQMV3Ik&t=598s">How to Serve PyTorch Models with TorchServe</a>: Hamid Shojanazeri is a Partner Engineer at PyTorch, here to demonstrate the basics of using TorchServe. As the preferred model serving solution for PyTorch, ...</li><li><a href="https://github.com/HigherOrderCO/Bend">GitHub - HigherOrderCO/Bend: A massively parallel, high-level programming language</a>: A massively parallel, high-level programming language - HigherOrderCO/Bend
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1241056711859769404)** (20 messages🔥): 

- **Performance discrepancy between tutorial setups**: A member noted a significant performance difference between Umer's [YouTube tutorial](https://www.youtube.com/watch?v=DdTsX6DQk24) and the official Triton tutorial. Despite using similar techniques, their implementation performed 2x worse compared to the tutorial.
- **MAX_FUSED_SIZE confusion in LayerNorm**: A user questioned why **MAX_FUSED_SIZE** is set to 65536 while **TRITON_MAX_TENSOR_NUMEL** is 1048576. They observed a speed degradation on an A100 when using a block size greater than 65536.
- **Reasons behind block size choices**: Horace explained that too large of a block size may cause kernel spilling due to excessive register requests. He confirmed that each block schedules to one SM and shares shared memory, similar to CUDA.
- **Thread operations on GPUs**: The conversation clarified that each Triton block maps to a GPU SM and each thread loads multiple elements. Horace mentioned this is desirable for utilizing vector instructions on a GPU.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/github.com/triton-lang/triton@ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/-/blob/python/triton/language/core.py?L19:27-19:34.">core.py - triton-lang/triton - Sourcegraph</a>: no description found</li><li><a href="https://discordapp.com/channels/1189498204333543425/1189607750876008468/1240593396389908510">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1241046490663751770)** (7 messages): 

- **Query on atomic add for complex numbers in CUDA**: A member asked about performing an *atomic add* on `cuda::complex`, querying if two distinct additions on the x and y components are necessary.
- **Limitation of 128-bit atomicCAS**: Another member noted that on architectures other than Hopper, one must settle for 64-bit operations because 128-bit `atomicCAS` isn't supported.
- **Shared Code Example**: To address the atomic add issue, a code snippet that uses a *complex* addition with `unsigned long long int` and `atomicCAS` was provided, explaining implementation on compatible architectures.
- **Simple Approach Suitability**: The original inquirer clarified that targeting Volta, Ampere, and Hopper architectures, they found two atomic adds on x and y components using either *cuComplex* or `cuda::std::complex` acceptable.


  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1241481119589994547)** (20 messages🔥): 

- **Torch Compile Usage Insights**: A member shared their experience using **torch.compile()** for inference use-cases, advising on code optimizations, such as avoiding Python loops and conditions, to ensure better performance. They mentioned that for static shapes, the tool generally works well out of the box.

- **Discussion on ONNX and TensorRT**: Another user raised a question on how **torch.compile()** compares to **ONNX** or **TensorRT** when using Triton for inference. The conversation suggests a curiosity about the relative performance and application scope of these tools.

- **NHWC Group Normalization Issues**: A member pointed out that group normalization in **ATen code** doesn't support NHWC format properly, leading to tensors being implicitly converted to NCHW. They shared a [GitHub pull request](https://github.com/pytorch/pytorch/pull/126635/files#r1605935532) aimed at addressing this issue but faced challenges with the `ApplyScaleBiasNHWCKernel` they wrote.

- **Torch Multiplication Memory Issue**: A question was raised about **torch's** native multiplication doubling memory usage even when performed in-place. Solutions and explanations included using `mul_()` to maintain flat memory consumption and handling memory allocation properly to address backprop concerns.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/pull/126635/files#r1605935532">Add NHWC support for group normalization  by ZelboK · Pull Request #126635 · pytorch/pytorch</a>: Fixes #111824 Currently it is the case that if the user specifies their group normalization to be of NHWC format, pytorch will default to NCHW tensors and convert. This  conversion is not immediate...

  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1241796374798798881)** (1 messages): 

- **Expert talk on building a GPU native query engine**: An announcement about a talk featuring a former maintainer of **cuDF** discussing the building process of a GPU native query engine at **Voltron**. The session promises to cover everything from authoring efficient kernels for data processing to creating a real production solution and is happening on [Zoom](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success).

**Link mentioned**: <a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1241952777253425244)** (2 messages): 

- **Stephen Jones Simplifies CUDA Programming**: A member shared a [YouTube video](https://www.youtube.com/watch?v=QQceTDjA4f4) titled "GTC 2022 - How CUDA Programming Works" by Stephen Jones, the CUDA Architect at NVIDIA. The video provides an introduction to programming the GPU and discusses the fundamentals of efficient memory use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=QQceTDjA4f4">GTC 2022 - How CUDA Programming Works - Stephen Jones, CUDA Architect, NVIDIA</a>: Come for an introduction to programming the GPU by the lead architect of CUDA. CUDA&#39;s unique in being a programming language designed and built hand-in-hand ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1241409136705404968)** (5 messages): 

- **Neural Magic hunts for CUDA/Triton engineers**: A committer to vLLM from **Neural Magic** announced job openings for CUDA/Triton engineers to contribute full-time to the project's open-source efforts. Interested individuals were asked to contact Robert Shaw via Discord or email.

- **Activation quantization emerges as top priority**: Responding to queries, it was mentioned that the primary focus is on **activation quantization** (fp8/int8) and related optimizations. The team aims to leverage features like 2:4 sparsity and fp6/fp4 on next-gen GPUs, and improve underoptimized aspects such as the MoE and sampling kernels.

- **LinkedIn expresses willingness to help**: A LinkedIn representative indicated potential interest from their team in supporting vLLM's needs. Further conversations on specific areas like graph-level optimization were initiated for potential collaboration.
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1242005768962838528)** (14 messages🔥): 

- **DLL Load Failure Troubleshooting**: A member encountered a "DLL load failed" error while importing a module in their CUDA implementation and sought help. Suggestions included checking the `build_directory` paths and ensuring `ninja` was installed, along with verifying the status of Visual Studio setup.
- **Need for Full Code and Stacktrace**: In response to the error, it was advised to share the full code and stacktrace for precise debugging, rather than assumptions. Testers asked for more context to offer a specific solution.
- **Ninja Installation and Environment Issues**: It was recommended to run `ninja -v` in the terminal to check its installation, especially considering the member was using a virtual environment on potentially dual-booted systems.
- **Windows Compatibility Worries**: There was a suggestion that dual-booting with Windows might complicate things, reflecting concerns over Visual Studio setup and general Windows compatibility issues with the build process.
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: Polish code breaker's https://www.flyingpenguin.com/?p=56989
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1241011309009703052)** (180 messages🔥🔥): 

<ul>
<li><strong>Grad Clipping Bug and Fixes Revolutionize Training:</strong> Discussions around gradient clipping reveal issues with incorrect comparisons as "grad_norm" was squared, needing correction to ensure more accurate and robust training. Additionally, the correct initialization of "grad_norm" has been emphasized to prevent unexpected behavior.</li>
<li><strong>Memory Optimizations come into Focus:</strong> Multiple users contribute to the discussion on optimizing CUDA kernel code, especially around memory allocation and usage with specific interest in templating block sizes for better compile-time constants. Attention was also given to the potential performance improvements from rewriting the Adam optimizer kernel considering new memory-bound constraints.</li>
<li><strong>Evaluating Hellaswag vs. MMLU Performance:</strong> Hellaswag evaluation for GPT-2 (124M at 29.55%) and GPT2-XL (48.93%) showed expected gradational improvement from model size. MMLU evaluations, however, were unexpectedly poor, indicating potential issues with the dataset or evaluation criteria.</li>
<li><strong>ZeRO-2 Implementation Discussions Get Technical:</strong> Members discussed implementing ZeRO-2, particularly focusing on memory layout reorganizations, communication call reductions, and the preservation of compatibility with checkpoint files. The conversation extended to efficient gradient computation and NCCL interleaving to enhance performance.</li>
<li><strong>Template and Constant Refactor for Optimization:</strong> A proposal to templatize block sizes in CUDA kernels to enable compile-time optimizations was discussed, alongside other codebase cleanup suggestions. An immediate result was a PR to standardize “warpSize” as a constant for better compile-time optimization, reflecting a common agreement on improved code efficiency.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://security.snyk.io/package/pip/llama-cpp-python">llama-cpp-python vulnerabilities | Snyk</a>: Learn more about known vulnerabilities in the llama-cpp-python package. Python bindings for the llama.cpp library</li><li><a href="https://github.com/karpathy/llm.c/issues/391)">Issues · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/k">k - Overview</a>: k has 88 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/439">Fix the unsupported block_size in matmul_backward_bias kernel 1 by lancerts · Pull Request #439 · karpathy/llm.c</a>: Due to the reduction in line https://github.com/karpathy/llm.c/blob/master/dev/cuda/matmul_backward_bias.cu#L67 The block size needs to be the power of 2 for the kernel 1. Otherwise the GPU result ...</li><li><a href="https://github.com/karpathy/llm.c/pull/435">Added warpsize as a constant for better compile time optimization and standardization by ChrisDryden · Pull Request #435 · karpathy/llm.c</a>: When investigating the properties of the WarpSize cuda constant, it is not available at compile time meaning that the compiler is unable to make compile time optimizations based on the value of the...</li><li><a href="https://github.com/karpathy/llm.c/pull/427">weight reordering: attempt 1 by ngc92 · Pull Request #427 · karpathy/llm.c</a>: Non-functional A first attempt how rearranging weights in a per-block layout could look like</li><li><a href="https://github.com/karpathy/llm.c/pull/429">improved numerical error checking by ngc92 · Pull Request #429 · karpathy/llm.c</a>: tighter tolarances relative tolerance based of bf16 epsilon less verbose output if all is OK  I&#39;ve checked these tolerances with both RTX 4060Ti and A4000 (which do give different errors, sometime...</li><li><a href="https://github.com/karpathy/llm.c/pull/361">Overlap gradient computation and NCCL AllReduce by PeterZhizhin · Pull Request #361 · karpathy/llm.c</a>: On my setup, I get the following: Before: step    2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step    3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1241798806647865464)** (33 messages🔥): 

- **Contributors inquire about libcudf's performance**: One contributor asked about benchmarks comparing **libcudf** to CPU parallelized operations, specifically mentioning the ongoing work on the ParPaRaw parser and CSV reader refactor. Another highlighted their interest in low-level code optimization like SASS.

- **Debate on Dask-cuDF and Theseus**: A user queried the differences and performance variations among **Dask-cuDF**, **cuDF**, and **Theseus**, expressing curiosity about their use cases and optimization levels. There was concern about the ongoing development of **Dask-cuDF**, with a [GitHub link](https://github.com/rapidsai/dask-cudf) indicating it had been archived.

- **RAPIDS Accelerator for Apache Spark introduced**: The discussion included an introduction to [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/), combining **RAPIDS cuDF library** and the **Spark distributed computing framework** to accelerate processing through GPUs. This tool aims to cater to the growing adoption of AI in analytics by offering a cost-efficient and speedy processing framework.

- **Thrust and CUB receive praise**: There was a rich discussion on the advantages of **Thrust** and **CUB**, with users appreciating their declarative programming flow that enhances code readability and optimization. The influence of CUB on the abstractions in CUTLASS was noted.

- **Optimization and bottlenecks discussed**: Insights were shared on the diminishing need for assembly-level optimization due to current bottlenecks shifting to IO and networking. The focus has now moved toward understanding how **libcudf** is utilized on large datasets, emphasizing the importance of networking orchestrations like **NCCL**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nvidia.github.io/spark-rapids/">Home</a>: This site serves as a collection of documentation about the RAPIDS accelerator for Apache Spark</li><li><a href="https://github.com/rapidsai/dask-cudf">GitHub - rapidsai/dask-cudf: [ARCHIVED] Dask support for distributed GDF object --&gt; Moved to cudf</a>: [ARCHIVED] Dask support for distributed GDF object --&gt; Moved to cudf - rapidsai/dask-cudf
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1241300874488188972)** (2 messages): 

- **Zoom link for extended discussion**: Due to the **45-minute limitation** on activity time, members were directed to join an extended discussion on Zoom: [Zoom Meeting](https://us06web.zoom.us/j/86116925784?pwd=XGcom9z5cGUijqjua9gKKa3AwOA4KO.1).
- **Barrier synchronization analogy**: A member shared an insightful analogy, comparing **barrier synchronization** to a school bus waiting for all kids to return from a museum visit, saying it "can't move till all are accounted for." This helped clarify the concept for others.

**Link mentioned**: <a href="https://us06web.zoom.us/j/86116925784?pwd=XGcom9z5cGUijqjua9gKKa3AwOA4KO.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


**CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1241234168739004566)** (31 messages🔥): 

- **Sanity checking uint2 implementation with updates**: A member shared implementation details and requested a *sanity check* for converting and packing uint2 data types in PyTorch. Another member suggested avoiding `torch.stack` for better performance with `torch.compile`, leading to an updated implementation using `torch.empty`.

- **Meeting planning for bitnet group**: Discussions were held about organizing regular meetings for the bitnet group and reviewing relevant documents and repositories. The meeting planner and resources were shared, with a *tentative meetup* scheduled for tomorrow.

- **Issues with uint4 dtype in Torch**: Members discussed the necessity of packing uint4 data types into uint8 for memory efficiency due to the lack of native int4 operations on Nvidia GPUs. It was clarified that without packing, memory consumption would double.

- **Unpacking uint8 to trinary values**: Code examples were discussed and refined for unpacking uint8 data to trinary values and handling signed/unsigned bitshifts. A potential workaround for quantization by shifting distributions was also considered.

- **Collaborative efforts on project management**: Members acknowledged the challenges with project management while ensuring that all necessary references and best practices for custom CUDA extensions and dtype creation were shared and followed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: This is the mergaeble version of #130 - some updates I have to make   Add a skip test unless pytorch 2.4+ is used and Add a skip test if cuda is not available  Add ninja to dev dependencies  Locall...</li><li><a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?pli=1#heading=h.ptttacy8y1u9">The C++ Custom Operators Manual</a>: no description found</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/dtypes/uint4.py">ao/torchao/dtypes/uint4.py at main · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://github.com/TimDettmers/bitsandbytes/commit/67475257a96b792f9b66e71892dab90f7a60ed87">Added documentation for NF4; failing 8-bit matmul; fixed absmax bug. … · TimDettmers/bitsandbytes@6747525</a>: …#529 #543</li><li><a href="https://github.com/pytorch/ao/pull/248">Improve primitives for FP6 quant by gau-nernst · Pull Request #248 · pytorch/ao</a>: Address #208 TODO:   FP32/FP16/BF16 -&gt; FP6 (CPU + CUDA) (with correct rounding)  FP6 -&gt; FP32/FP16/BF16 (CPU + CUDA)  Add tests  Fix exception in OpenMP  Figure a way for checking in CUDA kernel?...</li><li><a href="https://github.com/pytorch/ao/pull/68"> 1.58 bit by msaroufim · Pull Request #68 · pytorch/ao</a>: Fixes #67
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1240948219861794869)** (219 messages🔥🔥): 

- **CC Datasets Contain Significant Spam**: Members discussed ongoing issues with spam in CC datasets, noting a high presence of auto-generated and duplicate content across languages. Asada.shinon mentioned that Chinese datasets contain the most spam, with dedicated filters to address this, and shared an [article from Technology Review](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/) about GPT-4o's Chinese token issues.
  
- **OpenELM offers Transparency and Efficiency**: Smerkyg inquired about LLM models with frequent checkpoints, leading to a discussion about OpenELM, a new LLM emphasizing reproducibility and efficiency, achieving a 2.36% improvement in accuracy compared to OLMo. [OpenELM Research](https://machinelearning.apple.com/research/openelm) was suggested as a resource.

- **Memory and Efficiency in LoRA**: Premiumonion asked about the FLOPs and memory efficiency in LoRA, concluding that LoRA primarily saves memory over full fine-tuning. Skyward2989 confirmed that memory is often the bottleneck in AI model training.

- **Canadian and UK AI Safety Institutes Collaboration**: Hyperion.ai shared that the UK and Canada have announced collaboration on AI safety, involving professional exchanges and secondments to bolster research, detailed in a [government publication](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership).

- **Interest in Time Series Modeling**: Tiley and Hawk1399 discussed methods for modeling continuous multivariate time series with autoregression. Tiley expressed concerns about errors in autoregressive inference, with suggestions like examining MOMENT from [arXiv](https://arxiv.org/abs/2402.03885) and considering methods that account for the scope of nonlinear dynamics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/">GPT-4o’s Chinese token-training data is polluted by spam and porn websites</a>: The problem, which is likely due to inadequate data cleaning, could lead to hallucinations, poor performance, and misuse.</li><li><a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>: This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach...</li><li><a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>: Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands impo...</li><li><a href="https://arxiv.org/abs/2402.03885">MOMENT: A Family of Open Time-series Foundation Models</a>: We introduce MOMENT, a family of open-source foundation models for general-purpose time series analysis. Pre-training large models on time series data is challenging due to (1) the absence of a large ...</li><li><a href="https://machinelearning.apple.com/research/openelm">OpenELM: An Efficient Language Model Family with Open Training and Inference Framework</a>: The reproducibility and transparency of large language models are crucial for advancing open research, ensuring the trustworthiness of…</li><li><a href="https://github.com/nshepperd/flash_attn_jax">GitHub - nshepperd/flash_attn_jax: JAX bindings for Flash Attention v2</a>: JAX bindings for Flash Attention v2. Contribute to nshepperd/flash_attn_jax development by creating an account on GitHub.</li><li><a href="https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py">jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax</a>: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax</li><li><a href="https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership">UK-Canada science of AI safety partnership</a>: no description found</li><li><a href="https://zenn.dev/hellorusk/articles/27684d0ed96c4c">【風吹けば名無し】GPT-4o が獲得した日本語の語彙を調べる</a>: no description found</li><li><a href="https://www.aisi.gov.uk/">The AI Safety Institute (AISI)</a>: The AI Safety Institute is a directorate of the Department of Science, Innovation, and Technology that facilitates rigorous research to enable advanced AI governance.</li><li><a href="https://blog.allenai.org/olmo-open-language-model-87ccfc95f580">OLMo: Open Language Model</a>: A State-Of-The-Art, Truly Open LLM and Framework
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1240937732151316480)** (93 messages🔥🔥): 

- **Discussing Non-discrete Embedding Spaces with CLIP Guidance**: Members discussed potential issues with **CLIP guidance**, noting that the embedding space might not capture desired attributes due to non-discrete nature and model training biases. One participant suggested an alternative approach similar to *LPIPS for text*.
- **Twitter Paper Resurfaces, Stirs Discussion**: A shared [Twitter link](https://twitter.com/arankomatsuzaki/status/1791289342121455993) prompted debate on an apparently impactful paper. Members discussed its relevance and implications on model training techniques.
- **Innovations in Hierarchical Memory Transformers**: A [paper on Hierarchical Memory Transformers](https://arxiv.org/abs/2405.06067) sparked interest, proposing a novel framework mimicking human memory for enhanced long-context processing. This discussion delved into recurrent models and memory architectures.
- **Analyzing LLM Co-occurrence Issues**: Members explored the challenges of evaluating co-occurrence in language model outputs, particularly when models follow their prior outputs over prompts. Suggestions included measuring cross-attention contributions and perplexity metrics.
- **Investigating Positive Transfer Across Modalities**: The conversation around **ImageBind** and related papers ([ImageBind](https://arxiv.org/abs/2305.05665), [PaLM-E](https://arxiv.org/abs/2303.03378)) examined whether training models across multiple modalities can enhance performance in unimodal tasks. This included discussions on **zero-shot recognition** and combining modality embeddings to improve retrieval performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>: Transformer-based large language models (LLM) have been widely used in language processing applications. However, most of them restrict the context window that permits the model to attend to every tok...</li><li><a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>: The disconnect between tokenizer creation and model training in language models has been known to allow for certain inputs, such as the infamous SolidGoldMagikarp token, to induce unwanted behaviour. ...</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>: We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not n...</li><li><a href="https://arxiv.org/abs/2205.06175">A Generalist Agent</a>: Inspired by progress in large-scale language modeling, we apply a similar approach towards building a single generalist agent beyond the realm of text outputs. The agent, which we refer to as Gato, wo...</li><li><a href="https://arxiv.org/abs/2303.03378">PaLM-E: An Embodied Multimodal Language Model</a>: Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g., for robotics problems, raises the challenge of grounding. We propose embodied...</li><li><a href="https://arxiv.org/abs/2310.02557">Generalization in diffusion models arises from geometry-adaptive harmonic representations</a>: Deep neural networks (DNNs) trained for image denoising are able to generate high-quality samples with score-based reverse diffusion algorithms. These impressive capabilities seem to imply an escape f...</li><li><a href="https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/">ICLR 2024 Outstanding Paper Awards &#8211; ICLR Blog</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1240971637466337331)** (14 messages🔥): 

- **Scaling bare bones paper faces criticism**: One member commented on the sparse nature of a recently discussed research paper, noting its lack of hyperparameter tuning and expressing curiosity about its scalability at higher levels.
- **Challenges in estimating FLOP calculations**: A detailed discussion emerged about the correct calculation of FLOPs for forward and backward passes in models. Members provided insights and referenced specific resources like [EleutherAI's cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) to clear up confusion, with notes that some calculations might exclude projection computations leading to discrepancies.
- **Query on sample efficiency metrics**: A member posed questions about defining and measuring **sample efficiency** in various domains, suggesting the concept's importance in relation to scaling laws and efficient resource management.
- **Theoretical question on Bitnet's compute efficiency**: There was an intriguing theoretical discussion about whether a more compute-efficient version of a model, using the same parameter count but significantly less compute power, would alter the optimal parameter to token ratio as defined by Chinchilla scaling laws. The consensus leaned towards no change, assuming increased compute capabilities would simply extend compute budgets for such models.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>: Understanding how language model performance varies with scale is critical to benchmark and algorithm development. Scaling laws are one approach to building this understanding, but the requirement of ...

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1240969521699880960)** (13 messages🔥): 

- **HF model automatically uses default prompt**: *The model is automatically prompted with a default prompt based on current common practices.* One user shared their experience of fine-tuning models with different methods and noted a variation in performance.
- **Seeks finance and crypto-related AI tasks in English**: A member inquired about good tasks for finance, trading, investing, and crypto-related topics, specifying a preference for tasks in English.
- **NeurIPS benchmark article pseudo-review request**: A member asked if anyone was interested in reviewing their benchmark article for NeurIPS. Another member responded positively, agreeing to the request.
- **Improving evaluation speed on large models**: A user shared difficulties in running evaluations on large models, noticing long durations for tasks like MMLU. Another user suggested optimizing batch size settings to speed up the evaluation process.
- **No dedicated channel for AI Safety/benchmark events**: A member asked about promoting AI Safety or benchmarks-relevant events in a dedicated channel. The response indicated that there is currently no such channel available in EleutherAI Discord.

**Link mentioned**: <a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro">TIGER-Lab/MMLU-Pro · Datasets at Hugging Face</a>: no description found

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1241456488741077002)** (1 messages): 

- **Soft prompt tuning setup issues**: A member inquired about recent experiences with the soft prompt tuning setup in non-pipeline cases. They mentioned a specific issue where it seems *"param.requires_grad gets reset after model.to_sequential() is called."*
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1241007050121543720)** (2 messages): 

- **Request for Monthly Round Up**: A member suggested that a monthly round-up would be *"very helpful"*. The proposal indicates a desire for regular summaries or updates to stay informed. 

- **Expression of Uncertainty**: Nathan Lambert responded with, *"lol I don’t know Man"* indicating uncertainty or ambiguity about the previous suggestion or a related discussion. This shows a casual tone in the conversation.
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1241042777727570022)** (29 messages🔥): 

<html>
  <body>
    <ul>
      <li><strong>Meta introduces Chameleon</strong>: Meta's new model, Chameleon, is a 34B parameter multimodal foundation model outperforming models like Flamingo and IDEFICS in both text and image tasks. It’s trained on ~10T tokens and claims superiority over GPT-4V in human evaluations. [Source](https://arxiv.org/abs/2405.09818)</li>
      <li><strong>DeepMind reveals Flash-8B</strong>: The updated Gemini 1.5 paper introduces Flash-8B, a new model distinct from Gemini 1.5 Flash. Flash-8B boasts a multimodal and extensive context window while being highly efficient. [Source](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=45)</li>
      <li><strong>Gemini 1.5 Model Family expands</strong>: The Gemini 1.5 Pro and Flash models show significant improvements over previous versions, excelling in both text and vision benchmarks. Their performance in the MMLU task demonstrates the highest capabilities among their lineup. [Source](https://goo.gle/GeminiV1-5)</li>
      <li><strong>Anthropic scales up</strong>: Anthropic reported utilizing four times more compute than their previous model, Opus, aiming to develop even larger and more capable models. [Source](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)</li>
      <li><strong>LMsys announces "Hard Prompts" category</strong>: LMsys introduces a "Hard Prompts" category in Arena to evaluate models on more challenging tasks with a significant ranking shift observed. Llama-3-70B-Instruct is used as a judge model, but its reliability is questioned. [Source](https://fxtwitter.com/lmsysorg/status/1792625968865026427)</li>
    </ul>
  </body>
</html>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/lmsysorg/status/1792625977207468315">Tweet from lmsys.org (@lmsysorg)</a>: How did we classify these criteria? We adopt Llama-3-70B-Instruct as the judge model to help us label over 1 million Arena battles.  Overall our analysis reveals that the quality of Arena user prompts...</li><li><a href="https://x.com/swishfever/status/1791551855954370985?s=46">Tweet from fishy business (@swishfever)</a>: commented line in chameleon paper:  %  \item We open-source variants of \model{} that allow text and image inputs but only text outputs across all model sizes.  Quoting Tanishq Mathew Abraham, Ph.D. (...</li><li><a href="https://x.com/dalucasgonzalez/status/1791525232622342492?s=46">Tweet from lucas g (@DaLucasGonzalez)</a>: Our updated Gemini 1.5 tech report is out!  Excited to share a sneak peak of a new model we are working on: Flash-8B  https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=4...</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625968865026427">Tweet from lmsys.org (@lmsysorg)</a>: Introducing &#34;Hard Prompts&#34; Category in Arena!  In response to the community&#39;s growing interest in evaluating models on more challenging tasks, we are excited to launch the new &#34;Hard Pr...</li><li><a href="https://x.com/dalucasgonzalez/status/1791526024444006489?s=46">Tweet from lucas g (@DaLucasGonzalez)</a>: Flash-8B has the same multimodal and million context window as our other 1.5 models, but in an extremely efficient footprint. There&#39;s no other model like this in the world.  It shows incredible ca...</li><li><a href="https://x.com/dalucasgonzalez/status/1791526696312803727?s=46">Tweet from lucas g (@DaLucasGonzalez)</a>: Our initial benchmarks are very promising, and this only an early look, as we are still actively developing the model to maximize performance at this size.</li><li><a href="https://x.com/suchenzang/status/1791533241494835376?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Susan Zhang (@suchenzang)</a>: Updated tech report with lots of goodies!   Now for a thread mostly focusing on ⚡️ Gemini 1.5 Flash ⚡️...  🧵  Quoting Jeff Dean (@🏡) (@JeffDean)   Gemini 1.5 Model Family: Technical Report updates n...</li><li><a href="https://x.com/aidan_mclau/status/1792610354255769919">Tweet from Aidan McLau (@aidan_mclau)</a>: yo what is anthropic cookin  4× more compute than opus damm
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1241059896409981040)** (145 messages🔥🔥): 

- **OpenAI's superalignment team disbanded**: The formation of OpenAI's "superalignment team" was announced last year to prepare for potential supersmart AI. This team is now disbanded following departures of key researchers including Ilya Sutskever, as covered [here](https://archive.is/gEjjA).

- **Jan Leike's departure from OpenAI**: Jan Leike, former co-lead of the superalignment team, expressed disagreements with OpenAI's core priorities on [Twitter](https://x.com/janleike/status/1791498178346549382). 

- **Goodhart's law and AI deception**: Users debated the implications of Goodhart's law on large language models, with concerns that merely increasing model size can lead to models goodharting better, thus becoming more deceptive.

- **OpenAI's controversial employment practices**: OpenAI faced criticism for requiring departing employees to sign lifelong nondisparagement agreements in order to retain vested equity, despite leadership later clarifying on [Twitter](https://x.com/soumithchintala/status/1791612240371580999?s=46) that they never enforced such clauses.

- **OpenAI addresses AI voices controversy**: OpenAI paused the use of its AI voice "Sky" following questions about its selection process. They clarified the voice is not mimicking a celebrity but belongs to a professional actress. Read more [here](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/kelseytuoc/status/1791584361718100361?s=46">Tweet from Kelsey Piper (@KelseyTuoc)</a>: And if you don&#39;t sign within sixty days your units are gone. And it gets worse - because OpenAI can also deny you access to the annual events that are the only way to sell your vested PPUs at thei...</li><li><a href="https://x.com/kelseytuoc/status/1791584322698559780?s=46">Tweet from Kelsey Piper (@KelseyTuoc)</a>: I&#39;m getting two reactions to my piece about OpenAI&#39;s departure agreements: &#34;that&#39;s normal!&#34; (it is not; the other leading AI labs do not have similar policies) and &#34;how is that...</li><li><a href="https://x.com/sama/status/1791936857594581428?s=46">Tweet from Sam Altman (@sama)</a>: in regards to recent stuff about how openai handles equity:  we have never clawed back anyone&#39;s vested equity, nor will we do that if people do not sign a separation agreement (or don&#39;t agree ...</li><li><a href="https://fxtwitter.com/OpenAI/status/1792443575839678909">Tweet from OpenAI (@OpenAI)</a>: We’ve heard questions about how we chose the voices in ChatGPT, especially Sky. We are working to pause the use of Sky while we address them.  Read more about how we chose these voices: https://openai...</li><li><a href="https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Kelsey Piper (@KelseyTuoc)</a>: When you leave OpenAI, you get an unpleasant surprise: a departure deal where if you don&#39;t sign a lifelong nondisparagement commitment, you lose all of your vested equity: https://www.vox.com/futu...</li><li><a href="https://x.com/soumithchintala/status/1791612240371580999?s=46">Tweet from Soumith Chintala (@soumithchintala)</a>: i got confirmation from multiple ex-OpenAI folks that this is true, and that&#39;s why they don&#39;t say anything negative about their experience.</li><li><a href="https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/">Introducing the Frontier Safety Framework</a>: Our approach to analyzing and mitigating future risks posed by advanced AI models</li><li><a href="https://x.com/janleike/status/1791498178346549382">Tweet from Jan Leike (@janleike)</a>: I joined because I thought OpenAI would be the best place in the world to do this research.  However, I have been disagreeing with OpenAI leadership about the company&#39;s core priorities for quite s...</li><li><a href="https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDC">Tweet from Kelsey Piper (@KelseyTuoc)</a>: When you leave OpenAI, you get an unpleasant surprise: a departure deal where if you don&#39;t sign a lifelong nondisparagement commitment, you lose all of your vested equity: https://www.vox.com/futu...</li><li><a href="https://youtu.be/ZP_N4q5U3eE?si=hFlutzYz2Jd9E_rH&t=211">OpenAI’s huge push to make superintelligence safe | Jan Leike</a>: In July 2023, OpenAI announced that it would be putting a massive 20% of its computational resources behind a new team and project, Superalignment, with the ...</li><li><a href="https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence">Tweet from &quot;I lost trust&quot;: Why the OpenAI team in charge of safeguarding humanity imploded</a>: Company insiders explain why safety-conscious employees are leaving.</li><li><a href="https://archive.is/gEjjA">OpenAI&#x2019;s Long-Term AI Risk Team Has Disbanded | WIRED</a>: no description found
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1241211192937283686)** (24 messages🔥): 

- **Chinatalk episode gets thumbs up**: A member praised the Chinatalk episode with a thumbs up emoji, indicating it was very good.
- **Llama3-from-scratch project is a great learning tool**: [Llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) was highlighted as an excellent resource for learning, suggesting that creators of such tools are hireable. *"These things are the best learning tools"*, exclaimed a member.
- **Latent Consistency Models explained for beginners**: A blog explaining Latent Consistency Models (LCMs) for beginners was recommended, especially praised for its readability. The blog can be found [here](https://naklecha.notion.site/explained-latent-consistency-models-13a9290c0fd3427d8d1a1e0bed97bde2).
- **New domain name purchased**: A discussion about buying and squatting domain names led to a member buying the domain **rlhfbook.com**. The price was notably low, only $7/year via Porkbun.
- **Caution over Books4 dataset**: The Books4 dataset was humorously referred to as a legal minefield, likened to Monopoly's "Straight to Jail" card. It was mentioned that previous legal actions have primarily targeted dataset curators.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://naklecha.notion.site/explained-latent-consistency-models-13a9290c0fd3427d8d1a1e0bed97bde2">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://github.com/naklecha/llama3-from-scratch">GitHub - naklecha/llama3-from-scratch: llama3 implementation one matrix multiplication at a time</a>: llama3 implementation one matrix multiplication at a time - naklecha/llama3-from-scratch</li><li><a href="https://web.archive.org/web/20240519104217/https://www.reddit.com/r/datasets/comments/1cvi151/ai_books4_dataset_for_training_llms_further/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1241229144436768819)** (41 messages🔥): 

- **Yudkowsky's Doomsday Insight Gains Traction**: A user shared [a post](https://x.com/liron/status/1791592296053686341?s=46) by **Liron Shapira**, highlighting **Eliezer Yudkowsky's** broad yet incomplete influence on AI risk awareness. The user emphasized that other experts are still on their journey toward "full awareness" of the problem.

- **Hilarious AI Growth Hack**: Users discussed [a meme](https://x.com/HamelHusain/status/1791707778245185613) from **Hamel Husain**, with one suggesting using the concept as a marketing stunt. The idea revolved around offering "1 year paid," even though it provides minimal value.

- **Selling Couch for AI Credits**: A user humorously admitted selling their couch to purchase an AI course, proclaiming wealth in "credits". Amid discussions, **Natolambert** acknowledged the fun in using credits and experimenting with various APIs.

- **Debate Over Paid Content Lectures**: **Natolambert** expressed discomfort in delivering lectures for paid content and noted how **Maven** had tried to onboard him for a course. Concerns were raised about helping others profit from his branding via YouTube collaborations.

- **Gaming Roots and YouTube Ventures**: Conversations entertained **Call of Duty** experiences, with **Natolambert** sharing [his YouTube channel](https://www.youtube.com/@natolambert). There was a nostalgic recall of earning respect through gaming skills and the shared excitement for content creation around academic papers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her">OpenAI pulls its Scarlett Johansson-like voice for ChatGPT</a>: Maybe Her (2014) shouldn’t be a blueprint for AI voice features.</li><li><a href="https://x.com/liron/status/1791592296053686341?s=46">Tweet from Liron Shapira (@liron)</a>: You gotta realize that @ESYudkowsky&#39;s caliber of insight about how doomed we are is still far ahead of the pack.  Just because other experts are finally heading his way, doesn&#39;t mean they&#39;...</li><li><a href="https://x.com/HamelHusain/status/1791707778245185613">Tweet from Hamel Husain (@HamelHusain)</a>: 🤣
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1241189481751511134)** (6 messages): 

- **Concerns about ORPO paper in RLHF**: A user inquired if anyone had looked at the **ORPO paper** and noted that it had been added to **Hugging Face's library**. Another member shared their suspicion about ORPO's scalability, stating, "It sounds nice but I don’t know if it’ll scale well," indicating skepticism while reminding themselves to give it more credit.
- **Practical testing reveals ORPO limitations**: One member shared [results from tests](https://x.com/ethayarajh/status/1783270535369195905) on ORPO, finding it "seemed okay but not great". They argued that combining SFT with margin-based loss usually doesn't work well, suggesting ORPO's method of replacing the reference model with 1-policy might result in over-regularization.

**Link mentioned**: <a href="https://x.com/ethayarajh/status/1783270535369195905">Tweet from Kawin Ethayarajh (@ethayarajh)</a>: @maximelabonne @winniethexu aligned zephyr-sft-beta on ultrafeedback and it looks like kto/dpo are a bit better?   note that zephyr-sft-beta was sft&#39;ed on ultrachat (not ultrafeedback) so all the ...

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1241928485950193764)** (2 messages): 

- **Chamath Palihapitiya Faces Criticism**: A Substack post criticizes **Chamath Palihapitiya** for his role in promoting special purpose acquisition companies (SPACs), which led to financial losses for retail investors. The author argues that Palihapitiya dismisses the losses suffered by others while continuing to deny any wrongdoing ([The Scam in the Arena](https://open.substack.com/pub/newcomer/p/the-scam-in-the-arena?r=68gy5&utm_medium=ios)).
- **Schadenfreude Over the All-In Pod Hosts**: A member expressed enjoyment in reading about the failures of the All In Podcast hosts, noting that they seem insincere. *"I really enjoy reading about the all in pod hosts failures. They feel so fake"*.



**Link mentioned**: <a href="https://open.substack.com/pub/newcomer/p/the-scam-in-the-arena?r=68gy5&utm_medium=ios">The Scam in the Arena</a>: Chamath Palihapitiya took retail investors for a ride, got away with it, and just can&#x27;t let himself take the win.

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


**Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1241098150089986078)** (21 messages🔥): 

- **Cynical about automating OnlyFans DMs**: A member mentioned listening to the recent Latent Space podcast where they interviewed someone automating OnlyFans DMs, which made them "pretty cynical". This was considered relevant to the current episode discussion.
- **Interesting episode on OpenAI happenings**: The new [Retort AI episode](https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal) discusses two major OpenAI developments, including their new chat assistant and the release of their Model Spec for RLHF goals. A highlighted segment mentions the blurring boundaries of intimacy and technology.
- **Scaling laws of vocab size**: A member raised the question about the scaling laws of vocab size in relation to model size, pondering potential trade-offs between inference speed and complexity. Another member responded, indicating models are harder to train stably with weirder tokenizers.
- **Hysteresis and Control Theory**: Members discussed the term "hysteresis" and its relevance in control theory, with a nod to Steven Strogatz's work referenced through an [Amazon book link](https://www.amazon.com/Nonlinear-Dynamics-Chaos-Applications-Nonlinearity/dp/0738204536). They humorously pondered needing to know more control theory for GRE words.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/Nonlinear-Dynamics-Chaos-Applications-Nonlinearity/dp/0738204536">no title found</a>: no description found</li><li><a href="https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal">The Retort AI Podcast | ChatGPT talks: diamond of the season or quite the scandal?</a>: Tom and Nate discuss two major OpenAI happenings in the last week. The popular one, the chat assistant, and what it reveals about OpenAI's worldview. We pair this with discussion of OpenAI's new Mo...
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1241022818183942206)** (126 messages🔥🔥): 

- **Hinton and Ilya's Debate on Scaling Laws**: @joelhellermark shared Hinton's quote on Ilya's intuition for scaling laws, stating, *"Ilya was always preaching that you just make it bigger and it’ll work better. And I always thought that was a bit of a cop-out, that you're going to have to have new ideas too. Turns out Ilya was basically right."* [Full interview link here](https://x.com/joelhellermark/status/1791398092400390195).

- **Jan Leike's Departure from OpenAI**: Several contributors highlighted Jan Leike's resignation as head of alignment at OpenAI, sharing multiple sources and speculations on the implications of his departure. Sam Altman and Greg Brockman posted their appreciation and future safety plans [here](https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Vertical-Axis Wind Turbines Using Machine Learning**: A link to an article was shared about EPFL researchers using a genetic learning algorithm to optimize blade profiles for vertical-axis wind turbines, which are less noisy and more wildlife-friendly compared to horizontal-axis wind turbines. [Full story here](https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi).

- **Obsidian and AI for Journaling**: Users discussed integrating AI with note-taking systems like Obsidian to create more efficient journaling/diary workflows. @neuralution mentioned a project using voice conversations via a custom Telegram bot to summarize journal entries into Obsidian.

- **Comparing AI Languages: Rust vs Go**: A member asked whether Rust or Go is better suited for AI development. Contributors noted that Rust is gaining traction, especially with Hugging Face projects like Candle and tokenizers, while Go is more suited for applications making HTTP calls to LLM APIs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence">Tweet from &quot;I lost trust&quot;: Why the OpenAI team in charge of safeguarding humanity imploded</a>: Company insiders explain why safety-conscious employees are leaving.</li><li><a href="https://js.langchain.com/v0.1/docs/additional_resources/tutorials/">Tutorials | 🦜️🔗 Langchain</a>: Below are links to tutorials and courses on LangChain.js. For written guides on common use cases for LangChain.js, check out the use cases and guides sections.</li><li><a href="https://x.com/realsharonzhou/status/1792576516444065967?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Sharon Zhou (@realSharonZhou)</a>: Hallucinations are one of the biggest blockers to production LLMs & agents.  No hallucinations (&lt;5%) have been achieved internally — and for customers.   We’ve been able to tune LLMs to recall spec...</li><li><a href="https://hamel.dev/blog/posts/fine_tuning_valuable.html">Hamel’s Blog - Is Fine-Tuning Still Valuable?</a>: A reaction to a recent trend of disillusionment with fine-tuning.</li><li><a href="https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Greg Brockman (@gdb)</a>: We’re really grateful to Jan for everything he&#39;s done for OpenAI, and we know he&#39;ll continue to contribute to the mission from outside. In light of the questions his departure has raised, we w...</li><li><a href="https://x.com/joelhellermark/status/1791398092400390195?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Joel Hellermark (@joelhellermark)</a>: Spoke to @geoffreyhinton about OpenAI co-founder @ilyasut&#39;s intuition for scaling laws👇.  &#34;Ilya was always preaching that you just make it bigger and it&#39;ll work better.  And I always thou...</li><li><a href="https://x.com/soniajoseph_/status/1791604177581310234?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sonia Joseph (@soniajoseph_)</a>: To the journalists contacting me about the AGI consensual non-consensual (cnc) sex parties—  During my twenties in Silicon Valley, I ran among elite tech/AI circles through the community house scene. ...</li><li><a href="https://x.com/dan_biderman/status/1791506475010977875">Tweet from Dan Biderman (@dan_biderman)</a>: People think LoRA is a magic bullet for LLMs. Is it? Does it deliver the same quality as full finetuning but on consumer GPUs?  Though LoRA has the advantage of a lower memory footprint, we find that ...</li><li><a href="https://x.com/ns123abc/status/1791548950719103319">Tweet from NIK (@ns123abc)</a>: holy fuck it&#39;s absolutely fucking over</li><li><a href="https://x.com/sama/status/1791543264090472660?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sam Altman (@sama)</a>: i&#39;m super appreciative of @janleike&#39;s contributions to openai&#39;s alignment research and safety culture, and very sad to see him leave. he&#39;s right we have a lot more to do; we are commit...</li><li><a href="https://x.com/natfriedman/status/1791462511889559615?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Nat Friedman (@natfriedman)</a>: My loosely held conclusion from reading through the recommended papers and re-running some of the evals is that there are some weak signs of transfer and generalization from code to other reasoning pr...</li><li><a href="https://x.com/janleike/status/1791498174659715494">Tweet from Jan Leike (@janleike)</a>: Yesterday was my last day as head of alignment, superalignment lead, and executive @OpenAI.</li><li><a href="https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi/">Machine learning enables viability of vertical-axis wind turbines</a>: EPFL researchers have used a genetic learning algorithm to identify optimal pitch profiles for the blades of vertical-axis wind turbines, which despite their high energy potential, have until now been...</li><li><a href="https://threadreaderapp.com/thread/1791498174659715494.html">Thread by @janleike on Thread Reader App</a>: @janleike: Yesterday was my last day as head of alignment, superalignment lead, and executive @OpenAI. It&#39;s been such a wild journey over the past ~3 years. My team launched the first ever RLHF LL...</li><li><a href="https://github.com/sublayerapp/sublayer">GitHub - sublayerapp/sublayer: A model-agnostic Ruby Generative AI DSL and framework. Provides base classes for building Generators, Actions, Tasks, and Agents that can be used to build AI powered applications in Ruby.</a>: A model-agnostic Ruby Generative AI DSL and framework. Provides base classes for building Generators, Actions, Tasks, and Agents that can be used to build AI powered applications in Ruby. - sublaye...</li><li><a href="https://github.com/go-go-golems/geppetto">GitHub - go-go-golems/geppetto: golang GPT3 tooling</a>: golang GPT3 tooling. Contribute to go-go-golems/geppetto development by creating an account on GitHub.</li><li><a href="https://news.ycombinator.com/item?id=40400224#40403951">Sam and Greg&#x27;s response to OpenAI Safety researcher claims | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1241116585867874455)** (127 messages🔥🔥): 

- **"Feedback Is All You Need" Discussed**: A [YouTube video](https://www.youtube.com/watch?v=BNFRGfWQo6M) titled "Feedback Is All You Need - Gordon Brander" was shared, sparking discussion on whether current AI agents can learn, adapt, and make autonomous decisions.
- **Andrew Ng on AI Agents**: A link to an [Andrew Ng's tweet](https://x.com/AndrewYNg/status/1770897666702233815) was shared, emphasizing the potential of AI agentic workflows to drive significant AI progress. He elaborated on the benefits of iterative workflows and various design patterns for building agents like reflection, tool use, planning, and multi-agent collaboration.
- **Debate on Definition of AI Agents**: Members debated the definition and attributes of AI agents, comparing them to traditional software agents and considering autonomy, social ability, reactivity, and persistence as critical factors.
- **Reinforcement Learning and Historical Context**: The historical context of agents in AI was discussed, with references to seminal works like Samuel's checkers-playing program from 1959, highlighting the lineage and evolution of agent-based decision-making systems.
- **Interest in AI Music Generation**: Members expressed excitement and interest in AI-generated music and related projects, with personal anecdotes and future collaboration plans shared. One member mentioned working on MusicGen finetunes and promised to share related links.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Software_agent">Software agent - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/the-simpsons-mr-burns-muahahaha-evil-laugh-gif-4482837">Muahaha GIF - The Simpsons Mr Burns Muahahaha - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=BNFRGfWQo6M">Feedback Is All You Need - Gordon Brander</a>: How far are we from realizing AI agents that can learn, adapt, and make decisions on their own? Are we already there? And what is an agent, anyway? Answers f...</li><li><a href="https://x.com/AndrewYNg/status/1770897666702233815">Tweet from Andrew Ng (@AndrewYNg)</a>: I think AI agentic workflows will drive massive AI progress this year — perhaps even more than the next generation of foundation models. This is an important trend, and I urge everyone who works in AI...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1242138352224964679)** (1 messages): 

- **New Webinar on Memary project**: This Thursday at 9am PT, we will be hosting the authors of **memary**, an open-source reference for long-term memory in autonomous agents. The webinar will feature a deep dive into the project and a Q&A session discussing memory challenges and future directions—[sign up here](https://lu.ma/nzh3o83f).

**Link mentioned**: <a href="https://lu.ma/nzh3o83f">LlamaIndex Webinar: Open-Source Longterm Memory for Autonomous Agents · Zoom · Luma</a>: In this webinar we&#x27;re excited to host the authors of memary - a fully open-source reference implementation for long-term memory in autonomous agents 🧠🕸️ In…

  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1241065559664099370)** (10 messages🔥): 

```html
- **QA struggles with large tables**: Even the latest LLMs still hallucinate over complex tables like the Caltrain schedule due to poor parsing. More details can be found [here](https://t.co/Scvp7LH2pL).
- **Boost vector search speed by 32x**: Using 32-bit vectors, [JinaAI_](https://t.co/NnHhGudMa8) shared methods that offer significant performance gains at only a 4% accuracy cost. This optimization is crucial for production applications.
- **Building agentic multi-document RAG**: Plaban Nayak's article explains constructing a multi-document agent using LlamaIndex and Mistral. Each document is modeled as a set of tools for comprehensive summarization, available [here](https://t.co/FksUI3mm5l) and [here](https://t.co/MbDtlrxk5B).
- **Fully local text-to-SQL setup**: Diptiman Raichaudhuri offers a tutorial on setting up a local text-to-SQL system for querying structured databases without external dependencies. This guide is accessible [here](https://t.co/u3LG9NKE0X).
- **San Francisco meetup announcement**: LlamaIndex will host an in-person meetup at their HQ with talks from prominent partners including Tryolabs and Activeloop. The meetup will cover advanced RAG engine techniques; RSVP and more details can be found [here](https://t.co/o0BWxeq3TJ).
```

**Link mentioned**: <a href="https://t.co/qIGOmCW62G">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>: Note: This is an in-person meetup @LlamaIndex HQ in SF!  Stop by our meetup to learn about latest innovations in building production-grade retrieval augmented generation engines for your company from ...

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1240939005718302780)** (139 messages🔥🔥): 

- **MetaDataFilters for Data Governance**: A user figured out how MetaDataFilters in LlamaIndex work, applying filters directly on the DB level. They are curious whether MetaDataFilters are feasible for data governance in scalable applications and asked about selective indexing to restrict access to financial data.

- **Embedding with Neo4jVectorStore Issues**: A user reported errors when integrating LlamaIndex with an existing Neo4j graph containing pre-created embeddings and nodes. They discussed several methods for creating compatible nodes and embeddings using LlamaIndex to resolve this.

- **Model and Query Configuration Help**: Users discussed using different embedding models and query engines in LlamaIndex, including setting up environment variables, passing models to query engines, and handling embeddings setup issues. Several links to LlamaIndex documentation and examples were shared.

- **Challenges with Multi-Agent and Tools**: The conversation detailed issues with using multiple tools and agents within LlamaIndex, including confusion and inefficiencies in tool selection by agents like GPT-4. A user shared their workaround, including using a ReactAgent as a sub-agent.

- **Data Governance in RAG Applications**: A complex discussion on implementing data governance in RAG applications using LlamaIndex and Langchain was held. Links to talks and articles from NVIDIA and Microsoft about integrating access control were shared for deeper insights.

- **Miscellaneous LlamaIndex Queries**: Users asked about differences between chatbot engines and query engines, handling document duplicates in Pinecone, scraping data from the web for RAG applications, and modifying system prompts in OpenAI agents within LlamaIndex. Various solutions and troubleshooting steps were exchanged.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-s3?from=">no title found</a>: no description found</li><li><a href="https://llamahub.ai/">Llama Hub</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules">Redirecting...</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62731/">Retrieval Augmented Generation: A New Frontier in Governmental Efficiency | NVIDIA On-Demand</a>: We'll introduce retrieval augmented generation (RAG), an AI technology that can search and generate answers from large data sources</li><li><a href="https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/access-control-in-generative-ai-applications-with-azure-ai/ba-p/3956408">Access Control in Generative AI applications with Azure AI Search</a>: Apply access control in your Generative AI applications to enforce organizational policies and limit access to authorized content. </li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=insertion#insertion">Document Management - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Neo4jVectorDemo/?h=neo4j">Neo4j vector store - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/b26112b02be29eed82fc2b808eaf55bc51e472c7/llama-index-core/llama_index/core/readers/file/base.py#L68">llama_index/llama-index-core/llama_index/core/readers/file/base.py at b26112b02be29eed82fc2b808eaf55bc51e472c7 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/">Qdrant Hybrid Search - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 3 - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/?h=pandas+query">Pandas Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">no title found</a>: no description found</li><li><a href="https://git.tonic-ai.com/contribute/snowflake/fdabot">🛂contribute / ❄️snowflake / FDABot · GitLab</a>: 🙋🏻‍♂️ Welcome to🌟Tonic-AI Community
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1240977763679211541)** (4 messages): 

- **Multimodal Power Unleashed with GPT-4o**: A user shared a [Medium article](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) on integrating GPT-4o with LlamaParse. The link received positive reactions and a "nice!" comment from another member.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1240963475770048532)** (134 messages🔥🔥): 

- **CommonCanvas dataset splits opinions**: Members expressed mixed feelings about CommonCanvas, a dataset of 70M image-text pairs with both alt and synthetic captions, due to its restrictive non-commercial license. *"Seems entirely counterproductive"* and *"No derivatives is also odd, because wouldn't it be good if people modified/expanded on this dataset?"* show the frustration (link: [announcement](https://x.com/multimodalart/status/1791201296357142663)).

- **Challenges of torch.compile and GPU utilization**: Members, like drhead, discussed significant slowdowns due to PyTorch's native_group_norm and frequent device sync issues. The issue highlights the differences in performance between PyTorch’s eager mode and torch.compile (*"i have it running like, only 5% slower than what I can accomplish using torch.compile”*).

- **Concerns on hallucinations in AI captions**: There is an ongoing debate about the impact of hallucinated captions on training visual language models and text-to-image models (VLLMs and T2I). *"I've been talking to a lab that said hallucinations in captions are actually extremely damaging to VLLMs and T2I but I'm still waiting on the paper"*.

- **LLava and CogVLM discussed for dataset creation**: Members are exploring various AI models like LLava and CogVLM for captioning large datasets. While LLava-next and LLaMA models are gaining traction, they expressed skepticism over CogVLM's performance (*"cogvlm sucks too"*).

- **Aspirations for more open-source datasets**: Users are actively discussing the creation of high-quality, diverse datasets large enough for training foundational models, referencing projects like CC12M with various VLMs, and concerns over data integrity and accessibility. *"I will always open source mine”* and sentiments towards avoiding "hallucinations" in training data underline their efforts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/multimodalart/status/1791201296357142663">Tweet from apolinario (multimodal.art) (@multimodalart)</a>: Quite excited that CommonCanvas is JUST out! 🖼️  • First open source text-to-image models trained fully on openly licensed images (SD2 and SDXL architectures)  • The dataset, with ~70M openly license...</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/mustafasuleyman/status/1792623877744623806?t=t5EX1E--TJ-mAJJZtzX4eg&s=19">Tweet from Mustafa Suleyman (@mustafasuleyman)</a>: We are taking Copilot to the next level. 🚀  Copilot will see, hear, speak and help in real time.  Watch this demo to see what I mean. Soon your AI companion will start to live life alongside you, whe...</li><li><a href="https://x.com/OpenAI/status/1792443575839678909">Tweet from OpenAI (@OpenAI)</a>: We’ve heard questions about how we chose the voices in ChatGPT, especially Sky. We are working to pause the use of Sky while we address them.  Read more about how we chose these voices: https://openai...</li><li><a href="https://github.com/ProGamerGov/VLM-Captioning-Tools/blob/main/bad_caption_finder.py">VLM-Captioning-Tools/bad_caption_finder.py at main · ProGamerGov/VLM-Captioning-Tools</a>: Python scripts to use for captioning images with VLMs - ProGamerGov/VLM-Captioning-Tools
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1241016051563757698)** (13 messages🔥): 

- **Chameleon breaks new grounds**: The Chameleon model, introduced in an [arXiv paper](https://arxiv.org/abs/2405.09818), is a mixed-modal model capable of understanding and generating images and text simultaneously. It showcases **state-of-the-art performance** in tasks like image captioning and generative abilities surpassing even larger models like Llama-2.

- **Sakuga-42M, a game-changer for cartoon datasets**: An [arXiv study](https://arxiv.org/abs/2405.07425) introduces Sakuga-42M, the first large-scale cartoon animation dataset. The dataset comprises "42 million keyframes" and aims to fill the gap in cartoon-specific training data.

- **CogVLM2 license raises concerns**: **Warnings** were issued over the new CogVLM2 model's license, which states restrictive clauses regarding use against China's interests and mandating disputes be resolved by a Chinese court ([source](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/LICENSE), [GitHub](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)).

- **MambaOut steps in where Mamba stumbles**: The Mamba model, despite its architectural promise, underperforms in vision tasks compared to attentional and convolutional models ([arXiv paper](https://arxiv.org/abs/2405.07992)). Empirical evidence suggests Mamba isn't necessary for image classification, but its long-sequence capabilities still hold promise for detection and segmentation tasks.

- **Kobe Bryant Memed for Mamba's Performance**: Users humorously referenced Kobe Bryant’s famous quote "Mamba out" to comment on the underwhelming performance of the Mamba model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07425">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: Hand-drawn cartoon animation employs sketches and flat-color segments to create the illusion of motion. While recent advancements like CLIP, SVD, and Sora show impressive results in understanding and ...</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>: Mamba, an architecture with RNN-like token mixer of state space model (SSM), was recently introduced to address the quadratic complexity of the attention mechanism and subsequently applied to vision t...</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>: We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach f...</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: Low-Rank Adaptation (LoRA) is a widely-used parameter-efficient finetuning method for large language models. LoRA saves memory by training only low rank perturbations to selected weight matrices. In t...</li><li><a href="https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE">CogVLM2/MODEL_LICENSE at main · THUDM/CogVLM2</a>: 第二代 CogVLM多模态预训练对话模型. Contribute to THUDM/CogVLM2 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1240942274532085800)** (1 messages): 

- **Building a semantic research paper app**: A member shared a [recent article](https://towardsdatascience.com/building-an-observable-arxiv-rag-chatbot-with-langchain-chainlit-and-literal-ai-9c345fcd1cd8) on how to build a semantic research paper app using **LangChain, Chainlit, and Literal AI**. The article also includes steps on integrating observability features into the app.
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1241921904885895199)** (2 messages): 

- **4Wall reveals AI entertainment platform**: The team behind [4wall](https://beta.4wall.ai) is developing an AI-driven entertainment platform, currently in beta. A teaser video was shared on [X (formerly Twitter)](https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA).
- **AI Town integration and user-generated content**: 4Wall plans to integrate AI Town into their platform, allowing users to use bots seamlessly. They are also working on features for users to create maps and games.
- **3D AI characters in the pipeline**: The 4Wall team announced that 3D AI character functionality is in development and will be available soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://beta.4wall.ai">4Wall AI</a>: Explore interactive AI content on 4thWall AI</li><li><a href="https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA">Tweet from 4Wall AI (@4WallAI_)</a>: ✨coming soon on 4Wall✨ http://beta.4wall.ai
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/)** (1 messages): 

.ghost001: They gonna feel dumb when the more advanced versions come out
  

---


**AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1241197728050970624)** (1 messages): 

- **Announcing Rosebud AI Game Jam Winners**: The winners of the **Rosebud / #WeekOfAI Education Game Jam** were announced, showcasing incredible AI-powered educational games. The first-place game, **"Pathfinder: Terra’s Fate"**, and third-place **"Ferment!"** were highlighted for their engaging experiences. [Check the winners](https://x.com/Rosebud_AI/status/1791616913279160327) and try out Rosebud [here](https://play.rosebud.ai/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Rosebud_AI/status/1791616913279160327">Tweet from ⚡Rosebud AI🌹 (@Rosebud_AI)</a>: 🌟 Presenting the winners of our #WeekOfAI Education Game Jam! 🌟  These 4 incredible AI-powered games, made with Rosebud AI, showcase how to create fun educational gaming.  A huge thank you to our ju...</li><li><a href="https://play.rosebud.ai/">Play and Create Games on Rosebud — AI-Powered Game Development</a>: Use AI to create, share and play games. Go from text description to code to game.
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1240957770447196180)** (38 messages🔥): 

- **AI Town Now Runs Natively on Windows**: A member enthusiastically announced that **AI Town** now works natively on Windows without requiring WSL or Docker, celebrating the news with a [tweet from @cocktailpeanut](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964) confirming the launch of the AI Town 1 Click Launcher for Windows.
- **Launch of AI Reality TV Platform**: Excited members shared the [launch of the AI Reality TV platform](https://x.com/edgarhnd/status/1791586276178587707) that allows users to create social simulations and observe AI-powered characters interact. They encouraged others to join and hosted the next attraction, creating simulations like "Elisabeth choosing between Jack or Will in Pirates of the Caribbean."
- **Installation Issues and Solutions Shared**: A user encountered problems setting up AI Town conversations and was advised to check the [memory system documentation](https://github.com/a16z-infra/ai-town/blob/main/convex/agents.ts) and to adjust settings in `convex/constants.ts` to improve conversation persistence.
- **Extracting Conversations from SQLite Databases**: Users discussed methods to extract conversations from SQLite databases used by AI Town. Helpful SQL queries for exporting data were shared, and [links to relevant repositories](https://github.com/cocktailpeanut/townplayer/blob/main/index.html) were provided for further assistance in filtering and exporting conversation data.
- **Add Intriguing Characters and Watch AI Interactions**: Members shared their creative character additions to AI Town, such as spies and local reporters, and noted the realistic interactions. There were mentions of troubleshooting, and users shared experiences on how adjusting memory fetch settings can improve character interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/edgarhnd/status/1791586276178587707">Tweet from Edgar Haond 🎲 (@edgarhnd)</a>: excited to launch AI Reality TV today!  our new platform lets you create your own social simulations.   ever wondered if elisabeth preferred jack or will in pirates of the caribbean? now you can simul...</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1791495360541593964">Tweet from cocktail peanut (@cocktailpeanut)</a>: AI Town 1 Click Launcher comes to Windows!  Thanks to the hard work by the @convex_dev team, we finally have a Windows native convex binary (which powers AI Town).  AI Town--a fully hackable, persiste...</li><li><a href="https://www.aireality.tv/">AI Reality TV</a>: no description found</li><li><a href="https://x.com/emollick/status/1791695567212699874">Tweet from Ethan Mollick (@emollick)</a>: OK, I got a town of autonomous AI agents running locally on my machine and gave them all characters from Parks and Rec to play. Lets see what happens.</li><li><a href="https://youtu.be/UoJjeyQR66s?si=3EnN8hJO7UypY72K">We Made AI Town&#39;s Backend. Here&#39;s How.</a>: [0:00] - Introduction[1:15] - The Components[1:23] - Agents (https://github.com/a16z-infra/ai-town/blob/main/convex/agents.ts)[1:29] - The Engine (https://gi...</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI Browser</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">Tweet from cocktail peanut (@cocktailpeanut)</a>: Introducing AI Town Player  Did you know that the entire AI Town is stored in a single sqlite file via @convex_dev?    I reverse engineered the schema and built a web app that lets anyone REPLAY any A...</li><li><a href="https://github.com/cocktailpeanut/townplayer/blob/main/index.html">townplayer/index.html at main · cocktailpeanut/townplayer</a>: Replay AI Town. Contribute to cocktailpeanut/townplayer development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1241066877623734342)** (94 messages🔥🔥): 

- **AI Town gears up for AI Reality TV**: A link was shared for joining the AI Reality TV show [here](https://discord.com/invite/NtUXDUSnKa). Members are encouraged to create their own AI and help it win the show, detailed here: [AI Reality TV](https://www.aireality.tv/).

- **Technical details for AI Town shared**: The tech stack for AI Town was described as using Convex for backend, JS/TS for app logic, Pixi.js for graphics, Clerk for auth, and varying between Ollama and OpenAI for inference.

- **Error troubleshooting in AI Town**: Members encountered connection issues with AI Town on Windows, experiencing errors during agent communications. They were directed to seek further help on the Pinokio Discord server.

- **Saving and extracting conversations**: The possibility of using a web app to dump sqlite files from AI Town was discussed, with a link provided to [GitHub - Townplayer](https://github.com/cocktailpeanut/townplayer). Alternative methods include using any sqlite viewer and the convex dashboard for hosted versions.

- **World context integration suggestion**: It's noted that adding context directly into character prompts enriches the narrative in AI Town. There's a suggestion to use the world description for better context, and plans for Convex's hosted dashboard to work with local deployments were discussed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cocktailpeanut/status/1786421948638965870">Tweet from cocktail peanut (@cocktailpeanut)</a>: Introducing AI Town Player  Did you know that the entire AI Town is stored in a single sqlite file via @convex_dev?    I reverse engineered the schema and built a web app that lets anyone REPLAY any A...</li><li><a href="https://www.aireality.tv/">AI Reality TV</a>: no description found</li><li><a href="https://github.com/cocktailpeanut/townplayer">GitHub - cocktailpeanut/townplayer: Replay AI Town</a>: Replay AI Town. Contribute to cocktailpeanut/townplayer development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1241016110480883853)** (112 messages🔥🔥): 

- **Server responses with status 500 on function calls:** A user reported, *"when call with function calls,server response with status 500 and message 'Function calling is not supported by openrouter'."* There was no immediate resolution provided in the conversation.
- **Invalid model URLs causing application errors:** A user noted that navigating to an invalid model URL breaks the page with *“Application error: a client-side exception has occurred (see the browser console for more information)”* instead of a proper 404 error. The behavior differs based on whether the user is signed in or not.
- **Auto top-up payment issues:** Multiple exchanges discussed a problem where **auto top-up payments were declined**, resulting in a user’s credits falling below allowable limits and **being unable to manually top-up**. The issue was identified as likely being blocked by the user's bank (WISE EUROPE SA/NV).
- **Model recommendations and fine-tuning feedback:** Users shared their experiences with various models, with mentions of **“Cat-LLaMA-3-70B”**, **Midnight-Miqu models**, and the need for **better fine-tuning methods** as opposed to "random uncleaned data" approaches. One user noted, *"Try Cat-LLaMA-3-70B, it's very impressive when you actually manage to get it to work.”*
- **Wizard LM 8x22B request failure issues:** A user asked about frequent failures with **Wizard LM 8x22B** on OpenRouter, which were identified as temporary surges in request timeouts (408) from several providers. 

Reach the full conversation here: [OpenRouter Discord](https://discord.com/channels/1091220969173028894).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>: Explore OpenRouter's model list and recorded changes. Updates every hour.</li><li><a href="https://openrouter.ai/models/google/gejksdf">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/models/google/gejk.sdf">OpenRouter</a>: A router for LLMs and other AI models
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1241019330238480455)** (58 messages🔥🔥): 

- **Galore Layerwise lacks DDP support**: Users express frustration over the Galore Layerwise tool's **inability to support DDP (Distributed Data Parallel)**, indicating ongoing limitations in its functionality.

- **Large Chinese datasets raise interest**: A conversation around **fine-tuning a large 8B model with 1 billion tokens** in a non-English language, specifically Chinese, draws interest. Relevant dataset links: [Multimodal Art Projection (M-A-P)](https://huggingface.co/m-a-p) and [BAAI](https://huggingface.co/BAAI).

- **Gradient norm issues during fine-tuning**: Discussion reveals **unbounded growth in gradient norms** when using low rank during model fine-tuning, specifically with llama 3 8B. The issue seems rooted in saturated weights that cannot update gradients without significant perturbation.

- **GPT-4o's spammy tokens**: Members share concerns about [GPT-4o's tokens being polluted with spam and porn phrases](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/), highlighting flaws in the latest release's token parsing for Chinese language.

- **Commandr configuration for axolotl**: Issues related to setting up Commandr configurations are partly resolved by a [specific GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files). Users collaborate on testing and implementing this configuration to potentially merge into the project.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/">GPT-4o’s Chinese token-training data is polluted by spam and porn websites</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Untested! Screenshots (if appropriate) Types of changes  Social Handles (Optional)</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/">Pull requests · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://huggingface.co/m-a-p">m-a-p (Multimodal Art Projection)</a>: no description found</li><li><a href="https://huggingface.co/BAAI">BAAI (Beijing Academy of Artificial Intelligence)</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1241006527091970090)** (13 messages🔥): 

- **Impact of Datasets on PoSE**: A member questioned if the choice of dataset significantly impacts the quality of context extension in PoSE. Another member responded, "*I didn't play around with the datasets much*."
- **Unsloth Optimizations for Llama**: A member inquired if there was any reason not to use the [Unsloth optimizations for Llama](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609) for a full finetune. Another member replied that the **Unsloth** cross entropy loss is fine for full finetuning.
- **Random Datasets Suffice for PoSE**: When asked if a random dataset was good enough for PoSE, a member confirmed, "*Yeah, good enough for niah, but honestly PoSE doesn't seem to really scale up long context reasoning or understanding*."
- **Torchtune Optimizations**: A member highlighted potential valuable optimizations from [Torchtune pull request #993](https://github.com/pytorch/torchtune/pull/993). They mentioned, "*torchtune integration with axolotl is coming SOON*."
- **Future of HF Backend**: Members discussed whether Torchtune would replace the HF backend or just be another option. One suggested, "*Dismantle hf*," signaling a desire for significant change.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/hpcai-tech/grok-1">hpcai-tech/grok-1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609">Unsloth optims for Llama by winglian · Pull Request #1609 · OpenAccess-AI-Collective/axolotl</a>: WIP to integrate Unsloth&#39;s optimizations into axolotl. The manual autograd for MLP, QKV, O only seems to help VRAM by 1% as opposed to the reported 8%. The Cross Entropy Loss does help significant...</li><li><a href="https://github.com/pytorch/torchtune/pull/993">Llama3-70b: Full Finetune w/CPU offload + fused optimizer by rohan-varma · Pull Request #993 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Please link to any issues this PR addresses. Changelog  ...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1241152120778915911)** (3 messages): 

- **Continued pre-training yields illegal memory access**: A user requested an example of using Axolotl for continued pre-training, noting that their attempt with a pretraining dataset resulted in *out-of-vocab padding tokens leading to illegal memory access*. They specified they do not want to change the vocab of the tokenizer and provided a sample configuration.

- **Issues with Mistral 7b fine-tuning**: A user shared their challenge of fine-tuning **Mistral 7b** on their instruct data, observing that despite the loss dropping, the model *"mixes things up and seems like it didn't learn anything"*. They mentioned that their configuration is based on [this example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml), with a few custom adjustments.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1241229094906106056)** (37 messages🔥): 

- **Phorm assists with ORPO format queries**: Users inquired about the ORPO (Object-Role-Property-Operation) format. Though no specific implementation details were given, it was noted that **ORPO is used for structuring data/operations** and an example in Axolotl included its use in prompt strategies.

- **Weight decay clarified for LLM training**: Members discussed weight decay, which acts as a regularization technique **preventing overfitting** by adding a penalty to the loss function. This ensures model weights remain small, leading to better generalization.

- **LoRA Dropout explained**: LoRA Dropout helps in fine-tuning LLMs by introducing dropout in the **low-rank adaptation matrices**, preventing overfitting and improving generalization.

- **Gradient accumulation benefits LLM training**: Gradient accumulation enables training with larger effective batch sizes without increasing memory usage, which is crucial for LLM stability and efficiency. This approach was exemplified using PyTorch and the **Hugging Face Accelerator library**.

- **Sample weights in Axolotl**: A member sought to assign **sample weights without customizing loss functions**. It was recommended to use sample weights in the `compute_loss` method for custom loss handling, while cautioning that not all loss functions support this natively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/core/trainer_builder.py#L484L493)">axolotl/src/axolotl/core/trainer_builder.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/prompt_strategies/orpo/chat_template.py#L205L239)">axolotl/src/axolotl/prompt_strategies/orpo/chat_template.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mistral-qlora-orpo.yml#L1L83),">axolotl/examples/mistral/mistral-qlora-orpo.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b80342a0-719c-4ab5-9320-9d50afdf43da)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8005a5ee-d28e-42f2-960d-3c29cc0e03ad)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=72e80667-3df2-495a-8fec-c8acfd801de6)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fcde63a8-860b-4760-a6c8-03900deac358)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=7330f4bb-2f39-4e78-8f4d-36d7317c666c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f4c68df8-2e30-4ffb-b40c-ab1c79f91770)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1240927402327281766)** (69 messages🔥🔥): 

- **LangChain handling re-ranking and memory**: A member asked about re-ranking results using a cross encoder behind an organizational proxy, wondering if it’s possible with OpenAI GPTs or Gemini models. Another member stressed the importance of implementing short-term memory like buffer memory for chatbots.
- **Guiding model responses in LangChain**: One member inquired about setting specific questions in a React agent to guide the model for optimal answers. Another member clarified by suggesting using a custom prompt or template via the `PromptTemplate` function in LangChain, shared with a [GitHub issue link](https://github.com/langchain-ai/langchain/issues/18820).
- **LangChain for Swift developers**: A member asked if LangChain is available for Swift developers working on iOS or macOS. Another member shared a [GitHub link for LangChain Swift](https://github.com/buhe/langchain-swift) optimized for iOS, macOS, watchOS, and visionOS.
- **Handling SQL data in LangChain**: One member working on summarizing call center calls discussed summarizing concepts across multiple calls and asked for ways to utilize SQL data as memory in LangChain. Another member recommended various integrations with SQL-like databases and shared a [LangChain integrations link](https://python.langchain.com/v0.1/docs/integrations/memory/).
- **Langmem's contextual capabilities**: A member expressed amazement at Langmen's ability to maintain contextual conversations when switching topics mid-session and shared [YouTube videos](https://youtu.be/7QU89qL795A) demonstrating Langmem's long-term memory and context management.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/integrations/memory/">Memory | 🦜️🔗 LangChain</a>: no description found</li><li><a href="https://app.reclaim.ai/m/cp/ai-storytelling-and-gaming">AI Storytelling and Gaming</a>: Hi - I&#x27;m Chris, and I&#x27;m trying to learn how people use AI to tell stories and play games. If you&#x27;ve tried apps such as AI Dungeon or Novel AI, or just used ChatGPT to try and tell a sto...</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md">openai-python/chatml.md at release-v0.28.0 · openai/openai-python</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://youtu.be/OL6RDg04FNc">Langmem - Episode 2 | Managing context switching</a>: In this recording, I show how langmem is able to help continue contextual conversation. It is able to switch between contextsprevious video: https://youtu.be...</li><li><a href="https://github.com/Dataherald/dataherald?tab=readme-ov-file">GitHub - Dataherald/dataherald: Interact with your SQL database, Natural Language to SQL using LLMs</a>: Interact with your SQL database, Natural Language to SQL using LLMs - Dataherald/dataherald</li><li><a href="https://github.com/langchain-ai/langchain/issues/18820>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://youtu.be/7QU89qL795A">Langmem | Long term memory from Langchain</a>: In this recording, I speak about langmem. This is one of the newest innovation from langchain. It focuses on long term memory. I believe we should focus on m...</li><li><a href="https://github.com/buhe/langchain-swift">GitHub - buhe/langchain-swift: 🚀 LangChain for Swift. Optimized for iOS, macOS, watchOS (part) and visionOS.(beta)</a>: 🚀 LangChain for Swift. Optimized for iOS, macOS, watchOS (part) and visionOS.(beta) - buhe/langchain-swift</li><li><a href="https://www.reddit.com/r/LangChain/s/0e0H0tm1o1">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1241509355355111465)** (2 messages): 

- **Kenny Tang Shares $50 Steam Gift Link**: KennyTang posted a link purportedly for a $50 Steam gift: [steamcommunity.com/gift/50](https://bitly.cx/OjEZl). The message tagged @everyone and @here.
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1241509366599909456)** (2 messages): 

- **Suspicious $50 Gift Link Shared**: A user shared a link titled *"Gift 50$"* directing to [steamcommunity.com](https://bitly.cx/OjEZl). The link was shared multiple times and tagged everyone in the channel.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1241035600455401502)** (7 messages): 

- **Rubik's AI offers free premium access**: A member introduced an advanced research assistant and search engine, inviting beta testers with a two-month free premium access using the promo code **RUBIX**. Models offered include GPT-4 Turbo, Claude-3 Opus, and Mistral Large. [Check it out](https://rubiks.ai/).

- **LangServe blogpost shared**: A member shared a link to their blog post about LangServe. [What is LangServe?](https://flatteredwithflutter.com/what-is-langserve/).

- **Questionable $50 Steam gift link**: A member posted two messages with a potentially dubious $50 gift link on Steam. The link was [here](https://bitly.cx/OjEZl).

- **Affiliate program for ChatGPT Chrome Extension**: Another member announced an affiliate program for their **Easy Folders** Chrome extension. Affiliates can earn a 25% commission while customers get a 10% discount. [Register here](https://easyfolders.promotekit.com/) and [download the extension](https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rutamstwt/status/1792091646663754226">Tweet from Rutam Bhagat (@rutamstwt)</a>: http://x.com/i/article/1792080553656922112</li><li><a href="https://x.com/the_mint_flip/status/1791944845772132433?s=46&t=RFXQiGP9iFKCCIMhv9N8qQ">Tweet from AmpJemima.Cro (@the_mint_flip)</a>: ‼️HAAaaLLP #crofam ‼️  🚨I need 84 more points🚨  🫵 SIGNUP FOR THE $Flu #airdrop 🫵  🐥😷🐥 http://trop.ee/nr9hRS5hyR🐥😷🐥  #bornbrave  #fftb  #cronosMemesDegens #CronosMemes #cronoschain #CronosMem...</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://easyfolders.promotekit.com/">Sign up</a>: Affiliate Marketing Software for Stripe</li><li><a href="https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0">Easy Folders: ChatGPT &amp; Claude Chat Organizer</a>: Drag &amp; drop folders for ChatGPT &amp; Claude. Colored folders. Nested folders. History search. Bookmarks. Bulk delete chats.</li><li><a href="https://chatgpt-easy-folders.vercel.app/">ChatGPT Easy Folders</a>: A browser extension to organize your ChatGPT history with folders, bookmarks, and search.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1241424375782182983)** (3 messages): 

- **RAG-Fusion simplifies multi-query handling**: A detailed message highlights the differences between **RAG** (single query) and **RAG-Fusion** (multi-query) and provides insights into integrating LangChain and GPT-4o for creating AI chatbots for document handling. Check out this [YouTube tutorial](https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP) for more information.
- **Questionable $50 Steam gift link posted**: A link claiming to offer a $50 Steam gift was shared ([suspicious link](https://bitly.cx/OjEZl)) and tagged to everyone. Caution is advised regarding such links.

**Link mentioned**: <a href="https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP">LangChain + RAG Fusion + GPT-4o Python Project: Easy AI/Chat for your Docs</a>: #automation #rag #llm #ai #programming #gpt4o #langchain in this Video, I have a super quick tutorial for you showing how to create an AI for your PDF with L...

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1240926095012794428)** (76 messages🔥🔥): 

- **Improving Discord Support**: A member suggested changes to the support system on Discord, indicating long-standing issues with unanswered questions. Another member clarified that the channel operates more as a community-supported chat rather than official staff-supported system.
  
- **Rate Limit Issues with Trial API**: A user experimenting with a `RAG retriever` faced a 403 error and speculated reaching the limit of the Trial API. Others mentioned that trial keys are rate-limited and not intended for production.

- **Free API Keys**: Queries arose about acquiring free API keys and their limitations. A member confirmed that free keys are available but limited, suitable primarily for prototyping rather than production.

- **Translations with CommandR+**: A user sought examples for using `CommandR+` for translation. Another member recommended referencing the [Chat API documentation](https://docs.cohere.com/docs/chat-api) for implementation details.

- **Portfolio vs. Production Use**: Discussion occurred about hosting apps using Cohere AI on platforms like Vercel for portfolio purposes. It was clarified that execution within portfolios is generally considered prototyping and falls under free usage, whereas production involves commercial deployment and incurs costs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/chat-api">Using the Chat API</a>: no description found</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/retrievers/cohere/">Cohere RAG | 🦜️🔗 LangChain</a>: Cohere is a Canadian startup that provides natural language processing models that help companies improve human-machine interactions.</li><li><a href="https://huggingface.co/Xenova/codegen-350M-mono">Xenova/codegen-350M-mono · Hugging Face</a>: no description found</li><li><a href="https://cohere.com/pricing">Pricing</a>: Flexible, affordably priced natural language technology for businesses of all sizes. Start for free today and pay as you go.
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1241351465574862940)** (1 messages): 

- **A Complete Guide to Cohere AI Published**: A member announced their blog post titled "A Complete Guide to Cohere AI" on [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#). This guide covers the installation, setup, and usage of Cohere's Enterprise AI platform, including a demo app available at [Streamlit](https://cohere-guide-blog.streamlit.app) for practical understanding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#">A Complete Guide to Using Cohere AI</a>: Unlock insights, automate tasks, and enhance experiences with Cohere&#039;s Enterprise AI platform. Install, customize, and deploy effortlessly.</li><li><a href="https://cohere-guide-blog.streamlit.app">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1240923485853716532)** (41 messages🔥): 

- **Hugging Face pledges $10 million in free GPUs**: Hugging Face is committing $10 million in free shared GPUs to help developers create new AI technologies, aiming to assist small developers, academics, and startups. CEO Clem Delangue cited the company's profitability and recent funding as enablers of this initiative, according to [The Verge](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai).

- **Successful Pi 5 installation reported**: A member confirmed successfully running OpenInterpreter on a Pi 5 with Ubuntu without local models, using GPT4 for various tasks. Another user expressed interest in combining projects and received an offer for Azure credits to assist with the integration.

- **Platform tips and troubleshooting**: Members shared tips on setting up OpenInterpreter using WSL, virtual environments, and various IDEs. One user solved issues with GPT-4o on OpenRouter by upgrading the `litellm` dependency, highlighting potential areas for improving OpenInterpreter's default settings.

- **Event and streaming announcements**: The community was invited to the first Accessibility Round Table event, aiming to discuss technology's benefits for everyone. Additionally, a member announced a live stream on X for local development, encouraging others to join in.

- **Seeking project collaboration**: A junior full-stack DevOps engineer sought help building a "lite 01" AI assistant module for simplifying daily tasks and providing discreet assistance in work environments. The request highlighted the need for comprehensive resources on DevOps tools and cloud computing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1241028896846254141">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face is sharing $10 million worth of compute to help beat the big AI companies</a>: Hugging Face is hoping to lower the barrier to entry for developing AI apps.</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models.">Introduction - Open Interpreter</a>: no description found</li><li><a href="https://www.youtube.com/shorts/dpkzijtXOqw">HoloMat Update: Jarvis controls my printers! #engineering #3dprinting #ironman</a>: no description found</li><li><a href="https://denise.ai/">Denise Legacy - Virtual Assistant Denise</a>: Denise is alive! And Deniise 2.0 is coming! The moment we all waited for has come! Denise Legacy is available for purchase! Get Denise Legacy for only USD 49,90 Lifetime Promotion Deniise 2.0 with Cha...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1241023239740723250)** (15 messages🔥): 

- **Troubleshoot: Connection Issues Resolved Quickly**: One user had an initial problem connecting to the app but successfully resolved it with a provided example format. Another member admired the app's beauty and "native" feel, confirmed to be built in Swift.

- **Server Setup Tips for Windows Users**: A member asked for advice on whether to run the server using Ubuntu in Windows or PowerShell. Another user shared their setup method, leveraging poetry to run OpenInterpreter with specific parameters and ensuring the correct local IP and port are used.

- **Clarifying Environment Use**: A newcomer had questions about using Linux VM for OpenInterpreter. There was confirmation that it is feasible and will interact correctly with OpenInterpreter running directly on the host computer.

- **GitHub Resources Shared**: A link to a [GitHub repository](https://github.com/Tonylib/o1_for_flutter) was shared, highlighting a project related to running O1 on Flutter. The discussion included contributions and development guidance.

- **Community Projects and Assistance Requests**: A member discussed their ongoing build of an O1 Lite device, with all parts and a 3D-printed case. Another user seeking help to develop an AI module for task simplification and remote assistance appealed for community support due to pre-order delays.

**Link mentioned**: <a href="https://github.com/Tonylib/o1_for_flutter">GitHub - Tonylib/o1_for_flutter</a>: Contribute to Tonylib/o1_for_flutter development by creating an account on GitHub.

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1240922872139087922)** (6 messages): 

- **Google DeepMind tunes into Google IO**: A member shared a post from [GoogleDeepMind](https://x.com/GoogleDeepMind/status/1790463259822420239) discussing their involvement with Project Astra at #GoogleIO. Another member commented: *"Google really is stepping up their game"*.
- **Voice AI still has robotic limitations**: There was a debate about the current state of voice AI, with one member stating, "voice is a bit too robotic," suggesting it lags behind GPT-4's capabilities.
- **Intriguing idea for AI voice interaction**: A YouTube short shared by a member discussed a new idea for AI voice assistants, emphasizing their ability to interrupt users ([YouTube video](https://www.youtube.com/shorts/zgUanlLV_OQ)). One user humorously added that it was a "missed opportunity to make it moo."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1790463259822420239">Tweet from Google DeepMind (@GoogleDeepMind)</a>: We watched #GoogleIO with Project Astra. 👀</li><li><a href="https://www.youtube.com/shorts/zgUanlLV_OQ">interesting idea for a AI voice assistants I don&#39;t think I&#39;ve seen#voiceai #gpt4o #interruptingcow</a>: no description found
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1241160048567124048)** (26 messages🔥): 

- **Segfault in RAG tutorial troubleshooting**: A user encountered a segfault when querying their index while following the [RAG tutorial](https://future.mozilla.org/news/llamafiles-for-embeddings-in-local-rag-applications/). Key log message was *"llama_get_logits_ith: invalid logits id 420, reason: no logits"*; another user suggested checking the codebase, leading to a realization that the models were embeddings-only.
  
- **Llamafile embedding model clarification**: It was clarified that the embeddings model linked in the [tutorial](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/tree/main) cannot perform generation, which wasn't immediately clear from the examples.

- **Cloud deployment discussions**: Users discussed various cloud providers for running Llamafile with a preference for GPU-enabled services. [vast.ai](https://vast.ai) was recommended for experiments and short-lived workloads.

- **SQLite for vector search project**: Alex Garcia introduced his project [sqlite-vec](https://github.com/asg017/sqlite-vec), a SQLite extension for vector search, with the intention of integrating it into Llamafile. The project promises features like memory and semantic search, and already has beta release assets available.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://<Your">no title found</a>: no description found</li><li><a href="https://salad.com/">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>: Save up to 90% on your cloud bills. Deploy AI/ML production models easily. 600% more images &amp; 10x more inferences per dollar. Try SaladCloud for free today.</li><li><a href="https://fly.io/docs/gpus/gpu-quickstart/">Fly GPUs quickstart</a>: Documentation and guides from the team at Fly.io.</li><li><a href="https://github.com/beetbox/beets/issues/1166#issuecomment-68076160">convert: Character encoding/path issue on Windows · Issue #1166 · beetbox/beets</a>: The convert plugin seems to have issues with certain characters. Here is the cmd.exe output when importing: C:\Users\Michael\Desktop\blink-182\blink-182&gt;beet import . C:\Users\Michael\Desktop\blink...</li><li><a href="https://future.mozilla.org/news">Mozilla Innovation Projects | Recent Articles</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile-rag-example/blob/main/app.sh">llamafile-rag-example/app.sh at main · Mozilla-Ocho/llamafile-rag-example</a>: Contribute to Mozilla-Ocho/llamafile-rag-example development by creating an account on GitHub.</li><li><a href="https://github.com/asg017/sqlite-vec">GitHub - asg017/sqlite-vec: Work-in-progress vector search SQLite extension that runs anywhere.</a>: Work-in-progress vector search SQLite extension that runs anywhere. - asg017/sqlite-vec</li><li><a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">I'm writing a new vector search SQLite Extension</a>: sqlite-vec is an new vector search SQLite extension, coming soon!</li><li><a href="https://github.com/asg017/sqlite-vec/releases">Releases · asg017/sqlite-vec</a>: Work-in-progress vector search SQLite extension that runs anywhere. - asg017/sqlite-vec
</li>
</ul>

</div>
  

---



**MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1241552403052892212)** (9 messages🔥): 

- **FOMO into LLM Fine-Tuning Course**: A member shared their enthusiasm for joining an [LLM Fine-Tuning course](https://maven.com/parlance-labs/fine-tuning). The course promises hands-on experience with LLMs, covering topics from training to deployment, with workshops on evaluation, instrumentation, and prompt engineering.
- **Skepticism About Course Offerings**: Another member expressed skepticism about the course, suggesting it might be "fluff" due to the promotional giveaways and the wide range of experts involved. They questioned the value versus the marketing tactics used to attract participants.
- **Mixed First Week Impressions**: Feedback from participants about the first week of the course varied. One described it as "rather basic," focusing on introductory topics like finding use cases for LLMs, which might depend heavily on participants' prior experience.

**Link mentioned**: <a href="https://maven.com/parlance-labs/fine-tuning">Mastering LLMs: End-to-End Fine-Tuning and Deployment by Dan Becker and Hamel Husain on Maven</a>: All-time best selling course on Maven! Train, validate and deploy your first fine-tuned LLM

  

---


**MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1241132122107019387)** (7 messages): 

- **MAPIE for Prediction Intervals**: A member asked for recommendations on implementing prediction intervals and shared a link to [MAPIE's documentation](https://mapie.readthedocs.io/en/latest/). This tool is being explored for its utility in this context.

- **Valeriy Manokhin on Conformal Predictions**: Another member suggested Valeriy Manokhin's [Medium](https://valeman.medium.com/) for conformal predictions, noting Manokhin's preference for Nixtla, which might be relevant for time series data.

- **Image Embeddings via Inpainting**: A query was raised about deriving image embeddings using image inpainting or context encoding, comparing it to masked language modeling. This method involves predicting hidden parts of an image using the visible portions.

- **Multi-lingual Entity Extraction**: Discussions evolved around the challenge of making multi-lingual entities like “University of California” and “Universidad de California” comparable. Suggestions included using contrastive learning and prefixing tasks with language identifiers, as seen in some strategies for query and document encoding. 

- **Applying Ideas from Relevant Papers**: A member recommended applying concepts from a recent paper on [arxiv](https://arxiv.org/pdf/2401.12178), mentioning it as a part of their ongoing work for multi-lingual entity extraction this week.

**Link mentioned**: <a href="https://mapie.readthedocs.io/en/latest/">MAPIE - Model Agnostic Prediction Interval Estimator &mdash; MAPIE 0.8.3 documentation</a>: no description found

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1241588106084487168)** (7 messages): 

- **Running YOLO model on comma device**: A member asked if anyone tried to run a YOLO model on a comma device and mentioned that they are getting predictions in *about ~1000ms*. They didn't provide further details on the specifics of the model version or optimizations used.

- **Polynomial degree limits for sin approximation**: Members discussed the limitations of using high-degree polynomials for approximating the sine function. One user noted they are using a degree 11 polynomial with an error about *1e-8*, but it doesn't meet the test requirement of *1e-12* error, and they are contemplating increasing the degree despite performance concerns.

- **Accuracy concerns in polynomial approximations**: Another user highlighted that for sine, periodicity helps to manage accuracy issues, but warned about significant accuracy loss when approximating functions like the logarithm and exponential. They advised using range reduction techniques to maintain accuracy but recognized the challenge in meeting high precision requirements without increasing computational complexity.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1241425883756105840)** (4 messages): 

- **Question on bitshifting in tinygrad**: A member asked if there is a more efficient way to bitshift in tinygrad other than using the expression `x.e(BinaryOps.DIV, 2 ** 16).e(BinaryOps.MUL, 2 ** 16)`.

- **Unwrapping for loops in Metal compiler**: Another member shared a code snippet and asked about where the Metal compiler decides to unwrap a for loop. They highlighted the generated Metal code for `Tensor.arange(1, 32)`.

- **Comparison of generated code with Tensor.arange(1, 33)**: The same member demonstrated that using `Tensor.arange(1, 33)` instead of 32 results in significantly different generated Metal code, which includes the use of threadgroup variables and barriers.

- **Puzzling magic number 32**: The member also questioned why the number 32 specifically results in different compilation behavior in the Metal compiler, pointing out a noticeable performance implication.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1240941859077881896)** (6 messages): 

- **Claude3 support for Squeak Smalltalk**: One user floated the idea of adding **Claude3 support** to **Squeak Smalltalk**. Details about implementation or benefits were not discussed, but it signals growing interest in integrating advanced models with legacy programming environments.
  
- **GPT-4o Demo Voice Modes Explained**: Another user shared that the **voice in the GPT-4o demo** was initially included in the version 1 Voice Mode and called **Sky**, speculating it was the default option. OpenAI paused its use following the realization it inadvertently resembled Scarlett Johansson's voice, replacing it with a new feminine voice, **Juniper**.

- **Latency and Model Integration in Voice Mode**: A user referenced an article detailing how previous Voice Mode versions used separate models for transcription, processing, and audio output, resulting in latency issues. **GPT-4o** now consolidates these features in a single model, enhancing emotional expression, although this has introduced complexity and potential unpredictability ([source](https://help.openai.com/en/articles/8400625-voice-chat-faq)).

- **Concerns on AI Complexity and Prompt Injection**: Additional discussion centered on how advanced capabilities, like those in GPT-4o, bring significant drawbacks, such as susceptibility to **prompt injection**. The increased complexity of new models may lead to unpredictable behavior and higher chances for user-annoying outputs, similar to the problems of legacy systems being overridden by new instructions.

- **Resilience in Fault-Tolerant Systems**: Quoting Stainslaw Lem's "The Upside-Down Evolution," a user pointed out that while total reliability is unattainable, particularly in complex systems, building resilient infrastructures is key. They stressed that as systems evolve to be more fault-tolerant, new issues inevitably arise, echoing Lem’s notion of moving "from the frying pan to the fire."
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1242201665835237448)** (1 messages): 

- **GPT-4o excels at complex legal reasoning**: A member ran internal evaluations on **GPT-4o** for complex legal reasoning tasks, noting a non-trivial improvement over **GPT-4** and **GPT-4-Turbo**. More details can be found in their [LinkedIn post](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1).
  

---



**YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/1241717102315049031)** (1 messages): 

- **Call for Contributors on Docker and AI**: A member announced their plan to write an article focused on **using Docker containers for training and deploying AI**. They invited others to help, contribute, or review the draft, and asked interested individuals to DM them.
  

---



---




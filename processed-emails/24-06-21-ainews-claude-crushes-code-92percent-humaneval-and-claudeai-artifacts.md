---
id: 81ca9896-156e-42c9-ae5d-7dd740c11ba8
title: Claude Crushes Code - 92% HumanEval and Claude.ai Artifacts
date: '2024-06-21T07:27:45.182774Z'
original_slug: ainews-claude-crushes-code-92-humaneval-and
description: >-
  Anthropic announced Claude 3.5 Sonnet, a new AI model that outperforms GPT-4o
  on benchmarks like GPQA, MMLU, and HumanEval. It features improved speed, cost
  efficiency, and enhanced coding and content generation capabilities, including
  a new Artifacts feature for real-time content editing.
companies:
  - anthropic
models:
  - claude-3-5-sonnet
topics:
  - large-language-models
  - benchmark-performance
  - multitask-learning
  - code-generation
  - content-creation
  - artifacts-feature
---


<!-- buttondown-editor-mode: plaintext -->**Claude 3.5 Sonnet is all you need?**

> AI News for 6/19/2024-6/20/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**415** channels, and **3577** messages) for you. 
Estimated reading time saved (at 200wpm): **392 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The news of the day is nominally [Claude 3.5 Sonnet](https://x.com/AnthropicAI/status/1803790676988920098) - ostensibly Anthropic's answer to GPT-4o:

 ![image.png](https://assets.buttondown.email/images/4797f203-e0e9-400a-8559-4c5750b6db5a.png?w=960&fit=max) 

Including claiming SOTA on GPQA, MMLU, and HumanEval:

 ![image.png](https://assets.buttondown.email/images/bb851f1b-63b5-4011-b88a-bfeb88f23a72.png?w=960&fit=max) 

as well as "[surpassing Claude 3 Opus across all standard vision benchmarks](https://x.com/AnthropicAI/status/1803790684857536522)".

https://www.youtube.com/watch?v=dhxrHvgXpSM&embeds_referring_euri=https%3A%2F%2Fwww.anthropic.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.anthropic.com&feature=emb_title

The [model card](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf) demonstrates the Opus-level context utilization now extending to Sonnet:

 ![image.png](https://assets.buttondown.email/images/d9f417a7-7a67-44db-bce7-85987ac52a09.png?w=960&fit=max) 

We don't have a ton of technical detail on what drives the changes, but Anthropic is selling this as a Pareto improvement over 3 Sonnet and 3 Opus:


 ![image.png](https://assets.buttondown.email/images/33a9708c-36b1-432b-86bc-fb113a0bbf66.png?w=960&fit=max) 

> **Claude 3.5 Sonnet operates at twice the speed of Claude 3 Opus**. This performance boost, combined with cost-effective pricing, makes Claude 3.5 Sonnet ideal for complex tasks such as context-sensitive customer support and orchestrating multi-step workflows.

However the bigger focus of the messaging beyond general capability and efficiency improvements is Claude Sonnet's coding ability:

> "**Claude is starting to get really good at coding and autonomously fixing pull requests.** It's becoming clear that in a year's time, a large percentage of code will be written by LLMs." - [Alex Albert](https://x.com/alexalbert__/status/1803804682412007850)

https://www.youtube.com/watch?v=A598ESCoC70

 ![image.png](https://assets.buttondown.email/images/f8867131-8ce1-43ac-8b8c-6eaaff5bf347.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/680e9e6c-6ea1-4911-9688-9523eabaa880.png?w=960&fit=max) 

This seems to be backed up by Claude.ai's release of "Artifacts":

> a new feature that expands how users can interact with Claude. **When a user asks Claude to generate content like code snippets, text documents, or website designs, these Artifacts appear in a dedicated window alongside their conversation.** This creates a dynamic workspace where they can see, edit, and build upon Claude’s creations in real-time, seamlessly integrating AI-generated content into their projects and workflows. 

This would seem like Anthropic's answer to OpenAI's Code Interpreter or Cognition Labs' Devin.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Claude 3.5 Sonnet Release by Anthropic**

- **Performance**: [@alexalbert__](https://twitter.com/alexalbert__/status/1803804677701869748) noted Claude 3.5 Sonnet outperforms competitor models on key evaluations, at **twice the speed** of Claude 3 Opus and **one-fifth the cost**. It shows marked improvement in grasping nuance, humor, and complex instructions. [@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790676988920098) highlighted it now outperforms GPT-4o on several benchmarks like **GPQA, MMLU, and HumanEval**.
- **Artifacts Feature**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790681971859473) introduced Artifacts, allowing users to generate docs, code, diagrams, graphics, or games that appear next to the chat for real-time iteration. [@alexalbert__](https://twitter.com/alexalbert__/status/1803804686501507418) noted he's stopped using most simple chart, diagram, and visualization software due to this.
- **Coding Capabilities**: In Anthropic's internal pull request eval, [@alexalbert__](https://twitter.com/alexalbert__/status/1803804682412007850) shared Claude 3.5 Sonnet passed **64% of test cases vs 38% for Claude 3 Opus**. [@alexalbert__](https://twitter.com/alexalbert__/status/1803804689538171351) quoted an engineer saying it fixed a bug in an open source library they were using.
- **Availability**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790689408332059) noted it's available for free on claude.ai and the Claude iOS app. Claude Pro and Team subscribers get higher rate limits. Also available via Anthropic API, Amazon Bedrock, Google Cloud's Vertex AI.

**Ilya Sutskever's New Company: Safe Super Intelligence (SSI)**

- **Goal**: [@ilyasut](https://twitter.com/ilyasut/status/1803472979873128498) stated they will pursue safe superintelligence in a straight shot, with one focus, one goal, and one product through revolutionary breakthroughs by a small cracked team. 
- **Reactions**: Some like [@bindureddy](https://twitter.com/bindureddy/status/1803475778019164211) praised the focus on AGI without obsessing about money. Others like [@DavidSHolz](https://twitter.com/DavidSHolz/status/1803542447206879439) compared it to the Yahoo/AOL/pets dot com era of AI. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1803611506908529060) speculated this destroys possibility of a binding USA-China AGI/ASI treaty.
- **Funding**: [@ethanCaballero](https://twitter.com/ethanCaballero/status/1803494001091268756) questioned how SSI will raise $10B in one year or they're "dead on arrival". 

**AI Benchmarks and Evaluations**

- **Mixture of Agents (MoA)**: [@corbtt](https://twitter.com/corbtt/status/1803813970018791845) introduced MoA model+FT pipeline that beats GPT-4 but is 25x cheaper. Humans prefer MoA outputs vs GPT-4 59% of the time. New SOTA on Arena-Hard (84.8) and Alpaca Eval (LC 68.4).
- **Infinity Instruct**: [@_philschmid](https://twitter.com/_philschmid/status/1803679786079830449) shared this 3M sample deduplicated Instruction dataset. 10M sample version planned for end of June. SFT experiments for Mistral 7B achieve 7.9 on MT Bench, boost MMLU by 6% and HumanEval to 50%.
- **τ-bench**: [@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1803849363506237636) introduced τ-bench at Sierra Platform to evaluate critical agent capabilities omitted by current benchmarks: robustness, complex rule following, human interaction skills.

**Memes and Humor**

- Meme about Logi AI Prompt Builder on an AI mouse: [@nearcyan](https://twitter.com/nearcyan/status/1803583533690008030)
- Meme about Yahoo/AOL/pets dot com era of AI: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1803542447206879439) 
- Encrypted Shakespeare sonnet about Claude 3.5: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1803774865473696237)
- Meme about SSI raising $10B in funding: [@bindureddy](https://twitter.com/bindureddy/status/1803546758406086767)

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Companies and Developments**

- **Dell partnering with NVIDIA on "AI factory"**: In a [tweet](https://x.com/MichaelDell/status/1803385185984974941), Michael Dell announced Dell is building an "AI factory" with NVIDIA to power "grok for xAI", hinting at a major AI infrastructure initiative between the two tech giants.
- **Anthropic's Claude AI demonstrates strong legal reasoning**: According to an [analysis](https://adamunikowsky.substack.com/p/in-ai-we-trust-part-ii), Anthropic's Claude AI matched Supreme Court findings in **27 out of 37 cases**, showcasing its ability to comprehend and reason about complex legal issues.
- **Meta's Chameleon language model training datasets revealed**: Model files for Meta's Chameleon AI show it was [trained on](https://www.reddit.com/r/LocalLLaMA/comments/1dk5a5q/chameleon_model_files_list_the_datasets_meta_used/) diverse datasets spanning legal content, code, safety/moderation data and more, providing insight into the knowledge domains Meta prioritized.

**AI Capabilities and Benchmarks**

- **Microsoft open-sources Florence-2 vision models**: Microsoft [released](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) its Florence-2 vision foundation models under an open-source license, with the models demonstrating [strong performance](https://i.redd.it/31t6f0q9ti7d1.png) across tasks like visual question answering, object detection, and image captioning.
- **LI-DiT-10B claims to outperform DALLE-3 and Stable Diffusion 3**: A [comparison image](https://i.redd.it/4dz9eg6heh7d1.png) suggests the LI-DiT-10B model surpasses DALLE-3 and Stable Diffusion 3 in image-text alignment and generation quality, with a public API planned after further optimization.
- **70B parameter Llama-based story writing model released**: DreamGen Opus v1.4, a 70B parameter language model based on Llama 3 and focused on story generation, was [released](https://www.reddit.com/r/LocalLLaMA/comments/1djo3of/llama_3_70b_roleplay_story_writing_model_dreamgen/) along with a detailed usage guide and example prompts showcasing its creative writing capabilities.

**Discussions and Opinions** 

- **Concerns over Stability AI's business prospects**: An [opinion piece](https://i.redd.it/6lns2brtsm7d1.png) raised questions about the sustainability of Stability AI's business model and future outlook in light of issues with the Stable Diffusion 3 release and other factors.

**Memes and Humor**

- AI memes touched on the [rapid growth](https://i.redd.it/7tc9ugtk2n7d1.png) of AI startups, [poked fun](https://i.redd.it/m0pt9dvrqj7d1.jpeg) at OpenAI's closed model despite its name, and satirized Stability AI's [handling](https://i.redd.it/l2d0f7wxfi7d1.png) of the Stable Diffusion 3 problems.
- One meme [imagined](https://i.redd.it/o3rirwk52i7d1.jpeg) Doc Brown's stunned reaction to AI progress by 2045, nodding to the rapid pace of advancement.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Model Performance Optimization and Benchmarking**

- **[Quantization]** techniques like **AQLM** and **QuaRot** aim to run large language models (**LLMs**) on individual GPUs while maintaining performance. Example: [AQLM project](https://github.com/Vahe1994/AQLM) with **Llama-3-70b** running on RTX3090.

- Efforts to **boost transformer efficiency** through methods like **Dynamic Memory Compression (DMC)**, potentially improving throughput by up to 370% on **H100 GPUs**. Example: [DMC paper](https://arxiv.org/abs/2403.09636) by @p_nawrot.  

- Discussions on **optimizing CUDA operations** like fusing element-wise operations, using **Thrust library's `transform`** for near-bandwidth-saturating performance. Example: [Thrust documentation](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each).

- Comparisons of **model performance** across benchmarks like **AlignBench** and **MT-Bench**, with **DeepSeek-V2** surpassing GPT-4 in some areas. Example: [DeepSeek-V2 announcement](https://x.com/deepseek_ai/status/1787478986731429933).

**2. Fine-tuning Challenges and Prompt Engineering Strategies**  

- Difficulties in **retaining fine-tuned data** when converting **Llama3** models to GGUF format, with a [confirmed bug](https://github.com/ggerganov/llama.cpp/issues/7062) discussed.

- Importance of **prompt design** and usage of correct templates, including end-of-text tokens, for influencing model performance during fine-tuning and evaluation. Example: [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47).

- Strategies for **prompt engineering** like splitting complex tasks into multiple prompts, investigating **logit bias** for more control. Example: [OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).

- Teaching LLMs to use `<RET>` token for **information retrieval** when uncertain, improving performance on infrequent queries. Example: [ArXiv paper](https://arxiv.org/abs/2404.19705).

**3. Open-Source AI Developments and Collaborations**

- Launch of **StoryDiffusion**, an open-source alternative to Sora with MIT license, though weights not released yet. Example: [GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file).

- Release of **OpenDevin**, an open-source autonomous AI engineer based on Devin by Cognition, with [webinar](https://lu.ma/fp0xr460) and growing interest on GitHub.  

- Calls for collaboration on open-source **machine learning paper** predicting IPO success, hosted at [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html).

- Community efforts around **LlamaIndex** integration, with issues faced in Supabase Vectorstore and package imports after updates. Example: [llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack).

**4. Multimodal AI and Generative Modeling Innovations**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** focuses on elevated chat interactions, while **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** refines coding abilities.  

- The **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** model brings powerful AI chatbots to browsers via WebGPU.

- Combining **Pixart Sigma + SDXL + PAG** aims to achieve **DALLE-3**-level outputs, with potential for further refinement through fine-tuning.

- The open-source **[IC-Light](https://github.com/lllyasviel/IC-Light)** project focuses on improving image relighting techniques.

**5. Misc**

- **Stable Artisan Brings AI Media Creation to Discord**: Stability AI launched **Stable Artisan**, a Discord bot integrating models like **Stable Diffusion 3**, **Stable Video Diffusion**, and **Stable Image Core** for [media generation and editing directly within Discord](https://bit.ly/4aiVy6C). The bot sparked discussions about **SD3's open-source status** and the introduction of **Artisan as a paid API service**.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Ollama Gets Unslothed**: Engineers are keen on the [new support for Ollama](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) by Unsloth AI, providing a Colab link for tests and requesting bug reports for early adopters.

**Distillation of Distributed Training**: Deep dives into **distributed data parallelism (DDP)** focused on scaling models across multiple GPUs, highlighting the importance of model accuracy, token, and context handling in training.

**Anthropic Innovates with Claude 3.5 Sonnet**: Anthropic's announcement of [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) has captured engineers' attention for setting new industry model standards.

**CausalLM Confusion Cleared**: A slew of messages addressed confusion around **causalLM loss calculation** during training, comparing it to loss calculation in traditional Masked LM tasks, indicating its aggregate nature for next word prediction accuracy.

**Deployment Blues and Pretraining Queries**: AI engineers discussed practical challenges and solutions in deploying models, such as resolving **llama3 library version compatibility using Conda**, and strategies for **continued pretraining and fine-tuning instruct models**, with a helpful discussion found [here](https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Sparks Engineering Curiosity**: Engineers debated **GPT-4o's** reasoning capabilities, noting its advancement over other models and anticipating its implementation in larger models like the hypothetical GPT-5. Concerns centered around AI's theoretical limits and practical applications, with a specific focus on **OpenAI's** offerings vis-à-vis competitors like **Claude 3.5** and **Google’s Gemini**.

- **Pushing the Boundaries of ASI**: Discussions on Artificial Superintelligence (ASI) raised questions about achieving *"God-like intelligence"* and its ethical implications. Debates oscillated between the concerns over limitations of ASI and the enthusiasm for its unprecedented technological progression.

- **Practical Prompt Engineering Woes**: Engineers shared frustrations over **token usage** in OpenAI assistants, with unexpected high token counts for simple commands. On the creative side, limitations of **DALL-E** in generating asymmetrical images prompted suggestions for more diverse descriptive phrases but acknowledged limited success.

- **Voice of the Engineers: Call for Updates and Alternatives**: Users expressed dissatisfaction with stalled updates from OpenAI, such as a voice release from **Sam Altman**, and discussed chat experiences with **Google’s AI Studio**, noting **Gemini’s** superior performance in handling large context windows.

- **AI's Practical Limitations in Long Outputs and System Instructions**: **ChatGPT** was highlighted for difficulties in generating reliable long outputs due to its token limitations. Furthermore, reports on **GPT-3.5-turbo-0125** sometimes overlooking system instructions led to advice for clearer and simplified directives to ensure compliance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI CEO Under the Spotlight**: Shan Shan Wong has been confirmed as the CEO of Stability AI. Some members teased about possibly sharing exclusive updates in the future without providing specifics.

- **Licensing Woes for AI Artisans**: AI-generated images by **stabilityai/stable-diffusion-xl-base-1.0** model raised questions on licensing, with members exploring the use of various Creative Commons licenses. The model in question operates under the CreativeML Open RAIL++-M License.

- **Art Community Channels Axed**: Deletion of the Cascade and other art-related community channels due to inactivity and bot spamming led to a stir among members. A moderator noted that these channels could be restored should the community express a renewed interest.

- **Turbo Versus Finetuned Model Showdown**: Turbo models were valued for their speed and flexibility among some members, while others advocated for the use of finetuned models, like Juggernaut and Pony, for tasks needing specific detail or conceptual accuracy.

- **Introducing Mobius, The Anti-Bias Model**: The Mobius model was highlighted as a leader in debiased diffusion models, utilizing a domain-agnostic approach to reduce bias. Questions were raised about its size and requirements, such as clip skip 3 and its Lora compatibility was discussed. 

Links: [Hatsune Miku Gif](https://tenor.com/view/hatsune-miku-miku-hatsune-earthquake-plush-miku-death-gif-4018907532159793300), [Mobius on Civitai](https://civitai.com/models/490622/mobius), [ComfyUI_TensorRT GitHub](https://github.com/comfyanonymous/ComfyUI_TensorRT), [Google Colab notebook](https://colab.research.google.com/github/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb#scrollTo=9AZDrh-SUDt2).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's CEO Chats with Lex Fridman**: In a riveting podcast session, Perplexity's CEO discussed the powerful impact of AI on search and the internet, invoking inspiration from Larry Page with the mantra, "the user is never wrong." The video is available on [YouTube](https://youtu.be/e-gwvmhyU7A).

- **Technical Troubles and Triumphs**: Users have encountered issues with Pro Search's inability to find sources when toggled on, an inconsistency compared to the iPhone app's performance, prompting a community escalation. Meanwhile, there's anticipation for the upgrade to Claude 3.5 Sonnet, notably for its potential in creative writing, although HOW it integrates remains a point of curiosity.

- **AI Ethics in the Spotlight**: A Wired article sparked debate on Perplexity's adherence to robots.txt with some users defending AI's role in information retrieval for user requests, while others urge closer scrutiny. 

- **Prospects and Psychedelics**: Conversations took a turn from high-paying career paths for English literature majors to the financial speculations around Lululemon earnings, juxtaposed starkly with discussions on how psychedelic experiences can pivot personal belief systems.

- **API Adaptability Aches**: The Perplexity API showcases solid performance, particularly notable for running large LLMs, but is critiqued for its constrained customization and lack of certain features like Pages via API. However, resetting API keys is simplified via the [Perplexity API settings page](https://www.perplexity.ai/settings/api).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Character.AI Pushes Efficient INT8 Training**: Character.AI is working towards AGI with **INT8 optimization**, achieving more inference queries at a **rate about 20% of Google Search's volume**. Inquiry into their use of **Adaptive Quantization** (AQT) remains open. [Read more](https://research.character.ai/optimizing-inference/).

**Kernel Profiling and Triton Tackles**: **Nsight Compute** is the go-to for profiling CUDA kernels to squash performance bugs in the codebase, while **Triton 3.0.0** is hailed as a fix for numerous issues, with detailed upgrade instructions available. [GitHub profiling script](https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh) and [YouTube resource for kernel profiling](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR).

**Emerging AI Breakthroughs**: Advancements in **Qwen2**, **DiscoPOP**, and **Mixture of Agents** are shaping the future of AI with the potential to boost **LLM performance**. Unfolding research projects like Open Empathic and Advisory Board GPT offer creative angles on model utilization. [AI Unplugged Coverage](https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture).

**Optimizing with Quantization & Introducing FPx**: While Finetuning the details, the community evaluates **tinygemm** compatibility, embraces challenges with **FP8 quantization**, and ponders **XLA integration with quantized models**. The **uint2 quantization** and performance comparisons against **FP16** showcase promising speedups. [Quantization code reference](https://github.com/pytorch/ao/blob/e6460c22284df669b95dc912114cc29b40df2030/torchao/quantization/quant_primitives.py#L280-L289).

**Leveraging Newer Tech for Voltage Speed**: **H100 box** experimentation with a 1558M model demonstrates a 2.5x speedup over A100, providing tangible efficiency gains from the cutting edge of hardware advancements. Speed optimizations continue to be a focal point, with a 20% enhancement through **torch compile max autotune** mentioned.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 2 Theta Surpasses GPT-4**: **Hermes 2 Theta 70B** has scored 9.04 on the MT-Bench, a leap ahead of GPT-4-0314's 8.94 score, flaunting increased creativity and capabilities. It's a product of the collaboration between Nous Research, Charles Goddard, and Arcee AI, and download links for both FP16 and GGUF versions are available on Hugging Face.

- **General Chat Buzzes with Claude 3.5 Sonnet**: The community resonated with the release of **Claude 3.5 Sonnet** for its speed and problem-solving abilities, branding it a step forward in AI capabilities. Meanwhile, the debate on model parsing emphasized the importance of converting model-specific tool calls into a standard format, suggesting the incorporation of reverse templates into `tokenizer_config.json`.

- **Teasing New Resources**: Members hinted at an upcoming resource in the #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1253102983932154010) channel, sparking intrigue and anticipation among peers.

- **Model Integration Techniques Under Scrutiny**: A suggestion in the #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253065913649856542) channel described a direct method of merging tools into model prompts, potentially facilitating fluent use of multiple AI tools.

- **Music Video Diversifies Conversation**: In a lighter exchange, a [YouTube music video](https://youtu.be/E3Yt_qLUGJY) was shared by a member on the #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1253394167527247964) channel, offering a diversion from the technical discussions.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Direct Data Streaming On the Horizon**: Users highlighted current limitations with **Torchtune**, as in-memory datasets are still downloaded to local disk from **Hugging Face (HF) locations**. They are moving towards *streaming datasets* to bypass saving on disk.

- **Configuring HF Datasets: A Piece of Cake**: The community agreed on configuring HF datasets in `torchtune.dataset.chat_dataset` using `conversation_style: openai`, which should integrate effortlessly with Torchtune.

- **Sequence Length Debate Settles at 8k**: There was a debate on **llama3** maximum sequence length, resulting in a consensus of up to 8192 characters, though concerns were raised about **VRAM capacity limitations**.

- **Crash Course in Memory Management**: Amidst RAM-related crashes during model training, particularly with qlora and lora, it was suggested to offload layers to **CPU** and sort out **ROCm** setup quirks for smooth operation.

- **Navigating the ROCm Maze**: Discussions on setting up **ROCm for AMD GPUs** unearthed several issues, but community-shared resources, including a Reddit thread about successful ROCm operation on a **6900 XT**, proved to be valuable. Building from source was the recommended route for simplicity and effectiveness.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI integrations prove handy in scripting**: Users discussed integrating **Stable Diffusion** within VSCode and were advised to run commands via the terminal within the editor. There was also a mention of using a **stable-diffusion-3-medium-diffusers** model as a workaround for a missing model index in Stable Diffusion 3.

**LLMs debate over drug names and finetuning issues**: NLP models showed a preference for generic drug names (acetaminophen) over brands (Tylenol), suggesting possible data contamination as discussed in [this study](http://arxiv.org/abs/2406.12066) and demonstrated on a [leaderboard](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard). Meanwhile, a member encountered problems while fine-tuning **Llama 3** using TRL with QLoRa and linked to their code and potential solutions.

**Challenging assumptions with multi-table data synthesis**: A member scrutinized the challenge of generating synthetic multi-table databases, particularly those containing date columns, and an [article](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/) compared three data synthesis vendors. Additionally, **ToolkenGPT** was proposed in a [paper](https://doi.org/10.48550/arXiv.2305.11554) as a method for LLMs to use external tools via tokenization, aiming to bypass restrictions of fine-tuning and in-context learning.

**Protein predictions get a parallel processing power-up**: Users celebrated an update to **BulkProteinviz**, an open-source protein structure prediction tool that now enables simultaneous multiple predictions. This could significantly accelerate research in computational biology.

**LLama 3:70B seeks a size upgrade**: One engineer asked for tips to grow their training data for **Llama 3:70B** managed through **Ollama**, attempting to increase from 40GB to 200GB for more robust local training.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**MLIR's Kgen Dialect Causes Consternation**: Community members are baffled by the **`kgen` dialect** in MLIR as it lacks public documentation, with one user describing the code as *messy*. Suggested workarounds for implementing **256-bit integers** in MLIR include using **`SIMD[DType.int64, 4]`** or **defining an `i256` type**, supported by a [GitHub reference](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232).

**Mojo Rides the Open Source Wave**: Members are informed that Mojo language is partially open source with its compiler to be progressively open-sourced, as detailed in a [blog post](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source). Discussions revealed current practical limitations in Mojo for production environments and advice was shared against using Mojo in complex automation work until it matures.

**Evolving Mojo's Ecosystem with Package Managers and Livestreams**: The development of a package manager for Mojo is underway with community suggestions like [Hammad-hab's `pkm`](https://github.com/Hammad-hab/pkm). Additionally, the community was invited to a **Modular Community Livestream** to discuss MAX Engine and Mojo developments, available on [YouTube](https://www.youtube.com/watch?v=uookgZ7Ojg8).

**Blueprints for Burning Questions in Modular's 'engine' Room**: A detailed clarification about the `execute` function in the MAX Engine was provided, specifying that it can take a variadic `NamedTensor` or `Tuple[StringLiteral, EngineNumpyView]`, as stated in the [Model documentation](https://docs.modular.com/max/api/mojo/engine/model/Model#execute).

**Nightly, Handle Mojo with Care**: The release of the latest Mojo compiler version `2024.6.2005` was announced, and users can view the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) for details. Additionally, a new tool titled "mojo_dev_helper" for standard library contributors was introduced, with more details available on [GitHub](https://github.com/rd4com/mojo_dev_helper).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Spam Storm Strikes Discord**: Multiple channels within the Discord guild were plagued by spam bots promoting "18+ Free Content" including OnlyFans leaks, with a link to an illicit Discord server. The shared invite URL across all instances was [Join the Discord Server!](https://discord.gg/2AFWP2Qd2r).

- **Community Acts Against Spam**: Following the flood of inappropriate content, actions were taken by members to report and block the origins of the spam. There is confirmation that steps were taken against a reported user, indicating vigilance within the community.

- **Nitro Boost Giveaway Scam Warning**: In addition to adult content spam, there was mention of an alleged Nitro Boost giveaway, likely a part of phishing attempts or scams associated with the same spammed Discord link.

- **Repeated Targeted Channels**: The spam was not isolated but instead appeared across various channels, from #[committers](https://discord.com/channels/1122748573000409160/1122748682475950142/1253138286860304544) to #[ai-explained-cartoons](https://discord.com/channels/1122748573000409160/1249527870750195802/1253138355726843926), indicating a widespread issue.

- **Members' Concern and Prompt Response**: Amidst the spam, there was an expressed concern from members for the need to take swift actions, and there were affirmative responses indicating that the community is responsive and proactive in handling such disruptions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**New Horizons for LM Studio 0.2.23**: LM Studio's version 0.2.23 is hailed for its speed boost, greatly improving efficiency. Users report headaches with Deepseek Coder v2 due to "unsupported architecture" errors, but note that disabling flash attention and employing version 0.2.25's deepseek coder preset can mitigate the problem.

**Hardware Conundrums and GPU Debates**: Discussions revolve around the heavy VRAM demands of large language models (LLMs), suggesting 38GB+ of VRAM for seamless performance on 34GB models and debating the merits of Nvidia's 3090 vs 4090 in cost-effectiveness and VRAM capacity. AMD 7900XT's suitability for LLMs is questioned amid issues with ROCm support and general detection hitches on some systems.

**Seeking Frontend Flexibility**: Engineers are exploring frontend options for local LLM server deployment on various devices, with [every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui) and [awesome-chatgpt](https://github.com/uhub/awesome-chatgpt) repositories being common starting points. Some express frustrations over automated moderation in llama-related subreddits, which seem overly aggressive.

**Technical Quirks in Model Discussions**: Nvidia's new storytelling model garners interest for its balance in reinforcement content. The extent of Opus's context capacity sparks debates, with hopes pinned on extended limits. DeepSeek Coder V2 Lite has a peculiar inclination towards Chinese responses unless an older template is used. A preference emerges for a new model over Midnight Miqu's offerings following some hands-on tests.

**Bottlenecks in Beta and Tech Previews**: Latest beta testing of LM Studio reveals detection problems with Nvidia's 4070 GPU on Linux Mint and hiccups with DeepseekV2 models. M1 Mac users face inconsistencies when leveraging GPU acceleration, and AMD users are directed towards installing ROCm packages to ensure GPU compatibility.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **A Quicker, Cheaper, Better Claude**: The new [Claude 3.5 Sonnet](https://openrouter.ai/models/anthropic/claude-3.5-sonnet) from Anthropic has been launched, touting better performance than its predecessor Opus, while being 5x cheaper and 2.5x faster; it offers self-moderated versions alongside standard ones, with prices detailed in a [tweet](https://x.com/OpenRouterAI/status/1803802819708739717).
- **Stripe's Glitch in the Credits**: Stripe payment issues that caused credits to queue incorrectly have been resolved, with affected transactions from the last half-hour processed successfully.
- **Nemotron's Hosting Challenges**: Nemotron is not favored for hosting among providers, primarily due to its large size at 340 billion parameters and lack of compatibility with popular inference engines.
- **Dolphin Mixtral's Open Licensing Advantage**: Praise was shared for Dolphin Mixtral 1x22b model, which is available on [HuggingFace](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b) and recognized for its potential to replace Codestral while avoiding licensing restrictions.
- **Clarifying DeepSeek-Coder V2's Limits**: Confusion over the context length for DeepSeek-Coder V2 was addressed; despite its model card claiming 128K, further clarification revealed a 32K cap due to the OpenRouter hosting limitations.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **1B Internet Argument Solver? Cost vs. Practicality**: There's lively debate on the feasibility of training a 1B model specifically to resolve internet arguments, with concerns about high costs versus the model's training time, which can be under two days on an H100 node.

- **Tech Woes: Selectolax, Lexbor, and NumPy Miseries**: Engineers face technical issues with **Selectolax** and **Lexbor** causing segmentation faults, and struggle with **NumPy 2.0** compatibility in the `lm-eval-overview.ipynb`, even after downgrading.

- **Warc and the Speed Demons**: Discussion on **CC Warc file processing** has members sharing various optimizations, with reports of one Warc taking 60 seconds to process using 100 processes, and another approach leveraging parallel processing across 32 processes.

- **Data Hub Bonanza**: **Epoch AI's Data Hub** now catalogs over 800 models, aiming to benefit researchers, policymakers, and stakeholders and pointing to potential computational explosion in frontier AI by the 2030s, as discussed in a CNAS report.

- **Research Riches: From Token Datasets to Slot SSMs**: Discussion in the research channel spans diverse topics including the performance effects of the 4T token dataset from [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), the introduction of **SlotSSMs** for better sequence modeling in a [paper](https://arxiv.org/abs/2406.12272), models struggling with drug brand names in medical applications, post-training enhancement techniques like **LAyer-SElective Rank reduction (LASER)**, and domain conditional PMI to tackle surface form competition in LLMs.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude 3.5 Sonnet Takes the Lead**: [Anthropic](https://x.com/anthropicai/status/1803790676988920098?s=46) introduced **Claude 3.5 Sonnet**, boasting faster speeds and improved cost-efficiency, along with a promise of future models named Haiku and Opus. Meanwhile, [Character.AI](https://research.character.ai/optimizing-inference/) focuses on optimizing inference for their AGI, capable of handling 20,000 queries per second—comparatively 20% of Google Search's volume.

- **Youth Driven AI Engagement**: Character.AI is experiencing notable session times, particularly among younger users, which surpass the engagement seen with ChatGPT. Additionally, **Claude 3.5 Sonnet** tops aider’s code editing leaderboard, especially excelling in "whole" and "diff" editing formats.

- **Sour Grapes in AI Safety?**: Members expressed skepticism about the trust and implementation of AI safety, with ironic "Trust me bro" sentiments and references to [Eliezer Yudkowsky's challenge](https://x.com/ESYudkowsky/status/1803676608320192617) to AI alignment plans. Scott Aaronson's recount of **Ilya Sutskever's** quest for a theoretically robust alignment stance also surfaced.

- **Kling Outshines Sora**: [Kuaishou](https://kling.kuaishou.com/en) has launched **Kling**, a text-to-video generative AI model available to the public, which raises the bar with two-minute videos at 1080p and 30fps, unlike OpenAI’s **Sora**. Furthermore, there's curiosity about Meta's use of 5000 V100s for generating synthetic data, a topic being revisited by Nathan Lambert.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **CrewAI Teams Up with LlamaIndex**: CrewAI announced an enhancement to multi-agent systems by integrating with LlamaIndex, providing a way to define a "crew" of agents that leverage LlamaIndex capabilities for tasks. Details of this integration can be found in [their latest blog post](https://t.co/8Tjk888RL1).

- **AI Fair's Future Speaker**: The founder of LlamaIndex is scheduled to present at the AI Engineer's World's Fair, discussing the *Future of Knowledge Assistants* on June 26th with some major announcements, and another session on June 27th. For more information, enthusiasts can [learn more here](https://t.co/JMoAOAA4bI).

- **Vector Store Customization Queries**: Engineers are exploring the flexibility of LlamaIndex's VectorStoreIndex with questions about adding sequential identifiers, custom similarity scores, and asynchronous node retrieval, though some features might require custom implementation due to current limitations.

- **Knowledge Generation from Documents**: Discussion around generating questions from PDFs using LlamaIndex's `DatasetGenerator` was shared, including an example utilizing OpenAI's model for the task.

- **Persisting Indexes Made Easy**: A focus on storing indexes persisted with a conversational highlight on using `storage_context.persist()` to store a DocumentSummaryIndex in LlamaIndex, accompanied by practical code illustrations.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Speed Boost in Nemotrons API**: Members reported **Nemotrons API** improvements, highlighting significant speed increases and a newly released **reward model**.

- **Turbcat or Turbca?**: Clarification was made on the **Turbcat** debate; it's the model, with **Turbca** being the individual behind it. Issues with dataset configuration and tokenization methods prompted discussion and concern.

- **Tokenization Troubles and Solutions**: A robust debate emerged regarding tokenization and how to handle **end of text (EOT)** tokens, with a member presenting the [Multipack with Flash Attention documentation](https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html) to showcase the best practices.

- **Qwen's Biases Unraveled**: The community expressed concern over the **Qwen model's** biases and the need for adjustments, pointing to [Chinese LLM censorship analysis](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis) for insights into the model's potential propagandistic inclinations.

- **Layer-Pruning and QLoRA Hit the Spot**: The intersection of **layer-pruning** and QLoRA was brought up, with a member citing its successful application in improving model performance (MMLU scores by up to 10 points) and [a Hugging Face model card](https://huggingface.co/chargoddard/llama3-42b-v0) for practical application details.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Single Quotes Save the System**: A user discovered that substituting **backticks with single quotes** fixes a data injection issue in a **SystemMessage**.
  
- **Chunk and Conquer Large Text**: Strategies for handling large text data from web scraping were discussed, including token limits and how to effectively combine chunked responses, with links to [LangChain documentation](https://github.com/langchain-ai/langchain/issues/17783).

- **PDF Puzzles Vector Databases**: Retrieving data from **vector databases** using **PDF documents** has proved challenging for a user, who encountered non-informative "I don't know" answers from the system.

- **Manage Event Streaming Like a Pro**: Techniques for event filtering in **astream_event** were shared, with pointers to specific sections in the [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events) guiding users on the process.

- **Launching Foodie AI Assistants and Chatbots**: [TVFoodMaps](https://www.tvfoodmaps.com/foodtv-ai-chat) introduced an AI-powered feature to help users find restaurants featured on TV, requiring a premium membership, while a [guide to create SQL agents](https://git.new/SQLAgent) using **OpenAI & LangChain** was shared, inviting feedback. A new concept named **Conversational Time Machine** was introduced in an [article on Medium](https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587), exploring the development and uses of a **LangGraph Support Chatbot**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Bounty Hunters for Approximation**: In the pursuit of a bounty to implement Taylor approximations for LOG2, EXP2, and SIN in `function.py`, issues about adding bitwise operations to `ops.py` arose, with community concern about operation count inflation. Practicality trumps purity as the need for new operations competes with the aim for minimalism.

**Multi-GPU Quest Continues**: Clarifications around multi-GPU support with NVLink led to learning that GPUs connect via PCI-E, and a [GitHub resource](https://github.com/tinygrad/open-gpu-kernel-modules) was shared, evidencing NVIDIA's Linux open GPU kernel modules with P2P support.

**High Bar for Diffusion Models**: A community member's port of a diffusion model from PyTorch to tinygrad sparked a debate on code quality, with George Hotz setting the bar high for inclusion into the project. Contributors are encouraged to submit a PR for scrutiny.

**Clip, Clip, Hooray? Or Mayday?**: An intense technical dissection took place regarding `clip_grad_norm_` implementation in TinyGrad, where Metal's limitations forced a discussion on tensor chunking as a workaround. This signifies the ongoing struggles with optimization within hardware confines.

**Tying Weights, Loosing Bugs**: A suspected bug involving weight tying in TinyGrad was spotlighted, revealing that two ostensibly linked tensors were being optimized independently. The community is on the case, suggesting library corrections for consistent weight optimization.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Persistence of Discord Community Debated**: Members discussed the continued activity of the Discord server post-course, suspecting it would depend on member and moderator engagement, without concrete plans outlined.
  
- **Expert LLM Livestream Incoming**: A livestream with Eugene Yan from **Amazon** and Bryan Bischof from **Hex** discussing real-world **LLM applications** was announced, promising insights geared toward prompt engineering, evaluation, and workflow optimization. Interested members can register [here](https://lu.ma/e8huz3s6) and explore their learnings detailed in an [O'Reilly report](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/).

- **Finetuning Insights and Requests**: Regarding custom **LLM workloads**, discussions included needing fine-tuning for specific roles such as fraud detection, while general tasks like language translation may not. In a different vein, there was a buzz around **Jarvis Lab's upcoming Docker feature** and **Modal's user experience enhancement** for finetuning.

- **Credits and Access Issues Centre Stage**: Multiple members sought assistance regarding credits and account access across various platforms like **LangSmith** and **OpenAI**, often providing IDs or emails in a plea for help, indicating a level of confusion or technical problems.

- **Technical Glitches and Triumphs**: Amidst praise for a well-designed eval framework, users reported various technical issues from CORS errors at **Predibase** to credit visibility on **OpenAI**, showing a mix of user experience in the practical aspects of applying LLMs to projects.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Riches Beyond Just Wealth in AI Discussion**: Members joked about whether OpenInterpreter (OI) could make someone financially richer, leading to playful banter about achieving 100% richness instead of just 5%. In another thread, discussions around **Claude 3.5 Sonnet** revealed a preference for its dialogue style over **GPT-4**.

- **AI Models Face Off for Top Honors**: Debates surfaced concerning the best uncensored models, with "2.8 dolphin" and "mistral 3/31/24" mentioned as contenders. Opinions diverged, indicating varying user experiences with each model and no definitive best model emerged.

- **Memory Lane with Open Interpreter**: Queries regarding potential long-term memory capabilities in OpenInterpreter prompted discussion but yielded no conclusive solutions. Members are actively looking into how to equip OI with persistent memory.

- **OpenInterpreter's Tentative Manufacturing Milestone**: An update in #O1 indicated the expected shipping of the first 1,000 OpenInterpreter units between October 31st and November 30th, as per an announcement from Ben. Curiosity arose about order statuses and positioning within the first shipment batch.

- **Practical AI Magic with Local, Task-Oriented Controllers**: A demonstration showed a *fully local, computer-controlling AI* successfully connecting to WiFi by reading a password from a sticky note, illustrating the effectiveness of AI in executing everyday tasks. The example noted reflects AI's potential to simplify daily interactions with technology.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Graph-Based Captions Make a Leap**: The **GBC10M dataset**, a graph-based recaptioned version of **CC12M**, is now available on [Hugging Face](https://huggingface.co/datasets/graph-based-captions/GBC10M). Efforts are underway to secure a less restrictive license and transition the dataset to the **Apple organization** on Hugging Face, with plans to publish the accompanying paper on **arXiv** and release the code once it's refined.

- **Adversarial Robustness Debates Heat Up**: A scuffle erupts in academic circles as experts like Carlini and Papernot challenge the Glaze authors on adversarial robustness issues, particularly regarding a withheld codebase for perturbation budgets.

- **VAEs Channel Increase Sparks Debate**: Raising the channel count in VAE latent spaces from 4 to 16 sparked a technical debate, juxtaposing the complexity in latent space against computational costs, and noting the quadratic scaling of global attention with pixel count.

- **The Mystery of Overfitting Solved by Claude-3.5?:** An engineer's manual experiment suggests that **Claude-3.5-Sonnet** shows a promising ability to reason through problems without overfitting on recognizable patterns, unlike other models.

- **Chameleon Model Training Hits a Wall**: Engineers face an unexpected challenge with the Chameleon model as extreme gradient norms cause NaN values, with no remedy from standard fixes like reducing learning rates or switching to higher precision.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Multi-Language Chatbots with Cohere**: AI enthusiasts are employing the Cohere API for developing chatbots in various languages, and a discussion highlighted its compatibility with **OpenAI's API**, allowing integration into any environment through RESTful APIs or sockets.

- **Purple Praise**: Cohere's interface, notably its use of the color purple, received commendations for its stylish design in the community, sparking inspiration for future design endeavors among members.

- **Problem-Solving in Project Development**: A community member shared their experiences dealing with chat hang-ups potentially linked to API issues, with a commitment to addressing the issue through UI adjustments and ongoing troubleshooting.

- **Community Camaraderie**: Excitement was evident among participants who welcomed new members and shared their positive impressions of Cohere's unique and intelligent approach.

- **Platform Adaptability Discussions**: Dialogues emerged around utilizing Cohere's capabilities on different platforms, with a specific mention of creating chatbots in .NET on a Mac.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Toucan TTS Breaks Language Barriers**: The open-source [Toucan TTS](https://x.com/reach_vb/status/1803529768861610073?s=46) model is distinguished by its capability to support TTS in 7000 languages, featuring a text frontend for language-agnostic articulatory features and leveraging meta-learning for languages lacking data.
  
- **Claude 3.5 Sonnet Takes Efficiency to New Heights**: The new [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790676988920098?s=46) impresses the community by outperforming competitors, providing higher speeds and reduced costs. Members also celebrate the launch of the Artifacts feature, a Code Interpreter successor, enabling real-time doc, code, and diagram generation.

- **Consultancy Collaboration Creates AI Synergy**: Market buzz as Jason Liu's Parlance Labs merges with Hamel Husain's and Jeremy Lewi's teams, uniting to enhance AI product support and development, focusing on infrastructure, fine-tuning, and evaluations as noted in their [announcement](https://x.com/jxnlco/status/1803813743714844863?s=46).

- **Groq Steps Up with Whisper Support, but Concerns Linger**: Groq's new Whisper model support, which achieves speeds at 166x real-time, opens doors for faster AI processing; yet, the community raises questions about its current [rate limits](https://x.com/sjwhitmore/status/1803811998548812140?s=46) and the model's broader applicability.
   



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Aims for Model Diversity**: In discussions, it was proposed to harness **YOLOv10 PyTorch** and **OCR Safe Tensors** within a Llamafile structure. A solution offered entails converting these models to **gguf** format via llama.cpp Python scripts.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Infer Conference Ignites AI/ML Discussions**: *Hudson Buzby* and *Russ Wilcox* will spearhead conversations on **real-life recommender systems** and AI/ML challenges at [Infer: Summer '24](https://tinyurl.com/4dfvcte7), with a focus on optimizing AI pipelines and content accuracy, featuring experts from companies like Lightricks.

- **Network and Learn at RecSys Learners Virtual Meetup**: [RecSys Learners Virtual Meetup](https://lu.ma/7pvpp1cm), hosted by *Rohan Singh S Rajput* on 06/29/2024, provides a platform for professionals of all levels to connect and enhance their knowledge in recommendation systems.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Florence 2 Takes Handwriting OCR Up a Notch**: [Florence 2 by Microsoft](https://x.com/dylfreed/status/1803502158672761113) has received praise for its superior performance in handwriting recognition and OCR, especially useful for journalism. Microsoft's model stands out in processing public records.

**Test Drive Florence 2 on Hugging Face**: The Florence 2 model is available for hands-on experimentation at [Florence-2 on Hugging Face](https://huggingface.co/spaces/gokaygokay/Florence-2), showcasing its range of capabilities in vision-related tasks, which are crucial for AI development and research.

**Inside Florence 2’s Visual Prowess**: The model uses a prompt-based methodology for various vision and vision-language tasks, trained on the massive FLD-5B dataset containing 5.4 billion annotations, demonstrating mastery in multi-task learning and adaptability in both zero-shot and fine-tuned scenarios.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **"Don't Mention AI or Get Piledrived":** An entertaining blog post, *"I Will Fucking Piledrive You If You Mention AI Again"*, mocks the AI hype cycle, cautioning against the overzealous and impractical adoption of AI technology with a warning that it's a *"cookbook for someone looking to prepare a twelve course fucking catastrophe."* Engineers interested in cultural critiques of the industry might find it a different but relevant read, available [here](https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1253065876685586486)** (437 messages🔥🔥🔥): 

- **Test Ollama support with Unsloth**: Members are excited about the new support for Ollama in Unsloth. A [Colab link](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) is shared for early testing, with requests for bug reports.
- **Distilled Knowledge on Model Parameters and Training Techniques**: Discussions dive deep into distributed data parallelism (DDP) and the potential of scaling models across GPUs. There's a focus on model accuracy in various contexts, particularly around token and context handling during and after training.
- **Claude 3.5 Sonnet Released**: Timotheeee1 announced the release of Claude 3.5 Sonnet, a new model by Anthropic raising the industry standards. Full details and access guidelines are outlined in [Anthropic's news post](https://www.anthropic.com/news/claude-3-5-sonnet).
- **Training Tips and Troubleshooting with Unsloth**: Members sought advice on merging models and handling special tokens during fine-tuning and deployment, with valuable insights shared on end token signaling and accuracy assessment. Discussions included techniques using platforms like Hugging Face.
- **Upcoming Events and Recorded Talks**: Theyruinedelise hinted at an engaging scheduled talk with Daniel and Sebastien focused on fine-tuning and deploying models on Ollama. Details were shared via [Discord's invite link](https://discord.com/invite/EwGjYYBu?event=1251334371349233814).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1803446594228068474">Tweet from Daniel Han (@danielhanchen)</a>: I&#39;m heading over to San Fransciso for a few months! I&#39;ll be at @aiDotEngineer World&#39;s Fair for a 3 hour workshop and a talk!  The workshop will be super technical & fun! Topics: 1) Backpro...</li><li><a href="https://www.anthropic.com/news/claude-3-5-sonnet">Introducing Claude 3.5 Sonnet</a>: Introducing Claude 3.5 Sonnet—our most intelligent model yet. Sonnet now outperforms competitor models and Claude 3 Opus on key evaluations, at twice the speed.</li><li><a href="https://huggingface.co/fimbulvntr/llewd-8b-64k/tree/main">fimbulvntr/llewd-8b-64k at main</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>: no description found</li><li><a href="https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture">AI Unplugged 13: Qwen2, DiscoPOP, Mixture of Agents, YOLO v10, Grokked Transformers</a>: Insights over Information</li><li><a href="https://github.com/Kryptonions/RLHF">GitHub - Kryptonions/RLHF: pipeline for training LLM efficiently</a>: pipeline for training LLM efficiently. Contribute to Kryptonions/RLHF development by creating an account on GitHub.</li><li><a href="https://x.com/UnslothAI/status/1803767513215610974">Tweet from Unsloth AI (@UnslothAI)</a>: We&#39;ll be live on @Ollama&#39;s server at 12pm ET today, to show our new support for Ollama! 🦥🦙  First learn with Sebastien about &#39;Emotions in AI&#39;, then we&#39;ll teach & give early acces...</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq1">Google Colab</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7412#issuecomment-2120427347">CUDA: quantized KV cache demo by JohannesGaessler · Pull Request #7412 · ggerganov/llama.cpp</a>: This PR adds a simple implementation of a quantized KV cache for research purposes only. The goal is not to provide an implementation that could be merged or that is suitable for regular use but in...</li><li><a href="https://huggingface.co/Salesforce/SFR-Embedding-2_R">Salesforce/SFR-Embedding-2_R · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct">Alibaba-NLP/gte-Qwen2-7B-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit/blob/main/tokenizer_config.json">tokenizer_config.json · unsloth/llama-3-8b-bnb-4bit at main</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1253145504825479168)** (7 messages): 

- **Position Queries Answered**: A user asked about the positioning of certain elements, and another confirmed, saying "*yes!*" with a red checkmark emoji.
- **Shensmobile seeks causalLM loss calculation clarification**: A member asked, "*does anyone know how loss is calculated during training for causalLMs?*" They expressed confusion over interpreting the numbers they see, comparing it to the clearer loss calculation in traditional Masked LM downstream tasks.
- **Clarification on loss aggregation in causalLM**: Continuing the discussion, a member reflects on the nature of causalLM's next word prediction. They question "*Is the 'loss' just an aggregate of all of the loss for each word that was predicted?*"
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1253106974493507634)** (62 messages🔥🔥): 

- **Deploying llama3 with compatibility issues**: A member encountered errors while deploying llama3 due to library version compatibilities and CUDA dependencies. Thefanciestpeanut suggested using Conda to simplify dependency management and avoid headaches.
- **QLora finetuning and adapter question**: Ritx8 questioned the difference between `model.save_pretrained()` and `trainer.save_model()` in SFTTtrainer workflow. They eventually found that both methods essentially achieve the same purpose, related discussion [here](https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah).
- **Converting dataset format**: Jadon1 faced issues with data prep for an LLM and resolved it by re-uploading and restructuring the dataset to match the desired format instead of renaming columns in the dataset viewer.
- **Creating multiple responses for a single input**: Hieu1852002 was advised by karatsubabutslower to use `.generate` with beam search arguments to generate multiple sequences or responses using models like "unsloth/llama-3-8b-bnb-4bit".
- **Continued pretraining on instruct models**: Gbourdin inquired about the viability of continued pretraining on instruct models. Shensmobile responded that it's possible, and while instruct formats can be followed, it might be more effective to first domain adapt a base model before fine-tuning it for specific tasks.

**Link mentioned**: <a href="https://discuss.huggingface.co/t/what-is-the-purpose-of-save-pretrained/9167/2?u=aflah">What is the purpose of save_pretrained()?</a>: Hi there! The question is a bit weird in the sense you are asking: “Why does the model have this method when the Trainer has that model?”. The base answer is: &quot; because they are two different obj...

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1253064705035997285)** (417 messages🔥🔥🔥): 

- **GPT-4o's Reasoning Scrutinized**: Members debated the reasoning capabilities of GPT-4o, noting it often surpasses other non-OpenAI models despite its limitations. One user mentioned, *"After applying all of that to a bigger model for sure,"* indicating curiosity about GPT-5's potential.
  
- **Debate on Artificial Superintelligence (ASI)**: Extensive discussions ensued about the future of ASI, with arguments both for and against its potential to achieve *"God-like intelligence"* and immortality. Concerns over theoretical and practical limits of ASI were highlighted, alongside optimism for rapid technological advancements.

- **Claude 3.5 and Gemini Discussion**: Some users voiced skepticism about OpenAI's latest offerings like GPT-4o, favoring competitors like Claude 3.5 and Google’s Gemini for their advanced capabilities in large context windows and humor integration. *"OpenAI is getting outcooked right now,"* a user remarked, emphasizing competitive advancements.

- **Debate on Human vs. AI Capabilities**: A segment of the chat explored the unique strengths and weaknesses of human intelligence compared to AI, pointing out that while AI can process more data, real-world applications often limit its efficacy. This led to theoretical debates around computation, logic, and the boundaries of what AI can achieve, summarized by, *"large portion of extremely gifted people in terms of intelligence are not engineers."*

- **Shift from GPT to Google AI Studio**: Users shared experiences and tips on using Google’s AI Studio, especially praising its Gemini model. *"Large amount of text… and it acts as an expert on this topic,"* one member shared, illustrating how Gemini’s large context windows outperform other models like GPT in specific use cases.

**Link mentioned**: <a href="https://ai.google.dev/aistudio">no title found</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1253069870862438481)** (17 messages🔥): 

- **Users demand new voice update from Sam Altman**: One user expressed frustration over delayed updates, urging Sam Altman for the new voice release. They commented on the diminishing returns and the tactic of withholding to increase excitement.

- **ChatGPT can't generate long outputs reliably**: A member advised that prompts requiring 5000-word outputs from ChatGPT are impractical due to token limitations. Another member shared their experience of getting a reduced word count when requesting the output in video script format.

- **GPT created on website uses GPT-4o**: When asked whether the GPT models created on the website are based on GPT-4 or GPT4-o, a member confirmed they use **GPT-4o**.

- **Surface laptops as MacBook alternatives**: A user inquired about good laptops for web development as alternatives to MacBooks. Respondents suggested the **newest Surface laptops** but recommended considering a MacBook or visiting a server like BuildAPC for more advice.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1253094360543924274)** (11 messages🔥): 

- **High Token Usage Frustrates Users**: One user lamented that even a simple "hello" command took 384 tokens to process due to necessary functions. Suggested solutions included shortening function names and descriptions, or excluding unnecessary tools to save on token usage.
  
- **DALL-E Struggles with Asymmetry**: A member inquired about making DALL-E generate less symmetrical images, as attempts with terms like "asymmetrical" failed. Another user advised using phrases emphasizing randomness, like "rule of thirds" or "complementary positions," yet acknowledged only limited success.

- **System Instructions Not Always Heeded**: A user posted screenshots showing GPT-3.5-turbo-0125 sometimes ignoring system instructions. Another member advised focusing on clear directives and suggested removing unnecessary containers to fix related issues.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1253094360543924274)** (11 messages🔥): 

- **Reducing Token Usage in OpenAI Assistants:** A member raised concerns about high token usage for simple inputs like "hello." Another member suggested shortening the names and descriptions of functions or excluding unnecessary tools can help save tokens.
  
- **Seeking Asymmetrical Art with DALL-E:** A user frustrated with DALL-E's symmetrical outputs asked for ways to generate more unique, asymmetrical art. Another user acknowledged this limitation and recommended using phrases emphasizing randomness or imbalance, albeit with limited success.

- **Question on OpenAI's Memory Function:** Members discussed whether a new window in the chat is completely fresh and checked on the memory function. A member clarified that this behavior changed after the introduction of the memory function.

- **System Instructions Overlooked by GPT-3.5-turbo-0125:** A member shared screenshots highlighting intermittent issues where the model ignores certain system instructions. Another suggested the problem might be due to unnecessary container instructions and advised to remove them.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1253061950594613349)** (401 messages🔥🔥): 

- **CEO of Stability AI identified**: Members discussed the CEO of Stability AI, identifying **Shan Shan Wong**. One user mentioned they wouldn’t share inside information but hinted at occasionally dropping news.

- **License inquiries on AI-generated images**: A member asked if they could license images generated by **stabilityai/stable-diffusion-xl-base-1.0** under Creative Commons licenses like **CC-0-1.0, CC-BY-4.0, or CC-BY-SA-4.0**. The stable-diffusion-xl-base-1.0 model is licensed under the CreativeML Open RAIL++-M License.

- **Archive and community channels debate**: Members expressed frustration that certain community channels were deleted or archived, particularly pointing at **Cascade** and other art communities. **"The art communities were removed because they were inactive for nearly 2 months and were collecting bot spam,"** responded a mod, adding that these could be synced back if needed.

- **Opinions on Turbo and Finetuned Models**: Discussions highlighted differing views on "_Turbo_" models versus finetuned models like **Juggernaut** and **Pony**. Some members preferred the flexibility and speed of Turbo models while others argued that **finetunes serve specific purposes better**, particularly for detailed or concept-specific tasks.

- **Mobius model explanation**: A user introduced the **Mobius model** as a new state-of-the-art in debiased diffusion models, explaining its **domain-agnostic debiasing** technique. Queries about its large size and special requirements, like **clip skip 3**, were addressed by noting its training specifics and potential impact on **Lora compatibility**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/hatsune-miku-miku-hatsune-earthquake-plush-miku-death-gif-4018907532159793300">Hatsune Miku Miku Hatsune GIF - Hatsune miku Miku hatsune Earthquake - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/tasting-milk-antony-starr-the-homelander-the-boys-lick-gif-17834498">Tasting Milk Antony Starr GIF - Tasting Milk Antony Starr The Homelander - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/models/490622/mobius">Mobius - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Mobius: Redefining State-of-the-Art in Debiased Diffusion Models Mobius, a diffusion model that pushes the boundaries of domain-agnostic debiasing ...</li><li><a href="https://github.com/comfyanonymous/ComfyUI_TensorRT">GitHub - comfyanonymous/ComfyUI_TensorRT</a>: Contribute to comfyanonymous/ComfyUI_TensorRT development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/526316">Tsuki - v2 | Stable Diffusion Checkpoint | Civitai</a>: euler a, 30+ steps, use external vae flatcoloredponytest2+Bunny-XL2-V3-NS (0,0,0.0202546296296296,0.0787037037037037,0.171875,0.296296296296296,0.4...</li><li><a href="https://colab.research.google.com/github/mkshing/notebooks/blob/main/stable_video_diffusion_img2vid.ipynb#scrollTo=9AZDrh-SUDt2">Google Colab</a>: no description found</li><li><a href="https://civitai.com/articles/5800">Malicious Compliance | Civitai</a>: In response to CivitAI&#x27;s choice to significantly reduce user&#x27;s ability to gain buzz, as well as a particular attack on NSFW users, I propose as man...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1253075566790185050)** (319 messages🔥🔥): 

- **Perplexity CEO on Lex Fridman Podcast**: A YouTube video titled ["Aravind Srinivas: Perplexity CEO on Future of AI, Search &amp; the Internet | Lex Fridman Podcast #434"](https://youtu.be/e-gwvmhyU7A) was shared. A highlight includes Larry Page's philosophy, "the user is never wrong."

- **Pro Search Issues Reported**: Users discussed problems with the Pro Search feature not finding website sources with the pro toggle on, though it works fine on the iPhone app. Several users reported the same issue and it was escalated.

- **Perplexity vs. ChatGPT and Feature Requests**: Multiple users inquired about how Perplexity differs from ChatGPT, with explanations focusing on Perplexity's use of reputable web sources and detailed answers. There were also multiple requests and discussions about the need for more control over settings such as temperature settings and a potential "Advanced" UI mode for power users.

- **Claude 3.5 Sonnet Release Anticipation**: Users are looking forward to the release of Claude 3.5 Sonnet on Perplexity, with some highlighting its advantages for creative writing. There is excitement about its improved performance and concerns about how it will be integrated into the platform.

- **Controversy Over Respecting robots.txt**: A Wired article critique on Perplexity's adherence to robots.txt was shared and discussed. Users weighed in, pointing out the distinction between AI scraping information for training and retrieving information on behalf of a user, questioning whether the same rules should apply.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.wired.com/story/perplexity-is-a-bullshit-machine/">Perplexity Is a Bullshit Machine</a>: A WIRED investigation shows that the AI-powered search startup Forbes has accused of stealing its content is surreptitiously scraping—and making things up out of thin air.</li><li><a href="https://gizmodo.com/perplexity-ai-internet-rule-robots-exclusion-protocol-1851551095">Perplexity Is Reportedly Letting Its AI Break a Basic Rule of the Internet</a>: Perplexity is in hot water for its AI-generated articles.</li><li><a href="https://youtu.be/e-gwvmhyU7A">Aravind Srinivas: Perplexity CEO on Future of AI, Search &amp; the Internet | Lex Fridman Podcast #434</a>: Arvind Srinivas is CEO of Perplexity, a company that aims to revolutionize how we humans find answers to questions on the Internet. Please support this podca...</li><li><a href="https://docs.perplexity.ai/docs/feature-roadmap">Feature Roadmap</a>: no description found</li><li><a href="https://fxtwitter.com/perplexity_ai/status/1803861295432933801?s=19">Tweet from Perplexity (@perplexity_ai)</a>: This model outperforms Claude 3 Opus and GPT-4o on our internal benchmarks.</li><li><a href="https://x.com/AravSrinivas/status/1803870324213121362">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Claude 3.5 is now available on Perplexity Pro. In our international evaluations, it’s outperformed GPT 4o. Try it out!  Quoting Perplexity (@perplexity_ai)   🚨 Claude 3.5 Sonnet is now available on P...</li><li><a href="https://msty.app/">Msty - Using AI Models made Simple and Easy</a>: Interact with any AI model with just a click of a button</li><li><a href="https://rknight.me/blog/perplexity-ai-is-lying-about-its-user-agent/">Perplexity AI Is Lying about Their User Agent</a>: Perplexity AI claims it sends a user agent and respects robots.txt but it absolutely does not</li><li><a href="https://www.perplexity.ai/search/read-the-time-A0JXqn3iR86OjAnRVX4CEQ">read the time for me
think-aloud and write down your internal thoughts, check...</a>: Certainly, I&#x27;ll think through this step-by-step:  1. First, I&#x27;m looking at the hour hand. It&#x27;s clearly past the 2 but not quite at 3. 2. Now, the minute hand....</li><li><a href="https://www.perplexity.ai/page/Bananaclicking-game-tops-zW.nvAhGSzuXznHHEskL1Q">Banana-Clicking Game Tops Steam</a>: Banana, a simple clicker game where players repeatedly click on a virtual banana, has taken the Steam gaming platform by storm, amassing a staggering 884,469...</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/w5t5F1MtWc">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1253154378785886360)** (6 messages): 

- **Top 3 Self-Development Platforms**: Users discussed the top three self-development platforms such as Mindvalley, highlighting courses like "Unlimited Abundance" and "Superbrain" by Jim Kwik, which cover topics including mindfulness, health, and personal growth. [Learn more](https://www.perplexity.ai/search/What-are-the-3aHdlkCeTLunnzd4eTLozA).

- **Psychedelics and God Encounters**: A survey study revealed striking similarities between "God encounter" experiences induced by psychedelics and those that occur spontaneously, noting significant impacts on life satisfaction and meaning. A considerable portion of atheists changed their beliefs after such experiences. [Read more](https://www.perplexity.ai/search/What-is-the-cLSebvMCTH2CL_SgIVAQnQ).

- **Discover Today on Perplexity AI YouTube**: Perplexity AI's [YouTube video](https://www.youtube.com/embed/AXxR1aMNBls) covers various topics including AI safety, TikTok's creative tools, and recent astronomical discoveries.

- **High-Paying Jobs for English Literature Majors**: Jobs like technical writers and editors are among the highest paying for those with a master’s degree in English literature, offering average salaries of $74,000 and $63,000 respectively. [Explore more job options](https://www.perplexity.ai/search/What-jobs-can-zdbsXD8QQp2OGw79WcpQqw#5).

- **Lululemon Earnings Speculation**: Users discussed the imminent release of Lululemon Athletica Inc.'s financial results, with analysts predicting earnings per share of around $2.39 to $2.40. Speculations suggest these results could significantly influence the stock price. [More details here](https://www.perplexity.ai/search/LULU-is-set-eyOD1n8mQ..kH_kgodWYFg).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/LULU-is-set-eyOD1n8mQ..kH_kgodWYFg">LULU is set to release earnings today. Tell me more about any rumours about...</a>: Lululemon Athletica Inc. (NASDAQ: LULU) is scheduled to release its first-quarter financial results after the market closes today, June 5, 2024. There are...</li><li><a href="https://www.perplexity.ai/search/What-are-the-3aHdlkCeTLunnzd4eTLozA">What are the top 3 self-development platforms today?</a>: The top 3 self-development platforms today are:  1. Coursera  Coursera is an online learning platform that partners with top universities and organizations to...</li><li><a href="https://www.perplexity.ai/page/Piloting-an-AI-JDrMoyrVT6iNR_lul8DnQA">Piloting an AI System</a>: Is interacting with AI through chat more like piloting a system than having a real conversation? This thought-provoking analogy challenges how we typically...</li><li><a href="https://www.perplexity.ai/search/What-jobs-can-zdbsXD8QQp2OGw79WcpQqw#5">What jobs can someone with an ma in english literature get</a>: Here are some potential career paths for someone with a master&#x27;s degree in English literature:  Teaching College professor teaching English literature,...</li><li><a href="https://www.perplexity.ai/search/What-is-the-cLSebvMCTH2CL_SgIVAQnQ">What is the relationship between psychedelics and god</a>: Based on the provided search results, there appears to be a complex relationship between psychedelic experiences and encounters or perceptions related to God,...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1253210297334894602)** (5 messages): 

- **Perplexity API limits customization**: A member shared their impression that the Perplexity API hosts a single HTTPS endpoint without citation in responses and offers limited integration options for building custom agents. A positive aspect noted was its capability to run large open-source LLMs when users handle tokenization or text embeddings themselves.
- **Pages feature not accessible via API**: A user inquired about accessing the Pages feature via the API, but another responded that this is **not possible**.
- **Reset your API key easily**: To reset an API key, users can visit the [Perplexity API settings page](https://www.perplexity.ai/settings/api). The section includes options to "delete" and "generate" new keys.

**Link mentioned**: <a href="https://www.perplexity.ai/settings/api">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1253404964664119306)** (2 messages): 

- **Character.AI achieves efficient Int8 training**: A member shared a [link](https://research.character.ai/optimizing-inference/) discussing Character.AI's efforts towards AGI by optimizing inference to be more efficient, cost-effective, and scalable. *"We serve more than 20,000 inference queries per second, roughly 20% of the request volume served by Google Search."*
- **Curiosity about AQT usage**: Another member questioned whether Character.AI uses Adaptive Quantization (AQT) for their efficient inference process.

**Link mentioned**: <a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>: At Character.AI, we&#x27;re building toward AGI. In that future state, large language models (LLMs) will enhance daily life, providing business productivity and entertainment and helping people with e...

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1253064107762909285)** (8 messages🔥): 

- **Profiling Kernels with Nsight**: Members suggested profiling CUDA kernels using Nsight Compute to identify bottlenecks and optimize performance. They shared a [GitHub script for profiling](https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh) and a [YouTube playlist](https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR) as resources.
- **Understanding 'ncu' Command**: When asked about 'ncu' in a profiling script, it was explained that 'ncu' stands for Nsight Compute. A [user guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) was provided to give more details about using the CLI tool.
- **Using Triton with Nsight Compute**: It was confirmed that **Triton**, which compiles to **PTX**, is compatible with **Nsight Compute**. This allows for performance profiling in a similar manner to CUDA.
- **Upgrading to Triton 3.0.0**: One member suggested upgrading to Triton 3.0.0 to resolve issues and provided a detailed [installation guide](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/). The guide includes steps for uninstalling the current version, cloning the new repository, and setting it up either for development or regular use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/">Installing Triton 3.0.0</a>: As of June 13 2024, to get Triton 3.0 you have to install it from source, like so:</li><li><a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html">4. Nsight Compute CLI &mdash; NsightCompute 12.5 documentation</a>: no description found</li><li><a href="https://github.com/AnswerDotAI/bitlora/blob/master/benchmarks/forward_kernel/profile_forward_kernel.sh">bitlora/benchmarks/forward_kernel/profile_forward_kernel.sh at master · AnswerDotAI/bitlora</a>: Experimental q[X]ora kernel development code. Contribute to AnswerDotAI/bitlora development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR">CUDA Developer Tools</a>: This video series will help get you started with NVIDIA Nsight Developer Tools for CUDA. Grow your proficiency with the tools and apply the examples to your ...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1253361205662187630)** (1 messages): 

- **AI Unplugged Edition 13 Discusses Latest AI Trends**: The 13th edition of AI Unplugged covers recent advancements such as **Qwen2**, **DiscoPOP**, **Mixture of Agents**, **Grokked Transformers**, and **YOLO v10**. More details can be found in the [blog post](https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture).

- **Mixture-of-Agents Enhances LLM Capabilities**: A new paper from [Together.ai](http://together.ai) discusses how using multiple LLMs simultaneously can enhance performance, detailed in their [arxiv paper](https://arxiv.org/pdf/2406.04692) and associated [blog post](https://www.together.ai/blog/together-moa).

- **ChatGPT’s Custom GPT 'Advisory Board' Explored**: There's a custom GPT called [Advisory board](https://chatgpt.com/g/g-mhH7nIrJW-advisory-board-2-0-with-hats) that lets GPTs enact different characters to evaluate responses from multiple perspectives, a practical example of mixing LLMs to provide complex answer evaluations.

**Link mentioned**: <a href="https://datta0.substack.com/p/ai-unplugged-13-qwen2-discopop-mixture">AI Unplugged 13: Qwen2, DiscoPOP, Mixture of Agents, YOLO v10, Grokked Transformers</a>: Insights over Information

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1253094885901467768)** (4 messages): 

- **Nous Research seeks advanced ML Engineers**: *Nous Research is hiring CUDA/Triton engineers for implementing modeling code in pytorch and optimizing performance with Triton and CUDA.* They are looking for candidates who can write custom Triton Kernels to enhance training efficiency and are open to both full-time and contract positions. [Nous Research](https://nousresearch.com/) [Twitter](https://twitter.com/nousresearch/) [LinkedIn](https://www.linkedin.com/company/nousresearch/)
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1253063820629246023)** (4 messages): 

- **Native Files App Boosts Productivity**: A member shared their enthusiasm for using the native **Files app** on Mac and iPhone, *"awesome cause I can immediately open them in my mac & iphone’s native file browsers (and fiddle with in command line)"*. This setup enables seamless AirPlay to mirror content to a Mac and use it as a video source in Streamlabs.

- **Reflector 4 Offers Alternative Mirroring**: Another member suggested **Reflector 4** as a solution for mirroring content, which allows usage of a Mac while AirPlaying, but noted, *"you might need a higher spec mac than an M1 Pro, it can make mine crash while streaming"*.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1253415962322014288)** (3 messages): 

- **Quantization Methods in CUDA MODE**: The discussion highlights that **supported quantization methods** are quite specific, pointing out that terms like `int4_weight_only` quant are tailored to systems like **tinygemm**. As an example, *"it is int4 weight only quantization that's compatible with tinygemm, which is `uint4 weight-only asymmetric per-group quantization`"* ([repository link](https://github.com/pytorch/ao/blob/e6460c22284df669b95dc912114cc29b40df2030/torchao/quantization/quant_primitives.py#L280-L289)).

- **FP8 Quantization Challenges**: Concerns were raised about **FP8** quantization, highlighting the need for a flow to choose the best scale. It was suggested that users should have the option to skip quantizing a layer if the performance loss is unacceptable, saying *"people should be able to choose to not quantize a layer if the loss is too much"*.

- **Exporting Quantized Models for XLA**: The idea of creating a **quantized model** that includes quant/dequant operations for export and execution in **XLA** was proposed. More details are needed to explore how this could be implemented, but it aims for better integration with XLA graph execution.
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1253371751820431482)** (3 messages): 

- **Seeking tricks for speeding up diffusion models**: A member asked for favorite tricks to speed up diffusion models without architecture changes or retraining, mentioning they already achieved a **20% performance increase** using **torch compile max autotune**. The thread was automatically created for this discussion in a specific channel.
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1253177566055891054)** (4 messages): 

- **Zero in HQQ causes compatibility issues**: A user noted that HQQ quantizes a float tensor into an integer `qweight`, and float `zero` and `scale`, while algorithms like RNT and GPTQ use integer `zero`. They questioned if the quantized result of HQQ could reuse the `w4a16` kernel for GPTQ and RTN.

- **Rounded zero-point and fastest kernel**: Another member explained that the zero-point is rounded for 4-bit quantization to use `int8` and mentioned using the [torchao kernel](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py#L41), the fastest available for grouped quantization. They also noted support for Marlin in channel-wise quantization, but emphasized that torchao is preferable for compatibility and performance.

- **Performance gain details for Marlin**: A user inquired about the performance gain from using Marlin's split float `matmul` method. The other member responded with a [link to performance data](https://github.com/mobiusml/hqq/blob/master/imgs/llama_int4_4090.png?raw=true) and recommended using the `torchao` solution for better compatibility and speed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py#L41">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/marlin_int4_demo.py">hqq/examples/backends/marlin_int4_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1253063543951982602)** (230 messages🔥🔥): 

```html
<ul>
  <li>
    <strong>LR Schedulers PR is Ready</strong>: The <a href="https://github.com/karpathy/llm.c/pull/605">PR #605 on GitHub</a> for adding learning rate schedulers is refactored and simplified. "Main file is shorter now."
  </li>
  <li>
    <strong>Discussion on SLURM Issues</strong>: Several users discussed troubleshooting SLURM configurations and scripts, particularly issues with dummy GPU runs hanging or only recruiting a single node. One user noted, "i restarted the nodes last night."
  </li>
  <li>
    <strong>Updated Bias_Backward PR Feedback</strong>: The updated <a href="https://github.com/karpathy/llm.c/pull/619">bias_backward PR</a> showed a slight speedup but introduced concerns about potential deadlocks. "We were making sure not to call `__syncthreads` ... inside a conditional branch."
  </li>
  <li>
    <strong>H100 Box Experimentation</strong>: A user shared their experience running a 1558M model on an H100 box, achieving a 2.5x naive speedup compared to an A100. "From 8.1 days -> 3.2 days expected."
  </li>
  <li>
    <strong>Evaluating Using Latest Eval Harness</strong>: Members discussed the speed of different versions of the eval harness, showing significant speed improvements. "From 40 mins to 1 min ... suggested using the latest eval harness release for quicker comparison evals."
  </li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mdouglas/llmc-gpt2-774M-150B">mdouglas/llmc-gpt2-774M-150B · Hugging Face</a>: no description found</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress">NCCL and MPI &mdash; NCCL 2.21.5 documentation</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/619">Cast Get2dNoiseUint computation to uint by gordicaleksa · Pull Request #619 · karpathy/llm.c</a>: After the conversation with Squirrel (author of the noise rnd generator we&#39;re using) it might be a good idea to cast the intermediate computation to uint to avoid dealing with ints having UB (unde...</li><li><a href="https://github.com/karpathy/llm.c/pull/624">if available, use MPI env vars to initialize multi-gpu configs by ngc92 · Pull Request #624 · karpathy/llm.c</a>: let&#39;s see what windows thinks of this</li><li><a href="https://github.com/karpathy/llm.c/pull/623">feature/nccl only (delete MPI) by karpathy · Pull Request #623 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/605">Add learning rate schedulers by gordicaleksa · Pull Request #605 · karpathy/llm.c</a>: Refactored the learning rate schedulers code - we&#39;ll keep all definitions inside &quot;schedulers.h&quot; as per our offline discussion. Supported LR schedulers:  Cosine with warmup Cyclic triangu...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1253185718193360956)** (22 messages🔥): 

- **New FPx Kernel in FP6-LLM**: The new FPx kernel from FP6-LLM can perform FP16 operations with FPx for x ranging from 1 to 7, though FP1 and FP2 are less practical. Limitations include lack of support for group-wise quantization, as mentioned *"If someone has time and knowledge, probably can try add support for it 😂."*

- **FP6 Benchmarks Show Varied Results**: Benchmarks for different quantization methods on wikitext show varied perplexity and token speeds. Notably, FP4 and FP3 quantizations lead to significant perplexity distortions compared to others like FP5 or FP6.

- **End-to-End Test Case Development for uint2**: There is ongoing work to develop an end-to-end test case for uint2 quantization, alongside ongoing benchmarking tasks.

- **FP2 and FP4 Speedup over FP16**: Using FPx-LLM kernel, FP2 and FP4 multiplications show speedups ranging from 3.77x to 10.64x over FP16 multiplications, depending on matrix sizes.

- **FP16 to FP8 Conversion Challenges**: Discussion highlighted challenges in implementing fp16 -> fp8 conversions, noting potential slowdowns despite hardware instructions availability. One member noted *"fp16->fp8 can be done literally by chopping off 8 mantissa bits."* but acknowledged complications from scaling requirements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/gau-nernst/ao/blob/fp5_llm/torchao/prototype/fp6_llm/fp6_llm.py">ao/torchao/prototype/fp6_llm/fp6_llm.py at fp5_llm · gau-nernst/ao</a>: Native PyTorch library for quantization and sparsity - gau-nernst/ao</li><li><a href="https://github.com/gau-nernst/ao/blob/fp5_llm/torchao/csrc/cuda/fp6_llm/utils_parallel_dequant.cuh">ao/torchao/csrc/cuda/fp6_llm/utils_parallel_dequant.cuh at fp5_llm · gau-nernst/ao</a>: Native PyTorch library for quantization and sparsity - gau-nernst/ao</li><li><a href="https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/fp6_linear.cu#L177">fp6_llm/fp6_llm/csrc/fp6_linear.cu at 5df6737cca32f604e957e3f63f03ccc2e4d1df0d · usyd-fsalab/fp6_llm</a>: An efficient GPU support for LLM inference with x-bit quantization (e.g. FP6,FP5). - usyd-fsalab/fp6_llm</li><li><a href="https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/include/utils_parallel_dequant.cuh">fp6_llm/fp6_llm/csrc/include/utils_parallel_dequant.cuh at 5df6737cca32f604e957e3f63f03ccc2e4d1df0d · usyd-fsalab/fp6_llm</a>: An efficient GPU support for LLM inference with x-bit quantization (e.g. FP6,FP5). - usyd-fsalab/fp6_llm
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Tumxml3DCvM
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1253448619177279580)** (1 messages): 

- **Hermes 2 Theta 70B takes the spotlight**: The announcement introduces **Hermes 2 Theta 70B**, which is described as "smarter, more creative, and capable of more" compared to its predecessors. Hermes 2 Theta achieves a **9.04 score on MT-Bench**, surpassing GPT-4-0314's score of 8.94, and outperforms Llama-3 Instruct 70B in multiple benchmarks.
- **Advanced features and capabilities highlighted**: **Hermes 2 Theta** supports function calling, feature extraction, and JSON mode outputs for agentic capabilities, indicating advanced functionality for users. This highlights the potential applications in more complex AI-driven tasks and operations.
- **Collaboration with Arcee AI**: This release is a continuation of Nous Research's collaboration with **Charles Goddard** and **Arcee AI**, the team behind MergeKit. The model is a merged and further RLHF'ed version combining **Hermes 2 Pro** and **Meta's Llama-3 Instruct**.
- **Access and download options**: The FP16 version of the model can be downloaded from [Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B), while the quantized GGUF version is available [here](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF). Detailed model descriptions and comparisons are provided on the download pages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF">NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253065913649856542)** (272 messages🔥🔥): 

- **Model Parsing and Tokenization Debates**: Members discussed the complexities of building a parser to handle tool calls from model-specific formats into a universal format like OpenAI's. A consensus emerged around using "reverse templates" and potentially integrating it into `tokenizer_config.json` for added flexibility.

- **Claude 3.5 Sonnet Impresses**: The release of Claude 3.5 Sonnet sparked excitement, with several users noting its superior performance and capabilities compared to previous models. One member called it "superfast opus" and praised its competency in solving complex tasks.

- **Hermes 2 Theta's Moderate Uncensorship**: Discussion around **Hermes 2 Theta** highlighted its moderate uncensorship while maintaining informative responses, sparking debates on the ideal balance between refusal and realistic dialogue.

- **Integrating Tools with Models**: A member explained a process to integrate tools seamlessly with models, notably skipping special tokens and inserting tool responses directly into the prompt. This method, merging tools into the model's flow, was suggested to ease multi-tool integrations.

- **Claude 3.5 Sonnet's Impact**: The new Claude model demonstrated impressive capabilities, including understanding and solving obscure programming problems accurately. The model's speed and attentiveness were particularly praised, positioning it as a significant advancement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/fireworks-ai/firefunction-v2">fireworks-ai/firefunction-v2 · Hugging Face</a>: no description found</li><li><a href="https://www.anthropic.com/news/claude-3-5-sonnet">Introducing Claude 3.5 Sonnet</a>: Introducing Claude 3.5 Sonnet—our most intelligent model yet. Sonnet now outperforms competitor models and Claude 3 Opus on key evaluations, at twice the speed.</li><li><a href="https://x.com/teortaxesTex/status/1803611506908529060">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: To spell it out. In June 2024, Ilya Efimovich Sutskever has destroyed the possibility of a binding USA-China treaty on AGI/ASI.  It was always far-fetched, but from now on there is no way China will h...</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/mudler/LocalAI/blob/43f0688a95ce5a5f43228ae288020bef02770e8e/pkg/functions/parse.go#L124">LocalAI/pkg/functions/parse.go at 43f0688a95ce5a5f43228ae288020bef02770e8e · mudler/LocalAI</a>: :robot: The free, Open Source OpenAI alternative. Self-hosted, community-driven and local-first. Drop-in replacement for OpenAI running on consumer-grade hardware. No GPU required. Runs gguf, trans...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1253102983932154010)** (2 messages): 

- **Exciting new resource on the way**: A member praised another's question and hinted at creating something special to answer it. Another member expressed enthusiasm, saying, *"Incredible! Can't wait to see."*
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1253394167527247964)** (3 messages): 

- **Sonnet 3.5 drops, Opus 3.5 next?**: A member announced the release of **3.5 Sonnet** and speculated that **Opus 3.5** might be coming soon. "3.5 Sonnet is here! Opus 3.5 presumably not too far away. 😎".
- **Members Share Excitement Over New Release**: Another member expressed excitement following the announcement. Simple reaction: "exciting".
- **Link to Music Video Shared**: A member shared a [YouTube video](https://youtu.be/E3Yt_qLUGJY) titled "L'ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley & Khoe Wa (Official Video)," promoting the music album "Chickens In Your Town."

**Link mentioned**: <a href="https://youtu.be/E3Yt_qLUGJY">L&#39;ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley &amp; Khoe Wa (Official Video)</a>: &quot;Lobster Shwarama Feat Troy Berkley &amp; Khoe Wa&quot; taken from L&#39;Entourloop &quot;Chickens In Your Town&quot; album, available 👉  https://smarturl.it/LNTRLPChickensIYT♦︎ V...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1253061758453813300)** (254 messages🔥🔥): 

- **Can't use Hugging Face location directly**: A user clarified, "*With our current in-memory datasets, it will download from the HF location to local disk.*" They are working towards *streaming datasets* to avoid saving datasets on disk.
- **Setting up HF dataset in Torchtune**: Users discussed configuring the HF dataset in `torchtune.dataset.chat_dataset`, eventually deciding that using `conversation_style: openai` should work OOTB without additional converters.
- **Max sequence length confusion**: There was debate over **llama3**'s maximum sequence length, initially believed to be 4096 but could be up to 8192. However, a member noted, "*I doubt I could fit that size context on my vram.*"
- **Handling crashes and memory issues**: Users faced several RAM-related crashes when training models, specifically with qlora and lora. A possible solution discussed was to offload layers to *CPU* and troubleshoot ROCm setup.
- **Experiences with ROCm setup**: Members shared issues and resources related to setting up ROCm for AMD GPUs, with one saying, "*You need to build them from source or get them from an alternative source.*" They also noted a Reddit resource where someone successfully ran ROCm on a 6900 XT.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/chat.html">Fine-tuning Llama3 with Chat Data &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/discussions/1090">Has anyone run this with AMD ROCm (specifically RDNA3 / gfx1100) · pytorch/torchtune · Discussion #1090</a>: I submitted an issue to the ROCm team here ROCm/hipBLASLt#831 but just curious if anyone else (authors or uses) has ever tried running torchtune with ROCm? I am using a very standard install (lates...</li><li><a href="https://pytorch.org/torchtune/stable/install.html#install-nightly-build">Install Instructions &mdash; TorchTune  documentation</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clxhtp/amd_rocm_61_local_multigpu_6900xt_vii_setup/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html">System requirements (Linux) — ROCm installation (Linux)</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/ef6e196d8e47e9bc584bc9f7ce836f646443381f/recipes/lora_finetune_single_device.py#L277C9-L277C50">torchtune/recipes/lora_finetune_single_device.py at ef6e196d8e47e9bc584bc9f7ce836f646443381f · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/s1nMpiuc1Q">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1253447468239294606)** (1 messages): 

- **MidJourney image-captions dataset hops onto Github**: A 520k MidJourney image+caption [dataset](https://github.com/bghira/SimpleTuner/blob/main/documentation/QUICKSTART.md) has been released. It promises a wealth of visual data for training and experimentation.
  
- **PixArt Model highlights its 900M parameters**: Featuring public photos, DALLE-3 + Midjourney datasets, the 900M param PixArt [model](https://huggingface.co/ptx0/pixart-900m-1024-ft) is now available. This model leverages a diverse set of data sources for image generation tasks.

- **Bulkproteinviz ups the ante for protein predictions**: The updated [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) now supports multiple structure predictions at once. It's a significant step forward for those working in computational biology.

- **Powershell meets AI with function calling support**: The [Powershell + AI integration](https://github.com/rrg92/powershai) now includes support for function calling. This update offers enhanced functionality for developers and automation specialists.

- **Drug names swap disrupts model performance**: A recent study switched generic drug names to their brand names in biomedical benchmarks, revealing performance drops across most models. The full paper on this study can be found [here](http://arxiv.org/abs/2406.12066), and a leaderboard is available on [Hugging Face](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/taha_yssne/status/1802607279809630562)">Tweet from Taha Yassine (@taha_yssne)</a>: I just wrote a blog post about the temperature parameter in LLMs, but really it was just an excuse to play with Transformers.js. I had fun implementing an interactive demo of the impact of T on genera...</li><li><a href="https://x.com/shan23chen/status/1803459255518769509)">Tweet from Shan Chen (@shan23chen)</a>: 💊 We took your language model to the drug store… and it knew about acetaminophen (generic name) better than Tylenol (brand name)! @hughbzhang @scale_AI developed GSM1K last month, where they found ma...</li><li><a href="https://blog.cubed.run/5-chunking-techniques-in-rag-1250c8e1f49f)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1253061783716102299)** (142 messages🔥🔥): 

- **Stable Diffusion and VSCode Troubleshooting**: A user asked about integrating **Stable Diffusion** with VSCode. Another user clarified that since VSCode is just an editor, one should use the terminal within it to run the required commands.

- **Llama 3 Fine-Tuning Issues**: A user experienced issues while fine-tuning **Llama 3** using TRL with QLoRa and received semi-nonsensical outputs. They shared their code and discussed potential problems with parameters, linking to some research and trying different `lora_rank`.

- **Errors with Stable Diffusion 3**: Some users reported getting errors when trying to use **Stable Diffusion 3** due to a missing model index. One user mentioned using the **stable-diffusion-3-medium-diffusers** model as a workaround.

- **Hugging Face Service Outages**: Multiple users reported 504 errors and sporadic issues accessing **Hugging Face** services. The status page initially indicated all services were online, leading to confusion and speculations about server overload.

- **Florence-2 Model Impresses Users**: The **Florence-2** large model from Microsoft impressed users with its versatility and efficiency, supporting tasks like captioning, OCR, and object detection despite its smaller size. Conversations highlighted its potential use in low-power devices like Raspberry Pi and its comparison to other models like DINOv2.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/deep-rl-course/unitbonus1/train">Let’s train and play with Huggy 🐶 - Hugging Face Deep RL Course</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb">sample_inference.ipynb · microsoft/Florence-2-large at main</a>: no description found</li><li><a href="https://youtu.be/S_3TVEVE8y4?si=4zw5brZDugG4yFtY">Blade Runner Off-World 2055</a>: Blade Runner Off-World  AI / 3D short using MidJourney, RunwayML, ElevenLabs, Magnific Maya &amp; Zbrush. #sora #Ai #3dmodeling #ridleyscott  #giger Setting: The...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/iy9Z4DyHxvE?si=eHxvWI4rfSwk8YH1">How to hack a LLM using PyReft (using your own data for Fine Tuning!)</a>: 🚀 Sign up to the newslettergo.coursesfromnick.com/newsletter👨‍💻 Sign up for the Full Stack course and use YOUTUBE50 to get 50% off:https://www.coursesfrom...</li><li><a href="https://en.wikipedia.org/wiki/White_Christmas_(Black_Mirror)">White Christmas (Black Mirror) - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/blog/llama3">Welcome Llama 3 - Meta&#39;s new open LLM</a>: no description found</li><li><a href="https://tenor.com/view/frank-castle-wait-please-stop-please-no-please-gif-21133188">Frank Castle Wait GIF - Frank Castle Wait Please Stop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/PWhiddy/PokemonRedExperiments/blob/master/windows-setup-guide.md">PokemonRedExperiments/windows-setup-guide.md at master · PWhiddy/PokemonRedExperiments</a>: Playing Pokemon Red with Reinforcement Learning. Contribute to PWhiddy/PokemonRedExperiments development by creating an account on GitHub.</li><li><a href="https://github.com/PWhiddy/PokemonRedExperiments">GitHub - PWhiddy/PokemonRedExperiments: Playing Pokemon Red with Reinforcement Learning</a>: Playing Pokemon Red with Reinforcement Learning. Contribute to PWhiddy/PokemonRedExperiments development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ZyKswO6xDbTuyMQw5NTSlri1fRHw43_l?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1253168410032865371)** (3 messages): 

- **Synthesizing multi-table databases analyzed**: A shared [article](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/) focuses on the challenges of generating high-quality synthetic multi-table tabular data, particularly with date columns. It evaluates three vendors, highlighting difficulties with libraries like SDV, Gretel, and Mostly.ai regarding data integrity preservation, run time, and business rule compliance.
- **ToolkenGPT proposes innovative approach**: This [paper](https://doi.org/10.48550/arXiv.2305.11554) discusses **ToolkenGPT**, which embeds external tools as tokens to be used by large language models (LLMs) in a way akin to generating regular word tokens. This aims to overcome limitations of finetuning and in-context learning constrained by context length and the number of tools.
- **Drug names in biomedical benchmarks reveal performance drop**: A tweet highlighted a study where generic drug names were switched with brand names in biomedical benchmarks like MedQA and MedMCQA, noting a performance drop across most models. This experiment, detailed in a [paper](http://arxiv.org/abs/2406.12066) and a Hugging Face [leaderboard](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard), indicates possible data contaminations in public pre-training datasets.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/">Synthesizing Multi-Table Databases: Model Evaluation &amp; Vendor Comparison - Machine Learning Techniques</a>: Synthesizing multi-table tabular data presents its own challenges, compared to single-table. When the database contains date columns such as transaction or admission date, a frequent occurrence in rea...</li><li><a href="https://doi.org/10.48550/arXiv.2305.11554">ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings</a>: Augmenting large language models (LLMs) with external tools has emerged as a promising approach to solving complex problems. However, traditional methods, which finetune LLMs with tool demonstration d...</li><li><a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Tweet from Shan Chen (@shan23chen)</a>: 💊 We took your language model to the drug store… and it knew about acetaminophen (generic name) better than Tylenol (brand name)! @hughbzhang @scale_AI developed GSM1K last month, where they found ma...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1253125109837860914)** (21 messages🔥): 

- **Debate on promoting AI covers in server**: Members discussed whether promoting AI covers is appropriate on the HuggingFace server. One user clarified they prefer not to share models publicly and only promote AI covers of Kpop songs, not their own creations.
- **Biomedical NLP models recognize generic drug names better**: Users discussed a study showing NLP models recognize generic drug names like acetaminophen better than brand names like Tylenol. The study found evidence of data contamination and shared a [paper](http://arxiv.org/abs/2406.12066) and [leaderboard](https://huggingface.co/spaces/AIM-Harvard/rabbits-leaderboard) related to the dataset "RABBITS".
- **Open-source protein structure prediction tool update**: An announcement was made about the release of **BulkProteinviz**, a new feature in an open-source protein structure prediction tool allowing multiple predictions from a FASTA file, promoting enhanced research speed.
- **Massive data and model releases for image generation**: A comprehensive update included guides, datasets, and models such as 520k image files for Midjourney v6, training a 900M parameter PixArt, and new fine-tunes of SD3 models.
- **Open-source backgammon simulation project**: A new project was announced that runs high-speed backgammon simulations and is open for further development at [GitHub - C1N-S4/Backgamoon-A.I-tool](https://github.com/C1N-S4/Backgamoon-A.I-tool). The project aims to add user interface and optimization enhancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Tweet from Shan Chen (@shan23chen)</a>: 💊 We took your language model to the drug store… and it knew about acetaminophen (generic name) better than Tylenol (brand name)! @hughbzhang @scale_AI developed GSM1K last month, where they found ma...</li><li><a href="https://github.com/C1N-S4/Backgamoon-A.I-tool">GitHub - C1N-S4/Backgamoon-A.I-tool</a>: Contribute to C1N-S4/Backgamoon-A.I-tool development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1253065454411579443)** (6 messages): 

- **New user seeks comic style model recommendations**: A member expressed interest in fine-tuning a model with a dataset related to comic styles. The goal is *"to generate new ones in similar/same style."*
- **Deep Guided Posterior Regularization shared**: Another member provided a link to [DeGPR for Medical Object Detection](https://paperswithcode.com/task/medical-object-detection/latest). The discussion covered the challenges of applying general-purpose deep learning methods to medical images, such as class imbalance and tiny overlapping objects.
- **Request for Java-based object detection app**: A member inquired about creating an object detection app in Java. They are interested in features like custom detection and live detection, comparable to YOLO in Python.

**Link mentioned**: <a href="https://paperswithcode.com/task/medical-object-detection/latest">Papers with Code - Medical Object Detection</a>: Medical object detection is the task of identifying medical-based objects within an image.
 
 &lt;span style=&quot;color:grey; opacity: 0.6&quot;&gt;( Image credit: [Liver Lesion Detection from Weakly...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1253142876699885669)** (7 messages): 

- **Noteworthy Resources Shared**: A user shared resources and notebooks such as [this guide on using llama2-faiss and Langchain](https://github.com/murtuza753/llama2-faiss-langchain-qa-rag/blob/main/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb) and another on [LLM fine-tuning using PEFT](https://github.com/ashishpatel26/LLM-Finetuning).

- **Reranking Issue in RAG Pipelines**: A user raised a problem about reranking in their RAG pipeline where reranking scores are low for certain queries like "what are my products?". They are considering adjusting the reranking scores based on results from their vector database.

- **Help Requested for vLLM Fine-tuning**: A user asked for guidance on fine-tuning models with vLLM and encountered an issue related to loading a fine-tuned model and LoRA adapters with a specific error: *"ValueError: Cannot find any of ['adapter_name_or_path'] in the model's quantization config."*

- **Detailed Steps for Model Fine-tuning**: The same user detailed steps for fine-tuning, merging, saving, and loading models using PEFT, including attempts at inference with vLLM. The steps included saving checkpoints, merging the model, handling device maps, and the subsequent error encounter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/murtuza753/llama2-faiss-langchain-qa-rag/blob/main/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb">llama2-faiss-langchain-qa-rag/Using_llama2_faiss_and_langchain_for_question_answering_on_your_own_data.ipynb at main · murtuza753/llama2-faiss-langchain-qa-rag</a>: Contribute to murtuza753/llama2-faiss-langchain-qa-rag development by creating an account on GitHub.</li><li><a href="https://github.com/ashishpatel26/LLM-Finetuning">GitHub - ashishpatel26/LLM-Finetuning: LLM Finetuning with peft</a>: LLM Finetuning with peft. Contribute to ashishpatel26/LLM-Finetuning development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1253313451791613952)** (2 messages): 

- **Seeking help with Llama 3:70B training**: A user mentioned their installation of Llama 3:70B via **Ollama** but noted that the current training data is only 40GB. They are seeking advice on how to train additional datasets to increase the size to at least 200GB locally.
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1253135824279699496)** (53 messages🔥): 

- **MLIR: An Internal and Confusing Dialect**: A user noted confusion around the **`kgen` dialect** in MLIR, revealing it is an **internal dialect without public documentation**. Another user expressed frustration, adding, *"the code is sooo messy on top of the messyness of MLIR."*
  
- **Implementing Custom Types in MLIR**: Discussion on implementing **256-bit integers** in MLIR led to suggestions such as using **`SIMD[DType.int64, 4]`** or directly **defining an `i256` type**. Helpful references were provided, including a [GitHub link](https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232).

- **Understanding MLIR and Its Conversion**: A user was puzzled about how MLIR translates to LLVM IR for optimization, prompting explanations about **MLIR dialects and transformation infrastructure**. It was clarified that MLIR operations could be converted into assembly through custom backend or by lowering to LLVM IR.

- **Package Manager Progress for Mojo**: A user inquired about the progress of a Mojo package manager. Different users pointed out existing efforts, such as [Hammad-hab's `pkm`](https://github.com/Hammad-hab/pkm) on GitHub, and noted that the Modular team has discussed this in community meetings.

- **Modular Community Livestream Announcement**: The community was informed about a **live Modular Community Livestream** discussing new features in MAX Engine and Mojo, available on [YouTube](https://www.youtube.com/watch?v=uookgZ7Ojg8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#exception-is-actually-called-error">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://www.youtube.com/watch?v=uookgZ7Ojg8">Modular Community Livestream - New in MAX 24.4</a>: MAX 24.4 is now available! Join us on our upcoming livestream as we discuss what’s new in MAX Engine and Mojo🔥 - MAX on macOS, MAX Engine Quantization API, ...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L">mojo/stdlib/src/builtin/simd.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/simd.mojo#L231-L232">mojo/stdlib/src/builtin/simd.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2002.11054">MLIR: A Compiler Infrastructure for the End of Moore&#39;s Law</a>: This work presents MLIR, a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, sign...</li><li><a href="https://github.com/Hammad-hab/pkm">GitHub - Hammad-hab/pkm: Mojo&#39;s unoffical package manager</a>: Mojo&#39;s unoffical package manager. Contribute to Hammad-hab/pkm development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1253388280100294656)** (2 messages): 

- **New Modular Update Alert**: Modular recently shared two tweets related to new developments on their Twitter account. Check the updates [here](https://twitter.com/Modular/status/1803828734992207974) and [here](https://twitter.com/Modular/status/1803850466767573143) for more detailed information.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1253075360229363825)** (56 messages🔥🔥): 

- **Mojo is partially open source**: Mojo, the language, is open source and available on [GitHub](https://github.com/modularml/mojo). However, the compiler is still proprietary but freely licensed, with parts of it being progressively open-sourced ([detailed blog post](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)).
- **Intable type handling explained**: Users experienced issues with explicit casting in the `count_many_things` function when using `Intable`. It was clarified that explicit casting to `int` is required to prevent errors and that `Intable` indicates the type has an `__int__` method.
- **Current limitations of Mojo for production use**: It was suggested that Mojo lacks key features for production, such as a package manager, advanced stdlib functions, traits, async, and more. A broader release aimed at being more suitable for real software is targeted around December.
- **Advice on using Mojo in automation work**: Athena shared interest in using Mojo for its implicit parallelism and enhanced type safety. However, until Mojo matures and stabilizes further, it was advised against integrating it into production or complex automation work.
- **Claude's Sonnet 3.5 capabilities**: A brief mention highlighted that Claude's new Sonnet 3.5 model excels at generating Mojo code compared to GPT-4, suggesting it might be useful for developers looking to implement or experiment with Mojo code.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://www.perplexity.ai/">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://docs.modular.com/mojo/faq#will-mojo-be-open-sourced">Mojo🔥 FAQ | Modular Docs</a>: Answers to questions we expect about Mojo.</li><li><a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1253072418176761999)** (1 messages): 

- **Engine docs clarify 'execute' function details**: A member pointed out that the `execute` method can take a variadic `NamedTensor` or `Tuple[StringLiteral, EngineNumpyView]`, supporting this with a link to the [documentation](https://docs.modular.com/max/api/mojo/engine/model/Model#execute). They also provided a link to the [NamedTensor documentation](https://docs.modular.com/max/api/mojo/engine/tensor/NamedTensor) for further reference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/engine/model/Model#execute">Model | Modular Docs</a>: Represents a model that&#x27;s loaded and ready for execution.</li><li><a href="https://docs.modular.com/max/api/mojo/engine/tensor/NamedTensor">NamedTensor | Modular Docs</a>: A named input tensor.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1253128778792829070)** (8 messages🔥): 

- **New Mojo Compiler Released**: The latest nightly update for the Mojo compiler, version `2024.6.2005`, has been released. Users can update using `modular update nightly/mojo` and view the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) as well as the [raw diff](https://github.com/modularml/mojo/compare/d96acc9161ce91d93d9a24424cb8870906440e05...279ade23a9409a545a723236f271c5061d2f005b).
- **Tool for stdlib contributors**: A tool intended for standard library contributors, titled "mojo_dev_helper," was shared. More details can be found [here](https://github.com/rd4com/mojo_dev_helper).
- **Importing initialize_pointee_move issue**: A member encountered an issue importing `initialize_pointee_move`. It was clarified that this method is available as part of `UnsafePointer` in the nightly branch but may not be accessible in other versions.

**Link mentioned**: <a href="https://github.com/rd4com/mojo_dev_helper">GitHub - rd4com/mojo_dev_helper: 🦺 small tool for stdlib contributors.</a>: 🦺 small tool for stdlib contributors. Contribute to rd4com/mojo_dev_helper development by creating an account on GitHub.

  

---



### **AI Stack Devs (Yoko Li) ▷ #[committers](https://discord.com/channels/1122748573000409160/1122748682475950142/1253138286860304544)** (7 messages): 

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Committers Channel Summary</title>
</head>
<body>
  <ul>
    <li><strong>Spam Invites Flood Channel</strong>: Multiple messages promoting *18+ Free Content* and OnlyFans leaks were posted in the channel. Each message included a repeated invitation with a Discord link and a description of explicit content.</li>
  </ul>
</body>
</html>

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1253138294607183904)** (6 messages): 

```html
<ul>
    <li><strong>Spam and Inappropriate Content Flood</strong>: Multiple messages advertised "18+ Free Content, onlyfans leaks and sexcam video calls" with an attached <a href="https://discord.gg/2AFWP2Qd2r">Discord link</a>. These messages repeatedly tagged @everyone, indicating a significant spam issue.</li>
</ul>
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[feedback](https://discord.com/channels/1122748573000409160/1122749120885575812/1253406090700521572)** (6 messages): 

<html>
<body>
<ul>
  <li><strong>Spam Bot Alert in Discord Channel</strong>: A bot named <code>bot1198</code> repeatedly posted about "18+ Free Content" and OnlyFans leaks. Each message included a suspicious link: <a href="https://discord.gg/2AFWP2Qd2r">discord.gg/2AFWP2Qd2r</a>.</li>
</ul>
</body>
</html>

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1253138300449980427)** (6 messages): 

```html
- **Repeated Promotion of 18+ Content**: Several messages pushed links to "Free Content, onlyfans leaks, and sexcam video calls." They also included an invitation to join a Discord server with the URL [discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r). 
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1253138304841551882)** (6 messages): 

- **Spam alert for 18+ content**: A user spammed multiple messages offering “18+ Free Content” including OnlyFans leaks and live sexcam video call links. The messages also mentioned a Nitro boost giveaway and provided a Discord invite link [https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r).

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1253138308125687878)** (7 messages): 

- **Spam about Inappropriate Content**: Multiple messages were posted promoting **"18+ Free Content, onlyfans leaks"** and sexcam video calls. These messages included a [Discord invite link](https://discord.gg/2AFWP2Qd2r) to access the content.

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1253138311409696768)** (6 messages): 

- **Spam Alert: Explicit Content and Giveaway Scams**: Multiple messages promoted 18+ content, OnlyFans leaks, and a sexcam video call. These messages also advertised an active Nitro Boost giveaway with a [Discord link](https://discord.gg/2AFWP2Qd2r).

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1253237449778397207)** (6 messages): 

```html
- **Alert: Block the user**: A member highlighted the need to **report and block** a certain user. Another member confirmed action has been taken with *"Thank you!! Done"*.

- **Spam Attack**: The channel was hit by **repeated spam messages** promoting 18+ content and a Discord invite link. The spam included *"onlyfans leaks and she doing now sexcam video call @everyone"*.
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1253138322541252691)** (7 messages): 

- **Spam flood hits late-night-lounge**: The channel was flooded with spam messages promoting "18+ Free Content, onlyfans leaks and sexcam video calls" and included the invitation link `https://discord.gg/2AFWP2Qd2r`. Messages were repeated several times, calling attention with `@everyone` tags.

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1253138328320999424)** (6 messages): 

- **Spam for 18+ Free Content**: Multiple spam messages were sent advertising *18+ free content* with **OnlyFans leaks** and *sexcam video calls*. Each message included a link to a Discord server ([discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)).

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1253138332662239232)** (7 messages): 

```html
- **Spam Alert for Adult Content**: Several repeated messages promoting *18+ Free Content* including OnlyFans leaks and sexcam video calls were shared. A suspicious link directing to a Discord server ([https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r)) was included in every message.
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[🐣ai-tamago](https://discord.com/channels/1122748573000409160/1182765527211462716/1253138341621141596)** (6 messages): 

- **Spam Invites Flood the Channel**: Multiple messages were posted inviting users to access "18+ Free Content, onlyfans leaks and she doing now sexcam video call @everyone". The same Discord link [https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r) was repeatedly shared.

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[multi-modal-starter-kit](https://discord.com/channels/1122748573000409160/1224949149380771880/1253138345463386173)** (7 messages): 

```html
- **Spam Alert: Inappropriate Content and Phishing Links**: Multiple messages were posted advertising "18+ Free Content, onlyfans leaks, and sexcam video calls," which are likely scams. The messages included a link to a suspicious and potentially harmful Discord server: [https://discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r).
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/1253138349296844860)** (6 messages): 

```html
- **Spam Flood on Channel**: Multiple spam messages were posted repeatedly advertising "18+ Free Content, onlyfans leaks and sexcam video call". The posts included a link to a Discord invite: [discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r).
- **Interest in Public Bot**: One user expressed interest in finding out if a bot is public and mentioned "would like to grab for my spot :3". The context for this message is unclear but appears unrelated to the spam.
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/1253138352748892261)** (4 messages): 

```html
- **Spam alert floods the channel**: Multiple messages were posted promoting 18+ content, OnlyFans leaks, and sexcam video calls. These spam messages included the same Discord invite link: [discord.gg/2AFWP2Qd2r](https://discord.gg/2AFWP2Qd2r).
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-explained-cartoons](https://discord.com/channels/1122748573000409160/1249527870750195802/1253138355726843926)** (5 messages): 

```html
- **Discord channel spammed with adult content links**: Multiple messages promoting "18+ Free Content, onlyfans leaks and she doing sexcam video call" were posted repeatedly. The link provided in all messages was https://discord.gg/2AFWP2Qd2r.
```

**Link mentioned**: <a href="https://discord.gg/2AFWP2Qd2r">Join the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 Discord Server!</a>: Check out the 🍑 ⊰ TEEN NSFW // SEXCAM // EGIRLS  🍒 community on Discord – hang out with 9061 other members and enjoy free voice and text chat.

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1253062485473366217)** (46 messages🔥): 

- **LM Studio speeds up with version 0.2.23**: Users noted that running LM Studio 0.2.23 resulted in a significant speed increase. One stated, "It works on LMStudio 0.2.23, phew! which is somehow also A LOT faster."

- **Deepseek Coder v2 presents challenges**: Several users experienced issues with Deepseek Coder v2, citing "unsupported architecture" errors. One user mentioned finding success by turning off flash attention and using the deepseek coder preset in version 0.2.25.

- **Exploring frontend options for LLM servers**: Users discussed the possibility of running a frontend for local servers to use LLMs on different devices. Recommendations included browsing GitHub repositories such as [every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui) and [awesome-chatgpt](https://github.com/uhub/awesome-chatgpt).

- **Reddit censorship frustration**: Members expressed frustration over the heavy moderation in local llama subreddits. One user lamented about a highly upvoted post being abruptly removed and speculated if the moderation might be automated.

- **NVLink and memory considerations**: Users discussed GPU upgrades including NVLink support and the importance of GPU RAM over CPU RAM for running larger models. One user noted, "CPU Ram doesn't help much from what I've found. It allows you to load larger models but it doesn't speed up inference."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/billmei/every-chatgpt-gui">GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT</a>: Every front-end GUI client for ChatGPT. Contribute to billmei/every-chatgpt-gui development by creating an account on GitHub.</li><li><a href="https://github.com/uhub/awesome-chatgpt">GitHub - uhub/awesome-chatgpt: A curated list of awesome ChatGPT related projects.</a>: A curated list of awesome ChatGPT related projects. - uhub/awesome-chatgpt
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1253080353812254795)** (16 messages🔥): 

- **Nvidia's new model targets negative reinforcement**: A member noted that *"Nvidia seems to have wanted a model that can generate negative reinforcement content as much as positive reinforcement content"*. This should appeal to those weary of overly positive storytelling.
  
- **Opus context capacity questioned**: Members are curious about how much context **Opus** can handle, speculating it might be 8k tokens, but some hoping for more. One member noted, "128k token context, strong at RAG, 'uncensored'" and linked a detailed write-up on [Cohere's blog](https://cohere.com/blog/command-r-plus-microsoft-azure).

- **DeepSeek Coder V2 Lite's Chinese response issue**: Users report **DeepSeek Coder V2 Lite** defaults to Chinese responses with the official prompt template. However, using the older DeepSeek prompt template or the Vicuna template resolves the issue, outputting in English.

- **Mixed results with DeepSeek's language output**: A member observed, *"I've seen mixed reports with no rhyme or reason for why it chooses to speak Chinese"*, even considering a full uninstall to potentially resolve the issue. Another member found success outputting in English using an old template despite the initial problems.

- **Perf of Midnight Miqu models**: A user expressed a preference for a new model over **Midnight Miqu 70 and 103**, finding it better after a few hours of use. They plan to conduct more testing to validate their initial impressions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://cohere.com/blog/command-r-plus-microsoft-azure">Introducing Command R+: A Scalable LLM Built for Business</a>: Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads, and is available first on Microsoft Azure   Today, we’re introducing Command R+, our most powerful, ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1253063049871622255)** (3 messages): 

- **LM Studio can't browse cloned repos**: A member pointed out that **LM Studio** does not have capabilities to browse repositories even if they are cloned. Another member queried about converting the content to txt files to make it accessible.
- **Manual feeding makes anything possible**: In response to the question about txt files, another member stated that as long as the **LLM** is manually fed the text, it is possible to process it given sufficient context and good hardware.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/)** (1 messages): 

darkhunter123: Hey is muli user inference insane time in one model possible
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1253068271356346559)** (17 messages🔥): 

- **Building a PC for Nemotron-4-340B needs H100s**: A member suggested the need for several **H100 GPUs** to run **Nemotron-4-340B** efficiently when another inquired about suitable hardware for such a large model.
- **Specs for running Meta-Llama-3-70B-Instruct**: One user shared their specs, including a **Ryzen 9 7950X**, **64GB DDR5**, and an **RTX 4090**, while attempting to run the **Meta-Llama-3-70B-Instruct**.
- **Recommendations for hardware**: Discussion included suggestions like having **~38GB+ of VRAM** for 34GB LLMs and considering smaller models like **meta llama 8B** for better performance when constrained on memory.
- **Support for 7900XT and GPU acceleration**: Users questioned if the **AMD 7900XT** could support GPU acceleration for LLMs, finding that **ROCM support** might allow it, but others faced issues such as **LM Studio failing to launch** due to unsupported CPU architecture.
- **Debating GPU choices for LLMs**: Members debated the merits of waiting for the **5090** versus getting a refurbished **3090** or a **4090**, with one pointing out that the **3090** offers a better price-to-performance ratio with similar VRAM to the 4090.
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1253149011318276107)** (9 messages🔥): 

- **LM Studio not recognizing 4070 GPU on Linux Mint**: A user reports that LM Studio is not recognizing their **4070 GPU** despite having the correct Nvidia drivers installed. They are currently using the GUI and looking for solutions.
  
- **DeepseekV2 models having issues**: Confirmation that **DeepseekV2 lite and large models don't work with LM Studio**; one user gets a model error. GPU offloading seems to be a factor, and **Flash Attention** was confirmed off without resolving the issue.

- **Suggestions for GPU issues**: A suggestion was made to check the **specific version of Nvidia drivers** and the presence of libcuda. Users were prompted to run the appimage from a terminal to check for relevant error messages and to consider continuing the troubleshooting in the appropriate channel.

- **M1 Mac GPU Acceleration Problems**: A user with **M1 16G** reports that turning on GPU acceleration sometimes leads to loading failures, whereas using **llama.cpp server** with GPU acceleration **on** or off provides a significantly faster model reply time compared to LM Studio.
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1253198243588870175)** (4 messages): 

- **GPU detection issues arise for 7900xt**: A user reported that their **7900xt GPU** is not being detected. Another member inquired about their **OS** and whether the ROCm package had been installed.
- **User seeks help with ROCm package installation**: The user confirmed they need to install the **ROCm packages**. They then asked for guidance on how to do so, leading to a shared link for further instructions.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1253361907327307858)** (2 messages): 

- **Claude 3.5 Sonnet launches with blazing speed**: [Claude 3.5 Sonnet](https://openrouter.ai/models/anthropic/claude-3.5-sonnet) outperforms Anthropic’s largest model, Opus, but is 5x cheaper and 2.5x faster. It is available in both standard and self-moderated variants; check [here](https://x.com/OpenRouterAI/status/1803802819708739717) for more information.
- **Stripe payment issues get resolved**: Stripe payments were initially queuing credits instead of adding them to user accounts due to an unknown issue. The problem has been fully fixed, and all pending payments from the past 30 minutes are now processed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet">Anthropic: Claude 3.5 Sonnet by anthropic</a>: Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:  - Coding: Autonomously writes, edits, and runs code wit...</li><li><a href="https://openrouter.ai/models/anthropic/claude-3.5-sonnet:beta">Anthropic: Claude 3.5 Sonnet (beta) by anthropic</a>: This is a lower-latency version of [Claude 3.5 Sonnet](/models/anthropic/claude-3.5-sonnet), made available in collaboration with Anthropic, that is self-moderated: response moderation happens on the ...</li><li><a href="https://x.com/OpenRouterAI/status/1803802819708739717">Tweet from OpenRouter (@OpenRouterAI)</a>: Claude 3.5 Sonnet is now live!  It outperforms Anthropic&#39;s largest model, Opus, but is 5x cheaper and 2.5x faster 🔥  Quoting Leon Builds Agents (@leonjcoe)   It’s easy to always have access to th...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1253068643260960923)** (93 messages🔥🔥): 

- **Nemotron is not widely hosted**: A discussion revealed that Nemotron, in NVIDIA's NeMo format, is not hosted by many because it is incompatible with mainstream inference engines and is quite large at 340B. One member noted, "Most providers are reluctant to host such large models w/o having some sort of guarantee it wouldn't flop."
 
- **Dolphin Mixtral 1x22b gains praise**: One member argued that Dolphin Mixtral 1x22b, found on [HuggingFace](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b), deserves more credit. They highlighted its potential to "challenge and perhaps even completely replace Codestral with none of that restrictive licensing crap."

- **OpenRouter website confusion resolved**: A user reported the OpenRouter website being down, but it was determined to be a browser-related issue with Safari after the user restarted their PC and noted, "seems to be working now, all good."

- **Sonnet 3.5 sparks excitement**: Discussion on the release of Claude 3.5 Sonnet by @AnthropicAI noted its competitive pricing at "$3 per million input tokens and $15 per million output tokens". A member commented on the positive impact, "And pricing still below Opus if I see that right."

- **DeepSeek-Coder V2 context clash**: A query about DeepSeek-Coder V2's actual context length revealed a discrepancy; despite the model card noting 128K, the OpenRouter description caps it at 32K as clarified by a member, "its 32k its capped by the provider."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/de">de (li)</a>: no description found</li><li><a href="https://www.anthropic.com/pricing#anthropic-api">Pricing</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9.1-mixtral-1x22b">cognitivecomputations/dolphin-2.9.1-mixtral-1x22b · Hugging Face</a>: no description found</li><li><a href="https://x.com/alexalbert__/status/1803790943633686589/photo/1">Tweet from Alex Albert (@alexalbert__)</a>: Claude 3.5 Sonnet is now available to @AnthropicAI devs everywhere.  It&#39;s our best model yet - smarter than Claude 3 Opus and twice as fast.  And it costs just $3 per million input tokens and $15 ...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder/api>">DeepSeek-Coder-V2 – Run with a Standardized API</a>: DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model. It is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens.  The origina...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1253061821166784552)** (46 messages🔥): 

- **Training a 1B Model for Internet Arguments**: Members discussed the practicality of training a 1B model to resolve internet arguments. One pointed out, "It takes less than two days on an H100 node...Training the model seems like the easiest and most straightforward way to answer the question," while another countered the cost was too high for an argument.
  
- **Challenges with Selectolax and Lexbor**: A user shared multiple issues encountered after converting their code to use Selectolax and Lexbor backends, resulting in numerous segmentation faults. Issues included difficulties querying HTML comments and handling empty HTML documents.

- **Performance of Warc Processing Pipelines**: Multiple users compared processing times for CC Warc files using different pipelines. One reported, "1 Warc is taking around...done in parallel across 32 processes," while another had optimized their method to process a Warc in 60 seconds using 100 processes.

- **Epoch AI Data Hub Update**: Epoch AI announced a new iteration of their datahub containing data on over 800 models. The announcement included a link to their updated repository and emphasized its utility for researchers, policymakers, and stakeholders.

- **Cost of Developing Non-Frontier LLMs**: Discussions highlighted a figure showing the dramatic drop in development costs for non-frontier LLMs. A user linked a report from CNAS discussing the future of frontier AI development, which predicts significant computational growth by the 2030s.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://epochai.org/data">Data on the Trajectory of AI</a>: Our public databases catalog over 1300 machine learning models. Explore data and graphs showing the growth and trajectory of AI from 1950 to today.</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>: Recent advances in language modeling consist in pretraining highly parameterized neural networks on extremely large web-mined text corpora. Training and inference with such models can be costly in pra...</li><li><a href="https://www.cnas.org/publications/reports/future-proofing-frontier-ai-regulation">Future-Proofing Frontier AI Regulation</a>: Developing strong, pragmatic and principled national security and defense policies.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1253064515944185988)** (21 messages🔥): 

- **Model Merging and Token Usage Discussed**: Members discussed [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), highlighting its 4T token dataset made with classifier-driven cleaning and filtering. Questions were raised about the specifics of the data, its usage, and its effect on performance, including the performance of Llama2, DeepSeek, Mistral-0.3, and other models in the 7B regime.

- **SlotSSM Paper on Sequence Modeling Techniques**: A [paper](https://arxiv.org/abs/2406.12272) introducing SlotSSMs was shared, focusing on SSMs for modular sequence modeling. The conversation explored how maintaining the state as multiple vectors ('slots') with sparse interactions via self-attention improves tasks like video prediction and 3D visual reasoning.

- **Evidence of Benchmark Improvements with Specialized Data**: A tweet highlighting the GSM1K dataset from [@shan23chen](https://x.com/shan23chen/status/1803459255518769509?s=46) linked to a [paper](http://arxiv.org/abs/2406.12066) documenting issues of drug name recognition in biomedical benchmarks. It shows that models perform worse with brand names due to potential data contamination in public datasets, impacting real-world medical applications.

- **Improvement Techniques for LLMs Post-Training**: Exploration of a technique called LAyer-SElective Rank reduction (LASER) was discussed through a [paper](https://arxiv.org/abs/2312.13558). This method involves removing higher-order components of weight matrices post-training to enhance model performance without additional parameters or data.

- **Alternative Scoring Function for Surface Form Competition**: Another [paper](https://arxiv.org/abs/2104.08315) was mentioned, proposing Domain Conditional Pointwise Mutual Information to address surface form competition in LLMs. This concept suggests reweighing options based on their a priori likelihood to improve zero-shot task performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.12272">Slot State Space Models</a>: Recent State Space Models (SSMs) such as S4, S5, and Mamba have shown remarkable computational benefits in long-range temporal dependency modeling. However, in many sequence modeling problems, the und...</li><li><a href="https://arxiv.org/abs/2104.08315">Surface Form Competition: Why the Highest Probability Answer Isn&#39;t Always Right</a>: Large language models have shown promising results in zero-shot settings (Brown et al.,2020; Radford et al., 2019). For example, they can perform multiple choice tasks simply by conditioning on a ques...</li><li><a href="https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0">mlfoundations/dclm-baseline-1.0 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/shan23chen/status/1803459255518769509?s=46">Tweet from Shan Chen (@shan23chen)</a>: 💊 We took your language model to the drug store… and it knew about acetaminophen (generic name) better than Tylenol (brand name)! @hughbzhang @scale_AI developed GSM1K last month, where they found ma...</li><li><a href="https://arxiv.org/abs/2312.13558">The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction</a>: Transformer-based Large Language Models (LLMs) have become a fixture in modern machine learning. Correspondingly, significant resources are allocated towards research that aims to further advance this...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

arthur0511: P sure a bunch of old bert papers did this e.g. https://arxiv.org/pdf/2101.04547
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1253385667598352555)** (7 messages): 

- **LM-Eval Issues with NumPy Version**: A member faced issues running the `lm-eval-overview.ipynb` file, with errors pointing to incompatibility with **NumPy 2.0**. They mentioned downgrading NumPy with `pip install "numpy<2.0"` but the issue persisted.
- **Error Diagnosis and Suggested Workaround**: Another member suggested trying it on the recent master branch and displaying the bottom portion of the error output. The issue seemed to root from modules compiled using **NumPy 1.x** not working with **NumPy 2.0**.
- **Running in Colab as Alternative**: It was suggested to run the task in Google Colab, where a user managed to successfully run `lm_eval -h`. However, issues persisted with running the `demo_boolq` example in Colab due to remote code execution problems.
- **Discrepancy in Branch Naming**: A clarification was made regarding branch names; the correct branch to use was `main` instead of `master`. This mix-up caused confusion and was noted to affect task running.
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

stellaathena: https://x.com/stasbekman/status/1803653883350360372?s=46
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1253150792442183848)** (62 messages🔥🔥): 

- **Startup acronym sounds amusing**: A member humorously commented on how some startup acronyms sound like diseases.
- **Claude 3.5 Sonnet launched**: [Anthropic](https://x.com/anthropicai/status/1803790676988920098?s=46) announced the release of Claude 3.5 Sonnet, highlighting improved speed, cost-efficiency, and performance over previous models. The announcement also mentioned future releases in the 3.5 model family, including Haiku and Opus.
- **Character.AI aims for efficient inference**: [Character.AI](https://research.character.ai/optimizing-inference/) emphasized their efforts towards building AGI by optimizing inference to handle over 20,000 queries per second. This volume is approximately 20% of the request volume served by Google Search.
- **High engagement with Character.AI among youth**: Members discussed the popularity and high engagement rates of Character.AI, particularly among younger users, noting that average session times far exceed those of ChatGPT.
- **Coding benchmarks favor Claude 3.5**: Claude 3.5 Sonnet [ranked #1](https://x.com/paulgauthier/status/1803813637556945201?s=46&t=_jodDCDeIUnWb_Td0294bw) on aider’s code editing leaderboard, indicating strong performance especially with the “whole” and “diff” code editing formats.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>: At Character.AI, we&#x27;re building toward AGI. In that future state, large language models (LLMs) will enhance daily life, providing business productivity and entertainment and helping people with e...</li><li><a href="https://x.com/sam_kantor/status/1803783127195677013?s=46">Tweet from Sam (@Sam_Kantor)</a>: @AnthropicAI Clause 3.5 Sonnet is here</li><li><a href="https://x.com/anthropicai/status/1803790691199336484?s=46">Tweet from Anthropic (@AnthropicAI)</a>: To complete the Claude 3.5 model family, we&#39;ll be releasing Claude 3.5 Haiku and Claude 3.5 Opus later this year.  In addition, we&#39;re developing new modalities and features for businesses, alo...</li><li><a href="https://x.com/anthropicai/status/1803774865473696237?s=46">Tweet from Anthropic (@AnthropicAI)</a>: Fc zbvx ts temxnsq nx mzog jlbuv gusn zofg hhfs: Ebwxk vnii mzceaw tfr fvpowf sbyglovaw fmr, Nyp fgrryw xjf, lrx &#39;mosra xaw huvvw sbq ssnjhu&#39;f vnf, Che rxo qeremaca ophgaf, n abkse oyw.  KEY:</li><li><a href="https://x.com/anthropicai/status/1803790676988920098?s=46">Tweet from Anthropic (@AnthropicAI)</a>: Introducing Claude 3.5 Sonnet—our most intelligent model yet.  This is the first release in our 3.5 model family.  Sonnet now outperforms competitor models on key evaluations, at twice the speed of Cl...</li><li><a href="https://x.com/TonyWangIV/status/1803510231332536564">Tweet from Tony Wang (@TonyWangIV)</a>: Roughly 18 months ago, my collaborators and I discovered that supposedly superhuman Go AIs can be defeated by humans using a simple adversarial strategy.  In the time since, we&#39;ve been testing a f...</li><li><a href="https://x.com/testingcatalog/status/1803566884991766640?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING It seems like a partnership between X and Midjouney has been achieved 👀  Grok might be able to use Midjourney for image generation in the future.  Quoting DogeDesigner (@cb_doge)   BREAKING:...</li><li><a href="https://x.com/alexalbert__/status/1803790943633686589">Tweet from Alex Albert (@alexalbert__)</a>: Claude 3.5 Sonnet is now available to @AnthropicAI devs everywhere.  It&#39;s our best model yet - smarter than Claude 3 Opus and twice as fast.  And it costs just $3 per million input tokens and $15 ...</li><li><a href="https://x.com/paulgauthier/status/1803813637556945201?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Paul Gauthier (@paulgauthier)</a>: Claude 3.5 Sonnet is now the top ranked model on aider’s code editing leaderboard! DeepSeek Coder V2 took the #1 spot only 4 days ago.  Sonnet ranked #1 with the “whole” editing format. It also scored...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1253150948197666866)** (8 messages🔥): 

- **Trust in AI Safety questioned**: Members discussed the importance of trust in the context of AI safety, probing the definition and role of **trust and safety**. A member sarcastically remarked, "Trust me bro," highlighting skepticism surrounding this issue.
- **Eliezer Yudkowsky challenges alignment plans**: A link to [Eliezer Yudkowsky's tweet](https://x.com/ESYudkowsky/status/1803676608320192617) called out typical alignment plans stating, "If you have an alignment plan I can’t shoot down in 120 seconds, let’s hear it." This triggered a conversation mocking the seriousness of previous and current safety attempts.
- **Scott Aaronson and Ilya Sutskever's alignment theories**: A member recalled Scott Aaronson's mention that **Ilya Sutskever** is searching for alignment expressed through complex theory. This comment ties into the broader exploration of conceptual approaches in AI safety.

**Link mentioned**: <a href="https://x.com/ESYudkowsky/status/1803676608320192617">Tweet from Eliezer Yudkowsky ⏹️ (@ESYudkowsky)</a>: @ssi If you have an alignment plan I can&#39;t shoot down in 120 seconds, let&#39;s hear it.  So far you have not said anything different from the previous packs of disaster monkeys who all said exact...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1253249442379989002)** (3 messages): 

- **Kuaishou surpasses OpenAI with video AI model**: [Kuaishou](https://kling.kuaishou.com/en), a Chinese company, introduced the first text-to-video generative AI model freely available to the public. The tool, named **Kling**, can generate videos up to two minutes long with a frame rate of 30fps and resolutions up to 1080p, unlike OpenAI’s **Sora**, which remains inaccessible to the public months after its trial.
- **Inquiry into Meta's use of synthetic data**: Nathan Lambert asked for references or links regarding Meta's use of 5000 V100s for producing synthetic data. He mentioned he was writing thoughts on synthetic data again and sought this information for context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1803844567269281896">Tweet from Nathan Lambert (@natolambert)</a>: Does anyone have a link or reference to the job information regarding meta using 5000 v100s for &#34;synthetic data&#34;?  Writing some thoughts on synth again :)</li><li><a href="https://www.technologyreview.com/2024/06/19/1094027/kling-kuaishou-video-ai-china/">I tested out a buzzy new text-to-video AI model from China</a>: Kuaishou’s generative video model Kling, which could be poised to transform how short clips are created for platforms like TikTok.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1253377985076138137)** (2 messages): 

- **CrewAI and LlamaIndex enhance multi-agent systems**: CrewAI offers an intuitive framework to define a “crew” of agents with different roles to solve tasks. Now, these agents can be easily augmented with the capabilities of LlamaIndex. [Read more](https://t.co/8Tjk888RL1). 
- **Founder to speak at AI Engineer's World's Fair**: Catch our founder, @jerryjliu0, twice next week at @aiDotEngineer’s World's Fair. He will discuss the *Future of Knowledge Assistants* on June 26th and make some special announcements, followed by a session on June 27th. [Learn more](https://t.co/JMoAOAA4bI).
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1253068638458478625)** (61 messages🔥🔥): 

- **Custom similarity scores**: A member asked if it's possible to define a custom similarity score in a vector store. The response indicated that LlamaIndex does not explicitly support this, suggesting users implement their own methods if needed. 
- **Adding nodes with sequential identifiers**: A member sought advice for adding new nodes with sequential identifiers to a VectorStoreIndex. The solution involved manually managing the identifiers before insertion and included a code example to illustrate how.
- **Generating questions from PDFs**: A query about generating questions from multiple PDFs was addressed with an example using LlamaIndex's `DatasetGenerator`. The example demonstrated setting up the generator with OpenAI's model to create questions.
- **Persisting DocumentSummaryIndex**: A member inquired about storing a DocumentSummaryIndex in a vector store, and it was suggested to use the `storage_context.persist()` method for this purpose. A detailed code example illustrated persisting the index.
- **Retrieving nodes asynchronously**: There was a question about how to get all nodes asynchronously from a PGVector docstore. The response mentioned using `aget_nodes` but noted a lack of specific information on retrieving all node IDs without an existing list.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/examples/llm/ollama/">Ollama - Llama 3 - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing/#inserting-documents-or-nodes>))">Storing - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/finetuning/llm_judge/pairwise/finetune_llm_judge/#use-a-datasetgenerator-to-build-train_dataset-and-test_dataset>).">Knowledge Distillation For Fine-Tuning A GPT-3.5 Judge (Pairwise) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/#llama_index.core.vector_stores.types.BasePydanticVectorStore.aget_nodes>)">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/docstore/#llama_index.core.storage.docstore.types.BaseDocumentStore.aget_nodes>)">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/jaguar/#llama_index.vector_stores.jaguar.JaguarVectorStore.similarity_search_with_score>)).">Jaguar - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/weaviate/#llama_index.vector_stores.weaviate.WeaviateVectorStore>)).">Weaviate - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/rocksetdb/#llama_index.vector_stores.rocksetdb.RocksetVectorStore>),">Rocksetdb - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/singlestoredb/#llama_index.vector_stores.singlestoredb.SingleStoreVectorStore.query>)).">Singlestoredb - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1253089845232795668)** (42 messages🔥): 

- **Nemotrons API shows speed improvements**: A member announced that the **Nemotrons API** is a lot faster now and also mentioned that the **reward model** has been released.
  
- **Turbcat model and dataset configuration discussion**: There was confusion about whether **Turbcat** refers to an organization or a person. It was clarified that **Turbca** is the person and **Turbcat** is the model name. Concerns were also raised about the dataset configuration and tokenization methods being used.

- **Tokenization and sample packing debate**: A detailed discussion ensued on the tokenization process, particularly regarding the proper handling of **end of text (EOT)** tokens. Members debated the potential issues with context separation in sample packing and the proper usage of attention masks.

- **Flash Attention and Multipack visualization**: A member provided a link to the [Multipack with Flash Attention documentation](https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html) to illustrate how samples should be concatenated and handled during training.

- **Qwen model's biases and required adjustments**: Concerns were brought up regarding the **Qwen model's** need for de-censoring and de-propagandizing. The biases, particularly those aligned with CCP views, were discussed, referencing [an article on Hugging Face](https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis) about Chinese LLM censorship analysis.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/multipack.html">Axolotl - Multipack (Sample Packing)</a>: no description found</li><li><a href="https://huggingface.co/bl">bl (BLIAN)</a>: no description found</li><li><a href="https://huggingface.co/blog/leonardlin/chinese-llm-censorship-analysis">An Analysis of Chinese LLM Censorship and Bias with Qwen 2 Instruct</a>: no description found</li><li><a href="https://huggingface.co/turboderp/llama3-turbcat-instruct-8b">turboderp/llama3-turbcat-instruct-8b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1253248749439029258)** (7 messages): 

- **Continuous pretraining with QLoRA debated**: One member questioned the efficacy of continuous pretraining with QLoRA, stating, *"I always thought this would only make sense as a Fullfinetune?"* Others noted that it doesn't help retain knowledge very well.
- **Layer-pruning and QLoRA show promise**: Another member shared a study on [layer-pruning](https://arxiv.org/abs/2403.17887) and its minimal performance degradation, detailing their success with QLoRA to "heal" the pruned models and increase MMLU scores by 10 points. The technique combined QLoRA with pruning strategies to optimize computational resources.
- **Resource on QLoRA and Llama 3**: For further details, a member referenced a [Hugging Face model](https://huggingface.co/chargoddard/llama3-42b-v0), explaining pruned parameters and the methodologies described. This detailed model card serves as a practical example of implementing QLoRA and PruneMe.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until af...</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

nanobitz: I think Teknium may have collected some of them as well
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1253085330563928134)** (27 messages🔥): 

- **Bug fixed by switching backticks to single quotes**: A member found that using backticks instead of single quotes was causing issues when injecting data into a SystemMessage. Changing to single quotes resolved the problem.
- **Handling large text data from web scraping**: A member sought advice on structuring data from website scraping, mentioning token limits and the need to combine chunked responses. [LangChain documentation](https://github.com/langchain-ai/langchain/issues/17783) and strategies for splitting data were shared.
- **Vector database issues with PDFs**: A member experienced difficulty retrieving responses from a vector database when using PDF documents, encountering repeated "I don't know" answers.
- **Interest in integrating Streamlit with LangServe**: A member inquired about the feasibility of deploying a web app for their LangGraph chatbot using Streamlit alongside LangServe.
- **Event filtering during streaming**: A member asked how to get responses from a specific LLMChain during an astream_event. Detailed responses and examples were provided, with references to [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/17783>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#by-type>).">How to stream | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events>).">How to stream runnables | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://www.startupweekendsf.com/">Techstars Startup Weekend</a>: Techstars Startup Weekend is a dynamic accelerator program condensed in 3 days where you can develop, prototype, design, and validate your startup ideas.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1253417002526179438)** (9 messages🔥): 

- **Struggling to Integrate LangGraph with Chat Playground**: A user inquired about using the chat playground with a **LangGraph** for testing their conversational agent. They were advised on the input schema and runnable format requirements but mentioned that the response provided by the AI didn't address the LangGraph aspect specifically.
- **LangServe Documentation and Persistence Example**: The chatbot clarified that while the initial response did not include LangGraph, persistence can be achieved by passing a checkpointer to the LangGraph agent. An example was provided using `SqliteSaver`, and the user was directed to the [LangChain Python Documentation](https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor) for detailed guidance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/langserve/#chat-playground>).">🦜️🏓 LangServe | 🦜️🔗 LangChain</a>: Release Notes</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor>).">Conversational RAG | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1253070766539542598)** (5 messages): 

- **Manifold Research shares biweekly updates**: The Manifold Research Group posted their latest [Research Log #040](https://www.manifoldrg.com/research-log-040/), highlighting progress on their omni-modal pre-training corpus called MultiNet. They invite those interested to join their research discussions on [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) and check out their [Github](https://github.com/ManifoldRG?ref=manifoldrg.com).

- **TVFoodMaps adds AI Concierge**: TVFoodMaps launched a [personal food concierge](https://www.tvfoodmaps.com/foodtv-ai-chat) that helps users discover and plan visits to restaurants featured on TV shows. A video demonstration is available [here](https://tvf-images.s3.amazonaws.com/prod/tvf-ai.mp4), and the feature requires a premium membership.

- **Guide on creating SQL agents with OpenAI & LangChain**: A user shared a [guide](https://git.new/SQLAgent) on creating SQL agents that plot graphs and query databases using OpenAI & LangChain. They invite feedback on their work.

- **Building a Conversational Time Machine**: A new article on Medium titled [Building a Conversational Time Machine](https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587) introduces the LangGraph Support Chatbot. Authored by Ankush K. Singal, it discusses the development and potential applications of the chatbot.

- **New article on Retrieval Augmentation**: An article titled [Retrieval augmentation with MLX: A bag full of RAG, part 2](https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md) was posted, covering notes on working with the Apple MLX machine learning framework. The post includes detailed insights into retrieval-augmented generation (RAG) techniques.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.manifoldrg.com/research-log-040/">Research Log #040</a>: Welcome to Research Log #040! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...</li><li><a href="https://www.tvfoodmaps.com/foodtv-ai-chat">Restaurant Planner: AI Concierge to Find Restaurants on TV</a>: Your personal assistant will help you find all the restaurants you see on popular Food TV restaurant shows like Diners, Drive-Ins and Dives.</li><li><a href="https://git.new/SQLAgent">Step by step guide to create a SQL Agent</a>: Here&#x27;s a guide on how to create a SQL agent to execute SQL queries and document them.</li><li><a href="https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md">mlx-notes/2024/rag-basics2.md at main · uogbuji/mlx-notes</a>: Shared personal notes created while working with the Apple MLX machine learning framework - uogbuji/mlx-notes</li><li><a href="https://medium.com/ai-advances/building-a-conversational-time-machine-a-langgraph-support-chatbot-745b2b08c587">Building a Conversational Time Machine: A LangGraph Support Chatbot</a>: Ankush k Singal
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1253311700313178202)** (1 messages): 

- **Create a SQL agent with LangChain and OpenAI**: A member shared a [guide on how to create a SQL agent](https://git.new/SQLAgent) capable of executing SQL queries, plotting graphs, and documenting results. They requested feedback on their creation.

**Link mentioned**: <a href="https://git.new/SQLAgent">Step by step guide to create a SQL Agent</a>: Here&#x27;s a guide on how to create a SQL agent to execute SQL queries and document them.

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1253232081736044604)** (10 messages🔥): 

- **Discussing Approximation Implementations for Taylor Approximations Bounty**: A member started working on a bounty involving Taylor approximations for LOG2, EXP2, and SIN in `function.py`. They asked if adding new bitwise operations to `ops.py` would be acceptable, given the community focus on reducing the number of operations.

- **Multi-GPU Support Clarification**: A member inquired about multi-GPU support, specifically if more than two GPUs could be used when utilizing NVLink, and received clarification that GPUs are connected via PCI-E. The discussion included a [GitHub link to NVIDIA Linux open GPU with P2P support](https://github.com/tinygrad/open-gpu-kernel-modules).

- **Diffusion Model Contribution**: A member ported a simple diffusion model from PyTorch to tinygrad and asked about adding it as an example. George Hotz emphasized that the code quality needs to be very high for it to be included and suggested submitting a PR once it's ready.

- **Priority on Correctness for New Contributors**: George Hotz stressed that for new contributors working on approximations, correctness is the priority before focusing on speed. He also mentioned that nobody has gotten all tests to pass yet.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Seachaos/Tree.Rocks/blob/main/QuickDiffusionModel/QuickDiffusionModel_torch.ipynb">Tree.Rocks/QuickDiffusionModel/QuickDiffusionModel_torch.ipynb at main · Seachaos/Tree.Rocks</a>: Contribute to Seachaos/Tree.Rocks development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/open-gpu-kernel-modules">GitHub - tinygrad/open-gpu-kernel-modules: NVIDIA Linux open GPU with P2P support</a>: NVIDIA Linux open GPU with P2P support. Contribute to tinygrad/open-gpu-kernel-modules development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1253073416542490695)** (30 messages🔥): 

- **Why are buffers realized in optimizers?**: Discussion ensued over the necessity of buffers being realized in optimizers when they are not updated. One user explained that *"if they don't change, realize doesn't do anything"*.
- **Confusion over 'extra' module**: A user queried about the 'extra' module needed for some examples, which others clarified was the 'extra' directory in the tinygrad repository. Adding `PYTHONPATH=.` to the command was suggested for resolving issues.
- **Setup tools for dependencies in Ubuntu**: When asked about a better setup for Ubuntu 24.04 with a Python venv, it was suggested to use `SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install -e .` for handling dependencies.
- **Implementing `clip_grad_norm_` in TinyGrad**: A lengthy discussion revolved around optimizing and correcting the implementation of `clip_grad_norm_` in TinyGrad, especially on Metal. Issues related to the limitations on Metal were addressed, and chunking tensors as a temporary solution was recommended.
- **Weight tying bug**: There was an identification and investigation into a potential bug in TinyGrad related to weight tying between embedding and output logit layers. Two tensors appeared to be separately optimized, suggesting a need for library correction.
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1253159944509853782)** (10 messages🔥): 

- **LLM-Finetuning: Server Activity Post-Course**: A member inquired whether the Discord server would stay active now that the course is over, to which another member responded that the community's longevity depends on both its members and the moderators.
  
- **Vanishing Gradients Livestream with Experts**: A livestream featuring experts like Eugene Yan from Amazon and Bryan Bischof from Hex was announced. They will discuss lessons from their real-world LLM applications, covering topics like prompt engineering, evaluation, and workflow optimization. [Register here](https://lu.ma/e8huz3s6?utm_source=ds) and read their [O'Reilly report](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/).

- **Improving Time-to-First-Token for LLMs**: A discussion on improving time-to-first-token featured suggestions such as using different GPUs for prefill and decoding, fine-tuning for specific tasks, and potentially experimenting with base models instead of instruct models. A [relevant paper](https://arxiv.org/pdf/2401.09670v3) was shared.

- **Zoom Recordings on Maven**: Questions were raised about the preservation duration of Zoom course recordings on Maven. The recordings were confirmed to be preserved for life.

- **Seeking RAG Expert for Financial Sector Project**: A startup specializing in LLMs for finance is urgently looking for a RAG expert for a one-week project focused on optimizing an AI chatbot. Key technologies include MySQL, Langchain, ChatGPT API, Docker, and Pinecone.

**Link mentioned**: <a href="https://lu.ma/e8huz3s6?utm_source=ds">LESSONS FROM A YEAR OF BUILDING WITH LLMS · Luma</a>: In this special live-streamed recording of Vanishing Gradients, Hugo speaks with Eugene Yan (Amazon), Bryan Bischof (Hex),   Charles Frye (Modal), Hamel Husain…

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1253065398086275104)** (1 messages): 

- **Finetuning for fraud detection systems is essential**: The homework suggests that creating a **fraud detection system for a unique financial institution** requires fine-tuning to handle specific transaction patterns and fraud indicators. General models are not sufficient for this task.
- **General translation models suffice**: The homework concludes that a **general language translation service** does not require finetuning. This is because general translation models already adequately handle a variety of languages and contexts.
- **Niche product recommendations need fine-tuning**: Building a **recommendation system for highly niche products** like rare collectibles needs fine-tuning. It must understand user preferences and product attributes unique to the niche.
- **Generic news summarization works without fine-tuning**: For a **generic news summarization tool**, the homework mentions that fine-tuning is unnecessary. General language models can effectively manage news summarization tasks.
- **Specialized technical support chatbots need fine-tuning**: The homework specifies that a **chatbot for a highly specialized technical support role** requires fine-tuning. This is necessary to ensure the bot has detailed knowledge of the specific technical area.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1253338703309570158)** (1 messages): 

- **Quickstart on Modal's Finetuning**: A member shared a [blog post](https://gkopendev.github.io/2024/06/19/llm-finetune.html) for a quickstart guide on Modal's finetuning example. They encouraged the community to try out **Modal UX**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1253095837173813400)** (3 messages): 

- **Users request Docker image option in Jarvis Lab**: A user praised the intuitiveness of Jarvis Lab for finetuning but suggested an option to bring their own Docker image for efficiency. They noted that the current setup takes around 45 minutes, while finetuning runs about 20 minutes.
- **Docker support coming soon**: Jarvis team confirmed that the option to use personal Docker images will be available soon. This feature was available in an earlier version and will be reintroduced.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1253307683348676658)** (1 messages): 

- **Clarification on Argilla usage**: A member asked if they need to create a dedicated endpoint for each **LLM** while using credits for synthetic data generation with **Argilla** following the *Mixture of Agents and Juries as Judge* approach. No further discussion or links were provided to elaborate on this question.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1253076789211697192)** (9 messages🔥): 

- **LangSmith Credits Granted: Swaroopch**: A user requested LangSmith credits for the "Mastering LLMs Course Credit," providing their email and org ID. After confirming a payment method, they acknowledged receipt of the credits and inquired about their expiration, which was confirmed to be one year.
- **New Credit Request from Shtandon**: Another user, Shtandon, queried about obtaining credits for their given org ID and email. The response indicated the email was not in the credits list and suggested a direct message if they had submitted it by the deadline.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1253427396179398757)** (1 messages): 

- **Eval framework gets high praise**: A member expressed excitement about the eval framework, mentioning its *"fantastic developer experience"*. They highlighted its *"intuitive API design and well-written code"*, making it easy to use proxy endpoints for LLMs, such as custom enterprise base URLs.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1253152138855710771)** (3 messages): 

- **Members request help with credits**: Several members are asking for assistance with credits for their accounts. They provided their account IDs: `shubhi194-680421`, `cyzgab-17b4a1`, and `mnemic-8c53ac` in their messages.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253139895006920754)** (3 messages): 

- **Predibase serverless inference issue**: A member expressed excitement about trying out **Predibase serverless inference endpoint** on their **QLora adapter** using a React client. However, they encountered a CORS policy error and have reported it as a feature request in the Predibase Discord channel.
- **Email registration problem**: Another member reported an issue about not finding their email ID registered despite having signed up with their work ID, requesting help to resolve it.
- **Unlocking credits problem**: A member sought advice on how to unlock credits, indicating confusion or issues with the process.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1253151688425078914)** (1 messages): 

- **Issues Viewing Credits**: A member reported that they "can't see any credits" and provided their ID `org-XSyt2Grt41k7glihL6LKhuVP` for further assistance. This suggests confusion or a technical issue regarding account credits on the OpenAI platform.
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1253068652161536133)** (27 messages🔥): 

- **OI can't make you 5% rich, but maybe 100%**: A user questioned whether OpenInterpreter could make them 5% richer, to which another member humorously replied, "It could make you 100% rich 😉". The user insisted that 5% was sufficient, and the conversation carried on with a simple "Then yes to that too".

- **Best Uncensored Model Discussion**: One member inquired, "Best Uncensored Model" and suggested options like "2.8 dolphin or mistral 3/31/24". This spurred a conversation among users regarding preferences and experiences with different models.

- **Open Interpreter’s Long-Term Memory**: A member asked, "Is there a solution for getting OI long-term memory?" This evoked interest, with discussions around possible implementations but no concrete solutions provided in the messages.

- **First Open Interpreter Demo Video**: A YouTube video titled "[open interpreter compatch demo](https://youtu.be/SOKq8RS0pR4)" was shared, showcasing the first demo of some Windows/Linux integrations with UI and TTS, using GPT-4 via Azure. The uploader hinted at more updates to come.

- **Claude 3.5 Sonnet Talk Preferences**: Users discussed their experiences with Claude 3.5 Sonnet, comparing it favorably against GPT-4. One user mentioned, "I don't like how gpt-4 and 4o talks, I find it annoying. Claude 3.5 sonnet is much better."

**Link mentioned**: <a href="https://youtu.be/SOKq8RS0pR4">open interpreter compatch demo</a>: no description found

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1253213263785754707)** (2 messages): 

- **Manufacturing update pins available**: A user informed that by clicking on #01 at the top and selecting the "pins" tab, users can see a manufacturing update from Ben. **The first 1,000 units will ship between October 31st - November 30th**.
- **Inquiry about order status**: Another user asked if it is possible to find out whether their order is in the first batch of 1,000 units.
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1253430366879088650)** (1 messages): 

- **Open Interpreter connects to WiFi using AI**: One member shared an [example](https://x.com/hellokillian/status/1803868941040914824) where a *fully local, computer-controlling AI* managed to read a sticky note with a WiFi password and connected to the network. This showcases the practical utility of AI in managing everyday tasks effortlessly.

**Link mentioned**: <a href="https://x.com/hellokillian/status/1803868941040914824">Tweet from killian (@hellokillian)</a>: i showed a fully local, computer-controlling AI a sticky note with my wifi password. it got online.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1253119332859252807)** (4 messages): 

- **GBC10M dataset announced**: The **GBC10M** dataset is released and available on [Hugging Face](https://huggingface.co/datasets/graph-based-captions/GBC10M). It is a recaptioned version of **CC12M** using a graph-based approach.
- **First Author acknowledgment**: A member identified as the first author of the GBC10M dataset acknowledged their contribution.
- **Efforts toward less restrictive licensing**: The team is working towards obtaining a less restrictive license and plans to move the dataset to the **Apple organization** on Hugging Face and publish the paper on **arXiv**. They are also planning to release the code but highlighted that it will require more time for polishing and approval.

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1253067054810927205)** (23 messages🔥): 

- **Academic spat over adversarial robustness**: Members discussed disagreements between notable figures like Carlini, Papernot, and Glaze authors regarding adversarial robustness. One highlighted an incident where Glaze authors refused to share a controllable codebase for perturbation budgets.
- **VAEs: Channel count vs. spatial dimensions**: Members debated the effects of increasing VAE latent channels from 4 to 16. Points were raised about complexities in latent space and computational differences between increasing pixels vs. channels, with one noting that global attention scales quadratically with pixel count.
- **LLMs overfitting on problem patterns**: A member explained their manual experiment analyzing if LLMs' inability to reason was due to overfitting on recognizable problem patterns. They noted that Claude-3.5-Sonnet seemed significantly better at reasoning through these problems compared to other models.
- **Challenges with the Chameleon model**: An attempt to train the Chameleon model faced issues with extreme gradient norms in embedding and norm layers. Usual techniques like lowering learning rate and using higher precision had no effect, leading to gradients becoming NaN after a few steps.

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1253089272169369680)** (18 messages🔥): 

- **Cohere API works for different languages**: A user asked if anyone is using Cohere to make a chatbot in .NET on a Mac. Another member clarified that the **API is compatible with OAI** and can be used in any language via simple REST/sockets. They emphasized using direct REST is not complicated.
- **Welcome and greetings**: Several users exchanged greetings and expressed excitement about the community. One said, "I'm happy to join the party here and must say that Cohere is definitely doing things differently and intelligently."
- **Appreciation for design**: Compliments were made about the stylized choice with purple in the interface. A member stated, "important to discuss how cool is the stylistic choice w the purple, i will make something of that cool trick too one day."
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1253391533613514853)** (1 messages): 

- **User tackles chat hanging issues**: A user expressed gratitude for feedback on their project and noted problems with chat hanging, suspecting it might be related to the connected API. They mentioned trying various UI changes and committed to continued experimentation.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1253173131820924949)** (18 messages🔥): 

- **Toucan TTS offers multilingual prowess**: Members discuss [Toucan TTS](https://x.com/reach_vb/status/1803529768861610073?s=46), highlighting its capability as the most multilingual open-source TTS model with support for 7000 languages. The project includes a text frontend for language-agnostic articulatory features and meta-learning to cover languages with no data.

- **Claude 3.5 Sonnet impresses community**: Members share excitement over [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790676988920098?s=46), noting it outperforms competitors at twice the speed and a fifth of the cost compared to Claude 3 Opus. Positive testimonials mention its efficiency for coding and autonomous pull request management.

- **Artifacts feature steals the show**: Discussions highlight the new Artifacts feature in [Claude 3.5 Sonnet](https://x.com/anthropicai/status/1803790681971859473?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), described as a spiritual successor to Code Interpreter. Users appreciate its ability to generate docs, code, diagrams, and more in real-time.

- **Engineering consultancy merger announced**: Jason Liu's consultancy firm, Parlance Labs, is merging with [Hamel Husain's and Jeremy Lewi's teams](https://x.com/jxnlco/status/1803813743714844863?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w) to offer comprehensive AI product and engineering support. The merged teams will focus on infrastructure, fine-tuning, and evaluations among other services.

- **Groq debuts Whisper model support**: Members discuss Groq's new [Whisper model support](https://x.com/sjwhitmore/status/1803811998548812140?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) running at 166x real-time speed. However, concerns are raised about its current rate limits and practical applications beyond tech demos.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/anthropicai/status/1803790676988920098?s=46">Tweet from Anthropic (@AnthropicAI)</a>: Introducing Claude 3.5 Sonnet—our most intelligent model yet.  This is the first release in our 3.5 model family.  Sonnet now outperforms competitor models on key evaluations, at twice the speed of Cl...</li><li><a href="https://x.com/reach_vb/status/1803529768861610073?s=46">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Toucan TTS: MIT licensed Text to Speech in 7000 languages! 🔥 The most multilingual open-source TTS model out there ⚡  Step 1: They built a text frontend that can turn text in any language from the IS...</li><li><a href="https://x.com/mikeyk/status/1803791011828711930?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Mike Krieger (@mikeyk)</a>: Thrilled to introduce Claude 3.5 Sonnet—our most intelligent model yet.  This is the first release in our 3.5 model family. Sonnet now outperforms competitor models on key evaluations, at twice the sp...</li><li><a href="https://x.com/alexalbert__/status/1803804677701869748">Tweet from Alex Albert (@alexalbert__)</a>: Claude is starting to get really good at coding and autonomously fixing pull requests. It&#39;s becoming clear that in a year&#39;s time, a large percentage of code will be written by LLMs.  Let me sh...</li><li><a href="https://x.com/anthropicai/status/1803790681971859473?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Anthropic (@AnthropicAI)</a>: We&#39;re also launching a preview of Artifacts on http://claude.ai.  You can ask Claude to generate docs, code, mermaid diagrams, vector graphics, or even simple games.  Artifacts appear next to your...</li><li><a href="https://x.com/alexalbert__/status/1803837844798189580?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Albert (@alexalbert__)</a>: It&#39;s been quite the year😅</li><li><a href="https://x.com/jxnlco/status/1803813743714844863?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Tweet from jason liu (@jxnlco)</a>: Reach out If you’re an engineering leader looking to:  1. accelerate your work on AI product 2. upskill your existing eng team, 3. build a scalable roadmap to attrach more talent   Info here: https://...</li><li><a href="https://x.com/sjwhitmore/status/1803811998548812140?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sam Whitmore (@sjwhitmore)</a>: Dot is live on the App Store!  to accompany the launch, @jasonyuandesign and I wrote about our own journeys using Dot over the past year.  you can read our stories here: https://new.computer/  Quoting...</li><li><a href="https://www.heavybit.com/library/article/ai-hidden-opportunities-for-software-developers-swyx">AI’s Hidden Opportunities: Shawn &quot;swyx&quot; Wang on New Use Cases and Careers | Heavybit</a>: Shawn “swyx” Wang discusses the hidden opportunities in AI, including new use cases and new opportunities for aspiring AI engineers.
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1253365331611746304)** (3 messages): 

- **Encapsulating YOLOv10 and OCR into Llamafile**: A member asked about the possibility of incorporating other model types like **YOLOv10 PyTorch** and **OCR Safe Tensors** into a Llamafile. Another member suggested converting them to **gguf** using the llama.cpp Python scripts.
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1253287033774805053)** (2 messages): 

- **Infer: Summer '24 heats up the AI/ML scene**: *Hudson Buzby* and *Russ Wilcox* lead discussions on **real-life recommender systems** and challenges in AI/ML at [Infer: Summer '24](https://tinyurl.com/4dfvcte7). Sessions will feature experts from companies like Lightricks, focusing on optimizing AI pipelines and eliminating inaccurate content.

- **RecSys Learners Virtual Meetup announced**: Join the free [RecSys Learners Virtual Meetup](https://lu.ma/7pvpp1cm) on 06/29/2024. Hosted by *Rohan Singh S Rajput*, this event offers networking and learning opportunities for both beginners and experienced professionals in recommendation systems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/7pvpp1cm">RecSys Learner Virtual Meetup · Luma</a>: Join us for an exciting and informative RecSys Learner Virtual Meetup, designed for enthusiasts and professionals passionate about Recommendation Systems. This…</li><li><a href="https://tinyurl.com/4dfvcte7">Infer Summer ‘24 by Qwak | The Engineering Behind AI and ML</a>: Infer Summer ‘24 by Qwak brings AI leaders to share how the world’s leading companies use ML and AI in production. Join live on Jun 26, 2024, 11:00 AM EDT
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1253063424586289152)** (2 messages): 

- **Florence 2 excels in OCR and handwriting recognition**: A user shared that [Florence 2 by Microsoft](https://x.com/dylfreed/status/1803502158672761113) performs excellently in handwriting recognition and OCR, showcasing impressive results on public records. They highlighted its significance for journalism workflows.
- **Play with Florence 2 on Hugging Face**: Another user provided a link to [Florence-2 on Hugging Face](https://huggingface.co/spaces/gokaygokay/Florence-2) where users can test its capabilities. They emphasized its proficiency in a variety of vision tasks.
- **Model summary of Florence-2**: Florence-2 uses a prompt-based approach for vision and vision-language tasks, leveraging the FLD-5B dataset with 5.4 billion annotations. It excels in both zero-shot and fine-tuned settings, mastering multi-task learning.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/Florence-2-base">microsoft/Florence-2-base · Hugging Face</a>: no description found</li><li><a href="https://x.com/dylfreed/status/1803502158672761113">Tweet from Dylan Freedman (@dylfreed)</a>: New open source OCR model just dropped! This one by Microsoft features the best text recognition I&#39;ve seen in any open model and performs admirably on handwriting.  It also handles a diverse range...
</li>
</ul>

</div>
  

---



### **YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/1253413048052617268)** (1 messages): 

- **AI hype cycle critique gets a laugh**: A member shared a blog post titled *"I Will Fucking Piledrive You If You Mention AI Again"*, suggesting that readers clear out 10 calm minutes to fully enjoy it and mentioned its hilarious yet true take on the current AI hype cycle. They highlighted a quote criticizing the impracticality of most organizations implementing sophisticated AI technology: *"This isn't a recipe for disaster, it's a cookbook for someone looking to prepare a twelve course fucking catastrophe."* [Read the blog post](https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/)

**Link mentioned**: <a href="https://ludic.mataroa.blog/blog/i-will-fucking-piledrive-you-if-you-mention-ai-again/">I Will Fucking Piledrive You If You Mention AI Again — Ludicity</a>: no description found

  

---



---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

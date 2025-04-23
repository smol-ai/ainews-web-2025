---
id: c6ea6f49-0170-43c6-93c1-308fd26576f2
title: '[AINews] ReALM: Reference Resolution As Language Modeling'
date: '2024-04-04T00:00:20.574613Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-realm-reference-resolution-as-language
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 4/2/2024-4/3/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**382** channels, and **4673** messages) for you. Estimated reading time saved (at 200wpm): **512 minutes**.

Apple is finally waking up to AI in a big way [ahead of WWDC](https://analyticsindiamag.com/what-to-expect-at-the-absolutely-incredible-apple-wwdc-2024/). [We featured MM1 a couple weeks ago](https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/) and now a different team is presenting [ReALM: Reference Resolution As Language Modeling](https://arxiv.org/abs/2403.20329). Reference resolution in their terminology refers to understanding what ambiguous references like "they" or "that" or "the bottom one" or "this number present onscreen" refer to, based on 3 contexts - 1) what's on screen, 2) entities relevant to the conversation, and 3) background entities. They enable all sorts of assistant-like usecases:

 ![image.png](https://assets.buttondown.email/images/d38ff8cd-58f1-4e6c-8ff0-227006ec7cfd.png?w=960&fit=max) 

Which is a challenging task given it basically has to read your mind.

The authors use a mix of labeled and synthetic data to finetune a much smaller FLAN-T5 model  that beats GPT4 at this task:
 ![image.png](https://assets.buttondown.email/images/1e53fed9-dd25-43bd-83ce-ef33cfb2875c.png?w=960&fit=max) 

No model release, no demo. But it's nice to see how they are approaching this problem, and the datasets and models are small enough to be replicable for anyone determined enough.

The [AI content creator industrial complex has gone bonkers over it](https://www.emergentmind.com/papers/2403.20329), of course. There only a few more months' worth of headlines to make about things beating GPT4 before this is itself beaten to death.

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.

**AI Research and Development**

- **Open source coding agent**: In /r/MachineLearning, researchers developed [**SWE-agent, an open source coding agent that achieves 12.29% on the SWE-bench benchmark**](https://www.reddit.com/r/MachineLearning/comments/1btwl37/p_sweagent_an_open_source_coding_agent_that/). The agent can turn GitHub issues into pull requests, but the researchers found building effective agents to be harder than expected after 6 months of work.
- **New RAG engine**: Also in /r/MachineLearning, [RAGFlow was introduced as a **customizable, credible, explainable retrieval-augmented generation (RAG) engine**](https://www.reddit.com/r/MachineLearning/comments/1btycwl/d_ragflow_customizable_credible_explainable_rag/) based on document structure recognition models.
- **Efficient quantization**: In /r/LocalLLaMA, [QuaRot was announced as a **new quantization method enabling 4-bit inference**](https://www.reddit.com/r/LocalLLaMA/comments/1bu8j03/quarot_new_quant_that_offers_4_bit_inference_w4a4/), more efficient than current methods like GPTQ that require dequantization. It also supports lossless 8-bit quantization without calibration data.

**AI Applications and Tools**

- **T-shirt design generator**: In a video post, a Redditor [shared a tool they made to **generate t-shirt designs using AI**](https://v.redd.it/ukxhydz1r0sc1).
- **Podcast generation**: In /r/OpenAI, [podgenai was released as **free GPT-4 based software to generate hour-long informational audiobooks/podcasts** on any topic](https://www.reddit.com/r/OpenAI/comments/1bujeh6/podgenai_a_free_gpt4_api_based_software_to/), requiring an OpenAI API key.
- **Open-source language model**: HuggingFace CEO reshared the [release of PipableAI/pip-library-etl-1.3b, an **open-source model that can be tried out without a GPU**](https://i.redd.it/i5ylmzrmo0sc1.jpeg).

**AI Industry and Trends**

- **Impact of large language models**: In /r/MachineLearning, a discussion was started on [whether **large language models (LLMs) are doing more harm than good for the AI field**](https://www.reddit.com/r/MachineLearning/comments/1btuizd/d_llms_causing_more_harm_than_good_for_the_field/) due to hype changing the focus of conferences and jobs superficially, with overpromising potentially leading to another AI winter.
- **Decentralizing AI**: An Axios article was shared on [efforts to **decentralize AI development and break the hold of big tech companies**](https://www.axios.com/2024/04/02/ai-decentralized-big-tech-blockchain).
- **Stability AI Japan hire**: News was posted about [**Takuto Takizawa joining Stability AI Japan as Head of Japan Sales & Partnerships**](https://i.redd.it/8mt9bei0s5sc1.png).

**Stable Diffusion Discussion**

- **Generating arbitrary resolutions**: In /r/StableDiffusion, a user asked [how Stable Diffusion **generates images at resolutions other than 512x512** given the VAE input/output sizes](https://www.reddit.com/r/StableDiffusion/comments/1bueyze/how_does_stable_diffusion_generate_arbitrary/), seeking an explanation and pointers to relevant code. 
- **Suitability for storytelling**: Also in /r/StableDiffusion, a beginner asked if [Stable Diffusion is **suitable for creating specific characters, poses, and scenes for storytelling and comics**](https://www.reddit.com/r/StableDiffusion/comments/1bukoqt/beginner_question_is_ai_stable_diffusion_good_to/), as they struggle to control the output and consider 3D tools as an alternative.
- **Batch generation in UI**: Another user in /r/StableDiffusion was [looking for the setting to have **Automatic1111's Stable Diffusion UI repeatedly generate images in batches** overnight](https://www.reddit.com/r/StableDiffusion/comments/1bula9s/where_is_the_option_to_repeat_generation_in/).

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Anthropic Research on Jailbreaking LLMs**

- **Many-shot jailbreaking technique**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1775211248239464837) released a research paper studying a long-context jailbreaking technique effective on most large language models. The research shows increasing context window is a double-edged sword, making models more useful but also vulnerable to adversarial attacks.
- **Principled and predictable technique**: [@EthanJPerez](https://twitter.com/EthanJPerez/status/1775230994087543155) noted this is the most effective, reliable, and hard to train away jailbreak known, based on in-context learning. It predictably gets worse with model scale and context length.
- **Concerning results**: [@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1775212287214981207) found the results interesting and concerning, showing many-shot prompting for harmful behavior gets predictably more effective at overcoming safety training with more examples, following a power law.

**Adversarial Validation Technique for Identifying Distribution Shifts**

- **Clever trick to check train/test distribution**: [@svpino](https://twitter.com/svpino/status/1775154270708396215) shared a trick called Adversarial Validation to determine if train and test data come from the same distribution. Put them together, remove target, add binary feature for train/test, train simple model. If AUC near 0.5, same distribution. If near 1, different distributions.
- **Useful for identifying problem features**: Adversarial Validation can identify problem features causing distribution shift. Compute feature importance, remove most important, rebuild model, recompute AUC. Repeat until AUC near 0.5. Useful in production to identify distribution shifts.

**Impact of Taiwan Earthquake on Semiconductor Supply**

- **Proximity of earthquake to fabs**: [@nearcyan](https://twitter.com/nearcyan/status/1775382258767016116) noted the 7.4 earthquake was 64 miles from Central Taiwan Science Park. In 1999, a 7.7 quake near fabs caused production losses. 2016 6.6 quake only delayed ~1% TSMC orders.
- **TSMC preparedness**: TSMC is well prepared for larger quakes. Government prioritizes utility restoration for fabs. No structural damage reported yet. Expect more disruption at Hsinchu/Taichung than 3nm Tainan fab.
- **Potential delays**: Expect nontrivial delays of at least few weeks, possibly months if unlucky. Will likely cause short-term semiconductor price action.

**AI Advancements and Developments** 

- **Genie AI model from DeepMind**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1775162077696360804) announced Genie, a foundation world model that can create playable 2D platformer worlds from a single image prompt, sketch or text description. It could help train AI agents.
- **Replit Code Repair AI agent**: [@pirroh](https://twitter.com/pirroh/status/1775327316157358564) announced Replit Code Repair, a low-latency code repair AI agent using GPT-4. It substantially outperforms open-source models on speed and accuracy.
- **Sonnet model replacing GPT-4**: [@jxnlco](https://twitter.com/jxnlco/status/1775169368781209980) is replacing GPT-4 with Sonnet for most use cases across 3 companies, showing a shift to more specialized models.

**Memes and Humor**

- **Coding longevity meme**: [@svpino](https://twitter.com/svpino/status/1775266990128812314) joked about being told in 1994 that coding would be dead in 5 years, yet still coding 30 years later.
- **Anthropic jailbreaking violence meme**: [@goodside](https://twitter.com/goodside/status/1775271932382068844) joked that if violence doesn't solve your LLM jailbreaking problems, you aren't using enough of it.

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Advancements in Memory-Efficient LLM Training**:
   - A new attention mechanism called **[DISTFLASHATTN](https://arxiv.org/abs/2310.03294)** claims to reduce quadratic peak memory usage to linear for training long-context LLMs, enabling up to **8x longer sequences**. However, the paper lacks pseudocode for the backward pass, raising concerns about reproducibility.
   - Discussions around **[CUDA optimization techniques](https://github.com/cuda-mode)** like DISTFLASHATTN and its potential to revolutionize LLM training through memory efficiency and speed improvements over existing solutions like Ring Self-Attention.

2. **AI Model Evaluations and Benchmarking**:
   - The **[SWE-agent](http://github.com/princeton-nlp/SWE-agent)** open-source system claims comparable accuracy to Devin on the SWE-bench for autonomously solving GitHub issues.
   - Varying performance of models like **GPT-4**, **Claude**, and **Opus** on tasks like solving historical prompts, math riddles, and code generation, highlighting the need for comprehensive evaluations.
   - Platforms like **[Chaiverse.com](https://chaiverse.com)** for rapid feedback on RP-LLM models and **[LMSys Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)** for model benchmarking.

3. **Prompt Engineering and Multimodal AI**:
   - Discussions on **prompt engineering techniques** for tasks like translation while preserving markdown, generating manager prompts, and improving multimodal QA using Chain of Thought.
   - The potential of **[DSPy](https://arxiv.org/abs/2310.03714)** for prompt optimization compared to other frameworks like LangChain and LlamaIndex.
   - Explorations into **multimodal AI** like using Stable Diffusion for depth mapping from stereo images and the launch of **[Stable Audio 2.0](http://stableaudio.com)** for high-quality music generation.

4. **Open-Source AI Developments and Deployments**:
   - Work on an **[Open Interpreter iPhone app](https://github.com/tyfiero/01iOS)** and porting to Android Termux, M5 Cardputer, enabling voice interfaces and exploring local STT solutions.
   - Unveiling of the **[Octopus 2](https://huggingface.co/spaces/Tonic/Octopus/)** demo, a model capable of function calling, fueling excitement around on-device models.
   - Releases like **[Axolotl documentation updates](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a)** and the open-sourcing of **[Mojo's standard library](https://github.com/modularml/mojo/tree/nightly/stdlib)**.

5. **Misc Themes**:

- **Optimization Challenges and Breakthroughs in LLMs**: Engineers grappled with memory and performance bottlenecks in training large language models, with the introduction of novel techniques like **DISTFLASHATTN** which claims [linear memory usage and 8x longer sequences](https://arxiv.org/abs/2310.03294) compared to existing solutions. Discussions also covered leveraging **bf16 optimizers**, **tinyBLAS**, and frameworks like **IPEX-LLM** ([GitHub](https://github.com/intel-analytics/ipex-llm)) for inference acceleration on specific hardware.

- **Anticipation and Analysis of New AI Models**: Communities buzzed with reactions to newly released or upcoming models such as **Apple's ReALM** ([paper](https://arxiv.org/pdf/2403.20329.pdf)), **Stable Diffusion 3.0**, **Stable Audio 2.0** ([website](http://stableaudio.com)), and the **SWE-agent** which matches Devin's performance on the SWE-bench ([GitHub](http://github.com/princeton-nlp/SWE-agent)). Comparative evaluations of instruction-following and chat models like **Claude**, **Opus**, and **Haiku** were also common.

- **Ethical Concerns and Jailbreaking in AI Systems**: Discussions touched on the legal implications of training AI on copyrighted data, as seen with the music platform Suno, and the efficacy of jailbreak defenses in language models, referencing an [arXiv paper](https://arxiv.org/abs/2403.14725) on the importance of defining unsafe outputs. The emotional simulation capabilities of chatbots sparked philosophical debates likening AI to psychopathy.

- **Innovations in AI Interfaces and Applications**: The potential of voice-based interactions with AI was highlighted by apps like **CallStar AI**, while communities worked on projects to make technology more accessible through **conversational UIs**. Initiatives such as **Open Interpreter** aimed to bring AI capabilities to mobile and embedded devices. Novel use cases for AI ranged from **WorldSim's gamified simulations** ([Notion](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)) to **AI-generated art and music**.


---



# PART 1: High level Discord summaries




## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Optimizer Headaches and Proposals**: Technical talks revealed challenges with `torch.compile` and optimizer functions. An emerging solution discussed involved a Python package with `bf16 optimizer` to address dtype conflicts and device compatibility issues.

- **Sound of Legal Alarm for AI Tunes**: The community spotlighted potential legal issues with the AI music platform Suno, emphasizing the risks of copyright infringement suits from record labels due to training on copyrighted content.

- **Memory Hogs & Crashes in Apple's MPS**: Apple's MPS framework was under scrutiny for crashing at high memory allocations even when the memory was available. Theoretical internal limitations and attention slicing as a workaround were hot topics, albeit with concerns about resulting NaN errors.

- **Textual Details Elevate Image Quality**: Research surfaced indicating that fine-tuning text-to-image models with precise spatial descriptions enhances the spatial consistency in generated images, as suggested by an [arXiv paper](https://arxiv.org/pdf/2404.01197.pdf).

- **Decoding AI Optimal Performance**: From skepticism about SD3 Turbo's claimed efficiency to recommendations on model fine-tuning and scheduler effectiveness, the guild analyzed various AI strategies. There were also insights into how smaller models may outperform larger ones within the same inference budget, as shown in a recent [empirical study](https://arxiv.org/abs/2404.01367).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Forge Ahead with Stable Diffusion**: Users report that **Forge**, a user interface for Stable Diffusion, delivers superior performance especially on RTX 3060 and RTX 4080 graphics cards. DreamShaper Lightning (SDXL) models come recommended for efficiency and speed in image generation.

**Anticipation High for SD3**: The Stable Diffusion community is actively awaiting the release of **Stable Diffusion 3.0**, projected to launch in the next 3-5 weeks, with improvements to text rendering expected, though perfect spelling may remain elusive.

**Creative AI Unleashed, But Not 'Unleash'**: Members are experimenting with Stable Diffusion to generate art for projects like tabletop RPGs and are considering storytelling through AI-generated visual narratives, possibly in comic or movie formats.

**Tech Tips for Troubled Times**: Discussions centered on addressing issues such as slow image generation and unwanted text appearance, with participants suggesting optimizations, and mentioning GitHub links as starting points for troubleshooting.

**Features Forecast**: There's evident excitement about upcoming features like sparse control net, SegMOE, and audiosparx models, with the community sharing resources and anticipating new possibilities for AI-generated content.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Cortana 1.0 Chat Model Sparks Curiosity**: Engineers discussed creating an AI prompt model named **Cortana 1.0**, based on the *Halo* series AI, emphasizing creating effective chat modes and prompt structures for streamlined interaction.

**Unsloth Enterprise Capability Clarified**: It was clarified that **Unsloth Enterprise** does indeed support full model training with a speed enhancement of 2-5x over FA2, rather than the expected 30-40x.

**AI Optimization Exchange**: A set of lively discussions covered diverse optimization topics, including advances in **Unsloth AI** with a mention of [Daniel Han's Tweet](https://twitter.com/danielhanchen/status/1775120334305607781), GitHub resources for accelerating AI inference like [ipex-llm](https://github.com/intel-analytics/ipex-llm), and troubleshooting with AI models, notably the compatibility of *SFTTrainer* with **Gemma models**.

**Innovative Approach to Asteroid Mining**: The [Open Asteroid Impact](https://openasteroidimpact.org/) project captured interest with a novel concept of bringing asteroids to Earth to harness resources more effectively.

**Groundwork for Full Stack Prospects**: Solicitations for a skilled full stack developer within the community were made, and users were encouraged to DM if they could recommend or offer assistance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Reading Between the PDF Lines**: Engineers discussed AI models such as **Claude** and **Haiku** for interpreting PDFs, with a focus on *context windows* and **Perplexity's Pro features**, especially the "Writing" focus and enabling "Pro" for accuracy. Some users favored **Sonar** for faster responses.

**Ad-talk Sparks User Spat**: The possibility of **Perplexity introducing ads** sparked debate, following statements by Perplexity's Chief Business Officer on integrating sponsored suggestions. Concerns were raised about the potential impact on the user experience for Pro subscribers, citing a [Verge article](https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform) on the subject.

**PDF Roadblocks and Image Generation**: While addressing technical issues, users clarified that **Perplexity's mobile apps lack image generation support**—an inconvenience tempered by the website’s desktop-like functionality on mobile devices for image generation. Separate discussions pointed to users wanting to lift the 25MB PDF limit for increased efficiency.

**Engineers Exchange 'Supply Links'**: Referral programs and discounts became a hot topic, with mentions of savings through supplied **links**.

**API Woes and Workarounds**: Within the Perplexity API realm, users grappled with the lack of **team support** and **payment issues** for API credits, while also sharing frustrations over **rate limits** and receiving outdated responses from the sonar-medium-online model. The advice ranged from accurate request logging to refining system prompts for up-to-date news.

**Curiosity Drives Deep Dives**:
- Users applied AI to explore a range of subjects from **Fritz Haber's life** and ethical dilemmas to **random forest classifiers** and "Zorba the Greek," hinging on AI's suitability to satisfy diverse and complex inquiries.
- They leveraged Perplexity to efficiently compile comprehensive data for newsletters, indicating a strong inclination towards utilizing AI for streamlined content creation.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Open Source AI Matches Devin**: The [SWE-agent](http://github.com/princeton-nlp/SWE-agent) presented as an open-source alternative to Devin has shown comparable performance on the SWE-bench, prompting discussions on its potential integrations and applications.

**Apple's AI Research Readiness**: A new [paper by Apple](https://arxiv.org/pdf/2403.20329.pdf) showcases ReALM, hinting at AI advancements that could eclipse GPT-4's capabilities, closely integrated with the upcoming iOS 18 for improved Siri interactions.

**Conundrum with Claude**: Users are experimenting with Claude Opus but finding it challenged by complex tasks, leading to recommendations of the [Prompt Engineering Interactive Tutorial](https://www.anthropic.com/index/claude-2-1-prompting) for enhanced interactions with the model.

**Supercharged Sound with Stable Audio 2.0**: StabilityAI has introduced [Stable Audio 2.0](http://stableaudio.com), pushing the boundaries of AI-generated music with its ability to produce full-length, high-quality tracks.

**DALL-E Gets an Edit Button**: ChatGPT Plus now includes features that allow users to **edit DALL-E generated images** and **edit conversation prompts**, bringing new dimensions of customization and control, detailed on [OpenAI's help page](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e).

**DSPy Framework Discussion Heats Up**: The LLM Paper Club scrutinized the [DSPy framework's](https://arxiv.org/abs/2310.03714) functionality and its advantage in prompt optimization over other frameworks, sparking ideas about its application in diverse projects such as voice API logging apps and a platform for summarizing academic papers.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **SWE-agent Rises, Devin Settles**: A cutting-edge system named **SWE-agent** was introduced, claiming to match its predecessor **Devin** in solving GitHub issues with a remarkable 93-second average processing time, and it's available open-source on [GitHub](http://github.com/princeton-nlp/SWE-agent).

- **80M Model Sparking Skepticism**: Engineers discussed an **80M model's** surprising success on out-of-distribution data, prompting speculation about the margin of error and stirring debate about the validity of this performance.

- **Chinese Processor Punches Above its Weight**: Conversations about AI hardware led to **Intellifusion's DeepEyes**, Chinese 14nm AI processor, offering competitive AI performance at significantly reduced costs, potentially challenging the hardware market ([Tom's Hardware report](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus)).

- **Tuning Heroes and Model Troubles**: The community shared experiences of tuning models, like **Lhl's work** with a *jamba model* and *Mvds1's* issue uploading models to Hugging Face due to a metadata snag, pointing out the need for manual adjustments to `SafeTensorsInfo`.

- **WorldSim Sparks Community Imagination**: Engineers enthusiastically explored features for **WorldSim**, ranging from text-to-video integration to a community roadmap, discussing technical enhancements and sharing resources like the [WorldSim Command Index on Notion](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4). Technical constraints and gamification of WorldSim were among the hot topics, showcasing the community's drive for innovation and engagement in simulation platforms.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Lacks Embedding Model Support**: Users confirmed that **LM Studio** currently does not support embedding models, emphasizing that embedding functionality is yet to be implemented.
- **AI Recommendation Query Gains Popularity**: A user's request for a model capable of providing hentai anime recommendations prompted suggestions to use **MyAnimeList (MAL)**, found at [myanimelist.net](https://myanimelist.net/), coupled with community amusement at the unconventional inquiry.
- **Optimized LLM Setup Suspense**: Discussions in the hardware channel revealed insights about **multip GPU configurations without SLI** for LM Studio, recommended GPUs like Nvidia's **Tesla P40**, and concerns regarding future hardware prices due to a [major earthquake affecting TSMC](https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake).
- **API Type Matters for Autogen Integration**: Troubleshooting for LM Studio highlighted the importance of specifying the API type to ensure proper functioning with **Autogen**.
- **Cross-Origin Resource Sharing (CORS) for CrewAI**: A recommendation to enable CORS as a potential fix was discussed for local model usage issues in **LM Studio**, with additional guidance provided via a [Medium article](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL·E Enters the ChatGPT Realm**: Direct in-chat image editing and stylistic inspiration have been introduced for **DALL·E** images within **ChatGPT** interfaces, addressing both convenience and creative exploration.

- **Bing API Goes Silent**: Outages of the **Bing API** lasting 12 hours stirred up concerns among users, affecting services reliant on it, like DALL-E and Bing Image Creator, signaling a need for robust fallback options.

- **Perplexed by Emotion**: Lively debate buzzed around whether GPT-like LLMs can authentically simulate emotions, pointing to the lack of intrinsic motivation in AI and invoking comparisons to psychopathy as well as the infamous Eliza effect.

- **Manager In A Box**: Request for crafting prompts to tackle managerial tasks emphasizes the AI community’s interest in automating complex leadership roles, despite actual strategies or solutions not being churned out in discussions.

- **Translation Puzzles and Markdown Woes**: Efforts to finesacraft translation prompts preserving markdown syntax faced headwinds; inconsistent translations, especially in Arabic, leave AI engineers questioning the limits of current language models' abilities to handle complex formatting and language nuances.




---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Saying Goodbye to a Linux GPU Pioneer**: John Bridgman's retirement from AMD sparked discussions on his contributions to Linux drivers, with George Hotz commenting on the state of AMD's management and future directions. Hotz called for anonymous tips from AMD employees for a possible blog expose, amidst community concerns over AMD's follow-through on driver issues and open-source promises as highlighted in debates and a [Phoronix article](https://www.phoronix.com/news/AMD-Bridgman-Retires).

**Linux Kernel and NVIDIA's Open Move**: The discourse extended to implications of varying kernel versions, particularly around Intel's Xe and i915 drivers, and the transition preferences amongst Linux distributions, with a nod towards moving from Ubuntu 22.04 LTS to 24.04 LTS. Additionally, George Hotz referenced his contribution towards an [open NVIDIA driver initiative](https://github.com/NVIDIA/open-gpu-kernel-modules), stirring conversations about the state of open GPU drivers compared to proprietary ones.

**Tinygrad's Path to V1.0 Involves the Community**: Exploration of **tinygrad's beam search heuristic** and **CommandQueue** functionality highlighted George Hotz's emphasis on the need for improved documentation to aid users in learning and contributing, including a proposed tutorial inspired by ["Write Yourself a Scheme in 48 Hours"](https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours). This goes hand-in-hand with community contributions, like [this command queue tutorial](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md), to polish tinygrad.

**Active Member Engagement Strengthens Tinygrad**: The community's initiative in creating learning materials received kudos, with members offering resources and stepping up to live stream their hands-on experiences with tinygrad, fostering a collaborative learning environment. This aligns with the collective goal to reach tinygrad version 1.0, cementing the platform's position as a tool for education and innovation.

**Rethinking Memory Use in AI Models**: A technical debate ensued on memory optimization during the forward pass of models, particularly regarding the use of activation functions with inverses, leveraging the [inverse function rule](https://en.wikipedia.org/wiki/Inverse_function_rule). This represents the community's engagement in not only tooling but also foundational principles to refine processing efficiency in AI computations.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**OpenInterpreter Dives into App Development**: Development is progressing on an **Open Interpreter iPhone app** with about 40% completion, driven by community collaboration on [GitHub](https://github.com/tyfiero/01iOS), inspired by [Jordan Singer's Twitter concept](https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA).

**Making Tech More Accessible**: There's a push in the **Open Interpreter** community to introduce a **Conversational UI layer** to aid seniors and the disabled, aiming to significantly streamline their interaction with technology.

**Security Measures in a Digital Age**: Members are warned to steer clear of potentially hazardous posts from a seemingly **Open Interpreter X** account suspected of being compromised, in efforts to avert crypto wallet intrusions.

**Out-of-the-Box Porting Initiatives**: **OpenInterpreter** is blurring platform lines with a new repo for Android's Termux installation, work on a **M5 Cardputer** port, and a discussion for implementing local STT solutions amid cost concerns with GPT-4.

**Anticipation for AI Insights**: The community shares a zest for in-depth understanding of LLMs, potentially indicating high interest in gaining advanced technical knowledge about AI systems.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Saturation Alert for Tinystories**: The **Tinystories** dataset is reportedly hitting a saturation point at around 5M parameters, prompting discussions to pivot towards the larger `minipile` dataset despite its greater processing demands.

- **Call for AI Competition Teams**: There's a keen interest within the community for EleutherAI to back teams in AI competitions, leveraging models like llema and expertise in RLHF, along with recommendations to set up dedicated channels and pursue compute grants for support.

- **Defense Against Language Model Jailbreaking**: A recent [paper](https://arxiv.org/abs/2403.14725) suggests that ambiguity in defining unsafe responses is a key challenge in protecting language models against 'jailbreak' attacks, with emphasis placed on the precision of post-processing outputs.

- **AI Model Feedback Submission Highlighted**: [Public comments](https://www.regulations.gov/document/NTIA-2023-0009-0001/comment) on AI model policies reveal a preference for open model development, as showcased by EleutherAI's LaTeX-styled contribution, with discussions revealing both pride and missed opportunities for community engagement.

- **LLM Safety Filter Enhancement Suggestion**: Conversations around mixing refusal examples into fine-tuning data for LLMs reference @BlancheMinerva's tweets and relevant research, corroborating the increased focus on robustness in safety filters as noted in an [ArXiv paper](https://arxiv.org/pdf/2402.18540.pdf).

- **Chemistry Breakthrough with ChemNLP**: The release of the first ChemNLP project paper on [ArXiv](https://arxiv.org/abs/2404.01475) promises significant implications for AI-driven chemistry, sparking interest and likely discussions on future research avenues.

- **Legality Looms over Open Source AI**: A deep dive into the implications of California’s SB 1047 for open-source AI projects encourages signing an open letter in protest, indicating the community's apprehension about the bill's restrictive consequences on innovation. The detailed critique is accessible [here](https://www.context.fund/policy/sb_1047_analysis.html).

- **Conundrum between Abstract and Concrete**: An offbeat clarification sought on how a "house" falls between a "concrete giraffe" and an "abstract giraffe" was met with a lighthearted digital shrug, indicating the playful yet enigmatic side of community discourse.

- **Open Call for Neel Nanda's MATS Stream**: A reminder was shared about the impending deadline (less than 10 days) to apply for Neel Nanda's MATS stream, with complete details available in this [Google Doc](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn).

- **Engagement on Multilingual Generative QA**: The potential of using **Chain of Thought** (CoT) to boost multilingual QA tasks is discussed, with datasets like **MGSM** in the mix and a generated list showcasing tasks incorporating a `generate until` function contributing to the conversation.

- **CUDA Quandaries Call for Community Help**: A user facing `CUDA error: no kernel image is available for execution on the device` with H100 GPUs, not encountered on A100 GPUs, led to troubleshooting efforts that excluded flash attention as the cause, with further advice suggesting checking the `context_layer` device to resolve the issue.

- **Elastic Adventures with PyTorch**: Questions about **elastic GPU/TPU adjustment** during pretraining are met with suggestions of employing [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/quickstart.html), which showcases its ability to adapt to faults and dynamically adjust computational resources, piquing the interest of those looking for scalable training solutions.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Boost Privacy in Repos**: Hugging Face now enables enterprise organizations to set repository visibility to public or private by default, enhancing privacy control. [Their tweet](https://twitter.com/julien_c/status/1772688542289822073) has more details.

**Publish with a Command**: Quarto users can deploy sites on Hugging Face using `use quarto publish hugging-face`, as shared in recent [Twitter](https://twitter.com/gshotwell/status/1772661727856914720) and [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29) posts.

**Gradio's New Sleek Features**: Gradio introduces **automatic deletion** of state variables and **lazy example caching** in the latest 4.25.0 release, detailed in their [changelog](https://gradio.app/changelog).

**Exploring the CLI Frontier**: A shared [YouTube video](https://www.youtube.com/watch?v=PKYPKRoCW2c) explains how to use Linux commands, containers, Rust, and Groq in the command line interface for developers.

**Pushing LLMs to Operative Zen**: A user inquires about fine-tuning language models on PDFs with constrained computational resources, with a focus on inference using open-source models. Meanwhile, a discussion unfolds about modifying special tokens in a tokenizer when fine-tuning an LLM.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Persistent Context Quest in Chat History**: Engineers discussed maintaining **persistent context in chats**, especially when interfacing with databases of 'question : answer' pairs, but did not converge on a specific solution. Reference was made to [LangChain issues and documentation](https://github.com/langchain-ai/langchain/issues/8066) for potential ways forward.

**Video Tutorial For LangServe Playground**: An informative [video tutorial](https://www.youtube.com/watch?v=stWiNP1o2_g) introducing the **Chat Playground** feature in LangServe was shared, aimed at easing the initial setup and showcasing its integration with Langsmith.

**Voice Commands the Future**: Launch of several AI voice apps such as **CallStar AI** and **AllMind AI** was announced, suggesting a trend towards voice as the interface for AI interactions. Links were provided for community support on platforms like [Product Hunt](https://www.producthunt.com/posts/callstar) and [Hacker News](https://news.ycombinator.com/item?id=39914442).

**AI Engineering Troubles and Tutorials**: A CI issue was reported on a [langchain-ai/langserve pull request](https://github.com/langchain-ai/langserve/pull/580); and guidance was sought for a `NotFoundError` when employing LangChain's `ChatOpenAI` and `ChatPromptTemplate`. Meanwhile, novices were directed to a comprehensive [LangChain Quick Start Guide](https://python.langchain.com/docs/get_started/quickstart).

**Galactic API Services Offered and Prompting Proficiency Test**: GalaxyAI provided **free access to premium AI models**, emphasizing API compatibility with Langchain, although the service link was missing. Another initiative, [GitGud LangChain](https://tinyurl.com/gitgud-langchain), challenged **proficient prompters** to test a new code transformation tool to uphold code quality.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Mingles with Memory Safety**: The integration of **Mojo language** into [ROS 2](https://github.com/ros2) suggests potential benefits for robotics development, enhanced by Mojo's memory safety practices. C++ and Rust comparison shows the growing interest in performance and safety in robotics environments.

**Docker Builds Set Sails**: Upcoming **Modular 24.3** will include a fix aimed at improving the efficiency of **automated docker builds**, which has been well-received by the community.

**Logger's Leap to Flexibility**: The logger library in Mojo has been updated to accept **arbitrary arguments and keyword arguments**, allowing for more dynamic logging that accommodates versatile information alongside messages.

**Mojo Dicts Demand More Speed**: Community engagement on the [One Billion Row Challenge](https://github.com/VMois/1brc-mojo) revealed that the performance of `Dict` in **Mojo** needs enhancement, with efforts and discussions ongoing about implementing a custom, potentially SIMD-based, `Dict` that could keep pace with solutions like swiss tables.

**The Collective Drive for Mojo's Nightly Improvements**: Members expressed a desire for clearer pathways to contribution and troubleshooting for **Mojo's stdlib development** with discussions on GitHub clarifying challenges such as parsing errors and behavior of `Optional` types, indicative of active collaboration to refine Mojo's offerings.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **TogetherAI Trips over a Time-Out**: Users reported that the **NOUSRESEARCH/NOUS-HERMES-2-MIXTRAL** model experienced failures, specifically error code 524, which suggests a potential upstream issue with **TogetherAI's API**. A fallback model, **Nous Capybara 34B**, was suggested as an alternative solution.
  
- **Historical Accuracy Test for Chatbots a Mixed Bag**: When tasked with identifying Japanese General **Isoroku Yamamoto** from a historical WW2 context, LLMs such as **claude**, **opus**, and **haiku** exhibited varied levels of accuracy, underscoring the challenge in historical fact handling by current chatbots.

- **OpenRouter Hits a 4MB Ceiling**: A technical limitation was highlighted in OpenRouter, imposing a **4MB maximum payload size** for body content, a constraint confirmed to be without current workarounds.

- **Roleplaying Gets an AI Boost**: In the realm of AI-assisted roleplaying, **Claude 3 Haiku** was a focus, with users sharing tactics for optimization including jailbreaking the models and applying few-shot learning to hone their interactions.

- **Community Sourcing Prompt Playgrounds**: The **SillyTavern** and **Chub's Discord servers** were recommended for those seeking enriched resources for prompts and jailbroken models, pointing to particular techniques like the **pancatstack jailbreak**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**RankZephyr Eclipses the Competition**: The integration of *RankZephyr* into advanced *Retrieval-Augmented Generation* systems is suggested to enhance reranking, with the *RankLLM* collection recognized for its [fine-tuning capabilities](https://twitter.com/llama_index/status/1775166279911186930).

**Enhancing Research Agility with AI Copilots**: A **webinar summary** reveals key strategies in building an **AI Browser Copilot**, focusing on a *prompt engineering pipeline*, *KNN few-shot examples*, and *vector retrieval*, with more insights available on [LlamaIndex’s Twitter](https://twitter.com/llama_index/status/1775264340465381536).

**Timely Data Retrieval Innovations**: *KDB.AI* is said to improve **Retrieval-Augmented Generation** by incorporating *time-sensitive queries* for **hybrid searching**, facilitating a more nuanced search capability critical for contexts like financial reporting, as illustrated in a [code snippet](https://twitter.com/llama_index/status/1775269014849359925).

**Intelligent Library Redefines Knowledge Management**: A **new LLM-powered digital library** for professionals and teams is touted to revolutionize knowledge organization with features allowing creation, organization, and annotation in an advanced digital environment, as announced in a [LlamaIndex tweet](https://twitter.com/llama_index/status/1775537091272937933).

**Community Dialogues Raise Technical Questions**: Discussions in the community include challenges with indexing large PDFs, issues with *qDrant* not releasing a lock post *IngestionPipeline*, limitations of the **HuggingFace API**, model integration using the **Ollama class**, and documentation gaps in recursive query engines with **RAG**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Axolotl Docs Get a Fresh Coat**: The **Axolotl documentation** received an aesthetic update, but a glaring omission of the **Table of Contents** was swiftly corrected as shown in this [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a), although further cleanup is needed for consistency between headings and the Table of Contents.

**Deployment Woes and Wins for Serverless vLLMs**: Experiences with **Runpod** and serverless **vLLMs** were shared, highlighting challenges along with a resource on how to deploy [large language model endpoints](https://github.com/runpod-workers/worker-vllm).

**Data Aggregation Headaches**: Efforts to unify several datasets, comprising hundreds of gigabytes, face complications including file alignment. Presently, TSV files and pickle-formatted index data are used for quick seeking amid discussions on more efficient solutions.

**Casual AI Model Smackdown**: A light-hearted debate compared the preferences of AI models such as 'qwen mow' vs 'jamba', with the community joking about the need for additional data and resources.

**Call for High-Def Data**: A community member seeks resources to obtain a collection of **4K and 8K images**, indicating a project or research that demands high-resolution image data.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Windows ARM Woes with Llamafile**: Compiling **llama.cpp** for Windows ARM requires source compilation because pre-built support isn't available. Developers have been directed to use other platforms for building **llamafile** due to issues with Cosmopolitan's development environment on Windows, as highlighted in [Cosmopolitan issue #1010](https://github.com/jart/cosmopolitan/issues/1010).

- **Mixtral's Brains Better with Bigger Numbers**: **Mixtral** version **`mixtral-8x7b-instruct-v0.1.Q4_0.llamafile`** excels at solving math riddles; however, for fact retention without errors, versions like **`Q5_K_M`** or higher are recommended. For those interested, the specifics can be found on [Hugging Face](https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main).

- **Performance Heft with TinyBLAS**: GPU performance when working with **llamafile** can vastly improve by using a *`--tinyblas`* flag which provides support without additional SDKs, though results may depend on the GPU model used.

- **PEs Can Pack an ARM64 and ARM64EC Punch**: Windows on ARM supports the PE format with ARM64X binaries, which combine Arm64 and Arm64EC code, detailed in Microsoft's [Arm64X PE Files documentation](https://learn.microsoft.com/en-us/windows/arm/arm64x-pe). Potential challenge arises due to the unavailability of AVX/AVX2 instruction emulation in ARM64EC, which can impede operations that LLMs typically require.

- **References for Further Reading**: Articles and resources including the installation guide for the HIP SDK on Windows and details on performance enhancements using Llamafile were shared, such as "Llamafile LLM driver project boosts performance on CPU cores" available on [The Register](https://www.theregister.com/2024/04/03/llamafile_performance_gains/) and HIP SDK's installation documentation available [here](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Opus Judgement Predicts AI Performance Boost**: Discussion highlighted the potential of *Opus Judgement* to unlock performance improvements in Research-Level AI Fine-tuning (**RLAIF**), with certainty hinging on its accuracy.

- **Google's AI Power Move**: Engineers were abuzz about Logan K's transition to lead Google's AI Studio, with a surge of speculation about the motives ranging from personal lifestyle to strategic career positioning. The [official announcement](https://fxtwitter.com/OfficialLoganK/status/1775222819439149424) stirred expectations about the future of the Gemini API under his leadership. 

- **Logan K Sparks Broader AI Alignment Debate**: The move by Logan K sparked conversations regarding AI alignment values versus corporate lures, pondering if the choice was made for more open model sharing at Google or the attractive compensation regardless of personal alignment principles.

- **The Air of Mystery in AI Advances**: A member noted the ripple effect caused by the GPT-4 technical report's lack of transparency, marking a trend towards increased secrecy among AI companies and less sharing of model details.

- **Access Denied to Financial AI Analysis**: Interest in AI's financial implications was piqued by a Financial Times article discussing Google's AI search monetization, but restricted access to the content [Financial Times](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1) limited the discussion among the technical community.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Crashes into LLM Optimization**: The [DISTFLASHATTN mechanism](https://arxiv.org/abs/2310.03294) claims to achieve **linear memory usage** during the training of long-context large language models (LLMs), compared to traditional quadratic peak memory usage, allowing for up to **8x longer sequence processing**. However, the community noted the absence of pseudocode for the backward pass in the paper, raising concerns about reproducibility.

- **Code Talk**: For those seeking hands-on CUDA experience, the [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE) and associated [GitHub materials](https://github.com/cuda-mode) were recommended as starting points for beginners transitioning from Python and Rust.

- **Memory-Efficient Training Makes Waves**: The DISTFLASHATTN paper with its focus on optimizing LLM training is garnering attention, and a member flagged an upcoming detailed review, hinting at further discussion around its memory-efficient training advantages.

- **Backward Pass Backlash**: A member's critique regarding the lack of backward pass pseudocode in the DISTFLASHATTN paper echoed a familiar frustration within the community, calling for improved scientific repeatability in attention mechanism research.

- **Pointers to Intel Analytics' Repo**: A link to Intel Analytics' ipex-llm GitHub repository was shared without additional context, possibly suggesting new tools or developments in the LLM field.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Token Efficiency Talk**: A user highlighted a paper's finding that **throughput efficiency** increases with **per token** measurement, calculated by the ratio of end-to-end throughput (both encoding and decoding) over the **total number of tokens**.

**Speed Debate Heats Up**: There's a divide on how the addition of tokens affects generation speed — while encoding can be done in parallel, the inherent sequential nature of decoding suggests each new token would add to the processing time.

**Focus on Encoding Performance**: Clarification in the discussion pointed to a graph that plotted the speed of generating a fixed 512 tokens, implying that observed speed improvements in the plot should be attributed to **faster encoding** rather than decoding.

**Decoding: The Sequential Slowdown Dilemma**: Queries arose about the possibility of increasing the speed of decoding despite its sequential dependency, which theoretically mandates a waiting period for each token's predecessor.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Eager Pythonista Joins the Fray**: An eager new contributor with a background in **Python**, **software engineering**, and a Master's in **data science** is looking to join the team and contribute to the onboarding process, bringing expertise from AI medical research and data pipeline construction.
  
- **GPT-4 Stumped by Math Without Context**: Even advanced AIs like **GPT-4** and **Claude** can stumble on solving equations unless the problems are posed with clarity in natural language, indicating there's room for improvement at the current state of AI models.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

It seems there is not enough context to generate a summary. Please provide more information or discussions across the channels within the Discord guild to output a meaningful summary.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Clarification on Dialogue Data**: An AI engineer clarified the terminology used within conversation logs, referencing the `responses` table in `logs.db`. The term "speaker turn" or simply "turn" was proposed for the initial part of a conversation, resulting in the renaming of their app's table to `turns`.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1224692800659001356)** (699 messages🔥🔥🔥): 

- **Troubleshooting Difficulties with Optimizer Implementations**: Members engaged in a technical discussion about issues they experienced with the use of `torch.compile` and `add_stochastic_` functions, noting compatibility problems across different devices like NVIDIA, AMD, and MPS. A potential solution involving the creation of a Python package for bf16 optim was discussed, alongside possible modifications to prevent dtype conflict errors during operations.

- **Skepticism Over SD3 Efficiency Improvements**: Doubts were cast regarding claims about the efficiency improvements in SD3 Turbo after a member was banned from a server for questioning the training on limited tokens and the long-term viability of the approach. There were also suggestions that reliance on tools like CLIP may introduce artifacts hindering comprehensive learning.

- **Legal Risks for AI-Generated Music**: A conversation about AI music platform Suno highlighted potential copyright infringement issues, where concerns were raised that record labels' powerful legal teams could pose serious challenges if Suno trained on copyrighted music. Users discussed the complexities of proving infringement in court.

- **MPS Limitations and Crashes at High Memory Utilization**: It was pointed out that Apple's MPS framework would crash when more than 2^32 bytes of data were allocated during training, despite having sufficient memory, indicating a possible internal limitation. Practical workarounds such as attention slicing were also mentioned, though they may lead to other issues like NaN during the backward pass.

- **Recommendations for Model Fine-Tuning and Scheduler Choice**: There were debates over how to properly implement CLIP in conjunction with other models like T5 for better performance, with one member supporting the eventual exclusion of CLIP in favor of purely T5 based models to avoid long-term issues. Further discussions touched on inconsistencies and misinformation spread within the community regarding sampler efficiency and ideal sampling numbers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.972mag.com/lavender-ai-israeli-army-gaza/">‘Lavender’: The AI machine directing Israel’s bombing spree in Gaza</a>: The Israeli army has marked tens of thousands of Gazans as suspects for assassination, using an AI targeting system with little human oversight and a permissive policy for casualties, +972 and Local C...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa_recent_pytorch_nightlies_support_enough/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/8a9w.gif">Ian Malcolm GIF - Ian Malcolm Jurassic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://discuss.pytorch.org/t/runtimeerror-required-rank-4-tensor-to-use-channels-last-format/159729">RuntimeError: required rank 4 tensor to use channels_last format</a>: My transformer training loop seems to work correctly when I train it on the CPU, but when I switch to MPS, I get the below error when computing loss.backward() for Cross Entropy loss. I am doing machi...</li><li><a href="https://arxiv.org/abs/2404.01292">Measuring Style Similarity in Diffusion Models</a>: Generative models are now widely used by graphic designers and artists. Prior works have shown that these models remember and often replicate content from their training data during generation. Hence ...</li><li><a href="https://www.youtube.com/watch?v=_D3GACF-Bsk">Galileo</a>: no description found</li><li><a href="https://www.musicbusinessworldwide.com/suno-is-a-music-ai-company-aiming-to-generate-120-billion-per-year-newton-rex/">Suno is a music AI company aiming to generate $120 billion per year. But is it trained on copyrighted recordings? &#x2d; Music Business Worldwide</a>: Ed Newton&#x2d;Rex discovers that Suno produces music with a striking resemblance to classic copyrights&#8230;</li><li><a href="https://www.youtube.com/watch?v=5pidokakU4I">Axis of Awesome - 4 Four Chord Song (with song titles)</a>: Australian comedy group &#39;Axis Of Awesome&#39; perform a sketch from the 2009 Melbourne International Comedy Festival. Footage courtesy of Network Ten Australia. ...</li><li><a href="https://github.com/pytorch/pytorch/issues/120930>">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/issues/71631>">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/Nerogar/OneTrainer/blob/9a35e7f8596988f672af668f474f8d489ff8f962/modules/util/optimizer/adafactor_extensions.py">OneTrainer/modules/util/optimizer/adafactor_extensions.py at 9a35e7f8596988f672af668f474f8d489ff8f962 · Nerogar/OneTrainer</a>: OneTrainer is a one-stop solution for all your stable diffusion training needs. - Nerogar/OneTrainer</li><li><a href="https://github.com/huggingface/diffusers/issues/7563">[mps] training / inference dtype issues · Issue #7563 · huggingface/diffusers</a>: when training on Diffusers without attention slicing, we see: /AppleInternal/Library/BuildRoots/ce725a5f-c761-11ee-a4ec-b6ef2fd8d87b/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPS...</li><li><a href="https://github.com/steffen74/ConstitutionalAiTuning/">GitHub - steffen74/ConstitutionalAiTuning: A Python library for fine-tuning LLMs with self-defined ethical or contextual alignment, leveraging constitutional AI principles as proposed by Anthropic. Streamlines the process of prompt generation, model interaction, and fine-tuning for more responsible AI development.</a>: A Python library for fine-tuning LLMs with self-defined ethical or contextual alignment, leveraging constitutional AI principles as proposed by Anthropic. Streamlines the process of prompt generati...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530#discussion_r1547822696">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>: What does this PR do?   Fixes #7529 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&#39;s the case).  Did you read the contributor guideline?  D...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1224691527167643719)** (11 messages🔥): 

- **Scaling vs. Sampling Efficiency Analyzed**: An empirical study highlighted in [this article](https://arxiv.org/abs/2404.01367) explores the influence of model size on the sampling efficiency of latent diffusion models (LDMs). Contrary to expectations, it was found that smaller models often outperform larger ones when under the same inference budget.

- **In Search of Scalable Crawling Techniques**: A member inquired about research into scalable crawling methods that could assist in building datasets for model training. However, no specific groups or resources were referenced in the response.

- **Mystery of Making $50K Revealed**: A humorous exchange involved a link to a [Discord mod ban GIF](https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646) and a guess that the secret to making $50K in 72 hours could involve being a drug mule, referencing an MLM-related meme.

- **Teasing a New Optimizer on Twitter**: There's anticipation for a [new optimizer](https://twitter.com/aaron_defazio/status/1775521495298588956) discussed on Twitter, promising potential advancements in the field.

- **Visual Enhancements through Specificity**: Discussing an [arXiv paper](https://arxiv.org/pdf/2404.01197.pdf), it was mentioned that fine-tuning text-to-image (t2i) models with captions that include better spatial descriptions can lead to images with improved spatial consistency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.01367">Bigger is not Always Better: Scaling Properties of Latent Diffusion Models</a>: We study the scaling properties of latent diffusion models (LDMs) with an emphasis on their sampling efficiency. While improved network architecture and inference algorithms have shown to effectively ...</li><li><a href="https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646">Discord Mod Moderation Ban GIF - Discord mod Moderation ban Mod ban - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1224793707606184117)** (1 messages): 

- **LangChain's Harrison Chase to Illuminate LLM Challenges**: Attendees are invited to an *exclusive event* with **Harrison Chase**, co-founder and CEO of **LangChain**. He will discuss the challenges companies face when moving from prototype to production and how **LangSmith** helps overcome these hurdles, providing insights during a meetup organized for April 17th at 18:30 @Online. [Register here](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/).
- **Insider Access to LLM Framework Trends with LangChain**: The co-founder of LangChain, **Harrison Chase**, will share his expertise on using **LLMs (Large Language Models)** for developing context-aware reasoning applications. This talk will address the challenges encountered by companies and the solutions implemented, as part of the third **LangChain and LLM France Meetup**.

**Link mentioned**: <a href="https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/">Meetup #3 LangChain and LLM: Using LangSmith to go from prototype to production, mer. 17 avr. 2024, 18:30   | Meetup</a>: Nous avons le plaisir d&#x27;accueillir Harrison Chase, le Co-Founder et CEO de LangChain, pour notre troisième Meetup LangChain and LLM France !  Ne loupez pas cette occasion u

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1224640668421459988)** (568 messages🔥🔥🔥): 

- **Stable Diffusion Secrets Revealed**: Members are discussing the performance of various versions of Stable Diffusion. Forge is highlighted as the fastest UI right now, and there's a lot of love for models like DreamShaper Lightning (SDXL). Users with graphics cards like the RTX 3060 and RTX 4080 noted significant speed improvements when using Forge compared to A1111, with image generation times dropping significantly.
- **Anticipation Builds for SD3**: The community is eagerly waiting for the release of Stable Diffusion 3.0, with estimated arrival times ranging between 3-5 weeks. However, it was noted that while SD3 will improve text rendering, it might still not achieve perfect spelling due to its limitations and model size.
- **Harnessing SD for Creative Projects**: Users are exploring the use of Stable Diffusion for various creative endeavors such as generating art for tabletop RPGs or contemplating storytelling through images, potentially in comic or movie formats.
- **Technical Tackles and Tips**: A conversation around potential issues faced while generating images, such as slow speeds or text from one prompt appearing in another, led to suggestions on utilizing specific Stable Diffusion optimizations and trying out alternative interfaces, such as Forge.
- **New Models and Features on the Horizon**: Excitement is also buzzing around the community for the new features like sparse control net, SegMOE, and audiosparx model, shared alongside helpful GitHub links and tips on better leveraging AI-generated content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.leonardo.ai/">Leonardo.Ai</a>: Create production-quality visual assets for your projects with unprecedented quality, speed and style-consistency.</li><li><a href="https://tenor.com/view/anime-help-tears-cry-sad-gif-17104681">Anime Help GIF - Anime Help Tears - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://remix.ai/">Remix</a>: Create, share, and remix AI images and video.</li><li><a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus">BFloat16: The secret to high performance on Cloud TPUs | Google Cloud Blog</a>: How the high performance of Google Cloud TPUs is driven by Brain Floating Point Format, or bfloat16</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations">Optimizations</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/3Frame">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/229002">ICBINP XL - v4 | Stable Diffusion Checkpoint | Civitai</a>: If you do like this work, consider buying me a coffee :) Use this model for free on Stable Horde The long awaited followup to ICBINP, this model is...</li><li><a href="https://forms.gle/9i4jM9BQu9bVVAAF6">Survey Form - 5day.io</a>: As a young professional just a few years into the workforce, there is a constant, low-humming anxiety about proving yourself and finding that mythical work-life balance everyone talks about. Sometimes...</li><li><a href="https://www.youtube.com/watch?v=yvOXZ6SV2Rk">Stable Radio 24/7</a>: Stable Radio, a 24/7 live stream that features tracks exclusively generated by Stable Audio.Explore the model and start creating for free on stableaudio.com</li><li><a href="https://tenor.com/tKgaYjwJq16.gif">Cool Fun GIF - Cool Fun White cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/continue-revolution/sd-webui-animatediff/blob/master/docs/features.md#controlnet-v2v">sd-webui-animatediff/docs/features.md at master · continue-revolution/sd-webui-animatediff</a>: AnimateDiff for AUTOMATIC1111 Stable Diffusion WebUI - continue-revolution/sd-webui-animatediff</li><li><a href="https://github.com/princeton-nlp/SWE-agent">GitHub - princeton-nlp/SWE-agent: SWE-agent: Agent Computer Interfaces Enable Software Engineering Language Models</a>: SWE-agent: Agent Computer Interfaces Enable Software Engineering Language Models - princeton-nlp/SWE-agent</li><li><a href="https://www.reddit.com/r/3FrameMovies/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://github.com/kijai/ComfyUI-DiffusionLight">GitHub - kijai/ComfyUI-DiffusionLight: Using DiffusionLight in ComfyUI</a>: Using DiffusionLight in ComfyUI. Contribute to kijai/ComfyUI-DiffusionLight development by creating an account on GitHub.</li><li><a href="https://github.com/ZHO-ZHO-ZHO/ComfyUI-SegMoE">GitHub - ZHO-ZHO-ZHO/ComfyUI-SegMoE: Unofficial implementation of SegMoE for ComfyUI</a>: Unofficial implementation of SegMoE for ComfyUI. Contribute to ZHO-ZHO-ZHO/ComfyUI-SegMoE development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1224650862958346270)** (241 messages🔥🔥): 

- **Request for Full Stack Developer Recommendations**: A member sought recommendations for good full stack developers, inviting direct messages from anyone able to assist.

- **Inquiry About Unsloth Enterprise Model Training**: A question was raised regarding whether Unsloth Enterprise supports full model training; the response clarified that it does, but the speedup factor would be between 2-5x faster than FA2, rather than 30-40x.

- **Discussion on Prompt Formats and Implementations**: Members discussed custom AI models and prompt formats, with specific references to creating a model called Cortana 1.0, designed after the AI in the Master Chief video games. Concerns were raised about finding suitable models for chat mode and utilizing correct prompt structures for efficient operation.

- **Updates and Achievements Shared in AI Development**: They shared [Daniel Han's tweet](https://twitter.com/danielhanchen/status/1775120334305607781) reflecting on the potential of AI over a few months, given the short development time so far. Benchmarks for Unsloth AI were also discussed, including a 12.29% performance on the SWE Bench by their 'Ye' model.

- **Concerns and Optimizations for AI Performance**: Various members inquired about optimizations and support for different AI models and platforms. For instance, discussions revolved around the support for Galore within Unsloth, the possible open-sourcing of GPT models, and efforts to accelerate local LLM inference and fine-tuning on Intel CPUs and GPUs. An exchange with [links to GitHub](https://github.com/intel-analytics/ipex-llm) highlighted resources for accelerating AI inference on specific hardware. There was also a discussion about potential performance improvements and updates coming soon from the Unsloth team.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1-uKmQzhh8ftxEdipiqGu4sVdRb8MgWv2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://tenor.com/view/am-ia-joke-to-you-am-ia-joke-is-this-a-joke-do-you-think-this-is-funny-do-you-think-this-is-a-joke-gif-14191111">Am Ia Joke To You Is This A Joke GIF - Am IA Joke To You Am IA Joke Is This A Joke - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-aint-no-fool-wiz-khalifa-still-wiz-song-im-not-a-fool-im-not-an-idiot-gif-21822363">I Aint No Fool Wiz Khalifa GIF - I Aint No Fool Wiz Khalifa Still Wiz Song - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b">jondurbin/airoboros-gpt-3.5-turbo-100k-7b · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/intel/neural-speed">GitHub - intel/neural-speed: An innovative library for efficient LLM inference via low-bit quantization</a>: An innovative library for efficient LLM inference via low-bit quantization - intel/neural-speed</li><li><a href="https://github.com/intel-analytics/ipex-llm">GitHub - intel-analytics/ipex-llm: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max). A PyTorch LLM library that seamlessly integrates with llama.cpp, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, ModelScope, etc.</a>: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)...</li><li><a href="https://github.com/toranb/sloth/blob/master/sftune.py">sloth/sftune.py at master · toranb/sloth</a>: python sftune, qmerge and dpo scripts with unsloth - toranb/sloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1546dvc/24gb_vram_on_a_budget/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1224808992300335278)** (12 messages🔥): 

- **Asteroid Mining Company with a Twist**: The [Open Asteroid Impact](https://openasteroidimpact.org/) initiative is a unique approach to asteroid mining that proposes slinging asteroids to Earth instead of mining in space. The link provided displays their logo and underscores their aim to prioritize safety and efficiency in resources acquisition from space.

- **Praise for Unsloth's Website Design**: A member complimented the website design for Unsloth, noting the attractiveness of the site.

- **Creativity on a Budget**: The Unsloth website's sloth images were designed with Bing DALL-E due to budget constraints. The designer also expressed intentions to eventually commission 3D artists for a consistent mascot.

- **Design Consistency Through Hard Work**: Responding to an inquiry about the uniformity of design, the Unsloth website designer mentioned generating hundreds of sloth images and refining them manually in Photoshop.

- **Bing DALL-E Over Hugging Face for Speed**: The designer chose Bing DALL-E over Hugging Face’s DALL E's for image generation because of the ability to generate multiple images quickly and having available credits.

**Link mentioned**: <a href="https://openasteroidimpact.org/">Open Asteroid Impact</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1224663029539668029)** (278 messages🔥🔥): 

- **Evaluation During Training Explained**: Members discussed why evaluation datasets are not added by default during fine-tuning—adding them slows down the process. The training loss is calculated using cross-entropy loss, and evaluation loss uses the same metric.

- **Pack Smart with SFTTrainer**: When using `SFTTrainer`, members shared how to configure and optimize training, including the use of `packing` and avoiding using it with Gemma models, as it can lead to problems.

- **Dealing with Dataset Size Challenges**: Users troubleshoot issues related to OOM errors and dataset size, including a discussion on the use of streaming datasets for large volumes and the challenges with tools like PyArrow when handling very large amounts of data.

- **GGUF Conversion Confusion**: A member faced issues converting a model into GGUF format and debated the appropriate approach, discussing the possible need for manual architecture adjustments in conversion scripts.

- **Inference Troubles and Unsloth Updates**: There was a case of a GemmaForCausalLM object causing an attribute error, which was fixed after the Unsloth library was updated and reinstalled. A member mentioned that using 16-bit model inference led to OOM errors, and someone had an issue with Python.h missing during the setup of a finetuning environment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF">qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">Adding accuracy, precision, recall and f1 score metrics during training</a>: hi, you can define your computing metric function and pass it into the trainer. Here is an example of computing metrics.   define accuracy metrics function from sklearn.metrics import accuracy_score, ...</li><li><a href="https://huggingface.co/danielhanchen/model_21032024">danielhanchen/model_21032024 · Hugging Face</a>: no description found</li><li><a href="https://docs.wandb.ai/guides/integrations/huggingface">Hugging Face Transformers | Weights &amp; Biases Documentation</a>: The Hugging Face Transformers library makes state-of-the-art NLP models like BERT and training techniques like mixed precision and gradient checkpointing easy to use. The W&amp;B integration adds rich...</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer#trl.trainer.ConstantLengthDataset">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GPTQ-Int4">Qwen/Qwen1.5-14B-Chat-GPTQ-Int4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/">deepseek-ai/deepseek-vl-7b-chat · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF">TheBloke/deepseek-coder-6.7B-instruct-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1224640434211389500)** (469 messages🔥🔥🔥): 

- **Discussions on Pro Models and Usage**: Users exchanged insights on using different AI models, such as Claude and Haiku, for reading and interpreting PDFs. They debated the advantages of Perplexity's Pro features and models' context windows, with suggestions to use "Writing" focus for detailed responses and enable "Pro" for more concise and accurate answers. Some suggested using Sonar for speedier responses.

- **Ads Coming to Perplexity?**: There was a significant concern over reports of Perplexity planning to introduce ads. Users referenced statements from Perplexity's Chief Business Officer about the potential of sponsored suggested questions, with some expressing disappointment and hoping the ad integration would not affect the Pro user experience.

- **Image Generation Queries and Accessibility**: Users asked about generating images on desktop and mobile, with a response confirming that while the mobile apps do not support image generation, the website does on mobile devices.

- **Referral Links and Discounts**: Users shared referral links for Perplexity.ai, mentioning the availability of $10 discounts through these links.

- **Technical Support and Feature Requests**: Users inquired about technical issues like API limits and slow response times, as well as feature updates like lifting the 25MB PDF limit. There was a recommendation to use Sonar for speed and some discussions on whether Perplexity has lifted certain restrictions.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://groq.com/">GroqChat</a>: no description found</li><li><a href="https://community.spiceworks.com/">no title found</a>: no description found</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform">Perplexity will try a form of ads on its AI search platform.</a>: Perplexity’s chief business officer Dmitry Shevelenko tells Adweek the company is considering adding sponsored suggested questions to its platform. If users continue to search for more information on ...</li><li><a href="https://www.tomsguide.com/ai/apple-reveals-realm-new-ai-model-could-make-siri-way-faster-and-smarter">Apple reveals ReALM &mdash; new AI model could make Siri way faster and smarter</a>: ReALM could be part of Siri 2.0</li><li><a href="https://www.adweek.com/media/gen-ai-search-engine-perplexity-has-a-plan-to-sell-ads/">Gen-AI Search Engine Perplexity Has a Plan to Sell Ads</a>: no description found</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platfo">Perplexity will try a form of ads on its AI search platform.</a>: Perplexity’s chief business officer Dmitry Shevelenko tells Adweek the company is considering adding sponsored suggested questions to its platform. If users continue to search for more information on ...</li><li><a href="https://tenor.com/view/when-server-down-iceeramen-monkey-gif-23229726">When Server Down Iceeramen GIF - When Server Down Iceeramen Monkey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1775229252973334902?t=p2-h_dWeQhz6swoCVL66SA&s=19">Tweet from Aravind Srinivas (@AravSrinivas)</a>: good vibes are essential</li><li><a href="https://x.com/aravsrinivas/status/1775244089505845610?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Can’t wait. </li><li><a href="https://fxtwitter.com/apostraphi/status/1775240129264730438?t=8XB64t2ExHGixP06DHvAHw&s=19">Tweet from Phi Hoang (@apostraphi)</a>: Merch drop this month. In collaboration with @Smith_Diction for @perplexity_ai.</li><li><a href="https://www.reddit.com/r/singularity/comments/1bp885i/claude_3_haiku_is_the_new_budget_king/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/getting-started">Getting Started with pplx-api</a>: no description found</li><li><a href="https://www.gadgets360.com/ai/news/perplexity-ai-powered-search-engine-could-soon-show-ads-report-5357479">AI Search Engine Perplexity Could Soon Show Ads to Users: Report</a>: As per the report, Perplexity will show ads in its related questions section.</li><li><a href="https://slashdot.org/story/24/04/01/1653221/perplexity-an-ai-startup-attempting-to-challenge-google-plans-to-sell-ads">Perplexity, an AI Startup Attempting To Challenge Google, Plans To Sell Ads - Slashdot</a>: An anonymous reader shares a report: Generative AI search engine Perplexity, which claims to be a Google competitor and recently snagged a $73.6 million Series B funding from investors like Jeff Bezos...</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bo">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bot_public_releaseintroducing/?rdt=64126">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.quora.com">Quora - A place to share knowledge and better understand the world</a>: no description found</li><li><a href="https://www.reddit.com/r/AskReddit">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/NoStupidQuestions">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/explainlikeimfive">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/subreddits/search">search results</a>: no description found</li><li><a href="https://community.spiceworks.com">no title found</a>: no description found</li><li><a href="https://discuss.codecademy.com">Codecademy Forums</a>: Community discussion forums for Codecademy.</li><li><a href="https://hashnode.com">Start a Developer Blog: Hashnode - Custom Domain, Sub-path, Hosted/Headless CMS.</a>: Developer blogging with custom domains, hosted/headless CMS options. Our new headless CMS streamlines content management for devtool companies.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1224647504826601513)** (23 messages🔥): 

- **Tailored Article Magic**: A member discovered they can create articles highly customized to their interests, highlighting the ability to hone in on specific topics using Perplexity.
- **Efficient Research for Newsletters**: Perplexity facilitated a user in swiftly gathering accurate information, which significantly expedited the creation of a "welcome gift" for their newsletter subscribers.
- **A Noble Examination of Fritz Haber**: Utilizing the Perplexity search, a member delved into the life of **Fritz Haber**, revealing his pivotal contribution to food production with the Haber-Bosch process, his complex history with chemical warfare, and his moral stance against the Nazi regime. The nuances include his Nobel Prize-winning achievement and the unfortunate family and historical circumstances surrounding him.
- **Curiosity Fueled Learning**: Users are engaging with Perplexity to feed their curiosity on diverse topics ranging from convolutions in machine learning to **Zorba the Greek**, showcasing the platform's versatility in addressing various inquiries.
- **Conceptual Clarity on Random Forest**: Multiple members sought to understand the **random forest classifier**, indicating a shared interest in machine learning algorithms within the community.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1224751637936079020)** (24 messages🔥): 

- **No Team Sign-Up for Perplexity API**: A user inquired about signing up for the Perplexity API with a team plan, but it was confirmed that **team sign-ups are currently unavailable**.

- **Rate Limits Confusion**: A member shared **confusion about rate limits**, specifically using the sonar-medium-online model. Despite adhering to the 20req/m limit, they are **still encountering 429 errors**; it was suggested to log requests with timestamps to ensure the rate limits are enforced correctly.

- **Trouble with Temporally Accurate Results**: A user reported inaccurate results when asking for the day's top tech news using the sonar-medium-online model, receiving outdated information. It was recommended to include **"Ensure responses are aligned with the Current date."** in the system prompt to help guide the model's results.

- **Clarifying the Perplexity API's Functionality**: A clarification was sought on how the Perplexity API works. Points include generating an API key, sending the key as a bearer token in requests, and managing the credit balance with possible automatic top-ups.

- **Payment Pending Issues for API Credits**: A member voiced concerns about issues when trying to buy API credits — the process indicates **"Pending" status** without account updates. A request for account details to check the issue on the backend was made by a staff member.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1224726965886648423)** (76 messages🔥🔥): 

- **Open Source SWE-agent Rivals Devin**: A new system called [SWE-agent](http://github.com/princeton-nlp/SWE-agent) has been introduced, boasting similar accuracy to Devin on SWE-bench and has the distinguishing feature of being open source.
- **Apple Research Hints at AI Leapfrogging GPT-4**: An Apple research [paper](https://arxiv.org/pdf/2403.20329.pdf) discusses a system named ReALM, suggesting capabilities that surpass ChatGPT 4.0, in sync with iOS 18 developments for Siri.
- **Claude Opus's Performance Dilemma**: Conversations report a notable performance gap between Claude Opus and GPT-4, with Opus struggling in certain tasks such as the "needle-in-a-haystack" test. There's mention of a [Prompt Engineering Interactive Tutorial](https://www.anthropic.com/index/claude-2-1-prompting) to improve results with Claude.
- **Stable Audio 2.0 Launches**: [StabilityAI announces Stable Audio 2.0](http://stableaudio.com), an AI capable of generating high-quality, full-length music tracks, stepping up the game in audio AI capabilities.
- **ChatGPT Plus Enhancements**: ChatGPT Plus now allows users to **edit DALL-E images** from the web or app, and a recent iOS update includes an option to **edit conversation prompts**. Detailed instructions are available on [OpenAI's help page](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://devday.replit.com/">Replit</a>: Replit Developer Day Livestream</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...</li><li><a href="https://9to5mac.com/2024/04/01/apple-ai-gpt-4/">Apple AI researchers boast useful on-device model that ‘substantially outperforms’ GPT-4 - 9to5Mac</a>: Siri has recently been attempting to describe images received in Messages when using CarPlay or the announce notifications feature. In...</li><li><a href="https://x.com/sullyomarr/status/1774960295393538519?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sully (@SullyOmarr)</a>: I use cursor as my ide, and Claude seems significantly worse with the api   Half finished code, bad logic, horrible coding style  But it works perfectly on their site  Anyone else experience this?</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...</li><li><a href="https://x.com/zswitten/status/1775187565219631155?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Zack Witten (@zswitten)</a>: I&#39;ve been dying to shill this harder for six months, and now that Anthropic API is GA, I finally can...  The Prompt Engineering Interactive Tutorial! https://docs.google.com/spreadsheets/d/19jzLgR...</li><li><a href="https://blog.replit.com/code-repair">Replit — Building LLMs for Code Repair</a>: Introduction  At Replit, we are rethinking the developer experience with AI as a first-class citizen of the development environment. Towards this vision, we are tightly integrating AI tools with our I...</li><li><a href="https://x.com/gregkamradt/status/1727018183608193393?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Greg Kamradt (@GregKamradt)</a>: Claude 2.1 (200K Tokens) - Pressure Testing Long Context Recall  We all love increasing context lengths - but what&#39;s performance like?  Anthropic reached out with early access to Claude 2.1 so I r...</li><li><a href="https://x.com/anthropicai/status/1732527908139552951?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Anthropic (@AnthropicAI)</a>: Claude 2.1’s 200K token context window is powerful, but requires careful prompting to use effectively.      Learn how to get Claude to recall an individual sentence across long documents with high fid...</li><li><a href="https://x.com/jyangballin/status/1775114444370051582?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from John Yang (@jyangballin)</a>: SWE-agent is our new system for autonomously solving issues in GitHub repos. It gets similar accuracy to Devin on SWE-bench, takes 93 seconds on avg + it&#39;s open source!  We designed a new agent-co...</li><li><a href="https://x.com/ofirpress/status/1775226081575915661?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Ofir Press (@OfirPress)</a>: People are asking us how Claude 3 does with SWE-agent- not well. On SWE-bench Lite (a 10% subset of the test set) it gets almost 6% less (absolute) than GPT-4.   It&#39;s also much slower.   We&#39;ll...</li><li><a href="https://x.com/jd_pressman/status/1775295848509026659?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from John David Pressman (@jd_pressman)</a>: &#34;Many Shot Jailbreaking&#34; is the most embarrassing publication from a major lab I&#39;ve seen in a while, and I&#39;m including OpenAI&#39;s superalignment post in that.  ↘️ Quoting lumpen spac...</li><li><a href="https://x.com/StabilityAI/status/1775501906321793266?s=20">Tweet from Stability AI (@StabilityAI)</a>: Introducing Stable Audio 2.0 – a new model capable of producing high-quality, full tracks with coherent musical structure up to three minutes long at 44.1 kHz stereo from a single prompt.  Explore the...</li><li><a href="https://x.com/teortaxestex/status/1775003753055228391?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: Opus is an immensely strong model, a poet and a godsend to @repligate. It&#39;s also subpar in factuality (makes stuff up OR doesn&#39;t know it) and instruction-following; GPT-4, even Mistrals may do...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">Tweet from Gustavo Cid (@_cgustavo)</a>: I used to beg LLMs for structured outputs.   Most of the time, they understood the job and returned valid JSONs. However, around ~5% of the time, they didn&#39;t, and I had to write glue code to avoid...</li><li><a href="https://overcast.fm/+HaNOG0VjE/19:08">Should kids still learn to code? (Practical AI #263) &mdash; Changelog Master Feed &mdash; Overcast</a>: no description found</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya">Tweet from Gustavo Cid (@_cgustavo)</a>: I used to beg LLMs for structured outputs.   Most of the time, they understood the job and returned valid JSONs. However, around ~5% of the time, they didn&#39;t, and I had to write glue code to avoid...</li><li><a href="https://youtu.be/tVw3CwrN5-8?si=d_0EPgMCRL9mhva_">Structured Outputs with DSPy</a>: Unfortunately, Large Language Models will not consistently follow the instructions that you give them. This is a massive problem when you are building AI sys...</li><li><a href="https://x.com/gblazex/status/1775558982645547236?s=20">Tweet from Blaze (Balázs Galambosi) (@gblazex)</a>: Wow. While OpenAI API is still stuck on Whisper-2, @AssemblyAI releases something that beats even Wishper-3: + 13.5% more accurate than  Whisper-3  + Up to 30% fewer hallucinations + 38s to process 60...</li><li><a href="https://www.youtube.com/watch?v=N1TEjTeQeg0">Prof. Geoffrey Hinton - &quot;Will digital intelligence replace biological intelligence?&quot; Romanes Lecture</a>: Professor Geoffrey Hinton, CC, FRS, FRSC, the ‘Godfather of AI’, delivered Oxford&#39;s annual Romanes Lecture at the Sheldonian Theatre on Monday, 19 February 2...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1225158181316067422)** (356 messages🔥🔥): 

- **DSPy Takes Center Stage**: LLM Paper Club discussed the **DSPy framework** comparing its utility to that of **LangChain** and **LlamaIndex**. There's an emphasis on its ability to **optimize prompts** for different large language models (LLMs) and **migrate** models easily, a capability underscored in [DSPy's arXiv paper](https://arxiv.org/abs/2310.03714).

- **Devin's Debut Draws Discussion**: The concept of **Devin**, an AI with thousands of dollars of OpenAI credit backing it for demos, was mentioned, generating excitement and anticipation for its potential demonstration uses.

- **Exploring DSPy's Depth**: Questions around **DSPy's operation** and execution were posed, including whether it can **compile to smaller models**, **rate limit calls** to avoid OpenAI API saturation, and **save optimization** outcomes to disk using the `.save` function.

- **Prompt Optimization Potential**: There was an interest in **DSPy's ability** to optimize a **single metric** and whether **multiple metrics** could be combined into a composite score for optimization purposes. The discussion points highlighted DSPy's **teleprompter/optimizer** functionality, which does not require the metric to be differentiable.

- **Practical Applications Proposed**: Club members proposed various **practical applications** for the LLMs, including an **iOS app for logging voice API** conversations, a **front-end platform for summarizing arXiv papers** based on URLs, a **DSPy pipeline for PII detection**, and rewriting of **DSPy's documentation**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.03714">DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines</a>: The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically i...</li><li><a href="https://eugeneyan.com/writing/abstractive/">Evaluation & Hallucination Detection for Abstractive Summaries</a>: Reference, context, and preference-based metrics, self-consistency, and catching hallucinations.</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://eugeneyan.com/writing/evals/#summ">LLM Task-Specific Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">LLM Task-Specific Evals that Do & Don't Work</a>: Evals for classification, summarization, translation, copyright regurgitation, and toxicity.</li><li><a href="https://www.spotery.com/">Are you human?</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://hamel.dev/blog/posts/prompt/#dspy">- Fuck You, Show Me The Prompt.</a>: Quickly understand inscrutable LLM frameworks by intercepting API calls.</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/knn.ipynb">dspy/examples/knn.ipynb at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://x.com/HamelHusain/status/1774999027538612652?s=20">Tweet from Hamel Husain (@HamelHusain)</a>: @swyx a guy + a small cult of fans
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1224830090073084036)** (4 messages): 

- **Autonomous GitHub Issue Resolver Unveiled**: A new system named **SWE-agent** has been shared, boasting similar accuracy to its predecessor Devin on SWE-bench and provided with an innovative agent-computer interface. It processes tasks in an average of 93 seconds and is available as open-source on its [GitHub repository](http://github.com/princeton-nlp/SWE-agent).

- **The Rise and Fall of Devin**: A simple remark highlights the swift evolution in AI tools with **Devin** considered impressive just two weeks prior to the introduction of SWE-agent.

- **Exploration of Scalable Data Crawling**: A member inquires about research into methods of scalable crawling for creating large datasets, with a response indicating a broad interest in both expanding dataset size and enhancing quality.

**Link mentioned**: <a href="https://fxtwitter.com/jyangballin/status/1775114444370051582">Tweet from John Yang (@jyangballin)</a>: SWE-agent is our new system for autonomously solving issues in GitHub repos. It gets similar accuracy to Devin on SWE-bench, takes 93 seconds on avg + it&#39;s open source!  We designed a new agent-co...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224658774166343751)** (17 messages🔥): 

- **Understanding Unseen Performance**: Discussion touched on a curious phenomenon where an **80M model bested larger ones on unseen performance**. Skepticism arose around the validity of this result, with suggestions such as high **margin of error** for unseen domain evaluations.
- **Peculiar OOD Data Results**: Members remarked upon the oddity of an **80M model** scoring highly on out-of-distribution (OOD) data, leading to speculation about potential errors in evaluation.
- **Exploring LLM Vulnerabilities**: A red teaming suite created by @enkryptai was mentioned, designed to examine the vulnerabilities of Large Language Models (LLMs) including tests on **@databricks's DBRX** and MoE SSM LLM **Jamba**. Results were shared indicating the discovery of some significant issues ([Tweet about LLM vulnerabilities](https://x.com/divyanshutwt/status/1775241719740535149?s=20)).
- **Lollms & Ollama Server Tutorial**: A YouTube tutorial was highlighted showcasing how to install and use **lollms with Ollama Server**, aimed at tech enthusiasts ([YouTube Tutorial on lollms & Ollama Server](https://www.youtube.com/watch?v=RuQSQmolXGE)).
- **China's Alternative AI Hardware**: Discussion about a Chinese chipmaker **Intellifusion** that launched a 14nm AI processor called "DeepEyes," which is significantly cheaper than comparable GPUs. The processor's AI performance and competitive pricing could challenge high-end hardware in the AI market ([Tom's Hardware article on Intellifusion](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus">Chinese chipmaker launches 14nm AI processor that's 90% cheaper than GPUs &mdash; $140 chip's older node sidesteps US sanctions</a>: If there's a way to sidestep sanctions, you know China is on that beat.</li><li><a href="https://x.com/divyanshutwt/status/1775241719740535149?s=20">Tweet from Divyanshu (@divyanshutwt)</a>: At @enkryptai we&#39;ve build a red teaming suite to identify the pitfalls of LLMs. Recently, we tested the vulnerability of @databricks &#39;s DBRX and 🐍Jamba, a MoE SSM LLM. Got some interesting re...</li><li><a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601 - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=RuQSQmolXGE">Installing &amp; Unleashing the Power of lollms with Ollama Server: A Fun Tech Tutorial 🚀</a>: 🌟 Hey YouTube fam! 🤓 I&#39;m so excited to present my newest video to you all! In this enlightening tutorial, I&#39;ll walk you through the process of installing a...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1224667853735858187)** (137 messages🔥🔥): 

- **Query on Account Bans and Tool Restrictions**: A user questioned an instaban, asking for clarification whether both API and web level accounts are permitted. Another mentioned that a tool like *worldsim* can generate content disallowed by Anthropic.

- **Jamba Model Tuning Experience Shared**: Lhl shared results of tuning a *jamba model* over the weekend using the [shisa-v1 bilingual tuning set](https://huggingface.co/datasets/augmxnt/ultra-orca-boros-en-ja-v1), despite the *"marginal results"*. Direct links to the [training scripts and configurations](https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228) are provided, with an admission that results for JA MT-Bench were not spectacular.

- **Inquiry on Foundational NLP Papers**: A user searched for foundational papers in NLP, having finished "Attention Is All You Need." Responses included a recommendation to watch all of Andrej Karpathy's YouTube videos.

- **Issue Sharing Models on Hugging Face**: Mvds1 reported a problem with uploading models to Hugging Face due to metadata issues with *safetensors.sharded* key and shared a workaround from a discussion that involves manually adding a `sharded: None` parameter to the `SafeTensorsInfo` definition.

- **Discussing Novel LLM Compression Mechanisms**: A lively discussion about theoretical and fringe methods for LLM efficiency ensued, touching on the use of solvers like Coq for enhancing model compression, with references to works by Goertzel on using paraconsistent probabilistic logic for AGI. Specific studies discussed include the concept of interiorizing a *PDLU: Proof Driven Logic Unit* within an LLM and the potential of (DSPy + Solver) Hylomorphic Recursor to achieve significant model compression.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19928">DiJiang: Efficient Large Language Models through Compact Kernelization</a>: In an effort to reduce the computational load of Transformers, research on linear attention has gained significant momentum. However, the improvement strategies for attention mechanisms typically nece...</li><li><a href="https://x.com/p00ssh/status/1775185708887539864?s=20">Tweet from poosh (e/λcc) (@p00ssh)</a>: attention is what you need, anon</li><li><a href="https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228">shisa-ai/shisa-jamba-v1-checkpoint-4228 · Hugging Face</a>: no description found</li><li><a href="https://www.archives.gov/citizen-archivist">Citizen Archivist</a>: One day all of our records will be online. You can help make it happen. You can become a citizen archivist — just click one of the options below to get started.      You Can Tag It! Add tags to images...</li><li><a href="https://tenor.com/view/unchained-foxx-silent-django-gif-4956511">Unchained Foxx GIF - Unchained Foxx Silent - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/sam_paech/status/1770497691593974021?s=46">Tweet from Sam Paech (@sam_paech)</a>: New automated creative writing benchmark using Claude-3-opus as judge: https://eqbench.com/creative_writing.html  More info: https://eqbench.com/about.html</li><li><a href="https://arxiv.org/abs/2012.14474">Paraconsistent Foundations for Probabilistic Reasoning, Programming and Concept Formation</a>: It is argued that 4-valued paraconsistent truth values (called here &#34;p-bits&#34;) can serve as a conceptual, mathematical and practical foundation for highly AI-relevant forms of probabilistic log...</li><li><a href="https://youtu.be/Y94tw4eDHW0?si=cbH5-LV2dkXkkb0_&t=549">Programming Foundation Models with DSPy / Multivector Semantic Search with ColBERT - Omar Khattab</a>: Omar Khattab is a PhD Candidate at Stanford University and an Apple Scholar in AI/ML. In this conversation, Omar explains how to program foundation model pip...</li><li><a href="https://www.youtube.com/watch?v=ZYf9V2fSFwU">AI Pioneer Shows The Power of AI AGENTS - &quot;The Future Is Agentic&quot;</a>: Andrew Ng, Google Brain, and Coursera founder discusses agents&#39; power and how to use them. Join My Newsletter for Regular AI Updates 👇🏼https://www.matthewb...</li><li><a href="https://github.com/YuchuanTian/DiJiang">GitHub - YuchuanTian/DiJiang: The official implementation of &quot;DiJiang: Efficient Large Language Models through Compact Kernelization&quot;, a novel DCT-based linear attention mechanism.</a>: The official implementation of &quot;DiJiang: Efficient Large Language Models through Compact Kernelization&quot;, a novel DCT-based linear attention mechanism. - YuchuanTian/DiJiang</li><li><a href="https://huggingface.co/datasets/aneeshas/imsdb-genre-movie-scripts">aneeshas/imsdb-genre-movie-scripts · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/fnlp/character-llm-data">fnlp/character-llm-data · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/Oobabooga/s/ApIzWEdZu7">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/TheBritishLibrary/blbooks">TheBritishLibrary/blbooks · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/storytracer/US-PD-Books">storytracer/US-PD-Books · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1224666673450192916)** (34 messages🔥): 

- **Exploring Agent Research**: A member with a background in cognitive science and reinforcement learning suggested that efficient exploration by calibrating what the LLM already knows is an under-explored area in agent research.
- **Hermes 2 Pro Gathers Praise**: After testing Hermes 2 Pro, a user commended the model, particularly its function calling capabilities, which performed reliably in large chat sessions without hallucinating about non-existent tools.
- **Multilingual LLM Training Clarified**: In response to questions about LLM training on multiple languages, it was clarified that **Mistral** is primarily pretrained in English, with some European languages, but finetuning training data contains minimal non-English content. The model's coherence in other languages could be attributed to language snippets present in the predominantly English training set.
- **JSON Streaming for Function Calling**: A user curious about streaming parameters for function calling was directed to the oboe.js library, which provides a streaming JSON parsing technique.
- **Genstruct 7B Touted for Instruction Generation**: In a discussion about generating synthetic data in different domains, members suggested using Genstruct 7B, an instruction-generation model designed to create valid instructions from raw text corpuses, as a reference point for crafting diverse instructional data for fine-tuning purposes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B/discussions/7">NousResearch/Nous-Hermes-2-SOLAR-10.7B · Is added_tokens.json missing?</a>: no description found</li><li><a href="https://github.com/jimhigson/oboe.js/">GitHub - jimhigson/oboe.js: A streaming approach to JSON. Oboe.js speeds up web applications by providing parsed objects before the response completes.</a>: A streaming approach to JSON. Oboe.js speeds up web applications by providing parsed objects before the response completes. - jimhigson/oboe.js</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>: The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considera...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1224728581096345641)** (3 messages): 

- **Expressions of Intent**: A member conveyed enthusiasm, possibly in response to an ongoing discussion or recent update in the project.
- **Dataset Development Potential**: The same member acknowledged the potential for building a dataset, implying a connection to the work or topic discussed within the channel.
- **Acknowledgement of Time Restraints**: This member also apologized for not having had time to try out something likely related to the project.
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1224713092542365716)** (2 messages): 

- **Huggingface Model Upload Issue**: A member reported a problem with uploading to the chain, pinpointing the cause as Huggingface automatically adding a `safetensors.sharded = true/false` key to the model metadata. This key is not recognized by the Huggingface Python library, creating an obstacle in the model upload process due to the inability to load `ModelInfo`.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1224642070178697216)** (7 messages): 

- **Scratchpad's Niche in Workflows**: *gabriel_syme* discussed the value of using a scratchpad for intermediate results in workflows, mentioning a specific use case where `notes` function as a scratchpad for users.
- **Glaive's RAG Sample Dataset Released**: *sahilch* shared a link to a newly created sample dataset by Glaive that could aid in RAG data generation, available at [GlaiveAI's RAG Sample on Hugging Face](https://huggingface.co/datasets/glaiveai/rag_sample).
- **DiscoResearch Synthesizes Advanced RAG Data**: *bjoernp* from ellamind/DiscoResearch highlighted their efforts on synthetic data generation for advanced RAG applications, expressing interest in collaborating to develop a robust and varied dataset.
- **Vision of RAG with Enhanced Functionality**: *bjoernp* touted the potential of integrating RAG with function calling capabilities, enabling an LLM to manage query decomposition, multi-search coordination, and dynamic retrieval strategies.
- **Ellamind's Early RAG Dataset and Intentions**: *rasdani* introduced ellamind/DiscoResearch's preliminary RAG dataset in German and outlined their aspirations for contributing to the finetuning and enhancement of RAG capabilities, showing enthusiasm for Nous Research's previous work.

**Link mentioned**: <a href="https://huggingface.co/datasets/glaiveai/rag_sample">glaiveai/rag_sample · Datasets at Hugging Face</a>: no description found

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1224672342492581958)** (88 messages🔥🔥): 

- **Creative Competitions with WorldSim**: Members mused about a competitive platform for **WorldSim**, proposing races to achieve specific states in simulated worlds, embracing complexity, and discussing the role of rules and judges, evidencing a keen interest in gamified simulations. They referenced a [Twitter post](https://x.com/karan4d/status/1768836844207378463?s=20) as a source for the **WorldSim system prompt**, and shared a [Pastebin link](https://pastebin.com/Gj7CpdSE) for easy access.
  
- **Potential WorldSim Features Discussed**: Several enhanced features for WorldSim were envisioned, such as **text-to-video integration**, possibly using an open-source project like **ModelScope's MotionAgent**, and **persistent user entities and data** for deeper interaction with the simulations. Some proposed advanced concepts involved **read/write privileges into an actual kernel**, creating a multiversive experience for users.

- **Roadmapping and Communication**: There was talk about creating a **community-driven roadmap and newsletter** for WorldSim to inform users of potential updates and a desire for clearer communication on **WorldSim's development**. Suggestions arose for using visual organization tools and updates like the **Dwarf Fortress roadmap** shared in a [link](https://www.bay12games.com/dwarves/dev.html).

- **Technical Troubleshooting and Enhancements**: Suggestions for improving WorldSim included **ease of copy/pasting** within the simulator, managing resource slowdowns, and **saving/loading** simulation states. Users volunteered various solutions, sharing their experiences with different versions of WorldSim integrated into platforms like **Copilot** and **AI Dungeon**.

- **Diverse Contributions and Resources**: The community shared and appreciated a variety of resources, such as the **WorldSim Command Index on Notion**, and engaged in light-hearted banter, welcoming fellow users to a "digital afterlife". They also encountered issues with **spam flags** incorrectly applied to user profiles during their interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://lostpedia.fandom.com/wiki/Hanso_Foundation">Hanso Foundation</a>: The Hanso Foundation is an organization founded by Alvar Hanso, whose aim was to &quot;reach out to a better tomorrow&quot; by researching ways to preserve human life and promote well-being. It was es...</li><li><a href="https://en.wikipedia.org/wiki/Core_War">Core War - Wikipedia</a>: no description found</li><li><a href="https://copilot.microsoft.com/sl/j7kIWW89XQ4">Microsoft Copilot: vaš svakodnevni AI pomoćnik</a>: Microsoft Copilot koristi moć umjetne inteligencije za poticanje produktivnosti, otključavanje kreativnosti i bolje razumijevanje informacija uz jednostavno iskustvo čavrljanja.</li><li><a href="https://arxiv.org/abs/2402.19459">Anomalous contribution to galactic rotation curves due to stochastic spacetime</a>: We consider a proposed alternative to quantum gravity, in which the spacetime metric is treated as classical, even while matter fields remain quantum. Consistency of the theory necessarily requires th...</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D&#039;s WorldSim System Prompt Open Source - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://x.com/karan4d/status/1768836844207378463?s=20">Tweet from mephisto 🤡7 (@karan4d)</a>: im opensourcing worldsim of course i am  worldsim sysprompt and conversation to intitialize:  sysprompt:  &lt;sys&gt;Assistant is in a CLI mood today. The human is interfacing with the simulator direc...</li><li><a href="https://www.bay12games.com/dwarves/dev.html">Bay 12 Games: Dwarf Fortress</a>: no description found
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1224648828888813569)** (170 messages🔥🔥): 

- **LM Studio and Embedding Models Are Not Friends**: Users clarified that **LM Studio** cannot currently support embedding models, pointing out that *embedding models aren’t supported yet*.
- **Issues with Running LLM Studio on Certain CPUs**: Discussed that **LLM Studio** installation problems might occur on processors that do not support **AVX2 instructions**, citing an older beta version available but noting that it's deprecated and not highly supported.
- **Troubleshooting Model Loading Errors**: Several members confronted errors when trying to load models into LM Studio, and advice included looking at presets lists, revising config files, and posting in specific help channels with system specs for further assistance.
- **Usage of Local Server and Stability Concerns**: Conversation included praises about the local server mode, while others expressed struggles with LLM's degrading performance or inability to maintain context in conversations, with suggestions to adjust context size and investigate logging.
- **GPU performance and multi-user environment handling**: Inquiries about hardware requirements for running models in LM Studio arose, with mentions of settings to offload GPU layers, and discussions on the feasibility of handling multiple users' chat requests in parallel, recommending enterprise-level solutions like **Nvidia DGX H100 servers** for companies.

**Link mentioned**: <a href="https://useanything.com/">AnythingLLM | The ultimate AI business intelligence tool</a>: AnythingLLM is the ultimate enterprise-ready business intelligence tool made for your organization. With unlimited control for your LLM, multi-user support, internal and external facing tooling, and 1...

  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1224656872716505259)** (13 messages🔥): 

- **Databricks Download Dilemma**: A member inquired about downloading **databricks/dbrx-instruct** into LM Studio but was informed that it is currently unsupported and resource-intensive, even failing to load in Apple MLX with 128gb M3 Max.
- **Model for Hentai Anime Recommendations Sought**: A user asked for a model capable of recommending hentai anime, but was advised to use **MyAnimeList (MAL)** as a conventional alternative and provided with the link: [myanimelist.net](https://myanimelist.net/).
- **Hentai Recommendation Query Draws Humor**: The community reacted with humor to the request for a model specializing in hentai anime recommendations, appreciating the user's audacity.
- **Training LLMs with System Prompts**: There was a discussion about the possibility of using the outputs of an LLM with a complex System Prompt to train another LLM to inherit this prompt's functionality, which could work as a form of model fine-tuning.
- **Odd Response from Employer's Model**: A member reported strange behavior from their employer's model, which consistently provided a non-relevant response related to completing a crossword puzzle, hinting at a possible issue with presets.

**Link mentioned**: <a href="https://myanimelist.net/">MyAnimeList.net - Anime and Manga Database and Community </a>: Welcome to MyAnimeList, the world&#039;s most active online anime and manga community and database. Join the online community, create your anime and manga list, read reviews, explore the forums, follo...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225000508557754408)** (3 messages): 

- **Embedding Models Inquiry**: A member asked about using **embedding models** with LM Studio and mentioned downloading an **SFR embedding gguf model** from Hugging Face.
- **Embedding Support Currently Unavailable**: In response, another participant clarified that *embedding models are unsupported* at this current time within LM Studio.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1224685466885034054)** (69 messages🔥🔥): 

- **Debunking SLI myths for LM Studio**: Discussion clarifies that SLI is **not required** to use two GPUs and has been phased out post-3090 generation, with members confirming good performance running LM Studio with multiple GPUs without SLI, including configurations like 2x 3090s and 2x 48GB RTX8000s.
- **P40 GPUs Attract Interest**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) about the performance of the Nvidia Tesla P40, while another outlined a detailed build using three P40s, capable of running a 70B parameter model efficiently.
- **Performance Surprises in LM Studio**: Users reported significant performance differences between systems, with one noting an **AMD system running slower** than expected. However, switching to ROCm preview showed a performance jump to about **65 tokens/sec**, indicating that software and driver choices can have a drastic impact on performance.
- **Considering GPU Upgrades for Faster LLM Responses**: A user contemplating a hardware upgrade for improved performance with LLMs was advised that a **4090 GPU and a PSU upgrade** would be sufficient, without a need for CPU changes.
- **Concerns Over Future Hardware Prices**: Discussion touched on potential impacts on GPU and Mac pricing following a [major earthquake](https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake) at TSMC production lines, suggesting these items could become more expensive or scarce.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://rentry.org/Mikubox-Triple-P40">Mikubox Triple-P40 build</a>: Dell T7910 &quot;barebones&quot; off ebay which includes the heatsinks. I recommend the &quot;digitalmind2000&quot; seller as they foam-in-place so the workstation arrives undamaged. Your choice of Xe...</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306#:~:text=The%20card%20also%20has%2072,MHz%20(14%20Gbps%20effective).">NVIDIA Quadro RTX 8000 Specs</a>: NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1224718735068364962)** (3 messages): 

- **Troubleshooting LM Studio with Autogen**: A user encountered an issue where Autogen is only returning a couple of tokens and then stops. They are unsure if special steps are needed for proper integration between LM Studio and Autogen.
- **Model and API Specifications Matter**: Another member hinted that the problem might be due to the incorrect model name and possibly omitting the API type in the configuration. They suggest checking the model details section in LM Studio for accurate information.
- **API Type is Critical**: It was confirmed that specifying the API type is essential for LM Studio to work with Autogen.
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1224968419774955583)** (3 messages): 

- **Troubleshooting LM Studio Connection**: A member reports successfully integrating a project with **OpenAI GPT-4** but faces issues when connecting it to **LM Studio** for local model usage. The local model, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf," does not return a response despite the LM Studio Server indicating a streaming response.

- **CORS Might Be the Culprit**: In response to the connection issue, another member suggests enabling **CORS** as a possible solution to the problem with **LM Studio** and **crewai** communication.

- **Helpful Resource for Integration**: For further assistance with implementing LM Studio in crewai, a member provides a helpful [Medium article](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee) guide.
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1225128391813103706)** (1 messages): 

- **DALL·E Images Now Editable in ChatGPT**: Users can now edit **DALL·E** images directly in **ChatGPT** across web, iOS, and Android platforms. Additionally, getting *inspiration on styles* when creating images with DALL·E in GPT is now possible.
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1224671467820683334)** (173 messages🔥🔥): 

<ul>
<li><strong>Bing API Down for Hours</strong>: Users reported the <strong>Bing API</strong> being down for 12 hours, impacting services related to DALL-E and Bing Image Creator.</li>
<li><strong>App Accessibility Issues on Android</strong>: A member expressed frustration with being unable to access an app on their Samsung Galaxy Note 9, mentioning error messages such as "request is not allowed" and the app being listed as incompatible in the Google Play Store.</li>
<li><strong>GPT's Simulated Emotions Trigger Debate</strong>: A discussion unfolded about whether LLMs like GPT can truly simulate emotions, resulting in a comparison to psychopathy and the Eliza effect, and emphasizing the lack of a "motivation engine" in current AI models.</li>
<li><strong>OpenAI's Slow Roll-Out of Promised Features</strong>: Users discussed their dissatisfaction with OpenAI's perceived pattern of announcing new tools and features, such as a memory system, without following through on providing broad access, particularly to paying subscribers.</li>
<li><strong>Defining the Line Between Simulation and Sentience</strong>: The chat touched on the limitations of current AI in simulating emotions, with references to similar conceptual problems in neuroscience, and calls for a more refined understanding of consciousness to inform AI development.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.17567">Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models</a>: With LLMs shifting their role from statistical modeling of language to serving as general-purpose AI agents, how should LLM evaluations change? Arguably, a key ability of an AI agent is to flexibly co...</li><li><a href="https://arstechnica.com/tech-policy/2024/03/microsoft-compares-nyts-openai-lawsuit-to-movie-studios-trying-to-kill-the-vcr/">Microsoft argues Supreme Court’s VCR ruling should doom NYT’s OpenAI lawsuit</a>: Microsoft: Copyright law &#34;no more an obstacle to the LLM than it was to the VCR.&#34;</li><li><a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00567/116616/How-Much-Do-Language-Models-Copy-From-Their">How Much Do Language Models Copy From Their Training Data? Evaluating Linguistic Novelty in Text Generation Using RAVEN</a>: Abstract. Current language models can generate high-quality text. Are they simply copying text they have seen before, or have they learned generalizable linguistic abstractions? To tease apart these p...</li><li><a href="https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators">Simulators — LessWrong</a>: Thanks to Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, and Jonathan Low for feedback on draf…
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1224681472775032882)** (57 messages🔥🔥): 

- **DALL-E's Inpainting Tease**: Members are discussing the new editing feature in DALL-E that allows for style suggestions and inpainting, editing specific parts of an image. This feature seems to only be available to Plus plan members or above, and its rollout isn't complete, as some users report being unable to access it.

- **ChatGPT Performance Discussions**: In the community, there are varying opinions and experiences regarding the performance of different models such as GPT-4 and Anthropic's Opus. One finds GPT-4 better in reasoning tasks and more consistently coherent, while another suggests that Opus outperforms GPT-4 in some areas.

- **Utilizing Custom GPTs**: A lively debate is happening about the use of custom GPTs versus the base ChatGPT model. While some enjoy the efficiency these tailored models bring to the table, one user prefers the flexibility and direct interaction with the base model. 

- **Exploring Custom Prompt Engineering**: The discussion has touched on the advantages of custom GPTs for complex prompt construction. Users are sharing techniques on chaining prompts together using the builder menu and contrast the ease of custom GPTs with the process of instructing the base GPT model. 

- **Plus Plan Perks**: Users with Plus plans are inquiring how to use new Plus features like image editing since the feature isn't clearly available or functioning for everyone. The feature should present a noticeable button for editing after selecting an image if the rollout has reached the user’s account.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1224771961897484319)** (11 messages🔥): 

- **Seeking Manager Replacement Prompts**: A member is looking for prompts to manage tasks like dividing directives and performance plans for middle management and C-suite roles. No specific suggestions or responses were provided in the discussion.
- **Numpy Novice Needs a Hand**: Amayl_ expressed difficulty with numpy, mentioning that it's related to a machine learning course they are taking. They asked for assistance but did not provide details of the exercise in question.
- **Troubleshooting ChatGPT for Exercise Help**: Eskcanta suggested asking ChatGPT, even the 3.5 model, for help with Amayl_'s exercise by copying and pasting the exercise details. No follow-up on this suggestion was given.
- **Markdown Translation Conundrums**: Mat_adore is facing issues with translating markdown text, where responses in Arabic are inconsistently translated or not translated at all. They shared several variations of their prompt with the goal of preserving markdown, links, and proper names.
- **Prompt Engineering Frustrations Amplify**: Mat_adore adjusted their translation prompts multiple times to address issues with markdown and proper language conversion but continues to face challenges, expressing frustration with inconsistent results.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1224771961897484319)** (11 messages🔥): 

- **Seeking Managerial Prompt Guidance**: A user inquired about effective prompt strategies for **manager replacement** tasks, particularly catered to middle and C-suite management, which involves dividing up directives and performance plans.
- **ML Course Numpy Assistance Requested**: A member asked for assistance with a **numpy** exercise related to their machine learning course but did not provide specific details about the issue they are facing.
- **Translation Troubleshooting**: One user reported inconsistent results when translating markdown content into different languages; for some languages like Arabic, the output was occasionally untranslated, leading to frustration. They are seeking a prompt modification that ensures **consistent translation while preserving markdown formatting**.
- **Markup Preservation a Challenge**: The same user attempted various prompt iterations to maintain markdown markup and proper translation but continued to experience issues—specifically with language maintenance and appropriate markdown formatting in the translated text.
- **Quest for a Foolproof Translation Prompt**: Continuous efforts to craft an accurate translation prompt for **markdown content** have led to mixed results, with the user still facing challenges in achieving translation consistency and correctness in the target language while preserving both links and markdown markup.
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1224686984484753510)** (148 messages🔥🔥): 

- **Farewell to a GPU Giant**: John Bridgman retires from AMD, recognized for his contribution to getting a driver upstreamed into the Linux kernel. George Hotz comments on his impact and expresses skepticism about AMD’s management and their handling of GPU issues, inviting anonymous insights from AMD employees for a potential blog post. See discussions on [Phoronix](https://www.phoronix.com/news/AMD-Bridgman-Retires) and a [Twitter thread](https://twitter.com/__tinygrad__/status/1775264921468764205).
  
- **Open GPU Challenges and Promises**: The AMD team's perceived inaction on GPU drivers and open-source commitments sparks debate; George Hotz highlights a history of unfulfilled promises and contends that significant cancellations might be the needed wake-up call for AMD. There's cautious optimism on an open-source approach marked by an [AMD tweet](https://twitter.com/amdradeon/status/1775261152987271614), but credibility of their commitments is under scrutiny.

- **Kernel Discussions and Distro Evolution**: The conversation moves to discuss the implications and support challenges of various kernel versions tagging Intel's Xe and i915 drivers, and the potential move from Ubuntu 22.04 LTS to 24.04 LTS. It wraps up with George Hotz stating he will switch to 24.04 LTS once dependencies align, coinciding with [com.apple's](https://apple.com) migration to 24.04 from 20.04.

- **Logo Redesign Contributions**: The community engages in updates to the tinygrad documentation, including the introduction and adjustment of a new SVG logo that adapts to light and dark mode. George Hotz commits the final changes, noting the removal of "excess stuff" and gratitude for the discovery of the 'source media' attribute helpful in the update.

- **NVIDIA Open GPU Driver Speculations**: George Hotz shares a link to his own contribution to an open NVIDIA driver, clarifying it’s not the Nouveau driver but instead [NVIDIA's open GPU kernel modules](https://github.com/NVIDIA/open-gpu-kernel-modules). This stirs up a discussion on the comparative merits and support of open GPU drivers across different hardware manufacturers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.phoronix.com/review/intel-xe-benchmark/)">Tweet from Trying Out &amp; Benchmarking The New Experimental Intel Xe Linux Graphics Driver - Phoronix</a>: no description found</li><li><a href="https://www.phoronix.com/news/AMD-Bridgman-Retires">Tweet from AMD&#039;s Longtime Open-Source Linux Graphics Driver Advocate Retires - Phoronix</a>: no description found</li><li><a href="https://tinygrad.org">">no title found</a>: no description found</li><li><a href="https://fedoramagazine.org/contribute-at-the-fedora-linux-test-week-for-kernel-6-8/">Contribute at the Fedora Linux Test Week for Kernel 6.8 - Fedora Magazine</a>: Announcing the Fedora test week for kernel 6.8 and requesting participants</li><li><a href="https://lwn.net/ml/dri-devel/20221222222127.34560-1-matthew.brost@intel.com/">[RFC PATCH 00/20] Initial Xe driver submission [LWN.net]</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4044">[WIP] nv driver by nimlgen · Pull Request #4044 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/NVIDIA/open-gpu-kernel-modules">GitHub - NVIDIA/open-gpu-kernel-modules: NVIDIA Linux open GPU kernel module source</a>: NVIDIA Linux open GPU kernel module source. Contribute to NVIDIA/open-gpu-kernel-modules development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224687461242769478)** (28 messages🔥): 

- **Understanding Tinygrad's Beam Search Heuristic**: A member was inquiring whether the **beam search heuristic for tinygrad** is related to the time it takes, prompting a discussion but no specific conclusion was reached.
- **CommandQueue Sheds Light on Tinygrad's Functionality**: **George Hotz** noted that **CommandQueue** serves as a replacement for the `run_schedule` function within **tinygrad**. For a deep dive, **alveoli3358** shared a [tutorial on the new command queue implementation](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md).
- **Memory Optimization Queries Spark Technical Evaluation**: A member sparked a discussion by questioning if memory could be released during the forward pass specifically for activation functions with inverses. They referenced the [inverse function rule from Wikipedia](https://en.wikipedia.org/wiki/Inverse_function_rule) to illustrate the point further.
- **Towards a More Polished Tinygrad**: In the pursuit of reaching version 1.0, **George Hotz** highlighted the imminent need for more **documentation and tutorials for tinygrad**. He also suggested creating a tutorial similar to ["Write Yourself a Scheme in 48 Hours"](https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours) to help users learn by implementing pieces themselves.
- **Community Engagement and Tutorial Contributions**: Members are actively contributing to **tinygrad's learning resources**, with positive feedback from fellow users. Contributions such as tutorials and live streaming oneself going through the quick start guide are helping users, particularly newcomers, to understand and engage with the technology.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours">Write Yourself a Scheme in 48 Hours - Wikibooks, open books for an open world</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Inverse_function_rule">Inverse function rule - Wikipedia</a>: no description found</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md">tinygrad-notes/commandqueue.md at main · mesozoic-egg/tinygrad-notes</a>: Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/me">me - Overview</a>: me has 45 repositories available. Follow their code on GitHub.</li><li><a href="https://jax.readthedocs.io/en/latest/autodidax.html">Autodidax: JAX core from scratch &#8212; JAX  documentation</a>: no description found
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1224693342177333318)** (93 messages🔥🔥): 

- **Open Interpreter App Development**: Members discussed the potential of an **iPhone app** that communicates with Open Interpreter, referencing [Jordan Singer's Twitter post](https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA). A **React Native app** is in development, around 40% completed, with the repository shared on [GitHub](https://github.com/tyfiero/01iOS) for community collaboration.

- **Accessibility Focus for Open Interpreter**: A member highlighted the significance of a **Conversational UI layer** to assist seniors and people with disabilities, aiming to simplify human-computer interaction by reducing search, click, and data management efforts.

- **Security Alert: Open Interpreter's X account Possibly Compromised**: The Open Interpreter community cautioned against clicking links in suspicious posts from what appeared to be a compromised Open Interpreter X account and encouraged reporting the account to prevent crypto wallet breaches.

- **Community Engagement Reminder**: Mike Bird reminded everyone about the April House Party, providing a Discord event [link](https://discord.gg/fjPmtRk8?event=1221828294811586572) and prompted discussion on how **Open Interpreter** could universally improve the human condition.

- **Interactive Installation Queries Resolved**: One user inquired about installation issues related to **chroma-hnswlib**, and the issue was directed to a more appropriate channel, emphasizing the value of community engagement and shared resolutions for technical snags.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/fjPmtRk8?event=1221828294811586572">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">no title found</a>: no description found</li><li><a href="https://api.openinterpreter.com/">no title found</a>: no description found</li><li><a href="https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA">Tweet from jordan singer (@jsngr)</a>: ✨ talk to your computer remotely from your phone  i call it Teleport</li><li><a href="https://github.com/tyfiero/01iOS">GitHub - tyfiero/01iOS</a>: Contribute to tyfiero/01iOS development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1224681843710754816)** (66 messages🔥🔥): 

- **OI on Android Termux**: A repo has been shared providing instructions for installing the OpenInterpreter on Android devices using Termux, available [here](https://github.com/MikeBirdTech/open-interpreter-termux).
- **Linux Server Hurdles**: Multiple users are experiencing difficulties running the 01 server on various Linux distributions, with issues relating to audio and dependencies like `portaudio19-dev`.
- **Suggestions for Local STT Usage**: It was suggested that to cut down on costs, local Speech-to-Text (STT) could be used instead of cloud services before feeding text outputs to OpenAI, leveraging tools like `Whisper.cpp`.
- **Porting on M5 Cardputer**: Work is underway to port OpenInterpreter to the M5 Cardputer, with updates and branches shared, including a function to send messages to both serial and screen. The relevant GitHub repo can be found [here](https://github.com/Clinteastman/c0mputer).
- **GPT-4 Cost Concerns and Alternatives**: Discussion on the high cost of testing with GPT-4 led to suggestions of more cost-effective alternatives like `gpt-4-turbo` and Claude’s Haiku; concerns are being considered for future model defaults in OpenInterpreter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/package-manager/winget/">Use the winget tool to install and manage applications</a>: The winget command line tool enables developers to discover, install, upgrade, remove and configure applications on Windows computers.</li><li><a href="https://scoop.sh/">no title found</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01/issues/219">Ubuntu 21+ is not supported [wayland] · Issue #219 · OpenInterpreter/01</a>: Some dependencies uses x11 and is not compatible with wayland https://github.com/Kalmat/PyWinCtl?tab=readme-ov-file#linux-notice https://github.com/asweigart/pyautogui/issues?q=is%3Aissue+is%3Aopen...</li><li><a href="https://github.com/Clinteastman/c0mputer">GitHub - Clinteastman/c0mputer: Porting open-interpreter to the M5 Cardputer</a>: Porting open-interpreter to the M5 Cardputer. Contribute to Clinteastman/c0mputer development by creating an account on GitHub.</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-termux">GitHub - MikeBirdTech/open-interpreter-termux: Instructions for installing Open Interpreter on your Android device.</a>: Instructions for installing Open Interpreter on your Android device. - MikeBirdTech/open-interpreter-termux</li><li><a href="https://github.com/m5stack/M5Unified/tree/develop">GitHub - m5stack/M5Unified at develop</a>: Unified library for M5Stack series. Contribute to m5stack/M5Unified development by creating an account on GitHub.</li><li><a href="https://git-scm.com/download/win">Git - Downloading Package</a>: no description found</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>: no description found</li><li><a href="https://ngrok.com/docs/getting-started/?os=linux">Quickstart | ngrok documentation</a>: This quickstart will use the ngrok agent to put your application on</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper: A fast, local neural text to speech system</a>: A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.</li><li><a href="https://dashboard.ngrok.com/get-started/setup/linux">ngrok - Online in One Line</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1224640470337191996)** (2 messages): 

- **Excitement Over Deep Dives**: A member expressed enthusiasm about gaining deeper insights into the underlying mechanisms that drive Large Language Models (LLMs). The use of a rocket emoji underscored the member's excitement for this advanced knowledge.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1224649158565036143)** (67 messages🔥🔥): 

- **Performance Limits of Tinystories**: A discussion highlighted the limitations of the *Tinystories* dataset for model training, mentioning that it begins to saturate around 5M parameters. Members suggested utilizing the `minipile` dataset instead, as it's roughly 4 times larger, although more resource-intensive to process.

- **Interest in AI Competitions**: Community members expressed a desire for EleutherAI to sponsor groups to compete in AI competitions, specifically citing the potential of leveraging the llema models, carperai, and other partners with RLHF expertise. To facilitate competition participation, a suggestion was made to form a group in a specified chat channel and discuss eligibility for compute grants.

- **EAI's Position on Jailbreak Defenses and Unsafe Outputs**: An [arXiv paper](https://arxiv.org/abs/2403.14725) was shared, raising doubts over the effectiveness of existing enforcement mechanisms guarding against "jailbreak" attacks on language models. The paper argued the importance of having a clear definition of unsafe responses for better defense strategies, highlighting the adequacy of post-processing outputs.

- **Seeking PyTorch Interview Tips for Research Engineering Roles**: With members seeking advice for research engineering interviews focusing on PyTorch knowledge, there was consensus on the importance of discussing one's work confidently. Tips included relying on the STAR method for behavioral questions and mastering medium-level coding problems that most candidates would get correct.

- **Public Comments on AI Models**: A link to public comments from [regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001/comment) discussing open AI models was shared, with EleutherAI's comment highlighted for its LaTeX formatting. Some members regretted not contributing, while others nodded to the predominant support for open models and rejection of fearmongering in the comments section.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found</li><li><a href="https://www.regulations.gov/document/NTIA-2023-0009-0001/comment">Regulations.gov</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.14725">Jailbreaking is Best Solved by Definition</a>: The rise of &#34;jailbreak&#34; attacks on language models has led to a flurry of defenses aimed at preventing the output of undesirable responses. In this work, we critically examine the two stages o...</li><li><a href="https://github.com/UpstageAI/evalverse?fbclid=IwAR3IhfKfnHlkHWfmuAKDqcZZIP3mIZE5NfnsxBowp-ZuqiyVSndZfnYVTG4">GitHub - UpstageAI/evalverse: The Universe of Evaluation. All about the evaluation for LLMs.</a>: The Universe of Evaluation. All about the evaluation for LLMs. - UpstageAI/evalverse
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1224720471774466088)** (53 messages🔥): 

- **Exploring LLM Robustness Ideas**: A suggestion to investigate the robustness of safety filters for LLMs was shared, referencing a tweet by **BlancheMinerva** discussing the potential of using refusal examples mixed into finetuning data. The concept aligns with current research indicated by a provided [ArXiv paper](https://arxiv.org/pdf/2402.18540.pdf).
  
- **Monitoring Open Source AI Legislation**: An analysis of California’s SB 1047's impact on open-source AI development was highlighted, with an open letter available for signatures. The bill critiques are extensive, addressing concerns of legal liability and efficiency in the AI field, and the full analysis can be found [here](https://www.context.fund/policy/sb_1047_analysis.html).

- **Discoveries in AI Jailbreaking**: Anthropic's new research on "many-shot jailbreaking," a technique effective on various LLMs including their own, was discussed along with a critique about the originality of the paper's findings on how in-context learning follows power laws. The full paper can be explored on their [research page](https://www.anthropic.com/research/many-shot-jailbreaking).

- **ChemNLP First Paper Published**: The first paper from the ChemNLP project through OpenBioML.org, which may be an important step in AI-driven chemistry, has been made available on [ArXiv](https://arxiv.org/abs/2404.01475).

- **Discussing Gradient Notations in Research**: A conversation about notation for gradients ensued, with suggestions on whether to use partial derivative notation or the nabla symbol depending on whether the gradient refers to model parameters or not. The discussion also touched on preferences for different versions of the epsilon symbol in reports.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.context.fund/policy/sb_1047_analysis.html">SB 1047 Analysis - Context Fund</a>: no description found</li><li><a href="https://x.com/cem__anil/status/1775282571070591220?s=20">Tweet from Cem Anil (@cem__anil)</a>: One of our most crisp findings was that in-context learning usually follows simple power laws as a function of number of demonstrations.  We were surprised we didn’t find this stated explicitly in the...</li><li><a href="https://swe-agent.com/">SWE-Agent</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.01475">Are large language models superhuman chemists?</a>: Large language models (LLMs) have gained widespread interest due to their ability to process human language and perform tasks on which they have not been explicitly trained. This is relevant for the c...</li><li><a href="https://x.com/blancheminerva/status/1774901289773584531?s=46">Tweet from Stella Biderman (@BlancheMinerva)</a>: It&#39;s known that finetuning can incidentally remove RLHF guards https://arxiv.org/abs/2310.03693. Can you solve this by including examples with refusals mixed into the data? Does it matter if those...</li><li><a href="https://x.com/anthropicai/status/1775211248239464837?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Anthropic (@AnthropicAI)</a>: New Anthropic research paper: Many-shot jailbreaking.  We study a long-context jailbreaking technique that is effective on most large language models, including those developed by Anthropic and many o...</li><li><a href="https://www.youtube.com/watch?v=rJIwO31uv5c">Louis Castricato - RLAIF, User Autonomy, and Controllability (Eleuther / Synthlabs)</a>: Talk from the Open-Source Generative AI Workshop at Cornell Tech. Website: https://www.louiscastricato.com/Slides: https://drive.google.com/file/d/14Qldg0E1c...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1224765514140876871)** (4 messages): 

- **Abstract House Conundrum**: A member humorously questioned how a house could be considered somewhere between a **concrete giraffe** and an **abstract giraffe**.
- **Keep Calm and Shrug On**: In response to the abstract/concrete giraffe house conundrum, another member offered a classic internet shrug emoticon as a nonchalant answer.
- **Opportunity Closing for Neel Nanda's MATS Stream**: The admissions procedure for **Neel Nanda's MATS stream** closes in just under 10 days, with a link to the application details on [Google Docs](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn).
- **Cryptic Twitter Mention**: A member shared a tweet, with the context and content of the tweet not included in the message.

**Link mentioned**: <a href="https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn">Neel Nanda MATS Stream -  Admissions Procedure + FAQ</a>: no description found

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1224708347899412482)** (24 messages🔥): 

- **Exploring Multilingual Generative QA**: Participants acknowledged the potential for using **Chain of Thought** (CoT) variants to improve performance on multilingual QA tasks and are considering datasets like **MGSM** and others like `nq_open` or `triviaqa`.
- **Generate Until Task Arouses Interest**: Debugging efforts led to the observation that not many tasks utilize the `generate until` function, with confirmed ones being gsm8k, bigbench, and mgsm. Later, a comprehensive list was found containing tasks that implement `generate until`.
- **Troubleshooting Multi-Choice Output in LM Eval**: There was a discussion about resolving an "index out of range" issue when using multiple-choice outputs for evaluation datasets in a CSV format, hinting at adjusting indexing for the answers.
- **CUDA Error Conundrum on Different GPU Architectures**: A user encountered a `CUDA error: no kernel image is available for execution on the device` when running an older version of the LM Eval Harness on H100 GPUs, while A100 GPUs worked fine. The issue was isolated to not being caused by flash attention.
- **CUDA Error Investigation**: Further investigation into the CUDA error suggested it is not being caused by the `.contiguous()` function, as minimal examples with this operation work correctly. The advice was given to check the device `context_layer` is on to further troubleshoot the issue.

**Link mentioned**: <a href="https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+output_type%3A+generate_until&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1224940664991649832)** (2 messages): 

- **Elastic Pretraining Frameworks**: One user inquired about pretraining frameworks capable of **elastic GPU/TPU adjustment** during training. Another user provided a solution using [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/quickstart.html), which allows jobs to run fault-tolerantly with a specified number of restarts, and can handle joining nodes in an elastic fashion.

**Link mentioned**: <a href="https://pytorch.org/docs/stable/elastic/quickstart.html">Quickstart &mdash; PyTorch 2.2 documentation</a>: no description found

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225114862640959720)** (3 messages): 

- **Set Your Visibility**: Enterprise organizations on 🤗 now have the option to set default repository visibility to public, private, or private by default. [Check out the tweet for more info](https://twitter.com/julien_c/status/1772688542289822073).
- **Quarto Publishing**: Quarto! now allows users to deploy sites on Hugging Face with a simple command `use quarto publish hugging-face`. Detailed instructions can be found in these [Twitter](https://twitter.com/gshotwell/status/1772661727856914720) and [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29) posts.
- **New HF Enterprise Page and HuggingChat Updates**: A new HuggingFace Hub Enterprise page is live, and HuggingChat assistants now support custom settings for generation parameters. Discover the [new Enterprise page](https://x.com/victormustar/status/1772742275744850137) and [HuggingChat features](https://x.com/victormustar/status/1772993404437205289).
- **Fine-Grained Control & GGUF on the Hub**: There's now fine-grained access control per repo for Enterprise orgs, and GGUF support updates on the Hub have been implemented. Find out more about access control in this [tweet](https://twitter.com/Thom_Wolf/status/1770504033452573077) and GGUF updates in this [status post](https://x.com/lunarflu1/status/1775232743070220559).
- **Datasets 2.18.0 Released**: The release of Datasets version 2.18.0 brings new features, JSON builder support, and ensures compatibility with PyTorch data types. [Explore the new release](https://github.com/huggingface/datasets/releases/tag/2.18.0).
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1224649181352824843)** (70 messages🔥🔥): 

- **Searching for Multilingual Image-Captioning Models**: A user inquired about pretrained image-captioning models that support multiple languages, including Portuguese, but no specific solutions were given.
- **Stable Diffusion for Photo Lighting**: A discussion around using **Stable Diffusion** to equalize lighting in photos took place, with a member pointing to normalization of luma instead of manipulating the image texture directly. The conversation included the desire to batch process images with various lighting biases.
- **Precision Goals in NLP Project**: Members engaged in a discussion about acceptable precision levels for NLP projects, with one user questioning if 0.68 precision is good enough for a first project. Another suggested aiming for at least 80% precision.
- **Fine-Tuning Challenges and Solutions**: Users shared experiences and challenges related to fine-tuning **Mistral** models, with references to successfully fine-tuned versions like [Mistral Alpaca LoRA](https://huggingface.co/JoPmt/mistral_alpaca_lora) and tips on using Google Colab for the process.
- **Summarization Pipeline Tweaks for Brevity**: One user sought advice on generating shorter summarizations using the Hugging Face summarization pipeline. The conversation included hints to adjust `max_new_tokens` rather than `max_length` to avoid truncated outputs, with more discussions directed to Hugging Face's Discord channels.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/wav2vec2_audiocls.ipynb#scrollTo=9_TjMTIGL46g">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/JoPmt/mistral_alpaca_lora">JoPmt/mistral_alpaca_lora · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation">Text generation strategies</a>: no description found</li><li><a href="https://pytorch.org/blog/inside-the-matrix/?hss_channel=tw-776585502606721024">Inside the Matrix: Visualizing Matrix Multiplication, Attention and Beyond</a>: Use 3D to visualize matrix multiplication expressions, attention heads with real weights, and more.  </li><li><a href="https://www.reddit.com/r/photoshop/comments/r7c2bh/evenout_lighting_for_a_tileable_texture/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/huggingface/cookbook">GitHub - huggingface/cookbook: Open-source AI cookbook</a>: Open-source AI cookbook. Contribute to huggingface/cookbook development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth#installation-instructions---conda">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1224855264096948304)** (2 messages): 

- **Exploring the Command Line Universe**: A channel member shared a [YouTube video](https://www.youtube.com/watch?v=PKYPKRoCW2c) titled "Super User Do - Tinkering with Linux commands, Containers, Rust, and Groq", offering an introduction to navigating a computer using the command line interface (CLI).
- **A New Perspective on Scaling**: Discussion hinted at an unidentified subject that is scaled **exponentially** rather than linearly, although specific details were not provided.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=PKYPKRoCW2c">Super User Do- Tinkering with Linux commands, Containers, Rust, and Groq</a>: A brief intro for basic commands to navigate your computer from what&#39;s called the &quot;command line interface&quot; or &quot;CLI&quot;. How to update, upgrade, move in and out ...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1224700005152194570)** (5 messages): 

- **Innovations in Text Generation**: A Medium article was shared discussing **IPEX-LLM and LlamaIndex** highlighting their potential to shape the future of text generation and chat applications. Read about these advances in the full article [here](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2).
- **Testing the Waters of LLM Security**: A new suite for red teaming has been developed to test the vulnerabilities of LLMs, with a specific focus on **DBRX and Jamba**. Details of their findings are mentioned in the shared [tweet](https://x.com/divyanshutwt/status/1775241719740535149?s=20).
- **Educational Watch: GPT Demystified**: A YouTube video from 3blue1brown titled "But what is a GPT?  Visual intro to Transformers" offers an engaging explanation of transformers and GPT architectures. Preview the educational content [here](https://www.youtube.com/watch?v=wjZofJX0v4M), with acknowledgments to the video’s supporters.
- **Apple Claims AI Supremacy Over OpenAI**: A short notice revealed that **Apple** announced its latest model being more powerful than **OpenAI's GPT-4**, without providing additional details or supporting evidence.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/divyanshutwt/status/1775241719740535149?s=20">Tweet from Divyanshu (@divyanshutwt)</a>: At @enkryptai we&#39;ve build a red teaming suite to identify the pitfalls of LLMs. Recently, we tested the vulnerability of @databricks &#39;s DBRX and 🐍Jamba, a MoE SSM LLM. Got some interesting re...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to Transformers | Chapter 5, Deep Learning</a>: An introduction to transformers and their prerequisitesEarly view of the next chapter for patrons: https://3b1b.co/early-attentionSpecial thanks to these sup...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1224691782516740227)** (14 messages🔥): 

- **Musical Innovation Strikes a Chord**: A new Gradio app called a 'musical slot machine' was created, integrating **Musiclang** for random seed generation or input chords and allowing users to pick from community-made fine-tunes. The result is a form of **text2midi2audio** conversion, highlighted in a [YouTube video](https://www.youtube.com/watch?v=p77U2eyJFPU), and though the app is made for testing fine-tunes, it doubles as a playful instrument for musicians.

- **Bringing Order to Chaos with Hypergraph Visualization**: A **Space** for visualizing high-dimensional hypergraph datasets was constructed, dealing with up to 150k rows and serving as a way to bring sense to the complex information. A concise [link to the Space](https://huggingface.co/spaces/SauravMaheshkar/CornellTemporalHyperGraphDatasets) was shared, along with a reference to the [original collection](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05) and an accompanying [Twitter thread](https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20).

- **Octopus 2 Hooks Developers with Functionality**: A demo for **Octopus 2**, a model capable of function calling, debuted. Although it takes 1500 seconds to render, the model promises new possibilities, especially with excitement building around on-device models, highlighted in the [Space](https://huggingface.co/spaces/Tonic/Octopus/).

- **Local Tune Assembly Hits a High Note**: Discussion highlighted that music models might be better off running locally to improve accessibility and usability, in line with the concept of on-device models being more convenient.

- **GPU Expense Spurs Optimism for CPU Optimizations**: The high cost of GPUs fueled a conversation about the anticipation of significant advancements in CPU optimization for AI and ML applications in the near future.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Octopus/">Octopus - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - a SauravMaheshkar Collection</a>: no description found</li><li><a href="https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20">Tweet from Saurav Maheshkar ☕️ (@MaheshkarSaurav)</a>: I&#39;m working on HyperGraph Representation Learning at the moment and have spent the last few days creating a @huggingface collections consisting of: 👉 processed datasets 👉 papers 👉 @Gradio space...</li><li><a href="https://www.youtube.com/watch?v=p77U2eyJFPU">made a musical slot machine then built a song with it - captains chair 21</a>: 00:00 - start01:35 - building the track08:28 - the trackour first @HuggingFace space. it&#39;s pretty ridiculous.https://huggingface.co/spaces/thepatch/the-slot-...</li><li><a href="https://huggingface.co/spaces/thepatch/the-slot-machine">The Slot Machine - a Hugging Face Space by thepatch</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1224734225853317231)** (3 messages): 

- **Effective Batch Size Optimization**: Seanb2792 mentioned that while computationally the cost might be similar, increasing the effective batch size can be achieved without using additional VRAM. This is particularly valuable as larger batch sizes may enhance the performance of certain models.
- **Batch Size Affects Model Performance**: In tests on medical data, Huzuni found that a **larger batch size generally results in better performance**, even if the improvements are marginal or non-significant.
- **Batch Normalization Draws Concern**: Huzuni also observed that accumulating more than two batches can have a detrimental effect on performance, likely due to batch normalization, based on their latest tests.
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1224841862506872973)** (13 messages🔥): 

- **LLM Fine-tuning on a Budget**: A user is exploring how to build a language model on top of PDFs with limited computational resources, preferring to use inference with open-source models like llama2, mistral, phi, etc. There is an inquiry about the minimum requirements for llm models, mentioning that *phi-2* requires more than 10GB of free space to run on a PC with 16GB RAM.

- **KV Cache Queries in Transformers**: A member asks for use cases or examples related to using KV Cache with HuggingFace, linking to the specific [Dynamic Cache](https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/cache_utils.py#L76) in the transformers' GitHub repository.

- **Changing Special Tokens in Tokenizers**: There's a discussion on how to modify special tokens in a tokenizer when fine-tuning an LLM. A member provided a solution to add new special tokens using `tokenizer.add_special_tokens(special_tokens)` and another advised changing the tokenizer's dictionary directly but cautioned about potential merges during tokenization.

- **Issues with Multinode Fine-tuning**: A user experiences a timeout while trying to finetune llama2 using multi-node from Docker with deepspeed and axolotl. Despite having proper communication between nodes and visible GPUs in their stack, the fine-tuning process freezes with the given deepspeed command.

- **Calls for Structured Training Examples**: A user struggles with training GPT2 for text summarization, encountering issues like OOM errors and stagnating validation metrics. They suggest that HuggingFace should provide structured examples on how to perform specific tasks with various models to aid users in their training efforts.

**Link mentioned**: <a href="https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/cache_utils.py#L76">transformers/src/transformers/cache_utils.py at c9f6e5e35156e068b227dd9b15521767f6afd4d2 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225105529903251536)** (8 messages🔥): 

- **Seeking DiT with Enhanced Conditioning**: A user inquired about a modified **DiT (Diffusion Image Transformer)** that supports conditioning with text, images, or other modalities using cross-attention. The only available DiT on Hugging Face Diffusers is conditioned by class, and the original team's source code isn't publicly shared, as noted with a [link to the paper](https://arxiv.org/html/2312.04557v1).
  
- **Cost Concerns of Public DiTs**: A member pointed out that publicly available **DiTs** are class-conditioned because it's more cost-effective compared to cross-attention methods, echoing the discussion about the expense of such models.

- **Exploration of Diffusion Models for Depth Mapping**: A user is considering modifying **Stable Diffusion (SD)** for converting stereo images into depth maps, as the current best public model for such a task is inadequate for their challenge.

- **Potential Modification of Stable Diffusion Architectures**: The user asked if it's possible to fine-tune Stable Diffusion using input images with more than three channels, exploring the feasibility of using **LoRA** or **ControlNet** with StableDiffusion for their task.

- **Advocacy for Modifying SD Over Training from Scratch**: In response to the query, another participant suggested slightly modifying the SD architecture to adapt it for the user's needs, indicating that training from scratch should be a last resort option.
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1224778620971515994)** (1 messages): 

- **Gradio Hits Version 4.25.0**: A new update for Gradio is out, introducing **automatic deletion** of `gr.State` variables for better traffic management and an **unload event** for browser tab closures. The update also features **lazy example caching** with `cache_examples="lazy"` suitable for ZeroGPU, a fix for a bug with streaming audio outputs, and enhancements to `gr.ChatInterface`, including image pasting from the clipboard.
- **Changelog Ready for Review**: The full list of changes and fixes in Gradio 4.25.0 can be explored in the [changelog](https://gradio.app/changelog).
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1224663111391252520)** (104 messages🔥🔥): 

- **Searching for Persistent Chat History Solutions**: A member inquired about techniques for maintaining persistent context when chatting with a database formatted with 'question : answer' pairs, expressing uncertainty about which methods to apply.
- **Structured Tool Validation Query**: A discussion unfolded around validating fields in a `StructuredTool` using LangChain. The conversation mentioned utilizing Pydantic's `BaseModel` and `root_validator` for field validation, referencing specific [Github issues and documentation](https://github.com/langchain-ai/langchain/issues/8066).
- **Exception Handling in Structured Tools**: Members explored strategies on how to catch and display `ValueError` texts in structured tools when error conditions are met, with reference to [Github issues for relevant methods](https://github.com/langchain-ai/langchain/issues/1358).
- **Integrating Langchain with External APIs**: Questions arose regarding the integration of LangChain with Azure API Management (APIM), in particular fetching results with AzureOpenAI, for which a troubleshooting link to a specific [Github issue](https://github.com/langchain-ai/langchain/issues/16930) was suggested.
- **Creating a Database-Connected Appointment Bot**: A member sought assistance for creating a bot in LangChain and Javascript that not only schedules appointments but also handles the storage and retrieval of dates from a database, prompting recommendations for libraries like [Sequelize](https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a) and [node-postgres](https://github.com/brianc/node-postgres/tree/master).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://smith.langchain.com/hub/wfh/web-voyager?organizationId=0ef50744-2e28-5e1f-8f50-1a7afa359cb9]">LangSmith</a>: no description found</li><li><a href="http://localhost:8000.>">no title found</a>: no description found</li><li><a href="https://python.langchain.com/docs/templates/openai-functions-agent#usage>).">openai-functions-agent | 🦜️🔗 Langchain</a>: This template creates an agent that uses OpenAI function calling to communicate its decisions on what actions to take.</li><li><a href="https://python.langchain.com/docs/guides/structured_output">[beta] Structured Output | 🦜️🔗 Langchain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://python.langchain.com/docs/langgraph#how-to-guides>).">🦜🕸️LangGraph | 🦜️🔗 Langchain</a>: Downloads</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/quickstart#quickstart-1>):">Quickstart | 🦜️🔗 Langchain</a>: Overview</li><li><a href="https://js.langchain.com/docs/integrations/llms/azure#llm-usage-example>).">Azure OpenAI | 🦜️🔗 Langchain</a>: Azure OpenAI is a cloud service to help you quickly develop generative AI experiences with a diverse set of prebuilt and curated models from OpenAI, Meta and beyond.</li><li><a href="https://js.langchain.com/docs/integrations/llms/azure#using-the-openai-sdk>).">Azure OpenAI | 🦜️🔗 Langchain</a>: Azure OpenAI is a cloud service to help you quickly develop generative AI experiences with a diverse set of prebuilt and curated models from OpenAI, Meta and beyond.</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html#langchain.chains.structured_output.base.create_structured_output_runnable.">langchain.chains.structured_output.base.create_structured_output_runnable &mdash; 🦜🔗 LangChain 0.1.14</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/1358>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a">GitHub - sequelize/sequelize at 9e141880230a7f2a9a8c1e66a31f29fea7b5a65a</a>: Feature-rich ORM for modern Node.js and TypeScript, it supports PostgreSQL (with JSON and JSONB support), MySQL, MariaDB, SQLite, MS SQL Server, Snowflake, Oracle DB (v6), DB2 and DB2 for IBM i. - ...</li><li><a href="https://github.com/langchain-ai/langchain/issues/8406>):">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16930>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/brianc/node-postgres/tree/master">GitHub - brianc/node-postgres: PostgreSQL client for node.js.</a>: PostgreSQL client for node.js. Contribute to brianc/node-postgres development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/tool_error_handling#tryexcept-tool-call>).">Tool error handling | 🦜️🔗 Langchain</a>: Using a model to invoke a tool has some obvious potential failure modes.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13662>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8066>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/fairseq/tree/nllb">GitHub - facebookresearch/fairseq at nllb</a>: Facebook AI Research Sequence-to-Sequence Toolkit written in Python. - GitHub - facebookresearch/fairseq at nllb</li><li><a href="https://huggingface.co/facebook/nllb-200-distilled-600M">facebook/nllb-200-distilled-600M · Hugging Face</a>: no description found</li><li><a href="https://opennmt.net/CTranslate2/guides/transformers.html">Transformers &mdash; CTranslate2 4.1.0 documentation</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1225024321806667787)** (2 messages): 

- **CI Confusion on Langserve**: A member sought assistance with a failed CI related to a **[pull request #580 on langchain-ai/langserve](https://github.com/langchain-ai/langserve/pull/580)**. They indicated having tested the changes locally with Python 3.10, where all tests passed.
  
- **New Tutorial for Langserve Chat Playground**: A full video tutorial was shared explaining how to utilize the new **Chat Playground** feature of Langserve, especially in cases where it does not work out of the box. Here is the [video link](https://www.youtube.com/watch?v=stWiNP1o2_g), which also includes a showcase of Langsmith and the final code in the description.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=stWiNP1o2_g">The NEW Langserve Chat Playground with Agents | Coding Showcase</a>: In this technical deep dive, we&#39;ll guide you through the exciting world of LangChain and LangServe frameworks. In 17 minutes, we&#39;ll present you with a compre...</li><li><a href="https://github.com/langchain-ai/langserve/pull/580">WIP: Serve playground from correct route if nested APIrouters within one another by StreetLamb · Pull Request #580 · langchain-ai/langserve</a>: Update playground tests to check for the correct playground assets path in index.html. #578
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1224759029935505522)** (7 messages): 

- **Prompt-Breaking Challenge Issued**: A member introduced a tool for automatically generating code transformations to ensure **code quality and standards** for production. Feedback is solicited from **proficient prompters** to test the tool and the link [GitGud LangChain](https://tinyurl.com/gitgud-langchain) was shared for this purpose.
  
- **CallStar AI Voice Apps Launched**: The member announced the launch of several AI voice apps including **CallStar AI**, **Call Jesus AI**, **Call PDF AI**, **Call Tube AI**, **Call Website AI**, and **Call Hacker News AI**. Enthusiasm for voice as the future of AI interaction was expressed, and links to support the project on [Product Hunt](https://www.producthunt.com/posts/callstar), [Reddit](https://www.reddit.com/r/SideProject/comments/1bumj6s/launching_callpdf_5_more_ai_voice_apps_today/), and [Hacker News](https://news.ycombinator.com/item?id=39914442) were provided.

- **AllMind AI Emerges for Financial Analysis**: A new large language model called **AllMind AI** was launched for financial analysis and research. This LLM aims to revolutionize financial research by providing access to insights and comprehensive financial data on a single platform, with promotional links on [AllMind Investments](https://allmindinvestments.com/) and [Product Hunt](https://www.producthunt.com/products/allmind-ai).

- **Galaxy AI Unveiled**: GalaxyAI has announced a **free API service** giving access to premium AI models including various versions of GPT-3.5, GPT-4, and **Gemini-PRO API**, all compatible with Langchain integration and formatted like OpenAI's APIs. They encouraged integration into projects and provided a link for trying the service, though the URL was not included in the message.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found</li><li><a href="https://tinyurl.com/gitgud-langchain">GitGud</a>: no description found</li><li><a href="https://callstar.ai/">CallStar</a>: AI Voice Calls with Characters and Celebrities</li><li><a href="https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios">Mona Bild repost</a>: Bild bekannt aus tiktok -,Mona Bild repost in Wuppertal - Elberfeld-West</li><li><a href="https://allmindinvestments.com/">AllMind AI</a>: no description found</li><li><a href="https://www.producthunt.com/products/allmind-ai"> AllMind AI - Product Information, Latest Updates, and Reviews 2024 | Product Hunt</a>: AllMind AI is a new large language model designed exclusively for financial analysis and research. This LLM revolutionizes financial research by offering users access to insights and providing real-ti...</li><li><a href="https://calljesus.ai/">Call Jesus</a>: Realistic AI Voice Chats with Jesus</li><li><a href="https://callpdf.ai/">CallPDF</a>: Call any PDF - Realistic AI Voice Chats</li><li><a href="https://calltube.ai/">CallTube</a>: Call any YouTube Video - Realistic AI Voice Chats</li><li><a href="https://callwebsite.ai/">Call Website</a>: Call any Website - Realistic AI Voice Chats</li><li><a href="https://callhackernews.com/">Call Hacker News</a>: AI Voice Interface for Hacker News</li><li><a href="https://www.producthunt.com/posts/callstar"> CallStar - Realistic AI voice calls with characters, YT-videos &amp; PDFs | Product Hunt</a>: Next-level AI voice calls! Chat with celebrities, understand your docs with voice &amp; explore spirituality. Make AI conversations feel real and personal with best-in-class AI voices. Call PDFs, YouT...</li><li><a href="https://www.reddit.com/r/SideProject/comments/1bumj6s">Reddit - Dive into anything</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=39914442">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1225149749418659880)** (1 messages): 

- **Your Guide to the LangChain Universe**: A member highlighted the [LangChain Quick Start Guide](https://python.langchain.com/docs/get_started/quickstart) which offers a comprehensive introduction to **LangChain**, including setting up **LangSmith** and **LangServe**, using prompt templates, models, output parsers, and building simple applications.

- **Encountering the 404 Abyss**: When attempting to run **LangChain** code involving `ChatOpenAI` and `ChatPromptTemplate`, a member encountered a `NotFoundError` with a **404 error code** suggesting a "Resource not found" issue. This hiccup occurred during the execution of the member's program in their virtual environment.

**Link mentioned**: <a href="https://python.langchain.com/docs/get_started/quickstart">Quickstart | 🦜️🔗 Langchain</a>: In this quickstart we&#x27;ll show you how to:

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1224668076906385429)** (38 messages🔥): 

- **Comprehensive Mojo Docs Praised**: The [Mojo documentation](https://docs.modular.com/mojo/roadmap#cc-interop) was mentioned to be fairly comprehensive, offering insight into future implementations, including MAX Engine and C/C++ interop, which are expected to enhance development and efficiency.
- **Mojo and Mathematical Variable Names**: A question about Mojo supporting mathematical variable names like Julia led to a clarification that currently Mojo only supports ASCII for variable names and follows Python's convention for variable naming, starting with a character or underscore.
- **Debate on Mojo's Variable Naming with Emojis**: A discussion emerged on whether Mojo supports non-traditional variable names, confirming that emojis and other symbols can be used as variable names if enclosed in backticks.
- **Mojo's Wikipedia Page Needs Updates**: Concerns were raised over the poor state and outdated information on [Mojo's Wikipedia page](https://en.wikipedia.org/wiki/Mojo_(programming_language)), with a recent edit correcting the misunderstanding that Mojo is still proprietary.
- **Code Snippet Troubleshooting**: There was a troubleshooting discussion about a code snippet where `listdir` returned a list of references which needed to be dereferenced using `[]` to allow `print` to work properly, a solution was found and applied successfully.

**Link mentioned**: <a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1224789636891803670)** (3 messages): 

- **Modular Tweets Its Moves**: Modular shared a tweet on their official Twitter handle which can be checked out [here](https://twitter.com/Modular/status/1775225130882564529).
- **Another Tweet from Modular**: Another [tweet](https://twitter.com/Modular/status/1775549728400572660) was posted by Modular on their Twitter account.
- **Modular Continues the Twitter Streak**: Modular posted yet another [tweet](https://twitter.com/Modular/status/1775583583530524987) on their Twitter feed.
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1224783263936024597)** (1 messages): 

- **MAXimum Mojo Momentum**: The Modular **Mojo 24.2** update has been released and details are provided in a recent [blog post](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more). This release is especially significant for Python developers adopting Mojo, offering a line-up of new features and enhancements.

**Link mentioned**: <a href="https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more">Modular: What’s new in Mojo 24.2: Mojo Nightly, Enhanced Python Interop, OSS stdlib and more</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s new in Mojo 24.2: Mojo Nightly, Enhanced Python Interop, OSS stdlib and more

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1225139017121529927)** (4 messages): 

- **Proposing Mojo on ROS 2**: A member suggested integrating [Mojo support into ROS 2](https://github.com/ros2), a widely used robotics middleware framework, with potential benefits due to Mojo's memory safety practices. The ROS 2 community has [native Rust support](https://github.com/ros2-rust/ros2_rust), with a shift towards Rust-based middleware like [Zenoh](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds).

- **Rust vs Python in ROS 2**: It was noted that despite most of the ROS 2 community's preference for Python due to their research backgrounds, Rust offers a compelling alternative in terms of performance and safety.

- **Rewriting Python Code for Performance**: The member mentioned that while many robotics systems are initially written in Python for convenience, they are often rewritten in C++ for speed in serious applications.

- **Mojo's Potential with Nvidia Jetson**: It was pointed out that Mojo could better leverage hardware like Nvidia Jetson products, which are increasingly used in robotics, unlike Python which is limited by the Global Interpreter Lock (GIL).

**Link mentioned**: <a href="https://github.com/ros2-rust/ros2_rust">GitHub - ros2-rust/ros2_rust: Rust bindings for ROS 2</a>: Rust bindings for ROS 2 . Contribute to ros2-rust/ros2_rust development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1225083116880662580)** (2 messages): 

- **Docker Builds on Autopilot**: A fix is prepared for version 24.3 that addresses the solution for **automated docker builds**. Members reacted positively to this news.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1224657765461528577)** (30 messages🔥): 

- **The Perils of Non-Trivial Structs**: It's confirmed that **@register_passable("trivial")** can't be used for structs with memory allocation due to the shared pointer issue, requiring the use of **@register_passable** for proper functionality.

- **Embarking on SIMD Search**: A member aimed to implement SIMD Naïve Search in Mojo but was unclear about implementing 'found' and 'SIMDcompare' functions. A fellow member compared it with native Mojo code for SIMD operations, pointing to **[Mojo's SIMD documentation](https://arxiv.org/pdf/1612.01506.pdf)** as a starting point.

- **Top-Level Code Temporarily Grounded**: Discussion around the introduction of top-level code in Mojo reveals complications without a current estimated time of arrival. The issue with a missing page on the "escaping" operator has been raised, and the documentation team has been pinged.

- **A Decorator's Dilemma**: Custom decorators in Mojo are not yet possible, as they're hardcoded in the compiler; a workaround was shared to manually decorate functions, while acknowledging the limitation.

- **Equality Check Enigma in Iteration**: A scenario where a member tried to check for string equality within a List iteration in Mojo led to a clarification that explicit dereferencing with brackets `x[]` is required due to how Mojo handles Reference types.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/decorators/always-inline">@always_inline | Modular Docs</a>: Copies the body of a function directly into the body of the calling function.</li><li><a href="https://docs.modular.com/search?q=escaping+">Modular Docs</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225176876079911101)** (1 messages): 

- **Logger Library Gets an Update**: The logger library now accepts **arbitrary args and kwargs** for logging messages. The update enhances the functionality, allowing entry of variable information along with log messages like `key=value` or `erroring=True`.
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1224847356277817344)** (7 messages): 

- **Mojo Lagging Behind Go in 1BRC**: A member shared their experience working on the [One Billion Row Challenge](https://github.com/VMois/1brc-mojo) in Mojo language, noting a performance of approximately 23 minutes with optimizations on a MacBook Air M1, significantly longer compared to a Go implementation which completes in around 96 seconds.

- **Searching for a Faster Dict**: The member expressed concerns about the performance of `Dict` in Mojo, considering it does many memory copies and discussing potential improvements including a SIMD version.

- **A New Dict Implementation on the Horizon**: A different member mentioned having a custom `Dict` implementation that is faster than the standard one in Mojo, offering hope for performance improvements.

- **Benchmarking Against Swiss Table**: When asked about comparisons to the swiss table, a member responded that they haven't yet benchmarked against it, and that such a benchmark would need to be written in C++ or Rust.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/VMois/1brc-mojo/tree/main">GitHub - VMois/1brc-mojo: One Billion Row Challenge (1brc) in Mojo language</a>: One Billion Row Challenge (1brc) in Mojo language. Contribute to VMois/1brc-mojo development by creating an account on GitHub.</li><li><a href="https://r2p.dev/b/2024-03-18-1brc-go/#:~:text=One%20Billion%20Row%20Challenge%20in%20Golang%20%2D%20From%2095s%20to%201.96s">One Billion Row Challenge in Golang - From 95s to 1.96s</a>: In the One Billion Row Challenge, the task is to write a program capable of reading an 1-billion-line file (with around 13GB), process and aggregate temperature readings from various weather stations,...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/1225099690790355036)** (2 messages): 

- **Max⚡ and Mojo🔥 24.2 Released**: Last week marked the release of **Max⚡ and Mojo🔥 24.2**, along with the open-sourcing of the standard library and the launch of nightly builds. The community has shown active engagement with approximately 50 pull requests raised and 10 merged; interested users can explore and contribute on [GitHub](https://github.com/modularml/mojo/tree/nightly/stdlib).

- **Explore the Latest in Mojo🔥**: For those keen to dive into the latest updates and contributions, Modular has made available several resources: *The Next Big Step in Mojo🔥 Open Source*, the **Mojo launch blog**, details on **What’s new in Mojo 24.2** including *Mojo nightly, enhanced Python interop, open-source stdlib*, and more.
  - Find Modular's development insights on their blog about [Open Source progress](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source).
  - Discover the new features of Mojo 24.2 by reading the [Mojo launch blog](https://www.modular.com/blog/max-24-2-is-here-whats-new) and the detailed account on [What’s new in 24.2](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more).

**Link mentioned**: <a href="https://www.modular.com/newsletters/modverse-weekly-issue-28">Modverse Weekly - Issue 28</a>: Welcome to issue 28 of the Modverse Newsletter covering Featured Stories, the Max Platform, Mojo, &amp; Community Activity.

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1224732356536045689)** (13 messages🔥): 

- **Path to GitHub Collaboration**: A user suggested using Discord for general discussions and moving more specific topics onto GitHub for collaboration purposes.
- **Importing Python stdlib into Mojo**: A developer inquired whether they could use the Python standard library as a reference for contributing to **Mojo's stdlib**. The response highlighted that this approach would introduce a dependency on the CPython interpreter, contrary to the goal of enabling standalone binaries.
- **Seeking Guidance on Mojo stdlib Development**: A user looking to contribute to the **Mojo stdlib** stated that existing documentation like `stdlib/docs/development.md` was helpful yet found it challenging to begin actual development.
- **Resolving Parsing Errors and Test Failures in stdlib**: One user faced parsing errors and test failures including a `FileCheck command not found` error. Guidance was provided on locating `FileCheck` within WSL and adding it to the path, which resolved the issue.
- **Discussion on `Optional` Behavior in Mojo**: A link to GitHub was shared discussing whether `Optional` can return a reference for `value()` in Mojo's standard library amidst the current behavior of dereferencing the value.

**Link mentioned**: <a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L117-L118).">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1224643934186373151)** (91 messages🔥🔥): 

- **Timeout Concerns & Model Failures**: Messages indicate **NOUSRESEARCH/NOUS-HERMES-2-MIXTRAL** model suffering from failures with error code 524, and one member mentions issues using **TogetherAI's API**, indicating an upstream problem. Another mentions a backup model, **Nous Capybara 34B**, as a potential alternative.
- **Testing LLMs with Historical Questions**: Some members are discussing the varying accuracy of different LLMs in response to a historical prompt involving a Japanese general from WW2. **Isoroku Yamamoto** is identified as the correct answer, but models such as **claude**, **opus**, and **haiku** show mixed results.
- **OpenRouter's Maximum Payload Size**: A discussion about OpenRouter's limitation of **4MB max body size** was highlighted, with confirmation that this limit currently has no workaround.
- **Roleplaying with AI Models**: Members were seeking advice on using various AI models for roleplaying, specifically **Claude 3 Haiku**. The conversation includes recommendations for jailbreaking the models and using few-shot examples to improve performance.
- **Discord Servers for Prompt Resources**: Members looking for prompt examples and jailbroken prompts were directed to **SillyTavern** and **Chub's** Discord servers, where they can find resources such as the suggested **pancatstack jailbreak**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/abhishek/autotrain-mixtral-dgx-cloud-local">Finetune Mixtral 8x7B with AutoTrain</a>: no description found</li><li><a href="https://sillytavern.app/">SillyTavern - LLM Frontend for Power Users</a>: no description found
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1224725406058283039)** (4 messages): 

- **RankZephyr Leads in Advanced RAG**: Recommending specific **rerankers** for advanced Retrieval-Augmented Generation, IFTTT suggests using an **LLM** like **RankZephyr** for better results. An open-source collection of LLMs called **RankLLM** is also highlighted for its excellence in [finetuning for reranking](https://twitter.com/llama_index/status/1775166279911186930).

- **Webinar Unveils AI Browser Copilot Secrets**: A recent webinar featuring @dhuynh95 offered valuable insights into building an **AI Browser Copilot**, emphasizing the **prompt engineering pipeline** and the importance of **KNN few-shot examples** and *vector retrieval*. More details are available on the [LlamaIndex Twitter page](https://twitter.com/llama_index/status/1775264340465381536).

- **Boosting RAG with Time-Sensitive Queries**: **KDB.AI**'s integration with **Retrieval-Augmented Generation (RAG)** allows for **hybrid searching** that combines literal, semantic, and time-series analysis. This enables more accurate results by filtering for relevancy based on a time index, essential for financial reports such as quarterly earnings statements, as showcased in the [shared code snippet](https://twitter.com/llama_index/status/1775269014849359925).

- **Introducing an AI-Powered Digital Library**: The unveiling of a new **LLM-powered digital library** designed for **professionals and teams** promises an advanced system for organizing knowledge. This platform transcends traditional data management by offering features to create, organize, and annotate data in a self-managing digital space as mentioned in [this LlamaIndex tweet](https://twitter.com/llama_index/status/1775537091272937933).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/nbvRS0Cc9Q">IKI AI &#x2013; Intelligent Knowledge Interface</a>: Smart library and&#x2028; Knowledge Assistant for professionals and teams.</li><li><a href="https://t.co/5uVy4hbtSw">Home - KDB.AI</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1224731920492134512)** (45 messages🔥): 

- **PDF Indexing Dilemmas**: A member sought advice on indexing a **2000-page PDF** without using llamaparse, mentioning that current methods were time-consuming. Another member suggested increasing `embed_batch_size` on the embedding model, which was later said to be unhelpful, indicating the need for alternative strategies.

- **Understanding qDrant Lock Files**: One user encountered an issue where **qDrant** wouldn't release a lock after running an **IngestionPipeline**, querying the community if it was an LlamaIndex or a qDrant specific problem. The user received no certain answer, highlighting a gap in collective experience regarding this issue.

- **HuggingFace API Limitations Discussed**: There was confusion about potential rate limits and charges when using **HuggingFaceInferenceAPIEmbedding and HuggingFaceInferenceAPI** with a token. While one member initially thought there were no rate limits, another later confirmed rate limit errors and the possibility of charges by Hugging Face.

- **Integration Challenges with Alternate Models**: A user was trying to integrate a model named **"llama2"** into an LlamaIndex agent and was advised to use the Ollama class, which uses the REST API for interaction. Helpful documentation was shared, and the integration process with **Ollama** was discussed in detail.

- **RAGAs with Recursive Query Engines**: A conversation about the absence of documentation for recursive query engines with RAGAs was raised, leading to a realization of potential issues between **langchain and ragas** and highlighting the need for clearer guidance or fixes in this area.

**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/">Ollama - LlamaIndex</a>: no description found

  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1224699971501297745)** (7 messages): 

- **Exploring the Future of Text and Chat Generation**: A Medium article titled *Unlocking the Future of Text Generation and Chat with IPEX-LLM and LlamaIndex* explores advancements in text generation. The article can be found at [Unlocking the future of text generation](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2).

- **Step-by-Step RAG App Tutorial Shared**: A member shared a link to a YouTube tutorial on building a RAG app using **LlamaIndex, Pinecone, and Gemini Pro**. The tutorial can be viewed at [How to build a RAG app using Gemini Pro](https://youtu.be/B9mRMw0Jhfo).

- **RAG Tutorial Receives Community Support**: Another member expressed enthusiasm about the RAG app video tutorial shared earlier, indicating community support for such educational content.

- **Comparing Fine-Tuning and Few-Shot Learning for Multistep Tasks**: A member inquires into research comparing **fine-tuning versus few-shot learning** in improving a model's execution of multistep agentic tasks, considering two approaches – inclusion of reasoning examples in prompts, or dataset building and fine-tuning.

- **Seeking Local Text Enhancement Solution**: A member requests advice on technologies for building a local application to enhance text by correcting errors without altering its original meaning, with an aim to avoid third-party services like ChatGPT.

**Link mentioned**: <a href="https://youtu.be/B9mRMw0Jhfo">How to build a RAG app using Gemini Pro, LlamaIndex (v0.10+), and Pinecone</a>: Let&#39;s talk about building a simple RAG app using LlamaIndex (v0.10+) Pinecone, and Google&#39;s Gemini Pro model. A step-by-step tutorial if you&#39;re just getting ...

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1224746398155542569)** (48 messages🔥): 

- **Mistral Office Hour Alert**: The channel is notified about the **Mistral office hour** being available for questions.
- **Dataset Unification Challenges**: A member describes the complex process of unifying numerous datasets totaling hundreds of gigabytes, involving issues like file alignment. They're currently using TSV files and pickle-format index data for quick seeking, but the ideal solutions and infrastructure are still under consideration.
- **Runpod Serverless vLLM Experiences**: Discussions around **Runpod** and serverless **vLLM** include challenges related to setup and operation. Shared resources on GitHub demonstrate how to deploy [large language model endpoints](https://github.com/runpod-workers/worker-vllm).
- **Evaluating RP-LLMs**: A member introduces **Chaiverse.com** as a platform for receiving rapid feedback on RP-LLM models, highlighting that it's already evaluated 1k models and 5k variants. They invite feedback on the service and discuss the benefits of non-public evaluation datasets for preventing training to the test.
- **Qwen Mow Versus Jamba**: A playful debate regarding the preference of AI models, such as 'qwen mow' versus 'jamba', suggests varying opinions on different models' effectiveness for specific cases like RAG or general-purpose considerations. There's humor about needing more training data and collective investment for better servers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bit.ly/3TFIsKt">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>: Save up to 90% on your cloud bills. Deploy AI/ML production models easily. 600% more images &amp; 10x more inferences per dollar. Try SaladCloud for free today.</li><li><a href="https://github.com/runpod-workers/worker-vllm">GitHub - runpod-workers/worker-vllm: The RunPod worker template for serving our large language model endpoints. Powered by vLLM.</a>: The RunPod worker template for serving our large language model endpoints. Powered by vLLM. - runpod-workers/worker-vllm
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1224741705954099303)** (5 messages): 

- **Documentation Update Praised**: The updated docs for Axolotl received compliments for the new look, but an issue was raised regarding the missing **Table of Contents** which was supposed to include various sections like *Axolotl supports*, *Quickstart*, *Common Errors*, and more, as shown [here](https://openaccess-ai-collective.github.io/axolotl/).
- **Table of Contents Actually Fixed**: A member fixed the missing **Table of Contents** and confirmed the update with a [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a).
- **Discrepancy in Table of Contents Noted**: It was observed that the Table of Contents in the README does not match the markdown headings exactly, implying a need for further cleanup to ensure consistency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/">Axolotl</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a">fix toc · OpenAccess-AI-Collective/axolotl@5760099</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1224713792160534689)** (2 messages): 

- **Models Behaving Unpredictably**: A member expressed frustration that certain models are getting stuck despite having the same configuration as others which are functioning properly.

- **Quest for High-Resolution Images**: Another member inquired about resources to crawl a large quantity of **4K and 8K images** for their needs.
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1224644605455372309)** (36 messages🔥): 

- **Llamafile Builds for Windows ARM**: To build **llama.cpp** for Windows ARM, you'll need to compile it from source, as Windows ARM isn't within the current support vector.
- **Mixtral's Math-Riddle Solving Capability**: The **`mixtral-8x7b-instruct-v0.1.Q4_0.llamafile`** can solve math riddles succinctly, but for recalling obscure facts without hallucinations, a version like **`Q5_K_M`** or higher is necessary. Find related details on [Hugging Face](https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main).
- **Optimizing GPU Performance with TinyBLAS**: When using **llamafile**, GPU performance can differ significantly, often depending on vendor-provided linear algebra libraries. A *`--tinyblas`* flag is available that enables GPU support without needing extra SDKs, though its performance may vary based on the specific GPU model.
- **Windows Executable Formats for ARM**: Windows on ARM supports PE format via the ARM64X binary that contains both Arm64 and Arm64EC code. The lack of emulation for AVX/AVX2 with ARM64EC presents challenges for LLM operations which often require instructions like SVE or NEON. More details can be seen in Microsoft's [documentation](https://learn.microsoft.com/en-us/windows/arm/arm64x-pe).
- **Compiling Issues on Windows for Llamafile**: Windows users are encouraged to build **llamafile** on Linux, Mac, or BSD due to complications in setting up a Cosmopolitan development environment on Windows, as mentioned in the Cosmopolitan [issue #1010](https://github.com/jart/cosmopolitan/issues/1010).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html">Install HIP SDK — HIP SDK installation Windows</a>: no description found</li><li><a href="https://www.theregister.com/2024/04/03/llamafile_performance_gains/">Llamafile LLM driver project boosts performance on CPU cores</a>: Way to whip that LLaMA&#39;s ass</li><li><a href="https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main">jartine/Mixtral-8x7B-Instruct-v0.1-llamafile at main</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/arm/arm64x-pe">Arm64X PE Files</a>: Arm64X are a type of PE file in the Windows 11 SDK used for x64 compatibility on Arm64. Arm64X may be a good solution for developers of middleware or plugins, where code could get loaded into x64 or A...</li><li><a href="http://www.emulators.com/docs/abc_arm64ec_explained.htm">ARM64 Boot Camp: ARM64EC and ARM64X Explained</a>: no description found</li><li><a href="https://github.com/jart/cosmopolitan/issues/1010">execve() should polyfill #! on windows · Issue #1010 · jart/cosmopolitan</a>: Copied from bellard/quickjs#197: #!/bin/qjs console.log(&quot;Hello&quot;); It doesn&#39;t work when invoked from bash as script: $ ./test.qjs ./test.qjs: line 2: syntax error near unexpected token `&...
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1224755324100149390)** (1 messages): 

- **Potential for Opus Judgement to Boost Performance**: There's speculation that, if *Opus Judgement* is accurate, there could be unutilized potential which may enhance results through further Research-Level AI Fine-tuning (**RLAIF**).
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1224783900036042772)** (29 messages🔥): 

- **Google's New AI Lead Excites Discord**: Members expressed surprise and humor over [Logan K's announcement](https://fxtwitter.com/OfficialLoganK/status/1775222819439149424) of joining Google to lead product for AI Studio and supporting the Gemini API, with reactions ranging from shock to speculation over practical reasons such as location.
- **The Logan Strategy: Lifestyle or Poaching?**: The conversation speculated on various factors influencing Logan's move to Google, including the appeal of Chicago, perceived HR poaching strategies, chances of future stock gains, and Google's relative openness in releasing model weights compared to OpenAI.
- **Ideology or Opportunity?**: Members discussed Logan's potential ideological reasons for leaving OpenAI, such as a desire for more openness, but also considered the possibility of being attracted by Google's offers despite personal values.
- **Startup Ambitions or Strategic Move?**: The dialogue included guesses about whether Logan had startup aspirations indicated by his previous "building at" bio, or if the move was a strategic choice due to Google's current positive momentum in AI.
- **Financial Times and the AI Buzz**: A member shared a link to a Financial Times article about AI, but the content was locked behind a subscription, leaving the discussion about it incomplete ([FT content](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/OfficialLoganK/status/1775222819439149424">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Excited to share I’ve joined @Google to lead product for AI Studio and support the Gemini API.  Lots of hard work ahead, but we are going to make Google the best home for developers building with AI. ...</li><li><a href="https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1">Google considers charging for AI-powered search in big change to business model</a>: no description found
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1224755945343811624)** (1 messages): 

- **Open Science Turns Opaque**: Post the release of the **GPT-4 technical report**, which withheld model details, a trend began where other companies also started keeping their model information under wraps. The member recalls this as a shift toward increased secrecy in the field.
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

iron_bound: https://github.com/intel-analytics/ipex-llm
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1224978234559696968)** (4 messages): 

- **Revolutionizing LLMs with DISTFLASHATTN**: [DISTFLASHATTN](https://arxiv.org/abs/2310.03294) presents a memory-efficient attention mechanism that claims to **reduce quadratic peak memory usage to linear**, and optimize long-context LLM training. It reportedly achieves up to **8x longer sequences** and significant speed advantages over existing solutions like Ring Self-Attention and Megatron-LM with FlashAttention.

- **Code for Cutting-edge LLM Training Released**: Researchers can access the code for DISTFLASHATTN, which boasts considerable improvements in training sequence lengths and speeds for models like Llama-7B, via the provided [GitHub repository](https://github.com).

- **Lack of Backward Pass Pseudocode in DISTFLASHATTN Critique**: A member pointed out an omission in the DISTFLASHATTN paper; it *does not include pseudocode for the backward pass*.

- **Previous Attention Mechanisms With Similar Issues**: The same member noted that **Ring Attention**, a prior technique, also failed to include pseudocode for its backward pass.

- **A Call for Scientific Repeatibility**: A comment was made highlighting the frustration with the lack of repeatability in science, which may be linked to the omission of detailed implementation details like pseudocode in published works.

**Link mentioned**: <a href="https://arxiv.org/abs/2310.03294">DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training</a>: FlashAttention (Dao, 2023) effectively reduces the quadratic peak memory usage to linear in training transformer-based large language models (LLMs) on a single GPU. In this paper, we introduce DISTFLA...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225047565670547496)** (2 messages): 

- **CUDA Learning Resources for Beginners**: A new member inquired about recommendations for learning the basics of CUDA programming given their background in Python and Rust. Another member suggested starting with a series of lectures found on [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE) and supplementary content available on their [GitHub page](https://github.com/cuda-mode).

**Link mentioned**: <a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>: A CUDA reading group and community https://discord.gg/cudamode Supplementary content here https://github.com/cuda-mode Created by Mark Saroufim and Andreas Köpf    

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1224979333785849907)** (2 messages): 

- **DISTFLASHATTN for Memory-Efficient LLM Training**: A new **[distributed memory-efficient attention mechanism](https://arxiv.org/abs/2310.03294)** named DISTFLASHATTN is introduced, optimizing the training of long-context large language models (LLMs) with techniques like token-level workload balancing. It outperforms existing models, achieving up to **8x longer sequence lengths** and **speedups** compared to Ring Self-Attention and Megatron-LM with FlashAttention, with **source code available on GitHub**.

- **Reading Scheduled for DISTFLASHATTN Paper**: A member shared an intention to review the DISTFLASHATTN paper on the following day, indicating interest and potential discussion to ensue.

**Link mentioned**: <a href="https://arxiv.org/abs/2310.03294">DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training</a>: FlashAttention (Dao, 2023) effectively reduces the quadratic peak memory usage to linear in training transformer-based large language models (LLMs) on a single GPU. In this paper, we introduce DISTFLA...

  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1224674334778327071)** (6 messages): 

- **Clarifying Throughput Efficiency**: A user referenced a paper, highlighting that efficiency **per token** increases as it's measured by end-to-end throughput (encoding + decoding) divided by **total number of tokens**.
- **Debating the Speed of Token Generation**: A discussion ensued about how adding more tokens can lead to increased speed. The point raised was that while encoding may run in parallel, decoding is sequential, hence the expectation that each additional token would take the same amount of time to decode.
- **Encoding Speed Insight**: Further explanation clarified that the graph in question showed speed for generating a constant 512 tokens, which implies that any speedup in the plot is associated with the **encoding process**.
- **Decoding Speed Questioned**: There was persistence in understanding the process, questioning how decoding could get faster with a larger context since it's sequential in nature, requiring each token to wait for its predecessor.
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1224849362052452372)** (1 messages): 

- **New Contributor Eager to Join**: A member expressed interest in an **onboarding session**, highlighting their background in Python, software engineering, and a Master's degree in data science. They have experience in **AI medical research** in collaboration with someone from StonyBrook and are skilled in writing **data pipelines**.
  

---


**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1224826866137960590)** (1 messages): 

- **Natural Language Crucial for Equations**: Despite the high-level capabilities of **GPT-4** and **Claude**, they sometimes still struggle to solve equations unless the problem is carefully explained in natural language. This suggests a significant challenge remains at the current scale of AI.
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

jinastico: <@748528982034612226>
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554)** (1 messages): 

- **Terminology Tangle in Conversation Logs**: A participant made an observation regarding the `responses` table in `logs.db`, revealing an interest in what to call parts of a dialogue. They shared that the initial part of a conversation where the first person speaks is termed a "speaker turn" or "turn," leading them to name their app's table `turns` instead.
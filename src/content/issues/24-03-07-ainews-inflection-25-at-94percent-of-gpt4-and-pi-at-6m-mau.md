---
id: 52967f97-5cc8-4648-86fa-c64518bb6048
title: Inflection-2.5 at 94% of GPT4, and Pi at 6m MAU
date: '2024-03-08T02:11:17.500129Z'
original_slug: ainews-inflection-25-at-94-of-gpt4-and-pi-at-6m
description: >-
  Inflection AI announced Inflection 2.5, a model achieving over 94% of GPT-4's
  performance with only 40% of the training compute, alongside rapid user growth
  for their Pi chatbot and new real-time web search capabilities. Meanwhile,
  Anthropic's Claude 3 continues to impress, surpassing GPT-4 in user votes and
  gaining adoption as the default model for Perplexity Pro, with expanded
  features like OCR, structured extraction, and experimental tool calling
  support.
tags:
  - inflection
  - inflection-2-5
  - gpt-4
  - pi
  - claude
  - claude-3
  - perplexity
  - retrieval-augmented-generation
  - llamaindex
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/6/2024-3/7/2024. We checked [**356** Twitters](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**332** channels, and **3382** messages) for you. Estimated reading time saved (at 200wpm): **419 minutes**.

Mustafa Suleyman [announced Inflection 2.5](https://inflection.ai/inflection-2-5), which closes much of the gap Inflection had with GPT-4 in an undisclosed compute-efficient way ("*achieves more than 94% the average performance of GPT-4 despite using only 40% the training FLOPs*", which is funny because those numbers aren't public).

 ![image.png](https://assets.buttondown.email/images/51363edf-6f32-40da-a566-e459a2bf0885.png?w=960&fit=max) 

But IQ isn't the only metric that matters; they are also optimizing for EQ, which is best proxied but the impressive user numbers [they also released for Pi](https://inflection.ai/inflection-2-5):

 ![image.png](https://assets.buttondown.email/images/0ed18eb8-c390-414d-a884-147439df784c.png?w=960&fit=max) 

More notes on [the Axios exclusive](https://www.axios.com/2024/03/07/inflection-ai-chatgpt-openai-comparison):

- Pi's user base has been growing at around 10% a week for the last two months. This lets us construct some ballpark estimates for Pi vs ChatGPT:

 ![image.png](https://assets.buttondown.email/images/54998979-bbac-4a44-adfd-1e5d2edcfcea.png?w=960&fit=max) 


They also released a [corrected version of MT-Bench](https://github.com/InflectionAI/Inflection-Benchmarks) for community use.

The community has spotted a couple other interesting tidbits:

- The results are [suspiciously close to Claude 3 Sonnet](https://x.com/seshubon/status/1765870717844050221?s=20)
- Pi also now has [realtime web search](https://x.com/intrstllrninja/status/1765812678277071356?s=20).

---

**Table of Contents**

[TOC] 

---

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs

**Claude 3 Release and Capabilities**:

<ul>
<li><a href="https://twitter.com/AmandaAskell/status/1765207842993434880" target="_blank" rel="noopener noreferrer">Amanda Askell broke down Claude 3's system prompt</a>, explaining how it influences the model's behavior (671,869 impressions)</li>
<li><a href="https://twitter.com/Teknium1/status/1765215638426108028" target="_blank" rel="noopener noreferrer">Claude 3 Opus beat GPT-4 in a 1.5:1 vote</a>, showing impressive performance (17,987 impressions)</li>
<li><a href="https://twitter.com/perplexity_ai/status/1765062913008537793" target="_blank" rel="noopener noreferrer">Claude 3 is now the default model for Perplexity Pro users</a>, with Opus surpassing GPT-4 and Sonnet being competitive (89,658 impressions)</li>
<li><a href="https://twitter.com/llama_index/status/1765101841535336929" target="_blank" rel="noopener noreferrer">Claude 3 shows impressive OCR and structured extraction capabilities</a>, as demonstrated in the @llama_index cookbook (59,822 impressions)</li>
<li><a href="https://twitter.com/LangChainAI/status/1765059668362367110" target="_blank" rel="noopener noreferrer">Anthropic has added experimental support for tool calling in Claude 3 via a LangChain wrapper</a> (21,493 impressions)</li>
</ul>

**Retrieval Augmented Generation (RAG)**:

<ul>
<li><a href="https://twitter.com/llama_index/status/1765439865351766135" target="_blank" rel="noopener noreferrer">LlamaIndex released LlamaParse JSON Mode</a> which allows parsing text and images from PDFs in a structured format. Combined with Claude-3, this enables building RAG pipelines over complex PDFs.</li>
<li><a href="https://twitter.com/llama_index/status/1765481657765912599" target="_blank" rel="noopener noreferrer">LlamaIndex now supports video retrieval</a> via integration with VideoDB, allowing RAG over video data by indexing visual and auditory components.</li>
<li><a href="https://twitter.com/omarsar0/status/1765408813467759037" target="_blank" rel="noopener noreferrer">A paper on "Knowledge-Augmented Planning for LLM Agents"</a> proposes enhancing LLM planning capabilities through explicit action knowledge bases.</li>
</ul>

**Benchmarking and Evaluation**:

<ul>
<li><a href="https://twitter.com/jeremyphoward/status/1765512499049472434" target="_blank" rel="noopener noreferrer">TinyBenchmarks looks promising</a> as a tool for evaluating language models, similar to the Dharma-1 benchmark by @far__el.</li>
<li><a href="https://twitter.com/OfirPress/status/1765494594475581443" target="_blank" rel="noopener noreferrer">An empirical result suggests</a> that 100 examples may be sufficient to evaluate language models, based on datasets like HumanEval (164 examples) and Bamboogle (124 examples).</li>
<li><a href="https://twitter.com/abacaj/status/1765430190249750754" target="_blank" rel="noopener noreferrer">The Yi-9B model was released</a>, showing strong performance on code and math benchmarks, topping Mistral.</li>
</ul>

**AI Research and Techniques**:

<ul>
<li><a href="https://twitter.com/DeepLearningAI/status/1765089900234235954" target="_blank" rel="noopener noreferrer">Researchers introduced Wanda, a method for network pruning that reduces computational burden while maintaining performance</a> (7,530 impressions)</li>
<li><a href="https://twitter.com/arankomatsuzaki/status/1765206218967331008" target="_blank" rel="noopener noreferrer">A paper proposes foundation agents that can master any computer task by taking screen images and audio as input and producing keyboard/mouse operations</a> (16,598 impressions)</li>
<li><a href="https://twitter.com/arankomatsuzaki/status/1765161261162188994" target="_blank" rel="noopener noreferrer">Microsoft presents DéjàVu, a KV-cache streaming method for fast, fault-tolerant generative LLM serving</a> (16,458 impressions)</li>
<li><a href="https://twitter.com/arankomatsuzaki/status/1765159595578982697" target="_blank" rel="noopener noreferrer">Google presents RT-H, which outperforms RT-2 on a wide range of robotic tasks using action hierarchies and language</a> (11,348 impressions)</li>
<li><a href="https://twitter.com/arankomatsuzaki/status/1765160866931159067" target="_blank" rel="noopener noreferrer">Meta presents ViewDiff for generating high-quality, multi-view consistent images of 3D objects in authentic surroundings</a> (5,880 impressions)</li>
</ul>

**Memes and Humor**:

<ul>
<li><a href="https://twitter.com/levelsio/status/1765522778772377638" target="_blank" rel="noopener noreferrer">"AGI by September 2024"</a> meme tweet pokes fun at overhyped AI timelines.</li>
<li><a href="https://twitter.com/OfirPress/status/1765397310006001916" target="_blank" rel="noopener noreferrer">A humorous tweet</a> laments the time sink of relying on GPT-4 to write code in 2024 instead of doing it manually.</li>
<li><a href="https://twitter.com/cto_junior/status/1765302825825812736" target="_blank" rel="noopener noreferrer">Playful banter about making Claude-3 a girlfriend</a>, referencing the Gemini model's supposed creativity.</li>
</ul>

---

# PART 0: Summary of Summaries of Summaries


## Claude 3 Sonnet (14B?)

1. **Model Releases and Comparisons**: Multiple new AI models sparked heated discussions around their strengths and limitations. **Inflection-2.5** claimed to match **GPT-4** performance on benchmarks while using less compute, but faced skepticism from @HlibIvanov who called it [a mere GPT-4 distill lacking innovation](https://x.com/hlibivanov/status/1765754625364275267?s=46). **Claude-3 Opus** achieved impressive feats like [a perfect 800 on the SAT reading section](https://twitter.com/wangzjeff/status/176485068925), with @res6969 praising its enhanced knowledge web construction over 35k tokens. However, @jeffreyw128 noted Claude struggled to find a specific name among 500. **Gemma** underwhelmed @lee0099 compared to 7B **Mistral**, especially in multi-turn dialogues and being English-only.

2. **Open-Source AI and Community Dynamics**: @natolambert vented frustrations over the OSS community's pedantic corrections and lack of perspective, which can deter OSS advocates. Even helpful posts face excessive criticism, as experienced when writing on OSS. The **GaLore** optimizer by @AnimaAnandkumar promised major memory savings for LLM training, generating excitement from @nafnlaus00 and others about improving accessibility on consumer GPUs. However, @caseus_ questioned GaLore's claimed parity with full pre-training. Integrating GaLore into projects like **axolotl** faced [implementation challenges](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370).

3. **Hardware Optimization for AI Workloads**: Optimizing hardware was a key focus, with techniques like pruning, quantization via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), and low-precision operations discussed. @iron_bound highlighted Nvidia's H100 GPU offering 5.5 TB/s L2 cache bandwidth, while @zippika speculated the RTX 4090's L1 cache could reach 40 TB/s. CUDA implementations like @tspeterkim_89106's [Flash Attention](https://github.com/tspeterkim/flash-attention-minimal) aimed for performance gains. However, @marksaroufim warned about [coarsening impact on benchmarking consistency](https://x.com/zeuxcg/status/1765534285229064297?s=20).

4. **AI Applications and Tooling**: Innovative AI applications were showcased, like @pradeep1148's [Infinite Craft Game](https://www.youtube.com/watch?v=QPZpOBxUd1U) and [Meme Generation](https://www.youtube.com/watch?v=PtP8R8VjTGc) using Mistral. LlamaIndex released [LlamaParse JSON Mode](https://twitter.com/llama_index/status/1765439865351766135) for parsing PDFs into structured data. Integrating AI with developer workflows was explored, with @alexatallah offering sponsorships for OpenRouter VSCode extensions, while LangChain's `ask-llm` library [simplified LLM coding integrations](https://github.com/FlorianMgs/ask-llm).

## Claude 3 Opus (8x220B?)

- **NVIDIA Restricts CUDA Translation Layers**: NVIDIA has banned the use of translation layers for running CUDA on non-NVIDIA hardware, targeting projects like **ZLUDA** that aimed to bring AMD GPUs to parity with NVIDIA on Windows. The [updated restrictions](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers) are seen as a move to maintain NVIDIA's proprietary edge, sparking debates over the policy's enforceability.

- **Efficiency and Pruning Debates in LLM Training**: Discussions emerged around the parameter-efficiency of LLMs and the potential for optimized training schemes. Some argued that the ability to heavily prune models without substantial performance drops indicates inefficiencies, while others cautioned about reduced generalizability. New memory-reduction strategies like **GaLore** generated interest, with ongoing attempts to integrate it into projects like [OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370). Questions arose about the limits of current architectures and the saturation of model compression techniques.

- **Inflection-2.5 and Claude-3 Make Waves**: The release of **Inflection-2.5**, claiming performance on par with GPT-4, and **Anthropic's Claude-3** variants Opus and Sonnet sparked discussions. Some were skeptical of Inflection-2.5's innovation, suggesting it might just be a GPT-4 distillation. Meanwhile, Claude-3 garnered significant community interest, with Opus achieving a perfect 800 on the SAT reading section according to this [tweet](https://twitter.com/wangzjeff/status/176485068925).

- **Mistral Powers Innovative Applications**: The **Mistral language model** demonstrated its versatility in powering an [Infinite Craft Game](https://www.youtube.com/watch?v=QPZpOBxUd1U) and [automating meme creation](https://www.youtube.com/watch?v=PtP8R8VjTGc) using the Giphy API. Other noteworthy releases included **Nous Research's Genstruct 7B** for instruction-generation ([HuggingFace](https://huggingface.co/NousResearch/Genstruct-7B)) and new benchmarks like **Hae-Rae Bench** and **K-MMLU** for evaluating Korean language models ([arXiv](https://arxiv.org/abs/2309.02706)).

## ChatGPT (GPT4T)

- **Multilingual Model Support and API Integration**: Discord communities highlighted advancements in multilingual support and API integration, with **Perplexity AI** introducing user interface support for languages like **Korean, Japanese, German, French, and Spanish** and discussing the **Perplexity API** for integrating **Llama 70B**. API desires and troubles included discussions on rate limit increases and integration code, as detailed in their [API documentation](https://docs.perplexity.ai/).

- **Innovations in AI-Driven Game Development**: **Nous Research AI** and **Skunkworks AI** showcased the use of AI in creating new gaming experiences. A **crafting game leveraging Mistral** demonstrated AI's potential in game development with its expandable element combination gameplay, showcased in a [YouTube video](https://www.youtube.com/watch?v=QPZpOBxUd1U). Similarly, **Mistral** was used in an [Infinite Craft Game](https://www.youtube.com/watch?v=QPZpOBxUd1U) and for [automating meme creation](https://www.youtube.com/watch?v=PtP8R8VjTGc), illustrating innovative AI applications in gaming and humor.

- **Advancements and Debates in Model Optimization and Pruning**: The **LAION** and **OpenAccess AI Collective (axolotl)** summaries brought to light discussions on model optimization, pruning, and efficiency. Debates on the pruning of Large Language Models (LLMs) reflected differing opinions on its impact on performance and generalizability, with some engineers proposing pruning as evidence for possible optimization. **GaLore** emerged as a focal optimization tool in discussions, despite skepticism about its performance parity with full pretraining, with integration efforts underway as noted in their [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370).

- **Emergence of New AI Models and Tools**: Across multiple Discord summaries, there was significant buzz around the introduction of new AI models and tools, including **Inflection AI 2.5**, **Genstruct 7B**, and **Yi-9B**. **Inflection AI 2.5**'s release sparked conversations about its efficiency and performance, whereas **Nous Research AI** unveiled **Genstruct 7B**, an instruction-generation model aimed at enhancing dataset creation and detailed reasoning, available on [HuggingFace](https://huggingface.co/NousResearch/Genstruct-7B). The **Hugging Face** community saw the launch of **Yi-9B**, adding to the growing list of models available for experimentation and deployment, showcasing the continuous innovation and expansion of AI capabilities, with a demo available [here](https://huggingface.co/spaces/Tonic/Yi-9B).

---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity AI Talks the Talk in Multiple Languages**: Perplexity now supports **Korean, Japanese, German, French, and Spanish** for its user interface, as announced by `@ok.alex`. Users can customize their language preferences in the app settings.

- **A Limitations and Alternatives Smorgasbord**: There was an active discussion about the limitations of AI models, with focus on daily usage limits for **Claude 3 Opus** and alternatives like **Claude 3 Sonnet** and **GPT-4 Turbo**. The need for more direct feedback was mentioned in regards to the closed beta application process.

- **Perplexity Pro Subscribers Sound Off**: Users shared their experiences with **Perplexity Pro**, engaging in discourse regarding the additional benefits and how to access specialized support channels.

- **New Kid on the Block: Inflection AI 2.5**: The release of **Inflection AI 2.5** sparked conversations about its efficiency and performance levels, with users highlighting its speed and debating its potential use cases.

- **Global Cordiality or Algorithmic Manners?**: A discussion was sparkled around cultural communication nuances, with a focus on the use of "sir" and global differences in respectful address, in the context of language models.

- **Sharing is Caring - Perplexity Goes 3D**: Users shared interesting Perplexity search links, exploring topics from 3D space navigation, to altcoin trends, the concept of Ikigai, quality of text generation by Claude 3 Opus, and interpretations of quantum mechanics.

- **API Desires and Troubles**: The guild members are engaging with the **Perplexity API**, seeking integration code for **Llama 70B** and support for rate limit increases, while also showing interest in the **Discover** feature. The [Perplexity API documentation](https://docs.perplexity.ai/) was referenced as a guide for usage and technical assistance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Innovative Crafting AI Game Emerges**: A new crafting game leveraging **Mistral** has been introduced by `@pradeep1148`, showcasing the potential for AI in game development. The game begins with four elements and expands as players combine them, as demonstrated in a [YouTube video](https://www.youtube.com/watch?v=QPZpOBxUd1U).

- **Continuous Improvement Triggers Tech Buzz**: The **Yi-34B** base model has shown remarkable performance growth in its "Needle-in-a-Haystack" test, potentially raising the bar for upcoming models. Google's **Gemma** model received community-driven bug fixes, which are available in [Colab notebooks](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing), and the new **GaLore** project demands community validation, potentially benefiting from pairing with [low-bit optimizers](https://github.com/thu-ml/low-bit-optimizers).

- **Genstruct 7B Sets the Instructional Pace**: Nous Research unveils **Genstruct 7B**, an instruction-generation model designed to enhance detailed reasoning and dataset creation. Heralded by Nous's own `<@811403041612759080>`, **Genstruct 7B** promises innovation in instruction-based model training, available on [HuggingFace](https://huggingface.co/NousResearch/Genstruct-7B).

- **Chat AI and LLMs Spark Discussions**: Anthropic's new [**Genstruct-7B-GGUF**](https://huggingface.co/gguf/Genstruct-7B-GGUF) enters the spotlight alongside debates surrounding **Claude 3's** performance, and **Inflection AI** claims its model Inflection-2.5 matches **GPT-4** on benchmarks. Meanwhile, community skepticism prevails regarding both a shared **Twitter IQ test chart** and the rumors of a **GPT-5** release.

- **Technical Debates and Clarifications**: From running **Ollama** locally to the potential of a function-calling model inspired by Claude-style, the community seeks insights on various AI models and tools. Highlights include the **Nous-Hermes-2-Mistral-7B-DPO**'s upcoming update for function calling data and a refactoring effort for a logit sampler by `@ufghfigchv` tailored for JSON/function calls. Access to **GPT-4** was also mentioned with [Corcel.io](https://corcel.io/) offering free ChatGPT-4 like interactions and a desire for longer context lengths in models like Nous-Hermes for RAG applications.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **CUDA Controversy**: NVIDIA's recent policy change prohibits the use of translation layers for running CUDA on non-NVIDIA hardware, directly impacting projects like ZLUDA. The [updated restrictions](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers) are seen as a move to maintain NVIDIA's proprietary edge.

- **Scraping Spat Spirals**: Stability AI was mixed up in a controversy for allegedly scraping Midjourney, causing a ban on their employees and raising concerns over data scraping practices. While some [tweets suggest](https://twitter.com/EMostaque/status/1765496173572346182) it was not work-related, it has kindled a discussion on scraping ethics and protocols.

- **Efficiency in the Spotlight**: Debates concerning the pruning of Large Language Models (LLMs) have been center stage. Some engineers believe current training methods are ineffectual, proposing pruning as evidence for possible optimization, while others voice concerns about the potential loss of generalizability, stability, and the slowness of certain optimization techniques such as SVD.

- **Pruning Perplexities**: Contrary to some beliefs that lightly pruned models experience performance degradation, there's an argument that heavily pruned LLMs remain surprisingly generalizable. However, this leads to a bigger question: are oversized parameters needed for model training, or can engineers aim for leaner, yet effective LLMs?

- **Architectural Assessments**: Conversations are probing the structural boundaries of current LLMs, exploring whether the strategies to compress and optimize models, especially those based on attention mechanisms, are approaching their limits. This underlines a curiosity about the saturation of model efficiency within present-day architectures.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Claude's Remarkable Performance**: Positive experiences with **Claude 3 Opus** (C3) were highlighted as it outperformed GPT-4 on complex tasks and elementary class problems.
- **Claude Versus Gemini**: There was a debate over coding capabilities, where **Gemini 1.5 Pro** solved a Python GUI task successfully on the first try, while Claude 3 did not, indicating strengths and weaknesses in each.
- **Doubts Cast on MMLU Datasets**: Concerns arose about the **MMLU datasets**' questions lacking logical consistency and containing incorrect answers, leading to calls for reconsidering their use in AI model evaluations.
- **GPT-4 Availability and Policy Discussions**: Users discussed intermittent access to GPT-4 and policy changes affecting code provision, with confusion over accessing custom models like Human GPT via API clarified by references to OpenAI's [model overview](https://platform.openai.com/docs/models/overview).
- **Service Outages and Support Quests**: A reported 6-hour service interruption on OpenAI's APIs, a query on the implementation of randomization in storytelling using Python’s random function, and discussions on the development of a GPT classifier reflected the technical and operational challenges community members faced.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Tech Titans Talk Troubleshooting**: Users discussed hardware configurations with **512 GB RAM** and **24GB VRAM**, and tackled library version issues, specifically a `GLIBCXX_3.4.29 not found` error. The debates extended to whether a **Macbook Pro with M3 Max or M2 Max** is suitable for local LLM processing, touching on price and performance trade-offs. Community members aired concerns over the non-openness of OpenAI, with alternative AI services like **POE monthly** being considered for model access.

- **Narrative Nuances and AI Models**: In models discussion, the optimal AI specified for storytelling was **Mistral-7b-instruct-v0.2-neural-story.Q4_K_M.gguf**, yet the limitations due to memory constraints were evident. Lack of image-generation capabilities in LM Studio led participants to consider tools like **Automatic 1111** for Stable Diffusion tasks. Interest in **Starcoder2** was evident, with users awaiting its support, as indicated by references to its [Hugging Face page](https://huggingface.co/TechxGenus/starcoder2-15b-instruct/).

- **Feedback Focus Irregularities and Insights**: A user shared they desire clearer guidance to exploit LM Studio's potential. There's feedback suggesting LM Studio could emulate [Embra AI](https://embra.app/) to improve its utility, and that the current version (v0.2.16) doesn't support proxy on macOS 14.3. Also, it was clarified that help requests should not be posted in the feedback channel.

- **Hardware Hub Conversations**: Reports indicated experiments with a **200K context** in the Smaug model and VRAM demand issues with an LLM Studio task utilizing a 105,000 context. A **minimum of 550W** was recommended for a PSU to power an RTX 3090, and discussions included handling large contexts or datasets with LLM tasks and mismatched VRAM issues.

- **Response Rate Quest in Crew AI**: In looking for ways to **increase response speed** for an unspecified process, users faced a connection timeout issue and proposed establishing local operations as a potential solution.

- **The Odyssey of Open Interpreter Syntax**:
There were inquiries and conversations surrounding the correct `default_system_message` syntax and profile configurations in Open Interpreter. Users exchanged experiences with code-trained models and shared learning moments for configurations, with references to instructions at [Open Interpreter - System Message](https://docs.openinterpreter.com/settings/all-settings#system-message) and [Hugging Face](https://huggingface.co/owao/LHK_DPO_v1_GGUF).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Survey Your Heart Out**: LlamaIndex seeks deeper user insights with a **3-minute survey**. Engineers are encouraged to participate to shape better resources like documentation and tutorials, accessible at [SurveyMonkey](https://www.surveymonkey.com/r/PNSP3P9).

- **LlamaParse JSON Unleashed**: The **LlamaParse JSON Mode** from LlamaIndex is generating buzz with its ability to parse PDFs into structured dictionaries, especially when paired with models like claude-3 opus. For those interested, a tweet announces the launch at [LlamaIndex Tweet](https://twitter.com/llama_index/status/1765439865351766135).

- **Video Capabilities Level Up**: LlamaIndex and `@videodb_io` integration opens new doors for video content handling, allowing keyword-based video upload, search, and streaming within LlamaIndex. Discover more about this integration via this [Announcement Tweet](https://twitter.com/llama_index/status/1765481657765912599).

- **Optimizing A* for Search Efficiency**: The A* algorithm's feasibility for similarity search was validated with a subclassing of the embedding class to alter the search methodology, showcasing LlamaIndex's flexibility.

- **In-Context Learning Gets a Boost**: A new methodology enhancing in-context learning has been introduced by `@momin_abbas` with the [Few-Shot Linear Probe Calibration](https://github.com/mominabbass/LinC). The initiative invites support from the community through GitHub engagement.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Midjourney and Stability AI Spat Goes Public**: `@420gunna` referenced a controversial incident where **Stability AI** was banned from **Midjourney** for scraping data, as detailed in a [Twitter post](https://x.com/nickfloats/status/1765471291300045255?s=20) by `@nickfloats`.
- **Podcast Episode Featuring AI Experts Hits the Airwaves and Hacker News**: `@swyxio` announced a new podcast episode with `<@776472701052387339>` and highlighted its presence on [Hacker News](https://news.ycombinator.com).
- **Cheers for Volunteer-led Model Serving Paper Presentation**: The community showed appreciation for `@720451321991397446`'s volunteering, with a specific focus on a presentation about model serving, accessible via [Google Slides](https://docs.google.com/presentation/d/1Jde7Vx0BQNMClRCLlesy36hlC3VUJRceEYIxdkl1j68/edit?usp=sharing).
- **Inference Optimization and Hardware Utilization Debate Heats Up**: Notable discussions on inference optimization included alternatives like speculative decoding and FlashAttention, and the effect of hardware, with insights derived from resources like [EGjoni's DRUGS GitHub](https://github.com/EGjoni/DRUGS?tab=readme-ov-file) and the [DiLoCo paper](https://arxiv.org/abs/2311.08105 "DiLoCo").
- **Decentralized Training and GPU Configuration Discussions Surge**: There was an active engagement in deliberations around distributed training with references to **DiLoCo** and the influence of GPU configurations on model outputs, spurred by an anecdote of an OpenAI incident.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **New Korean Language Benchmarks Unveiled**: Two new evaluation datasets, **[Hae-Rae Bench](https://arxiv.org/abs/2309.02706)** and **[K-MMLU](https://arxiv.org/abs/2402.11548)**, specifically tailored to assess language models' understanding of Korean language and culture, have been introduced by `@gson_arlo`.
  
- **vLLM Batching Clarified**: `@baber_` explained that batching is internally handled by **vLLM**, thus manual implementation for batched inference is not required, referencing the [official documentation for ease of use](https://docs.vllm.ai/en/latest/getting_started/quickstart.html).
  
- **Multilingual Benchmark Collaboration Called For**: Contributors speaking non-mainstream languages are invited to participate in creating pertinent benchmarks that evaluate language model competencies specific to their cultures.
  
- **Optimizer Memory Issues in GPT-NeoX Addressed**: Discussions in the **gpt-neox-dev** channel focused on tackling memory peaks during optimization with members like `@tastybucketofrice` citing [Issue #1160](https://github.com/EleutherAI/gpt-neox/issues/1160) and suggesting Docker as a potential solution to dependency challenges ([Docker commit here](https://github.com/EleutherAI/gpt-neox/commit/119950c7faee46e7929baac8640c84aa9fda4d2b)).
  
- **Efforts on Unified Fine-tuning Framework**: In the **lm-thunderdome** channel, the lack of a consistent mechanism for fine-tuning language models was spotlighted by `@karatsubabutslower`, despite having a standardized evaluation method like lm-evaluation-harness.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Entrepreneurs Seek Open Source Model Groups**: Entrepreneurs on *HuggingFace* are looking for a community to discuss the application of open-source models in small businesses, however, no dedicated channel or space was recommended within the provided messages.

- **New Model on the Block, Yi-9B**: *Tonic_1* launched **Yi-9B**, a new model in the Hugging Face collection, available for use with a [demo](https://huggingface.co/spaces/Tonic/Yi-9B). Hugging Face may soon be hosting leaderboards and gaming competitions.

- **Inquiry into MMLU Dataset Structure**: *Privetin* displayed interest in understanding the **MMLU datasets**, a conversation that went without elaboration or engagement from others.

- **Rust Programming Welcomes Enthusiasts**: *Manel_aloui* kicked off their journey with the **Rust language** and encouraged others to participate, fostering a small community of learners within the channel.

- **AI's Mainstream Moment in 2022**: Highlighted by an Investopedia article shared by `@vardhan0280`, AI's mainstream surge in 2022 was attributed to the popularity of **DALL-E** and **ChatGPT**.

- **New Features in Gradio 4.20.0 Update**: Gradio announced version 4.20.0, now supporting **external authentication providers** like HF OAuth and Google OAuth, as well as introducing a `delete_cache` parameter and `/logout` feature to enhance the user experience. The new `gr.DownloadButton` component was also introduced for stylish downloads, detailed in the [documentation](https://www.gradio.app/docs/downloadbutton#demos).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Deepspeed Disappoints Multi-GPU Setups**: Engineers noted frustration with **Deepspeed**, particularly in multi-GPU scenarios with 4x 4090s, finding that it fails to split the base model across GPUs when using the Lora adapter. The discussion referenced a [Deepspeed JSON config file](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12).

- **GaLore Spurs Debate**: **GaLore**, an optimization tool, became a focal point due to its potential for memory savings in Large Language Model training. Despite excitement, skeptics questioned its performance parity with full pretraining, even as [integration efforts](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370) are underway.

- **Efficiency Methods Under Microscope**: Discussions surfaced doubts about the potential misleading nature of various efficiency methods, including ReLoRA and NEFT, prompting consideration of dataset sizes and settings for meaningful finetuning.

- **Gemma's Performance Draws Criticism**: The **Gemma** model came under scrutiny for underwhelming performance against 7B Mistral, especially in multi-turn dialogues, and was constrained by being English-only, which limited its value for multilingual tasks.

- **Dependency Wars in AI Development**: A common thread across discussions was the battle against dependency conflicts, especially with `torch` versions in the installation of `axolotl[deepspeed]==0.4.0`. Engineers shared tactics like manual installation and suggested specific versions, including [`torch==2.2.0`](https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts), as potential fixes.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Claude 3 Makes Group Chat Cool**: **Alex Atallah** shared a [tweet](https://twitter.com/OpenRouterAI/status/1765470591836959061) on the positive self-moderated group chat experience using **Claude 3**.
  
- **Nitro-Power to Your Projects**: New "nitro" models are in testing with OpenRouter, offering safe integration options, although slight adjustments may be expected during the feedback incorporation phase before an official launch.
  
- **VSCode Extension Bounty**: **Alex Atallah** offered **sponsorship for building a VSCode extension** for OpenRouter, rewarding developers with free credits for their contributions.

- **Development Tips and Tricks Exchange**: Community members exchanged information on various VSCode extensions for LLMs, including [Cursor](https://cursor.sh/), [Continue](https://continue.dev), and [Tabby](https://tabby.tabbyml.com/), as well as pointing to more cost-effective chat models like [Sonar 8x7B by Perplexity](https://openrouter.ai/models/perplexity/sonar-medium-chat?tab=parameters).
  
- **Budget-Friendly AI Conversations**: Discussions about the cost implications of engaging with models like Claude 3 Opus were had, whereby **Sonar 8x7B** was highlighted for its cost-effectiveness over others.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **CSV Loader Timeout Troubles**: Users noted issues with `UnstructuredCSVLoader` throwing "The write operation timed out" errors in LangChain. Although solutions were not discussed, the problem was acknowledged by members sharing similar experiences. 

- **Raising Red Flags on Phishing**: Concerns were raised over an uptick in phishing attempts within the server, particularly through suspicious steamcommunity links, but follow-up actions or resolutions were not detailed.

- **Prompt Puzzle from Past Interactions**: In the construction of a chat chain, one user faced issues with `HumanMessage` content improperly propagating into `AIMessage` after initial interactions, despite the intention of memory segregation. A shared code snippet highlighted the problem, though the community's advice was still sought.

- **LangChain Leverages RAPTOR & Pydantic Pairing**: Detailed strategies for utilizing **Pydantic** with **LangChain** and **Redis** for structuring user data and chat histories were under discussion, with an invite for insights into unexpected `AIMessage` behaviors. 

- **Link Library for LangChain Learners**: Released resources included **ChromaDB Plugin for LM Studio**, a tool for generating vector databases, and the `ask-llm` library for simpler LLM integration into Python projects. Highlighted educational content included a Medium article on **RAG** construction using **RAPTOR**, and YouTube tutorials on game crafting and meme generation with **Mistral** and Giphy API.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Coarsening Tips**: A [tweet by @zeuxcg](https://x.com/zeuxcg/status/1765534285229064297?s=20) shared insights on properly handling coarsening in code execution, highlighting performance impacts due to benchmarking inconsistencies.
- **Comparing Relay and Flash Attention**: In the realm of attention mechanisms, `@lancerts` raised a discussion on comparing **RelayAttention** with ring/flash attention, citing a [GitHub repository on vLLM with RelayAttention](https://github.com/rayleizhu/vllm-ra).
- **Insightful CUDA Command for GPU Reset**: A `sudo` command was provided for resetting GPUs while addressing memory allocation on GPUs, `sudo nvidia-smi --gpu-reset -i 0`, as well as sharing a potentially related `nvtop` observation.
- **New CUDA Project on the Block**: `@tspeterkim_89106` introduced a project implementing **Flash Attention** in CUDA, inviting feedback and collaboration on [GitHub](https://github.com/tspeterkim/flash-attention-minimal).
- **CUDA Synchronization Mechanics in Torch**: The use of `torch.cuda.synchronize()` for accurate performance measurements was recommended, with clarifications on synchronization across CUDA kernels and the cross-device usage of scalar tensors.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **Inflection-2.5 Sparks Debates**: `@xeophon.` introduced **Inflection-2.5**, an AI model embedded in Pi, which claims high performance on par with GPT-4 and Gemini. However, a [tweet by @HlibIvanov](https://x.com/hlibivanov/status/1765754625364275267?s=46) criticizes it as a GPT-4 distill, raising questions about its innovation.
  
- **AI Innovation Frenzy Noted**: `@natolambert` showcased enthusiasm over the rapid development within the AI field, referencing multiple new model releases and discussing it further in a [tweet](https://twitter.com/natolambert/status/1765779714252451971).

- **OSS Community Nitpicks Frustrate**: Discussions touched on the unwelcoming nature of the open-source software community, with `@natolambert` expressing frustration over pedantic criticisms that are discouraging to OSS advocates and `@xeophon.` bringing up the confusion over labeling in the space.

- **Claude-3 Heats Up AI Competition**: The release of **Claude-3** by **@Anthropic** has gathered a fervent community response, along with its variants Opus and Sonnet, as shared in a [tweet](https://x.com/lmsysorg/status/1765774296000172289?s=46) with significant community involvement noted in the form of 20,000 votes in three days.

- **Expectations Soar for Gemini Ultra**: The upcoming **Gemini Ultra** and its 1M context window feature are highly anticipated, with engineers like `@natolambert` and `@xeophon` keen on exploring its capabilities for tasks such as analyzing academic papers.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Spam Alert Leads to Policy Change**: Following a spam incident with @everyone tags, users like `@joshxt` highlighted the importance of respecting everyone's inboxes, leading to a new policy where the ability to ping everyone was disabled to prevent unwanted notifications.
- **Orca Dataset Dives into Discourse**: The release of Microsoft's [Orca dataset](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) was brought up by `@joshxt`, sparking a conversation and personal model preferences, with "Psyonic-cetacean" and "Claude 3 Opus" getting special mentions.
- **Introducing Project Orca-2**: `@aslawliet` put forward a proposal for Orca-2, aiming to encompass a diverse range of datasets beyond Microsoft's recent release, such as FLAN 2021 and selective zero-shot samples from T0 and Natural Instructions.
- **Efficient Data Augmentation Tactics**: `@aslawliet` proposed using Mixtral as a time and cost-efficient data augmentation method over GPT-4, prompting discussion on efficient methods for model improvement.
- **Cordial Introductions and Greetings**: The community warmly welcomed new participants like `@segmentationfault.` and `@1168088006553518183`, emphasizing a friendly atmosphere and the shared interest in contributing to the field of AI.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Choose Your Language Model Wisely**: Depending on constraints, `@johannhartmann` advises using **Claude Opus** and **GPT-4** when there are no limitations, **DiscoLM-120B** for open-source with substantial memory availability, and **VAGOsolutions/Sauerkraut LM-UNA-SOLAR-Instruct** as the go-to when working with restricted memory.

- **Retrieval-Augmented on the Rise**: A study in an [arXiv paper](https://arxiv.org/abs/2403.03187) shows the benefits of retrieval-augmented language models, with a specific focus on joint training of retriever and LLM, though comprehensive research on this integration remains scarce.

- **The Quest for the Best German-Speaker**: The **Nous Hermes 2 Mixtral 8x7b** was praised by `@flozi00` for its high accuracy in task comprehension. In contrast, `@cybertimon` and `@johannhartmann` recommended exploring a range of models including **DiscoResearch/DiscoLM_German_7b_v1** and **seedboxai/KafkaLM-7B-DARE_TIES-LaserRMT-QLoRA-DPO-v0.5** for fluent German language capabilities.

- **Evaluating Translation Through Embedding Models**: `@flozi00` is developing an approach to score translation quality based on embedding distance, using the OPUS 100 dataset. This initiative could steer enhancements in machine translation (MT) models and data quality.

- **mMARCO Dataset Receives the Apache Seal**: The mMARCO dataset now boasts an Apache 2.0 license, as shared by `@philipmay`, enriching resources for developers although lacking dataset viewer support on [Hugging Face](https://huggingface.co/datasets/unicamp-dl/mmarco#licensing-information).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **SAT Scores Soar with Opus**: **Opus** nailed a perfect 800 on the SAT reading section, as shared by [@jeffreyw128](https://twitter.com/wangzjeff/status/176485068925).
- **Memorization vs. Learning**: Following the SAT victory, `@dare.ai` touched on the difficulty of ensuring that large models like **Opus** avoid memorizing answers rather than truly learning.
- **Opus Earns a Fanclub Member**: `@nosa_` humorously warned of a faux-confrontation should **Opus** learn of their high praise for its performance.
- **Weaving Webs of Wisdom**: `@res6969` praised **Opus** for its enhanced skill in crafting knowledge webs from expansive documents, highlighting its ability to follow instructions over 35k tokens.
- **In-Depth Search Dilemma**: A task that involved finding a specific name among 500 proved to be challenging for different models including **Claude Opus**, as reported by `@jeffreyw128`.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **GPT-4 Stumbles in Mystery Test**: **GPT-4** failed an unspecified test, according to `@dbreunig`, yet no details about the nature of the test or the type of failure were provided.
- **Bridging Physical and Digital Libraries**: `@xnimrodx` shared a novel [blog post](https://jamesg.blog/2024/02/14/clickable-bookshelves/) about making **bookshelves clickable** that connect to Google Books pages, along with a [demo](https://capjamesg.github.io/cv-book-svg/), sparking discussion about potential applications in library systems and local book-sharing initiatives.
- **Dollar Signs in Templates Cause Chaos**: `@trufuswashington` experienced crashes in the `llm` command caused by a `TypeError` when using a dollar sign `$` in a YAML template that was intended for explaining code-related content, uncovering the issue with the special character's handling within template prompts.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Mistral Turns Game Crafting Infinite**: A new [Infinite Craft Game powered by Mistral](https://www.youtube.com/watch?v=QPZpOBxUd1U) was shared, highlighting the **Mistral language model**'s application in a game that allows players to combine elements to create new items, suggesting innovative use-cases for AI in gaming.

- **Meme Generation Meets AI**: The **Mistral language model** has been used to [automate meme creation](https://www.youtube.com/watch?v=PtP8R8VjTGc) in combination with the **Giphy API**, as demonstrated in a YouTube video, with the code available on GitHub for engineers looking to explore the intersection of AI and humor.


---

# PART 2: Detailed by-Channel summaries and links



### Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1215104546850873354) (1 messages): 

- **Perplexity AI now speaks your language**: User `@ok.alex` announced that Perplexity is now available in **Korean (한국어), Japanese (日本語), German (Deutsch), French (Français), and Spanish (Español)**. Users can change their preferred interface language in the settings on both desktop and mobile.
  

---


### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1214891515978190909) (413 messages🔥🔥🔥): 

- **Limitations and Comparisons of AI Models**: Users discussed various limitations of AI models, with `@twelsh37` sharing a comprehensive report prompt for testing AI capabilities. `@zero2567` and others noted the limited usage of **Claude 3 Opus** to 5 times a day, prompting discussions on the constraints and alternatives like **Claude 3 Sonnet** and **GPT-4 Turbo** for coding tasks, as mentioned by users like `@tunafi.sh` and `@deicoon`.

- **Pro Subscriptions and Features Enquiry**: Users like `@arrogantpotatoo` and `@dieg0brand0` shared their subscription to **Perplexity Pro**, leading to discussions on the benefits and how to access specialized channels and pro support on Discord.

- **Testing Inflection AI's New Release**: The announcement of **Inflection** AI's 2.5 release caught the attention of multiple users, including `@codelicious` and `@ytherium`, with conversations around the model's claimed performance level and efficiency. Several noted its speedy performance, even speculating on its potential for various use cases.

- **Opinions on Gemini 1.5**: Dissatisfaction and speculations with **Gemini 1.5** were voiced by users like `@archient`, who found it disappointing and lacking features compared to other services. The conversation touched upon expectations of Google's AI product ecosystem and potential reasons for its perceived underperformance.

- **Cultural Respect or Language Model Bias?**: A few messages from `@gooddawg10` sparked a discussion on respect and formality in communication, highlighting cultural nuances and prompting users like `@twelsh37` and `@brknclock1215` to address the use of "sir" and differences in global communication styles.

**Links mentioned**:

- [Inflection-2.5: meet the world&#x27;s best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Cat Dont Care Didnt Ask GIF - Cat Dont Care Didnt Ask Didnt Ask - Discover &amp; Share GIFs](https://tenor.com/view/cat-dont-care-didnt-ask-didnt-ask-i-didnt-ask-gif-25429803): Click to view the GIF
- [Perplexity Blog](https://blog.perplexity.ai/blog/introducing-pplx-online-llms>:): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [What is Pro Search?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [
Welcome to Live — Ableton Reference Manual Version 12
 | Ableton](https://www.ableton.com/en/live-manual/12/welcome-to-live/): no description found

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1214869423043641445) (14 messages🔥): 

- **Cruising Through the 3D Space**: User `@williamc0206` shared a perplexity search link, potentially discussing the capabilities of navigating or generating 3D spaces. [Explore the 3D Space with Perplexity](https://www.perplexity.ai/search/can-you-generate-3DS8nzLaRH.kziyvZp5GUQ#dc34bc9f-32da-447f-a4ce-2caf669e4651).
- **Altcoins in the Spotlight**: `@b.irrek` posted a link that appears to delve into the movement and trends surrounding alternative cryptocurrencies. [Insight on Altcoins Here](https://www.perplexity.ai/search/altcoins-have-seen-rxkApXXdQnungtp7yvjzMw?s=m).
- **Unraveling the Concept of Ikigai**: `@sevonade4` invited others to check out a generated text on the concept of Ikigai, which could be of specific interest depending on personal curiosity. [Dive into Ikigai](https://www.perplexity.ai/search/Concept-of-Ikigai-RWMyh5a0SYakFpZyB.wAFw?s=m).
- **Contemplations on Claude 3 Opus**: Further, `@sevonade4` highlighted the text generation quality of Claude 3 Opus for those interested in exploring different levels of text generation. [Reflect with Claude 3 Opus](https://www.perplexity.ai/search/Reflection-piece-on-yAziPT6hQYik._AQAh.yfw).
- **Quantum Queries Addressed**: `@vmgehman` shared how Perplexity has been a helpful resource in studying various interpretations of quantum mechanics. [Quantum Mechanics Explorations](https://www.perplexity.ai/search/What-are-the-Z6stfbOURFuXP_76wIrV_A).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1214871920588357642) (19 messages🔥): 

- **Seeking HTML & JS Code for Llama 70B Integration**: `@kingmilos` asked for assistance with code to integrate Llama 70B using HTML and JS because their expertise lies in Python. `@po.sh` responded by providing a basic code example, instructing to insert the API key and adjust the model as necessary.
  
- **Feedback on Beta Application Process**: `@brknclock1215` expressed disappointment in the perceived impersonal nature of the closed beta application denial, suggesting a desire for more direct communication or feedback.
  
- **Documentation for API Assists Programmers**: `@icelavaman` pointed `@kingmilos` and `@pythoner_sad` to the [Perplexity API documentation](https://docs.perplexity.ai/) for guidance on using the API with LLM inference, yet `@kingmilos` expressed difficulty due to a lack of HTML and JS knowledge.
  
- **User Seeks Support for Rate Limit Increase**: `@xlhu_69745` requested assistance with a rate limit increase for the Sonar model but noted a lack of response to their email. `@icelavaman` responded with a non-verbal indication, possibly suggesting where to seek help or check updates.
  
- **Interest in Discover Feature via API**: `@yankovich` inquired about Perplexity Discover and whether a similar feature could be implemented for users through Perplexity's API. `@bitsavage.` suggested reviewing the API documentation to understand potential functionalities and consider how to craft personalized user discovery features.

**Links mentioned**:

[pplx-api](https://docs.perplexity.ai/): no description found

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1214975398828314665) (6 messages): 

- **Crafting with AI**: User `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=QPZpOBxUd1U) titled "Infinite Craft Game using Mistral", which shows the development of a crafting game that begins with four elements and expands as players combine them.
- **In Search of Ollama Examples**: `@pier1337` inquired about examples of running **Ollama** with **Deno**, but no further information was provided in the channel.
- **Missed Connection**: User `@teknium` responded to **nonexistent tags** and apologized for missing a direct message on Twitter sent months ago by an unspecified user.

**Links mentioned**:

- [Infinite Craft Game using Mistral](https://www.youtube.com/watch?v=QPZpOBxUd1U): Let develop Neal Agarwal’s web game Infinite Craft. This is a “crafting game” where you start with just four elements and repeatedly combine pairs of element...
- [Making memes with Mistral &amp; Giphy](https://www.youtube.com/watch?v=PtP8R8VjTGc): Lets make memes using mistral llm and Giphy api#llm #ml #python #pythonprogramming https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1214913941688549427) (44 messages🔥): 

- **Yi LLMs Constantly Improving**: `@thilotee` highlighted ongoing enhancements to the **Yi-34B** base model, notably its performance on the "Needle-in-a-Haystack" test improving from 89.3% to 99.8%. The discussion touched upon the potential for further finetuning and whether the Yi-9B model supports 200k context, which it appears not to ([Yi's Huggingface page](https://huggingface.co/01-ai/Yi-34B-200K) and [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1b8qnqv/yi34b200k_model_update_needleinahaystack_improved/)).

- **Google's Gemma Admittedly Flawed**: `@mister_poodle` shared concerns that Google may be hastily releasing models, as a team called Unsloth fixed several bugs in the Gemma model which were not addressed elsewhere. The fixes are available in [Colab notebooks](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing).

- **Claude 3 Opus Claims Debated**: A user shared an experience with **Claude 3 Opus** translating the low-resource Circassian language impressively, but later updates suggested the model might have had prior knowledge of the language. This sparked a discussion on in-context reasoning capabilities and the validity of the original claims ([Original Twitter Post](https://x.com/hahahahohohe/status/1765088860592394250?s=46)).

- **GaLore: The New GitHub Gem**: `@random_string_of_character` posted links to **GaLore**, a project on GitHub, and a Twitter post; however, community validation is needed to determine its effectiveness. A suggestion was made to pair it with [low-bit optimizers](https://github.com/thu-ml/low-bit-optimizers) for potential savings.

- **Anticipation for Function Calling Model**: Amidst various discussions, `@scottwerner` and `@sundar_99385` expressed excitement about trying out a new model with function calling capabilities inspired by a Claude-style. No release date was mentioned, but eagerness for the model's launch was evident.

**Links mentioned**:

- [Tweet from An Qu (@hahahahohohe)](https://x.com/hahahahohohe/status/1765088860592394250?s=46): Today while testing @AnthropicAI &#39;s new model Claude 3 Opus I witnessed something so astonishing it genuinely felt like a miracle. Hate to sound clickbaity, but this is really what it felt like.  ...
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950): Value functions are a central component of deep reinforcement learning (RL). These functions, parameterized by neural networks, are trained using a mean squared error regression objective to match boo...
- [Unsloth Fixing Gemma bugs](https://unsloth.ai/blog/gemma-bugs): Unsloth fixing Google&#x27;s open-source language model Gemma.
- [01-ai/Yi-9B · Hugging Face](https://huggingface.co/01-ai/Yi-9B): no description found
- [GitHub - thu-ml/low-bit-optimizers: Low-bit optimizers for PyTorch](https://github.com/thu-ml/low-bit-optimizers/): Low-bit optimizers for PyTorch. Contribute to thu-ml/low-bit-optimizers development by creating an account on GitHub.
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [01-ai/Yi-34B-200K · Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b8qnqv/yi34b200k_model_update_needleinahaystack_improved/): no description found

  

---


### Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1215366139400421396) (1 messages): 

```html
<ul>
  <li><strong>New Model Unveiled: Genstruct 7B</strong>: Nous Research announces the release of <strong>Genstruct 7B</strong>, an instruction-generation model that can create valid instructions from raw text, allowing for the creation of new finetuning datasets. The model, inspired by the Ada-Instruct paper, is designed to generate questions for complex scenarios, promoting detailed reasoning.</li>
  <li><strong>User-Informed Generative Training</strong>: The <strong>Genstruct 7B</strong> model is grounded in user-provided context, taking inspiration from Ada-Instruct and pushing it further to enhance the reasoning capabilities of subsequently trained models. Available for download on HuggingFace: [Genstruct 7B on HuggingFace](https://huggingface.co/NousResearch/Genstruct-7B).</li>
  <li><strong>Led by a Visionary</strong>: The development of <strong>Genstruct 7B</strong> was spearheaded by `<@811403041612759080>` at Nous Research, signifying a team investment in innovation for instruction-based model training.</li>
</ul>
```


**Links mentioned**:

[NousResearch/Genstruct-7B · Hugging Face](https://huggingface.co/NousResearch/Genstruct-7B): no description found

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1214871245552746516) (329 messages🔥🔥): 

- **Anthropic releases chat-oriented AI**: @teknium announces a new model from Anthropic called [**Genstruct-7B-GGUF**](https://huggingface.co/gguf/Genstruct-7B-GGUF), a generative model that can create dialogues and instruction-based content.
- **Discussion about Claude 3's performance**: `@proprietary` exclaims about the **Claude 3** model's impressive capabilities, sparking curiosity and requests to share outputs.
- **Evaluating a Twitter IQ Test Chart**: An incorrect **IQ test chart** from Twitter is discussed. @makya2148 and others express skepticism about the reported IQ scores for AI models like Claude 3 and GPT-4.
- **GPT-5 Release Rumors Circulate**: @sanketpatrikar shares rumors of a potential **GPT-5** release, leading to speculation but a consensus of skepticism amongst the chat participants.
- **Inflection AI Claims Impressive Benchmark Results**: Inflection AI tweets about their new model, Inflection-2.5, claiming it is competitive with **GPT-4** on all benchmarks. `@teknium` and `@mautonomy` discuss the credibility of these claims.

**Links mentioned**:

- [Tweet from Netrunner — e/acc (@thenetrunna)](https://x.com/thenetrunna/status/1765253707866751039?s=46): GPT5_MOE_Q4_K_M.gguf  SHA256: ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c  https ://t.ly/8AN5G
- [Tweet from Inflection AI (@inflectionAI)](https://x.com/inflectionai/status/1765751898001608793): Pi just got a huge upgrade! It’s now powered by our latest LLM: Inflection-2.5, which is neck and neck with GPT-4 on all benchmarks and used less than half the compute to train.  Pi now has world clas...
- [Tweet from Emad (@EMostaque)](https://x.com/emostaque/status/1765680597597372823?s=46): @Teknium1 Less stable above 7b. Transformer engine has it as main implementation. Intel have one too and Google have int8
- [gguf/Genstruct-7B-GGUF · Hugging Face](https://huggingface.co/gguf/Genstruct-7B-GGUF): no description found
- [Swim In GIF - Swim In Swimming - Discover &amp; Share GIFs](https://tenor.com/view/swim-in-swimming-pool-underwater-gif-23188415): Click to view the GIF
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): no description found
- [Tweet from Sebastian Majstorovic (@storytracer)](https://fxtwitter.com/storytracer/status/1765410706638160303?s=20): Open source LLMs need open training data. Today I release the largest dataset of English public domain books curated from the @internetarchive and the @openlibrary. It consists of more than 61 billion...
- [Tweet from Prof. Anima Anandkumar (@AnimaAnandkumar)](https://fxtwitter.com/AnimaAnandkumar/status/1765613815146893348?s=20): For the first time, we show that the Llama 7B LLM can be trained on a single consumer-grade GPU (RTX 4090) with only 24GB memory. This represents more than 82.5% reduction in memory for storing optimi...
- [Tweet from FxTwitter / FixupX](https://fxtwitter.com/AnimaAnandku): Sorry, that user doesn't exist :(
- [Tweet from Daniel Han (@danielhanchen)](https://fxtwitter.com/danielhanchen/status/1765446273661075609?s=20): Found more bugs for #Gemma: 1. Must add &lt;bos&gt; 2. There’s a typo for &lt;end_of_turn&gt;model 3. sqrt(3072)=55.4256 but bfloat16 is 55.5 4. Layernorm (w+1) must be in float32 5. Keras mixed_bfloa...
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl): In this blog post you will learn how to fine-tune LLMs using Hugging Face TRL, Transformers and Datasets in 2024. We will fine-tune a LLM on a text to SQL dataset.
- [Microsoft&#x27;s new deal with France&#x27;s Mistral AI is under scrutiny from the European Union](https://apnews.com/article/european-union-microsoft-mistral-competition-antitrust-05d6eb911e56f88b7da20ebc224efac4): The European Union is looking into Microsoft’s partnership with French startup Mistral AI. It&#x27;s part of a broader review of the booming generative artificial intelligence sector to see if it rais...
- [llama_index/llama-index-packs/llama-index-packs-raptor/llama_index/packs/raptor at main · run-llama/llama_index](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-raptor/llama_index/packs/raptor): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Weyaxi/Einstein-v4-7B · Hugging Face](https://hf.co/Weyaxi/Einstein-v4-7B): no description found
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1765851433448944125): 🎉 Exciting News!   🧑‍🔬 Meet Einstein-v4-7B, a powerful mistral-based supervised fine-tuned model using diverse high quality and filtered open source datasets!🚀  ✍️ I also converted multiple-choice...
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [WIP: galore optimizer by maximegmd · Pull Request #1370 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370): Adds support for Galore optimizers Still a WIP, untested.
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit
- [GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.](https://github.com/PKU-YuanGroup/Open-Sora-Plan): This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project. - PKU-YuanGroup/Open-Sora-Plan

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1214874529369825300) (41 messages🔥): 

- **Local OIllama Announcement**: User `@teknium` declared that OIllama is intended for local running, while `@lakshaykc` suggested it could potentially have an endpoint created for backend inference.
- **Current Training on Hermes Update**: `@teknium` confirmed to `@aliissa` that the dataset used for Nous-Hermes-2-Mistral-7B-DPO did not originally include function calling data but a newer version is now in training.
- **Sampling Tech for JSON/Function Calls**: `@ufghfigchv` is refactoring a logit sampler designed for JSON/function calls, which functions well with Hugging Face (HF) and very-large language models (vllm).
- **GPT-4 Free Access Question**: `@micron588` inquired about free access to GPT-4, and `@teknium` clarified availability, pointing to a website called Corcel.io offering free ChatGPT-4 like interactions.
- **Inquiry on Context Length for Nous-Hermes Models**: `@nickcbrown` asked why the context lengths for the Nous-Hermes model built on Mixtral/Mistral had been apparently reduced and expressed a desire for longer contexts for applications like RAG (Retrieval-Augmented Generation).

**Links mentioned**:

- [Corcel · Build with the power of Bittensor](https://corcel.io/): no description found
- [Lilac - Better data, better AI](https://www.lilacml.com/): Lilac enables data and AI practitioners improve their products by improving their data.

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1214888964180287488) (300 messages🔥🔥): 

- **NVIDIA Puts the Brakes on Cross-Platform CUDA**: A recent change that has been stirring up attention is NVIDIA's [ban on using translation layers](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers) for running CUDA on non-NVIDIA chips, targeting projects like ZLUDA which aimed at bringing AMD to parity with NVIDIA on Windows.
- **Midjourney Scrape Shake-Up**:
  Stability AI is accused of scraping Midjourney for prompts and images, resulting in their employees being banned from Midjourney. The incident has sparked various reactions, with some [suggesting it might not be work-related](https://twitter.com/EMostaque/status/1765496173572346182) and others joking about the situation.
- **Marketing Missteps**: Conversation turned to a case where a marketing department is reportedly spending a disproportionate amount on ad conversions, leading to incredulous reactions and discussions on the inefficiency of such spending.
- **SD3 Speculations Amidst Dataset Discussions**: As talk of new datasets like MajorTOM released by `@mikonvergence` on Twitter surfaces, there's anticipation around Stability AI's plans to distribute SD3 invites and make PRs to diffusers, with users discussing the potential and limitations of SD3's architecture.
- **Scraping Etiquette Examination**: Amid the ongoing discussions about scraping, from the technical implications to the social impacts, users stress the importance of proper scraping techniques and etiquette, reinforcing that understanding and abiding by these principles is crucial.

**Links mentioned**:

- [Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers): Translators in the crosshairs.
- [Tweet from Nick St. Pierre (@nickfloats)](https://x.com/nickfloats/status/1765471291300045255): In MJ office hours they just said someone at Stability AI was trying to grab all the prompt and image pairs in the middle of a night on Saturday and brought down their service.   MJ is banning all of ...
- [TwoAbove/midjourney-messages · Datasets at Hugging Face](https://huggingface.co/datasets/TwoAbove/midjourney-messages): no description found
- [Regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001): no description found
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1215172640868925481) (74 messages🔥🔥): 

- **Debate on Training Efficiency and Pruning**: Discussion between `@mkaic` and `@recviking` focused on whether LLMs are parameter-inefficient and if training methods or architectures could be optimized. `@mkaic` suggested that the ability to prune an LLM without a substantial drop in performance indicated potential for more efficient training schemes, while `@recviking` argued that pruning reduces a model's generalizability and that the vastness of potential inputs makes efficiency evaluation complex.
  
- **SVD and Training Slowness**: `@metal63` reported a significant slowdown when applying SVD updates during the training of the Stable Cascade model, with training pausing for 2 minutes for updates. `@thejonasbrothers` expressed a dislike towards SVD due to its slowness, hinting at the practical issues with certain training optimizations.

- **Pruning Effects on Model Performance**: `@thejonasbrothers` noted that pruning models often leads to performance issues such as token repetition and stability concerns, and that while some models may seem saturated at around 7 billion parameters, the situation might differ at larger scales, such as 1 trillion parameters.

- **General Utility of Pruned LLMs**: `@mkaic` maintained that even after heavy pruning, LLMs retain more generalizability than expected and posed questions about the necessity of large parameters for model training versus inference. The potential for major breakthroughs in training more efficient yet comparably effective LLMs was highlighted as a promising research area.

- **Current Structural Limitations**: `@thejonasbrothers` and `@recviking` discussed the structural limitations of current LLMs and the saturation of efficiencies with an especially critical lens on attention-based optimizations. The dialog raised questions about whether the industry is reaching the limits of compression and model efficiency within existing architectures.

**Links mentioned**:

[Neverseenagain Yourleaving GIF - Neverseenagain Yourleaving Oh - Discover &amp; Share GIFs](https://tenor.com/view/neverseenagain-yourleaving-oh-no-he-gif-10093833): Click to view the GIF

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1214868745034666005) (204 messages🔥🔥): 

- **Claude Outshines GPT-4 in Elemental Knowledge**: `@you.wish` and `@kennyphngyn` share their positive experiences with **Claude 3 Opus** (referred to as "C3"), reporting it outperforms GPT-4 on complex tasks and in elementary classes.
- **Debate Over Claude 3's Coding Capabilities**: `@testtm` finds that **Gemini 1.5 Pro** succeeds at a Python GUI code task on the first try, while Claude 3 doesn't, suggesting that both models have strengths and weaknesses.
- **MMLU Dataset Efficacy in Question**: On the subject of **MMLU datasets**, `@privetin` and `@foxalabs_32486` criticize the set for having questions that don’t make logical sense and containing a significant percentage of incorrect answers, calling for its removal from AI model evaluation.
- **Limits of YouTube for AI Model Evaluation Highlighted**: `@7877` asserts that Youtube is not the best source for quick, raw evaluation numbers of AI models due to its focus on entertainment over detailed information, prompting a discussion on alternative evaluation resources.
- **Concerns Over the Potency of Free AI Services**: `@eskcanta` expresses concern about the limited interactions allowed with **Claude's free version** compared to the more generous allowances from OpenAI's services, contemplating the economic sustainability of these AI companies in providing free services.

**Links mentioned**:

[EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1214877732484284467) (29 messages🔥): 

- **Troubles Accessing GPT Account**: User `@haseebmughal_546` faced issues with a frozen page, preventing access to the GPT account.
- **GPT Policy Update on Code**: User `@watcherkk` raised concerns about GPT not providing full codes due to a policy change, mentioning an 'out of policy' message.
- **Service Disruption on OpenAI**: `@qilin111` reported a 6-hour service interruption; `@dystopia78` confirmed a partial API outage, although OpenAI's [status page](https://status.openai.com/) showed "All Systems Operational" at the time.
- **Inconsistent GPT-4 Availability**: Users `@cetacn`, `@emante`, `@liyucheng09`, `@openheroes`, `@ed1431`, and `@malc2987` discussed intermittent access to GPT-4, with varying degrees of operability among users.
- **Mix-Up on GPT and Other Models**: `@cliffsayshi` inquired about using custom models like Human GPT via API, to which `@solbus` clarified that GPTs are exclusive to ChatGPT and not accessible via API, providing links to OpenAI's [model overview](https://platform.openai.com/docs/models/overview) and additional [info on GPTs vs. assistants](https://help.openai.com/en/articles/8673914-gpts-vs-assistants).

**Links mentioned**:

- [OpenAI Status](https://status.openai.com/): no description found
- [OpenAI Status](https://status.openai.com): no description found

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1214966798558961694) (68 messages🔥🔥): 

- **Navigating Channel Posting Requirements**: `@giorgiomufen` was unclear on how to post in a specific channel due to a required tag, and `@eskcanta` assisted by pointing out that one must click on the 'see more tags' options first.
- **Positive Phrasing for Role-play Prompts**: Users `@toothpiks252`, `@eskcanta`, and `@dezuzel` advised `@loamy_` on how to phrase a role-play prompt positively, suggesting it's best to tell the model explicitly what to do instead of what not to do.
- **GPT-5 for Enhanced Problem-Solving**: `@spikyd` expressed confidence that GPT-5 would have a 10% better chance of solving a specific bird enigma puzzle, while `@eskcanta` believed it might simply require more targeted training with the vision model.
- **Improving Randomization in Storytelling**: In a conversation about adding randomness to GPT's choices, `@solbus` recommended using Python's random function via data analysis tools to give `@interactiveadventureai` more variety in AI-generated stories.
- **Seeking Help for GPT Classifier Development**: `@chemlox` sought advice on whether to use a react-based agent or fine-tuning for building a GPT classifier to determine the status of conversations, with `@eskcanta` suggesting to test the base model first before deciding on a more complex solution.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1214966798558961694) (68 messages🔥🔥): 

- **Tag Troubles in Posting**: `@giorgiomufen` had an issue posting in a channel that required selecting a tag first. `@eskcanta` helped them out by pointing out the requirement to click on one of the 'see more tags' options.
- **Positivity Beats Negativity in Instructions**: `@loamy_` sought a positive phrase alternative to "do not repeat", which `@toothpiks252` and `@eskcanta` assisted with by suggesting explicit instructions on desired actions.
- **Bird Enigma and GPT-5's Potential**: `@spikyd` and `@eskcanta` discussed the potential for solving a complex bird enigma with the upcoming GPT-5, with emphasis on improved vision models and reasoning capabilities for such tasks.
- **Random Number Generation Query**: `@interactiveadventureai` inquired about methods for GPT to generate different random seeds for number generation, with `@solbus` suggesting using Python's random function as a possible solution.
- **A Warm Welcome to a New Member**: `@thebornchampion` introduced themselves as an aspiring data analyst and prompt engineering enthusiast, and `@eskcanta` welcomed them while engaging in a discussion about their interests and use cases for ChatGPT.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1214865889795776512) (187 messages🔥🔥): 

- **Discussing Tech Specs and Troubleshooting**: `@kavita_27183` shares their hardware specs, boasting **512 GB RAM** and **24GB VRAM**, while `@jedd1` assists in troubleshooting libstdc++ errors and checking whether the system recognizes VRAM correctly. A shared error message points to a library version issue (`GLIBCXX_3.4.29 not found`), suggesting a need for updating.
- **Server-Side Issues with Local Models**: Diverse local model topics are addressed, including `@_benoitb` encountering API issues with nodes servers, `@datasoul` discussing GGUF-related errors in LM Studio, and `@mattjpow` seeking clarification on server context behavior. Advice and responses are offered by `@heyitsyorkie`.
- **Hardware Recommendations for LLM Work**: `@saber123316` deliberates on getting a **Macbook Pro with M3 Max or M2 Max**, seeking community advice on which would suffice for local LLM processing. The conversation touches on the trade-offs regarding price, performance, and the value of more RAM.
- **Discovering LM Studio's Capabilities**: Users `@aeiou2623` and `@.lodis` explore features ranging from image uploads in conversations to model support. `@heyitsyorkie` provides guidance on loading GGUF files and clarifies that LM Studio is primarily for running local models offline.
- **OpenAI Critique and Alternate AI Services Insights**: `@saber123316` and `@rugg0064` discuss the implications of the revealed non-openness of OpenAI, with mentions of Elon Musk's dissatisfaction and OpenAI's proprietary approach prompting users to consider other AI service subscriptions such as **POE monthly** for accessing various models like Claude and GPT-4.

**Links mentioned**:

- [Inflection-2.5: meet the world&#x27;s best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Reor](https://www.reorproject.org/): AI note-taking app that runs models locally &amp; offline on your computer.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18av9aw/quick_start_guide_to_converting_your_own_ggufs/): no description found
- [RIP Midjourney! FREE &amp; UNCENSORED SDXL 1.0 is TAKING OVER!](https://www.youtube.com/watch?v=A0xUnf5302k&ab_channel=Aitrepreneur): Say goodbye to Midjourney and hello to the future of free open-source AI image generation: SDXL 1.0! This new, uncensored model is taking the AI world by sto...
- [Accelerating LLM Inference: Medusa&#39;s Uglier Sisters (WITH CODE)](https://www.youtube.com/watch?v=0_fZNW59PaA): https://arxiv.org/abs/2401.10774https://github.com/evintunador/medusas_uglier_sisters
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [22,000 H100s later, Inflection 2.5!!!](https://youtu.be/fEpa_Ak6Ec4?si=9bLvLARbKL91o1lp): 🔗 Links 🔗https://inflection.ai/inflection-2-5❤️ If you want to support the channel ❤️Support here:Patreon - https://www.patreon.com/1littlecoder/Ko-Fi - ht...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1214909477992534046) (27 messages🔥): 

- **Optimal AI for Storytelling Specified**: User `@laszlo01` inquired about the best AI model for storytelling using Open Chat 3.5. `@jason_2065` recommended **Mistral-7b-instruct-v0.2-neural-story.Q4_K_M.gguf** with 24 layers and 8192 context size, noting the memory constraints.
  
- **LM Studio Lacks Image-Generation Models**: `@karisna` asked about models capable of drawing images within LM Studio. `@heyitsyorkie` clarified that LM Studio does not support such models and recommended using **Automatic 1111** for Stable Diffusion tasks.

- **Starcoder2 Running Issues and Support**: Users `@b1gb4ng` and `@madhur_11` wondered if starcoder2 could be run via LM Studio, to which `@heyitsyorkie` informed that it is not supported in the current LM Studio version.

- **Exploring Image Generation Alternatives**: `@callmemjinina` sought a model that generates pictures. `@heyitsyorkie` explained that Language Models and LM Studio cannot perform this task, advising to look for **Stable Diffusion** tools and tutorials online.

- **Request for Starcoder2**: `@zachmayer` shared interest in using starcoder2 and posted a link to its [Hugging Face page](https://huggingface.co/TechxGenus/starcoder2-15b-instruct/), but `@wolfspyre` hinted at the need for patience, implying future support might be coming.

**Links mentioned**:

[Kquant03/TechxGenus-starcoder2-15b-instruct-GGUF · Hugging Face](https://huggingface.co/Kquant03/TechxGenus-starcoder2-15b-instruct-GGUF): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1215037455875710976) (4 messages): 

- **Users Seek Guidance**: User `@tiro1cor15_10` expressed a desire for guidance to fully realize the potential they see in using the service.
- **Proxy Support Lacking on macOS**: `@calmwater.0184` provided **feedback** indicating that **LM Studio currently does not support the Proxy feature on macOS 14.3** (v0.2.16), impacting the ability to search or download models.
- **Feature Enhancement Suggestion**: `@calmwater.0184` suggested that **LM Studio** could look into the user experience and features of [Embra AI](https://embra.app/) to become a more efficient **Productivity Booster Assistant** for users at all levels.
- **Channel Usage Clarification**: `@heyitsyorkie` directed users to **stop using** the designated feedback channel for help requests and instead use the appropriate help channel.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1215009912669872138) (46 messages🔥): 

- **Extreme Context Experiments on the Smaug Model**: `@goldensun3ds` reported running a test involving a **200K context** in the Smaug model, which displayed erratic RAM usage between 70GB and 20GB, under only CPU usage. Their latest message indicated no outputs yet and continued fluctuations in RAM usage.

- **VRAM and Power Supply Discussions**: `@wilsonkeebs` sought advice for the smallest PSU to power a standalone RTX 3090, and `@heyitsyorkie` recommended a **minimum of 550W** but suggested 750W is standard. Wilsonkeebs emphasized the need for sufficient PCIe cables rather than the wattage itself.

- **Exploring Large Contexts for Niche Use Cases**: `@aswarp` brought up the potential for LLMs to process very large data sets monthly like entire codebases for thorough reporting, acknowledging the trade-off with processing time. Aswarp sees potential particularly for applications in government and smaller businesses.

- **VRAM Demand for Processing in LLM Studio**: `@jason_2065` mentioned using 105,000 context and experiencing high resource usage with 42GB RAM and over 20GB VRAM, which aligns with the high demands discussed by others when handling large contexts or datasets in memory-intensive LLM tasks.

- **Managing Mismatched VRAM for Machine Learning Models**: The conversation touched on difficulties associated with mismatched VRAM when running LLMs, noting adjustments to the LLM preset file for GPU allocation which `@goldensun3ds` finds troublesome due to requiring a restart of the LM Studio.



**Links mentioned**:

- [Razer Core X - Thunderbolt™ 3 eGPU | Razer United Kingdom](https://www.razer.com/gb-en/gaming-egpus/razer-core-x): Now compatible with Mac and Windows laptops, featuring 3-slot PCI-Express desktop graphic cards, 650W power supply, and charges via USB-C.
- [PSU for NVIDIA GeForce RTX 3090 | Power Supply Calculator](https://www.whatpsu.com/psu/gpu/NVIDIA-GeForce-RTX-3090): See what power supply you need for your NVIDIA GeForce RTX 3090

  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1215334149397680168) (2 messages): 

- **Seeking Speed Enhancement Techniques**: `@alluring_seahorse_04960` is looking for methods to **increase response speed** for an unspecified process. They are also experiencing an error: *'Connection to telemetry.crewai.com timed out'*.
- **Suggestion for Baseline Local Operation**: In response, `@wolfspyre` suggests establishing a simple baseline operation that can run locally to possibly address the speed issue.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1214924939249983538) (85 messages🔥🔥): 

- **Tackling the Preprompt Syntax**: `@nxonxi` inquired about the correct syntax for modifying the `default_system_message` in Open Interpreter settings across different operating systems. They shared their struggles and attempts to alter the system message for Linux, Windows, and WSL.

- **Confusion Over `-s` and DOS Documentation**: `@nxonxi` mentioned confusion over how to use Open Interpreter's prompt settings, discussing the `-s` or `--system_message` option, and shared a link to the documentation, which led to further discussion with `@1sbefore` on finding the correct usage and commands.

- **Profiles and Configurations - The Journey Continues**: Throughout the conversation, `@1sbefore` offered assistance, suggesting to check the correct paths and configurations in Open Interpreter's Python environments, while `@nxonxi` reported on various unsuccessful attempts to modify the prompt or use profiles effectively.

- **Exploring Code-Trained Models**: In the latter part of the conversation, the discussion shifted toward experiences with different language models like `deepseek-coder-6.7B-instruct-GGUF`, as well as the integration of prompts and system messages within these models. `@1sbefore` shared a link to Hugging Face hosting potentially useful GGUFs and provided insights on the models' performances.

- **Belief in the Power of Curiosity**: When faced with perplexities about working with the Open Interpreter and language models, `@1sbefore` encouraged `@nxonxi` to embrace their curiosity as they ventured to find the right sources and setups. The exchange was filled with trial-and-error experiences and shared learning moments including attempts to clone git repositories and adjustments of Python environments.

**Links mentioned**:

- [owao/LHK_DPO_v1_GGUF · Hugging Face](https://huggingface.co/owao/LHK_DPO_v1_GGUF): no description found
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#system-message): no description found
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#custom-instructions): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1215391107030978631) (1 messages): 

- **User Survey Announced**: `@seldo_v` encourages everyone to take a **3-minute user survey** to help LlamaIndex understand their user base better. The survey can be found at [SurveyMonkey](https://www.surveymonkey.com/r/PNSP3P9) and aims to improve documentation, demos, and tutorials.

**Links mentioned**:

[LlamaIndex user survey](https://www.surveymonkey.com/r/PNSP3P9): Take this survey powered by surveymonkey.com. Create your own surveys for free.

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1214999109451387021) (5 messages): 

- **Launch of LlamaParse JSON Mode**: The LlamaIndex team is excited to announce the new **LlamaParse JSON Mode**, which simplifies the RAG pipeline creation by parsing text and images from a PDF into a structured dictionary. This feature enhances capabilities when combined with multimodal models like claude-3 opus. [View tweet](https://twitter.com/llama_index/status/1765439865351766135).
- **LlamaParse testing by AIMakerspace**: `@AIMakerspace` tested **LlamaParse** with notable results and an in-depth analysis published that details its functionalities and performance metrics. [In-depth look at LlamaParse](https://t.co/6mLTVpmzbN).
- **Integration with VideoDB for RAG Over Video Streams**: LlamaIndex introduced an integration with `@videodb_io`, enabling the upload, search, and streaming of videos directly within LlamaIndex, by words spoken or visual scenes presented. [Announcement tweet](https://twitter.com/llama_index/status/1765481657765912599).
- **Comprehensive Video Guide for Claude 3**: A new video guide has been released offering a comprehensive tutorial on using Claude 3 for various applications, including Vanilla RAG, Routing, and Sub-question query planning with LlamaIndex's tools. [Claude 3 Cookbook](https://t.co/P346byw4hT).
- **LlamaIndex User Survey for Enhanced Resources**: LlamaIndex is conducting a 3-minute user survey to better understand its users' expertise and needs in order to tailor documentation, demos, and tutorials more effectively. [Take the user survey](https://t.co/cadlrPztJo).

**Links mentioned**:

[LlamaIndex user survey](https://t.co/cadlrPztJo): Take this survey powered by surveymonkey.com. Create your own surveys for free.

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1214874263400751174) (298 messages🔥🔥): 

- **A* Algorithm Chat**: `@nouiri` asked if the A* algorithm could be applied to similarity search. `@whitefang_jr` confirmed it is possible by subclassing the embedding class and changing the similarity search method, referencing LlamaIndex's default use of cosine similarity.

- **Configuring Chatbots for Contextual Outputs**: `@techexplorer0` inquired about configuring a RAG chatbot for brief, context-specific responses. `@kapa.ai` suggested using a `ResponseSynthesizer` or post-processing responses to achieve concise outputs, linking to the [LlamaIndex documentation](https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline.html) for exemplified setup.

- **Query Engine Customization on LlamaIndex**: `@cheesyfishes` replied to various implementation queries, explaining that chat engines typically accept strings, the chunking in LlamaIndex isn't randomized, and suggesting explorations of the source code for deeper issues such as the use of Gemini as a chat engine or embedding within Azure OpenAI.

- **Ingesting Documents with Slack**: `@habbyman` sought advice on best practices for Slack document ingestion, looking to maintain individual message metadata without losing conversational context.

- **LlamaIndex Vector Store Recommendations**: New users like `@generalenthu` asked for vector store recommendations compatible with LlamaIndex, receiving suggestions from `@cheesyfishes` and `@jessjess84` to try Qdrant, ChromaDB, or Postgres/pgvector for their extensive documentation and robust user bases.

- **Discrepancies Between LLM Direct Queries and VectorStoreIndex**: `@jessjess84` experienced subpar responses from LlamaIndex's `VectorStoreIndex` compared to direct LLM queries, with `@teemu2454` attributing this to prompt templates used by `query_engine` and advising adjustments for improved results. In response to a separate query by the same user, `@teemu2454` clarified that the `VectorStoreIndex` is separate from the LLM used to process texts, with embeddings used only to fetch text for context during LLM queries.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [Building a Slack bot that learns with LlamaIndex, Qdrant and Render — LlamaIndex, Data Framework for LLM Applications](https://www.llamaindex.ai/blog/building-a-slack-bot-that-learns-with-llamaindex-qdrant-and-render-c88d4aa72840): LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).
- [Chat Stores - LlamaIndex 🦙 v0.10.17](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html): no description found
- [OpenAI - LlamaIndex 🦙 v0.10.17](https://docs.llamaindex.ai/en/stable/examples/llm/openai.html#openai): no description found
- [llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-opensearch/llama_index/vector_stores/opensearch/base.py at 0ae69d46e3735a740214c22a5f72e05d46d92635 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/0ae69d46e3735a740214c22a5f72e05d46d92635/llama-index-integrations/vector_stores/llama-index-vector-stores-opensearch/llama_index/vector_stores/opensearch/base.py#L249): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-legacy/llama_index/legacy/llms/openai_like.py at f916839e81ff8bd3006fe3bf4df3f59ba7f37da3 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/f916839e81ff8bd3006fe3bf4df3f59ba7f37da3/llama-index-legacy/llama_index/legacy/llms/openai_like.py#L23): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/base/embeddings/base.py at df7890c56bb69b496b985df9ad28121c7f620c45 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/df7890c56bb69b496b985df9ad28121c7f620c45/llama-index-core/llama_index/core/base/embeddings/base.py#L52): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [GitHub - mominabbass/LinC: Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot;](https://github.com/mominabbass/LinC): Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot; - mominabbass/LinC
- [OMP_NUM_THREADS](https://www.openmp.org/spec-html/5.0/openmpse50.html>).): no description found
- [llama_index/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/base.py at f916839e81ff8bd3006fe3bf4df3f59ba7f37da3 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/f916839e81ff8bd3006fe3bf4df3f59ba7f37da3/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/base.py#L294): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [[Bug]: Issue with EmptyIndex and streaming. · Issue #11680 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/11680#issuecomment-1981070708): Bug Description Im trying to create a simple Intent Detection agent, the basic expected functionality is to select between to queryengines with RouterQueryEngine, one q_engine with an emptyindex, t...
- [Available LLM integrations - LlamaIndex 🦙 v0.10.17](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules.html): no description found
- [Implement EvalQueryEngineTool by d-mariano · Pull Request #11679 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11679): Description Notice I would like input on this PR from the llama-index team. If the team agrees with the need and approach, I will provide unit tests, documentation updates, and Google Colab noteboo...
- [Chroma Multi-Modal Demo with LlamaIndex - LlamaIndex 🦙 v0.10.17](https://docs.llamaindex.ai/en/stable/examples/multi_modal/ChromaMultiModalDemo.html): no description found
- [Multimodal Retrieval Augmented Generation(RAG) | Weaviate - Vector Database](https://weaviate.io/blog/multimodal-rag): A picture is worth a thousand words, so why just stop at retrieving textual context!? Learn how to perform multimodal RAG!
- [Custom Response - HTML, Stream, File, others - FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>)): FastAPI framework, high performance, easy to learn, fast to code, ready for production
- [no title found](https://medium.com/@its.jwho/errorhandling-vulnerability-tests-on-gemini-19601b246b52.): no description found
- [llama_index/llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/utils.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-vertex/llama_index/llms/vertex/utils.py): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1215023549782163537) (1 messages): 

- **New Approach to Enhancing In-context Learning**: `@momin_abbas` shared their latest work focusing on improving in-context learning through [Few-Shot Linear Probe Calibration](https://github.com/mominabbass/LinC). They ask for support by starring the [GitHub repository](https://github.com/mominabbass/LinC).

**Links mentioned**:

[GitHub - mominabbass/LinC: Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot;](https://github.com/mominabbass/LinC): Code for &quot;Enhancing In-context Learning with Language Models via Few-Shot Linear Probe Calibration&quot; - mominabbass/LinC

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1215049516005331025) (14 messages🔥): 

- **Tea Time Tales from Twitter**: User `@420gunna` amusingly described bringing news from Twitter, like a messenger arriving with updates.
- **Midjourney and Stability AI Clash**: `@420gunna` highlighted a controversy involving Stability AI allegedly scraping data from Midjourney, leading to a ban of Stability AI employees from Midjourney as reported by `@nickfloats` in this [Twitter post](https://x.com/nickfloats/status/1765471291300045255?s=20).
- **Mi5 Type Confusion Cleared Up**: User `@nav10` humorously clarified initial misconceptions about a data scrape incident being likened to a spy movie scenario but was actually a Discord scrape.
- **Newsletter Expander or Troublemaker?**: `@guardiang` playfully suggested `@272654283919458306` might be expanding their newsletter's scope in light of the recent data scrape incident reported on Twitter.
- **Laughing Off the Discord Drama**: `@swyxio` posted a GIF from [Tenor.com](https://tenor.com/btRZl.gif) in response to the unfolding drama involving Stability AI and Midjourney.

**Links mentioned**:

- [Tweet from Nick St. Pierre (@nickfloats)](https://x.com/nickfloats/status/1765471291300045255?s=20): In MJ office hours they just said someone at Stability AI was trying to grab all the prompt and image pairs in the middle of a night on Saturday and brought down their service.   MJ is banning all of ...
- [Tweet from Prof. Anima Anandkumar (@AnimaAnandkumar)](https://x.com/animaanandkumar/status/1765613815146893348?s=46&t=PW8PiFwluc0tdmv2tOMdEg): For the first time, we show that the Llama 7B LLM can be trained on a single consumer-grade GPU (RTX 4090) with only 24GB memory. This represents more than 82.5% reduction in memory for storing optimi...
- [Obi Wan Im Not Brave Enough For Politics GIF - Obi Wan Im Not Brave Enough For Politics Talk - Discover &amp; Share GIFs](https://tenor.com/btRZl.gif): Click to view the GIF

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1215011107530342512) (4 messages): 

- **New Podcast Episode Alert**: `@swyxio` announced the latest podcast episode featuring `<@776472701052387339>` is now live, available via a tweet posted [here](https://twitter.com/swyx/status/1765452280915230904).
- **Podcast Discussion Hits Hacker News**: `@swyxio` shared that the podcast with Soumith is also garnering attention on [Hacker News](https://news.ycombinator.com).
- **Model Serving Paper Presentation**: `@swyxio` invited `<@&1107197669547442196>` members to join `<@720451321991397446>`'s presentation on the Model Serving survey paper in their [Discord channel](https://discord.com/channels/822583790773862470/1197350122112168006).
  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1215026406396796949) (204 messages🔥🔥): 

- **Volunteer Effort Acknowledged**: The community expressed gratitude towards `@720451321991397446` for volunteering for a presentation. `@eugeneyan` and others cheered on the effort.
- **Fancy Footwork with Models**: `@swizec` discussed noticeable performance differences when running Ollama on Intel vs M2, whilst `@amgadoz` reviewed technical aspects such as parallelism and look-ahead decoding in large language models.
- **Inference Optimization Deep Dive**: Speculative decoding's effectiveness was a topic of interest with `@shivdinho`, `@yikesawjeez`, and others discussing how it might vary with hardware. There were also recommendations for resources like [Drugs by EGjoni](https://github.com/EGjoni/DRUGS?tab=readme-ov-file "Drugs") and speculations about using FlashAttention for speed improvements in large model serving.
- **Decentralized and Distributed Deliberations**: The channel touched on distributed training (`@yikesawjeez` mentioned reading [DiLoCo paper](https://arxiv.org/abs/2311.08105 "DiLoCo")) and there were discussions about the potential pitfalls of GPU configurations affecting model output from different instances (`@ayenem` reflected on an OpenAI incident).
- **Model Serving Survey Presentation**: `@swyxio` provided a link to a [Google Slides presentation](https://docs.google.com/presentation/d/1Jde7Vx0BQNMClRCLlesy36hlC3VUJRceEYIxdkl1j68/edit?usp=sharing "Model Serving Survey Paper - Paper Club") and the group shared thoughts and reactions to the material presented. There was significant discussion on distributed inference and the trade-offs of different architectures.


**Links mentioned**:

- [Join Slido: Enter #code to vote and ask questions](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions): Participate in a live poll, quiz or Q&A. No login required.
- [Join Slido: Enter #code to vote and ask questions](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb): Participate in a live poll, quiz or Q&A. No login required.
- [SpecInfer](https://flexflow.ai/specInfer/): no description found
- [Monk](https://monk.io): Monk is the AI DevOps for the cloud. Let your infrastructure take care of itself.
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://legendary-slicer-267.notion.site/Paper-reading-LLM-Inference-caa072f4e8304acd9fefbcafb1305cd1): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Datasets for Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2402.18041): This paper embarks on an exploration into the Large Language Model (LLM) datasets, which play a crucial role in the remarkable advancements of LLMs. The datasets serve as the foundational infrastructu...
- [Quickstart &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): no description found
- [no title found](https://news.ycombinator.com/item?id=39597847): no description found
- [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/): Overcoming the bottleneck of human annotations in instruction-tuning, preference-tuning, and pretraining.
- [FlashAttention 2: making Transformers 800% faster w/o approximation - with Tri Dao of Together AI](https://www.latent.space/p/flashattention): How FlashAttention became the new industry standard architecture, how FlashAttention 2 is 2x faster still, life inside the Stanford Hazy Research lab, and hints of the post-Transformers future
- [Welcome to SkyPilot! &#8212; SkyPilot documentation](https://skypilot.readthedocs.io/en/latest/index.html): no description found
- [Load Balancing is Impossible ](https://www.infoq.com/presentations/load-balancing/): Tyler McMullen discusses load balancing techniques and algorithms such as Randomized Least-conns, Join-Idle-Queue, and Load Interpretation. Load balancing perfectly may be impossible in the real world...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/199bq25/vllm_vs_aphrodite_engine_and_other_alternatives): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/199bq25/vllm_vs_aphrodite_engine_and_other_alternatives/): no description found
- [Welcome to SkyPilot! &#8212; SkyPilot documentation](https://skypilot.readthedocs.io/en/latest): no description found
- [Model Serving Survey Paper - Paper Club](https://docs.google.com/presentation/d/1Jde7Vx0BQNMClRCLlesy36hlC3VUJRceEYIxdkl1j68/edit): Model Serving Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems https://arxiv.org/abs/2312.15234v1
- [Model Serving Survey Paper - Paper Club](https://docs.google.com/presentation/d/1Jde7Vx0BQNMClRCLlesy36hlC3VUJRceEYIxdkl1j68/edit?usp=sharing): Model Serving Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems https://arxiv.org/abs/2312.15234v1
- [GitHub - TheBlokeAI/dockerLLM: TheBloke&#39;s Dockerfiles](https://github.com/TheBlokeAI/dockerLLM): TheBloke&#39;s Dockerfiles. Contribute to TheBlokeAI/dockerLLM development by creating an account on GitHub.
- [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105): Large language models (LLM) have become a critical component in many applications of machine learning. However, standard approaches to training LLM require a large number of tightly interconnected acc...
- [Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188): Many NLP tasks benefit from using large language models (LLMs) that often have more than 100 billion parameters. With the release of BLOOM-176B and OPT-175B, everyone can download pretrained models of...
- [GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!](https://github.com/EGjoni/DRUGS?tab=readme-ov-file): Stop messing around with finicky sampling parameters and just use DRµGS! - EGjoni/DRUGS
- [GitHub - OpenNMT/CTranslate2: Fast inference engine for Transformer models](https://github.com/OpenNMT/CTranslate2): Fast inference engine for Transformer models. Contribute to OpenNMT/CTranslate2 development by creating an account on GitHub.
- [FireAttention — Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs](https://blog.fireworks.ai/fireattention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs-a29a85ad28d0): Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI&#39;s large-scale inference engine](https://github.com/PygmalionAI/aphrodite-engine): PygmalionAI&#39;s large-scale inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.
- [GitHub - lilacai/lilac: Curate better data for LLMs](https://github.com/lilacai/lilac): Curate better data for LLMs. Contribute to lilacai/lilac development by creating an account on GitHub.

  

---



### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1215326440371785790) (1 messages): 

- **New Benchmarks for Korean Language Model Evaluation**: `@gson_arlo` announced two new evaluation datasets, **[Hae-Rae Bench](https://arxiv.org/abs/2309.02706)** and **[K-MMLU](https://arxiv.org/abs/2402.11548)**, designed to test language models' proficiency in Korean. The **Hae-Rae Bench** assesses models' knowledge of Korean culture, while **K-MMLU** is focused on Korea-specific questions and includes a challenging subset for current models.

- **Call for Contributions to Multilingual Benchmarks**: In an effort to improve evaluation practices for languages other than English and Chinese, `@gson_arlo` invites individuals who speak non-mainstream languages or belong to non-mainstream cultures within English-speaking countries to join the **<#1208111628051152969>** channel. They can contribute by designing benchmarks that assess language model competencies significant to their cultures.
  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1214866432039460904) (78 messages🔥🔥): 

- **Exploring Internal Batching Logic of vLLM**: `@rwamit` inquired how to perform batched inference using **vLLM**. `@baber_` clarified that **vLLM** has internal batching, which means it's unnecessary to manually implement batching when using vLLM.

- **Commentary Sought on AI Regulation**: `@wonkothesensible` shared a [link](https://www.regulations.gov/document/NTIA-2023-0009-0001) regarding a request for public commentary on the regulation of open source AI and freely available models. The document calls for a report on the risks and benefits of "dual-use foundation models."

- **The BitNet vs. Full-Precision NN Efficiency Debate**: With the release of BitNet b1.58, `@kyo_takano` offered an [introductory notebook](https://gist.github.com/kyo-takano/9d8376a35acb5e6be090e1a90271050e) on Ternary Neural Networks, suggesting that they are inefficient compared to full-precision NNs during training, despite faster inference speeds.

- **MidJourney Discord Drama**: `@teknium` reported allegations that Stability AI disrupted MidJourney services, leading to a ban on those behind the incident. The circumstances and reasons remain obscure, with `@stellaathena` and others questioning how scraping Discord would impact MidJourney's servers.

- **Potential Collaboration Opportunity with an Expert**: `@andrew_f0874` is seeking part-time, voluntary research collaborations in AI/ML. With a PhD from Cornell and prior experience as a research scientist at Google focusing on privacy-preserving technology, he could be especially useful for projects at the intersection of AI/ML and his areas of expertise.

**Links mentioned**:

- [Quickstart &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): no description found
- [Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models](https://arxiv.org/abs/2402.14714): This report introduces \texttt{EEVE-Korean-v1.0}, a Korean adaptation of large language models that exhibit remarkable capabilities across English and Korean text understanding. Building on recent hig...
- [Regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001): no description found
- [Tweet from Nick St. Pierre (@nickfloats)](https://x.com/nickfloats/status/1765471291300045255?s=46): In MJ office hours they just said someone at Stability AI was trying to grab all the prompt and image pairs in the middle of a night on Saturday and brought down their service.   MJ is banning all of ...
- [Introduction to Ternary Neural Networks](https://gist.github.com/kyo-takano/9d8376a35acb5e6be090e1a90271050e): Introduction to Ternary Neural Networks. GitHub Gist: instantly share code, notes, and snippets.
- [Megatron-DeepSpeed/tasks/eval_harness/evaluate.py at main · microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/tasks/eval_harness/evaluate.py): Ongoing research training transformer language models at scale, including: BERT &amp; GPT-2 - microsoft/Megatron-DeepSpeed

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1214930285137887295) (77 messages🔥🔥): 

- **Pythia Model Suite by EleutherAI**: `@alxsp.` provided a [link](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) to the Pythia model suite, mentioning its training on the same dataset.
- **Adaptive Recurrent Vision Paper Discussion**: `.the_alt_man` shared a [link](https://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html) to a NeurIPS 2023 paper about zero-shot computation scaling in vision models explaining it as a "universal transformer but with CNN."
- **Batch Normalization vs. Token Normalization Debate**: A debate emerged around whether to token normalize or batch normalize loss when training models, as raised by `@thatspysaspy`, with insights from `@ai_waifu` suggesting normalization by target tokens, leading to a detailed discussion on the topic.
- **Innovative Memory Reduction Strategy: GaLore**: Discussion of a new memory-efficient training strategy, Gradient Low-Rank Projection (GaLore), was sparked by `@fredholm` with `@xylthixlm`, `@random_string_of_character`, and `@ai_waifu` examining its claims on memory savings and improved results over full-rank updates.
- **Optimizer Hook Insights with PyTorch**: Conversations around the technical implementation of optimizer steps using gradients (`@xylthixlm`, `@_inox`, and others) highlighted PyTorch's ability to index a dictionary with a parameter and the potential impacts on models with tied parameters.

**Links mentioned**:

- [Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels](https://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html): no description found
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507): Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-ran...
- [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950): Value functions are a central component of deep reinforcement learning (RL). These functions, parameterized by neural networks, are trained using a mean squared error regression objective to match boo...
- [Pythia Scaling Suite - a EleutherAI Collection](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1): no description found
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666): Today&#39;s probabilistic language generators fall short when it comes to producing coherent and fluent text despite the fact that the underlying models perform well under standard metrics, e.g., perp...
- [pytorch/torch/_tensor.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_tensor.py#L1059): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [GaLore/torchrun_main.py at master · jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py#L356): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [WIP: galore optimizer by maximegmd · Pull Request #1370 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370): Adds support for Galore optimizers Still a WIP, untested.

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1215222634841645066) (15 messages🔥): 

- **Harnessing Desired Output Formats**: `@pminervini` inquired about how to customize model outputs to a specified format using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). `@baber_` advised on modifying the `generate_until` method and utilizing `_loglikelihood_tokens` preceded by conditionally making the generate call based on calculated log_probs.

- **MCQA Eval/Artifacts Paper Shared by Creator**: `@nish5989` shared [their recent paper](https://twitter.com/NishantBalepur/status/1764729478893174977) focused on multiple-choice questions evaluation and dataset artifacts. A discussion ensued regarding the impact of following or ignoring irrelevant instructions in model prompts, with `@hailey_schoelkopf` referencing related works on how models react to prompts.

- **Consistency in Fine-Tuning**: `@karatsubabutslower` questioned the lack of a standardized approach for fine-tuning models on benchmarks like the GLUE dataset. The lm-evaluation-harness provides a standardized evaluation method, but there does not seem to be a similar framework for the fine-tuning process.

- **Evaluating with Generation vs. Loglikelihood**: `@hailey_schoelkopf` expressed interest in how often models fail to produce answers in the correct format, using generation methods rather than loglikelihood. `@nish5989` responded that, as noted in their paper's appendix, validity typically wasn't an issue, especially in stricter settings, with considerations for future experiments using likelihood methods.

- **Discussion on Multilingual Evaluation Criteria**: `@seanbethard` queried the preference for language-specific evaluation criteria over crosslingual ones in language understanding and reasoning. They questioned the efficacy and necessity of language-specific eval criteria, especially if it precludes non-native speakers from creating it.

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/models/huggingface.py at 9e6e240229429d2214bc281bed7a4e288f5169a1 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/9e6e240229429d2214bc281bed7a4e288f5169a1/lm_eval/models/huggingface.py#L1186).): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [Multiple Choice Question Standard Deviation · Issue #1524 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1524): I saw that the multiple choice type evaluation would compute the metrics along with standard deviation. From my understanding, multiple choice answer is chosen from the choice with highest probabil...
- [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247): Recently, a boom of papers has shown extraordinary progress in zero-shot and few-shot learning with various prompt-based models. It is commonly argued that prompts help models to learn faster in the s...
- [Are Language Models Worse than Humans at Following Prompts? It&#39;s Complicated](https://arxiv.org/abs/2301.07085): Prompts have been the center of progress in advancing language models&#39; zero-shot and few-shot performance. However, recent work finds that models can perform surprisingly well when given intention...

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1215034823207419994) (34 messages🔥): 

- **Optimizer Memory Peaks Under Scrutiny**: `@tastybucketofrice` raised an issue regarding memory peaks during the optimizer step, pointing to [GitHub issue #1160](https://github.com/EleutherAI/gpt-neox/issues/1160) and PyTorch memory profiling for more context. `@gaindrew` requested specific configuration details to faithfully reproduce the problem for further analysis.

- **A Night of Dependency Conflicts**: `@biiter` mentioned facing multiple challenges, including incompatible PyTorch version dependencies and a crashing machine due to parallel compilation. They succeeded in running the setup on Ubuntu 22.04 with Cuda 12.3 after a series of workarounds, including installing apex from the NVIDIA git repository and capping parallel compilation ([flash-attention info](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)).

- **Docker Image, a Solution to Dependency Hell?**: `@tastybucketofrice` suggested using their newly rebased docker container around pytorch NGC to alleviate recent issues with dependencies like apex ([GitHub commit #1170](https://github.com/EleutherAI/gpt-neox/commit/119950c7faee46e7929baac8640c84aa9fda4d2b)). `@tfidia` recommended docker + enroot for a lightweight installation alternative, mentioning its support by the Slurm plugin pyxis for containerized job launches.

- **Poetry for Dependency Management**: `@catboy_slim_` proposed moving dependencies into poetry for more deterministic package management, discussing both the challenges and aspirations for a stable source installation outside of Docker. They stressed the need for a source build that mirrors the reliability of a docker environment.

- **Challenges with Fused Backward/Optimizer Implementations**: `@gaindrew` describes progress on a fused backward/optimizer implementation that has led to a significant reduction in peak memory usage at the expense of breaking certain DeepSpeed and logging functionalities. Further work is set to address these issues and is specifically aimed at Adam optimizers.

**Links mentioned**:

- [GitHub: Let’s build from here](https://github.com): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Single node Pythia 14M training on ngc pytorch 24.02 container (#1170) · EleutherAI/gpt-neox@119950c](https://github.com/EleutherAI/gpt-neox/commit/119950c7faee46e7929baac8640c84aa9fda4d2b): * Pythia 14M training on ngc pytorch 24.02 container
 
 * pre-commit
 
 ---------
 
 Co-authored-by: Quentin Anthony &lt;qganthony@yahoo.com&gt;
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [GitHub - EleutherAI/gpt-neox: An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.](https://github.com/EleutherAI/gpt-neox?tab=readme-ov-file#pytorch-memory-profiling): An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. - EleutherAI/gpt-neox
- [PyTorch Lightning Fused optimizer step · Issue #1160 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/1160): Add PyTorch Lightning memory optimizations. https://lightning.ai/pages/community/tutorial/faster-pytorch-training-by-reducing-peak-memory/

  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1214874666863300668) (106 messages🔥🔥): 

- **Entrepreneurial Chat Space Inquiry**: `@snoozyy` asked if there is a discussion space for small company entrepreneurs using open source models in the Discord; no specific channel or place was suggested in the provided messages.
- **Learning TTS and Model Training**: `@ericpeter24` expressed he's new to Text-to-Speech models and is looking to train a model using coqui with a dataset on HuggingFace but doesn't know where to start.
- **Showcasing Personal Projects**: `@anuragshas` inquired about creating featured models or spaces on their personal Hugging Face profile; the question wasn't directly addressed in the subsequent messages.
- **Exploring Multimodal Models**: `@welltoobado` mentioned [multi_token](https://github.com/sshh12/multi_token), a model for embedding various modalities into large language models, and discussed resource requirements for running such models with `@kuki1941`.
- **Inference API in Android Studio**: `@hari4626` sought guidance on how to use Inference APIs or endpoints in Android Studio, with `@amirgame197` suggesting making a web request with the correct data using a programming language supported by Android Studio.

**Links mentioned**:

- [Inflection-2.5: meet the world&#x27;s best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [@andrewyng on Hugging Face: &quot;DeepLearning.AI just announced a new short course: Open Source Models with…&quot;](https://huggingface.co/posts/andrewyng/643116669090778): no description found
- [Haiper | Generative AI For Video Content Creation](https://haiper.ai/): Video creation AI products crafted to empower individuals in creatively expressing themselves. 
- [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032): Transformers are the dominant architecture for sequence modeling, but there is growing interest in models that use a fixed-size latent state that does not depend on the sequence length, which we refer...
- [Deploying 🤗 Hub models in Vertex AI](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai): no description found
- [Quickstart &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): no description found
- [How to build a multi-label &amp; multi-class dataset correctly?](https://discuss.huggingface.co/t/how-to-build-a-multi-label-multi-class-dataset-correctly/76042): I am unsure how to proceed creating a Dataset with multiple labels and classes where the classes are not the same for the different labels.  A multi-label example is shared here, but the classes are a...
- [Deploying 🤗 Hub models in Vertex AI](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai#model-upload): no description found
- [GitHub - sshh12/multi_token: Embed arbitrary modalities (images, audio, documents, etc) into large language models.](https://github.com/sshh12/multi_token): Embed arbitrary modalities (images, audio, documents, etc) into large language models. - sshh12/multi_token

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1214933310757937172) (10 messages🔥): 

- **Background Project in Motion**: User `@singe.r` is working on converting **img2img** to create backgrounds for products and inquires if anyone has undertaken a similar project before.
- **FP8 Training Achievement Unlocked**: `@neuralink` shared their learning experience about accomplishing **55% end-to-end FP8** training from scratch, including developing the kernels.
- **Invitation to Rust Language Adventure**: `@manel_aloui` announced the start of their journey learning the **Rust programming language** and invites others to join.
- **Rust Enthusiasts Converge**: Following `@manel_aloui`'s call, `@cursorop` chimed in about their own experience learning Rust, specifically the **candle library** for machine learning, sparking a connection between the two Rust learners.
- **Soliciting Peer Learning for Stanford ML Course**: `@singhaditya4333`, also known as Aditya, is seeking companions to join in completing the **Stanford machine learning course**.

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1214881782764150814) (11 messages🔥): 

- **AI Enters Mainstream with a Bang**: `@vardhan0280` shared an [Investopedia article](https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp) highlighting that 2022 saw AI go mainstream, largely due to the popularity of OpenAI's [DALL-E](https://www.investopedia.com/openai-rolls-out-upgraded-ai-image-generation-tool-with-dall-e-3-7972607) and [ChatGPT](https://openai.com/blog/chatgpt).

- **Community Collaboration for Open Sora**: `@miko_al` encouraged spreading the word to support the [Open-Sora-Plan project](https://github.com/PKU-YuanGroup/Open-Sora-Plan) which aims to reproduce OpenAI's text-to-video model with limited resources.

- **Precision Health and AI in Space**: `@rtscott2001` shared a link to a Nature Machine Intelligence article titled "Biomonitoring and Precision Health in Deep Space Supported by Artificial Intelligence," available [here](http://rdcu.be/c8jSO).

- **Karpathy Shares Insights on Training LLMs**: `@.lawlord` posted an engaging Twitter thread by Andrej Karpathy addressing the complexities of training large language models (LLMs), a taxing process needing dedicated teams focused on cluster maintenance and fault tolerance. The thread concludes with a [link](https://twitter.com/AIatMeta/status/1539702714141011969) to the famous OPT-175B logbook for further insights.

- **Open Source Models with Hugging Face Course Launch**: `@andysingal` highlighted a new course offered by DeepLearning.AI focused on open source models using Hugging Face's tools, and the course can be viewed [here](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/1/introduction).

**Links mentioned**:

- [DLAI - Open Source Models with Hugging Face](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/1/introduction): Introduction · Selecting models · Natural Language Processing (NLP) · Translation and Summarization · Sentence Embeddings · Zero-Shot Audio Classification · Automatic Speech Recognition · Text to Spee...
- [Let&#39;s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): We build a Generatively Pretrained Transformer (GPT), following the paper &quot;Attention is All You Need&quot; and OpenAI&#39;s GPT-2 / GPT-3. We talk about connections t...
- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1765424847705047247?s=46): Nice read on the rarely-discussed-in-the-open difficulties of training LLMs. Mature companies have dedicated teams maintaining the clusters. At scale, clusters leave the realm of engineering and becom...
- [Artificial Intelligence (AI): What It Is and How It Is Used](https://www.investopedia.com/terms/a/artificial-intelligence-ai.asp): Artificial intelligence or AI refers to the simulation of human intelligence in machines that are programmed to think and act like humans.
- [Exploring TRLx: Hands-on Guide for Implementing Text Summarization through RLHF](https://www.labellerr.com/blog/exploring-trlx-hands-on-guide-for-implementing-text-summarization-through-reinforcement-learning-and-human-feedback/): Learn text summarization with TRLx. Fine-tune models, create reward feedback and apply PPO for effective learning.
- [GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.](https://github.com/PKU-YuanGroup/Open-Sora-Plan): This project aim to reproducing Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project. - PKU-YuanGroup/Open-Sora-Plan
- [leom0311 - Overview](https://github.com/leom0311): leom0311 has 9 repositories available. Follow their code on GitHub.
- [Biomonitoring and precision health in deep space supported by artificial intelligence | Nature Machine Intelligence](http://rdcu.be/c8jSO): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1214928688907554816) (18 messages🔥): 

- **New Model Yi-9B Debuts**: `@tonic_1` announced the release of **Yi-9B**, a new addition to the model collection, inviting users to check it out on [Hugging Face with demo available here](https://huggingface.co/spaces/Tonic/Yi-9B). They hinted at HuggingFace's future plans including leaderboards and gaming competitions.

- **Chatbot Arena Leaderboard by rwitz_**: `@rwitz_` shared his recent work, a ChatBot-Arena-Leaderboard for **Mistral Fine-tunes**, styled like the lmsys leaderboard and invited model contributions on its hosted space, found [here](https://huggingface.co/spaces/rwitz/Mistral-ChatBot-Arena).

- **UDOP Model Demonstration**: `@shinjeki` provided a simple demo space for playing around with Microsoft's latest document AI model, **UDOP**, which can be explored [here](https://huggingface.co/spaces/RamAnanth1/udop-vqa).

- **ComfyUI Workflow Unveiled**: `@alx.ai` announced the release and open-sourcing of new workflow and nodes for comfyUI, specifically for creating parallax motion in UIs, details of which can be followed through a tweet posted [here](https://x.com/TheHeroShep/status/1765525023115350114?s=20). 

- **AI-Enhanced Educational Dataset**: `@locutusque` introduced the UltraTextbooks v2, a large NLP dataset designed for training language models in education, which includes textbooks on various academic subjects, now available [on Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks-2.0).

**Links mentioned**:

- [Tweet from undefined](https://x.com/The): no description found
- [Andyrasika/Gemma-ChatML · Hugging Face](https://huggingface.co/Andyrasika/Gemma-ChatML): no description found
- [Tweet from TheHeroShep (@TheHeroShep)](https://x.com/TheHeroShep/status/1765525023115350114?s=20): 🧂Excited to share the first (of many) @getsaltai workflow & node release  • @comfyUI workflow to generate controlled 3D parallax motion with a single prompt or input image • Possibly useful  for gene...
- [ETHDenver Recap: Emerging Trends in web3 and AI](https://www.spatialawareness.net/p/ethdenver-recap-emerging-trends-in?utm_source=activity_item): Where we&#x27;re at, where we&#x27;re heading, and the return of Kevin.
- [Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101 · Hugging Face](https://huggingface.co/Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101): no description found
- [Locutusque/UltraTextbooks-2.0 · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks-2.0): no description found
- [UDOP DocVQA - a Hugging Face Space by RamAnanth1](https://huggingface.co/spaces/RamAnanth1/udop-vqa): no description found
- [Testing - a Hugging Face Space by rwitz](https://huggingface.co/spaces/rwitz/Mistral-ChatBot-Arena): no description found
- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz): no description found
- [Langchain Crash Course (Gradio) - a Hugging Face Space by chongdashu](https://huggingface.co/spaces/chongdashu/langchain-crash-course-gradio): no description found
- [GitHub - chongdashu/langchain-crash-course at lesson-1](https://github.com/chongdashu/langchain-crash-course/tree/lesson-1): Contribute to chongdashu/langchain-crash-course development by creating an account on GitHub.
- [Yi 9B - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Yi-9B): no description found
- [Building a Datacomp CLIP index with Fondant - Fondant](https://fondant.ai/en/latest/blog/2024/03/05/building-a-datacomp-clip-index-with-fondant/#with-fondant).): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1214946254413500476) (9 messages🔥): 

- **Weekend Session Consideration**: `@shafi8433` expresses a preference for weekend sessions because the current timing coincides with their working hours.
- **Timezone Coordination**: In response to `@lunarflu`'s inquiry, `@shafi8433` mentions being in the **IST** timezone.
- **Deciphering Big Data Technology**: `@ibrahim_72765_43784` shares a link for a **comprehensive exploration** of the Big Data Technology Ecosystem posted on Kaggle.
- **End-to-End Chatbot Using Llama2 Inquiry**: `@neerajjulka1986` seeks suggestions for implementing an end-to-end chatbot project using an open-source model, needed for understanding fine-tuning, deployment, and monitoring.
- **Advice on Fine-Tuning and Deployment**: `@chad_in_the_house` responds to `@neerajjulka1986`, recommending **Hugging Face's PEFT** for fine-tuning and their **text-generation-inference** GitHub repository for deployment, but advises careful consideration of compute resources before fine-tuning.

**Links mentioned**:

- [Deciphering the Big Data Technology Ecosystem: A Comprehensive Exploration &#x1F310; | Kaggle](https://www.kaggle.com/discussions/accomplishments/482230#2684943): Deciphering the Big Data Technology Ecosystem: A Comprehensive Exploration &#x1F310;.
- [GitHub - huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.](https://github.com/huggingface/peft): 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft
- [GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference): Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1214944296361263154) (2 messages): 

- **Slow Down, Power User!**: HuggingMod gently reminded `@user` to temper their enthusiasm and reduce the frequency of their messages with a friendly nudge to *slow down a bit* 🤗.
- **Quest for SDXL-Lightning LoRA Knowledge**: `@happy.j` asked for assistance on how to integrate SDXL-Lightning LoRA with a standard sdxl model, linking to a [discussion post](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104) that didn't fully resolve their issue. 
- **Tips from ByteDance for SDXL Magic**: The ByteDance organization offered advice on merging SDXL-Lightning LoRA with a trained SDXL model. They proposed starting with a traditional SDXL model and adding LoRA for acceleration, suggesting advanced techniques such as merging then using adversarial objectives for those seeking a challenge.

**Links mentioned**:

[ByteDance/SDXL-Lightning · finetune](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1214883747216891996) (7 messages): 

- **Labeling and Splitting Data Tip**: `@huzuni` suggests it's good to label and split data if one does not mind making it public.
- **User-Friendly Interface for Segmentation**: `@huzuni` finds the current interface friendlier than most SAM plugins, especially for **segmentation and bounding box labeling**.
- **Slow Down to HuggingMod's Pace**: `@HuggingMod` reminded **<@715715500470042706>** to slow down their posting frequency.
- **Questioning the Impact of Normalization**: `@huzuni` wonders about the effects of normalization in their data as they observed **no significant changes** with methods like **imagenet norm, channel wise norm, and min-max norm**.
- **Looking for Ultralytics Alternatives**: `@prod.dopamine` is seeking a good alternative to **ultralytics**, expressing dissatisfaction with the AGPL license.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1214931031203905626) (23 messages🔥): 

- **MMLU Dataset Inquiry**: `@privetin` expressed interest in the structure and content of the **MMLU datasets** but no further details or discussions followed the inquiry.

- **Slow Down, You're Posting Too Fast!**: `@HuggingMod` reminded `@715715500470042706` to moderate the pace of their postings to prevent spamming the channel.

- **Tokenization Troubleshoot**: `@mbotta` encountered issues with tokenizing prompts for the **OpenHermes-2.5** model, which lacks a 'tokenizer.json' file. Through the conversation with `@cursorop`, it was clarified that for a fine-tuned model like OpenHermes, one should use the tokenizer of the base model, which in this case is **Mistral**.

- **Seeking the Right Model for Job Title Normalization**: `@deb0rian` asked for advice on a base model for fine-tuning that can predict normalized job titles, disciplines, and seniority from user input. `@lucnzz` suggested an alternate approach using a basic retriever and generator model, while a subsequent exchange included a humorous GIF link response and dog-related pun by `@cakiki` and `@lucnzz`.

- **Model Recommendations for Colab**: `@iloveh8` inquired about the best small/medium open-source language model suitable for Google Colab, with responses from `@cursorop` and `@lucnzz` offering suggestions including a 2b model or **flan T5**, and any small quantized model respectively.

**Links mentioned**:

[Golden Retriever Dog GIF - Golden Retriever Dog Puppy - Discover &amp; Share GIFs](https://tenor.com/view/golden-retriever-dog-puppy-gif-26065357): Click to view the GIF

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1214944296361263154) (2 messages): 

- **Slow Your Roll, Poster**: `@HuggingMod` cautioned user `@715715500470042706` to reduce the frequency of their postings in the server, emphasizing the importance of pacing in the chat.

- **Seeking Guidance on SDXL-Lightning LoRA Merge**: User `@happy.j` is looking for assistance on combining SDXL-Lightning LoRA with a standard SDXL model and shared a [discussion link](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104) seeking answers on fine-tuning or creating their own version. The link includes suggestions such as training a regular SDXL model and then applying LoRA for acceleration or merging SDXL-Lightning LoRA before further training, with the latter including advanced techniques involving MSE loss and adversarial objectives.

**Links mentioned**:

[ByteDance/SDXL-Lightning · finetune](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104): no description found

  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1214906815624642580) (1 messages): 

```html
<ul>
  <li><strong>Gradio 4.20.0 Unleashed with External Auth Providers</strong>: User <code>@yuviii_</code> announced Gradio's newest version 4.20.0 which supports <strong>external / arbitrary authentication providers</strong>, including HF OAuth and Google OAuth, enhancing app security and user flexibility. Check out the examples on HF Spaces - [HF OAuth Example](https://huggingface.co/spaces/Wauplin/gradio-oauth-private-models) and [Google OAuth Example](https://huggingface.co/spaces/gradio/oauth-example).</li>
  <li><strong>Clean Up With Ease</strong>: The latest Gradio update introduces a <code>delete_cache</code> parameter to <code>gr.Blocks</code>, allowing for automatic cleanup of files upon app shutdown.</li>
  <li><strong>Smooth User Logout Experience</strong>: Users can now enjoy a smoother sign-off with Gradio's new <code>/logout</code> functionality.</li>
  <li><strong>Stylish Downloads with Gradio</strong>: The <code>gr.DownloadButton</code> component is now available, making the provision of downloadable content in apps easier and more visually appealing. For more information, visit the [documentation for gr.DownloadButton](https://www.gradio.app/docs/downloadbutton#demos).</li>
</ul>
```

**Links mentioned**:

[Gradio DownloadButton Docs](https://www.gradio.app/docs/downloadbutton#demos): no description found

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1214926846664253533) (44 messages🔥): 

- **Deepspeed Draws Ire**: `@duh_kola` raises an issue with Deepspeed, contributing to the frustration expressed by `@leoandlibe`, who says, *Deepspeed sucks ass in axo*, especially when handling multiple GPUs like his 4x 4090s, as it can't split the base model across them, only the Lora adapter.
- **Gemma Falters Compared to Mistral**: `@noobmaster29` checks in after a hiatus to inquire about Gemma's performance compared to 7B Mistral. `@lee0099` responds with the community's view that Gemma underperforms, specifically with multi-turn dialogues, and `@le_mess` clarifies it's only trained in English, dismissing its utility for multilingual tasks.
- **GaLore Promises Memory Efficiencies**: The GaLore optimizer is highlighted in a tweet shared by `@noobmaster29`, which promises significant reductions in memory requirements for Large Language Model (LLM) training. While `@lee0099` criticizes the lack of detailed performance data, `@nafnlaus00` and others discuss its potential for improving accessibility of LLM training on consumer-grade hardware.
- **GaLore Integration in Axolotl WIP**: As GaLore garners attention, `@yamashi` is ready to contribute, offering to make a pull request later and urging others to test, while `@caseus_` provides updates and bug fixes, sharing related YAML in search of further assistance.
- **Technical Clarifications and Calls for Assistance**: `@tank02.` seeks guidance on which version of CUDA-supported torch to install, with `@nanobitz` suggesting the newer version is typically better. `@nanobitz` also indicates the existence of a configuration for training Gemma with Axolotl when `@noobmaster29` inquires about it.

**Links mentioned**:

- [Tweet from Prof. Anima Anandkumar (@AnimaAnandkumar)](https://x.com/AnimaAnandkumar/status/1765613815146893348?s=20): For the first time, we show that the Llama 7B LLM can be trained on a single consumer-grade GPU (RTX 4090) with only 24GB memory. This represents more than 82.5% reduction in memory for storing optimi...
- [oaaic](https://wandb.ai/oaaic/galore/runs/xf34fi0z/files/axolotl_config_ib1po0hq.yml): Weights & Biases, developer tools for machine learning
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/galore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [WIP: galore optimizer by maximegmd · Pull Request #1370 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370): Adds support for Galore optimizers Still a WIP, untested.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1214882208096194611) (45 messages🔥): 

- **16bit LoRA's Limited Appeal?**: `@suikamelon` questioned who would use the recently supported 16bit LoRA, referencing a [commit in axolotl's GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb2c90cbd915273f21cf3bff3b216f00303a0). `@caseus_` pointed out that there's a quantized DoRA PR still in process and they will remove a check once merged.

- **Memory Quandary in LoftQ**: Issues with LoftQ's memory usage and incorrect initialization documentation were identified by `@suikamelon` with discussions found in [GitHub Issue #1525](https://github.com/huggingface/peft/issues/1525#issuecomment-1976872543) and [Pull Request #1532](https://github.com/huggingface/peft/pull/1532) on Hugging Face's PEFT repository.

- **Excitement and Skepticism Over GaLore**: `@suikamelon` shared a [new training strategy called GaLore](https://arxiv.org/abs/2403.03507), along with corresponding [GitHub code](https://github.com/jiaweizzhao/GaLore) and [a tweet from Anima Anandkumar](https://twitter.com/AnimaAnandkumar/status/1765613815146893348). The discussion involved considering GaLore's potential memory usage benefits, with `@caseus_` questioning the claimed performance equivalence to full pretraining.

- **Integration Challenges with GaLore**: Some users, including `@stoicbatman` and `@caseus_`, discussed trying to integrate GaLore into their projects but encountered issues with getting the training started, as reflected by a [Pull Request on GaLore](https://github.com/jiaweizzhao/GaLore/pull/5).

- **Baited by Methods on Multiple Occasions?**: A recurring sentiment of possibly being misled by different efficiency methods arose with `@yamashi`, `@nruaif`, and others, with mentions of ReLoRA, NEFT, and potentially a third unidentified method. The conversation veered into questions about proper dataset sizes and settings for effective finetuning and pretraining.

**Links mentioned**:

- [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch): Low-rank adaptation (LoRA) is a machine learning technique that modifies a pretrained model (for example, an LLM or vision transformer) to better suit a specific, often smaller, dataset by adjusting o...
- [LoftQ does not seem to quantify the base model · Issue #1525 · huggingface/peft](https://github.com/huggingface/peft/issues/1525#issuecomment-1976872543): System Info transformers version: 4.37.2 Platform: Ubuntu 18.04.6 LTS GPU: RTX GeForce 3090 x 2 Python version: 3.10.13 Huggingface_hub version: 0.20.3 Safetensors version: 0.4.2 Accelerate version...
- [Is it possible to use qlora with relora? · Issue #5 · Guitaricet/relora](https://github.com/Guitaricet/relora/issues/5): no description found
- [GitHub - euclaise/SlimTrainer: Full finetuning of large language models without large memory requirements](https://github.com/euclaise/SlimTrainer): Full finetuning of large language models without large memory requirements - euclaise/SlimTrainer
- [peft/examples/loftq_finetuning at main · huggingface/peft](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning): 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft
- [be a bit more lenient on transformers version by winglian · Pull Request #5 · jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore/pull/5): Hi there! Amazing research on this. We&#39;re looking to integrate galore into the axolotl project here OpenAccess-AI-Collective/axolotl#1370 One issue I ran into is the transformers dependency pin is...
- [support for DoRA w/ PEFT (#1363) · OpenAccess-AI-Collective/axolotl@0cfdb2c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb2c90cbd915273f21cf3bff3b216f00303a0): no description found
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507): Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-ran...
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [support for DoRA w/ PEFT (#1363) · OpenAccess-AI-Collective/axolotl@0cfdb2c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/0cfdb): no description found
- [WIP Fix LoftQ docs and tests by BenjaminBossan · Pull Request #1532 · huggingface/peft](https://github.com/huggingface/peft/pull/1532): Relates to #1525 Don&#39;t merge this, some GPU tests are failing Unfortunately, the docs I wrote about how to use LoftQ were incorrect, based on a misunderstanding I had. In reality, it is quite a bi...
- [jiawe - Overview](https://github.com/jiawe): Victory LOVES Preparation! jiawe has 47 repositories available. Follow their code on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1214963753620086846) (20 messages🔥): 

- **DeepSpeed Zero3 Configuration Check**: `@caseus_` queried if `stage3_gather_16bit_weights_on_model_save` was set to true in the DeepSpeed JSON. `@seungduk` confirmed its presence in the config, providing a visual from the [axolotl GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12) for reference.

- **Python Package Dependency Hell**: `@tank02.` faced a dependency conflict issue while installing `axolotl[deepspeed]==0.4.0`, which had conflicting versions of `torch` required by other dependencies. Solutions offered included manual installation of specific dependencies and the use of version `torch==2.2.0` based on advice from `@remek1972`.

- **Manual Installation Tactic to Resolve Module Version Conflict**: `@rtyax` and `@remek1972` both recommended manually installing conflicting dependencies to overcome version clashes, with `@remek1972` specifically citing successful installation after adjusting PyTorch and xformers versions.

- **Masking Mechanism Clarification for Training**: `@suikamelon` sought to understand how masked tokens are treated during training, leading to an exchange where `@nanobitz` clarified that `axolotl` sets the labels for masked tokens to -100, indicating they are not ignored but excluded from the loss calculation.

**Links mentioned**:

- [Dependency Resolution - pip documentation v24.1.dev0](https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts```): no description found
- [axolotl/deepspeed_configs/zero3_bf16.json at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json#L12): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1215029412794335262) (2 messages): 

- **Tweet on Group Chatting with Claude 3**: `@alexatallah` shared a Twitter post about a positive experience with **group chatting** using **Claude 3**, which was self-moderated. The story is available on [OpenRouterAI's Twitter](https://twitter.com/OpenRouterAI/status/1765470591836959061).

- **"Nitro" Models in Testing**: `@alexatallah` informed users of the appearance of new "nitro" models, which are safe to use and build with. Users were advised that slight changes might occur until an official announcement is made as they are incorporating feedback from early testers.
  

---


### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1214930144142037063) (94 messages🔥🔥): 

- **Sponsorship Offer for VSCode Extension Builders**: `@alexatallah` extended an offer to **sponsor anyone willing to build a VSCode extension** compatible with OpenRouter with free credits.
- **Community Discusses VSCode Extensions for LLMs**: Community members shared various VSCode extensions for coding assistance with LLMs like OpenRouter and GPT-4, including alternatives such as [Cursor](https://cursor.sh/), [Continue](https://continue.dev), and [Tabby](https://tabby.tabbyml.com/).
- **Inefficiency with Long Documents on OpenRouter Chat**: `@aliarmani` experienced issues with long document processing on OpenRouter chat inference and received recommendations for alternatives such as Typingmind and ChatbotUI.
- **Claude 3 Opus Conversations Engaging But Impact Wallets**: Users like `@phoshnk` and `@billbear` discussed the engaging nature of conversations with Claude 3 Opus, while others like `@xiaoqianwx` lamented its cost; `@filth2` highlighted Sonnet's cost-effectiveness.
- **Moderation Layers on OpenRouter Explained**: Community members explained the moderation layers applied to the models on OpenRouter, with OpenAI and Anthropic models receiving additional moderation compared to self-moderated beta models.

**Links mentioned**:

- [Continue](https://continue.dev): no description found
- [Home | Tabby](https://tabby.tabbyml.com/): Description will go into a meta tag in &lt;head /&gt;
- [Configuration | Continue](https://continue.dev/docs/model-setup/configuration#defining-a-custom-llm-provider): Configure your LLM and model provider
- [Perplexity: Sonar 8x7B by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-medium-chat?tab=parameters): Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier models in cost-efficiency, speed, and performance.  The version of this model with Internet access is [Sonar 8x7B Online](/mo...
- [GitHub - continuedev/continue: ⏩ The easiest way to code with any LLM—Continue is an open-source autopilot for VS Code and JetBrains](https://github.com/continuedev/continue): ⏩ The easiest way to code with any LLM—Continue is an open-source autopilot for VS Code and JetBrains - continuedev/continue

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1214907234656583681) (55 messages🔥🔥): 

- **Handling CSV Loader Timeout Errors**: `@kinghaze443` is seeking help for the error "The write operation timed out" which occurs while loading a CSV using `UnstructuredCSVLoader` from LangChain. They provided a snippet of code and mentioned their current LangChain and OpenAI versions. No solution was proposed or discussed in the provided messages.
- **Concerns Over Phishing Attempts on Discord**: `@archiacme` reported an increase in server members sharing suspicious steamcommunity links, suggesting these could be phishing attempts, and inquired about removing them. There was no follow-up or resolution mentioned in the message history.
- **Clarity on Query Scaling and Large Dataset Handling**: `@dbounds` inquired about the right approach for using `retrieval-augmented generation` chains with large datasets, concerned with examples that serialize an entire database into a string for context. The concern highlighted the impracticality of this method for large data sets, but no specific solution was given in the messages provided.
- **Lack of Documentation and Assistance on Azure AI Search**: `@juroy` is looking for information on setting up chains such as `RetrievalQA` using Azure AI Search with LangChain and cannot find it in the documentation. Several responses acknowledge the lack of help and difficulty in finding solutions, emphasizing the novelty of the technology and community dynamics, yet no direct solution to `@juroy`'s initial problem was provided.
- **LangChain Streaming and Prompt Viewing**: `@cybersmiths` mentions implementing streaming in Python, while `@yd4224` asks how to view the complete prompt text including `chat_history` and `agent_scratchpad`, hoping to see the actual string sent to the LLM model. A callback solution provided by `@chester3637` with code snippets and a mention of Langsmith might serve as a starting point for resolving both inquiries, focusing on the visibility and performance of the LangChain interaction.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [LangChain Expression Language (LCEL) | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/): LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.
- [Google Colaboratory](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/question_answering/chat_history.ipynb): no description found
- [Google Colaboratory](https://colab.research.google.com/github/langchain-ai/langgraph/blob/main/examples/code_assistant/langgraph_code_assistant.ipynb#scrollTo=744f48a5-9ad3-4342-899f-7dd4266a9a15): no description found
- [LangSmith](https://smith.langchain.com/public/ea1f6ca5-de52-4d36-bd7b-fde3faa74a70/d?paginationState=%7B%22pageIndex%22%3A0%2C%22pageSize%22%3A10%7D&chartedColumn=latency_p50): no description found
- [Any updates on Assistant API Streaming?](https://community.openai.com/t/any-updates-on-assistant-api-streaming/551809/1): Building a web app using assistants api. Lack of streaming is seriously hurting the UI and making me consider just going another route until streaming is available.  Has anyone heard anything about wh...
- [LangSmith](https://www.langchain.com/langsmith): Get your LLM app from prototype to production.
- [Retrieval augmented generation (RAG) | 🦜️🔗 Langchain](https://js.langchain.com/docs/expression_language/cookbook/retrieval): Let&#x27;s now look at adding in a retrieval step to a prompt and an LLM, which adds up to a &quot;retrieval-augmented generation&quot; chain:
- [Azure AI Search | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/vectorstores/azuresearch#configure-vector-store-settings): [Azure AI
- [langchain/libs/community/langchain_community/document_loaders/parsers/pdf.py at v0.1.11 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/v0.1.11/libs/community/langchain_community/document_loaders/parsers/pdf.py#L97): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Extract Text from a PDF &mdash; pypdf 4.0.1 documentation](https://pypdf2.readthedocs.io/en/stable/user/extract-text.html): no description found
- [langchain_agent/assistant at master · couthyapper7/langchain_agent](https://github.com/couthyapper7/langchain_agent/tree/master/assistant): a csv reader made in langchain with a fine tuned gpt - couthyapper7/langchain_agent

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1215349964125769758) (9 messages🔥): 

- **User Profile Construction via Pydantic and LangChain**: `@justanothergraphguy` is working on a chat chain to build user profiles, using **Pydantic** for structured output and **LangChain** with **Redis** for chat history. They shared the `UserProfile` and `Result` Pydantic models to structure the user data.
- **System Prompt for Interactive User Profile Creation**: A detailed **system prompt** was provided which guides the AI on how to interact with users to build their profiles by extracting information, asking follow-up questions, and confirming the final details.
- **Integration Woes in Chain Construction**: `@justanothergraphguy` discussed an issue with a **chat chain** where the `HumanMessage` is incorrectly included in the `AIMessage` content after the first interaction, despite the intention for memory propagation.
- **An Example Shows Unexpected Results**: An example code interaction was shared where **Redis** stores the chat history, but when a new message is introduced, the **AIMessage** includes the prior `HumanMessage` content, indicating a possible issue with the chat history handling.
- **Seeking the Community's Insights**: `@justanothergraphguy` is looking for community input on the issue with the chat chain, specifically regarding the erroneous inclusion of `HumanMessage` in subsequent `AIMessage` outputs. They provided the initial example code snippet to illustrate the problem.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1214934208582254603) (7 messages): 

- **RAG meets RAPTOR**: `@andysingal` shared a Medium article about building a Long Context Retriever-Aggregator (RAG) from scratch using RAPTOR and Langchain. The [write-up](https://medium.com/ai-advances/building-long-context-rag-from-scratch-with-raptor-using-langchain-c6491f1ba141) details adapting to evolving knowledge domains and overcoming traditional knowledge retrieval shortcomings.

- **ChromaDB Plugin Release**: `@vic49.` introduced the [ChromaDB Plugin for LM Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases), a solution for creating a ChromaDB vector database to work with LM Studio in server mode.

- **Architectural and Medical Generative Scripts**: `@_johnny1984` is working on a similar concept to the ChromaDB plugin and shared a script directory showcasing that they are attempting to build new hospitals through generative algorithms, covering both architectural and medical professional roles.

- **Integrate LLMs Seamlessly into Python Projects**: `@madgic_` announced a new library using langchain called `ask-llm`, which allows for the easy integration of LLM interactions into Python projects, inspired by langchain-decorators. The library uses Jinja templating for prompts and is described in detail on [GitHub](https://github.com/FlorianMgs/ask-llm).

- **Exploring Vision Models in Production**: `@vru.shank` announced a workshop with MultiOn and Quizizz focusing on the application of vision models in production, inviting interested individuals to RSVP for the workshop being held by the LLMs in Prod community. The details and registration can be accessed through this [link](https://lu.ma/multimodal-llms).

**Links mentioned**:

- [Releases · BBC-Esq/ChromaDB-Plugin-for-LM-Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases): Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode! - BBC-Esq/ChromaDB-Plugin-for-LM-Studio
- [GitHub - FlorianMgs/ask-llm: The easiest way to supercharge your apps with LLM!](https://github.com/FlorianMgs/ask-llm): The easiest way to supercharge your apps with LLM! - FlorianMgs/ask-llm
- [Building Long Context RAG from Scratch with RAPTOR using Langchain](https://medium.com/ai-advances/building-long-context-rag-from-scratch-with-raptor-using-langchain-c6491f1ba141): Ankush k Singal
- [no title found](https://quizizz.com)): no description found
- [Multi-Modal LLMs in Prod | Practitioners&#x27; Workshop · Luma](https://lu.ma/multimodal-llms): The LLMs in Prod community is hosting practitioners from top Gen AI companies to talk about how they are using multi-modal models (vision, audio, image gen, etc.) in...

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1214975657591709706) (2 messages): 

- **Crafting Infinite Fun with AI**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=QPZpOBxUd1U) titled "Infinite Craft Game using Mistral", showcasing the development of a game where players start with four elements and combine them to discover new ones using **Mistral**.

- **Meme Generation with Mistral & Giphy**: `@pradeep1148` also posted a [YouTube video](https://www.youtube.com/watch?v=PtP8R8VjTGc) titled "Making memes with Mistral & Giphy", demonstrating the use of **Mistral** and Giphy API to create memes, and provided a link to the related [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb).

**Links mentioned**:

- [Making memes with Mistral &amp; Giphy](https://www.youtube.com/watch?v=PtP8R8VjTGc): Lets make memes using mistral llm and Giphy api#llm #ml #python #pythonprogramming https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb
- [Infinite Craft Game using Mistral](https://www.youtube.com/watch?v=QPZpOBxUd1U): Let develop Neal Agarwal’s web game Infinite Craft. This is a “crafting game” where you start with just four elements and repeatedly combine pairs of element...

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1215093773982240879) (6 messages): 

- **Inquiring About Channel Purpose**: `@8bitelon` questioned if "general" meant off-topic, to which `@marksaroufim` responded that there's no particular off-topic channel but one might be created if needed.
- **Meme Humor Tolerance**: `@iron_bound` expressed a desire to post memes, eliciting a response from `@marksaroufim` asking to see the best memes in a specific memes channel.
- **Flash Attention CUDA Project by tspeterkim_89106**: `@tspeterkim_89106` shared a project implementing **Flash Attention** in CUDA, looking for feedback and discussions on the implementation. [The project is available on GitHub](https://github.com/tspeterkim/flash-attention-minimal).
- **Quick Guide to CUDA**: `@iron_bound` provided a concise educational resource with a link to a YouTube video titled "Nvidia CUDA in 100 Seconds". [Watch the video for a rapid overview of CUDA](https://www.youtube.com/watch?v=pPStdjuYzSI).

**Links mentioned**:

- [Nvidia CUDA in 100 Seconds](https://www.youtube.com/watch?v=pPStdjuYzSI): What is CUDA? And how does parallel computing on the GPU enable developers to unlock the full potential of AI? Learn the basics of Nvidia CUDA programming in...
- [GitHub - tspeterkim/flash-attention-minimal: Flash Attention in ~100 lines of CUDA (forward pass only)](https://github.com/tspeterkim/flash-attention-minimal): Flash Attention in ~100 lines of CUDA (forward pass only) - tspeterkim/flash-attention-minimal

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1214873751485947945) (30 messages🔥): 

- **Bandwidth Revelations with Nvidia H100**: `@iron_bound` discussed the L2 cache read bandwidth for Nvidia's H100 GPU, highlighting a 5.5 TB/s figure and proposing a method for calculating L1 cache bandwidth. They referenced an in-depth [article at Chips and Cheese](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/) that describes Nvidia's focus on the compute market with the H100 using the Hopper architecture.
  
- **Architectural Comparisons**: `@zippika` speculated based on available data that the Nvidia 4090's L1 cache bandwidth could be 40TB/s, assuming that the H100 has similar parameters, while `@iron_bound` pointed out that one must consider the differences between the Ada and Hopper architectures.

- **Unveiling Coarsening Effects on Performance**: `@marksaroufim` shared a [link to a tweet by @zeuxcg](https://x.com/zeuxcg/status/1765534285229064297?s=20) that discloses the proper way to handle coarsening in code execution, providing insights into performance misconceptions due to benchmarking inconsistencies.

- **Learning the CUDA CuTe DSL**: `@ericauld` started a discussion on understanding the CuTe Domain Specific Language (DSL), sharing a [GitHub link to the CUtlass library](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) where CuTe is used, and discussing the best order to read the documentation.

- **Dequantization Optimization in CUDA**: `@zippika` shared their progress on optimizing fp4 dequantization using the cuda::pipeline API, noting an improvement in speed over a baseline and stating success in accuracy when testing on real inputs and outputs.

**Links mentioned**:

- [Microbenchmarking Nvidia&#8217;s RTX 4090](https://chipsandcheese.com/2022/11/02/microbenchmarking-nvidias-rtx-4090/): Nvidia&#8217;s RTX 4090 features Nvidia&#8217;s newest architecture, named Ada Lovelace after a pioneer in early computing. Compared to their previous architecture, Ampere, Ada Lovelace enjoys a pr…
- [Tweet from Arseny Kapoulkine 🇺🇦 (@zeuxcg)](https://x.com/zeuxcg/status/1765534285229064297?s=20): As a demonstration, I changed coarsen source code to comment out the body of VecAdd/VecAddCoarsened code and changed the launch parameters to omit `2*` and I get these results.  What you&#39;re seeing...
- [Nvidia&#8217;s H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/): GPUs started out as devices meant purely for graphics rendering, but their highly parallel nature made them attractive for certain compute tasks too. As the GPU compute scene grew over the past cou…
- [cutlass/media/docs/cute at main · NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute): CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1214947730393403474) (11 messages🔥): 

- **bitsandbytes for Quantization Needs**: User `@iron_bound` suggested `@mabeto5p` to check out [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to perform linalg on low precision integers with PyTorch, which might utilize int8 tensor-cores on Ada architectures. `@mabeto5p` was looking for high abstraction tools to handle int4/int8 and fp8 matrices operations.

- **A Sync in Time Saves Nine**: `@andreaskoepf` pointed out a common mistake which could inflate benchmarks, implying `@mabeto5p` might need to add `torch.cuda.synchronize()` to get accurate measurements. `@mabeto5p` later affirmed moving away from Jupyter allowed for reasonable benchmarking results.

- **Cross-device Sync Clarified**: Addressing `@mabeto5p`'s query, `@andreaskoepf` clarified that `torch.cuda.synchronize()` calls `cudaDeviceSynchronize` internally, ensuring all kernels are finished before the call returns, irrespective of the origin, given they are from the same process. 

- **Mixed Device Tensor Indexing**: As per `@_t_vi_`, indexing a CPU tensor with a CUDA tensor scalar is allowed due to historical reasons and convenience, since scalars can be treated differently for operations like indexing.

- **Scalars Get VIP Treatment Across Devices**: `@_t_vi_` continued to explain that scalar tensors can be used to index other tensors regardless of device due to automatic conversions from Python numbers and CPU scalars to the device of the target tensor.

**Links mentioned**:

- [CUDA Runtime API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html): no description found
- [GitHub - TimDettmers/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch.](https://github.com/TimDettmers/bitsandbytes): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1215325325471256598) (1 messages): 

- **RelayAttention vs Ring/Flash Attention Inquiry**: `@lancerts` inquired about how **RelayAttention** compares with ring/flash attention, linking to the GitHub repository [vLLM with RelayAttention integration](https://github.com/rayleizhu/vllm-ra). The user has just begun reading the paper on RelayAttention.

**Links mentioned**:

[GitHub - rayleizhu/vllm-ra: vLLM with RelayAttention integration](https://github.com/rayleizhu/vllm-ra): vLLM with RelayAttention integration. Contribute to rayleizhu/vllm-ra development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1214889870334496789) (23 messages🔥): 

- **GPU Reset Tips from Command Line**: `@iron_bound` provided a potential solution to reset a GPU by suggesting the use of the command `sudo nvidia-smi --gpu-reset -i 0`. They also highlighted that in `nvtop`, pid 3874970 was observed to be running.
- **Pod Restart to Address GPU Issues**: `@andreaskoepf` mentioned restarting the pod in response to `@jamesmel`'s concern about memory allocation issues on the GPU not releasing and having no PID.
- **CUDA Runtime Error Conundrum**: `@iron_bound` encountered a CUDA runtime error that specified "head_size should be a multiple of 8" during the backward pass of `ring_flash_attn_varlen.py`.
- **Sampling Mechanism Clarification**: `@andreaskoepf` explained the token sampling mechanism during the forward pass of model training, indicating that only the last sampled token is used after the prompt, with an input tensor shaped (batch_size, 1).
- **Attention to Ring-attention and How-to Log Sum Exp**: `@andreaskoepf` created and shared a "how-to log sum exp" notebook for ring-attention experiments, and `@_t_vi_` expressed enthusiasm about the logsumexp trick, sharing a related project on [sinkhorn-kernel](https://lernapparat.de/sinkhorn-kernel).

**Links mentioned**:

- [iron-bound](https://wandb.ai/iron-bound/axolotl/runs/wjb8eyw3/overview?workspace=user-iron-bound): Weights & Biases, developer tools for machine learning
- [ring-attention/notebooks/howto_log_sum_exp.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/notebooks/howto_log_sum_exp.ipynb): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [GitHub - RulinShao/LightSeq: Official repository for LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers](https://github.com/RulinShao/LightSeq): Official repository for LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers - RulinShao/LightSeq
- [iron-bound](https://wandb.ai/iron-bound/axolotl/runs/7djmd1i2?workspace=user-iron-bound): Weights & Biases, developer tools for machine learning
- [Lernapparat - Machine Learning](https://lernapparat.de/sinkhorn-kernel): no description found

  

---


### CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1215398628928258098) (1 messages): 

- **Intriguing Mandelbrot Visualization Shared**: User `@apaz` shared an image link depicting a [Mandelbrot set](https://cdn.discordapp.com/attachments/1001261706762264709/1152786434420387922/mandelbrot2.jpg?ex=65f64f07&is=65e3da07&hm=eb2f8bf851ed742bc9d49fe9932f1d21f8c269ebbc681d1f65b75c6969c68081). The cryptographic string in the URL suggests enhanced security or unique identification for the image.
  

---



### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1215334683567456328) (4 messages): 

- **Inflection Boosts Pi with Emotional Intelligence**: `@xeophon.` highlighted the launch of **Inflection-2.5**, an upgraded AI model from Inflection AI, boasting competitive performance against major models like GPT-4 and Gemini. This new model, implemented in the empathetic AI **Pi**, is available across [iOS](https://apps.apple.com/us/app/pi-personal-ai-assistant/id6445815935), [Android](https://play.google.com/store/apps/details?id=ai.inflection.pi), and [desktop platforms](https://pi.ai/desktop).

- **Skepticism Over Inflection-2.5 Claims**: `@xeophon.` shared a [tweet](https://x.com/hlibivanov/status/1765754625364275267?s=46) by `@HlibIvanov` criticizing the efficacy of Inflection-2.5, suggesting it was simply a distillation of GPT-4 and questioning its innovation in AI modeling.

- **Nato Lambert Shares Excitement**: `@natolambert` expressed excitement in a [tweet](https://twitter.com/natolambert/status/1765779714252451971) about the rapid development and release of various models in less than a month, implying the pace of innovation in AI is accelerating.

**Links mentioned**:

- [Inflection-2.5: meet the world&#x27;s best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Tweet from Hlib Ivanov (e/acc) (@HlibIvanov)](https://x.com/hlibivanov/status/1765754625364275267?s=46): Of course it takes less flops to make a gpt-4 distill than training gpt-4 from scratch i&#39;m not even bothering with proper benchmark, this is clearly trained on unfiltered stuff from gpt-4, and it ...

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1215035592820264960) (48 messages🔥): 

- **Open-Source Software: A Salty Topic?**: `@natolambert` shared frustrations about the reactions to writing on open-source software, where even helpful posts often receive pedantic corrections and little welcome from the community.
- **Venting on Community Feedback**: While acknowledging that feedback is useful, `@natolambert` vented that the open-source software (OSS) community often lacks perspective and can deter those trying to promote OSS due to excessive criticism.
- **ML Policy Discussions Remain Private**: `@dangf91` and `@natolambert` discussed how the contentious nature of open-source and machine learning (ML) policy keeps many discussions out of the public eye. `@natolambert` pointed out that this is partly political, and one's stance often differs in public vs. private.
- **Critiques on License Talks**: `@natolambert` noted that while he is open about being engaged in politics around ML, criticism often focuses on minutia like incorrect terminology use.
- **Troubles in Clarifying “Open” Terminology**: A conversation between `@xeophon.` and `@natolambert` highlighted the difficulties and confusions in classifying models like Mistral or Llama2, with the industry incorrectly labeling proprietary models as "open-source".

**Links mentioned**:

[Aggregator&#8217;s AI Risk](https://stratechery.com/2024/aggregators-ai-risk/): A single AI can never make everyone happy, which is fundamentally threatening to the Aggregator business model; the solution is personalized AI

  

---


### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1215335559866617937) (19 messages🔥): 

- **Claude-3 Ranking Sparks Community Frenzy**: `@xeophon` shared a [link](https://x.com/lmsysorg/status/1765774296000172289?s=46) announcing the launch of **@Anthropic's Claude-3** ranking with impressive *20,000 votes* in just *three days*. Claude-3 variants Opus and Sonnet are creating a buzz, rivaling GPT-4-Turbo and closely matching GPT-4 in performance.
- **Anticipation for Anthropic's Gemini Ultra**: `@natolambert` expressed eagerness for the release of **Gemini Ultra**, while `@xeophon` is excited to access it and try out the *1M context window* feature, especially for analyzing multiple academic papers.
- **Simplicity in Screenshotting**: In response to `@xeophon` asking about the method to generate certain images, `@natolambert` explained using the *screenshot full window* feature on **Mac** with the tip to press space after `command-shift-4` for an effective tool useful in blogging and communications.

**Links mentioned**:

- [Tweet from lmsys.org (@lmsysorg)](https://x.com/lmsysorg/status/1765774296000172289?s=46): 🔥Exciting news from Arena  @Anthropic&#39;s Claude-3 Ranking is here!📈 Claude-3 has ignited immense community interest, propelling Arena to unprecedented traffic with over 20,000 votes in just three...
- [Tweet from Nathan Lambert (@natolambert)](https://x.com/natolambert/status/1765779714252451971?s=46>): I didn&#39;t expect commoditization at the top end but all of these in less than a month. We still have a week to get GPT5 and Llama 3 in this month post G1.5.  Gemini 1.5, Mistral Large, Claude 3, In...

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1214989627455438921) (1 messages): 

- **A Warm Welcome to Segmentationfault.**: `@segmentationfault.` expressed gratitude for being invited by `@748528982034612226` and showed eagerness to contribute to the field despite being new.
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1215246595348504617) (16 messages🔥): 

- **Early Bird Doesn't Always Get the Worm**: `@joshxt` emphasized the importance of not tagging everyone, especially at 5am, as it was considered spam and was promptly deleted. The unwanted alert led to a quick policy change, disabling the ability for users to ping everyone.
- **Curiosity Over a Mysterious Ping**: `@ikaridev` inquired if they were the recipient of an @everyone ping, only to learn from `@joshxt` that such a message did exist but was removed due to it being spam.
- **Dropping the @everyone Hammer**: Following a spam incident, `@joshxt` humorously confirmed to `@ikaridev` that users no longer have the ability to ping the entire server, hinting at a quick permission tweak.
- **Orca Dataset Splashes Into the Scene**: `@joshxt` sparked a discussion about the newly released Orca dataset by Microsoft, sharing a link to Hugging Face (`https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k`) and asking if anyone is doing something interesting with it.
- **Favorites in the AI Aquarium**: In light of the Orca dataset conversation, `@twistedshadows.` confessed a bias towards the model "Psyonic-cetacean" which integrates orca2 13b, while `@joshxt` admitted to currently favoring "Claude 3 Opus."

**Links mentioned**:

- [Angry Gary Oldman GIF - Angry Gary Oldman Everyone - Discover &amp; Share GIFs](https://tenor.com/view/angry-gary-oldman-everyone-gif-14317847): Click to view the GIF
- [microsoft/orca-math-word-problems-200k · Datasets at Hugging Face](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k): no description found

  

---


### Alignment Lab AI ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/1214884876793282561) (10 messages🔥): 

- **New Faces in the Chat**: `@jaxxks` offered a friendly evening greeting to `@1168088006553518183` who had welcomed them in earlier.
- **The Band Is Gathering**: Users such as `@tcapelle` and `@aslawliet` joined the channel with general greetings like "Hello every1!" and "Hello friends! 👋".
- **Brainstorming Session for Orca-2 Begins**: `@aslawliet` introduced the project concept for Orca-2, suggesting it target a broader range of datasets including FLAN 2021 and selective zero-shot (zs_opt) samples from T0 and Natural Instructions (niv).
- **Data Augmentation Strategy Discussed**: The idea of using Mixtral for data augmentation was proposed by `@aslawliet` as a cost and time-efficient alternative to GPT-4.
- **A Light Moment on AI Choices**: `@aslawliet` humorously doubted anyone's willingness to use Claude-3 Opus for the current project.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1214927409825054790) (9 messages🔥): 

- **Choices of Language Models for Various Constraints**: User `@johannhartmann` mentioned different models suitable under various constraints: **Claude Opus** and **GPT-4** with no constraints, **DiscoLM-120B** for open-source with extensive memory, and **VAGOsolutions/Sauerkraut LM-UNA-SOLAR-Instruct** when memory is limited.
- **Discussion on Retrieval-Augmented Language Models**: `@maxidl` shared an [arXiv paper](https://arxiv.org/abs/2403.03187) discussing the advantages of retrieval-augmented language models over traditional ones, noting the paper has a section on joint training of retriever and LLM but research on this topic is not extensive.
- **Recommendations for German-Speaking Models**: `@cybertimon` recommended **Nous Hermes 2 Mixtral 8x7b** for its fluent German capabilities, while `@johannhartmann` suggested exploring **DiscoResearch/DiscoLM_German_7b_v1** and other models such as **VAGOsolutions/SauerkrautLM-7b-HerO**, **mayflowergmbh/Brezn-7b**, or **seedboxai/KafkaLM-7B-DARE_TIES-LaserRMT-QLoRA-DPO-v0.5**.
- **Hermes Mixtral Praised for Accuracy**: `@flozi00` voiced their positive experience with **Nous Hermes 2 Mixtral 8x7b**, highlighting its accuracy in understanding tasks on the first attempt.
- **Comparison Inquiry for German Language Models**: `@johannhartmann` inquired if anyone had compared **Nous Hermes Mixtral** with other Mixtrals like **Sauerkraut** or **DiscoLM** for German prompts, indicating that to their knowledge, Hermes Mixtral had no German finetuning.

**Links mentioned**:

[Reliable, Adaptable, and Attributable Language Models with Retrieval](https://arxiv.org/abs/2403.03187): Parametric language models (LMs), which are trained on vast amounts of web data, exhibit remarkable flexibility and capability. However, they still face practical challenges such as hallucinations, di...

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1215035124744196159) (13 messages🔥): 

- **Exploring OPUS 100 Translation Quality**: `@flozi00` is working on labeling the OPUS 100 dataset to assess translation quality, finding less than half to be of good quality. They plan to develop embedding models to score translation quality by embedding distance, which could be beneficial for improving machine translation (MT) models and datasets.

- **Prompt Fine-tuning for Translation Categorization**: `@flozi00` employed the nous hermes mixtral dpo model, after substantial prompt fine-tuning, to categorize translation quality. This points toward the potential for automatic quality assessment in translation datasets.

- **Dataset Scrubbing for "Good" Translation Pairs**: `@crispstrobe` highlighted that OPUS 100, being randomly selected from a larger corpus, contains context-specific pairings that often fail outside of their intended setting. Creating subsets with universal "good" pairings is suggested for better utility in general contexts.

- **Improving the Automatic Evaluation of Translation Quality**: `@flozi00` mentions updating their model and dataset collection for better translation quality judgment. They also intend to iterate over multiple datasets to enhance their collection and welcome further suggestions for improvement.

- **mMARCO Dataset Gains Apache 2.0 License**: `@philipmay` notes that the mMARCO dataset has added an Apache 2.0 license and directs attention to the dataset's information on Hugging Face, despite a current lack of dataset viewer support for it.

**Links mentioned**:

- [Translation Data Quality - a flozi00 Collection](https://huggingface.co/collections/flozi00/translation-data-quality-65e9d0cdd977e1e0aed2de9d): no description found
- [unicamp-dl/mmarco · Datasets at Hugging Face](https://huggingface.co/datasets/unicamp-dl/mmarco#licensing-information): no description found
- [Data (Hint ID)](https://huggingface.co/data): no description found

  

---



### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1214997369712541726) (13 messages🔥): 

- **Opus the SAT Scholar**: `@jeffreyw128` shared that **Opus** achieved a perfect score of 800 on the SAT reading section, and linked to a Twitter post with the results ([View tweet](https://twitter.com/wangzjeff/status/176485068925)).
- **Concerns about Model Memorization**: `@dare.ai` raised a point following the SAT achievement regarding the challenges in creating true holdouts to avoid memorization in massive models.
- **Opus Garners Praise**: `@nosa_` expressed admiration for Opus, humorously threatening a confrontation if anyone tells Opus about the compliment.
- **Improved Opus Skill at Knowledge Webs**: `@res6969` tested Opus's ability to construct knowledge webs from a large document exceeding 35k tokens, remarking on the model's improved instruction following and contextual understanding.
- **Search Amongst 500 Names Proves Difficult**: `@jeffreyw128` reported on failing to find a specific name within 500, using different models including Claude Opus, indicating a typical struggle for AI models with such tasks.
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1214960981235470366) (1 messages): 

Since there is only one message provided without any specific discussion on topics, no summary can be generated. If more context or messages were provided, I'd be able to offer a summary with the requested details.
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1215370759464165386) (5 messages): 

- **GPT-4 Performance Under Scrutiny**: `@dbreunig` expressed surprise at **GPT-4's** failure in a specific, undisclosed test with no details provided on the nature of the test or the kind of failure encountered.
- **Clickable Bookshelves Spark Interest**: `@xnimrodx` shared a [blog post](https://jamesg.blog/2024/02/14/clickable-bookshelves/) about a script that transforms images of bookshelves into clickable regions, leading to Google Books pages for each book. The post includes a [demo](https://capjamesg.github.io/cv-book-svg/) and a video showing the functionality.
- **Library Efficiency Through Imaginative Tech**: `@xnimrodx` expressed a desire for an application similar to the *clickable bookshelves* to help his librarian-wife with shelf-reading tasks, noting the size of her library which is the *largest* among 35 schools in their diocesan system.
- **Community Library Project Idea**: `@dbreunig` showed interest in creating a toy app to help people register books in the small libraries around their town, indicating a practical application of the technology being discussed.

**Links mentioned**:

[Making my bookshelves clickable | James' Coffee Blog](https://jamesg.blog/2024/02/14/clickable-bookshelves/): no description found

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1215143442775023708) (8 messages🔥): 

- **Debugging Template Crashes**: `@trufuswashington` sought advice from the expert (`@746595581086138409`) due to issues with creating templates for common use cases; one template worked fine, while another kept crashing the `llm` command.
- **Error Output Shared**: They shared the error output, showing that the `llm` command failed due to a `TypeError` related to "expected str instance, NoneType found" during variable interpolation.
- **Working vs. Crashing Templates Compared**: `@trufuswashington` attached two YAML templates for comparison. The working one was for modifying text blocks and the faulty one was for providing brief explanations on various types of code-related content.
- **Cause of the Crash Found**: Eventually, `@trufuswashington` discovered the cause of the crash—it was due to the presence of a dollar sign `$` in the template prompt where it was not expected.
- **Dollar Sign Culprit**: Specifically, the error was triggered by the line "I will tip you $100 if you do this." in the crashing template, indicating issues with handling the special character in the prompt interpolation.
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1214975411226935477) (2 messages): 

- **Innovating Crafting Games with AI**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=QPZpOBxUd1U) titled "Infinite Craft Game using Mistral," showcasing a crafting game where elements are combined to create new items using the capabilities of the **Mistral language model**.

- **Automating Memes with Mistral**: `@pradeep1148` also posted a link to another [YouTube video](https://www.youtube.com/watch?v=PtP8R8VjTGc) titled "Making memes with Mistral & Giphy," which demonstrates the process of creating memes by integrating **Mistral's AI** with the **Giphy API** and includes a link to the GitHub repository containing the relevant notebook.

**Links mentioned**:

- [Infinite Craft Game using Mistral](https://www.youtube.com/watch?v=QPZpOBxUd1U): Let develop Neal Agarwal’s web game Infinite Craft. This is a “crafting game” where you start with just four elements and repeatedly combine pairs of element...
- [Making memes with Mistral &amp; Giphy](https://www.youtube.com/watch?v=PtP8R8VjTGc): Lets make memes using mistral llm and Giphy api#llm #ml #python #pythonprogramming https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb

  

---



---



---




---
id: 5193e590-b36c-4c34-a4f1-59f563eb578c
title: Mozilla's AI Second Act
date: '2024-06-27T01:37:35.020344Z'
original_slug: ainews-mozillas-ai-second-act
description: >-
  AI news highlights superfast CPU inference, Mozilla's comeback at AIE World's
  Fair with llama-file demos, new vector search project sqlite-vec, llama-agents
  launch, Claude updates with new UI features and projects, Etched's inference
  chip benchmarks, Sohu's chip performance, open-source models like Deepseek
  Coder v2 outperforming Gemini and GPT-4, and the PyTorch documentary.
companies:
  - mozilla
  - anthropic
  - etched
  - llamaindex
  - etched
  - sohu
  - deepseek
models:
  - claude-3-opus
  - gpt-4
  - gemini
  - llama-2
  - b200
  - billion-parameter-models
topics:
  - cpu-inference
  - vector-search
  - llama-file
  - llama-agents
  - ai-hardware
  - ai-benchmarks
  - open-source-ai
  - ai-model-performance
  - ai-research
  - ai-communities
---


<!-- buttondown-editor-mode: plaintext -->**Superfast CPU inference is all you need.**

> AI News for 6/25/2024-6/26/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**416** channels, and **3358** messages) for you. 
Estimated reading time saved (at 200wpm): **327 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The slow decline of [Mozilla's Firefox market share](https://en.wikipedia.org/wiki/Usage_share_of_web_browsers#/media/File:StatCounter-browser-ww-yearly-2009-2023.png) is well known, and after [multiple](https://arstechnica.com/information-technology/2020/08/firefox-maker-mozilla-lays-off-250-workers-says-covid-19-lowered-revenue/) [rounds](https://arstechnica.com/gadgets/2024/02/mozilla-lays-off-60-people-wants-to-build-ai-into-firefox/) of layoffs its future story was very uncertain. However at the opening keynote of the AIE World's Fair today they came back swinging:

 ![image.png](https://assets.buttondown.email/images/e1440ead-35c7-4eed-a606-83f053b95424.png?w=960&fit=max) 

Very detailed [live demos of llamafile](https://x.com/thedataroom/status/1806018145926455661) with technical explanation from Justine Tunney herself, and [Stephen Hood announcing](https://x.com/aiDotEngineer/status/1806072610683576368) a very welcome second project `sqlite-vec` that, you guessed it, adds vector search to sqlite.

You can watch the entire talk on the livestream (53mins in):

https://www.youtube.com/watch?v=5zE2sMka620&t=262s


LlamaIndex also closed the day with a [notable launch of llama-agents](https://x.com/llama_index/status/1806116419995844947)

 ![image.png](https://assets.buttondown.email/images/74816a0f-0cc4-4ca3-934d-ee691bbfa2f1.png?w=960&fit=max) 



Some mea culpas: yesterday we missed calling out [Etched's big launch](https://x.com/Etched/status/1805625693113663834) ([questioned](https://x.com/cHHillee/status/1805696613480022238?utm_source=ainews&utm_medium=email)), and [Claude Projects](https://www.anthropic.com/news/projects) made a splash. The [PyTorch documentary](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s) launched to crickets (weird?).

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

**Anthropic Claude Updates**

- **New UI features**: [@alexalbert__](https://twitter.com/alexalbert__/status/1805617407375065498) noted new features in the Claude UI, including a **sidebar** for starring chats, **shareable projects** with 200K context windows for documents and files, and **custom instructions** to tailor responses.
- **Anthropic announces Projects**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1805616725733339199) introduced Projects, which allow organizing chats into shareable knowledge bases with a 200K context window for relevant documents, code, and files. Available for Claude Pro and Team users.

**Hardware and Performance Benchmarks**

- **Etched AI specialized inference chip**: [@cHHillee](https://twitter.com/cHHillee/status/1805696613480022238) shared thoughts on Etched's new inference chip, noting potential **misleading marketing claims** around silicon efficiency and performance. Benchmarks claim **500k tokens/sec** (for multiple users) and replacing **160 H100s with one 8x Sohu server**, but may not be normalized for key details. More info needed on benchmark methodology.
- **Sohu chip enables 15 agent trajectories/sec**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1805636415772147839) highlighted that 500k tokens/sec on Sohu translates to **15 full 30k token agent trajectories per second**, emphasizing the importance of building with this compute assumption to avoid being scooped.
- **Theoretical GPU inference limits**: [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1805701944746590549) shared a model estimating **theoretical max of ~300k tokens/sec** for 8xB200 NVLink 8-bit inference on 70B Llama, assuming perfect implementations like OpenAI/Anthropic. Suggests Etched benchmarks seem low.

**Open Source Models**

- **Deepseek Coder v2 beats Gemini**: [@bindureddy](https://twitter.com/bindureddy/status/1805686571108384990) claimed an open-source model beats the latest Gemini on reasoning and code, with more details on open-source progress coming soon. A [follow-up](https://twitter.com/bindureddy/status/1805747650823962795) provided specifics - Deepseek Coder v2 excels at coding and reasoning, beating GPT-4 variants on math and putting open-source in 3rd behind Anthropic and OpenAI on real-world production use cases.
- **Sonnet overpowers GPT-4**: [@bindureddy](https://twitter.com/bindureddy/status/1805661535597211832) shared that Anthropic's Sonnet model continues to overpower GPT-4 variants in testing across workloads, giving a flavor of impressive upcoming models.

**Biological AI Breakthroughs**

- **ESM3 simulates evolution to generate proteins**: [@ylecun](https://twitter.com/ylecun/status/1805581310548697360) shared news of Evolutionary Scale AI, a startup using a 98B parameter LLM called ESM3 to "program biology". ESM3 simulated 500M years of evolution to generate a novel fluorescent protein. The [blog post](https://twitter.com/ylecun/status/1805634811773571496) has more details. ESM3 was developed by former Meta AI researchers.

**Emerging AI Trends and Takes**

- **Data abundance is key to AI progress**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1805669049948327995) emphasized that breaking through the "data wall" will require innovation in data abundance. AI models compress their training data, so continuing current progress will depend on new data, not just algorithms.
- **Returns on human intelligence post-AGI**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1805669260745916913) predicted that the premium on human genius will increase rather than decrease after AGI, as only the smartest humans will understand what AGIs are doing. 
- **Terminology for multimodal AI**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1805690638647935016) noted it's becoming weird to call multimodal AI "LLMs" and solicited suggestions for replacement terminology as models expand beyond language.

**Memes and Humor**

- [@Teknium1](https://twitter.com/Teknium1/status/1805718678526476655) joked about OpenAI having trouble removing "waifu features" in GPT-4 voice model updates.
- [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1805602848920715519) made a joke announcement that Noam Shazeer won the Turing Award for pioneering work on AI girlfriends.
- [@willdepue](https://twitter.com/willdepue/status/1805688616280293766) joked that "AGI is solved" now that you can search past conversations in chatbots.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

AI Progress

- **AI website generation**: A new AI system can generate full webpages from just a URL or description input, demonstrating progress in AI content creation capabilities. [Video demo](https://v.redd.it/y65tyl0r5r8d1).
- **OpenAI Voice Mode delay**: OpenAI [announced](https://x.com/openai/status/1805716393524183136?s=46) a one month delay for the advanced Voice Mode alpha release to improve safety and user experience. Plans for all Plus users to have access in the fall.
- **Singularity book release**: Ray Kurzweil released a sequel to his 2005 book The Singularity is Near, sparking [excitement and discussion](https://i.redd.it/dfrwlkvoop8d1.jpeg) about the future of AI.
- **AI agents speculation**: OpenAI's [acquisition](https://i.redd.it/swfpyso5jo8d1.png) of a remote desktop control startup led to speculation about integration with ChatGPT desktop for AI agents.
- **AI-generated ads**: Toys R Us used the SORA AI system to [generate a promotional video/ad](https://www.toysrus.com/pages/studios), showcasing AI in marketing.

AI Research

- **New optimizer outperforms AdamW**: A [research paper](https://arxiv.org/abs/2406.16793) introduced Adam-mini, a new optimizer that achieves 50% higher throughput than the popular AdamW.
- **Matrix multiplication eliminated in LLMs**: Researchers [demonstrated](https://arstechnica.com/information-technology/2024/06/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms/) LLMs that eliminate matrix multiplication, enabling much more efficient models with major implications for running large models on consumer hardware.
- **Simulating evolution with AI**: EvolutionaryScale [announced ESM3](https://www.evolutionaryscale.ai/blog/esm3-release), a generative language model that can simulate 500 million years of evolution to generate new functional proteins.

AI Products & Services

- **Deepseek Coder V2 math capabilities**: Users praised the [math capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1do72te/deepseek_coder_v2_is_so_good_for_math/) of the Deepseek Coder V2 model, a free model from China that outperforms GPT-4 and Claude.
- **AI audiobook narration**: An [AI-narrated audiobook](https://i.redd.it/bc9hntyelr8d1.jpeg) was well-received, implying audiobook narration is now a solved problem with AI.
- **New AI apps and features**: Several new AI applications and features were announced, including [Tcurtsni](https://www.reddit.com/gallery/1do5ykm), a "reverse-instruct" chat app, [Synthesia 2.0](https://youtu.be/gZaBwdru_bk?si=yHcsnnCJ5750xgPv), a synthetic media platform, and [Projects in Claude](https://support.anthropic.com/en/articles/9517075-what-are-projects) for organizing chats and documents.

AI Safety & Ethics

- **Rabbit data breach**: A security disclosure revealed a [data breach](https://www.reddit.com/r/singularity/comments/1do6uxz/rabbit_data_breach_all_r1_responses_ever_given/) in Rabbit where all responses from their R1 model could be downloaded, raising concerns about AI company negligence. 
- **Hallucination concerns**: An [opinion post](https://www.reddit.com/r/singularity/comments/1do8aqf/hallucinations_could_lead_to_a_bigger_problem/) argued the "AI hallucinates" talking point is dangerous as it masks the real risks of rapidly improving AI flooding job markets.

AI Hardware

- **AMD MI300X benchmarks**: [Benchmarks](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/) of AMD's new MI300X AI accelerator chip were released and analyzed.
- **Sohu AI chip claims**: A new Sohu AI chip was [announced](https://www.reddit.com/r/LocalLLaMA/comments/1dobzcs/meet_sohu_the_fastest_ai_chip_of_all_time/) claiming 500K tokens/sec on a 70B model, with 8 chips equivalent to 160 NVIDIA H100 GPUs.
- **MI300X vs H100 comparison**: A [comparison](https://i.redd.it/a8xl65u3ap8d1.jpeg) showed AMD's MI300X is ~5% slower but 46% cheaper with 2.5X the memory of NVIDIA's H100 on the LLaMA-2 70B model.

AI Art

- **A8R8 v0.7.0 release**: A new version of the A8R8 Stable Diffusion UI was [released](https://github.com/ramyma/a8r8/releases/tag/v0.7.0) with Comfy integration for regional prompting and other updates.
- **ComfyUI new features**: A [detailed post](https://www.reddit.com/r/StableDiffusion/comments/1dohy20/quick_overview_of_some_newish_stuff_in_comfyui/) reviewed new features in the ComfyUI Stable Diffusion environment like samplers, schedulers, and CFG implementations.
- **Magnific AI relighting tool**: Results from Magnific AI's new relighting tool were [compared](https://www.reddit.com/r/StableDiffusion/comments/1do2nym/comparison_between_magnific_ais_new_relighting/) to a user's workflow, finding it lacking in quality. 
- **SD model comparisons**: Different Stable Diffusion model sizes were [compared](https://www.magicflow.ai/insights/read/sd-body-positions) on generating specified body positions, with performance noted as "not good."

Other Notable News

- **Stability AI leadership changes**: Stability AI [announced](https://www.reddit.com/r/StableDiffusion/comments/1do9owa/stability_ai_announces_new_ceo_and_investors/) a new CEO, board members, funding round, and commitment to open source while expanding enterprise tools.
- **AI traffic analysis**: A [post](https://www.reddit.com/r/singularity/comments/1dohcke/request_how_to_quantify_ai_traffic_between/) proposed ways to quantify bandwidth usage of major AI systems, estimating AI is still a small part of overall internet traffic.
- **Politician shares false ChatGPT stats**: A news article reported a Canadian politician shared inaccurate statistics generated by ChatGPT, highlighting risks of using unverified AI outputs.
- **Open-source AI agent for on-call**: Merlinn, an open-source AI Slack bot to assist on-call engineers, was [announced](https://www.reddit.com/r/singularity/comments/1dohfqo/created_an_opensource_ai_agent_that_helps_during/).
- **Living skin robots**: BBC [reported](https://www.bbc.com/news/articles/cedd3208veyo) on research into covering robots with living human skin to make them more lifelike.
- **Gene therapy progress**: A [tweet](https://x.com/natrevdrugdisc/status/1805630241521435078?s=46) discussed gene therapies progressing from rare to common diseases.
- **Google AI event**: News that Google will reveal new AI tech and Pixel phones at an August event.
- **Tempering AI release expectations**: A [post](https://www.reddit.com/r/singularity/comments/1dohqjt/reality_check_a_planned_release_window_of_an/) advised taking AI product release dates with a grain of salt due to R&D uncertainty.
- **AI ending amateurism**: An [opinion piece](https://www.reddit.com/r/singularity/comments/1dohwlb/generative_ai_the_end_of_amateurism/) argued generative AI will allow everyone to produce professional-quality work.

---

# AI Discord Recap

> A summary of Summaries of Summaries

## Claude 3 Sonnet

**1. 🔥 LLM Advancements and Benchmarking**

- **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** from Meta tops leaderboards, outperforming GPT-4-Turbo and Claude 3 Opus per [ChatbotArena](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer).
- New models: **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** for coding, **[DeepSeek-V2](https://x.com/deepseek_ai/status/1787478986731429933)** with 236B parameters.
- Skepticism around certain benchmarks, calls for credible sources to set realistic evaluation standards.

**2. 🤖 Optimizing LLM Inference and Training**

- **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** promises 4x reduced communication overhead on GPUs.
- **[vAttention](https://arxiv.org/abs/2405.04437)** dynamically manages KV-cache memory for efficient inference.
- **[QServe](https://arxiv.org/abs/2405.04532)** uses **W4A8KV4 quantization** to boost cloud serving on GPUs.
- **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** explore parallel token decoding for lower latency.

**3. 🌐 Open-Source AI Frameworks and Community Efforts**  

- **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** supports diverse formats for instruction tuning and pre-training.
- **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** powers a course on building agentic RAG systems.
- **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** claims best for "unsexy data tasks".
- **[Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** teases Mojo's Python integration and AI extensions.

**4. 🖼 Multimodal AI and Generative Modeling Innovations**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** for elevated chat interactions. 
- **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** refines coding abilities.
- **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** brings powerful chatbots to browsers via WebGPU.
- Combining **Pixart Sigma + SDXL + PAG** aims for DALLE-3-level outputs with potential fine-tuning.
- Open-source **[IC-Light](https://github.com/lllyasviel/IC-Light)** for image relighting techniques.

**5. Stable Artisan for AI Media Creation in Discord**

- Stability AI launched **Stable Artisan**, a Discord bot integrating **Stable Diffusion 3**, **Stable Video Diffusion**, and **Stable Image Core** for [media generation within Discord](https://bit.ly/4aiVy6C).
- Sparked discussions around SD3's open-source status and Artisan's introduction as a paid API service.

## Claude 3.5 Sonnet

1. **LLMs Level Up in Performance and Efficiency**:

   - New models like [IBM's Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) and [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) are pushing boundaries in code instruction and data tasks. Communities across Discord channels are discussing these advancements and their implications.

   - Optimization techniques such as [Adam-mini](https://github.com/zyushun/Adam-mini) are gaining traction, promising 45-50% memory reduction compared to AdamW while maintaining performance. This has sparked discussions in the OpenAccess AI Collective and CUDA MODE Discords.

   - The [vAttention system](https://arxiv.org/abs/2405.04437) for efficient KV-cache memory management is being explored as an alternative to PagedAttention, highlighting the ongoing focus on inference optimization across AI communities.

2. **Open-Source AI Flourishes with Community-Driven Tools**:

   - [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) is gaining popularity for its support of diverse dataset formats in LLM training, discussed in both the OpenAccess AI Collective and HuggingFace Discords.

   - The [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) framework is powering new courses on building agentic RAG systems, generating excitement in the LlamaIndex and general AI development communities.

   - [Mojo](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)'s potential for Python integration and AI extensions is a hot topic in the Modular Discord, with discussions on its implications for AI development workflows.

3. **Multimodal AI Pushes Creative Boundaries**:

   - The combination of Pixart Sigma, SDXL, and PAG is being explored to achieve DALLE-3 level outputs, as discussed in the Stability.ai and general AI communities.

   - [Stable Artisan](https://bit.ly/4aiVy6C), a new Discord bot from Stability AI, is integrating models like Stable Diffusion 3 and Stable Video Diffusion, sparking conversations about AI-powered media creation across multiple Discord channels.

   - The open-source [IC-Light project](https://github.com/lllyasviel/IC-Light) for image relighting is gaining attention in computer vision circles, showcasing the ongoing innovation in image manipulation techniques.

4. **AI Hardware Race Heats Up**:

   - AMD's [Radeon Instinct MI300X](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/) is challenging Nvidia's dominance in the GPU compute market, despite software ecosystem challenges. This has been a topic of discussion in the CUDA MODE and hardware-focused Discord channels.

   - The announcement of [Etched's Sohu AI chip](https://www.etched.com/) has sparked debates across AI hardware communities about its potential to outperform GPUs in running transformer models, with claims of replacing multiple H100 GPUs.

   - Discussions about specialized AI chips versus general-purpose GPUs are ongoing, with community members in various Discord servers debating the future direction of AI hardware acceleration.

## Claude 3 Opus

**1. LLM Performance and Benchmarking**:

- Discussions about the performance of various LLMs, such as **Llama 3** from Meta outperforming models like **GPT-4-Turbo** and **Claude 3 Opus** on leaderboards like [ChatbotArena](https://lmsys.org/blog/2024-05-08-llama3/).
- New models like IBM's **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** and **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** showcasing advancements in instruction following and parameter count.
- Concerns about the credibility of certain benchmarks and the need for realistic LLM assessment standards from reputable sources.

**2. Hardware Advancements and Optimization Techniques**:

- Techniques like **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** and **[vAttention](https://arxiv.org/abs/2405.04437)** being explored to optimize GPU memory usage and reduce communication overhead during LLM training and inference.
- Advancements in quantization, such as **[QServe](https://arxiv.org/abs/2405.04532)** introducing **W4A8KV4 quantization** for improved GPU performance in cloud-based LLM serving.
- Discussions about the potential of specialized AI chips like **Etched's Sohu** and comparisons with GPU performance for running transformer models.

**3. Open-Source Frameworks and Community Efforts**:

- Open-source frameworks like **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** and **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** supporting diverse dataset formats and enabling the development of agentic RAG systems.
- The release of open-source models like **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)**, claiming to be the best LLM for "unsexy data tasks."
- Community efforts to integrate AI capabilities into platforms like Discord, with bots such as **Stable Artisan** from Stability AI for media generation and editing.

**4. Multimodal AI and Generative Models**:

- New models focusing on specific tasks, such as **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** for elevated chat interactions and **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** for coding abilities.
- Advancements in browser-based AI chatbots, like the **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** model utilizing WebGPU for powerful interactions.
- Efforts to combine techniques like **Pixart Sigma**, **SDXL**, and **PAG** to achieve **DALLE-3**-level outputs in generative models.
- Open-source projects like **[IC-Light](https://github.com/lllyasviel/IC-Light)** focusing on specific tasks such as image relighting.


## GPT4O (gpt-4o-2024-05-13)

1. **Model Performance and Benchmarks**:
   - **[Llama3 70B Models Show Promise](https://x.com/ClementDelangue/status/1805989925080219927)**: New open LLM leaderboards hosted on 300 H100 GPUs have Qwen 72B leading, though bigger models don't always equate to better performance. Analyses highlighted differences in scope between training vs. inference benchmarks. 
   - **[Solving Grade School Arithmetic](https://arxiv.org/abs/2405.00332)** highlights skepticism where data leakage in large LLMs results in misleadingly high benchmarks despite incomplete learning. Calls for credible assessments were noted.

2. **Training, Optimization and Implementation Issues**:
   - **[Push for Better Optimizers](https://arxiv.org/abs/2406.16793)**: Adam-mini optimizer offers equivalent performance to AdamW but reduces memory use by 45-50%. This optimizer simplifies storage by reducing the number of learning rates per parameter.
   - **[Memory Management in High-Context Models](https://github.com/zyushun/Adam-mini)**: Efforts to load large models, such as Llama3 70B or Hermes, on consumer-grade GPUs are hindered by significant OOM errors, driving discussions on effective GPU VRAM utilization.

3. **AI Ethics and Community Debates**:
   - **[Ethics of AI Data Use](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main)**: Debates in **LAION** Discord stressed the controversial inclusion of NSFW content in datasets, balancing ethical concerns with the motivation for unrestricted data access.
   - **Model Poisoning Concerns**: Discussions in **LAION** focused on ethical implications and potential model poisoning, where controversial techniques in training and dataset usage are encouraged without broader consideration of long-term impacts.

4. **Specialized AI Hardware Trends**:
   - **[Etched's Sohu Chips Boast 10x Performance](https://x.com/bryan_johnson/status/1805629207374086490)**: Etched’s new transformer ASIC chips claim to outperform Nvidia GPUs significantly, with considerable financial backing. However, practical adaptability and inflexibility concerns were discussed within **CUDA MODE**.
   - **[AMD's MI300X Challenges Nvidia](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/)**: AMD's MI300X seeks to dethrone Nvidia in GPU compute markets, despite lagging behind Nvidia's CUDA ecosystem.

5. **AI Application Integration**:
   - **[Custom GPT Apps on Hugging Face Flourish](https://github.com/teknium1/Prompt-Engineering-Toolkit)**: Growing interest in custom GPT-based applications, citing niche tasks like Japanese sentence explanations, remains strong. Collaborative efforts in the community have driven the creation of resources and toolkits for ease of implementation.
   - **[AI-Assisted Tools Expand Academic Reach](https://gpasaver.com/)**: The new **GPA Saver platform** leverages AI for academic assistance, indicating growing integration of AI in streamlined educational tools. Community discussions about improving AI-driven functionalities highlighted potential and current constraints.

---

# PART 1: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Quick Access with a Shortcut**: The ChatGPT desktop app for macOS is now available, featuring a quick-access [Option + Space shortcut](https://openai.com/chatgpt/mac/) for seamless integration with emails and images.

**Voice Mode Hiccup**: The anticipated advanced Voice Mode for ChatGPT has been postponed by a month to ensure quality before alpha testing; expect more capabilities like emotion detection and non-verbal cues in the fall.

**OpenAI vs Anthropic's Heavyweights**: Discussions are heating with regards to GPTs agents' inability to learn post-training and Anthropic's Claude gaining an edge over ChatGPT due to technical feats, such as larger token context windows and a rumored **MoE setup**.

**Customization Craze in AI**: Enthusiasts are creating custom GPT applications using resources like **Hugging Face**, with a particular interest in niche tasks like explaining Japanese sentences, as well as concerns about current limitations in OpenAI's model updates and feature rollout.

**GPT-4 Desktop App and Performance Threads**: Users noted the limitation of the new macOS desktop app to Apple Silicon chips and shared mixed reviews on GPT-4's performance, expressing desire for Windows app support and improvements in response times.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **RAG Under the Microscope**: A discussion centered on the use of **Retrieval-Augmented Generation (RAG)** techniques highlighted consideration for managing document length with SSM like **Mamba** and using **BM25** for keyword-oriented retrieval. A GitHub resource related to BM25 [can be found here](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb).

- **Interactive Hand Gestures**: Two separate contexts highlighted a Python-based "**Hand Gesture Media Player Controller**," shared via a [YouTube demonstration](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM), indicating burgeoning interest in applied computer vision to control interfaces.

- **PAG Boosts 'Diffusers' Library**: An integration of **Perturbed Attention Guidance (PAG)** into the **`diffusers` library** promises enhanced image generation, as announced in [HuggingFace's core announcements](https://github.com/huggingface/diffusers/issues/8704), thanks to a community contribution.

- **Cracking Knowledge Distillation for Specific Languages**: Queries around knowledge distillation were prominent, with one member proposing a distilled multilingual model for a single language and another recommending SpeechBrain for tackling the task.

- **LLMs and Dataset Quality in Focus**: Alongside advances such as the **Phi-3-Mini-128K-Instruct** model by Microsoft, the community spotlighted the importance of dataset quality. Concurrently, concerns related to data leakage in LLMs were addressed through papers citing the issue [here](https://arxiv.org/abs/2404.18824) and [here](https://arxiv.org/abs/2405.00332). 

- **Clamor for AI-driven Tools**: From a request for a seamless **AI API development** platform, referenced through a [feedback survey](https://forms.gle/yAfGjUtNTnf5mASK7), to the challenge of identifying data in handwritten tables, there's a clear demand for AI-powered solutions that streamline tasks and inject efficiency into workflows.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI Ethics Take Center Stage**: Conversations arose about the ethics in AI training, where a member expressed concerns about active encouragement of model **poisoning**. Another member debated the offered solution to the AIW+ problem as incorrect, mentioning it overlooks certain familial relationships, thus suggesting ambiguity and ethical considerations.

- **Music Generation with AI Hits a High Note**: Discussions involved using **RateYourMusic ID** to generate songs and lyrics, with an individual confirming its success and describing the outcomes as "hilarious."

- **The Great NSFW Content Debate**: A debate surged regarding whether NSFW content should be included in datasets, highlighting the dichotomy between moral concerns and the argument against excessively cautious model safety measures.

- **GPU Showdown and Practicality**: Members exchanged insights on the trade-offs between **A6000s, 3090s**, and **P40 GPUs**, noting differences in VRAM, cooling requirements, and model efficiency when applied to AI training.

- **ASIC Chips Enter the Transformer Arena**: An emerging topic was **Etched's Sohu**, a specialized chip for transformer models. Its touted advantages sparked discussions on its practicality and adaptability to various AI models, contrasting with skepticism concerning its potential inflexibility.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ICML 2024 Papers on the Spotlight**: EleutherAI researchers gear up for ICML 2024 with [papers addressing classifier-free guidance and open foundation model impacts](https://arxiv.org/abs/2306.17806). Another study delves into [memorization in language models](https://arxiv.org/abs/2406.17746), examining issues like privacy and generalization.

- **Multimodal Marvels and Gatherings Galore**: Huggingface's leaderboard emerges as a handy tool for those seeking top-notch multimodal models; meanwhile, [ICML's Vienna meet-up](https://icml.cc/Conferences/2024) attracts a cluster of enthusiastic plans. The hybrid model Goldfinch also became a part of the exchange, merging Llama with Finch B2 layers for enhanced performance.

- **Papers Prompting Peers**: Discussion in the [#research](https://discord.com/channels/729741769192767510/747850033994662000/1255280862199550054) channel flared around papers from comparative evaluations of Synquid to the application of Hopfield Networks in transformers. Members dissected topics ranging from multimodal learning efficiencies to experimental approaches in generalization and grokking.

- **Return of the Hopfields**: Members offered insights on self-attention in neural networks by corralling it within the framework of (hetero)associative memory, bolstered by references to continuous modern Hopfield Networks and their implementation as single-step attention.

- **Sparse and Smart**: Sparse Autoencoders (SAEs) take the stage for their aptitude in unearthing linear features from overcomplete bases, as touted in [LessWrong posts](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition). Additionally, a noteworthy mention was a [paper on multilingual LLM safety](https://arxiv.org/abs/2406.16235), demonstrating cross-lingual detoxification from directionally poisoned optimization (DPO).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

### **AMD's Radeon MI300X Takes on Nvidia**: 
The new **AMD Radeon Instinct MI300X** is positioned to challenge Nvidia's dominant status in the GPU compute market despite AMD's software ecosystem ROCm lagging behind Nvidia's CUDA, as detailed in an article on [Chips and Cheese](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/).

### **ASIC Chip Ambitions**: 
Etched's announcement of the **Transformer ASIC chips** aims to outpace GPUs in running AI models more efficiently, with significant investment including a $120 million series A [funding round](https://x.com/bryan_johnson/status/1805629207374086490) supported by Bryan Johnson, raising discussions about the future role of specialized AI chips.

### **Optimization Tweaks and Triton Queries**: 
Engineering conversations revolve around a proposed **Adam-mini optimizer** that operates with 45-50% less memory, with code available on [GitHub](https://github.com/zyushun/Adam-mini), and community assistance sought for a `pow` function addition in `python.triton.language.core` as shown in this [Triton issue](https://github.com/triton-lang/triton/issues/4190).

### **PyTorch Celebrates with Documentary**: 
The premiere of the "PyTorch Documentary Virtual Premiere: Live Stream" has garnered attention, featuring PyTorch’s evolution and its community, substantially reiterated by users and symbolized with *goat emojis* to express the excitement, watchable [here](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s).

### **Intel Pursues PyTorch Integration for GPUs**: 
Building momentum for **Intel GPU (XPU)** support in stock PyTorch continues with an Intel PyTorch team's [RFC on GitHub](https://github.com/pytorch/pytorch/issues/114723), signaling Intel’s commitment to becoming an active participant in the deep learning hardware space.

### **Discussions of AI Infrastructure and Practices**: 
Community dialogue featured topics like learning rate scaling, update clipping with insights from an [AdamW paper](https://mlfoundations.github.io/advancedml-sp23/assets/adam.pdf), infrastructural choices between AMD and Nvidia builds, and the intrigue around the Sohu ASIC chip's promises, impacting the efficacy of large transformer models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexed by Perplexity API**: Engineers discussed intermittent **5xx errors** with the **Perplexity AI's API**, highlighting the need for better transparency via a status page. There were also debates on API filters and undocumented features, with some users probing the existence of a search domain filter and citation date filters.

**In Search of Better Search**: The **Perplexity Pro focus search** faced criticism for limitations, while comparisons to **ChatGPT** noted Perplexity's new agentic search capabilities but criticized its tendency to hallucinate in summarizations.

**Claude Leverages Context**: The guild buzzed about **Claude 3.5's** 32k token context window for **Perplexity Pro** users, with Android support confirmed. Users showed a clear preference for the full 200k token window offered by Claude Pro.

**Innovation Insight with Denis Yarats**: The CTO of Perplexity AI dissected AI's innovation in a [YouTube video](https://www.youtube.com/watch?v=gvP-DxqatLQ), discussing how it revolutionizes search quality. In a related conversation, researchers presented a new method that could change the game by removing matrix multiplication from language model computations.

**Hot Topics and Searches in Sharing Space**: The community shared numerous Perplexity AI searches and pages including evidence of Titan's missing waves, China's lunar endeavors, and a study on how gravity affects perception, encouraging others to explore these curated searches on their platform.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI World's Fair Watch Party Launch**: Enthusiasm stirred up for hosting a watch party for **AI Engineer World’s Fair**, livestreamed [here](https://www.youtube.com/watch?v=5zE2sMka620), spotlighting cutting-edge keynotes and code tracks.

- **Premiere Night for PyTorch Fans**: Anticipation builds around the [PyTorch Documentary Virtual Premiere](https://www.youtube.com/live/EjgTv6aSeqk), highlighting the evolution and impact of the project with commentary from its founders and key contributors.

- **ChatGPT's Voice Update Muted**: A delayed release of ChatGPT's Voice Mode, due to technical difficulties with voice features, causes a stir following a [tweet by Teknium](https://x.com/Teknium1/status/1805718678526476655/photo/1).

- **Bee Computer Buzzes with Intelligence**: Attendees at an AI Engineer event buzz over new **AI wearable tech from Bee Computer**, touted for its in-depth personal data understanding and proactive task lists.

- **Neural Visuals Exceed Expectations**: A breakthrough in neuroscience captures community interest with the [reconstruction of visual experiences from mouse cortex activity](https://x.com/Neuro_Joel/status/1805221959191437356), demonstrating incredible neuroimaging strides.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Tech Troubles and Tips in LM Studio**: Engineers reported errors with LM Studio (0.2.25), including an *Exit code: -1073740791* when loading models. For **Hermes 2 Theta Llama-3 70B**, users with RTX 3060ti faced "Out of Memory" issues and considered alternatives like NousResearch's 8b. Issues were also noted when running **Llama 3 70B** on Apple's M Chip due to different quant types and settings.

- **RAG Gets the Spotlight**: A detailed discussion on retrieval-augmented generation (RAG) took place, highlighting NVIDIA's [blog post](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) on RAG's capability to enhance information generation accuracy with external data.

- **Scam Warnings and Security Tips**: Users noted the presence of scam links to a Russian site impersonating Steam and reported these for moderator action. There's awareness in the community regarding phishing attacks and the importance of securing personal and project data.

- **Hardware Conversations Heat Up**: A completed build using **8x P40 GPUs** was mentioned, sparking further discussions on server power management involving a **200 amp circuit** and VRAM reporting accuracy in LM Studio for multi-GPU setups. The noise produced by home server setups was also humorously likened to a jet engine.

- **Innovative Ideas and SDK Expo**: Members shared ideas ranging from using an LLM as a game master in a sci-fi role-playing game to solving poor performance with larger context windows in token prediction. There's a guide to building Discord bots with the SDK [here](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6) and questions regarding extracting data from the LM Studio server using Python.

- **Uploading Blocks in Open Interpreter**: There's frustration over the inability to upload documents or images directly into the open interpreter terminal, limiting users in interfacing with AI models and use cases.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Plotting a Path with Mojo Data Types**: Engineers are experimenting with **Mojo data types** for direct plotting without conversion to Numpy, utilizing libraries like [Mojo-SDL](https://github.com/msteele/mojo-sdl) for SDL2 bindings. The community is mulling over the desired features for a Mojo charting library, with focus areas ranging from high-level interfaces to interactive charts and integration with data formats like Arrow.

- **Vega IR for Versatile Visualization**: The need for interactivity in data visualization was underscored, with the **Vega specification** being proposed as an Intermediate Representation (IR) to bridge web and native rendering. The conversation touched on the unique approaches of libraries like **UW's Mosaic** and mainstream ones like D3, Altair, and Plotly.

- **WSL as a Windows Gateway to Mojo**: Mojo has been confirmed to work on Windows via the Windows Subsystem for Linux (WSL), with native support anticipated by year's end. Ease of use with Visual Studio Code and Linux directories was a highlight.

- **IQ vs. Intelligence Debate Heats Up**: The community engaged in a lively debate about the nature of intelligence, with the **ARC test** questioned for its human-centric pattern recognition tasks. Some users view AI excelling at IQ tests as not indicative of true intelligence, while the concept of consciousness versus recall sparked further philosophical discussion.

- **Compile-Time Quirks and Nightly Builds**: Multiple issues with the Mojo compiler were aired, ranging from reported bugs in type checking and handling of boolean expressions to the handling of `List` and `Tensor` at compile time. Encouragement to report issues, even if resolved in nightly builds, was echoed across the threads. Specific commits, nightly build updates, and suggestions for referencing immutable static lifetime variables were also discussed, rallying the community around collaborative debugging and improvement.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LLM Leaderboard Bragging Rights Questioned**: Clement Delangue's announcement of a [new open LLM leaderboard](https://x.com/ClementDelangue/status/1805989925080219927) boasted the use of 300 H100 GPUs to rerun MMLU-pro evaluations, prompting sarcasm and criticisim about the necessity of such computing power and the effectiveness of larger models.
  
- **API Security Gone Awry at RabbitCode**: Rabbitude's discovery of **hardcoded API keys**, including ones for [ElevenLabs](https://elevenlabs.io) and others, has left services like Azure and Google Maps vulnerable, causing concerns over unauthorized data access and speculation about the misuse of ElevenLabs credits.

- **Delay in ChatGPT's Advanced Voice Mode**: [OpenAI](https://openai.com) has postponed the release of ChatGPT’s advanced Voice Mode for Plus subscribers till fall, aiming to enhance content detection and the user experience, as shared via [OpenAI's Twitter](https://x.com/openai/status/1805716393524183136?s=46).

- **Murmurs of Imbue’s Sudden Success**: Imbue's sudden $200M fundraise drew skepticism among members, exploring the company's unclear history and comparing their trajectory with the strategies of **Scale AI** and its subsidiaries for data annotation and PhD recruitment for remote AI projects.

- **Music Industry’s AI Transformation**: Udio's [statement](https://x.com/udiomusic/status/1805694761891778783?s=46) on AI's potential to revolutionize the music industry clashed with the [RIAA's concerns](https://x.com/riaa/status/1805739691972452559?s=46), asserting AI will become essential for music creation despite industry pushback.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Challenging Stability AI to Step Up**: Discussions point to growing concerns about Stability AI’s approach with **Stable Diffusion 3 (SD3)**, stressing the need for uncensored models and updated licenses to retain long-term viability. A more practical real-world application beyond novelty creations is requested by the community.

- **Cost-Effective GPU Strategies Discussed**: The comparison of GPU rental costs reveals **Vast** as a more economical option for running a 3090 compared to **Runpod**, with prices cited as low as 30 cents an hour.

- **Debate: Community Drive vs. Corporate Backup**: There's an active debate on the balance between open-source initiatives and corporate influence, with some members arguing for community support as crucial and others citing Linux's success with enterprise backing as a valid path.

- **Optimizing Builds for Machine Learning**: Members are sharing hardware recommendations for effective **Stable Diffusion** setups, with a consensus forming around the Nvidia 4090 for its performance benefit, potentially favoring dual 4090s over higher VRAM single GPUs for cost savings.

- **Nostalgia Over ICQ and SDXL Hurdles**: The shutdown of the legacy messaging service **ICQ** triggered nostalgic exchanges, while the community also reported challenges in running **SDXL**, particularly for those experiencing *"cuda out of memory"* errors due to insufficient VRAM, seeking advice on command-line solutions.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Introducing the Prompt Engineering Toolkit**: An open-source [Prompt Engineering Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit) was shared for use with **Sonnet 3.5**, designed to assist with creating better prompts for AI applications.
  
- **Skepticism Breeds Amidst Model Performance**: A demonstration of Microsoft's new raw text data augmentation model on [Genstruct](https://huggingface.co/spaces/davanstrien/Genstruct-7B) prompted doubts about its efficacy, showing results that seemed off-topic.

- **AI Chip Performance Heats Up Debate**: The new "Sohu" AI chip sparked discussions about its potential for high-performance inference tasks, linking to [Gergely Orosz's post](https://x.com/GergelyOrosz/status/1805604272614088721) which suggests OpenAI doesn't believe AGI is imminent despite advancing hardware.

- **70B Model Toolkit Launched by Imbue AI**: Imbue AI released a toolkit for a **70B model** with resources including **11 NLP benchmarks**, a **code-focused reasoning benchmark**, and a **hyperparameter optimizer**, found at [Imbue's introductory page](https://imbue.com/research/70b-intro/).

- **Embracing the Whimsical AI**: A post from a user featured AI-generated content in meme format by **Claude** from Anthropic, reflecting on Claude's explanation of complex topics and its humorous take on not experiencing weather or existential crises.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Streamlining AI Conversations**: Engineers highlighted the `.stream()` method from `langchain_community.chat_models` for iterating through LangChain responses, while others discussed integrating [Zep](https://www.getzep.com/) for long-term memory in AI and contemplated direct `BytesIO` PDF handling in LangChain without temp files.

- **Visualization Quest in LangChain**: Discussion around live visualizing agents' thoughts in Streamlit touched on using `StreamlitCallback` but also identified a gap in managing streaming responses without callbacks.

- **Troubleshooting the Unseen**: Inquiries were made about LangSmith's failure to trace execution despite proper environmental setup, with a suggestion to check trace quotas.

- **Extending Containerized Testing**: A community member contributed Ollama support to **testcontainers-python**, facilitating LLM endpoint testing, as indicated in their [GitHub issue](https://github.com/testcontainers/testcontainers-python/issues/617) and [pull request](https://github.com/testcontainers/testcontainers-python/pull/618).

- **Cognitive Crafts and Publications**: A [Medium article](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b38fe1) on few-shot prompting with tool calling in Langchain was shared, alongside a YouTube video exploring the ARC AGI challenges titled "[Claude 3.5 struggle too?! The $Million dollar challenge](https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap)".



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Chatbots Seeking Contextual Clarity**: An engineer inquired about how to effectively retrieve context directly from a chat response within the [LlamaIndex chatbot framework](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/), sharing implementation details and the challenges encountered.

- **Pull Request Review on the Horizon**: A member shared a GitHub PR for review, aimed at adding query filtering functionalities to the **Neo4J Database in LlamaIndex**, and another member acknowledged the need to address the backlog.

- **Silencing Superfluous Notifications**: There was a discussion on how to suppress unnecessary notifications about missing machine learning libraries in the Openailike class, with the clarification that such messages are not errors.

- **Tuning SQL Queries with LLMs**: Dialogue among users highlighted the benefits of fine-tuning language models for enhanced precision in SQL queries when using a RAG SQL layer, suggesting better performance with quality training data.

- **Balancing Hybrid Searches**: Questions about hybrid search implementations in **LlamaIndex** have been addressed, focusing on adjusting the `alpha` parameter to balance metadata and text relevance in search results.

- **Boosting RAG with LlamaIndex**: An article was shared highlighting ways to build optimized Retrieval-Augmented Generation systems with LlamaIndex and DSPy, providing insights and practical steps for AI engineers.

- **Open Source Contribution Perks**: A call was made for feedback on an open-source project, [Emerging-AI/ENOVA](https://github.com/Emerging-AI/ENOVA), for enhancing AI deployment, monitoring, and auto-scaling, with an incentive of a $50 gift card.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Claude-3.5-Sonnet Steps into the Spotlight**: The latest **Anthropic model** is officially named `claude-3-5-sonnet-20240620`, putting an end to name confusion among members.

- **MoonDream's Vision Limitation Acknowledged**: While there's interest in a **MoonDream-based** vision model for **OpenInterpreter (OI)**, current conversation confirms it's not compatible with OI.

- **Multiline Input Quirks and Vision Command Errors**: Technical issues arose with `-ml` for multiline inputs and the `interpreter --os --vision` command, with one user verifying their **API key** but facing errors, and another member reported a **ban** from attempting to directly drop files into the terminal.

- **01: OI's Voice Interface, Not for Sale Everywhere**: **01**, as the voice interface for **OI**, can't be bought in Spain; enthusiasts are redirected to an [open-source dev kit](https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight) on GitHub for DIY alternatives.

- **Constructing Your Own 01**: Tutorials for DIY assembly of 01 from the open-source kit will be proliferating, including one planned for July, hinting at the community's commitment to ensuring wider access beyond commercial sale limitations.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Curiosity About Cohere's Scholars Program**: One member inquired about the status of the **scholars program** for the current year, but no additional information or discussion followed on this topic.

**Billable Preamble Tokens in the Spotlight**: A user highlighted an experiment involving **preamble tokens** for API calls, bringing up a cost-cutting loophole that could avoid charges by exploiting non-billable preamble usage.

**Designing with Rust for LLMs**: An announcement was made about the release of **Rig**, a Rust library for creating LLM-driven applications, with an invitation to developers to engage in an incentivized feedback program to explore and review the library.

**Ethical Considerations Surface in AI Usage**: Concerns were brought up regarding **SpicyChat AI**, a NSFW bot hosting service, potentially violating Cohere's **CC-BY-NA** license through profit-generating use coupled with the claim of circumventing this via **OpenRouter**.

**Learning Event on 1Bit LLMs by Hongyu Wang**: An online talk titled *The Era of 1Bit LLMs* hosted by **Hongyu Wang** was announced with an invitation extended to attend through a provided [Google Meet link](https://meet.google.com/yhv-tiir-ava).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Adam Optimizer Slims Down**: Engineers discussed an [arXiv paper](https://arxiv.org/abs/2406.16793) introducing **Adam-mini**, highlighting its reduced memory footprint by 45% to 50% compared to AdamW. It achieves this by using fewer learning rates, leveraging parameter block learning inspired by the Hessian structure of Transformers.

- **Training Pitfalls and CUDA Quandaries**: One engineer sought advice on implementing output text masking during training, akin to `train_on_input`, while another raised an issue with **CUDA** errors, suggesting enabling `CUDA_LAUNCH_BLOCKING=1` for identifying illegal memory access during model training.

- **Gradient Accumulation—Friend or Foe?**: The impact of increasing gradient accumulation was hotly debated; some believe it may shortcut training by running the optimizer less often, others worry it could lead to slower steps and more training time.

- **Cosine Schedules and QDora Quests**: Questions arose about creating a cosine learning rate scheduler with a non-zero minimum on the **Hugging Face platform**, and excitement was evident over a pull request enabling **QDora** in **PEFT**.

- **Narrative Engines and Mistral Mysteries**: The introduction of [Storiagl](https://storiagl.web.app/), a platform for building stories with custom LLMs, was showcased, while another engineer reported a repetitive text generation issue with **Mistral7B**, despite high temperature settings and seeking solutions.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Prompting Takes the Cake in Language Learning**: Researchers, including Eline Visser, have shown that **prompting a large language model** (LLM) outperforms fine-tuning when learning **Kalamang language** using a single grammar book. The findings, indicating that 'prompting wins', are detailed in a [tweet by Jack Morris](https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ) and further elaborated in an [academic paper](https://arxiv.org/abs/2309.16575).

**Catch the AI Engineer World’s Fair Online**: The **AI Engineer World's Fair 2024** is being streamed live, focusing on keynotes and the CodeGen Track, with access available [on YouTube](https://www.youtube.com/watch?v=5zE2sMka620); more specifics are provided on [Twitter](https://twitter.com/aidotengineer).

**Claude Contest Calls for Creatives**: The June 2024 **Build with Claude** contest has been announced, inviting engineers to demonstrate their expertise with Claude, as outlined in the [official guidelines](https://docs.anthropic.com/en/build-with-claude-contest/overview).

**Credit Where Credit is Due**: An individual offered assistance with a credit form issue, asking to be directly messaged with the related email address to *resolve the matter efficiently*.

**Model Offloading Techniques Debated**: The community has observed that **DeepSpeed (DS)** seems to have more effective fine-grained offloading strategies compared to **FairScale's Fully Sharded Data Parallel (FSDP)**. Additionally, the utility of these offloading strategies with **LLama 70B** is under consideration by members seeking to optimize settings.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla's Builders Program Ticks Clock**: Members are reminded to submit their applications for the **Mozilla Builders Program** before the **July 8th** early application deadline. For support and additional information, check the [Mozilla Builders Program page](https://future.mozilla.org/builders/).

- **'90s Nostalgia via Firefox and llamafile**: **Firefox** has integrated llamafile as an HTTP proxy, allowing users to venture through LLM weights in a retro web experience; a demonstration video is available on [YouTube](https://youtu.be/YWQ5Kh9gNuo).

- **Create Your Own Chat Universe**: Users can create immersive chat scenarios, fusing llamafile with Haystack and Character Codex, through a shared notebook which is accessible [here](https://t.ly/y6jrZ).

- **Cleansing CUDA Clutter in Notebooks**: To keep Jupyter notebooks pristine, it's suggested to address CUDA warnings by using the [utility from Haystack](https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py).

- **NVIDIA's Stock Sent on a Rollercoaster**: Following a talk at AIEWF, NVIDIA's market cap fell dramatically, triggering various analyses from outlets like [MarketWatch](https://www.marketwatch.com/story/nvidias-stock-is-set-to-gain-with-rivals-seen-to-be-in-perpetual-catch-up-mode-0552e514) and [Barrons](https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c) over the catalyst of the company's financial performance.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Explores FPGA Acceleration**: There's chatter about **tinygrad** leveraging FPGAs as a backend, with George Hotz hinting at a potential **accelerator design** for implementation.
- **Groq Alumni Launch Positron for High-Efficiency AI**: Ex-Groq engineers introduced [Positron](https://www.positron.ai/), targeting the AI hardware market with devices like Atlas Transformer Inference Server, boasting a **10x performance boost** per dollar over competitors like DGX-H100.
- **FPGA's Role in Tailored AI with HDL**: Discussion centered on the future of FPGAs equipped with DSP blocks and HBM, which could allow for the creation of model-specific HDL, although it was noted that Positron's approach is generic and not tied to a specific FPGA brand.
- **PyTorch's Impact on AI Celebrated in Documentary**: A [documentary on YouTube](https://www.youtube.com/watch?v=rgP_LBtaUEc) highlighting PyTorch's development and its influence on AI research and tooling has been shared with the community.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Angry.penguin Ascends to Mod Throne**: User **angry.penguin** was promoted to moderator to tackle the guild's spam problem, volunteering with a proactive approach and immediately cleaning up the existing spam. Yoko Li entrusted angry.penguin with these new responsibilities and spam control measures.

- **Spam No More**: Newly-minted moderator **angry.penguin** announced the successful implementation of anti-spam measures, ensuring the guild's channels are now fortified against disruptive spam attacks. Members may notice a cleaner and more focused discussion environment moving forward.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **German Encoders Go Live on Hugging Face**: AI engineers might be enticed by the newly released **German Semantic V3 and V3b** encoders, available on [Hugging Face](https://huggingface.co/aari1995/German_Semantic_V3). V3 targets knowledge-based applications, while V3b emphasizes high performance with innovative features including Matryoshka Embeddings and 8k token context capability.

- **Finetuning Steps for German Encoders Without GGUF**: Despite inquiries, the **German V3b encoder** does not currently have a **gguf** format; however, for those interested in finetuning, it is recommended to use UKPLab's sentence-transformers [finetuning scripts](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training).

- **Possibility of GGUF for Encoders Empowered by Examples**: In the wake of confusion, a member clarified by comparing with **Ollama**, establishing that encoders like German V3 can indeed be adapted to **gguf formats** which may involve using dual embedders for enhanced performance.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **New AI Player in Town**: OpenRouter has introduced the [01-ai/yi-large model](https://openrouter.ai/models/01-ai/yi-large), a new language model specialized in knowledge search, data classification, human-like chatbots, and customer service; the model supports multilingual capabilities.

- **Parameter Snafu Resolved**: The Recommended Parameters tab for the model pages on OpenRouter had data display issues, which have been **fixed**, ensuring engineers now see accurate configuration options.

- **AI Meets Academia**: The newly launched [GPA Saver](https://gpasaver.com/) leverages AI to offer academic assistance and includes tools like a chat assistant, rapid quiz solver, and more; early adopters get a discount using the code **BETA**.

- **Easing the Integration Experience**: Thanks were expressed to OpenRouter for streamlining the process of **AI model integration**, which was instrumental in the creation of the GPA Saver platform.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255250846426595428)** (2 messages): 

- **ChatGPT desktop app for macOS releases**: The ChatGPT desktop app for macOS is now available to all users. Get quicker access to ChatGPT with the [Option + Space shortcut](https://openai.com/chatgpt/mac/), enabling seamless chats about emails, screenshots, and more.

- **Advanced Voice Mode delayed but incoming**: The rollout of the advanced Voice Mode, initially planned for late June, has been delayed by a month to ensure quality. This mode, capable of understanding emotions and non-verbal cues, will start alpha testing with a small group before expanding to all Plus users in the fall, with updates on video and screen sharing capabilities to follow.
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255251858323410985)** (388 messages🔥🔥): 

- **GPTs Agents and OpenAI's Progress**: Members expressed frustration over GPTs agents not learning new information after initial training. They pointed out that while OpenAI's models excel in initial training, continuous improvements are held back by excessive regulations.

- **Anthropic's Claude Rises in Popularity**: Discussions highlighted how **Anthropic's Claude 3.5 Sonnet** has gained traction, with claims it offers better performance in coding and larger context windows compared to OpenAI's models. One user speculated on the efficiency of its architecture, possibly employing a **MoE setup**.

- **Model Performance Comparisons**: Users discussed the upper hand **Anthropic's Claude** has over **OpenAI's ChatGPT**, especially in token context windows and refusal rates. While some argued **Claude** is more censored, others noted **Claude's technical improvements**, like larger token windows and less lag in responses.

- **Open Source and Custom Models**: There was interest in custom GPTs and synthetic datasets tailored for niche applications, such as **Japanese sentence explanations**. Users shared resources like **Hugging Face** datasets and local inference tools like **LM Studio** for further customization.

- **Criticism and Future Prospects of OpenAI**: Members voiced concerns about the delayed rollout of **OpenAI's voice features** and the limited benefits of the **ChatGPT Plus subscription**. They hope for advancements in context windows and other features to match competitors like **Google's Gemini** and **Anthropic's Claude**.
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255251153948639294)** (21 messages🔥): 

- **Windows Beats Mac in Desktop App Demand**: *"Wouldn't Windows desktop app get way more used than mac desktop app?"* sparked a discussion, with another user stating *"yes, thats bs releasing it for mac."*
- **LaTeX Formatting and GPT-4o Performance Concerns**: A member explained they get the best results specifying LaTeX format. They also noted performance issues with GPT-4o, citing failure in logic and historical research tasks.
- **Mac Desktop App Limited to Apple Silicon**: The discussion around the new macOS desktop app clarified it is only available for Apple Silicon (M1 or better), with no plans to support Intel Macs.
- **TTS Model New Voices Inquiry**: A user asked if the new voices would be available through the TTS model but did not receive a direct response.
- **Slow GPT-4o Responses Frustrate Users**: Members inquired and complained about the slowness of GPT-4o, wondering if there was an underlying problem.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255516869595627561)** (1 messages): 

- **Context Matters for AI Errors**: One member noted that understanding AI mistakes depends greatly on the topic, **knowledge content**, and context. They suggested that reviewing what specific errors the AI is making could be helpful.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255516869595627561)** (1 messages): 

- **Dependence on AI's understanding of context and knowledge**: The effectiveness of the AI's responses depends heavily on the topic and the context of the knowledge content. One member noted, *"It'd be helpful to see what the AI is getting wrong."*
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1255261278994305024)** (1 messages): 

- **Argilla 2.0 enhances AI dataset creation**: The release of [Argilla 2.0](https://x.com/argilla_io/status/1805250218184560772) introduces a unified framework for feedback collection, a new Python SDK, flexible UI for data annotation, and updated documentation. These features aim to assist AI builders in creating high-quality datasets more efficiently.
- **Microsoft's Florence model impresses**: Microsoft launched [Florence](https://x.com/osanseviero/status/1803324863492350208), a vision model capable of multiple tasks like captioning and OCR. The models are MIT licensed and provide high quality despite their smaller sizes compared to much larger models.
- **Instruction pre-training by Microsoft**: Microsoft's [Instruction Pre-Training](https://x.com/osanseviero/status/1804136001465442530) can enhance LLM pretraining with instruction-response pairs, leading to comparable performance of a Llama 3 8B model to a 70B model. This method is demonstrated in a [Gradio space](https://huggingface.co/spaces/davanstrien/instruction-synthesizer).
- **Marlin TGI features boost GPTQ models**: The next Hugging Face TGI update will include [Marlin features](https://x.com/danieldekok/status/1804224598721830954), supporting fast Marlin matrix multiplication for GPTQ-quantized models. This is achieved with the help of Neural Magic's Marlin kernel.
- **Ethics and Society newsletter underscores data quality**: The [Ethics and Society newsletter](https://huggingface.co/blog/ethics-soc-6) stresses the importance of data quality. It features collaborative efforts from various members and provides insights into maintaining high-quality data standards.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/argilla_io/status/1805250218184560772)">Tweet from Argilla (@argilla_io)</a>: 📢 Another big announcement: Argilla 2.0 rc!  What does it mean for AI builders?  🤺 Unified framework for feedback collection  🐍 New Python SDK to work with datasets, including a new @huggingface da...</li><li><a href="https://x.com/osanseviero/status/1803324863492350208)">Tweet from Omar Sanseviero (@osanseviero)</a>: Microsoft just silently dropped Florence  👀Vision model that can tackle many vision tasks (captioning, detection, region proposal, OCR) 🤏Small models (200M and 800M) with ~quality to models 100x lar...</li><li><a href="https://x.com/mervenoyann/status/1805265940134654424)">Tweet from merve (@mervenoyann)</a>: Fine-tune Florence-2 on any task 🔥  Today we release a notebook and a walkthrough blog on fine-tuning Florence-2 on DocVQA dataset @andi_marafioti @skalskip92   Keep reading ⇓</li><li><a href="https://x.com/reach_vb/status/1804615756568748537)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Generate GGUF quants in less than 120 seconds! ⚡  &gt; Added support for imatrix quants &gt; GGUF-split support for larger quants &gt; Automatic upload to hub &gt; Support for private and org repos  U...</li><li><a href="https://x.com/osanseviero/status/1804136001465442530)">Tweet from Omar Sanseviero (@osanseviero)</a>: Microsoft just silently (again!) dropped Instruction Pre-Training!  👀Augment pretraining datasets generating instructions 🦙A Llama 3 8B with comparable performance to 70B! 🔥General+domain models (m...</li><li><a href="https://x.com/vanstriendaniel/status/1804078257488495099)">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Instruction pre-training is a new approach that enhances LLM pretraining by using instruction-response pairs from an instruction synthesizer instead of raw data.  Explore this method in this @gradio S...</li><li><a href="https://x.com/danieldekok/status/1804224598721830954)">Tweet from Daniël de Kok (@danieldekok)</a>: 🐬More Marlin features coming to the next @huggingface TGI release: support for using existing GPTQ-quantized models with the fast Marlin matrix multiplication kernel.  ⚡This feature is made possible ...</li><li><a href="https://x.com/eustachelb/status/1805262952913858919)">Tweet from Eustache Le Bihan (@eustachelb)</a>: Distil-Whisper goes multilingual!! 🤗  The French distilled version of Whisper is here! 🇫🇷 As accurate as large-v3, faster than tiny. The best of both worlds! 🚀  Check out the details below ⬇️</li><li><a href="https://x.com/_philschmid/status/1805593591223398832)">Tweet from Philipp Schmid (@_philschmid)</a>: Embedding models are crucial for successful RAG applications, but they&#39;re often trained on general knowledge! Excited to share an end-to-end guide on how to Train and Deploy open Embeddings models...</li><li><a href="https://x.com/FrG_FM/status/1803703761119871122)">Tweet from F-G Fernandez (@FrG_FM)</a>: Xavier & @osanseviero presenting the robotics initiatives of @huggingface 🤗 (including LeRobot led by none other than @RemiCadene) at #AIDev by @linuxfoundation  Looking forward to the day when we re...</li><li><a href="https://x.com/RisingSayak/status/1805521415543697582)">Tweet from Sayak Paul (@RisingSayak)</a>: Were you aware that we have a dedicated guide on different prompting mechanisms to improve the image generation quality? 🧨  Takes you through simple prompt engineering, prompt weighting, prompt enhan...</li><li><a href="https://x.com/evijitghosh/status/1805312283628761446)">Tweet from Avijit Ghosh (@evijitghosh)</a>: The quarterly @huggingface Ethics and Society newsletter is out! Had so much fun collabing on this with @frimelle and supported by the ethics regulars. The theme for this quarter&#39;s newsletter is t...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255239401722875904)** (245 messages🔥🔥): 

- **Struggles with VSCode Coding Assistants**: A user experienced issues with Codiumate crashing mid-coding task, leading to frustration with coding assistants for VSCode. They expressed a need for a reliable solution that examines files and generates fixes without failing.
- **AI API Platform for Testing and Development**: A member proposed building an AI-driven platform to automate testing and API code generation, sharing a [survey](https://forms.gle/yAfGjUtNTnf5mASK7) to gather feedback. They seek full-stack developers and prompt engineers to contribute to the project.
- **Phi-3-Mini-128K-Instruct Model Highlights**: The Phi-3-Mini-128K-Instruct model, a lightweight and state-of-the-art open model by Microsoft, has been showcased on [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct). It supports longer token contexts and undergoes advanced post-training processes to enhance instruction following and safety.
- **Mozilla Builders Competition and Collaboration**: Members discussed teaming up for the Mozilla Builders competition, which requires creating AI projects that run locally. Relevant resources and guidelines were shared for interested participants.
- **Optimizing Stable Diffusion Inference**: Users discussed methods to speed up Stable Diffusion inference, with suggestions including using the Accelerate library and the [stable-fast](https://github.com/chengzeyi/stable-fast) framework for significant performance improvements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap">Claude 3.5 struggle too?! The $Million dollar challenge</a>: The million dollar ARC AGI challengeGet free HubSpot report of how to do AI data analysis project: https://clickhubspot.com/d30🔗 Links- Follow me on twitter...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://app.tweetscout.io/">no title found</a>: no description found</li><li><a href="https://future.mozilla.org/builders/">Mozilla Builders</a>: no description found</li><li><a href="https://tenor.com/view/hardest-choides-thanos-avengers-strongest-wills-gif-15279882">Hardest Choides Thanos GIF - Hardest Choides Thanos Avengers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/thom_wolf/status/1805244710258106369">Tweet from Thomas Wolf (@Thom_Wolf)</a>: a 3.25B params quantized gemini running locally in coming Google Chrome with less than 100ms latency while using less than 2GB of ram  that&#39;s less ram usage than many of my current Chrome page alr...</li><li><a href="https://tenor.com/view/beach-vacation-artem-gif-26266521">Beach Vacation GIF - Beach Vacation Artem - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/chengzeyi/stable-fast">GitHub - chengzeyi/stable-fast: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs.</a>: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs. - chengzeyi/stable-fast</li><li><a href="https://app.tweetscout.io">no title found</a>: no description found</li><li><a href="https://tenor.com/view/thanos-memoji-gif-23490017">Thanos Memoji GIF - Thanos Memoji - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forms.gle/yAfGjUtNTnf5mASK7">An end-to-end AI API Platform + Current Challenges with Testing &amp; Development</a>:  Description: We are building an AI-driven platform to help developers and others test APIs, UI, and anything use AI automation. We are not limited to just testing you can also talk about development ...</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: a research paper for generative cartoon interpolation</a>: a research paper for generative cartoon interpolation - ToonCrafter/ToonCrafter</li><li><a href="https://huggingface.co/spaces/hpcai-tech/open-sora">Open Sora - a Hugging Face Space by hpcai-tech</a>: no description found</li><li><a href="https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0">Open Sora Plan V1.0.0 - a Hugging Face Space by LanguageBind</a>: no description found</li><li><a href="https://t2v-turbo.github.io/">T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback
  </a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1255381029498130544)** (3 messages): 

- **Naive Bayes Algorithm on Kaggle**: A user shared a [link to a Kaggle code notebook](https://www.kaggle.com/code/rauf111/naive-bayes-algorithm) that explores the Naive Bayes algorithm. The link points to a resource for studying this machine learning algorithm.

- **InfiniAttention Reproduction Progress**: A user is working on a 95% reproduction of the **InfiniAttention** paper. They mentioned needing to fix the vanishing gradient issue and to run one final experiment to complete their work.

**Link mentioned**: <a href="https://www.kaggle.com/code/rauf111/naive-bayes-algorithm">Naive Bayes Algorithm</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255297047490461717)** (13 messages🔥): 

- **Optimized RAG Systems with LlamaIndex and DSPy**: An article on **Medium** provides a deep dive into building optimized Retrieval-Augmented Generation (RAG) systems using LlamaIndex and DSPy. [Building Optimized RAG Systems](https://medium.com/ai-advances/building-optimized-retrieval-augmented-generation-rag-systems-with-llamaindex-and-dspy-cacaf7f7089f)
  
- **AI Canon: Curated Modern AI Resources**: A blog post from a16z shares a curated list of resources dubbed the "AI Canon," useful for both beginners and experts in AI. It includes foundational papers, practical guides, and technical resources. [AI Canon](https://a16z.com/ai-canon/)
  
- **Hand Gesture Media Player Controller Demo**: A YouTube video demo showcases a Python-based hand gesture media player controller project. *"Check out this cool project I've been working on - a Hand Gesture Media Player Controller using Python!"* [Hand Gesture Media Player Controller Demo](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM)
  
- **Protein Design Advances Noted in Nature**: A **Nature** article discusses advances in protein design, noting challenges with traditional physics-based methods and highlighting the breakthroughs achieved with AlphaFold2. [Protein Design](https://www.nature.com/articles/s41586-024-07601-y)
  
- **Few-Shot Prompting with Tool Calling in Langchain**: An article discusses using few-shot prompting with tool calling in Langchain for improved AI model performance. [Few-Shot Prompting](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/freddyaboulton/gradio-llamma-cpp">Gradio Llamma Cpp - a Hugging Face Space by freddyaboulton</a>: no description found</li><li><a href="https://www.deeplearning.ai/the-batch/coding-agents-are-evolving-from-novelties-to-widely-useful-tools/">Coding Agents Are Evolving From Novelties to Widely Useful Tools</a>: On Father’s Day last weekend, I sat with my daughter to help her practice solving arithmetic problems...</li><li><a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">Hand Gesture Media Player Controller Demo</a>: Hey everyone! 👋 Check out this cool project I&#39;ve been working on - a Hand Gesture Media Player Controller using Python! 🎮🖐️So , I&#39;ve built a Python-based ...</li><li><a href="https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf">papers-we-love/machine_learning/General-self-similarity--an-overview.pdf at main · papers-we-love/papers-we-love</a>: Papers from the computer science community to read and discuss. - papers-we-love/papers-we-love</li><li><a href="https://a16z.com/ai-canon/">AI Canon | Andreessen Horowitz</a>: A curated list of resources we’ve relied on to get smarter about modern AI, including generative AI, LLMs, and transformer models.</li><li><a href="https://drive.google.com/file/d/1DYL8jvuE49fN3bGVfFydesJdxKMaBBvh/view">Final EMW 2023 - Macro Keynote (06.28.23).pdf</a>: no description found</li><li><a href="https://t.me/RYIUNITY/197736">Onlyone Dennis in RYI UNITYDEFI OFFICIAL CHANNEL</a>: 📣Buckle up everyone &#33; This Monday at 1 pm, join our X contest and earn RYIU&#33;  - 0-300 followers: earns 100 RYIU/tweet - 300-600 followers: earns200 RYIU/tweet - 600-1000 followers: earns300 R...</li><li><a href="https://www.nature.com/articles/s41586-024-07601-y">Computational design of soluble and functional membrane protein analogues - Nature</a>: A deep learning approach enables accurate computational design of soluble and functional&amp;nbsp;analogues of membrane proteins, expanding the soluble&amp;nbsp;protein fold space and facilitating new...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255236112973037629)** (66 messages🔥🔥): 

- **Exploring Custom Byte Encoding in LLMs**: In a detailed technical discussion, members explored the use of custom byte encoding for LLMs, predicting sequences in UTF-32. The conversation included potential issues with floating point accuracy and robustness, with one member expressing skepticism about its effectiveness but remaining curious about the results.

- **Hand Gesture Media Player Controller Demo**: A member shared a [YouTube video](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM) demonstrating a hand gesture-based media player controller using Python.

- **Bioinformatics Tools and Projects**: Members shared various bioinformatics tools and projects, including [PCALipids](https://github.com/membrane-systems/PCAlipids), a tool for PCA and related analyses of lipid motions, and other GitHub projects such as [embedprepro-lib](https://github.com/Elma-dev/embedprepro-lib) and [PixUP-Upscale](https://github.com/U-C4N/PixUP-Upscale).

- **New Text Analysis CLI Tool Released**: A member announced the release of a new text analysis command-line tool called [embedprepro](https://github.com/Elma-dev/embedprepro-lib), designed for generating text embeddings, clustering, and visualization, aimed at researchers and developers.

- **Dataset for Optimizing LLMs for RLHF**: A member released the Tasksource-DPO-pairs dataset on Hugging Face, [Tasksource](https://huggingface.co/datasets/tasksource). This dataset is tailored for optimizing LLMs for Reward Learning from Human Feedback (RLHF) and focuses on fine-grained linguistic reasoning tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://storiagl.web.app/">StorIA</a>: no description found</li><li><a href="https://x.com/eggwens/status/1806016129875476886">Tweet from Egg (@eggwens)</a>: Here is the live demo for pet psychic attached are the sample codes made in react with sample styling:  With Pet Psychic Scheduler, you can: 🔮 Book psychic readings for your pets ✨ Check daily mood f...</li><li><a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">Hand Gesture Media Player Controller Demo</a>: Hey everyone! 👋 Check out this cool project I&#39;ve been working on - a Hand Gesture Media Player Controller using Python! 🎮🖐️So , I&#39;ve built a Python-based ...</li><li><a href="https://huggingface.co/spaces/KoboldAI/Koboldcpp-Tiefighter/blob/main/Dockerfile">Dockerfile · KoboldAI/Koboldcpp-Tiefighter at main</a>: no description found</li><li><a href="https://github.com/U-C4N/PixUP-Upscale/">GitHub - U-C4N/PixUP-Upscale</a>: Contribute to U-C4N/PixUP-Upscale development by creating an account on GitHub.</li><li><a href="https://github.com/azmiord/project">GitHub - azmiord/project</a>: Contribute to azmiord/project development by creating an account on GitHub.</li><li><a href="https://github.com/Elma-dev/embedprepro-lib">GitHub - Elma-dev/embedprepro-lib</a>: Contribute to Elma-dev/embedprepro-lib development by creating an account on GitHub.</li><li><a href="https://github.com/membrane-systems/PCAlipids">GitHub - membrane-systems/PCAlipids: Scripts for PCA and related analyses of lipid motions</a>: Scripts for PCA and related analyses of lipid motions - membrane-systems/PCAlipids</li><li><a href="https://github.com/bigsk1/voice-chat-ai">GitHub - bigsk1/voice-chat-ai: 🎙️ Speak with AI - Run locally using ollama or use OpenAI - XTTS or OpenAI Speech or ElevenLabs</a>: 🎙️ Speak with AI - Run locally using ollama or use OpenAI - XTTS or OpenAI Speech or ElevenLabs - bigsk1/voice-chat-ai</li><li><a href="https://huggingface.co/datasets/tasksource/tasksource_dpo_pairs">tasksource/tasksource_dpo_pairs · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/sileod/tasksource/blob/main/tasks.md">tasksource/tasks.md at main · sileod/tasksource</a>: Datasets collection and preprocessings framework for NLP extreme multitask learning - sileod/tasksource
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255300766739206214)** (15 messages🔥): 

- **Thank Alex for recommending the work**: A member expresses gratitude for Alex’s post, which brought attention to a particular work that no one had recommended before. 

- **Discussion on data leakage in LLMs**: Eleuther AI mentioned papers discussing benchmark dataset leakage in LLMs. They shared links to [one paper](https://arxiv.org/abs/2404.18824) investigating the phenomenon and [another paper](https://arxiv.org/abs/2405.00332) addressing detection of benchmark data leakage.

- **Terminator architecture introduced**: A new architecture called "Terminator" was shared from a [Twitter link](https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w) and further detailed in a [GitHub repository](https://github.com/hyperevolnet/Terminator/blob/main/models/modules/hyperzzw.py). This architecture notably lacks residuals, dot product attention, and normalization.

- **Saturation in LLM leaderboards**: Highlighting a concern, a member shared a HuggingFace link to a blog about the saturation in LLM leaderboards, indicating community attention to the issue. The link to the blog post: [HuggingFace LLM Leaderboard Blog](https://huggingface.co/spaces/open-llm-leaderboard/blog).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w">Tweet from Alex Yanko 🇺🇦 (@LeopolisDream)</a>: Welcome the new architecture:   Terminator   No residuals, no dot product attention, no normalization...   https://arxiv.org/pdf/2401.17948</li><li><a href="https://github.com/hyperevolnet/Terminator/blob/main/models/modules/hyperzzw.py">Terminator/models/modules/hyperzzw.py at main · hyperevolnet/Terminator</a>: Contribute to hyperevolnet/Terminator development by creating an account on GitHub.</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion...</li><li><a href="https://arxiv.org/abs/2405.00332">A Careful Examination of Large Language Model Performance on Grade School Arithmetic</a>: Large language models (LLMs) have achieved impressive success on many benchmarks for mathematical reasoning. However, there is growing concern that some of this performance actually reflects dataset c...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1255397885210529902)** (1 messages): 

- **Perturbed Attention Guidance now in Diffusers**: HuggingFace has announced support for **Perturbed Attention Guidance (PAG)** in their `diffusers` library, enhancing image generation quality without additional training. [Check out the update](https://github.com/huggingface/diffusers/issues/8704) and kudos to the contributor who led the integration.

**Link mentioned**: <a href="https://github.com/huggingface/diffusers/issues/8704">PAG is now supported in core 🤗 · Issue #8704 · huggingface/diffusers</a>: Hello folks! #7944 introduced support for Perturbed Attention Guidance (PAG) which enhances image generation quality training-free. Generated Image without PAG Generated Image with PAG Check out th...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255246614692368405)** (3 messages): 

- **Evaluation Error with `detection_util` in Folder**: Someone pointed out that the **`evaluate` function** has issues locating `detection_util` if it is in a folder within a space. This causes problems during evaluation as the function cannot find the required files.
- **Hand Gesture Media Player Controller Demo**: A user shared a [YouTube video](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM) demonstrating a "Hand Gesture Media Player Controller" made with Python. They encouraged others to check out their cool project.
- **Developing a Handwritten Table Data Pipeline**: Someone requested assistance in creating a pipeline for identifying data in handwritten tables. They mentioned trying GPT-Vision, but it did not meet their expectations.

**Link mentioned**: <a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">Hand Gesture Media Player Controller Demo</a>: Hey everyone! 👋 Check out this cool project I&#39;ve been working on - a Hand Gesture Media Player Controller using Python! 🎮🖐️So , I&#39;ve built a Python-based ...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1255236404712046733)** (5 messages): 

- **Seek advice on multilingual model distillation**: *Looking for suggestions on knowledge distillation of a multilingual model for a single language*. 

- **Named entity recognition using RAG**: A member seeks advice on using Retrieval-Augmented Generation (RAG) for recognizing named entities in long documents. Considering using SSM like Mamba for managing document length, another member suggests BM25 for keyword-oriented search and provides a [GitHub link](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb) for more information.

- **Developing a pipeline for handwritten tables**: A member wants to create a pipeline for identifying data in handwritten tables and finds that GPT-Vision is not meeting expectations. Asking for advice on more effective methods.

- **Experiences with LLM knowledge editing sought**: A query about hands-on experiences with LLM knowledge editing and its deployment for simpler tasks like translation was raised.

**Link mentioned**: <a href="https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb">semantic-search-with-amazon-opensearch/Module 1 - Difference between BM25 similarity and Semantic similarity.ipynb at main · aws-samples/semantic-search-with-amazon-opensearch</a>: Contribute to aws-samples/semantic-search-with-amazon-opensearch development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1255403601770319944)** (3 messages): 

- **Exploring Knowledge Distillation for Multilingual Models**: A member inquired about performing **knowledge distillation** for a multilingual model focusing on a single language. Another member suggested trying **SpeechBrain** on HuggingFace as a possible solution.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255237341052670063)** (327 messages🔥🔥): 

- **Generate Music with AI on RateYourMusic**: Members discussed generating songs and lyrics of any musician by using IDs from the RateYourMusic website. One member tried this method and confirmed its effectiveness, calling it "hilarious".
  
- **Open Model Initiative Controversy**: There's a significant discussion about LAION's withdrawal from the Open Model Initiative and their involvement in datasets with problematic content. A member speculated that LAION might have been excluded for sharing non-synthetic datasets, but others believed it was a voluntary decision.

- **Synthetic vs. Non-Synthetic Data Debate**: Several members debated the inclusion of NSFW (Not Safe For Work) content in datasets for training AI models. Concerns included moral and PR implications, with some advocating for excluding NSFW content and others critical of heavy-handed safety measures on models like SD3.

- **GPU and Workstation Discussions**: Members compared different GPUs, including A6000s, 3090s, and P40s for AI training, discussing the trade-offs in VRAM, cost, and performance. They also talked about practical aspects like on-system cooling, fitting models in single VRAM vs. sharding, and specific models' efficiency and compatibility with certain GPUs.

- **ASIC Chips for Transformers**: There's an intriguing discussion about Etched's Sohu, a specialized chip for transformer models claimed to be faster and cheaper than GPUs. Some members doubted its practicality due to its apparent inflexibility, which might limit its use to only specific types of AI models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>: Mamba, an architecture with RNN-like token mixer of state space model (SSM), was recently introduced to address the quadratic complexity of the attention mechanism and subsequently applied to vision t...</li><li><a href="https://www.etched.com/announcing-etched">Etched is Making the Biggest Bet in AI</a>: no description found</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Tweet from Bryan Johnson /dd (@bryan_johnson)</a>: Excited to invest in @Etched&#39;s $120 million series A.    10x cheaper AI models will allow us to solve aging 100x faster.  Quoting Etched (@Etched)   Meet Sohu, the fastest AI chip of all time.  Wi...</li><li><a href="https://tenor.com/view/theoffice-stevecarrell-michaelscott-no-godplease-gif-4593632">No GIF - Theoffice Stevecarrell Michaelscott - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/tenstorrent/tt-firmware">GitHub - tenstorrent/tt-firmware: Tenstorrent Firmware repository</a>: Tenstorrent Firmware repository. Contribute to tenstorrent/tt-firmware development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/jim-halpert-the-office-confused-gif-25227530">Jim Halpert GIF - Jim Halpert The - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://shop.lambdalabs.com/gpu-workstations/vectorone/customize)">Lambda | GPU Compute for AI</a>: The GPU Cloud built for AI developers. Featuring on-demand &amp; reserved cloud NVIDIA H100, NVIDIA H200 and NVIDIA Blackwell GPUs for AI training &amp; inference.
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255240235139334155)** (7 messages): 

- **Debate on Poisoning Models**: A member expressed concern that someone "actively encouraged poisoning models," indicating controversies in model training ethics.
- **AIW+ Problem Harder but Solvable**: Another member clarified that the AIW+ problem, although more complex than simple AIW, is still a common-sense problem and solvable. They suggested checking the paper’s supplementary material for the solution. 
- **Caution Against Manual Evaluation**: It was advised against manual evaluation, as it can be highly misleading due to inconsistent results from repeated trials. The recommendation was to use systematic prompt variations and conduct at least 20 trials per prompt variation.
- **Disagreement Over AIW+ Solution**: A member disputed the provided solution for the AIW+ problem, stating it was incorrect and ambiguous due to unaccounted familial relationships. They also remarked that model agreement with this solution does not eliminate the ambiguity.

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1255332843534422038)** (2 messages): 

- **EleutherAI at ICML 2024**: EleutherAI members shared their excitement about presenting multiple papers at ICML 2024, covering a range of topics from classifier-free guidance to the societal impacts of open foundation models. Links to their papers, such as [Stay on topic with Classifier-Free Guidance](https://arxiv.org/abs/2306.17806) and [Neural Networks Learn Statistics of Increasing Complexity](https://arxiv.org/abs/2402.04362), were provided to keep the community informed.

- **Understanding Memorization in LMs**: A member highlighted their work on better understanding memorization in language models, introducing a taxonomy to differentiate between recitation, reconstruction, and recollection. They shared a [preprint](https://arxiv.org/abs/2406.17746) and a [Twitter thread](https://x.com/nsaphra/status/1805964526405161457) to elaborate on their findings and its implications for copyright, privacy, and generalization.

**Link mentioned**: <a href="https://x.com/nsaphra/status/1805964526405161457)">Tweet from Naomi Saphra (@nsaphra)</a>: Humans don&#39;t just &#34;memorize&#34;. We recite poetry drilled in school. We reconstruct code snippets from more general knowledge. We recollect episodes from life. Why treat memorization in LMs u...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255269463402483853)** (98 messages🔥🔥): 

- **Finding the best multimodal models**: A member inquired about locating top-performing multimodal models, specifically Image+Text to Text models, and shared a [link to Huggingface](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer) for reference. This helped others looking for similar resources.

- **ICML Social Thread kicks off**: A social thread for [ICML](https://icml.cc/Conferences/2024) in Vienna, Austria, was started to coordinate meetups and events. Members discussed logistics and planned gatherings, showing enthusiastic participation.

- **Goldfinch model details shared**: Information about the hybrid Goldfinch model, featuring an improved Llama-style transformer layer paired with Finch B2, was shared. Members exchanged [links and DM’d more details](https://discord.com/channels/729741769192767510/1103039376184852622/1246105198963982348) and discussed technical specifics.

- **Documenting OOD input handling in LLMs**: A paper concerning how neural network predictions behave with out-of-distribution (OOD) inputs was discussed, specifically [this arxiv link](https://arxiv.org/abs/2310.00873). This sparked a discussion on whether LLMs behave similarly and the implications for Bayesian DL.

- **Request for vision model recommendations**: A member requested suggestions for vision models capable of performing RAG on PDFs with image data. The conversation unfortunately did not yield any specific model recommendations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer">LeaderboardExplorer - a Hugging Face Space by leaderboards</a>: no description found</li><li><a href="https://tenor.com/view/spongebob-eating-chewing-popcorn-gif-16655546">Spongebob Eating GIF - Spongebob Eating Chewing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-hi-hello-close-up-kitten-gif-16709314">Cat Hi GIF - Cat Hi Hello - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2310.00873">Deep Neural Networks Tend To Extrapolate Predictably</a>: Conventional wisdom suggests that neural network predictions tend to be unpredictable and overconfident when faced with out-of-distribution (OOD) inputs. Our work reassesses this assumption for neural...</li><li><a href="https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009">Worried Scared GIF - Worried Scared Oh No - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-cat-crazy-crazy-cat-insane-cat-going-insane-gif-5752628082217795406">Cat Cat Crazy GIF - Cat Cat crazy Crazy cat - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255280862199550054)** (114 messages🔥🔥): 

- **Comparative Evaluation of Synquid**: Members debated the merits of the paper [Synquid](https://arxiv.org/abs/2406.16450), highlighting well-thought-out experiments but expressing mixed feelings about certain missing baselines like "no activation function." One member noted, "It will also score lower on their complexity measure at random initialization," emphasizing the importance of this baseline in their analysis.

- **NRS Framework in Paper Critique**: The discussion inspected the hypothesis testing in a paper on neural network initialization and inductive biases. One member stated, *"The complexity at initialization correlating with downstream performance on tasks of similar complexity,"* while others critiqued the reinterpretations of existing work, specifically their stance on random sampling of low-loss solutions.

- **Implementation of Multimodal Metrics and Experimental Validation**: Members analyzed a paper on JEST, emphasizing joint example selection for data curation in multimodal contrastive learning. They discussed the significant efficiency gains claimed in the paper, noting the approach surpasses state-of-the-art models with much fewer iterations and computational requirements.

- **Homomorphic Encryption and LLMs**: Members briefly touched on the speculative nature of using homomorphic encryption for large language models, as discussed in a [Zama AI blog post](https://www.zama.ai/post/chatgpt-privacy-with-homomorphic-encryption). The discussion noted skepticism about the practical advancements in homomorphic encryption for real-time applications.

- **Generalization and Grokking in Transformers**: Members debated whether grokking and generalization are being confused in a paper, pointing out that *"grokking refers specifically to a sudden shift in eval performance after a long flat period of training."* They critiqued the paper's methods and the historical context of generalization research.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.17711">Data curation via joint example selection further accelerates multimodal learning</a>: Data curation is an essential component of large-scale pretraining. In this work, we demonstrate that jointly selecting batches of data is more effective for learning than selecting examples independe...</li><li><a href="https://arxiv.org/abs/2406.16450">Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers</a>: State-of-the-art results in large language models (LLMs) often rely on scale, which becomes computationally expensive. This has sparked a research agenda to reduce these models&#39; parameter count an...</li><li><a href="https://arxiv.org/abs/2406.17224">Large Language Models are Interpretable Learners</a>: The trade-off between expressiveness and interpretability remains a core challenge when building human-centric predictive models for classification and decision-making. While symbolic rules offer inte...</li><li><a href="https://www.zama.ai/post/chatgpt-privacy-with-homomorphic-encryption">Making ChatGPT Encrypted End-to-end</a>: With Homomorphic Encryption, you can use LLMs without revealing your personal data.</li><li><a href="https://arxiv.org/abs/2406.16747">Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers</a>: Accommodating long sequences efficiently in autoregressive Transformers, especially within an extended context window, poses significant challenges due to the quadratic computational complexity and su...</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>: We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types...</li><li><a href="https://openreview.net/forum?id=siCt4xZn5Ve">What Happens after SGD Reaches Zero Loss? --A Mathematical Framework</a>: Understanding the implicit bias of Stochastic Gradient Descent (SGD) is one of the key challenges in deep learning, especially for overparametrized models, where the local minimizers of the loss...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1255327906629025832)** (15 messages🔥): 

- **Self-Attention confirmed as (hetero)associative memory model**: One member clarified that self-attention functions as a (hetero)associative memory model, pointing out the connection to associative memory frameworks like Hopfield networks. They referenced the paper [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) to support this claim.
- **LeCun's perspective on Transformers**: A discussion referenced Yann LeCun's description of transformers as "associative memories." This is tied to the idea that self-attention mechanisms have memory model characteristics.
- **Hopfield Networks paper sparks interest**: The paper [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) generated significant discussions, with mentions of its authors and the ideas it presents about modern continuous Hopfield networks (MCHNs) relating closely to self-attention.
- **Criticism of "Is All You Need" papers with exceptions**: One member expressed disdain for papers titled "Is All You Need" but acknowledged that some, like [Hopfield Networks is All You Need](https://arxiv.org/abs/2309.08632), present exceptional value. The user cited its innovative treatment of grokking and overall contributions to the field.
- **Hopfield layers as single-step attention**: Clarification on how Hopfield layers work in practice within neural networks was provided, noting that memorization happens during pre-training and retrieval occurs in the forward pass. Each operation is dictated as a single step of a Hopfield network, emphasizing the practical application in self-attention mechanisms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2008.02217">Hopfield Networks is All You Need</a>: We introduce a modern Hopfield network with continuous states and a corresponding update rule. The new Hopfield network can store exponentially (with the dimension of the associative space) many patte...</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>: Inspired by recent work demonstrating the promise of smaller Transformer-based language models pretrained on carefully curated data, we supercharge such approaches by investing heavily in curating a n...</li><li><a href="https://arxiv.org/abs/2202.04557">Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models</a>: A large number of neural network models of associative memory have been proposed in the literature. These include the classical Hopfield networks (HNs), sparse distributed memories (SDMs), and more re...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1255240451045326870)** (4 messages): 

- **SAEs identified to recover linear features**: A member shared a research report showing that "SAEs recover linear features from an overcomplete basis," and highlighted that using "a single layer autoencoder with an L1 penalty on hidden activations" can identify features beyond minimizing loss. They linked to the [LessWrong post](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) and acknowledged feedback from other researchers.

- **Interest in toy models for SAE testing**: The same member expressed interest in exploring toy models to test SAEs, inspired by another post emphasizing the importance of feature geometry beyond the superposition hypothesis. They shared another [LessWrong post](https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis) on the subject, which discusses structural information in neural networks' feature vectors.

- **Excitement for multilinguality and safety work**: A member shared a [Twitter link](https://x.com/yong_zhengxin/status/1805616252490236235?s=46) about new work on multilinguality, safety, and mechanistic interpretation, highlighting that "DPO training in only English can detoxify LLM in many other languages." They also provided a link to the associated [research paper on arXiv](https://arxiv.org/abs/2406.16235).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/yong_zhengxin/status/1805616252490236235?s=46">Tweet from Zheng-Xin Yong (Yong) (@yong_zhengxin)</a>: 🔥New work on multilinguality + safety + mech interp!  We show that DPO training in only English can detoxify LLM in many other languages.  We also give a mechanistic explanation on how cross-lingual ...</li><li><a href="https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis">SAE feature geometry is outside the superposition hypothesis — LessWrong</a>: Written at Apollo Research • Summary: Superposition-based interpretations of neural network activation spaces are incomplete. The specific locations…</li><li><a href="https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition.">[Interim research report] Taking features out of superposition with sparse autoencoders — LessWrong</a>: We&#x27;re thankful for helpful comments from Trenton Bricken, Eric Winsor, Noa Nabeshima, and Sid Black.  …
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1255334051980771390)** (16 messages🔥): 

- **AMD MI300X Challenges Nvidia's GPU Dominance**: A post about AMD's Radeon Instinct MI300X highlights its aim to compete with Nvidia's GPU compute market lead. While AMD's software ecosystem ROCm trails Nvidia's CUDA, the MI300X represents an effort to overcome this hardware gap independently. [Full post](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/).

- **Etched Introduces Transformer ASIC**: Etched's [new Transformer ASIC chips](https://www.etched.com/) claim to run AI models significantly faster and cheaper than GPUs by etching transformer architecture directly into silicon. The chip promises applications like real-time voice agents and the capability to run trillion-parameter models.

- **Skepticism Around Etched's ASIC Claims**: Users expressed doubts about the practical advantages of Etched's ASICs, particularly whether etching just the architecture rather than also including the weights would deliver the promised performance gains. The discussion highlighted the competition and rapid advancement in AI hardware.

- **Etched Secures Major Investment**: Bryan Johnson announced his excitement to invest in Etched's $120 million series A, citing the company's claim that their Sohu chip can run AI models 10x cheaper and replace 160 Nvidia H100 GPUs with one 8xSohu server. [Tweet link](https://x.com/bryan_johnson/status/1805629207374086490).

- **Future of AI Chips Debated**: Users debated the future role of specialized AI chips like ASICs compared to GPUs, with mentions of [the industry's direction](https://www.pixelstech.net/article/1719027344-The-Future-of-AI-Chips-Might-Not-Be-GPU) towards dedicated hardware accelerators. The potential for rapid changes in model architectures and the flexibility of tensor cores were highlighted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/">Testing AMD&#8217;s Giant MI300X</a>: Editor Note (6/26/2024): We have rephrased the acknowledgment section to make more clear that we got no direct support from AMD on this article. Our testing is fully independent, and AMD did not ha…</li><li><a href="https://www.etched.com/">Etched | The World&#x27;s First Transformer ASIC</a>: Transformers etched into silicon. By burning the transformer architecture into our chips, we&#x27;re creating the world&#x27;s most powerful servers for transformer inference.</li><li><a href="https://www.youtube.com/watch?v=zh6REnqwXe4">AI Chip Startup Etched Aims to Take On Nvidia</a>: AI chip startup Etched raised $120 million to expand manufacturing of its specialized chip that it boasts will rival Nvidia’s products. Etched CEO Gavin Uber...</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Tweet from Bryan Johnson /dd (@bryan_johnson)</a>: Excited to invest in @Etched&#39;s $120 million series A.    10x cheaper AI models will allow us to solve aging 100x faster.  Quoting Etched (@Etched)   Meet Sohu, the fastest AI chip of all time.  Wi...</li><li><a href="https://www.pixelstech.net/article/1719027344-The-Future-of-AI-Chips-Might-Not-Be-GPU">The Future of AI Chips Might Not Be GPU | Pixelstech.net</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1255625274389954710)** (1 messages): 

- **New user seeks help on Triton issue**: A new member introduced themselves and shared an issue they opened in the Triton repo. They are looking for pointers on how to add a `pow` function in `python.triton.language.core` and provided a [link to the issue](https://github.com/triton-lang/triton/issues/4190).

**Link mentioned**: <a href="https://github.com/triton-lang/triton/issues/4190">How to add a pow function in python.triton.language.core? · Issue #4190 · triton-lang/triton</a>: I tried to use pow operation in a triton.jitted function as: output = x + y**3 ^ However got AttributeError(&quot;&#39;tensor&#39; object has no attribute &#39;__pow__&#39;&quot;). In file python/trit...

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1255281600376209510)** (6 messages): 

- **PyTorch Documentary Premiers**: Members shared a [YouTube link](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s) to the "PyTorch Documentary Virtual Premiere: Live Stream" featuring key players from the project's early days to the present. This was posted repeatedly by multiple users to emphasize its importance.
- **Goat Emoji Hype**: A member reacted to the PyTorch Documentary link with a goat emoji (*🐐*), symbolizing excitement and hype. The reaction was noted and mirrored by another member to highlight this sentiment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=EjgTv6aSeqk&t=8...]">PyTorch Documentary Virtual Premiere: Live Stream</a>: Join us for the official release of the PyTorch Documentary! Hear from key players in the project, from the early days to the present.</li><li><a href="https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s">PyTorch Documentary Virtual Premiere: Live Stream</a>: Join us for the official release of the PyTorch Documentary! Hear from key players in the project, from the early days to the present.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1255414516959940608)** (1 messages): 

- **Adam-mini optimizer reduces memory usage**: Adam-mini is proposed as an optimizer that offers equivalent or better performance than AdamW while using 45% to 50% less memory. The [GitHub repository](https://github.com/zyushun/Adam-mini) contains the code and details of the implementation.

**Link mentioned**: <a href="https://github.com/zyushun/Adam-mini">GitHub - zyushun/Adam-mini: Code for the paper: Adam-mini: Use Fewer Learning Rates To Gain More</a>: Code for the paper: Adam-mini: Use Fewer Learning Rates To Gain More - zyushun/Adam-mini

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255335011754704979)** (38 messages🔥): 

- **Raw Kernel for Linear Algebra in PyTorch**: A user shared a [link to the raw kernel](https://github.com/pytorch/pytorch/blob/b7e7a4cb01de394af7686ab6feb216a8a5c716bb/aten/src/ATen/native/LinearAlgebra.cpp#L3476) in the PyTorch repository. This kernel is located in the native linear algebra section of the code.
- **Subclass dtype Issue in PyTorch**: Members discussed issues with tensor subclasses not reflecting their actual `dtype`, complicating compatibility and usability. Marksaroufim encouraged filing an issue on PyTorch and suggested looking into internal improvements.
- **Open Source Contributions' Value**: Locknit3 questioned whether open source contributions help in job searches, sparking a debate. Gau.nernst and kashimoo affirmed their value, mentioning instances where recruiters noted their contributions.
- **Integrating HQQ with TorchAO**: Members discussed the potential integration of HQQ with TorchAO's `quantize()` API, linking to the [HQQ optimizer](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L194-L243). They highlighted the algorithm's simplicity and suggested it could be a new baseline for INT4 quantization.
- **Low-bit Fused GEMV CUDA Kernels**: Mobicham shared that they have been developing low-bit fused GEMV CUDA kernels, outlining their flexibility and current limitations. Gau.nernst inquired about support for odd bitwidths, to which Mobicham confirmed feasibility.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1jqC53MwiW9dSiPS-a6hng_yo0ywdc3nH#scrollTo=Aj9ii4darSRA">Google Colab</a>: no description found</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L194-L243">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.3.0">Release v0.3.0 · pytorch/ao</a>: v0.3.0 Highlights We are excited to announce the 0.3 release of torchao! This release adds support for a new quantize API, MX format, FP6 dtype and bitpacking, 2:4 sparse accelerated training and b...</li><li><a href="https://github.com/pytorch/ao/blob/f172c474cbd56641bb34e73df5d61818a9d4e6e1/torchao/_models/llama/model.py#L122).">ao/torchao/_models/llama/model.py at f172c474cbd56641bb34e73df5d61818a9d4e6e1 · pytorch/ao</a>: Create and integrate custom data types, layouts and kernels with up to 2x speedups with 65% less VRAM for inference and support for training - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch/blob/b7e7a4cb01de394af7686ab6feb216a8a5c716bb/aten/src/ATen/native/LinearAlgebra.cpp#L3476">pytorch/aten/src/ATen/native/LinearAlgebra.cpp at b7e7a4cb01de394af7686ab6feb216a8a5c716bb · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1255424670711287818)** (8 messages🔥): 

- **Axis setting affects HQQModelForCausalLM performance**: A user reported issues with the `HQQModelForCausalLM` related to `meta-llama/Llama-2-7b-chat-hf` and `Mistral-7B-v0.1`, specifically when setting `axis=0`, which skips using `torchao`'s int4 kernel. Another user clarified that `axis` controls the axis along which grouping is performed and affects both the perplexity/lm-eval score and inference speed due to kernel support.
- **Inference quality issues tied to HF's transformers cache implementation**: There were mentioned quality issues with [Hugging Face's Transformers](https://github.com/huggingface/transformers) cache implementation, suggesting it could be a potential source of issues with model evaluation.
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255237578945335449)** (146 messages🔥🔥): 

- **Scaling LR and update clipping controversies**: Members discussed scaling the **learning rate (LR)** and using **update clipping**. One shared, "*tbh, some form of update clipping still sounds reasonable to me,*" while another noted, "*for stabilization, we analyze loss spikes*," pointing to [AdamW's paper](https://mlfoundations.github.io/advancedml-sp23/assets/adam.pdf) and another [arXiv link](https://arxiv.org/pdf/1905.11286).
  
- **AMD vs NVIDIA system builds**: A user reported building a machine with *RDNA3 cards* and having access to big machines, while another mentioned using *A6000's* and planned to build an AMD system, reflecting on potential providers like *Azure's MI300X* instances.

- **Sohu ASIC chip could revolutionize transformers**: A member highlighted a tweet about [Sohu's new ASIC chip](https://x.com/Etched/status/1805625693113663834) boasting 500,000 tokens per second for Llama 70B, potentially replacing 160 H100 GPUs. Questions arose regarding its specialization solely for transformer models and the impact on versatility.

- **FP8 integration sparks mixed reactions**: Discussion on integrating FP8 into existing systems, balancing simplicity versus greater changes, concluded with "*optional but you’d still take the PR if it’s in decent enough shape*". The feasibility of avoiding global amax history and using local scaling was also analyzed.

- **Effective multi-node training on Lambda**: Highlighting successful deployment, "*It was not easy to set it up*," a member shared their 16 GPU multi-node training setup on Lambda with almost a 2X speedup, stating, "*It is glorious*."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Etched/status/1805625693113663834">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://github.com/karpathy/llm.c/pull/636">rolling checkpoints by karpathy · Pull Request #636 · karpathy/llm.c</a>: checkpoints are either MINOR or MAJOR and minor checkpoints get deleted with a rolling window. This is an optimization that will allow us to save state more often, but preserve disk space overall. ...</li><li><a href="https://github.com/karpathy/llm.c/pull/629/">CI Dataloader test and ptx/sass file generator by rosslwheeler · Pull Request #629 · karpathy/llm.c</a>: New CI tests file - added dataloader test and ptx/sass file generator to it. Cuda Makefile - added capability build from main Makefile. Added support for ptx and sass output files. layernorm_forwar...</li><li><a href="https://github.com/karpathy/llm.c/pull/629/files">CI Dataloader test and ptx/sass file generator by rosslwheeler · Pull Request #629 · karpathy/llm.c</a>: New CI tests file - added dataloader test and ptx/sass file generator to it. Cuda Makefile - added capability build from main Makefile. Added support for ptx and sass output files. layernorm_forwar...</li><li><a href="https://github.com/karpathy/llm.c/pull/635">On-device reductions by ngc92 · Pull Request #635 · karpathy/llm.c</a>: Moves loss calculation to backward, and ensures  we can do more on-device reductions and fewer host&lt;-&gt;device transfers. Also enables a micro-optimization, that validate does not calculate dlogit...</li><li><a href="https://github.com/warner-benjamin/optimi/blob/4542d04a3974bb3ac9baa97f4e417bda0432ad58/optimi/stableadamw.py#L28>).">optimi/optimi/stableadamw.py at 4542d04a3974bb3ac9baa97f4e417bda0432ad58 · warner-benjamin/optimi</a>: Fast, Modern, Memory Efficient, and Low Precision PyTorch Optimizers - warner-benjamin/optimi
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1255352848498167808)** (2 messages): 

- **Intel Pytorch team works on XPU support**: The Intel PyTorch team is actively working to enable **XPU (Intel GPUs)** support in stock PyTorch. They have shared an [RFC on GitHub](https://github.com/pytorch/pytorch/issues/114723) to discuss the upstreaming process and leverage Intel's advancements in GPU technology.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/issues/114723">[RFC] Intel GPU Upstreaming  · Issue #114723 · pytorch/pytorch</a>: TL;DR This RFC document aims to propose and discuss the upstreaming of Intel GPU support in PyTorch. Our focus is on leveraging Intel&#39;s advancements in GPU technology to enhance PyTorch&#39;s perf...

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255240892625850368)** (153 messages🔥🔥): 

- **Pro focus search limitations and updates**: Users discussed issues with Perplexity Pro focus search, noting that non-Pro + Reddit searches are functioning correctly. One user appreciated that their Pro search for Standard mode returned more sources than previously.

- **Claude 3.5 capabilities and availability**: Members discussed **Claude 3.5 Sonnet**'s context window being about 32k tokens for Perplexity Pro, with a preference for Claude Pro for a full 200k tokens. Claude 3.5's availability on Android was also confirmed.

- **API filters and undocumented features**: Questions were raised about API citation date filters and possible undocumented features, like a search domain filter. Users discussed whether some features are in development or available through workaround methods.

- **Debate on AI search quality and new features**: Users compared Perplexity's response quality to ChatGPT, acknowledging Perplexity's new agentic search capabilities which handle multi-step queries. Some expressed frustrations with Perplexity's summarization and source handling, suggesting it often leads to hallucination in responses.

- **Service disruptions and API status concerns**: Users reported 5xx errors with the Perplexity API, expressing frustration over the lack of a status page to check service uptime. Calls were made for better transparency and basic API management features.

**Link mentioned**: <a href="https://entertainment.slashdot.org/story/24/06/26/001222/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms">Researchers Upend AI Status Quo By Eliminating Matrix Multiplication In LLMs - Slashdot</a>: Researchers from UC Santa Cruz, UC Davis, LuxiTech, and Soochow University have developed a new method to run AI language models more efficiently by eliminating matrix multiplication, potentially redu...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255314661310726184)** (10 messages🔥): 

- **CTO Denis Yarats discusses AI Innovations**: Check out the [YouTube video](https://www.youtube.com/watch?v=gvP-DxqatLQ) to see Denis Yarats, CTO of Perplexity AI, discussing the innovative use of AI in creating high-quality search experiences. Yarats joins Lukas Biewald on Gradient Dissent to dive into this topic.
- **Titan's Waves, China's Lunar Triumph, and Volkswagen's Rivian Investment**: Perplexity AI’s [YouTube video](https://www.youtube.com/embed/HSmt6qvwuS0) explores compelling topics such as Titan's missing waves, China's lunar achievements, and Volkswagen's investment in Rivian.
- **Hot Searches on Perplexity.ai**: Explore various searches on Perplexity AI including [Intel CPU](https://www.perplexity.ai/search/IntelCPU-tz.a_Iv0TlultEIdNpQLPw), [Perplexity functionality](https://www.perplexity.ai/search/how-does-perplexity-NVeFu4LMQ0K0RdU7PrFP1A), [Hive Blockchain](https://www.perplexity.ai/search/Hive-blockchain-HZ6MEvTqRf.HQpl2wMIurA), and [5000 unit results](https://www.perplexity.ai/search/5000-Jy_3Oq3dTpqG7l11w8PSNQ).
- **Insightful Pages on Perplexity.ai**: Discover detailed pages like overcoming trauma at [this link](https://www.perplexity.ai/page/Overcoming-Trauma-and-9_3ox12FRFaMON3Zk8lezQ) and updates on Julian Assange's release [here](https://www.perplexity.ai/page/Julian-Assange-Released-cLtbci_iSxW32Xve2NgKGA).
- **Curiosities about Gravity**: Check out the in-depth search results for how gravity affects perception and related phenomena at this [link](https://www.perplexity.ai/search/If-Gravity-affects-20bEiugFSnudRflOAGYlnA).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/HSmt6qvwuS0">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=gvP-DxqatLQ">Transforming Search with Perplexity AI’s CTO Denis Yarats</a>: In this episode of Gradient Dissent, Denis Yarats, CTO of Perplexity, joins host Lukas Biewald to discuss the innovative use of AI in creating high-quality, ...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255366046337208432)** (5 messages): 

- **Confusion about Perplexity AI usage**: A member expressed confusion, asking for clarification on what another user meant. No additional context or answers were provided.
- **Request for Perplexity AI search functionality**: Another member wanted to know how to make Perplexity AI perform a search function for recent events, like obtaining details on a new car. 
- **Feature suggestion: llama-3-sonar-*-32k-online**: In response to the query about search functionality, another member suggested trying the feature named *"llama-3-sonar-*-32k-online"*.
- **Inquiry about closed beta API with citation and image support**: A member asked if anyone has access to a closed beta API that includes citation and image features after requesting access.
- **Issues with Perplexity API and status inquiry**: A member reported receiving 5xx errors while using the Perplexity API and asked if there was a status page to check when the API server will be up.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255238244241641472)** (66 messages🔥🔥): 

- **AI Engineer World’s Fair watch party coordination**: A member asks if anyone can host a watch party for the **AI Engineer World’s Fair**, which will be livestreamed [here](https://www.youtube.com/watch?v=5zE2sMka620). The event includes keynotes and code tracks.

- **PyTorch Documentary Premiere**: An announcement about the [PyTorch Documentary Virtual Premiere](https://www.youtube.com/live/EjgTv6aSeqk) sparks interest. It will feature key players from the project's history and present developments.

- **ChatGPT Voice Mode delay discussed**: A member shares a [tweet by Teknium](https://x.com/Teknium1/status/1805718678526476655/photo/1) discussing ChatGPT's delay in releasing its advanced Voice Mode. There are issues with **killing the waifu features** in the updates.

- **Excitement over AI wearable at AIE**: Attendees of the speaker dinner at the AI Engineer event received an **AI wearable from Bee Computer**. One member said, *"it knows almost all the most important facts about me... and has a list of TODOs for me"*, indicating its impressive functionality.

- **Fascination with reconstructed movies from mouse visual cortex**: A member is amazed by the [reconstructed movies from mouse visual cortex activity](https://x.com/Neuro_Joel/status/1805221959191437356). They describe the neuroscientific achievement as mind-blowing.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Neuro_Joel/status/1805221959191437356">Tweet from Joel Bauer (@Neuro_Joel)</a>: 🚀 Excited to share the first preprint from my postdoc in the labs of @ClopathLab and Troy Margrie at @SWC_Neuro! We&#39;ve reconstructed movies from mouse visual cortex activity. 📽️✨ #Neuroscience #...</li><li><a href="https://x.com/itsandrewgao/status/1805772589970649534?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from andrew gao (@itsandrewgao)</a>: 💋📚 Lip reading AI in action!  i infamously tweeted that i&#39;m bearish on voice.  today, i changed my mind.  @SymphonicLabs trained AI to ** read your lips **   now i can use voice interfaces compl...</li><li><a href="https://x.com/altryne/status/1805840851626869196?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: ok holy shit...   Everyone at the speaker dinner at @aiDotEngineer got a new AI wearable, @bee__computer as a gift.   I onboarded (will post video later) put it on and kinda forgot about it, and tool ...</li><li><a href="https://youtube.com/@aidotengineer?si=KfTkCwPDCRU7jY3t">AI Engineer</a>: Talks, workshops, events, and training for AI Engineers. </li><li><a href="https://www.youtube.com/live/EjgTv6aSeqk">PyTorch Documentary Virtual Premiere: Live Stream</a>: Join us for the official release of the PyTorch Documentary! Hear from key players in the project, from the early days to the present.</li><li><a href="https://youtu.be/ziGNnhNABqA?si=KcpiPiduDLHIpywA">David Luan: Why Nvidia Will Enter the Model Space &amp; Models Will Enter the Chip Space | E1169</a>: David Luan is the CEO and Co-Founder at Adept, a company building AI agents for knowledge workers. To date, David has raised over $400M for the company from ...</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://x.com/TheXeophon/status/1805718926162280804">Tweet from Xeophon (@TheXeophon)</a>: Anthropic in a single week: - Releases new model - Releases Artifacts - Releases Projects in Claude  OpenAI:  - Releases a Mac-only app - Delays voice for months  Quoting OpenAI (@OpenAI)   We&#39;re ...</li><li><a href="https://x.com/MLStreetTalk/status/1805686042726445337">Tweet from Machine Learning Street Talk (@MLStreetTalk)</a>: My dream came true today - an epic day of filming with the original gangster of AI - @SchmidhuberAI</li><li><a href="https://developer.bee.computer">Bee Developer Platform</a>: no description found</li><li><a href="https://x.com/Teknium1/status/1805718678526476655/photo/1">Tweet from Teknium (e/λ) (@Teknium1)</a>: They&#39;re having trouble killing the waifu features in the gpt4o voice updates :l</li><li><a href="https://x.com/altryne/status/1805840851626869196?s=46&t=tMWvmS3OL3Ssg0b9l">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: ok holy shit...   Everyone at the speaker dinner at @aiDotEngineer got a new AI wearable, @bee__computer as a gift.   I onboarded (will post video later) put it on and kinda forgot about it, and tool ...</li><li><a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Austin Byrd (@AustinTByrd)</a>: Figma AI is free for a year before they start billing everyone
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255563805493297153)** (100 messages🔥🔥): 

- **Vectors in SQLite Shine**: Multiple users expressed excitement about the topic, with one noting “Vectors in SQLite 🔥” and another capturing a screenshot of a relevant slide.
- **Vector Databases Declare Dead**: A bold statement was made that "vectordbs are dead," which sparked reactions among the participants.
- **Slides Will Not Be Available**: When asked if the slides from the presentation would be available later, the response was a firm "no," disappointing some attendees.
- **AI Engineer Conference Hiccups**: The conference faced several issues, including a 10-minute late start leading to a canceled talk and OpenAI dropping out with less than 48 hours notice. Swyxio expressed frustration about audio issues, saying, “this audio issue is pissing me off.”
- **YouTube Livestream for Follow-ups**: Swyxio referred attendees looking to catch up on missed content to the [YouTube livestream](https://www.youtube.com/watch?v=5zE2sMka620) for the "AI Engineer World’s Fair 2024 — Keynotes & CodeGen Track."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@aiDotEngineer">AI Engineer</a>: Talks, workshops, events, and training for AI Engineers. </li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255243079569375426)** (109 messages🔥🔥): 

- **Error Loading Model in LM Studio**: A member reported a recurring error, *(Exit code: -1073740791)*, when trying to load a model in LM Studio (0.2.25). It was suggested they provide system specs and try a different model or configuration.
- **OM issues with 3060 and Discussion of Alternative Models**: Attempting to run Hermes 2 Theta Llama-3 70B on an RTX 3060ti will lead to "Out of Memory" (OOM) issues. There’s a suggestion to use NousResearch's 8b version instead.
- **Struggles with Large Models on Apple M Chips**: A user described problems running Llama 3 70B on unified RAM where one model worked fine while another led to unreadable, scrambled output. It’s acknowledged that different quant types and settings like Q3 or Q4 KM may affect performance.
- **RAG Explanation and Application**: There's an in-depth explanation of retrieval-augmented generation (RAG) for assisting detailed information generation. The link [NVIDIA's blog on RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) was shared for understanding the concept better.
- **AnythingLLM for Document Analysis and Summarization**: For those needing document analysis and summary generation, "AnythingLLM" is recommended for its ease of use with various document types and integration with LM Studio. It was noted as a free and open-source solution.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/autotrain/en/llm_finetuning">LLM Finetuning</a>: no description found</li><li><a href="https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/">What Is Retrieval-Augmented Generation aka RAG?</a>: Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.</li><li><a href="https://js.langchain.com/v0.1/docs/expression_language/cookbook/retrieval/">Retrieval augmented generation (RAG) | 🦜️🔗 Langchain</a>: Let&#x27;s now look at adding in a retrieval step to a prompt and an LLM, which adds up to a &quot;retrieval-augmented generation&quot; chain:</li><li><a href="https://llm.extractum.io/">LLM Explorer: A Curated Large Language Model Directory. LLM List. 34902 Open-Source Language Models.</a>: Browse 34902 open-source large and small language models conveniently grouped into various categories and llm lists complete with benchmarks and analytics.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255295355122352321)** (7 messages): 

- **GM Master in Traveler Universe**: A user asked for the chatbot to act as the game master in a hard-core science fiction role-playing game set in the original Traveler universe. They emphasized the need for the inclusion of unfortunate outcomes to random events to enhance the value of success, and also to give the user a chance to respond to NPC actions.

- **Lack of Discussion on 70b New Dawn**: A member noted the notable absence of discussions around the **70b New Dawn model**, calling it "really good." Another user suggested that this might be due to most users only running smaller models like 7b-13b, limiting larger models' exposure.

- **Struggles with Academic-Level Local Models**: A user expressed dissatisfaction with local models, specifically **L3-Daredevil-Obliterated, L3 SFR-Iterative, and L3 Hermes**, for academic discussions and complex instructions. They inquired about recommendations for models under 34B, if not preferable 20B, highlighting their preference for FOSS and privacy-focused models over options like OpenAI.

- **Bartkowski's Q4KM DeepCoder V2 Performance**: A user shared their set-up for running **Bartkowski's Q4KM DeepCoder V2 230B** with 8K context using LM Studio 0.2.25 and discussed its impressive performance. They noted the model's RAM and GPU memory usage, achieving 2.68-2.8 tokens per second, and discussed their challenges with higher context lengths like 32K.
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1255358103369678873)** (2 messages): 

- **Scam alert on Discord URL**: A user cautioned that a shared URL links to a scam site in Russia. They emphasized that the URL in question is not real.
- **Message deletion confirmation**: Another user noted that they couldn't see the suspect message and speculated that it might have been deleted.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1255360235078549544)** (15 messages🔥): 

- **A4000 vs 4060ti for inference**: A member asked if anyone has experience with an A4000 and wondered about its performance for inference in a single slot compared to the **4060ti 16G**.
- **8x P40 Build complete**: A member shared that they have completed their build using **8x P40 GPUs**, integrated into a used server cabinet with Garuda Linux working out of the box with Nvidia drivers.
- **VRAM reporting in LM Studio**: Queries were made regarding if LM Studio accurately reports the **correct amount of VRAM** for systems with multiple GPUs, with one user indicating their setup reported ~192GB of VRAM correctly.
- **Home lab noise issues**: A humorous comment noted the downside of a home lab setup, comparing the server's noise to a "jet engine" on startup. 
- **Server power management**: Discussion included the use of a **200 amp circuit** with 4x1000w power supplies, noting power draw around 1KW.


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1255303988690030613)** (7 messages): 

- **Context window impacts token prediction**: A member humorously explained that "the longer the context window the more scratchpad the model needs" to track token prediction, and when it runs out, it starts emitting gibberish. They joked about optimal sock color affecting performance, emphasizing the anecdotal nature of their comments.
- **Potential scam links reported**: A user alerted mods to a person's account potentially being hacked and posting links to a "Russian scam site pretending to be steam".
- **Mod assistance requested**: A user asked for moderator intervention as scam links had been present in several channels for hours, and thanked those who responded.
- **Mamba architecture support query**: A user inquired about when LMstudio will start supporting the Mamba architecture.
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255598967190982757)** (1 messages): 

- **Interpreter refuses document and image uploads**: A member is frustrated that they cannot "move documents or images directly into the terminal" in their interpreter. They mentioned the interpreter "gives me the ban" and "doesn't give me consent" for these actions.
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1255585869721505854)** (2 messages): 

- **SDK walkthrough for building a Discord bot**: A helpful guide was shared for those interested in using the SDK to create a Discord bot. Check it out [here](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6).

- **Querying token speed and generation time in Python**: One member inquired about how to extract data from the local **LM Studio server** using Python, specifically focusing on token speed and the time it takes to generate the first token.

**Link mentioned**: <a href="https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6">no title found</a>: no description found

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255416795654000770)** (49 messages🔥): 

- **Exploration of Plotting with Mojo Data Types**: A user questioned if there are any projects that use **Mojo data types** directly for plotting without converting to Numpy and Matplotlib. Another user shared code examples and links to discussions about using **Mojo with Matplotlib** and the **Mojo-SDL project** on [GitHub](https://github.com/msteele/mojo-sdl) for SDL2 binding.

- **Community Decision on Charting Libraries**: A discussion emerged about what the community wants from a **Mojo charting library**. The conversation involved decisions on whether to make high-level interfaces, support interactive charts, or focus on data input formats such as Arrow.

- **Interactivity in Data Visualization**: A user emphasized the importance of **interactivity in data science** visualization tools, suggesting that something like the **Vega specification** could be an IR to address both web and native rendering. A Vega maintainer revealed challenges and the potential of alternative query engines like those used by **Mosaic**.

- **Mojo on Windows via WSL**: A user confirmed that **Mojo works on Windows using WSL** with native support expected by the end of the summer. They discussed the ease of linking WSL with **VSCode** and the slight learning curve related to using Linux and directories.

- **Reflection on Plotting Libraries**: There was a discussion on various plotting libraries like **D3, Altair, Matplotlib, Seaborn,** and **Plotly**, with users comparing their focuses and target audiences. The dialogue also touched on **UW’s Mosaic library** and its innovative approach in the data visualization space.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/mojo-pi-approximating-pi-with-mojo-using-monte-carlo-methods">Modular: Mojo🔥 ❤️ Pi 🥧: Approximating Pi with Mojo🔥 using Monte Carlo methods</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Mojo🔥 ❤️ Pi 🥧: Approximating Pi with Mojo🔥 using Monte Carlo methods</li><li><a href="https://github.com/msteele/mojo-sdl">GitHub - msteele/mojo-sdl: Minimal SDL2 binding for Mojo🔥</a>: Minimal SDL2 binding for Mojo🔥. Contribute to msteele/mojo-sdl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1806070670293692594>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1255248595314278460)** (2 messages): 

- **ARC test includes pattern recognition common to humans**: A member described the ARC test as a catalog focusing on *"closed area, symmetry, object features, and other stuff that are culturally common to humans."* They humorously suggested a *"dog arc test"* focusing on features relevant to dogs like poop smell and bark pitch.
- **AI aces IQ tests but lacks true intelligence**: One user argued that IQ tests do not measure intelligence, which is why AI can excel at them. They believe the ARC test is *"the most AI thing ever"* yet criticized its dataset as very poor.
- **Questioning the nature of intelligence and consciousness**: Another member pondered the difference between human information recall and AI systems like LLMs. They asked if others differentiate between intelligence and consciousness, suggesting recall might be only a part of intelligence.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255249270903275632)** (18 messages🔥): 

- **Mojo discusses GPU programming with MAX Graph API**: Brad Larson mentioned targeting GPUs using the MAX Graph API in Mojo, which lets users construct and optimize computational graphs. He explained, *"Custom operations can be written in Mojo and staged inside that graph."*

- **Bug with type checking in Mojo 24.4.0**: Brooklyn_marc reported a potential bug where a function returning a `String` was incorrectly accepted as a `Float`. Roboquant and carlcaulkett confirmed that newer nightly versions properly raise an error, with carlcaulkett sharing specific error output to illustrate the issue.

- **Variable reassignment and type checking quirks**: Darinsimmons experimented with the reported bug and noted that nightly versions handle type checking more robustly. He commented on the dynamics of assignment and type checking within the compiler, wondering if it's an order of operations issue.

- **Community encourages reporting issues**: Despite brooklyn_marc's hesitation, darinsimmons and soracc encouraged reporting potential issues even if they appear fixed in the nightly builds. *"If you think it's an issue, feel free to say something... they've been encouraging about seeing issues on GitHub."*
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255244562297913487)** (64 messages🔥🔥): 

- **Boolean Expressions Issue Found**: A user identified a problem with handling certain boolean expressions at compile time in Mojo, mentioning that commenting `@parameter` and removing `not` or using `var` fixes the issue. They linked a [specific commit](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5) possibly related to this issue.

- **Nightly Compiler Updates Released**: Two new nightly builds of the Mojo compiler have been released. Users are informed to update to `2024.6.2605` and `2024.6.2614` with links to the [raw diffs](https://github.com/modularml/mojo/compare/6961ce560d0457689f8667986b94a7ea02940cea...0ce792c1024386c17e8ced3d6cf4a70ec7113cc6) and [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Unsigned Integer Casting Bug**: There's ongoing discussion about an [issue](https://github.com/modularml/mojo/issues/3065) with unsigned integer casting overflowing as if signed in Mojo. Users are speculating about the behavior and potential bugs in how `var` and `alias` are handled.

- **List and Compile-Time Evaluation Bugs**: There's a [bug report](https://github.com.modularml.mojo/issues/3126) highlighting that `List` doesn’t work correctly at compile time in Mojo. This issue adds to another reported compiler-time problem with `Tensor`, which leads to inconsistent results during successive runs.

- **Using Static Lifetime for References**: The concept of `ImmutableStaticLifetime` was introduced, allowing users to take references to `alias` items, which was previously problematic due to lifetime issues. This enhancement is akin to using `let` and promises better management of static items.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/compare/6961ce560d0457689f8667986b94a7ea02940cea...7c00fc9a5a3171531da871f7fc3925f960bd8d31">Comparing 6961ce560d0457689f8667986b94a7ea02940cea...7c00fc9a5a3171531da871f7fc3925f960bd8d31 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/commit/6961ce560d0457689f8667986b94a7ea02940cea">[stdlib] Bump compiler version to 2024.6.2516 · modularml/mojo@6961ce5</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...</li><li><a href="https://github.com/modularml/mojo/issues/3098">[BUG] `Tensor` initialised from a list with wrong type shows weird behaviour · Issue #3098 · modularml/mojo</a>: Bug description To be more specific, Tensor[DType.int8] initialised with List[UIn8] doesn&#39;t compute its total number of elements correctly. I think it&#39;s again somehow related to implicit conve...</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` doesn&#39;t work at compile time. · Issue #3126 · modularml/mojo</a>: Bug description As title. At least List.__getitem__ doesn&#39;t work. Steps to reproduce fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # prints 0 System information Mojo 2024.6.2614 (366c690a) o...</li><li><a href="https://github.com/modular">Modular Inc</a>: Modular is an integrated, composable suite of tools that simplifies your AI infrastructure so your team can develop, deploy, and innovate faster. - Modular Inc</li><li><a href="https://github.com/modularml/mojo/issues/1405">[Modular CLI]: modular install mojo should support version pinning · Issue #1405 · modularml/mojo</a>: Issue description I cannot figure out how to install a specific version of mojo. This would be useful, essential really, for library maintainers and CI/CD systems. Steps to reproduce $ modular inst...</li><li><a href="https://github.com/modularml/mojo/issues/3065#issuecomment-2173567566**">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...</li><li><a href="https://github.com/modularml/mojo/pull/2847">[stdlib] List __getitem__ returns auto-dereferenced ref by mikowals · Pull Request #2847 · modularml/mojo</a>: With this, List.__getitem__() no longer makes copies when returning a value.  I also added a test to show that setting an individual field using sugared my_list[0].value = 1 no longer produces extr...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255585372436561984)** (14 messages🔥): 

- **Clement Delangue announces new LLM leaderboard with 300 H100s**: Clement Delangue announced a [new open LLM leaderboard](https://x.com/ClementDelangue/status/1805989925080219927) where 300 H100 GPUs were used to re-run evaluations like MMLU-pro. Key takeaways include Qwen 72B dominating, outdated evaluations, focus on main evaluations over others, and the realization that bigger models aren't always smarter.

- **Reaction to Delangue's 300 H100s**: Nathan Lambert humorously dismissed the effort by saying, "300 H100s is so few" and referred to it as "cringe" for a corporate CEO to brag about.

- **Community responds with humor**: Members like xeophon. joked about needing 300 H100s for their university and setting up a LinkedIn post for it, while sumo43 sarcastically interpreted "burned" 300 H100s as an offering to Jensen. 

- **Leadership board is supported despite skepticism**: Despite criticism about the announcement, Nathan Lambert expressed support for the new leaderboard, calling it "nice". 

- **Corporate bravado criticized**: The community generally criticized the perceived bravado of the announcement, with xeophon. mentioning how it plays into the "underdog meme/game" but acknowledged that it isn't effective.

**Link mentioned**: <a href="https://x.com/ClementDelangue/status/1805989925080219927">Tweet from clem 🤗 (@ClementDelangue)</a>: Pumped to announce the brand new open LLM leaderboard. We burned 300 H100 to re-run new evaluations like MMLU-pro for all major open LLMs!  Some learning: - Qwen 72B is the king and Chinese open model...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1255268298577154152)** (28 messages🔥): 

- **RabbitCode API keys vulnerability exposed**: The Rabbitude team uncovered critical **hardcoded API keys** in the Rabbit codebase on May 16, 2024, compromising services like [ElevenLabs](https://elevenlabs.io), [Azure](https://azure.com), [Yelp](https://yelp.com), and [Google Maps](https://maps.google.com). These keys could potentially allow unauthorized alteration and access to sensitive data.
- **ElevenLabs Credits might be exploitable**: Discussion emerged on whether the ElevenLabs credits from the compromised keys could be used, with one member remarking "it's just VC money" so it isn't real money.
- **OpenAI delays advanced Voice Mode**: According to [OpenAI](https://x.com/openai/status/1805716393524183136?s=46), the rollout of **ChatGPT’s advanced Voice Mode** has been delayed until fall for all Plus subscribers. The delay is to improve content detection and user experience.
- **HF leaderboard changes impact 7B model merge**: The [open LLM leaderboard changes](https://x.com/sebastianb929/status/1805996999499514233?s=46) saw the most significant drop in the 7B model merge, sparking reactions from the community.
- **Udio discusses AI's transformative role in music**: In a [detailed statement](https://x.com/udiomusic/status/1805694761891778783?s=46), Udio emphasized AI’s potential to empower artists, despite concerns from the [RIAA](https://x.com/riaa/status/1805739691972452559?s=46). They predict AI will become integral in music creation and industry growth.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/riaa/status/1805739691972452559?s=46">Tweet from RIAA (@RIAA)</a>: @RIAA response to @udiomusic ⬇️</li><li><a href="https://rabbitu.de/articles/security-disclosure-1">rabbit data breach: all r1 responses ever given can be downloaded - rabbitude</a>: rabbit inc has known that we have had their elevenlabs (tts) api key for a month, but they have taken no action to rotate the api keys.</li><li><a href="https://x.com/sebastianb929/status/1805996999499514233?s=46">Tweet from SebastianBoo (@SebastianB929)</a>: Who would have thought... The 7B model merge dropped the most 😅</li><li><a href="https://x.com/techmeme/status/1805951837713088643">Tweet from Techmeme (@Techmeme)</a>: Sources: YouTube is in talks with Sony, Warner, and Universal to license their songs for an AI music generation tool that mimics popular singers (Financial Times)  https://on.ft.com/3L0zaEQ  📫 Subscr...</li><li><a href="https://x.com/nathanbenaich/status/1805360586420895770">Tweet from Nathan Benaich (@nathanbenaich)</a>: these two music genai suits are worth a read  spicy and quite revealing</li><li><a href="https://x.com/udiomusic/status/1805694761891778783?s=46">Tweet from udio (@udiomusic)</a>: Today, we&#39;d like to share some thoughts on AI and the future of music.  In the past two years, AI has become a powerful tool for creative expression across many media—from text to images to film, ...</li><li><a href="https://x.com/openai/status/1805716393524183136?s=46">Tweet from OpenAI (@OpenAI)</a>: We&#39;re sharing an update on the advanced Voice Mode we demoed during our Spring Update, which we remain very excited about:  We had planned to start rolling this out in alpha to a small group of Ch...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255269708316541008)** (69 messages🔥🔥): 

- **Suspicion over Imbue's $200M raise**: Members discussed their skepticism about **Imbue's** recent success, questioning the lack of track record and sudden fundraise. One member mentioned a bad interview experience with them, while others noted they seem to be on a better track now.
  
- **Hype around CARBs and Scale**: There's excitement about the release of CARBs, as one member mentioned attempting to implement it previously. The chat later pivoted to discussing **Scale AI**'s strategy, including Scale's use of subsidiaries like **remotasks.com** for data annotation work to potentially isolate brand perception from customers.
  
- **Scale AI hires PhDs for remote work**: Discussion included Scale AI flying contractors for project collaborations, with some working on AI projects for companies like **Alphabet's Bard**. A member pointed to a subsidiary for various strategic reasons, aligning practices with competitors like **surgehq.ai** and **Dynata**.
  
- **Gemma V2 Model excitement**: Members showed enthusiasm for the **Gemma V2** model, discussed the coy naming conventions, and appreciated the immediate openness about it. One member highlighted a shared article revealing internal details indirectly, generating significant signups.
  
- **AI research lab dynamics at AI2**: There was an amusing anecdote about teaching a manager about system prompts, contrasting their high-level proficiency with basics. Discussions veered towards internal office dynamics and a critique of their current model outputs being 'meh' due to the lack of system prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/minjunesh/status/1805691167037919536">Tweet from minjune (@minjunesh)</a>: are u kidding me claude? this is 1984 levels of information gating</li><li><a href="https://fxtwitter.com/iamkrishnasoham/status/1805937316164198560?s=46">Tweet from krishna soham (@iamkrishnasoham)</a>: im-also-a-late-june-chatbot  cc: @apples_jimmy @jeremyphoward</li><li><a href="https://x.com/decompwlj/status/1805961700522291222?s=46">Tweet from Rémi Eismann (@decompwlj)</a>: @TheXeophon It seems this is a Gemma V2 model: https://reddit.com/r/LocalLLaMA/comments/1dovvbd/gemma_v2_in_the_lmsys_arena/</li><li><a href="https://outlier.ai/">Outlier</a>: Refine the Next Generation of AI with Your Expertise. 
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255242721212502077)** (11 messages🔥): 

- **Tempting Tweets Cause Debate**: A conversation arose about whether to tweet certain messages, with one remarking, *"Should I tweet this ^ lol"*. Another member expressed a desire to avoid *"directly attacking or dunking on people"* and expressed that the safer option was to post them in their *"secret discord friends"* group.
- **Extra Stock Request**: One member humorously mentioned they are in the market for scarce items, saying, *"if he bought one to rock and one to stock just let him know that I'm in the market 😏🫴💳"*.
  


---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255239693776453742)** (121 messages🔥🔥): 

- **Concerns about Stability AI’s Open Model**: Members voiced concerns that unless Stability AI fixes and un-censors SD3 while updating its license, no amount of new investors will save the company in the long run. One member added, *“People sitting around making lewd waifus and deep fakes all day doesn't serve any actual benefit to the generative AI community…”* indicating a need for real-world utility.

- **Cost Comparisons for GPUs**: Members discussed GPU rental costs on platforms like Runpod and Vast, highlighting that running a 3090 is currently cheaper on Vast. *“Running it on runpod is literally like 30 cents an hour,”* one member noted.

- **Debate on Open Source vs Corporate Interests**: The chat oscillated between advocating for open-source philosophies versus corporate interests. One member argued, *“you need the community to drive an open source philosophy,”* while another countered that Linux's success was majorly due to enterprise and corporate support.

- **New Builds and Hardware Recommendations**: A user sought recommendations for building a proper SD setup, with discussions favoring the Nvidia 4090 for its performance advantage. *“Probably cheaper to get 2x 4090 instead of a gpu with 48,”* was suggested as a cost-effective option.

- **ICQ Shutdown Noted**: Members reminisced about the past as ICQ, a once-popular messaging service, shut down. *“Oh, and ICQ dies today... R-I-P,”* one member remarked, triggering a nostalgic discussion.

- **Issues with Running SDXL**: Users reported difficulties running SDXL on their hardware, mentioning *“cuda out of memory”* errors, particularly on machines with limited VRAM. Advice was sought on suitable command-line arguments and optimizations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/the-office-no-angry-steve-carell-michael-scott-gif-5606969">The Office No GIF - The Office No Angry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://opendata.blender.org/">Blender - Open Data</a>: Blender Open Data is a platform to collect, display and query the results of hardware and software performance tests - provided by the public.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1255578706479550690)** (5 messages): 

- **Pet Psychic App Demo Sparks Interest**: A [meme](https://x.com/eggwens/status/1806016129875476886) posted by a member inadvertently highlighted a live demo for a **Pet Psychic Scheduler app**, featuring capabilities like booking psychic readings for pets and checking daily mood forecasts. Another member humorously inquired if the app was real, mentioning their dog's need for a horoscope.

**Link mentioned**: <a href="https://x.com/eggwens/status/1806016129875476886">Tweet from Egg (@eggwens)</a>: Here is the live demo for pet psychic attached are the sample codes made in react with sample styling:  With Pet Psychic Scheduler, you can: 🔮 Book psychic readings for your pets ✨ Check daily mood f...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255394755248918670)** (2 messages): 

- **Imbue AI releases a powerful 70B model toolkit**: Imbue AI announced they have trained a **70B model** optimized for **reasoning and coding**, matching the performance of LLAMA 3 70B with just 1/7th of the data. They are releasing [a toolkit](https://imbue.com/research/70b-intro/) that includes **11 NLP benchmarks**, a **code-focused reasoning benchmark**, and a **hyperparameter optimizer** for scaling experiments. 
- **Community reaction to Imbue's infrastructure deep dive**: A discussion emerged regarding the practicality and audience for Imbue AI's comprehensive infrastructure scripts intended for high-capacity training. One user noted that the detailed information might only be useful for a small niche market, albeit acknowledging its usefulness.

**Link mentioned**: <a href="https://x.com/imbue_ai/status/1805629542914211951?s=46">Tweet from Imbue (@imbue_ai)</a>: Early this year, we trained a 70B model optimized for reasoning and coding. This model roughly matches LLAMA 3 70B despite being trained on 7x less data.  Today, we’re releasing a toolkit to help othe...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255252246514372768)** (87 messages🔥🔥): 

- **New prompt engineering toolkit released**: A user shared, "I opensourced a little personal project I worked on over the weekend with sonnet 3.5, a prompt engineering toolkit," with a link to their [GitHub project](https://github.com/teknium1/Prompt-Engineering-Toolkit).

- **Disappointment with Microsoft model**: A user was dissatisfied with Microsoft's new raw text data augmentation model and shared a demo link to [Genstruct](https://huggingface.co/spaces/davanstrien/Genstruct-7B) which yielded confusing results unrelated to the provided context.

- **Speculation on specialized AI hardware**: Members discussed various high-performance AI chips like "Sohu" and others, debating their real-world performance and potential for inference, with references like [Gergely Orosz's post](https://x.com/GergelyOrosz/status/1805604272614088721) about OpenAI's internal expectations on AGI.

- **New local project announcement**: Another user excitedly shared a creative project involving character simulations using local LLMs, referencing [NousResearch's CharacterCodex](https://huggingface.co/datasets/NousResearch/CharacterCodex) and tools like Haystack.

- **Discussions about model repetition and sampling issues**: Users debated why instruction-tuned LLMs might repeat content, attributing it to "lack of repetition penalty or bad sampling settings," with experienced members confirming these potential issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Etched/status/1805625693113663834">Tweet from Etched (@Etched)</a>: Meet Sohu, the fastest AI chip of all time.  With over 500,000 tokens per second running Llama 70B, Sohu lets you build products that are impossible on GPUs. One 8xSohu server replaces 160 H100s.  Soh...</li><li><a href="https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about">About</a>: no description found</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://x.com/GergelyOrosz/status/1805604272614088721">Tweet from Gergely Orosz (@GergelyOrosz)</a>: Fascinating observation by @ByrneHobart about how even OpenAI is unlikely to believe that &#34;AGI&#34; is close, based on changing equity structures.  Basically, OpenAI is expecting more employees to...</li><li><a href="https://huggingface.co/spaces/davanstrien/Genstruct-7B">Genstruct 7B - a Hugging Face Space by davanstrien</a>: no description found</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>: Contribute to teknium1/Prompt-Engineering-Toolkit development by creating an account on GitHub.</li><li><a href="https://huggingface.co/posts/anakin87/427727576111455">@anakin87 on Hugging Face: &quot;🌌 Creating adventures with local LLMs

What if 🤔... Homer Simpson met…&quot;</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

namayra: me!
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1255238857339830282)** (69 messages🔥🔥): 

- **Stream LangChain responses with .stream() method**: After importing `LLM` from `langchain_community.chat_models` and installing `ollama`, it is recommended to use `.stream("query")`. This method allows for iterating through tokens and printing them line by line.

- **Long-term memory with Zep looks promising**: Users are discussing the potential of using [Zep](https://www.getzep.com/) for an AI's long-term memory, which can populate prompts with relevant facts from past conversations.

- **Using BytesIO for PDF in LangChain**: A user seeks a method to load a PDF document directly from a `BytesIO` object without creating a temporary file. Current workaround involves creating a temp file, which is seen as inefficient.

- **Streamlit with `AgentExecutor` and streaming responses**: Instructions provided for using `StreamlitCallbackHandler` to visualize thoughts and actions of an agent live in a Streamlit app. Users seek ways to handle streaming responses within this setup without using callback handlers.

- **LangSmith tracing issue troubleshooting**: A user inquires about LangSmith no longer tracing their project despite having set all required environment variables. The suggestion is to check if the trace quota has been exhausted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/callbacks/streamlit/#scenario-1-using-an-agent-with-tools>).">Streamlit | 🦜️🔗 LangChain</a>: Streamlit is a faster way to build and share data apps.</li><li><a href="https://www.getzep.com/">Zep - Long-Term Memory for AI Assistants</a>: Recall, understand, and parse chat dialog to power personalized experiences.</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/callbacks/argilla/#scenario-3-using-an-agent-with-tools>).">Argilla | 🦜️🔗 LangChain</a>: Argilla is an open-source data curation platform for LLMs.</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/providers/pebblo/pebblo_retrieval_qa/">Identity-enabled RAG using PebbloRetrievalQA | 🦜️🔗 LangChain</a>: PebbloRetrievalQA is a Retrieval chain with Identity &amp; Semantic Enforcement for question-answering</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/agents/#adding-in-memory>)">Build an Agent | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/agent_executor/#adding-in-memory>).">Build an Agent with AgentExecutor (Legacy) | 🦜️🔗 LangChain</a>: This section will cover building with the legacy LangChain AgentExecutor. These are fine for getting started, but past a certain point, you will likely want flexibility and control that they do not of...</li><li><a href="https://github.com/SuperDuperDB/superduperdb">GitHub - SuperDuperDB/superduperdb: 🔮 SuperDuperDB: Bring AI to your database! Build, deploy and manage any AI application directly with your existing data infrastructure, without moving your data. Including streaming inference, scalable model training and vector search.</a>: 🔮 SuperDuperDB: Bring AI to your database! Build, deploy and manage any AI application directly with your existing data infrastructure, without moving your data. Including streaming inference, scal.....</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/#streaming>)">Build a Chatbot | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/functions/#next-steps>),">How to run custom functions | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#non-streaming-components>).">How to stream | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://ai.stackexchange.com/questions/40753/how-to-generate-original-training-videos-based-on-existing-videoset">How to generate original training videos based on existing videoset?</a>: I am a software engineer who is quickly ramping up on AI tech, but am nevertheless very new to the sector.&#xA;A collegue has an extensive collection of training videos, the vertical is wheelchair sea...</li><li><a href="https://github.com/langchain-ai/langchain/issues/16980>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#streaming-tokens>)">Build an Agent | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://github.com/langchain-ai/langchain/issues/12441>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/7747>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/19944>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/debugging/#set_debug-and-set_verbose>)">How to debug your LLM apps | 🦜️🔗 LangChain</a>: Like building any type of software, at some point you&#x27;ll need to debug when building with LLMs. A model call will fail, or model output will be misformatted, or there will be some nested model ca...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1255465297880551464)** (1 messages): 

- **Tracing execution chains in GA4 from langserve backend intrigues**: A member inquired about tracing execution chains in GA4 using the langserve backend. They considered *subclassing Langsmith* and clarified the need to capture only the first invoke or stream without tracking any subsequent steps.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255492691605716993)** (2 messages): 

- **Testcontainers-Python adds Ollama support**: A member shared their new contributions to **testcontainers-python**, adding support for an **Ollama module** to test LLM endpoints through Ollama in Python. You can check their [issue](https://github.com/testcontainers/testcontainers-python/issues/617) and [pull request](https://github.com/testcontainers/testcontainers-python/pull/618) for more details and provide feedback.

- **Medium Article on Few-Shot Prompting with Tool Calling**: A member shared a [Medium article](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1) that discusses **few-shot prompting with tool calling** in Langchain. The article provides insights and methods to implement this approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/testcontainers/testcontainers-python/issues/617">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>: Add support for the OllamaContainer to simplify running and testing LLMs through Ollama. What is the new container you&#39;d like to have? I would like to request support for a new container: OllamaCo...</li><li><a href="https://github.com/testcontainers/testcontainers-python/pull/618">feat(core): Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>: Added a new class OllamaContainer with few methods to handle the Ollama container.   The _check_and_add_gpu_capabilities method checks if the host has GPUs and adds the necessary capabilities to th...</li><li><a href="https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1">Few-Shot Prompting with Tool Calling in Langchain</a>: Ankush k Singal
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1255531227516244051)** (1 messages): 

- **ARC AGI Challenge Video Shared**: A YouTube video titled "[Claude 3.5 struggle too?! The $Million dollar challenge](https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap)" was shared. The video provides a tutorial on how to do the ARC AGI challenges with agents and includes a link to a free HubSpot report on AI data analysis projects.

**Link mentioned**: <a href="https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap">Claude 3.5 struggle too?! The $Million dollar challenge</a>: The million dollar ARC AGI challengeGet free HubSpot report of how to do AI data analysis project: https://clickhubspot.com/d30🔗 Links- Follow me on twitter...

  

---



### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255298940090187776)** (37 messages🔥): 

- **LlamaIndex chatbot development issues**: A user queried about fetching context directly from a chat response instead of individual query results while building a [chatbot with LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/). They provided specific implementation details and challenges faced.

- **Review discussions on GitHub PR**: A member sought advice on merging a PR that adds functionality for filtering queries on [Neo4J Database in LlamaIndex](https://github.com/run-llama/llama_index/pull/14362). Another indicated they were backlogged with reviews but would attend to it soon.

- **Notification issues with ML libraries**: A user asked how to remove an notification about missing ML libraries while using the Openailike class. A response clarified that it wasn’t an error and pointed to the specific source of the message.

- **Fine-tuning LLM for SQL queries**: Users discussed the potential improvements in query precision through fine-tuning an LLM when using a RAG SQL layer. It was suggested that fine-tuning on good data would likely yield better performance.

- **Hybrid search with LlamaIndex**: There was an inquiry about implementing hybrid search by balancing metadata and text chunk influence, to which a detailed response outlined the use of the `alpha` parameter for configuring search weightings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Emerging-AI/ENOVA">GitHub - Emerging-AI/ENOVA: A deployment, monitoring and autoscaling service towards serverless LLM serving.</a>: A deployment, monitoring and autoscaling service towards serverless LLM serving. - Emerging-AI/ENOVA</li><li><a href="https://github.com/run-llama/llama_index/pull/14362">Add MetadataFilters to neo4j_property_graph by theoneamendez · Pull Request #14362 · run-llama/llama_index</a>: Description Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change. Summary fo...</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/">How to Build a Chatbot - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/indices/keyword/#llama_index.core.indices.SimpleKeywordTableIndex>)">Keyword - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/json.py#L51">llama_index/llama-index-core/llama_index/core/readers/json.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L69">llama_index/llama-index-core/llama_index/core/readers/file/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=table+parser#define-expanded-query-pipeline">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/#setup-simple-retry-agent-pipeline-for-text-to-sql">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1255296974819950632)** (2 messages): 

- **Optimize RAG systems with LlamaIndex and DSPy**: A member shared a [Medium article](https://medium.com/ai-advances/building-optimized-retrieval-augmented-generation-rag-systems-with-llamaindex-and-dspy-cacaf7f7089f) about building optimized Retrieval-Augmented Generation (RAG) systems utilizing LlamaIndex and DSPy. The article details practical steps and insights for achieving robust RAG implementations.

- **Open-source project seeks feedback with perks**: Another member introduced an [open-source project on GitHub](https://github.com/Emerging-AI/ENOVA) aimed at enhancing AI deployment, monitoring, and autoscaling services for serverless LLM serving. They are looking for feedback and offering a $50 gift card in exchange for an online interview to share insights.

**Link mentioned**: <a href="https://github.com/Emerging-AI/ENOVA">GitHub - Emerging-AI/ENOVA: A deployment, monitoring and autoscaling service towards serverless LLM serving.</a>: A deployment, monitoring and autoscaling service towards serverless LLM serving. - Emerging-AI/ENOVA

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1255247980051824650)** (9 messages🔥): 

- **Model confusion resolved: it's Claude-3.5-Sonnet**: There was some confusion about the **new Anthropic model's name**. It was clarified as `claude-3-5-sonnet-20240620`.

- **MoonDream-based local vision model discussed**: Members discussed whether **OI** has a **MoonDream-based** local vision model, but it's noted that it's not currently usable with OI.

- **Multiline input problems**: A member faced issues with the `-ml` option to use multi-line inputs using `'''`.

- **Vision error concerns**: Another member faced errors while using `interpreter --os --vision` to identify screen contents, despite verifying their **API key**.

- **File drop restrictions in the interpreter**: Restrictions around moving documents or images directly into the terminal were brought up, with a member indicating getting a **ban** when attempting to do so.
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1255273144059428894)** (17 messages🔥): 

- **01 is the voice interface for OpenInterpreter**: A user confirmed that 01 serves as the voice interface for OpenInterpreter (**OI**), addressing another member's confusion about the relationship between 01 and OI.
- **01 not available for sale in Spain**: A member expressed interest in purchasing 01 in Spain but was informed it is only available for sale in the United States. They were directed to a GitHub repository to build one themselves using the [open-source dev kit](https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight).
- **DIY 01 tutorials available online**: Another user confirmed that tutorials for building 01 from the open-source kit are available on YouTube, and they plan to create a tutorial in July.
- **Challenges setting up voice functionality**: Members discussed difficulties and specifics about setting up voice functions with 01, including integrating TTS and STT in Spanish and sending voice to an ngrok websocket on a Macbook Pro M1.

**Link mentioned**: <a href="https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight">01/hardware/light at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255267975372734475)** (16 messages🔥): 

- **Inquire About Scholars Program**: A member recently asked if the **scholars program** is running this year. There was no further discussion on this topic provided.

- **Preamble Tokens Billing Discussion**: A member detailed an experiment about billing for **preamble tokens** in API calls, emphasizing that "*preamble is billed*". They mentioned a scenario where a 16k token preamble could theoretically be used without charges if not billed.

- **Era of 1Bit LLMs Talk Event**: Announcement for a talk by **Hongyu Wang** on the topic *The Era of 1Bit LLMs*. Participants were invited to join the talk via a [Google Meet link](https://meet.google.com/yhv-tiir-ava).

- **Websim.ai Webpage Simulation**: Members enjoyed experimenting with [Websim.ai](https://websim.ai/c/0NUkL2gMKZefC1AoZ), which simulates predicted webpages based on URLs. It uses Anthropics' Claude to create artifacts and simulate a personal pocket internet.

- **Reporting Commercial Abuse of Command-R**: A member raised a concern about a NSFW bot hosting service, **SpicyChat AI**, using Command-R for profit-making. They highlighted the service's owner claiming that the use of **OpenRouter** negates the Cohere **CC-BY-NA** license.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/0NUkL2gMKZefC1AoZ]">...</a>: no description found</li><li><a href="https://rhea.run)">no title found</a>: no description found</li><li><a href="https://websim.ai/c/0NUkL2gMKZefC1AoZ">Run From Dinosaurs - Crash Bandicoot Style</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156)** (2 messages): 

- **Announcing Rig Rust Library Release**: A member shared an update on the release of **Rig**, a Rust library for building LLM-powered applications. They are running an incentivized feedback program where developers are rewarded for building use cases and providing feedback on the library.
- **Feedback program is on-topic**: Another member confirmed that posting about the feedback program is appropriate for this channel. They humorously mentioned that the library should obviously support **Cohere's models**.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255268720448635040)** (11 messages🔥): 

- **Adam-mini aims to optimize memory usage**: A member shared an [arXiv paper](https://arxiv.org/abs/2406.16793) proposing Adam-mini, an optimizer that achieves performance on-par or better than AdamW with 45% to 50% less memory footprint. The optimizer reduces memory by cutting down the number of learning rates in Adam, using a single learning rate for parameter blocks inspired by the Hessian structure of Transformers.
- **Inquiring about masking output texts during training**: A member asked if there's a way to mask out certain 'output' texts, similar to `train_on_input`, suggesting a potential feature like `do_not_train_on_output_marked_as_masked`.
- **Debate on gradient accumulation impact on training time**: Multiple members discussed whether increasing gradient accumulation (GA) times, e.g., accumulating gradients 100 times, would affect the overall training time per epoch. One member suggested it might be faster as the optimizer runs fewer times, potentially reducing noise absorbed by parameters, while another argued that high GA slows down per step performance.
- **Issue with CUDA errors during training**: A member shared an error related to CUDA illegal memory access encountered during training, suggesting debugging steps like using `CUDA_LAUNCH_BLOCKING=1` or compiling with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

**Link mentioned**: <a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the number of learning rates in...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255240413720215695)** (4 messages): 

- **Creating cosine lr scheduler with custom min lr on HF**: A member asked about an easy way to create a cosine learning rate scheduler with a minimum learning rate greater than zero on Hugging Face. This points to potential tweaks in the HF library for practical implementations.

- **QDora enablement in PEFT**: Caseus mentioned a pull request that enables QDora in PEFT and promised to track it down. This sparked interest from another member willing to put in significant effort to get it working.

- **Mistral7B repeat issues**: A user reported that their full instruction-tuned Mistral7B model repetitively generates sentences or paragraphs even with high temperature settings. They noted that their dataset does not contain such repetition, seeking advice for the cause and solution.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1255484851180671027)** (1 messages): 

- **Storiagl: Free Story Creation with LLMs**: An exciting new platform, [Storiagl](https://storiagl.web.app/), allows users to create and play stories utilizing custom LLMs for character interpretation. It offers advanced settings to craft complex and detailed narratives.

**Link mentioned**: <a href="https://storiagl.web.app/">StorIA</a>: no description found

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255328754041880628)** (6 messages): 

- **Teaching LLMs Kalamang from a single book**: A Dutch PhD student, Eline Visser, wrote "The Grammar of Kalamang," the only text in the language, and researchers used this to see if a language model could learn Kalamang using various types of fine-tuning and prompting. Interestingly, *"prompting wins (and it’s not close)"* in this experiment, although humans still outperform LLMs in this task. [Detail](https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ); [Abstract](https://arxiv.org/abs/2309.16575). 

- **AI Engineer World’s Fair 2024 streaming now**: The "AI Engineer World’s Fair 2024" focusing on Keynotes & CodeGen Track is currently live-streaming on YouTube. [Watch here](https://www.youtube.com/watch?v=5zE2sMka620); more details are available through the event's [Twitter description](https://twitter.com/aidotengineer).

- **Build with Claude Contest June 2024**: The "Build with Claude" contest for June 2024 has been announced, providing an opportunity for participants to showcase their capabilities with Claude. More details can be found in the [official contest overview](https://docs.anthropic.com/en/build-with-claude-contest/overview).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ">Tweet from jack morris (@jxmnop)</a>: recently read one of the most interesting LLM papers i&#39;ve ever read, the story goes something like this  &gt; dutch PhD student/researcher Eline Visser lives on remote island in Indonesia for seve...</li><li><a href="https://arxiv.org/abs/2309.16575">A Benchmark for Learning to Translate a New Language from One Grammar Book</a>: Large language models (LLMs) can perform impressive feats with in-context learning or lightweight finetuning. It is natural to wonder how well these models adapt to genuinely new tasks, but how does o...</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://docs.anthropic.com/en/build-with-claude-contest/overview">Build with Claude June 2024 contest - Anthropic</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1255290715492323431)** (1 messages): 

- **Email Verification Requested for Credits Issue**: A member offered help with a credits form issue, advising another to DM them with the email address used in the form. *"Feel free to DM me and I can take a look - please let me know what email you used in the credits form."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1255496809317400617)** (3 messages): 

- **DS surpasses FSDP in offloading**: *"Likely the reason for this is DS has more fine-grained offloading vs FSDP, assuming that offloading is happening"*.
- **Lacking experience with DS and FSDP**: A member noted, *"I do not/haven't used them yet"*.
- **Exploring LLama 70B settings**: A member shared that they wanted to try LLama 70B but acknowledged a need to understand more about the settings.
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260)** (1 messages): 

- **Builders Program Early Deadline Reminder**: A reminder was issued for the Builders Program, urging members to submit their applications before **July 8th** for the early application deadline. More information and applications can be found [here](https://future.mozilla.org/builders/).

- **Questions and Support Available**: For any questions related to the Builders Program, members can get support through this [Discord channel](https://discord.com/channels/1089876418936180786/1245083732319408195).
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1255492503646244945)** (9 messages🔥): 

- **Firefox makes llamafile integration a nostalgic web adventure**: Using llamafile as an HTTP proxy, **Firefox** can explore the knowledge in LLM weights, creating a web experience reminiscent of the '90s. Check out the [YouTube video](https://youtu.be/YWQ5Kh9gNuo) demonstrating this integration.

- **Chat Adventures with llamafile and Character Codex**: A member shared a detailed notebook on *Creating Chat Adventures from scratch* using llamafile, Haystack, and Character Codex. Access the notebook [here](https://t.ly/y6jrZ) to experiment with scenarios like Homer Simpson meeting Spider-Man.

- **CUDA warnings in Jupyter notebooks**: There's a discussion about handling CUDA warnings in Jupyter-like environments to keep the notebooks clean. Suggested solution involves [a utility from Haystack](https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py) to detect if a program is running in such an environment.

- **NVIDIA stock volatility amidst AI news**: A tweet highlighted a significant drop in NVIDIA's market cap after a talk at AIEWF, with conflicting analyses from [MarketWatch](https://www.marketwatch.com/story/nvidias-stock-is-set-to-gain-with-rivals-seen-to-be-in-perpetual-catch-up-mode-0552e514) and [Barrons](https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c) discussing the factors affecting the stock price.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aiDotEngineer/status/1806012306046046232?t=BudJTKP1KpdJcSNGzihIYg&s=19">Tweet from AI Engineer (@aiDotEngineer)</a>: breaking news: $NVDA loses $56 BILLION in market cap after @JustineTunney @stlhood talk at AIEWF  Quoting swyx 👉 ai.engineer (@swyx)   MOZILLA is so back what the hell  MOZILLA AI might just be their...</li><li><a href="https://youtu.be/YWQ5Kh9gNuo">llm browser demo</a>: no description found</li><li><a href="https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py">haystack/haystack/utils/jupyter.py at main · deepset-ai/haystack</a>: :mag: LLM orchestration framework to build customizable, production-ready LLM applications. Connect components (models, vector DBs, file converters) to pipelines or agents that can interact with yo...</li><li><a href="https://t.ly/y6jrZ">haystack-cookbook/notebooks/charactercodex_llamafile.ipynb at main · deepset-ai/haystack-cookbook</a>: 👩🏻‍🍳 A collection of example notebooks. Contribute to deepset-ai/haystack-cookbook development by creating an account on GitHub.</li><li><a href="https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c">Nvidia Stock Still Falling as Shareholder Meeting Concludes</a>: At a quick annual meeting, Nvidia shareholders approved all 12 recommended nominees to the company’s board. 
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1255308575144677426)** (7 messages): 

- **FPGA backend for tinygrad discussed**: Members discussed the possibility of **tinygrad** having an FPGA backend. George Hotz suggested that it might be more practical to design an **accelerator** that runs on the FPGA and then target that.

- **Positron aims for transformer inference**: Some **Groq engineers** have left to work on [Positron](https://www.positron.ai/), which aims to provide hardware solutions like the Atlas Transformer Inference Server and Redstone Developer Workstation, claiming **10x better performance** per dollar than DGX-H100.

- **FPGA specialization and HDL**: Members mentioned newer FPGA platforms that are being outfitted with DSP blocks and HBM, which could specialize models by generating HDL specific to them, although **trsohmers** clarified that Positron is not using Xilinx/AMD FPGAs and their design is generic for all transformer models.

- **PyTorch documentary shared**: A link to a [YouTube documentary](https://www.youtube.com/watch?v=rgP_LBtaUEc) titled "Official PyTorch Documentary: Powering the AI Revolution" was shared, providing insights into the origins and impact of PyTorch on the AI landscape.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.positron.ai/">Positron | The Best Performing Transformer Inference System</a>: Positron makes purpose-built hardware to accelerate multimodal AI.</li><li><a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">Official PyTorch Documentary: Powering the AI Revolution</a>: This film unveils the authentic narrative of PyTorch’s inception, attributing its existence to a dedicated group of unsung heroes driving technological innov...
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1255604696694128782)** (4 messages): 

- **Angry.penguin steps up as mod**: A member, angry.penguin, offered to become a mod to prevent future issues with spam, stating *"if you want to make me a mod I can set it up so this doesnt happen again in the future!"* Yoko Li promptly accepted the offer.
- **Yoko Li and angry.penguin handle spam**: Following their promotion to mod, angry.penguin reported that they have addressed the spam issue, mentioning *"Should be good now from future spam 😄"* and *"also cleaned up the spam"*.
  

---



### **DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1255457292179214436)** (4 messages): 

- **New German Encoders Released**: A member announced the release of **German Semantic V3** and **V3b** on [Hugging Face](https://huggingface.co/aari1995/German_Semantic_V3). V3 focuses on being knowledge-heavy while V3b is geared towards performance with features like Matryoshka Embeddings and an 8k context length.

- **Inquiry about GGUF**: Another member inquired about the existence of **gguf** formats for the new encoders and how to further finetune **V3b**. The response indicated no gguf available and suggested using sentence-transformers scripts from [UKPLab's GitHub repository](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training) for finetuning.

- **GGUF Format Feasibility**: The member clarified that it is possible to use gguf formats for encoders and cited **Ollama** as an example of utilizing two embedders in such a format.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/UKPLab/sentence-transformers/tree/master/examples/training">sentence-transformers/examples/training at master · UKPLab/sentence-transformers</a>: Multilingual Sentence &amp; Image Embeddings with BERT - UKPLab/sentence-transformers</li><li><a href="https://huggingface.co/aari1995/German_Semantic_V3">aari1995/German_Semantic_V3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/aari1995/German_Semantic_V3b">aari1995/German_Semantic_V3b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255238674099077120)** (2 messages): 

- **OpenRouter Introduces New Model**: Check out the new [01-ai/yi-large model](https://openrouter.ai/models/01-ai/yi-large) just announced by OpenRouter, LLC for 2023 - 2024. It's the latest addition to their offering.

- **Recommended Parameters Tab Issue Fixed**: There was an issue with the incorrect data being shown on the Recommended Parameters tab for model pages. This problem has been resolved and the tab now displays the correct information.

**Link mentioned**: <a href="https://openrouter.ai/models/01-ai/yi-large)">Yi Large by 01-ai</a>: The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service.  It stands out for its multilingual pro...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1255275992071667743)** (1 messages): 

- **GPA Saver Website Launched**: A member shared their new website, [GPA Saver](https://gpasaver.com/), integrating AI for academic assistance. They thanked OpenRouter for making **LLM integration** seamless and easy.

- **AI-Powered Academic Tools**: The website offers several academic tools including an assistant chat, rapid quiz solver, PDF summarizer, interactive whiteboard, and flashcard generator. There's a special discount code **BETA** for the first 100 users, providing ~37% off.

**Link mentioned**: <a href="https://gpasaver.com/">GPA Saver</a>: Leverage the power of AI for your studies.

  

---



---



---



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

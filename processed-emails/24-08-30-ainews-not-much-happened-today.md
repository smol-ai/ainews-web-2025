---
id: 7de5091f-80d2-45af-b049-8df262bd2e14
title: not much happened today
date: '2024-08-31T00:41:42.203560Z'
original_slug: ainews-not-much-happened-today-5498
description: >-
  AI news highlights model updates including Google Gemini, Cohere Command R,
  LLaMA 3.1 adoption, long context models, style control, and new multimodal
  models. It also covers AI safety initiatives, tools, and industry trends like
  AI hype cycles and call center disruption.
companies:
  - google
  - cohere
  - meta
  - alibaba
  - openai
  - lmsys
models:
  - gemini
  - command-r
  - llama-3-1
  - ltm-2-mini
  - qwen2-vl
  - chatgpt-4o-latest
  - claude-3.5-sonnet
topics:
  - model-updates
  - long-context
  - style-control
  - multimodal
  - ai-safety
  - ai-regulation
  - ai-tools
  - market-trends
  - industry-disruption
---


<!-- buttondown-editor-mode: plaintext -->**3 day weekends are all you need.**

> AI News for 8/29/2024-8/30/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**213** channels, and **3131** messages) for you. Estimated reading time saved (at 200wpm): **340 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

A smattering of things we considered:

- Highlighting the [Google Gemini](https://x.com/OfficialLoganK/status/1828508078955696337) and [Cohere Command R](https://x.com/itsSandraKublik/status/1829519989969133757) ([blogpost](https://docs.cohere.com/changelog/command-gets-refreshed), but no leaderboard updates yet) model updates this week
- Lmsys responding to criticism by [introducing style control](https://x.com/lmsysorg/status/1829216988021043645) leaderboards, though ChatGPT-4o-latest still destroys everyone else
- Meta's AI assistant [announcing](https://x.com/Ahmad_Al_Dahle/status/1829541138736509102) 400m MAU, 185m WAU, 40m DAU.

But nothing seemed must-know.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**AI Model Developments and Benchmarks**

- **LLaMA 3.1 Adoption**: Meta announced significant adoption of LLaMA models, with nearly 350 million downloads on Hugging Face and widespread use across industries. [@AIatMeta](https://twitter.com/AIatMeta/status/1829157383052111946) highlighted the importance of open source AI in extending benefits to everyone.

- **Long Context Models**: Magic AI Labs introduced LTM-2-Mini, a model with a 100 million token context window. [@magicailabs](https://twitter.com/magicailabs/status/1829206893765767282) claimed this is equivalent to 10 million lines of code or 750 novels. They also introduced HashHop, a new evaluation method for long-context models.

- **Style Control in AI Evaluations**: LMSys introduced style control in their regression model for Chatbot Arena, aiming to separate the impact of style from substance in rankings. [@lmsysorg](https://twitter.com/lmsysorg/status/1829216988021043645) reported that models like Claude 3.5 Sonnet and Llama-3.1-405B rose significantly when style was controlled.

- **Qwen2-VL Release**: Alibaba released Qwen2-VL, a new multimodal LLM available in 2B and 7B sizes under Apache 2.0 license. [@_philschmid](https://twitter.com/_philschmid/status/1829190887399673908) noted its competitive performance with GPT-4o mini on various benchmarks.

**AI Safety and Regulation**

- **US AI Safety Institute Testing**: OpenAI CEO [@sama](https://twitter.com/sama/status/1829205847731515676) announced an agreement with the US AI Safety Institute for pre-release testing of future models, emphasizing the importance of national-level testing.

- **Concerns About AI Takeover**: [@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1829214030629876106) discussed the need for preventative measures against potential AI takeover, questioning how to build consensus and willingness to act before catastrophic harm occurs.

**AI Applications and Tools**

- **Web Crawling Tool**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1829158662964691159) shared information about firecrawl, an open-source tool for crawling entire websites and converting them into LLM-ready markdown or structured data.

- **PDF Processing Challenges**: [@svpino](https://twitter.com/svpino/status/1829137471717658884) highlighted the difficulties of processing PDF documents with current AI models and suggested preprocessing documents into text format for better results.

**AI Industry and Market Trends**

- **AI Hype Cycles**: [@fchollet](https://twitter.com/fchollet/status/1829258691100737701) observed that peak AI hype in the tech community was in Q1-Q2 2023, while peak AI greed in public markets was in Q1-Q2 2024, noting that progress in AI research and applications continues regardless.

- **Call Center Industry Disruption**: A viral Reddit post discussed the potential impact of AI on the call center industry, suggesting that AI agents could replace human workers within two years. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1829204901957706037) shared this, noting the implications for customer service and employment.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advancements in Long Context AI Inference**

- **Local 1M Context Inference at 15 tokens/s and ~100% "Needle In a Haystack": InternLM2.5-1M on KTransformers, Using Only 24GB VRAM and 130GB DRAM. Windows/Pip/Multi-GPU Support and More.** ([Score: 114, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1f3xfnk/local_1m_context_inference_at_15_tokenss_and_100/)): KTransformers project has introduced **local 1M context inference** for the **InternLM2-1M model**, achieving **15 tokens/s** inference speed and ~100% accuracy on a "Needle In a Haystack" challenge using only **24GB VRAM** and **130GB DRAM**. The project implements an **efficient sparse attention operator for CPUs**, based on research like H2O, InfLLM, Quest, and SnapKV, resulting in a **6x speed increase** and **92.88% success rate** on the 1M challenge, while maintaining **100% accuracy** on the 128K test.
  - The **RULER benchmark** suggests **InternLM2.5** has an "effective" context length of only **4K tokens**, after which it performs worse than **Llama2-7b**. The project developer noted they will test RULER later, emphasizing that their demo showcases the **sparse attention operator's** effectiveness.
  - Users expressed interest in adding support for **Mistral Large 2** to the project's model list, which already includes **Mixtral-8x22B**. The project's progress has been described as "exciting to track" by some commenters.
  - Some users reported installation issues, with one encountering **404 errors** from pip during the cmake process. This suggests potential technical challenges in setting up the project for some users.


**Theme 2. California's SB 1047: Implications for AI Development**

- **[SB 1047 got passed.  Do you think this will affect LLAMA?](https://i.redd.it/68cnmukzxpld1.png)** ([Score: 52, Comments: 68](https://reddit.com//r/LocalLLaMA/comments/1f4lbfy/sb_1047_got_passed_do_you_think_this_will_affect/)): **SB 1047**, a bill addressing **AI-generated content**, has been passed in California. The legislation requires **disclosure of AI-generated content** in certain contexts, which could potentially impact **LLAMA** and other AI language models. While the specific effects on LLAMA are uncertain, the bill's passage may necessitate changes in how AI-generated content is presented and used, particularly in commercial and political applications.
  - The bill's **$100 million training cost threshold** sparked debate about its impact on **open-source AI**. Some argue it won't affect local models, while others believe it could impact larger models like **LLAMA 405B** and its distillations.
  - Critics expressed concerns about the bill's potential to **stifle innovation** and favor large corporations. Some users called **Governor Newsom's office** to oppose **SB 1047**, citing worries about unnecessary regulations and increased costs for AI companies.
  - The legislation requires **safety measures** for large AI models, including **shutdown capabilities**, **third-party audits**, and **whistleblower protections**. Some view these as reasonable precautions, while others see them as potential threats to open-source development and free speech.


- **California assembly passed SB 1047** ([Score: 165, Comments: 73](https://reddit.com//r/LocalLLaMA/comments/1f4jftq/california_assembly_passed_sb_1047/)): The California assembly passed **SB 1047**, a bill that could significantly impact **open-source AI models**. The legislation reportedly includes provisions requiring model authors to have the ability to **shut down their models**, potentially making it impractical for state-of-the-art AI models to be open source and potentially concentrating AI development among a limited number of corporations.
  - **Meta** may face significant challenges due to the bill, as they are **headquartered in California**. Users speculate the company might **move to Seattle** or **spin off a subsidiary** to circumvent the law, while others suggest they may simply stop releasing **open-source models**.
  - The bill's **$100 million training cost** threshold for covered models was reportedly determined arbitrarily by **Eric Schmidt and colleagues**, according to a [YouTube video](https://youtu.be/7PMUVqtXS0A) at 20:15. Some users argue this legislation could drive innovation out of California and benefit Chinese AI development.
  - Legal scholars suggest companies **doing business in California** would need to comply with the bill regardless of location, due to the state's economic importance. Some users view this as California **handicapping the entire industry**, while others see it as **big tech corporations wanting regulation** to limit competition.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Video Generation and Visual Effects**

- **AI-generated monster movie clips**: A [video showcasing AI-generated sea monster scenes](https://www.reddit.com/r/singularity/comments/1f4iebb/fishing_for_megalodons_cousins_the_best_ai_video/) sparked discussion about the current state of AI video generation. While impressive, many commenters noted it still falls short of Hollywood quality, citing issues with physics, geometry, and human reactions.

- **AI movies on the horizon**: A [post about upcoming AI-generated movies](https://www.reddit.com/r/singularity/comments/1f4fv05/ai_movies_are_coming/) received significant attention, indicating growing interest in AI's potential impact on the film industry.

**AI Model Advancements**

- **Magic's 100 million token context window**: [Magic has trained a model with a 100 million token context window](https://www.reddit.com/r/singularity/comments/1f4917u/magic_has_trained_their_first_model_with_a_100/), equivalent to 10 million lines of code or 750 novels, representing a significant advancement in model context capacity.

**AI Safety and Regulation**

- **Anthropic's agreement with US AI Safety Institute**: [Anthropic has reached an agreement with the US AI Safety Institute](https://www.reddit.com/r/singularity/comments/1f47y4n/sama_we_are_happy_to_have_reached_an_agreement/) for pre-release testing of their future models, indicating a step towards more regulated AI development.

**AI in Gaming and Interactive Environments**

- **AI playing Minecraft**: A [video demonstrating an AI playing Minecraft like a human](https://www.reddit.com/r/singularity/comments/1f4ap60/ai_playing_minecraft_with_me_like_a_human/) showcases advancements in AI's ability to interact in complex, open-world gaming environments.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. LLM Advancements and Benchmarking**

- **Llama 3 Tops Leaderboards**: **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** from Meta has rapidly risen to the top of leaderboards like **ChatbotArena**, outperforming models like **GPT-4-Turbo** and **Claude 3 Opus** in over 50,000 matchups.
   - The community expressed excitement over Llama 3's performance, with discussions on its potential impact on the AI landscape and how it compares to proprietary models.
- **Grok 2 Impresses in Code Generation**: Discussion highlighted performance comparisons between **Grok 2**, **Gemini**, and **ChatGPT**, with Grok 2 noted as particularly strong in **code generation** tasks.
   - Users speculated on upcoming models such as Grok 3, raising questions about potential performance edges backed by robust hardware setups.
- **Word Game Bench Challenges LLMs**: The newly developed **[Word Game Bench](https://wordgamebench.github.io)** serves as a benchmark to evaluate language models on word puzzle games like **Wordle**, with no model currently achieving over a **50% win rate**.
   - This benchmark focuses on model interaction and reasoning, emphasizing the challenges LLMs face in dynamic, game-like environments.
  


**2. Open Source AI Developments**

- **Re-LAION-5B Dataset Launch**: The launch of **[Re-LAION-5B](https://laion.ai/blog/relaion-5b/)**, a cleaned version of the LAION-5B dataset, was celebrated by the community for addressing previous safety concerns.
   - This updated dataset, created in partnership with key organizations, marks a significant milestone in ensuring safety and compliance in large-scale AI training data.
- **RunwayML Deletes Stable Diffusion Repos**: **RunwayML** deleted all their **Stable Diffusion 1.5** repositories on HuggingFace and GitHub, causing frustration among users and breaking functionalities in Diffusers 1.5.
   - The community speculated about potential legal issues behind the deletions, highlighting the impact of such actions on the open-source AI ecosystem.
- **GameNGen: Neural Game Engine Breakthrough**: **[GameNGen](https://gamengen.github.io/)**, the first game engine powered entirely by a neural model, can simulate **DOOM** at over 20 frames per second on a single TPU, achieving a PSNR of 29.4.
   - This breakthrough demonstrates the potential of neural models in real-time game simulation, with human raters struggling to distinguish between real gameplay and simulations.
  


**3. Model Optimization Techniques**

- **Dynamic Expert Routing Enhances Adaptability**: The concept of allowing models to define their own experts during training, instead of using a fixed configuration, was discussed as a way to improve adaptability.
   - This idea is linked to ongoing research like the methods proposed in the **[LayerSkip paper](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e)**, aiming to enhance model performance and efficiency.
- **Quantization Techniques for Large Models**: Discussions highlighted quantization techniques like **AQLM** and **QuaRot** aimed at running large language models (**LLMs**) on individual GPUs while maintaining performance.
   - Members shared implementation details and benchmarks, such as running **Llama-3-70b** on RTX3090, showcasing the potential of these optimization methods.
- **Finite Scalar Quantization (FSQ) as VQ-VAE Alternative**: The introduction of **finite scalar quantization (FSQ)** was discussed as a potentially effective and simpler alternative to traditional vector quantization techniques in VQ-VAEs.
   - The FSQ method promises improved performance across various tasks, as noted in a [linked paper](https://arxiv.org/abs/2309.15505), with implications for token utilization in language models.
  


**4. AI Deployment and Infrastructure**

- **Tinygrad Launches Affordable Cloud Service**: **Tinygrad** announced a new cloud service offering a **4090 GPU** and **500 GB of storage** for just **$60/month**, making it 3x cheaper than competitors like Vast AI.
   - The service introduces a 'CLOUD=1' feature, allowing users to run Tinygrad locally while leveraging cloud speed for performance enhancements with 10-step processing.
- **OpenRouter Stealth Launch Goes Live**: **OpenRouter** successfully launched, serving **Llama 3.1-405B-instruct** with **128k context** and function calling support at a competitive price of **$2.5/mil tokens**.
   - The team emphasized building reliable infrastructure over referral-based compensation, highlighting their focus on service quality and accessibility.
- **Cohere's Command R Series Update**: Cohere announced refreshed **Command R** and **R+** models with improvements in performance for reasoning, coding, and multilingual RAG, now available under new aliases.
   - The updated models feature lower pricing per token, with R being significantly cheaper at **$0.15** for input tokens, showcasing advancements in both performance and cost-efficiency.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Debate on Fine-Tuning vs RAG**: Discussion revealed that while **RAG** might reduce hallucinations, controlled overfitting is crucial in fine-tuning processes. The effectiveness largely hinges on dataset size and hyperparameters like rank and alpha.
   - Participants emphasized that neither method clearly outranks the other and both strategies must be tailored based on specific project requirements.
- **Diverse Use Cases for LLMs**: LLMs are currently employed across various industries, with companies like **AT&T** using them for customer support and others for proprietary research applications. Instruction-based models akin to **GPT** dominate the deployment landscape.
   - The versatility shown in these applications indicates a strong trend towards integrating LLMs into practical daily operations.
- **OpenRouter Launch Hits the Ground Running**: The **OpenRouter** successfully went live with the **Llama 3.1-405B-instruct**, featuring **128k context** and function calling capabilities at an inviting price of **$2.5/mil tokens**.
   - Clarifications highlighted that the developer's compensation is unaffected by referral link usage, focusing instead on building reliable infrastructure.
- **Upcoming Models and New Pricing Trends**: Speculation around **Meta's** soon-to-be-announced **Llama models** has generated buzz, though specifics about **Llama 4** are still unclear. Concurrently, **OpenAI** revealed reduced pricing for their **GPT-4o model**, which now costs **$4 per 1M tokens**.
   - The adjustments provide a pathway for developers to optimize costs while accessing newer models and features, such as structured outputs aligning strictly with JSON Schemas.
- **Community Collaboration on Finetuning Goals**: A community member expressed eagerness to finetune an LLM without a specific objective, just for the fun of it. This openness highlights the exploratory spirit within the community.
   - Such a mindset may inspire other developers to experiment and innovate outside of fixed project frameworks.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini Model Generates Mixed Reactions**: The new **Gemini model** is causing a stir with claims of enhanced performance, but users maintain a cautious stance regarding its effectiveness compared to existing models like **Sonnet**.
   - Skepticism focuses on the model's practical utility in Aider scenarios, leading to user experiences being shared for validation.
- **Sonnet Keeps Delivering**: Recent benchmarks confirm that **Sonnet** remains consistent in performance, countering previous speculations of decline.
   - Users express continued interest in the model's capabilities and reliability based on its stable benchmark scores.
- **Investment Talks Heat Up for Aider**: Community buzz surrounds potential investments in **Aider**, especially the need for a refined GUI to broaden its usability.
   - Suggestions include enhancing the leaderboard feature with user-generated data to better reflect performance metrics.
- **Long Context Models Gaining Traction**: Discussions around models that can manage **100 million tokens** could significantly impact coding workflows, with tools like **Magic dev** mentioned as game-changers.
   - User curiosity about the practical applications of these models in AI-assisted development continues to grow.
- **Swift Support Lacking in Aider**: The current lack of **Swift** support in Aider, due to the **tree-sitter** package's limitations, is causing frustration among developers.
   - Users acknowledge that adding backend support for Swift may require additional custom development efforts.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Personalization of LLMs Gains Traction**: Members expressed strong interest in **personalization** of language models, advocating for customizable personalities and **long-term memory** to enhance user interactions.
   - Concerns over high implementation costs and maintenance complexities emerged, with ideas like **RAG** (Retrieval-Augmented Generation) considered as potential solutions.
- **Crafting Chatbots with OpenAI API**: The community discussed leveraging the OpenAI API for custom **chatbot development**, addressing the requirement for programming skills and suited use cases.
   - While suggestions for no-code solutions like **Zendesk** emerged, limitations in automation and integration with systems like **Jira** were acknowledged.
- **Grok 2 Stands Out in Performance Testing**: Discussion highlighted performance comparisons between **Grok 2**, **Gemini**, and **ChatGPT**, marking Grok 2 as notably strong in **code generation** tasks.
   - Speculation on upcoming models such as Grok 3 stirred excitement, raising questions about their potential performance edge backed by robust hardware.
- **AGI Development Fuels Global Concerns**: Participants voiced apprehension regarding which nation might first achieve **AGI** and the ensuing power shift implications.
   - Emphasis was placed on the necessity for the US to maintain technological superiority to mitigate risks to global stability.
- **Challenges in CV Matching Scores**: A user reported difficulties in scoring CVs against job descriptions via API prompts, noting a perplexing score of **65** for an unrelated commercial director position.
   - Adjusting scoring parameters showed no improvement, with significant misalignment issues persisting across different engineering roles.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Inference Endpoints Are Down**: Members reported issues with **inference endpoints** likely due to a bug related to payment methods, creating urgency for fixes as production websites rely on them.
   - A pull request was opened, and the team indicated that the problem is being addressed.
- **Discussion on Training Models and Performance**: Users explored the nuances of training dialogue data with various models, discussing the effectiveness of incorporating system prompts vs learning from context.
   - Concerns arose regarding VRAM limitations for local models, leading to suggestions of using **Colab** for more robust resources.
- **Human Feedback crucial for Model Evaluation**: A paper emphasized that **human feedback** is essential for training **Large Language Models**, albeit influenced by biases.
   - The researchers highlighted that while preference scores assist evaluation, they often don't represent crucial aspects like **factuality** ([View PDF](https://arxiv.org/abs/2309.16349)).
- **Efficient Layer Pruning in LLMs**: A study reviewed a layer-pruning strategy for LLMs finding minimal performance degradation until **up to half** the layers were removed.
   - This technique involves **parameter-efficient finetuning (PEFT)** and [quantization](https://arxiv.org/abs/2403.17887) to recover model performance post-pruning.
- **FLUX LoRA Training Simplified**: A guide titled [FLUX LoRA Training Simplified](https://youtu.be/nySGu12Y05k) instructs users on utilizing Kohya SS GUI for training with an 8GB GPU.
   - The tutorial enables novices to start their training journey smoothly.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Flash Attention Faces Memory Challenges**: Users are struggling with shared memory sizes in their **flash attention kernel**, particularly a size demand reaching **131,072 bytes** for Q, raising concerns about efficiency on non-Hopper GPUs.
   - When testing with NVIDIA GeForce RTX **3090**, users encountered an `OutOfMemoryError` while using the Hugging Face example, indicating challenges in memory management with the current package version.
- **LayerNorm Kernel Updates Enhance Performance**: The integration of LayerNorm custom kernels was confirmed with the merge of [PR #169](https://github.com/linkedin/Liger-Kernel/pull/169) in the Liger Kernel repository, tested for correctness on RTX 3090.
   - Further discussions centered on dynamic dispatch for atomic operations to optimize performance in multi-GPU setups.
- **Returning to FP8 for Development**: A member is reverting to **FP8** code development to solidify their understanding and push forward on the ongoing project, feeling good about their earlier progress.
   - This suggests a focus on enhancing performance and compatibility in the current environment where further optimization is anticipated.
- **L2 Side Aware Optimization Sees Speed Boost**: The L2 Side Aware code achieved a consistent speed of **1823GB/s** for GELU forward, marking a **2% increase** over earlier performance with **x128** configurations.
   - Despite this improvement, discussions indicated a need for further simplifications to sustain optimization and reduce power consumption.
- **Community Questions Quantization Techniques**: In discussions of quantizing attention layers, members raised concerns over accuracy in QKV projections, suggesting a need for refining strategies to maintain latency in system performance.
   - Notably, issues were identified with the AWQ performance degrading when using floating point integers, prompting inquiries into optimal implementation for high performance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **IP Adapter for Flux sparks mixed reactions**: Members discussed the recent introduction of an **IP adapter for Flux**, noting mixed performance results among users.
   - *Despite varying opinions* on its effectiveness, many are still excited about this addition to their toolkit.
- **Training Models with Limited VRAM presents challenges**: Experiences were shared regarding training with limited VRAM on an **RTX 3060**, revealing that higher resolutions (like **1024**) consume huge amounts of memory.
   - It was suggested that lowering resolution can help, especially since **12GB RAM** may not be enough for complex tasks.
- **Segmentation in Image Processing raises questions**: Discussion emphasized the concept of **SEG (Segmentation)** in image processing workflows, particularly its role in systems like **ComfyUI**.
   - Members voiced confusion over its implementation, questioning its necessity compared to simpler alternatives.
- **RunwayML SD 1.5 repos vanish from platforms**: **RunwayML** has deleted all **Stable Diffusion 1.5** repos on HuggingFace and GitHub, stirring conversation on the implications of this move.
   - Users speculated if this marks a departure from **1.5 models**, which seem to have dropped in utilization.
- **SDXL vs SD 1.5 creates debate**: One user considered transitioning from **SD 1.5** to **SDXL**, balancing concerns over generation times and storage needs for their GPU.
   - Advice focused on **optimizing performance** using command line arguments to suit weaker GPU capabilities.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Amnesia Mode Reveals Professionalism in Hermes 3**: Users reported that the 'amnesia mode' in **Hermes 3** favors professionalism over casual language, limiting its conversational flexibility.
   - One user showed frustration, stating that the model maintains a 'family-friendly' demeanor, prompting speculations about its predefined behavior.
- **Training Techniques Yield Better AI Output**: Discussions highlighted that training models on outputs alone leads to better benchmarks compared to incorporating user inputs during instruction tuning.
   - Members agreed that this specific training method enhances coherence and reduces unwanted 'AI-y' responses.
- **Gradient Strategies Could Cut Communication Costs**: A user proposed leveraging low-rank approximations for gradient synchronization in distributed training to minimize communication overhead.
   - This sparked discussions on effectively combining various optimization techniques to enhance model training performance.
- **Introducing the Word Game Bench for AI Assessment**: The new 'Word Game Bench' benchmark captures language model performance via word puzzle games like Wordle, allowing unique interaction based on previous actions.
   - Community members displayed curiosity about its engaging methodology and potential for evaluating model behavior.
- **GameNGen Transforms Game Development Landscape**: _GameNGen_, the first neural model game engine, enables real-time **DOOM** simulations without conventional tools, achieving over **20 fps**.
   - Human raters struggled to differentiate between simulated and actual footage, showcasing its advanced realism potential.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **API Inference Speed Cap Discussion**: A user raised questions about capping inference speed on the API; another member noted that multiple requests using different models is viable.
   - The user prefers sticking to the same model for VRAM conservation but recognizes the limitations.
- **User Feedback on LM Studio Version 0.3**: Concerns emerged regarding the latest LM Studio update leading to reduced AI responsiveness and unusual repeated output.
   - Members suggested this might be tied to prompt settings or template parsing, advising tweaks for improvement.
- **M2 Ultra Mac ready for development**: One member set up their **M2 Ultra Mac** with **192 GB** Unified Memory for exploring LLMs, with a **2 TB** drive for storage.
   - They are also using a separate PC as a server to augment their development environment.
- **Exploring LLM performance on RTX 4090s**: Discussions highlighted running the **405b model** on **6 RTX 4090s**, yielding around **1 token per second**, influenced by offload settings.
   - One member experimented with various GPU configurations, finding memory linking can enhance speeds when models are well-distributed.
- **Impact of PCIe lane settings on performance**: Members discussed running **RTX 4090s** on gen4 x8 vs. x16 settings, examining potential impacts on speed for multi-GPU environments.
   - While gen4 x8 might not matter for single GPUs, it could hinder performance in setups with denser models.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash models now free!**: The **Gemini Flash 8B (EXP)** model is now available for use at [this link](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp), with the **Gemini Flash Experiment** also confirmed free until pricing is finalized for **AI Studio**.
   - Users celebrated the availability of **Gemini Experimental models**, marking a significant step towards broader access.
- **Cheers to Daun.ai's launch!**: Community members expressed excitement over the **Daun.ai** launch, marking it as a noteworthy addition to AI tools.
   - The enthusiasm reflects an increasing demand for innovative AI solutions in the developer community.
- **Cohere model updates spark interest**: Recent updates to **Cohere's Command R models** introduced new features and pricing changes, igniting a buzz among users eager to explore the enhancements.
   - Concerns about the handling of safety modes in **OpenRouter** were raised, highlighting the community's attention to secure implementations.
- **Experimental models hit rate limits**: Users reported **rate limit errors** while trying out experimental models, indicating challenges in accessing new features during peak use.
   - Consequential discussions arose on managing safety settings through the **API**, pointing to a need for clearer documentation.
- **Concerns over infrastructure stability**: A spate of recent **downtime issues** attributed to database capacity has prompted concerns in the community, with ongoing upgrades proposed as a solution.
   - Developers acknowledged the ongoing effects of these outages, ensuring plans are in place to enhance stability moving forward.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Embedding Weights Hit NaN Early**: A user reported that embedding weights became **NaN** just a few steps into training, likely due to a loss function denominator rounding to zero, exacerbated by a data-dependent decay term.
   - Members tracked gradients to better understand the complexity of this situation, providing insights into loss function optimization.
- **Seeking Insights on Compression Techniques**: Jeremy Vonderfecht is requesting feedback on his research involving compressing images with diffusion models like **Stable Diffusion**, recognizing the need for collaboration.
   - Members suggested using specific channels for ongoing discussions to foster constructive dialogue.
- **Dynamic Expert Routing Boosts Adaptability**: The discussion highlighted the potential of dynamic expert routing, allowing models to define their own experts during training for enhanced adaptability.
   - This is linked to ongoing research such as the methods in the [LayerSkip paper](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e).
- **Launching Word Game Bench to Challenge Models**: **Word Game Bench** is a new benchmark for evaluating language models on word games like **Wordle**, with no model surpassing a **50% win rate**; it focuses on dynamic interactions.
   - More information can be found at [Word Game Bench](https://wordgamebench.github.io) and a [tweet announcement](https://x.com/zafstojano/status/1829398835585520076).
- **Addressing Tokenization Challenges**: Participants discussed the significant limitations of tokenization, especially for non-Latin languages, and its influence on model training efficiency.
   - Concerns were raised about how tokenization can obscure crucial data features, making optimization slower.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord server celebrates 100K members!**: The Discord server has officially reached **100K members**, marking a significant community milestone, with heartfelt thanks to all members for their support.
   - The team expressed excitement for continued growth, underscoring the contributions from every member that enrich the group's atmosphere.
- **Pro API credits missing for users**: Users reported not receiving their **$5 PPLX API credits** after purchasing Pro, leading to calls for urgent support to resolve the issues.
   - Members are sharing account details for quicker resolution, emphasizing concern over the usage and accessibility of API credits.
- **Concerns over Pro Searches functionality**: There was uncertainty regarding the functionality of **Pro Searches** through the API, especially for users running **llama-3.1-sonar-huge-128k-online**.
   - The absence of **Pro** in the API left users questioning when this feature would become available.
- **Users experience API Rate Limit errors**: Several users reported encountering a **429 Client Error: Too Many Requests** when accessing the API, bringing attention to potential usage caps.
   - This situation signals underlying issues that may affect overall API functionality for engineers relying on consistent performance.
- **Feedback on AI Model behavior and performance**: Users scrutinized their AI models, noticing inconsistent outputs even after switching models, which indicated possible bugs impacting user experience.
   - Queries on model behavior sparked discussions around recent updates, suggesting a need for clarity on outputs and model identification.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **MMLU's Lack of Practical Correlation**: Members noted that **MMLU** does not correlate well with practical utility in building LLMs, highlighting outdated examples like Freud's theories, and remarked on recent model refreshes improving data relevance from the internet.
   - This sparked a discussion on the future of benchmark metrics in evaluating LLM applicability in real-world scenarios.
- **Command R+ Impresses with Updates**: Cohere announced significant performance improvements for the refreshed **Command R** and **R+** models, featuring better multilingual RAG and a cost-efficient **$0.15** per input token.
   - Members confirmed the updates are available on [Hugging Face](https://huggingface.co/) and noted the need for quantization before deployment on other platforms.
- **Cohere Chat Interface Remains Unchanged**: Users raised concerns about the **Cohere chat interface**, questioning if updates align with new model features, notably the absence of a dark mode option.
   - The call for enhancements in user interface options indicates a growing desire for improved user experience in model interactions.
- **API Trial Key Limitations Cause Frustration**: A user faced a **rate limit error (429)** using a trial API key, lamenting the **1,000 API calls/month limit**, with peers confirming the necessity for a production key.
   - The discussion emphasized the importance of optimizing API usage for enhanced performance and broader experimentation.
- **Launch of Maya LLaVA-Pretrain Dataset**: The newly available **Maya LLaVA-Pretrain** dataset contains **4,404,776** entries across **8 languages**, developed for pretraining large models, and expanded via machine translation.
   - Members expressed appreciation for addressing queries around batch processing and API capabilities related to this dataset.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Codeium bags $150M in Series C**: Codeium successfully raised **$150 million** led by General Catalyst, now valued at **$1.25 billion** with total funding reaching **$243 million** since its inception. Co-founder Varun Mohan mentioned they still have not tapped into their **$65 million** Series B funds.
   - This strategic reserve may signal a cautious approach as they navigate market demands.
- **Meta AI Assistant hits 400M MAU**: Meta's AI Assistant soared to **400 million Monthly Active Users (MAU)** and **40 million Daily Active Users (DAU)**, showcasing its expanding user base and engagement. Discussion highlighted the potential necessity for licensing as user numbers continue to rise.
   - Such metrics reflect a significant adoption rate, spurring discussions about future scaling needs.
- **Google DeepMind rolls out customizable Gems**: Google DeepMind introduced **customizable Gems**, specialized iterations of their Gemini model tailored for specific domains like **Learning Coach** and **Coding Partner**. The initiative aims to enhance user experience through targeted functionality.
   - Feedback focused on the effectiveness of these Gems and their usability in real-world scenarios.
- **Tome pivots to focus on enterprise AI**: Tome announced a shift toward becoming an AI assistant designed to help users penetrate new enterprise accounts, marking a significant change in its business focus. The news was confirmed by a company representative outlining the strategic journey.
   - Members expressed interest in how this pivot might redefine Tome's market positioning and goals.
- **New Podcast with Nicholas Carlini**: The latest episode of the [Latent Space podcast](https://x.com/latentspacepod/status/1829173832877519152) showcases insights from Nicholas Carlini of **Google DeepMind** on LLM benchmarks and extraction methodologies of training data. Key highlights involved critical perspectives on the cessation of *OpenAI logprobs*.
   - Carlini’s reflections prompted community dialogue about benchmarking practices in AI.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Potential in Blockchain Protocols**: Discussions are ongoing about using **Mojo** for blockchain protocols, with developers noting its current immaturity compared to **Go, Rust, and C++**.
   - *One developer remarked that Mojo and Go are the most competent languages, but Go's **20% performance loss** could be crucial for some projects.*
- **Questions on Mojo's Open Source Future**: Inquiries arose about the availability of the **Mojo compiler's source code**, which remains closed source for now.
   - *The Modular team indicated they may not know when or if it will be open-sourced while balancing development speed with community engagement.*
- **Performance Comparison Insights**: Members debated the performance of **Go** versus **C**, highlighting Go's limitations in various tasks.
   - *Darkmatter pointed out that Go's performance may significantly drop, citing **30 requests per second** capacity compared to **C's 100**.*
- **Architect's Role in Memory Management**: A member argued that if a programmer is unsure about memory management, it signifies a flaw in the system's design.
   - *They emphasized the need for solid architectural design to minimize concerns for application programmers.*
- **Exciting Export Ideas for Fastai**: A proposed enhancement involves overriding **Learner.export** in fastai to export **Mojo** code along with the **PyTorch model**.
   - *This tactic could improve integration between the input pipeline and the model for streamlined production use.*



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Embraces Function Calling & Streaming**: A member struggled with using **LangChain v2.0** for function calling and streaming, citing documentation gaps. Another clarified that function calling is supported, but streaming outputs need careful setup in JavaScript.
   - Exploring resources like the [AgentExecutor documentation](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html) might help clarify configurations.
- **Docker Tales: Ollama Connection Woes**: One user faced a connection refusal error with their LangChain app in Docker while trying to use the **Ollama API**. They later resolved it by correcting the base URL to a direct **Ollama host URL**.
   - This issue highlights the importance of proper URL settings in containerized environments, especially when leveraging tools like Docker.
- **Custom GPT for HR Sparks Ideas**: A user expressed a desire to create a specialized **GPT** for their HR team, targeting hallucination reduction and feedback mechanisms. The discussion turned toward enhancing LLM interactions with fine-tuning and RAG techniques.
   - Implementing feedback loops could significantly improve performance, especially when adapting existing manual content.
- **Challenges with LangChain Streaming Outputs**: A user reported difficulties with LangChain agent executors that collect outputs before the final response is delivered, rather than streaming in real-time. Suggestions emerged to utilize the `streamRunnable` option for real-time output delivery.
   - Leveraging this feature could streamline response times, enhancing user experience in real-time applications.
- **GraphRAG vs Traditional RAG: A Preference Battle**: Discussion emerged around the effectiveness of hybrid RAG methods, with a member favoring **traditional RAG** techniques for their process. They pointed out that exploring new methods like self-query and large context RAG might prove worthwhile.
   - This conversation potentially opens the door to more advanced exploration in RAG methodologies for response enhancement.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GymNation partners with LlamaIndex for success**: GymNation partnered with LlamaIndex, resulting in a **20% increase** in digital lead to sales conversion and an **87% conversation rate** with digital leads. For more details, check their [full success story](https://t.co/CXsiySj4zq).
   - *Remarkable outcomes* showcase how LlamaIndex enhances user engagement effectively.
- **LLMs in Production insights shared**: An upcoming discussion on **September 9th** will feature insights on deploying LLMs effectively. Details are available [here](https://t.co/Ozb1xTF2Lh).
   - Attendees can expect *practical tips* on real-world LLM applications.
- **MLFlow Podcast features LlamaIndex**: Co-founder discussed the **MLFlow integration** with LlamaIndex on their podcast, focusing on streamlined logging and evaluating applications. Watch the demo and insights [here](https://t.co/2wwvn7HRBm).
   - *Powerful enhancements* in managing AI applications were showcased during the session.
- **LLM x Law Hackathon announced**: An **LLM x Law Hackathon** on **September 8th** invites participants to explore AI in legal practices. More information can be found [here](https://t.co/AksB9V6akr).
   - This event will feature *multiple tracks*, emphasizing innovation in AI-legal integrations.
- **Financial Data Analysis with MoW**: Innovative financial data analysis employing **Mixture of Workflows (MoW)** and Corrective RAG was discussed, utilizing models like **Phi-3**, **Qwen-2**, and others. Further details can be found [here](https://t.co/CIaEwmWB0S).
   - This method provides **context-aware analysis** of financial statements, promising better insights.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **House Party Next Week**: Join us for a **House Party** next week at an earlier time to boost community engagement! [Join the Discord Event](https://discord.gg/open-interpreter-1146610656779440188?event=1278796923892924498).
   - This event aims to create a fun atmosphere and encourage discussions about **potential applications**.
- **Seeking Terminal App Suggestions**: A member is looking for alternatives to the **Konsole** terminal app on KDE due to screen bleeding issues. Users reported similar problems while using **GPT-4o-mini** in standard terminal setups.
   - This highlights ongoing concerns about terminal performance in high-demand environments.
- **Obsidian OI Plugin Installation Help Needed**: A user praised resources on the **Obsidian OI plugin** but is struggling with global installation issues. They were advised to share their installation details in a designated channel for further support.
   - This reflects a collaborative effort within the community to resolve technical challenges.
- **GameNGen: A Leap in Game Simulation**: _GameNGen_ now simulates **DOOM** at over **20 frames per second** using a neural model, showcasing exceptional performance on a single TPU, with a PSNR of **29.4**.
   - The experience left human raters hard-pressed to tell apart real gameplay from its simulations, marking a significant advancement in game technology.
- **Excitement for AgentOps Developments**: Members are buzzing with enthusiasm for upcoming initiatives from **Adam and the AgentOps** team. This excitement underlines the community's interest in next-gen agent tech breakthroughs.
   - This anticipation signals a healthy dialogue about the future prospects in smart agent systems.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Google's GPU Acquisition Sparks Curiosity**: Members questioned why **Google** is purchasing **GPUs** from **NVIDIA** despite their own **TPUs**, suggesting a potential gap or interest in NVIDIA technologies.
   - *Is the TPU not enough?* One member mused about Google's strategic choices in hardware.
- **RunwayML Deletes All Stable Diffusion Repos**: Discussion erupted over **RunwayML** deleting all their **Stable Diffusion 1.5** repositories on **HuggingFace** and **GitHub**, leaving many users frustrated.
   - One member noted that this action broke many functionalities in **Diffusers 1.5**, particularly impacting single file loading.
- **Disruption from Repo Deletions**: Members expressed annoyance about the seemingly thoughtless nature of RunwayML's deletions, with one stating it felt like they wanted to cause **disruption**.
   - Speculation arose around potential legal issues, but no specific reasons were confirmed for the deletions.
- **Generating Realistic Images for Book Covers**: A member sought advice on generating **comic book-style** or cartoonish images for their novel covers, struggling with overly realistic outputs from **DALL·E**.
   - Despite attempts, they found DALL·E not catering to the specific style they desired.
- **Launch of Re-LAION-5B**: Members celebrated the launch of **Re-LAION-5B**, a cleaned version of the **LAION-5B** dataset, which addresses previous concerns following a [safety revision procedure](https://laion.ai/blog/relaion-5b/).
   - The dataset was updated in partnership with key organizations to ensure safety and compliance, marking a significant milestone.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tech Giants Eye OpenAI**: Nvidia, Apple, and Microsoft are in discussions to invest in **OpenAI** as part of a new **$100 billion funding round** [source](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round). This move indicates strong interest in driving AI funding and innovation from major players.
   - *Chatbot wars are heating up* as these companies jockey for pivotal stakes in AI development.
- **Chatbot Wars Heat Up**: **ChatGPT** has surpassed **200 million weekly users**, posing a challenge for rivals like **Meta AI**, which is also increasing its market traction [source](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx). This competitive landscape raises questions about user engagement and effectiveness of different platforms.
   - Concerns exist regarding the real utilization of **Meta AI**, as only **40 million DAUs** could suggest accidental engagement with its offerings.
- **Tinygrad Launches Affordable Cloud Solution**: Tinygrad introduced a new cloud service featuring a **4090 GPU** and **500 GB of storage** for only **$60/month**, significantly undercutting competitors like Vast AI [source](https://x.com/__tinygrad__/status/1829379908017238210?s=46). This new model promises a cost-effective solution for developers looking to leverage advanced hardware.
   - *Coming soon: CLOUD=1* enables users to operate Tinygrad locally while taking advantage of cloud processing speed for efficient handling.
- **Inquiry on System Prompts Impact**: Members are probing into the **impact of system prompts** on evaluation scores, sparking interest in whether different prompting techniques can significantly adjust results. There’s a call for research papers to support this exploration.
   - This inquiry highlights the ongoing desire to refine AI performance metrics through thoughtful prompt design.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QLoRA Faces Memory Puzzles**: Concerns arose as a member questioned the memory sufficiency for **QLoRA** after encountering a **CUDA error** indicating illegal memory access with **4 48GB GPU cards**.
   - This highlights potential pitfalls in hardware setup that need careful consideration when configuring memory resources.
- **A6000 GPUs Get Confused**: Clarifications confirmed that **A6000 GPUs** have been upgraded to **48GB**, thus ensuring four of these cards should meet the required capacity.
   - Members suggested CPU offloading and sequence length adjustments could additionally impact memory distribution during training.
- **Training Sequence Lengths Under Scrutiny**: A member experimented with different training sequence lengths (**8K** and **4K**), indicating how these variations may affect **vRAM** usage.
   - Probing into these specifics showcases the essential balancing act between sequence configuration and memory demands.
- **Interest in Multi-GPU Evaluation**: Inquiries about the existence of **multi-GPU evaluation** support in **TorchTune** suggest a keen interest in optimizing performance.
   - This reflects a broader trend where AI engineers seek scalability and efficiency in handling demanding training setups.
- **Debugging CUDA Errors for Data Integrity**: A member received debugging tips such as setting **CUDA_LAUNCH_BLOCKING=1** to address illegal memory access errors during training.
   - This points to the ongoing complexities of executing distributed training with **PyTorch** while managing memory constraints effectively.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Confusion Over Repo Connections**: A member expressed confusion about the connection between their statement and the [GitHub repository](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI), clarifying that the repo was separate but showcased to inspire community involvement.
   - *It’s getting over 2k likes each day*, indicating significant interest in the **LinkedIn Auto Jobs Applier** tool.
- **Concerns on LinkedIn Tool Performance**: Another member raised concerns regarding the performance of the **LinkedIn Auto Jobs Applier**, pointing to GitHub issues that reveal room for improvement.
   - This highlights ongoing feedback suggesting there’s still much to enhance in the tool's capabilities.
- **Workshop for Reliable AI Agents**: A member shared a link to the [YouTube video](https://www.youtube.com/live/-aKRsvgDEz0) for a workshop focusing on **Useful and Reliable AI Agents**, which tackles accuracy, reliability, and cost-effectiveness.
   - The workshop addresses the active research on AI agents and their effective utilization in real-world applications.
- **AgentOps Tools for AI Development**: [AgentOps](https://agents.staf.ai/AgentOps) offers resources for building agents, featuring tools that streamline the development process by eliminating guesswork in prompting.
   - This transparency aims to enhance how developers approach AI solutions.
- **DSPy Seminar at Bay Area AI Meetup**: The upcoming Bay Area AI meetup will feature Michael Ryan discussing **DSPy: Prompt Optimization for LM Programs**, showcasing his work on the MIPROv2 algorithm.
   - The meetup is sponsored by [Neo4j](https://x.com/ChiefScientist/status/1829231009344434400?t=wow3U2BluHEv16-MI2YcaQ&s=19) and promises to deliver valuable insights.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl GitHub Docs Needs Dark Mode**: A member requested the [Axolotl GitHub documentation](https://github.com/axolotl) to offer a **dark mode**, citing discomfort with the current light mode during frequent visits.
   - They emphasized challenges with checking configuration parameters in the current theme.
- **Hardware for Training LLaMA 70B**: Discussion revolved around the **hardware requirements** for training the **LLaMA 70B** model, with speculations that only a few **NVIDIA A6000 GPUs** might be needed.
   - A member confirmed that **3x A6000 GPUs** should be sufficient for training the full model, highlighting potential advancements in GPU capabilities.
- **Llama 3.1 Still Struggles with Special Tokens**: Concerns were raised about **Llama 3.1 base** still experiencing issues with uninitialized special tokens and out-of-distribution embeddings.
   - Members expressed ongoing challenges with managing special tokens, which could impact model performance.
- **Potential Fix for Untrained Tokens**: A new option, `fix_untrained_tokens: true`, was introduced to address uninitialized special tokens in Llama 3.1, signaling a step towards improvement.
   - This fix reflects a continued effort to refine model interactions and performance.
- **New Assistant Prefill Feature Launch**: The recent [Pull Request #33198](https://github.com/huggingface/transformers/pull/33198) at **Hugging Face** adds a long-requested **assistant prefill** feature that automatically initiates model responses.
   - This update aims to enhance user experience in the **TextGenerationPipeline**, employing a creative approach to response generation.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Groq Waits for Leaderboard PRs**: **Groq** has not yet been added to the leaderboard as the team is still waiting for their PRs, expected around next week.
   - This delay sparked discussions about their integration and anticipated performance implications.
- **Model Steps Documentation is Essential**: A member confirmed that documenting model steps is crucial for reproducibility, enhancing model understandability.
   - Proper documentation ensures usability and minimizes confusion during model implementation.
- **Java Test Case Reveals GIS Issues**: A user reported performance issues in a **Java** test case related to GIS geometry initialization.
   - They concluded that simpler direct examples might serve better than complex function calls, given user queries.
- **Queries on Evaluation Temperature Settings**: Members questioned if evaluations are conducted with a greedy decode and temperature of 0 for fair metrics.
   - Discussions referenced recent GitHub links on leaderboard evaluation criteria, contemplating randomness in output.
- **OSSHandler Default Parameters Discussed**: The default temperature for **OSSHandler** is set at 0.001, and adjustments were briefly considered but ultimately rejected.
   - This choice aligns with maintaining consistent function outputs and overall model performance optimization.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Exploring tinygrad's limitations**: *codeman3786* questioned if **tinygrad** is effective for **statically scheduled operations** but struggles with **semi-structured sparsity** options. George Hotz's invitation for specific examples of tinygrad's shortcomings highlights community curiosity about its operational limits.
   - The ensuing discussion signals a shared interest in dissecting the real-world applicability of tinygrad, especially in the context of complex data handling.
- **Tensor.cat's trouble with sharded tensors**: A user ran into issues using **Tensor.cat** with sharded tensors, receiving an error about *padding not supported*. They devised a workaround utilizing `unsqueeze`, but additional reshaping errors kept cropping up.
   - This indicates a need for clarity on whether the limitation stems from core functionality or is merely unsupported behavior, as the user considers adapting the code for batch dimension support.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1278796860781232260)** (459 messages🔥🔥🔥): 

> - `Fine-Tuning vs RAG`
> - `Use Cases for LLMs`
> - `Quantization Techniques`
> - `Model Training Challenges`
> - `Hopfield Networks` 


- **Fine-Tuning and Hallucinations**: It is often debated whether RAG is better at reducing hallucinations compared to fine-tuning; however, some participants argued that neither is definitively superior and that controlled overfitting is an essential consideration in training.
   - The effectiveness of fine-tuning is influenced by the dataset size and the model's hyperparameters, such as rank and alpha, which define how weights are trained and their influence on learning.
- **Use Cases for LLMs**: Participants discussed various applications of LLMs, highlighting that companies like AT&T utilize models for customer service, while others use them for proprietary research and search functionalities.
   - It was noted that many enterprises use instruction-based models similar to GPT for effective deployment in real-world tasks.
- **Quantization Techniques**: There were discussions about quantization types for model inference, specifically the current support for 4-bit loading, while 8-bit support remains absent.
   - The conversation delved into the effects of varying rank sizes in quantization, where higher ranks may offer better results in model training, particularly with respect to stability and accuracy.
- **Challenges in Model Training**: Many participants expressed the importance of understanding the dynamics of model training, emphasizing the necessity of experimenting with different techniques to find optimal configurations.
   - Training models involves a lot of trial and error, and sharing knowledge about successful approaches is vital for newcomers to navigate the complexities of fine-tuning.
- **Hopfield Networks and Memory**: Hopfield networks were referenced as foundational models for associative memory, with one participant sharing a YouTube video that discusses their principles and applications.
   - The humor about memory decay and the utility of such networks in comparison to newer models showcased a blend of nostalgia and contemporary relevance in neural network discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mathstral/">MathΣtral</a>: As a tribute to Archimedes, whose 2311th anniversary we're celebrating this year, we are proud to release our first Mathstral model, a specific 7B model designed for math reasoning and scientific disc...</li><li><a href="https://arxiv.org/abs/2405.05904">Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?</a>: When large language models are aligned via supervised fine-tuning, they may encounter new factual information that was not acquired through pre-training. It is often conjectured that this can teach th...</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora">In-depth guide to fine-tuning LLMs with LoRA and QLoRA</a>: In this blog we provide detailed explanation of how QLoRA works and how you can use it in hugging face to finetune your models.</li><li><a href="https://tenor.com/view/fumo-touhou-fumo-touhou-gif-23545090">Fumo Touhou GIF - Fumo Touhou Fumo Touhou - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://www.kaggle.com/code/mohsenghafari/kaggle-mistral-7b-unsloth">Kaggle Mistral 7b Unsloth &#x645;&#x62D;&#x633;&#x646; &#x6A9;&#x631;&#x647;</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/posts/dylanebert/255000504996462">@dylanebert on Hugging Face: &quot;Here&#39;s a 1-minute video tutorial on how to fine-tune…&quot;</a>: no description found</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows</a>: Research update on ultra-long context models, our partnership with Google Cloud, and new funding.</li><li><a href="https://tenor.com/view/orange-cat-smile-cat-smile-orenge-cat-smiling-gif-23133369">Orange Cat Smile Orenge Cat Smiling GIF - Orange Cat Smile Cat Smile Orenge Cat Smiling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts">Fine Tuning Is For Form, Not Facts | Anyscale</a>: Fine tuning is one approach to domain-specific model refinement (DSMR), but it’s not a silver bullet for improving domain-specific performance.</li><li><a href="https://www.youtube.com/watch?v=1WPJdAW-sFo">A Brain-Inspired Algorithm For Memory</a>: Get 20% off at https://shortform.com/artemIn this video we will explore the concept of Hopfield networks – a foundational model of associative memory that un...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers</li><li><a href="https://github.com/mlabonne/llm-autoeval?">GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab</a>: Automatically evaluate your LLMs in Google Colab. Contribute to mlabonne/llm-autoeval development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/57">Benchmark against unsloth · Issue #57 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch hey, did you run any benchmark against unsloth which uses similar kernels? I guess your project can be used as a dropdown replacement with multi gpu support. Alt.....</li><li><a href="https://x.com/BramVanroy/status/1827090122363564251">Tweet from Bram (@BramVanroy)</a>: @hsu_byron Is this stable? If so, a downstream integration with @huggingface trainer would be extremely valuable :o I&#39;d need be through accelerate cc @TheZachMueller maybe</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: SmolLM 135M Instruct Trained on DEVINator Data for Open Hands</li><li><a href="https://github.com/unslothai/unsloth/issues/636">Storing models to huggingface is not working · Issue #636 · unslothai/unsloth</a>: Hello, I think instructions for storing model to hugging face are not very clear. Following line in notebook tries to push model to HF model repository (&quot;hf/model&quot;, tokenizer, quantization_m...</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_pos">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)!</a>: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)! - e-p-armstrong/augmentoolkit</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_post_comment-text">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.</a>: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...</li><li><a href="https://github.com/unslothai/unsloth/pull/974">Fix for multi gpu setup training with a single GPU. by Sehyo · Pull Request #974 · unslothai/unsloth</a>: check_nvidia() originally spawns a new process for nvidia-smi, thus bypassing that GPU count might be limited by an OS environmental variable as this won&amp;#39;t be reflected in the new process. Add...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1278854896661168159)** (12 messages🔥): 

> - `Easy AI Training Scripts`
> - `Upcoming Models from Meta`
> - `OpenAI's New Pricing for GPT-4o`
> - `Gemini 2.0 Updates`
> - `LLM Providers as Cloud Services` 


- **Simplifying AI Training with One Script**: A member is creating **2 scripts** that allow anyone to train AI easily on local or cloud setups without using complex libraries like Unsloth or Deepspeed.
   - The scripts require minimal dependencies, and specific instructions for running them were shared along with a link to the [text generation web UI](https://github.com/oobabooga/text-generation-webui).
- **Meta's Upcoming Model Reveals**: Discussion about **Meta** potentially announcing updates and the next **Llama models** soon, though it's unclear if it will include **Llama 4**.
   - Speculation suggests the release may feature **multimodal Chameleon-type models**.
- **OpenAI's New GPT-4o Pricing**: The new **GPT-4o model** has been announced with significantly reduced costs of **4$ per 1M tokens** for input and **33% cheaper** for output tokens.
   - This model also supports **Structured Outputs**, allowing outputs to adhere strictly to JSON Schemas.
- **Gemini 2.0 Sparks Interest**: **Gemini 2.0** was referenced with excitement, suggesting it may be related to experimental models within **AI Studio**.
   - A user pointed towards a [Reddit post](https://www.reddit.com/r/Bard/comments/1f4xamv/wow_gemini_20/) discussing the new features of Gemini 2.0.
- **LLM Providers as App Store Models**: One user compared **LLM providers** like Anthropic and OpenAI to the **App Store** model, implying they prefer developers to create applications instead of taking a cut on sales.
   - This led to discussions about the similarities with **cloud services** like **Firebase**, indicating a broader trend in monetizing access to models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1820987573793386527?utm_campaign=The+Batch&utm_source=hs_email&utm_medium=email">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Our newest GPT-4o model is 50% cheaper for input tokens and 33% cheaper for output tokens.  It also supports Structured Outputs, which ensures model outputs exactly match your JSON Schemas.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f43ep8/meta_to_announce_updates_and_the_next_set_of/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/Bard/comments/1f4xamv/wow_gemini_20/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>: A Gradio web UI for Large Language Models. Contribute to oobabooga/text-generation-webui development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1278799638295744522)** (67 messages🔥🔥): 

> - `Learning Rate Scheduler`
> - `GPU Rental vs. Ownership`
> - `DPO Model RAM Optimization`
> - `Fine-tuning Parameters`
> - `Tokenizer Management` 


- **Understanding Learning Rate Scheduler Effects**: A member inquired about how the cosine learning rate scheduler with warmup steps impacts the LR graph during training.
   - The discussion highlighted the importance of observing graceful decay in learning rates for better model performance.
- **Debate on Renting vs. Owning GPUs**: Members delved into the advantages of renting GPUs over owning them, arguing that renting is significantly cheaper operationally.
   - One user emphasized that rental options allow flexibility and cost-effectiveness, especially for occasional use.
- **Optimization Tips for DPO Models on Limited RAM**: Several members discussed encountering out-of-memory (OOM) errors when trying to run a DPO model on systems with limited RAM, such as Colab's T4 with 16GB.
   - General advice included reducing batch size and sequence length, but some noted that DPO models demand more VRAM compared to regular fine-tuning.
- **Parameter Tuning for Fine-tuning**: A user sought clarification on how to choose parameters for training models, especially regarding batch size based on available memory.
   - Insights noted that lower batch sizes may be necessary when working under strict memory constraints, particularly with models requiring larger context lengths.
- **Managing Tokenizer After Training**: A question arose about when to push a tokenizer after training a model, with the consensus favoring pushing changes only when new tokens are added.
   - Members discussed that if the tokenizer remains unchanged during training, it is not necessary to push updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: Fine-tune and run Meta&#x27;s updated Llama 3.1 model with 6x longer context lengths via Unsloth!</li><li><a href="https://hastebin.com/share/ilelinosan.python">Hastebin</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1278885321509179485)** (4 messages): 

> - `OpenRouter Launch`
> - `Llama 3.1 Model` 


- **OpenRouter Stealth Launch Goes Live!**: After weeks of effort, the product is now live on [OpenRouter](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers) with real users, serving **Llama 3.1-405B-instruct** with **128k context** and function calling support.
   - The pricing is **$2.5/mil tokens**, making it the cheapest option available.
- **Payment Clarified Despite Link Usage**: The member clarified that they receive payment regardless of whether users access the service through their link or not, emphasizing pride in building the infrastructure.
   - *“I don’t make any extra money or commission or referral or anything”* was mentioned to highlight the focus on the effort rather than commission.



**Link mentioned**: <a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers">Meta: Llama 3.1 405B Instruct – Provider Status</a>: See provider status and make a load-balanced request to Meta: Llama 3.1 405B Instruct - The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, th...

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 messages): 

hamchezz: I want to finetune a llm on some undefined goal just because 😄
  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1278793009789534239)** (285 messages🔥🔥): 

> - `Gemini Model Performance`
> - `Sonnet Benchmark Updates`
> - `Investment in Aider`
> - `Long Context Models`
> - `Coding with AI Tools` 


- **Discussions on the Gemini Model**: The new Gemini model is generating excitement, though some users express suspicion about its effectiveness for Aider usage compared to other models.
   - Users are keen to verify the claims of improved performance while sharing experiences and skepticism regarding its real-world applications.
- **Updates on Sonnet's Performance**: Recent benchmarks indicate that Sonnet continues to perform well without significant degradation, despite rumors suggesting otherwise.
   - Users remain interested in Sonnet's capabilities, especially in the context of its current performance metrics.
- **Potential for Investment in Aider**: Community members speculate about Aider's future and potential investment interest, contemplating the benefits of a polished GUI version for wider appeal.
   - Some suggest that Aider's leaderboard functionality could be improved by incorporating user-generated data to provide a more accurate performance assessment.
- **Exploration of Long Context Models**: There are ongoing discussions about a model capable of reasoning with 100 million tokens, potentially transforming coding tasks and AI integration.
   - Users express curiosity about emerging tools like Magic dev and their implications for future AI-assisted software development.
- **Impact of AI Tools on Coding Profession**: Microsoft CEO Satya Nadella highlights GitHub Copilot's success, suggesting it has surpassed previous GitHub revenue benchmarks in total user contributions.
   - The discussion underscores the growing dependence on AI tools among developers, emphasizing their impact on productivity and coding efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/openaidevs/status/1823510395619000525?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: This model is also now available in the API as `chatgpt-4o-latest`. We recommend `gpt-4o-2024-08-06` for most API usage, but are excited to give developers access to test our latest improvements for c...</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet seems as good as ever</a>: Sonnet’s score on the aider code editing benchmark has been stable since it launched.</li><li><a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/homer-brain-monkey-gif-11098413">Homer Brain GIF - Homer Brain Monkey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.continue.dev/features/codebase-embeddings">Codebase Retrieval | Continue</a>: Talk to your codebase</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows</a>: Research update on ultra-long context models, our partnership with Google Cloud, and new funding.</li><li><a href="https://github.com/nu">Nu Deployment</a>: Nu Deployment has 3 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 30.67% tasks (pass@1) in SWE-bench lite and 38.40% tasks (pass@1) in SWE-bench verified with each task costs less than $0.7.</a>: A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 30.67% tasks (pass@1) in SWE-bench lite and 38.40% tasks (pass@1) in SWE-bench verified wi...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1278853802145091595)** (69 messages🔥🔥): 

> - `Aider and Swift Language Support`
> - `Automating Command Entry in Aider`
> - `File Detection in Aider`
> - `Repo Size Impact on Aider Performance`
> - `Using GitHub Copilot with Aider` 


- **Aider struggles with Swift language support**: A user inquired about adding **Swift** support to Aider, but another member pointed out that the **tree-sitter** package does not parse Swift files. They referenced documentation indicating that Aider has limitations with certain languages.
   - Further discussion led to the realization that augmenting the repo-map for new languages may require additional effort or custom implementation.
- **Automating Commands in Aider**: A member expressed frustration with Aider providing a list of commands instead of executing them, comparing it to Cursor Compose functionality. They were advised to use different LLM models like *Sonnet* or *gpt-4o* for better results.
   - It was noted that using `aider --deepseek` could help streamline some processes, but users still desired a more integrated experience.
- **Detecting Files Automatically in Aider**: A user asked how to refresh Aider to automatically detect newly created files rather than using the `/add` command. Although commands like `/drop` and `/clean` were discussed, it was concluded that manual addition via `/add` was necessary.
   - A few users confirmed that the autocomplete feature could suggest files once they were recently created, but noted there may be some git-related limitations.
- **Repo Size and Aider Performance**: A user raised a question about the size at which Aider struggles with repo complexity, prompting discussions about experiences with larger repos like *Wine* and blockchain code bases. Members emphasized that managing focus is critical for making changes in larger repositories.
   - Aider performs better on files relevant to the task, and users were encouraged to avoid overwhelming the model with unnecessary files to maintain efficiency.
- **Potential Use of GitHub Copilot API with Aider**: A user asked whether Aider could theoretically use the **GitHub Copilot API**, as their organization has approved Copilot but not other LLMs. This highlights the complexities of organizational approval processes for various AI tools.
   - The intersection of using Aider alongside widely accepted tools like Copilot could pave the way for more flexible integrations in corporate environments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>: Manage your keys or create new ones</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>: Tips for AI pair programming with aider.</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: Using a .env file to store LLM API keys for aider.</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://aider.chat/docs/languages.html#how-to-add-support-for-another-language">Supported languages</a>: Aider supports pretty much all popular coding languages.</li><li><a href="https://github.com/paul-gauthier/grep-ast/issues/7">`py-tree-sitter-languages` is unmaintained · Issue #7 · paul-gauthier/grep-ast</a>: Hi @paul-gauthier , thanks for your work on aider. I&#39;ve been having a blast using it. This project uses https://github.com/grantjenks/py-tree-sitter-languages, but that project is unmaintained and...</li><li><a href="https://github.com/ChimeHQ/SwiftTreeSitter">GitHub - ChimeHQ/SwiftTreeSitter: Swift API for the tree-sitter incremental parsing system</a>: Swift API for the tree-sitter incremental parsing system - ChimeHQ/SwiftTreeSitter
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1278937913689899091)** (1 messages): 

> - `Anthropic Prompt Engineering`
> - `Jupyter Notebooks`
> - `uvx tool`
> - `Anthropic API`
> - `Documentation quality` 


- **Explore Anthropic's Prompt Engineering Tutorial**: Check out [Anthropic's Prompt Engineering Interactive Tutorial](https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/) that showcases their documentation prowess through Jupyter notebooks.
   - It's noted that Anthropic continues to lead in **documentation quality** among LLM vendors.
- **Setting Up Jupyter with uvx Made Easy**: Implementation of Jupyter notebooks was described using **uvx**, demonstrating how to set up a server swiftly by running a few commands.
   - Using `git clone` followed by `uvx --from jupyter-core jupyter notebook courses` started the Jupyter server and opened the browser almost instantly.
- **Basic Prompt Demonstrations via Anthropic API**: The tutorial begins with fundamental chapters displaying basic prompts executed through the **Anthropic API** using `%pip install anthropic` for package management.
   - This emphasizes the importance of keeping installations organized in the correct virtual environment.
- **Engaging with Anthropic's Community**: A user actively contributed to Anthropic's community by filing an issue and creating a pull request on their GitHub course repository.
   - This demonstrates the importance of community engagement and collaboration in software development.



**Link mentioned**: <a href="https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/">Anthropic’s Prompt Engineering Interactive Tutorial</a>: Anthropic continue their trend of offering the best documentation of any of the leading LLM vendors. This tutorial is delivered as a set of Jupyter notebooks - I used it …

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1278796810671882260)** (318 messages🔥🔥): 

> - `Personalization of LLMs`
> - `OpenAI API for chatbots`
> - `Grok 2 performance`
> - `AGI development concerns`
> - `Creating custom AIs` 


- **Discussion on Personalization of LLMs**: Members emphasized the desire for **personalization** in AI, such as customizable personalities and long-term memory for meaningful interactions. They discussed the feasibility and challenges of implementing these features in a user-friendly manner.
   - Concerns were raised about the potential high costs and complexities of maintaining personalized AI, with ideas like **RAG** (Retrieval-Augmented Generation) being considered.
- **Chatbot Development Using OpenAI API**: A conversation ensued about building custom chatbots using the OpenAI API, highlighting the need for programming skills and understanding of specific use cases. Members pointed out existing no-code solutions like **Zendesk**, but acknowledged limitations in automation and support capabilities.
   - Key features for effective chatbots were outlined, including local vector databases and integration with existing systems like Jira and Sharepoint.
- **Performance Comparisons of AI Models**: Users compared the performance of various models, including **Grok 2**, **Gemini**, and **ChatGPT**, noting differences in code generation capabilities. It was suggested that Grok 2 was surprisingly effective, while some members expressed disappointment with model outputs on specific coding tasks.
   - The community speculated on the upcoming releases of new models, like Grok 3 and others, considering their potential performance and the advantages of large-scale hardware setups.
- **Concerns About AGI Development**: Participants expressed concerns about the implications of which country achieves **AGI** first, particularly regarding global power dynamics. There was a consensus that AGI development should be carefully monitored to prevent monopolization by any entity.
   - Discussions highlighted the necessity for countries like the US to maintain a lead in AI technology to prevent any adverse effects on global stability.
- **Creating Custom AIs**: Members provided insights on how to create custom AIs, recommending starting with simpler projects before tackling LLMs. Suggested resources included **TensorFlow**, **Colab**, and beginner-friendly models like image upscalers.
   - Encouragement was given for individuals to focus on programming skills and foundational knowledge in AI development.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

smilebeda: 👍
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 messages🔥): 

> - `Prompt Engineering Discussions`
> - `Job Description Matching`
> - `API Utility for Document Analysis`
> - `Deep Document Analytics`
> - `Batch Processing` 


- **Job Description Matching Scores**: A user described challenges in scoring resumes against job descriptions via prompts, noting specific cases where the API returned unexpected similarity scores.
   - One example included a commercial director position where a candidate received a score of **65** despite being an engineering student in IoT.
- **API Design for Document Analysis**: Another user inquired whether to use multiple API calls or a single prompt for extracting various details from large documents, such as summaries and application information.
   - A suggestion was made that separate requests would help minimize hallucinations and enhance coherence.
- **Batch Processing Discussion**: A community member recommended exploring [batch processing](https://platform.openai.com/docs/guides/batch/getting-started) to improve efficiency.
   - The context included discussions about minimizing responses' complexity by handling questions separately.
- **Seeking Deep Document Analytics Discussions**: A user expressed interest in discussing techniques for deep document analytics and plans for fine-tuning after collecting sufficient ChatGPT data.
   - They asked for guidance on available spaces for this topic within the community.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 messages🔥): 

> - `Prompt Engineering for CV Matching`
> - `Document Analytics with ChatGPT` 


- **Prompt adjustments lead to incorrect scoring**: A user adjusted their prompt for evaluating CVs against job descriptions but still received inaccurate similarity scores, such as a **65** for completely unrelated qualifications like that of an engineering student for a Commercial Director role.
   - Adding strict scoring rules didn't help either, as a Cloud Engineer received a score of **5** despite relevant experience due to misalignment in job focus.
- **Reducing hallucinations with separate API calls**: A user inquired whether to use multiple queries for extracting information from large documents, leading to a suggestion that separate requests minimize chances of hallucinations.
   - It was noted that larger, complex prompts may hinder coherent responses, supporting the idea of breaking inquiries into smaller, clearer segments.
- **Exploring batch processing for efficiency**: One user mentioned the potential benefits of batch processing in API calls to streamline operations, providing a helpful [link](https://platform.openai.com/docs/guides/batch/getting-started) for guidance.
   - Another user expressed interest in using ChatGPT responses as a starting point for fine-tuning, indicating a longer-term goal of improving document analytics.
- **Engagement in deep document analytics discussions**: A user asked about platforms for discussing deep document analytics, particularly in relation to gathering data for model fine-tuning.
   - They were directed to a specific channel dedicated to the topic, suggesting community support for their exploration.


  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1278799856135311360)** (223 messages🔥🔥): 

> - `Inference Endpoints Issues`
> - `Training Models`
> - `Video Processing`
> - `LLMs and AI Projects`
> - `AI Powered Applications` 


- **Inference Endpoints Are Down**: Members reported issues with inference endpoints likely due to a bug related to payment methods, creating urgency for fixes as production websites rely on them.
   - A pull request was opened to address the issue, and a response was received indicating that the problem was being looked into.
- **Discussion on Training Models and Performance**: Users explored the nuances of training dialogue data with various models, discussing the effectiveness of incorporating system prompts vs learning from conversation context.
   - Concerns were raised regarding the limitations of running models locally due to VRAM constraints, leading to suggestions of using Colab for more powerful resources.
- **Challenges with Video Processing and Uploads**: One member shared a strategy of chunking video files into smaller sizes for uploading to Hugging Face, acknowledging the limitations of file sizes when using Git LFS.
   - The group discussed experiences with video processing speed and resource usage, noting challenges encountered when running certain models.
- **Exploration of AI-Powered Applications**: Members expressed interest in the practical applications of AI, citing examples such as automating ID card creation through model training.
   - There were insights shared about integrating AI with other technologies, showcasing potential imaginative uses of AI in real-world projects.
- **Mood and Community Engagement**: Members celebrated their achievements and shared enthusiasm for their projects in AI development, promoting camaraderie and collaboration.
   - Conversations highlighted the fun and excitement surrounding experimenting with AI, with references to popular culture icons like J.A.R.V.I.S from Iron Man, sparking creativity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NM_Reid/status/1825997577151525338">Tweet from Noah Reid (@NM_Reid)</a>: uhh, anaconda just sent a message to our HPC admins that we&#39;re in violation their ToS and we now need to pay for a license or remove all their software from our system?</li><li><a href="https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to">torch.Tensor.to &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/repositories-licenses">Licenses</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=IW7jFq3vQbw">AI process thousands of videos?! - SAM2 deep dive 101</a>: Build your own SAM2 AI to analyse/edit video clipsDownload Free Python Introduction Ebook: https://clickhubspot.com/1sf7🔗 Links- Get full code breakdown &amp; J...</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: SmolLM 135M Instruct Trained on DEVINator Data for Open Hands</li><li><a href="https://youtu.be/0Ef7K18Eyxc">Pandas : Grouping and Sorting</a>: In this video, I&#39;ll be discussing how to Group or Sort in pandas with some examples and code. If you&#39;d like to see the resources or code, check the repositor...</li><li><a href="https://youtube.com/shorts/c1QI7r9AP_g?si=GWgdeHiWcPm9DfvE">TCP TIME_WAIT causing &quot;address already in use&quot; error</a>: System Design for SDE-2 and above: https://arpitbhayani.me/masterclassSystem Design for Beginners: https://arpitbhayani.me/sys-designRedis Internals: https:/...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1278900366250872914)** (4 messages): 

> - `Human Feedback in Model Training`
> - `Low Bit Quantisation`
> - `Training Requirements for AI Models` 


- **Human Feedback crucial for Model Evaluation**: A recent paper discusses how **human feedback** has become essential in evaluating and training **Large Language Models** but may be influenced by subjective biases.
   - The paper emphasizes that while preference scores cover many aspects, they **under-represent important criteria** such as factuality ([View PDF](https://arxiv.org/abs/2309.16349)).
- **Exploration of Low Bit Quantisation**: One member mentioned their focus on **low bit quantisation**, referencing a foundational paper on the topic.
   - This technique is crucial for optimizing models while maintaining efficiency ([Read Paper](https://arxiv.org/pdf/1609.07061)).
- **Training AI Models requires GPU**: A suggestion was made emphasizing that **training AI models** should not be done without a **GPU**, recommending platforms like **Colab** and **Kaggle**.
   - The insistence was clear that GPU access is **essential** for effective training.



**Link mentioned**: <a href="https://arxiv.org/abs/2309.16349">Human Feedback is not Gold Standard</a>: Human feedback has become the de facto standard for evaluating the performance of Large Language Models, and is increasingly being used as a training objective. However, it is not clear which properti...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1278820480358420603)** (4 messages): 

> - `LLM Pruning`
> - `Text-to-Speech ML`
> - `Multi-Party Chat Agents`
> - `Qwen2-VL Vision Language Models` 


- **Efficient Layer Pruning in LLMs**: A study explored a layer-pruning strategy for open-weight pretrained LLMs, finding minimal performance degradation until **up to half** of the layers are removed. The team employed methods like **parameter-efficient finetuning (PEFT)** and [quantization](https://arxiv.org/abs/2403.17887) techniques to recover model performance after pruning.
   - This suggests that pruning can help lower computational costs while enhancing memory and inference speeds.
- **GitHub Repository for Text-to-Speech ML**: A new repository titled [Text-to-Speech-ML](https://github.com/Azymack/Text-to-Speech-ML-) has been launched, aimed at contributions and development in the field of text-to-speech models. This project is a collaborative effort and invites users to engage.
   - The repository showcases the latest advancements and provides tools for further development in the text-to-speech domain.
- **Exploring Multi-Party Conversations for AI**: Research on multi-party conversations has shown that existing models trained on pairwise dialogues struggle with group dynamics, identifying critical skills lacking in these models. The study released a new dataset, **MultiLIGHT**, to improve AI's performance in multi-participant dialogues for [AI chatbots](https://arxiv.org/abs/2304.13835).
   - This work emphasizes the importance of conversational context and coherent interactions among multiple characters.
- **Qwen2-VL's State-of-the-Art Vision Language Model**: The **Qwen2-VL** series has been released, achieving state-of-the-art performance in visual understanding benchmarks such as **MathVista** and **DocVQA**. This advanced model can understand videos over **20 minutes long**, enhancing versatility in vision-language integration.
   - Qwen2-VL's release emphasizes its capability to comprehend images of varying resolutions, showcasing a significant evolution in the Qwen model family.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until af...</li><li><a href="https://arxiv.org/abs/2304.13835">Multi-Party Chat: Conversational Agents in Group Settings with Humans and Models</a>: Current dialogue research primarily studies pairwise (two-party) conversations, and does not address the everyday setting where more than two speakers converse together. In this work, we both collect ...</li><li><a href="https://qwenlm.github.io/blog/qwen2-vl/">Qwen2-VL: To See the World More Clearly</a>: DEMO GITHUB HUGGING FACE MODELSCOPE API DISCORD After a year&rsquo;s relentless efforts, today we are thrilled to release Qwen2-VL! Qwen2-VL is the latest version of the vision language models based o...</li><li><a href="https://github.com/Azymack/Text-to-Speech-ML-">GitHub - Azymack/Text-to-Speech-ML-</a>: Contribute to Azymack/Text-to-Speech-ML- development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1278811373656215704)** (12 messages🔥): 

> - `FLUX LoRA Training`
> - `ToonGPT Launch on Product Hunt`
> - `Word Game Bench for Language Models`
> - `VividNode Chatbot Release`
> - `Thoth Bot CLI Tool` 


- **FLUX LoRA Training Simplified**: A tutorial guide titled [FLUX LoRA Training Simplified](https://youtu.be/nySGu12Y05k) walks users through using Kohya SS GUI for training with an 8GB GPU on Windows.
   - This guide aims to make the training process accessible for users starting from scratch.
- **ToonGPT Now Live!**: ToonGPT has officially launched on [Product Hunt](https://www.producthunt.com/products/toontales-kiddiegpt), offering an interactive AI-powered companion for kids inspired by personal experiences.
   - The creator expresses the desire for feedback and support as they bring a unique approach to children's engagement through technology.
- **Evaluating Language Models with Word Game Bench**: The newly developed **Word Game Bench** serves as a benchmark to evaluate language models on various word puzzle games, currently with no model achieving over a 50% win rate.
   - It focuses on two tasks: **Wordle** for word guessing and **Connections** for word association, emphasizing model interaction and reasoning.
- **VividNode Chatbot Launch**: The open-source chatbot called **VividNode** has been released, featuring GPT and image generation capabilities, highlighting the creator's growth in skills.
   - A [tutorial article](https://medium.com/@yjg30737/what-is-vividnode-how-to-use-it-4d8a9269a3c0) shares details about its usage and future plans for feature additions.
- **Introducing Thoth Bot CLI Tool**: [Thoth Bot](https://github.com/U-C4N/Thoth-Bot) is an AI-powered CLI tool designed for chat, Python code generation, and improvements using multiple LLMs via Groq API, streamlining coding workflows.
   - It offers automation for code generation, execution, and error fixing, enhancing productivity for developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>: no description found</li><li><a href="https://dev.to/p3ngu1nzz/scaling-up-parallel-training-with-tau-llm-and-unity-ml-agents-53bh">no title found</a>: no description found</li><li><a href="https://airesearch.wiki/index.html">ai-research-agent</a>: no description found</li><li><a href="https://medium.com/@yjg30737/what-is-vividnode-how-to-use-it-4d8a9269a3c0">What is VividNode &amp; How to Use It</a>: VividNode is a software that allows you to directly experience GPT chatbot (ChatGPT) and image generation features on your desktop without…</li><li><a href="https://www.producthunt.com/products/toontales-kiddiegpt"> ToonTales - KiddieGPT - Product Information, Latest Updates, and Reviews 2024 | Product Hunt</a>: Introducing ToonGPT: A delightful AI-powered companion crafted for kids! Inspired by my daughter Becky, ToonGPT combines the magic of cartoons with interactive fun, sparking creativity and joy in ever...</li><li><a href="https://github.com/U-C4N/Thoth-Bot">GitHub - U-C4N/Thoth-Bot: AI-powered CLI tool for chat, Python code generation, and improvement using multiple LLMs via Groq API. Streamlines coding workflow with automated code generation, execution, and error fixing.</a>: AI-powered CLI tool for chat, Python code generation, and improvement using multiple LLMs via Groq API. Streamlines coding workflow with automated code generation, execution, and error fixing. - U-...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1278904332636389430)** (5 messages): 

> - `Meta FAIR's Transfusion`
> - `Multimodal modeling advancements`
> - `GitHub updates` 


- **Meta FAIR unveils Transfusion breakthrough**: Meta FAIR's research on **Transfusion** represents a significant leap in **multimodal modeling**, allowing concurrent prediction of tokens and image diffusion within a unified framework.
   - The model showcases *impressive scalability* and has demonstrated **superior performance** compared to traditional methods, which could **revolutionize multimodal applications**.
- **Community excitement for Transfusion**: Members expressed excitement over **Transfusion**, acknowledging its game-changing capabilities in handling vast datasets for multimodal tasks.
   - One noted the *significance of its performance* by mentioning the abundance of **gen AI keywords** present in the paper.
- **GitHub update for record keeping**: A member updated the community about the **GitHub** repository for better record keeping and requested feedback on any issues encountered.
   - Another member expressed curiosity about the **quality of Transfusion**, indicating they would check it out.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1278908354969997403)** (13 messages🔥): 

> - `Document Quality Assessment`
> - `Transfer Learning Challenges`
> - `OpenCV Techniques`
> - `GitHub Repo for Document Classifier`
> - `Networking and Friend Requests` 


- **Using Image Processing for Document Quality**: One member suggested utilizing **image processing techniques** and pre-trained models, such as **OpenCV**, to assess document quality through methods like blur detection and histogram analysis.
   - They also proposed exploring **CNNs** like **VGG** and **ResNet** to fine-tune for specific document quality requirements.
- **Transfer Learning Struggles with Document Data**: Another member tried applying **transfer learning** on datasets manipulated by adding brightness and blur but noted it didn't perform well with real-world documents, prompting a search for strategies.
   - They expressed a desire for resources on kernel applications and highlighted the significance of this problem in **organizations**.
- **Sharing GitHub Repo for Document Classifier**: A user shared their GitHub repository containing a notebook detailing their transfer learning efforts with the **FUNSD** dataset, emphasizing data augmentation techniques used.
   - The project link is [here](https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb), showcasing the various images and methods applied.
- **Late Night Discussion Plans**: Members discussed the late hour and suggested continuing their conversations the next morning, indicating a collaborative approach.
   - One member indicated that they sent a friend request to facilitate further discussions.
- **Friend Request Acceptance**: A member acknowledged the friend request sent and expressed gratitude, fostering a friendly environment for collaboration.
   - This gesture highlights the interpersonal aspect of their ongoing discussions.



**Link mentioned**: <a href="https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb">noisy_doc_clf/notebooks/train.ipynb at main · ajkdrag/noisy_doc_clf</a>: Contribute to ajkdrag/noisy_doc_clf development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1278803082410721300)** (9 messages🔥): 

> - `LLaMA 3 models`
> - `Inference GPUs`
> - `GPU RAM configurations` 


- **Guidance sought for LLaMA 3 models**: A user requested assistance with **LLaMA 3** models while planning to build **RAG applications** and needed advice on suitable on-premise GPUs.
   - They specifically asked for GPU and RAM configurations relevant to different model sizes: **8B**, **70B**, and **405B**.
- **Recommendation for GPU**: One member suggested that the **Nvidia A100** is the best option for running the models, though they did not specify RAM requirements.
   - Questions about which RAM to pair with the **A100** and which model to use were raised, indicating a need for more detailed recommendations.
- **Clarifying LLaMA 405B requirements**: Another member noted that running the **LLaMA 405B** model requires at least **300Gb of GPU RAM**, depending on precision.
   - They warned that using such large models is extremely expensive, recommending exploring cloud-based methods instead.
- **Skepticism about provided advice**: A member expressed doubt about the accuracy of the previous replies, suggesting that one response was generated by a model and was factually incorrect.
   - This led to further speculation that the answer could have originated from **LLaMA 3** itself.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278919068015005746)** (2 messages): 

> - `Animating Fireball in Photos`
> - `Using AnimateDiff with IP Adapter Plus or SVD` 


- **Ask About Animating Fireball in Photo**: A user inquired whether it's possible to animate only the **fireball** in a photo they've uploaded.
   - This highlights interest in techniques for selective animation in images.
- **Recommendation to Use AnimateDiff**: Another member suggested using **AnimateDiff** with **IP Adapter Plus** or **SVD** as a solution for animating the fireball.
   - Their recommendation indicates potential interest in AI tools for animation tasks.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

iron_bound: sounds like their LTM architecture has an RNN for attention
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1278902836868022282)** (1 messages): 

> - `Triton atomic_add functionality`
> - `Multiple GPU configurations`
> - `Scope definitions in Triton` 


- **Clarification on scope=GPU in Triton**: A member asked about the implications of using **scope=GPU** for the `atomic_add` function when working with multiple GPUs.
   - They questioned whether the default **scope=GPU** operates effectively in a multi-GPU setup.
- **Understanding scope=system in Triton**: The discussion also covered what **scope=system** means, specifically whether it refers to multiple GPUs or includes interaction with the **host**.
   - One member expressed confusion over whether **scope=system** entails GPU alongside **host** operations.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1279199308410781848)** (3 messages): 

> - `FX pass with Triton kernels`
> - `Calling Triton from PyTorch`
> - `FX pass examples`
> - `Triton code reference` 


- **Inquiring about FX pass for Triton**: A member questioned whether it's possible to implement an **FX pass** that maps **aten ops** onto a custom **Triton kernel**.
   - This inquiry suggests ongoing interest in optimizing PyTorch's performance with Triton's capabilities.
- **Calling Triton Code Natively**: It was clarified that you can directly call **Triton code** from a **PyTorch program** natively, allowing it to function with **torch.compile**.
   - This emphasizes **Triton's integration** within the PyTorch ecosystem for enhanced functionality.
- **Resource for FX pass examples**: Members mentioned that for examples of **FX passes**, reviewing the **Triton code** would be beneficial.
   - A specific link to [pre_grad.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py) was shared as a reference.



**Link mentioned**: <a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py">pytorch/torch/_inductor/fx_passes/pre_grad.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1278803556308353197)** (25 messages🔥): 

> - `Quantization Techniques`
> - `AWQ Implementation Issues`
> - `Low-Bit Optimizer Code`
> - `VLLM Integration`
> - `Layer Quantization Strategies` 


- **Quantization of Attention Layers**: It appears that quantizing the QKV projections in attention layers is common, where the default filter function handles **2D Linear layers** by checking their shape.
   - Members expressed concern over maintaining accuracy in these layers, leading to debates on whether such quantization is intentional.
- **AWQ Performance with Zero Points**: Members discussed that AWQ performance significantly deteriorates when using **floating point integers** for quantization compared to integers, leading to increased perplexity.
   - *Rounding during quantization seems to affect compatibility*, with members sharing implementation details from an old investigation.
- **Investigating Low-Bit Optimizer Code**: Concerns were raised over a questionable line in the low-bit optimizer code regarding **non-sign bits**, which is believed to be copied from another project.
   - Suggestions were made to simplify parts of the code, although there are limitations on kernel fusions for certain functions.
- **VLLM and AWQ Integration**: There was interest in exploring how the **newer VLLM version utilizes AWQ**, as past implementations prompted challenges when manipulating quant/dequant functions.
   - Members highlighted the need for accurate comparisons across quantization techniques, especially as they relate to embeddings.
- **Testing Low-Bit Quantization Strategies**: A discussion about mixed precision quantization revealed a GitHub prototype that might provide helpful insights for different model sizes.
   - Members are encouraged to check this repository as it offers a potential avenue for understanding quantization results better.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/tree/main/torchao/quantization/prototype/mixed_precision">ao/torchao/quantization/prototype/mixed_precision at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L28C5-L28C54).">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://gist.github.com/mobicham/8b3147742beb3b302064453a15ced428#file-awq_hqq_test-py-L52">awq_hqq_test.py</a>: awq_hqq_test.py. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype">ao/torchao/prototype at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorc">pytorc - Overview</a>: pytorc has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L69-L106">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://pytorch.org/docs/stable/generated/torch.searchsorted.html">torch.searchsorted &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/functional.py#L360">bitsandbytes/bitsandbytes/functional.py at e4674531dd54874c0abbc786ad5635c92c34dc3e · bitsandbytes-foundation/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - bitsandbytes-foundation/bitsandbytes</li><li><a href="https://github.com/pytorch/ao/pull/769">Fixed the llama model by yiliu30 · Pull Request #769 · pytorch/ao</a>: If we don&#39;t pass the input_pos to the model, freqs_cis = self.freqs_cis[input_pos] will select the whole freqs_cis. Test  pytest  -sv ./test/test_ao_models.py  cc @HDCharles
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1279136959381639301)** (2 messages): 

> - `Flash Attention Kernel`
> - `Shared Memory Sizes in FA`
> - `NVIDIA GeForce RTX 3090 Support`
> - `Attention Heads and Model Dimensions` 


- **Struggles with Shared Memory Sizes in Flash Attention**: A user mentioned difficulties in writing a flash attention kernel, specifically regarding shared memory sizes; they noted substantial memory demands with block sizes reaching **131,072 bytes** for Q.
   - This raised the question of how Flash Attention (FA) operates efficiently on non-Hopper GPUs with smaller SRAM capacities.
- **NVIDIA GeForce RTX 3090 Issues**: Another user reported running into issues with the flash_attn package on NVIDIA GeForce RTX 3090 GPUs, both equipped with Compute Capability **8.6**.
   - They linked a [GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/190) discussing the encountered problems while running the package specific to this hardware.
- **Question on Dimension Splitting Across Attention Heads**: There was a query regarding whether large model dimensions are divided across attention heads, suggesting that each FA head only processes smaller inner dimensions around **64 or 128**.
   - This speculation highlights the mechanics of Flash Attention and its potential adaptability to different underlying architectures.



**Link mentioned**: <a href="https://github.com/Dao-AILab/flash-attention/issues/190">Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6 · Issue #190 · Dao-AILab/flash-attention</a>: Issue description: Hello, I am using the flash_attn package on a system with two NVIDIA GeForce RTX 3090 GPUs, both of which have a Compute Capability of 8.6. When trying to run the package, I enco...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1278869677178884159)** (15 messages🔥): 

> - `Twitter Profile Recommendations`
> - `Pros and Cons of Twitter`
> - `Twitter for Research and Networking`
> - `Logistics for CUDA Mode Event` 


- **Twitter Profile Recommendations**: A user asked for recommendations on Twitter profiles to follow, and one highlighted a [specific list](https://x.com/marksaroufim/following) by marksaroufim.
   - There was some skepticism from another user who suggested that perhaps it's better not to make an account at all.
- **Debate on Twitter's Value**: A poll was initiated asking users to reflect if their time on Twitter in Summer '24 was a **net positive** or **net negative** with varying opinions shared.
   - Some users agreed that Twitter is beneficial for engaging with cutting-edge research and sharing personal work.
- **Concerns about Twitter Usage**: Participants discussed the careful curation of follows on Twitter to enhance their experience, with one stating that it's mainly for reading selected content.
   - Another user humorously noted the need to regularly mark posts as 'not interested' to clean up their feed.
- **Query about CUDA Mode Event Logistics**: A new member inquired about the logistics for the upcoming CUDA mode event, particularly regarding accommodation and meal provisions.
   - They asked if hotel booking would be necessary and for any additional details on the event structure.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1278873486789447681)** (6 messages): 

> - `L2 Side Aware optimization`
> - `FP8 switching`
> - `Loss landscape stationary points`
> - `Training sample dropping` 


- **L2 Side Aware code achieves speed boost**: The 'L2 Side Aware' code has been fixed and simplified, consistently hitting **1823GB/s** for GELU forward, outperforming **1791GB/s** from a previous kernel with x128.
   - The improvements include a **2% speed increase** and significantly lower power consumption, though further simplifications and optimizations are still needed.
- **Return to FP8 for development**: A member plans to switch back to **FP8** code development tomorrow to refresh their understanding before progressing further on the current project.
   - They expressed satisfaction with the progress made on the L2 Side Aware code but recognize additional optimization is necessary.
- **Discussion on loss landscape and training constraints**: A user discussed the implications of a stationary point in the loss landscape when optimized over the full weight space, questioning the actual constraints it imposes compared to traditional methods.
   - They emphasized the need for an implementation of vanilla fine-tuning to verify the quality of the minima achieved.


  

---


### **CUDA MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

mobicham: https://x.com/JamesLiuID/status/1829554782287413513
  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1278791518769119324)** (140 messages🔥🔥): 

> - `Release v0.2.0 Discussion`
> - `LayerNorm Kernel Updates`
> - `Memory Issues with Hugging Face example`
> - `Debugging RMS Norm Kernel`
> - `Documentation Enhancements` 


- **Release v0.2.0 Discussion**: The community discussed the release of [v0.2.0](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.2.0), highlighting improvements in the API and model support, along with the introduction of new features and bug fixes.
   - However, some users reported memory issues, as one user experienced an `OutOfMemoryError` while running the Hugging Face example with this version.
- **LayerNorm Kernel Updates**: [PR #169](https://github.com/linkedin/Liger-Kernel/pull/169) was merged, integrating LayerNorm custom kernels and LigerLayerNorm modules, with tests run for correctness on an RTX 3090.
   - Updates discussed included profiling results and a potential dynamic dispatch for atomic operations, aiming for better performance in multi-GPU scenarios.
- **Memory Issues with Hugging Face example**: After testing with v0.2.0, users noted that the example was less memory-efficient compared to v0.1.1, raising concerns over its default settings.
   - A user confirmed running the example without Liger resulted in immediate OOM errors, indicating that Liger integration was crucial for running large batch sizes.
- **Debugging RMS Norm Kernel**: A contributor reported a recurring failure in a specific test when rewriting the rms_norm kernel to use partial aggregation, with behavior becoming deterministic by manually setting the seed.
   - Further investigation revealed more mismatches and potentially a bug in the `assert_verbose_allclose` function, suggesting the condition should check for greater than 0 mismatched values.
- **Documentation Enhancements**: A new section regarding LayerNorm was added to the README, providing clarity on its functionality and implementation in the library.
   - The community expressed interest in creating a documentation website and tutorials to aid users in integrating custom operations and better utilizing the tool.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hidet.org/docs/stable/gallery/developer-guides/add-operator-resolve-rule.html">Add Operator Resolve Rule &#8212; Hidet Documentation</a>: no description found</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/174">torch.compile() throws exception when LigerKernel is used · Issue #174 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug ... File &quot;/home/tromero/workspace/seahorse/.venv/lib/python3.11/site-packages/torch/_inductor/async_compile.py&quot;, line 173, in triton kernel = TritonCodeCache.load(kernel_...</li><li><a href="https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#kernels">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/releases/tag/v0.2.0">Release v0.2.0 Release Note · linkedin/Liger-Kernel</a>: Opening Thoughts 🫶 Thank You! We&#39;d love to take this chance to express our sincere gratefulness to the community! 2500+ ⭐ , 10+ new contributors, 50+ PRs, plus integration into Hugging Face 🤗, a...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/179)">Issues · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`">CUDA semantics &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://github.com/linkedin/Liger-Kernel.git">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/169">[Operators] LayerNorm Kernels + LigerLayerNorm by AndreSlavescu · Pull Request #169 · linkedin/Liger-Kernel</a>: Summary  integrated layernorm custom kernels + LigerLayerNorm module  Testing Done  tested layernorm kernels for correctness   Hardware Type: RTX 3090  run make test to ensure correctness  run make...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/135/files#diff-0b31d056b2cdb59db1baaba4c4e7e0a79ed70b445ca67ff928ec57ffa89c6d0fR71">custom Embedding kernel by AndreSlavescu · Pull Request #135 · linkedin/Liger-Kernel</a>: Summary  Added Embedding forward/backwards kernels + LigerEmbedding class which maps to nn.Embedding nn.Embedding is useful for encoder-only models such as BERT ref: #131   Testing Done   tested ag...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/180">[Documentation] LayerNorm added to README by AndreSlavescu · Pull Request #180 · linkedin/Liger-Kernel</a>: Summary  Added LayerNorm description to README  Testing Done  N//A   Hardware Type: RTX 3090  run make test to ensure correctness  run make checkstyle to ensure code style  run make test-convergenc...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1278792548558835713)** (187 messages🔥🔥): 

> - `IP Adapter for Flux`
> - `Training Models with Limited VRAM`
> - `Segmentation in Image Processing`
> - `RunwayML SD 1.5 Repo Deletion`
> - `SDXL vs SD 1.5` 


- **IP Adapter for Flux gains attention**: Members discussed the recent introduction of an IP adapter for Flux, which has shown mixed results in performance, with some users finding it less effective.
   - One member noted that despite varying opinions, it is still an exciting development in the community.
- **Training Models on Limited VRAM**: Users shared experiences about training with limited VRAM, particularly with an RTX 3060, indicating that higher resolutions (like 1024) consume significant memory.
   - It was suggested to work with lower resolutions to reduce memory footprint, with confirmation that 12GB RAM may not suffice for complex tasks.
- **Segmentation in Image Processing**: Discussion highlighted the concept of SEG (Segmentation) in image processing workflows, particularly how it is connected to existing nodes in systems like ComfyUI.
   - Participants expressed confusion over its implementation and its necessity compared to simpler alternatives.
- **RunwayML removes SD 1.5 repositories**: The community noted that RunwayML has deleted all their Stable Diffusion 1.5 repos on HuggingFace and GitHub, prompting varied reactions about the implications of this move.
   - Users speculated whether this deletion signifies a shift away from 1.5 models, which are reportedly less utilized.
- **Comparing SDXL with SD 1.5**: A user contemplated switching from SD 1.5 to SDXL, weighing the concerns of generation times and model storage requirements with their existing GPU.
   - Advice was given to optimize performance with command line arguments to accommodate weaker GPU capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/ygD5YMm">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://imgur.com/Xr44AHl">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://www.amazon.de/Fantastische-Fabelwesen-Stressabbau-Entspannung-Fantasie-Kreaturen/dp/B0CN5B8WTG/ref=sr_1_1?crid=3IBODT2J8X6H6&dib=eyJ2IjoiMSJ9.-3XggVW3uObjvvXQqObf-g-EWf_V6QDcBkrHerEySuY2P3W0J8JG92mAOXoFt2DWOwZHT1w0m6M4IrDxhUwXVi523Affpx6n5y5TI3Pal5iMGXUuSJEje7x1BSRxDuAhRJqcESyU0awWBpc07xA90cucn7Z_uETG34wev0if1-ON4ICntYnPnlLPGVH6WUk532dqEr89fXftuzS4TrhIrYMCKNik-WVzuMj3aU2Vvr8.d_Vd1P3m4memC-Dd8Agtfsyxu8CgD6J3vjQdJ--SaDo&dib_tag=se&keywords=fabelwesen+malbuch&qid=1724956770&sprefix=Fabelwesen+%2Caps%2C126&sr=8-1">no title found</a>: no description found</li><li><a href="https://youtu.be/3LnbI5pcQko">The weirdest AI app I&#39;ve used</a>: This AI breaks social media. It&#39;s like Instagram, but everyone is AI. #ainews #ai #agi #socialmedia #npcTurboType helps you type faster with keyboard shortcu...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings">Command Line Arguments and Settings</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://youtu.be/v9KGQoaqhkw">CogVideoX 5B In ComfyUI Better Quality - Local AI Video Model That Truly Works!</a>: CogVideoX 5B Better Quality A Local AI Video Model Truly Works!In this engaging video, we delve into the latest advancements in AI technology with the CogVid...</li><li><a href="https://arxiv.org/abs/2408.16232">Enhancing Conditional Image Generation with Explainable Latent Space Manipulation</a>: In the realm of image synthesis, achieving fidelity to a reference image while adhering to conditional prompts remains a significant challenge. This paper proposes a novel approach that integrates a d...</li><li><a href="https://github.com/kshitij79/CS-7476-Improvements-in-Diffusion-Model">GitHub - kshitij79/CS-7476-Improvements-in-Diffusion-Model</a>: Contribute to kshitij79/CS-7476-Improvements-in-Diffusion-Model development by creating an account on GitHub.</li><li><a href="https://mp.weixin.qq.com/s/ZKJieSzqISyzCB8Iz9tY8A">【AI行业报告】Top 100 AI 产品 (第3期)</a>: AI行业报告第三期，来看看哪些AI产品上榜了？
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1278805467065352263)** (118 messages🔥🔥): 

> - `Amnesia Mode in AI Models`
> - `Training Techniques for LLMs`
> - `Gradient Communication Strategies`
> - `Hermes 3 Model Behavior`
> - `New AI Evaluation Framework` 


- **Amnesia Mode Experiences**: Users discussed the 'amnesia mode' of Hermes 3, highlighting its preference for professionalism over casual slang. One user expressed frustration at the model's insistence on being 'family-friendly' despite casual greetings.
   - The model displayed peculiar responses even when users attempted casual interactions, prompting discussions about whether it was a predefined behavior.
- **Training Techniques for LLMs**: A member shared they're training a Llama 3 on synthetic and real instruction data from platforms like Reddit. They aim to investigate if this process reduces 'AI-y' responses by making data more instruct-oriented.
   - The community engaged in discussions about handling training losses, experiences with odd training behaviors, and the importance of managing gradient issues.
- **Exploring Gradient Communication Strategies**: A user proposed low-rank approximations for gradients during model synchronization, aiming to reduce communication overhead. They highlighted possible enhancements by analyzing the gradient impact from data-parallel nodes.
   - Discussion revolved around combining various optimization techniques to facilitate more effective distributed training strategies.
- **Hermes 3 Model Behavior Insights**: Users noted that Hermes 3 displays certain behavior patterns, including potential preferences for communication styles. There were questions about the underlying reasons for these behaviors and how they might be influenced by system prompts.
   - Interactions revealed that certain phrases triggered unexpected responses, suggesting a blend of amnesia modes, prompting members to share experiences.
- **New AI Evaluation Framework: Word Game Bench**: A new benchmark called 'Word Game Bench' was introduced, aimed at evaluating language models through word puzzle games like Wordle and Connections. The creator allowed for unique interaction where models create outputs based on previous game actions.
   - Members expressed interest in the benchmark's approach and its implications for assessing model performance in an engaging and interactive manner.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1829205847731515676?s=19">Tweet from Sam Altman (@sama)</a>: we are happy to have reached an agreement with the US AI Safety Institute for pre-release testing of our future models.  for many reasons, we think it&#39;s important that this happens at the national...</li><li><a href="https://x.com/wingsoverheaven/status/1829024789693968628">Tweet from wings (@wingsoverheaven)</a>: no description found</li><li><a href="https://wordgamebench.github.io">Word Game Bench</a>: no description found</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">Tweet from zafir (@zafstojano)</a>: Excited to share &#34;Word Game Bench&#34; - a fun benchmark for evaluating language models on word puzzle games!   It is a relatively hard benchmark, where no model currently scores above 50% average...</li><li><a href="https://arxiv.org/abs/2311.08105">DiLoCo: Distributed Low-Communication Training of Language Models</a>: Large language models (LLM) have become a critical component in many applications of machine learning. However, standard approaches to training LLM require a large number of tightly interconnected acc...</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>: A Gradio web UI for Large Language Models. Contribute to oobabooga/text-generation-webui development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, rea...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1278815792594686032)** (43 messages🔥): 

> - `Instruction Tuning`
> - `Hermes 3 Performance`
> - `Full Precision vs 8 Bit Models`
> - `Hardware Requirements for Large Models`
> - `100 Million Token Context Window` 


- **Instruction Tuning Insights**: A member questioned whether instruction tuning typically involves training on the user end of conversations, with another confirming it's better to train only on outputs.
   - *Training solely on outputs resulted in significantly better benchmarks* compared to including user inputs.
- **Seeking Full Precision Hermes 3 Model**: A user expressed frustration trying to find a host for the full precision **Hermes 3 model** (bf16), which reportedly has no current providers.
   - Discussion revealed that no provider has yet offered this model, with concerns over *efficiency and hardware requirements* being primary obstacles.
- **Quantization Impact on Model Performance**: It was noted that larger models tend to be *more quantization resistant*, affecting performance at lower bit quantization levels.
   - For instance, a **70B model** at 2-bit can still produce coherent text, unlike smaller models which see degradation.
- **Concerns About Hosting Large Language Models**: Discussion highlighted that serving models like **Hermes 3** (405B) requires extensive hardware setups, often needing multinode configurations.
   - Members noted the challenge of balancing demand and hardware capabilities, leading many providers to stick with lower bit quantization models.
- **Magic of 100 Million Context Windows**: A user highlighted the intriguing news about a **100 million token context window**, possibly representing a breakthrough comparable to Q*.
   - Others humorously remarked on the perceived *magical* aspects of such advancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lam">Lambda Docs</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B">NousResearch/Hermes-3-Llama-3.1-405B · Hugging Face</a>: no description found</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-chat-completions-api">Using the Lambda Chat Completions API | Lambda Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 messages🔥): 

> - `GameNGen`
> - `Real-time Game Simulation`
> - `Neural Network Integration in Gaming`
> - `Unique Hallucinations in Gaming`
> - `Potential for Horror Games` 


- **GameNGen: Neural Model Takes the Spotlight**: A discussion emerged around _GameNGen_, the first game engine entirely powered by a neural model that can simulate the classic game **DOOM** at over **20 frames per second** without traditional game engine tools.
   - Participants expressed excitement over this **proof of concept**, with interests in how mainstream engines like **Unreal Engine** might integrate similar technology.
- **Trippy Gameplay Experience Enthralls Players**: Footage of gameplay reveals that _GameNGen's_ simulations appear **trippy** and even **dreamlike**, sparking interest in future applications beyond just replicating existing games.
   - One member noted the potential for these unique hallucinations to inspire a **fully original horror IP**, adding a refreshing twist to the genre.
- **Challenges in AI-Driven Game Creation**: Discussions highlighted the need for **guidance** while working with the neural model, hinting at the complexities involved in crafting a coherent gameplay experience.
   - As the tech evolves, questions arose concerning the balance between AI creativity and player interaction to achieve engaging and coherent gameplay.



**Link mentioned**: <a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 messages🔥): 

> - `GameNGen Neural Model`
> - `DOOM Simulation`
> - `Integration with Game Engines`
> - `Unique Hallucinations in Gaming`
> - `Original Horror IP Potential` 


- **GameNGen simulates DOOM without a game engine**: The _GameNGen_ neural model enables real-time simulation of the classic game [DOOM](https://en.wikipedia.org/wiki/Doom_(1993_video_game)) with no traditional game engine involved.
   - It achieves over **20 frames per second** using a single TPU and shows promising results in realism, as human raters struggle to distinguish between simulated and real clips.
- **Excitement about integrating technology with Unreal Engine**: A member expressed enthusiasm about seeing how major game engines like **Unreal Engine** could integrate this neural simulation technology in the future.
   - They also showed interest in replicating the technology themselves, highlighting its potential for innovative game development.
- **Trippy gameplay footage sparks discussion**: The gameplay footage from _GameNGen_ was described as **trippy** and **dreamlike**, opening up discussions about its unique visual experience.
   - Members shared thoughts on using these qualities for original game designs, particularly in the horror genre, which could offer a refreshing perspective.
- **Interest in unique hallucinations for games**: There was a shared interest in how the unique hallucinations produced by the neural model could be harnessed for creating an original horror IP.
   - This approach could offer players a distinct gameplay experience that diverges from traditional gaming mechanics.
- **Model requires hand-holding for effective use**: Concerns were raised regarding the need for substantial oversight and guidance when leveraging the neural model for complex game creation.
   - The need for ‘hand-holding’ suggests challenges in making the technology user-friendly and effectively implemented in original content.



**Link mentioned**: <a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1278791887024689195)** (93 messages🔥🔥): 

> - `API Inference Speed`
> - `LM Studio Update Stability`
> - `Model Performance and Compatibility`
> - `Text to Image/Voice Integration` 


- **API Inference Speed Cap Discussion**: A user inquired about capping inference speed on the API, and another member clarified that multiple requests with multiple models loaded are feasible.
   - The user indicated a preference for using the same model to conserve VRAM but acknowledged this may not be possible.
- **User Feedback on LM Studio Version 0.3**: A member expressed concerns about the latest LM Studio update reducing their AI's responsiveness, with unusual repeated output being mentioned.
   - Other users suggested that this issue may relate to prompt settings or template parsing, recommending adjustments to resolve it.
- **Evaluating Model Performances**: Discussions emerged around the performance comparison between models Gemma 2 and Yi 1.5, with some regarding Gemma 2 as overly censored.
   - Additionally, users evaluated potential alternatives, emphasizing the need for a general-purpose, uncensored model.
- **Query on Text to Image/Voice Capabilities**: A user inquired about the possibility of integrating text-to-image or text-to-voice functionalities within LM Studio.
   - Current discussions indicated a lack of such features or support for those functionalities in the existing LM Studio setup.
- **Setting Up on CPU**: One participant queried about the slower initial prompt processing when using CPU, leading to a discussion on expected performance outcomes.
   - It was suggested that the limitations of using CPU for processing are likely unavoidable given the architecture of the models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5/blob/main/tokenizer_config.json#L31">tokenizer_config.json · sophosympatheia/Midnight-Miqu-70B-v1.5 at main</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/xLAM-7b-r-GGUF/tree/main">lmstudio-community/xLAM-7b-r-GGUF at main</a>: no description found</li><li><a href="https://github.com/YorkieDev/LMStudioWebUI">GitHub - YorkieDev/LMStudioWebUI: A wip version of a simple Web UI to use with LM Studio</a>: A wip version of a simple Web UI to use with LM Studio - YorkieDev/LMStudioWebUI</li><li><a href="https://github.com/THUDM/CogVideo">GitHub - THUDM/CogVideo: Text-to-video generation: CogVideoX (2024) and CogVideo (ICLR 2023)</a>: Text-to-video generation: CogVideoX (2024) and CogVideo (ICLR 2023) - THUDM/CogVideo
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1278858932751695872)** (82 messages🔥🔥): 

> - `M2 Ultra Mac setup`
> - `LLM performance on GPUs`
> - `Parallel processing with multiple GPUs`
> - `Power consumption management`
> - `Model loading and inference speeds` 


- **M2 Ultra Mac ready for development**: A member mentioned setting up their new **M2 Ultra Mac** with **192 GB** Unified Memory to establish a developer environment before experimenting with LLMs.
   - They noted a **2 TB** drive is designated for this, utilizing a separate PC as a server.
- **Exploring LLM performance on RTX 4090s**: Discussions highlighted that running the **405b model** with **6 RTX 4090s** produced speeds around **1 token per second**, with offload settings affecting performance.
   - A member tested multiple GPU configurations, observing how memory linking across GPUs could potentially increase speeds when models were well-distributed.
- **Testing parallel processing capabilities**: Multiple users debated whether **LM Studio** supports true parallel processing across multiple GPUs, discussing its implications on inference speeds.
   - One member noted that splitting model layers and utilizing memory offload in Python might be effective for achieving better performance at higher token speeds.
- **Managing power consumption in GPU setups**: Concerns were raised about power consumption, particularly when running multiple **RTX 4090s**, with setups often needing shared phases to avoid tripping breakers.
   - A member explained how they configured their power supply units (PSUs) to accommodate the high demand while splitting loads across different circuits.
- **Impact of PCIe lane settings on performance**: Discussion ensued regarding the effect of running **RTX 4090s** on gen4 x8 settings instead of x16, particularly when using multiple GPUs with dense models.
   - Members concurred that while gen4 x8 configuration might not significantly affect performance for single GPU setups, it could hinder speed in multi-GPU environments.



**Link mentioned**: <a href="https://tenor.com/view/power-usage-auxiliary-nuclear-gif-22138997">Power Usage Auxiliary Nuclear GIF - Power Usage Auxiliary Nuclear - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278938553975308310)** (2 messages): 

> - `Gemini Flash models`
> - `Database downtime` 


- **Gemini Flash models are now available and free**: The **Gemini Flash 8B (EXP)** model is now available at [this link](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp) and the **Gemini Flash Experiment** can be found [here](https://openrouter.ai/models/google/gemini-flash-1.5-exp).
   - All **Gemini Experimental models** are now confirmed to be free until further pricing is determined for **AI Studio**.
- **Downtime caused by database error**: A **15-minute downtime** was recorded due to a database mistake, but the issue has since been reverted.
   - No additional details on the impact of this downtime were provided.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp">Gemini Flash 8B 1.5 Experimental - API, Providers, Stats</a>: Gemini 1.5 Flash 8B Experimental is an experimental, 8B parameter version of the [Gemini 1. Run Gemini Flash 8B 1.5 Experimental with API</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5-exp>">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1278792284514549924)** (2 messages): 

> - `Daun.ai Launch`
> - `AI Chat CLI Tool` 


- **Congrats on Daun.ai Launch!**: Excitement was expressed in the community as members congratulated the team behind **Daun.ai** for their recent launch.
   - The sentiment reflects a growing interest and positive reception towards new AI tools.
- **All-in-One AI CLI Tool on GitHub**: A member shared a link to the [AI Chat CLI Tool](https://github.com/sigoden/aichat), which features Chat-REPL, Shell Assistant, RAG, AI tools & agents with access to various platforms including OpenAI and Claude.
   - The project is touted as a comprehensive solution for AI interactions, integrating multiple functionalities for enhanced user experience.



**Link mentioned**: <a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. - sigoden/aichat

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1278791605289095269)** (146 messages🔥🔥): 

> - `OpenRouter Feedback`
> - `Cohere Model Updates`
> - `Rate Limiting on Experimental Models`
> - `Perplexity Model Issues`
> - `Infrastructure Downtime` 


- **OpenRouter users report issues and suggestions**: Users expressed concerns about default models in chat and issues with the frontend, prompting requests for improvements and direct communications with developers.
   - One user noted the possibility of providing screen recordings to facilitate troubleshooting of these frontend issues.
- **Cohere updates bring excitement**: Discussion centered around recent updates to Cohere's Command R models, highlighting new features and pricing structures for API access.
   - Users were eager to try out the new capabilities but questioned how safety modes would be handled by OpenRouter.
- **Experimental models experiencing rate limits**: Users reported encountering rate limit errors while using experimental models, highlighting the challenges and limitations in testing these new features.
   - There was discussion about the implications of needing to handle safety settings through the API and confusion regarding defaults set at the endpoint.
- **Perplexity model errors reported**: A user reported receiving an error regarding a model that was no longer valid, suggesting issues with model IDs and availability.
   - Another user confirmed that this issue was being actively addressed and to use a specific channel for further discussions.
- **Infrastructure upgrades amidst downtime concerns**: Concerns about increasing downtime were raised, prompting responses about ongoing infrastructure upgrades intended to alleviate pressure on systems.
   - Developers acknowledged recent outages, attributing them to database capacity issues, and outlined plans to improve overall system stability in the near future.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alibaba_qwen/status/1829187292038115413?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Qwen (@Alibaba_Qwen)</a>: To access Qwen2-VL-72B, temporarily you should use our official API in the following way:</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://api.together.ai/models/Qwen/Qwen1.5-4B-Chat">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards.","type":"invalid_model","code":400}}">Getting Started with Perplexity API - Perplexity</a>: no description found</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct)">Qwen 2 7B Instruct - API, Providers, Stats</a>: Qwen2 7B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.  It features SwiGLU activation, attention QKV bias, and grou...</li><li><a href="https://cohereforai-c4ai-command.hf.space/">Cohere Command Models</a>: Command R models are optimized for a variety of use cases including reasoning, summarization, and question answering. Developed by Cohere and Cohere For AI.</li><li><a href="https://x.com/OfficialLoganK/status/1828922199425548486)">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @DaveManouchehri Free in AI Studio. I don’t know off the top of my head if Vertex’s experimental endpoint is free or not</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental#pricing)">no title found</a>: no description found</li><li><a href="https://x.com/OfficialLoganK/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/issues">Issues · Pythagora-io/gpt-pilot</a>: The first real AI developer. Contribute to Pythagora-io/gpt-pilot development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: Language models ranked and analyzed by usage across apps</li><li><a href="https://marketplace.visualstudio.com/items?itemName=PythagoraTechnologies.gpt-pilot-vs-code&ssr=false#review-details">Pythagora&#32;(GPT&#32;Pilot)&#32;Beta&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;The&#32;first&#32;real&#32;AI&#32;developer.</li><li><a href="https://huggingface.co/CohereForAI">CohereForAI (Cohere For AI)</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1278825342605201611)** (56 messages🔥🔥): 

> - `Embedding Weights NaN Issue`
> - `Research Feedback on Compression Project`
> - `SAE Discussion`
> - `Regularization Techniques`
> - `Vision Embedding vs. Vision Token` 


- **Embedding Weights go NaN during Training**: A user reported that embedding weights became **NaN** just a few steps into training, possibly due to a denominator in the loss function rounding to zero.
   - Further investigation indicated that their data-dependent decay term was the source of the issue, as tracking gradients helped pinpoint the problem.
- **Seeking Feedback on Compression Research**: Jeremy Vonderfecht, a PhD student, is seeking feedback on research ideas related to compressing images using flagship diffusion models, like **Stable Diffusion**.
   - Members suggested using the current channel and another designated one for sharing ideas, indicating a welcoming environment for discussion.
- **Clarifications on SAE and Inputs**: There was a discussion clarifying the term **x** in the context of an SAE, with misunderstandings about its role in the network.
   - Members emphasized the importance of specifying premises in discussions, particularly when addressing the function of inputs to the vector of activations.
- **Research on Regularization Techniques**: A user discussed potential regularization strategies, like enforcing a mean of zero on inputs or using batch normalization to stabilize training.
   - It was clarified that anything potentially slowing down the optimization process could be detrimental, emphasizing careful design of loss functions.
- **Vision Embedding vs. Vision Token Advantages**: A question was raised regarding the advantages of **vision token** vision embedding over traditional approaches, highlighting a lack of clarity on their pros and cons.
   - The discussion acknowledged that vision tokens may have more native application, prompting further exploration of their benefits in the context of vision tasks.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1278803733127495800)** (88 messages🔥🔥): 

> - `Dynamic Expert Routing in Models`
> - `Adversarial Approaches in AI Safety`
> - `Tokenization Challenges in Language Models`
> - `Multi-Token Prediction Efficiency`
> - `Model Quantization Techniques` 


- **Dynamic expert routing enhances model training**: The concept of allowing models to define their own experts during training, instead of using a fixed configuration, has been discussed as a way to improve adaptability.
   - Members noted that this idea is linked to ongoing research like the methods proposed in the [LayerSkip paper](https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e).
- **Exploring adversarial methods in AI safety**: A suggestion was made to focus on adversarial strategies as a key area of interest in AI safety discussions.
   - This sentiment emphasizes the importance of exploring underlying vulnerabilities in AI systems.
- **Tokenization poses challenges for language models**: Participants discussed the limitations of tokenization, especially regarding non-Latin languages and the complexity it adds to model training.
   - Concerns were raised about tokenization obfuscating important data features and slowing down training efficiency.
- **Multi-token prediction's effectiveness debated**: Discussions highlighted that the efficiency of multi-token prediction (MTP) might not significantly benefit smaller language models, nor improve training speed even in larger models.
   - There is ongoing debate about whether the computational costs of MTP justify the potential gains in model performance.
- **Exploring model quantization methods**: The introduction of finite scalar quantization (FSQ) was discussed as a potentially effective and simpler alternative to traditional vector quantization techniques.
   - The FSQ method promises improved performance across various tasks, as noted in a linked paper, and its implications for token utilization were considered important.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.15495">Remove Symmetries to Control Model Expressivity</a>: When symmetry is present in the loss function, the model is likely to be trapped in a low-capacity state that is sometimes known as a &#34;collapse.&#34; Being trapped in these low-capacity states can...</li><li><a href="https://arxiv.org/abs/2403.00417">Rethinking Tokenization: Crafting Better Tokenizers for Large Language Models</a>: Tokenization significantly influences language models(LMs)&#39; performance. This paper traces the evolution of tokenizers from word-level to subword-level, analyzing how they balance tokens and types...</li><li><a href="https://arxiv.org/abs/2408.16532">WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling</a>: Language models have been effectively applied to modeling natural signals, such as images, video, speech, and audio. A crucial component of these models is the codec tokenizer, which compresses high-d...</li><li><a href="https://arxiv.org/abs/2406.07548">Image and Video Tokenization with Binary Spherical Quantization</a>: We propose a new transformer-based image and video tokenizer with Binary Spherical Quantization (BSQ). BSQ projects the high-dimensional visual embedding to a lower-dimensional hypersphere and then ap...</li><li><a href="https://arxiv.org/abs/2309.15505">Finite Scalar Quantization: VQ-VAE Made Simple</a>: We propose to replace vector quantization (VQ) in the latent representation of VQ-VAEs with a simple scheme termed finite scalar quantization (FSQ), where we project the VAE representation down to a f...</li><li><a href="https://arxiv.org/abs/2310.05737">Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation</a>: While Large Language Models (LLMs) are the dominant models for generative tasks in language, they do not perform as well as diffusion models on image and video generation. To effectively use LLMs for ...</li><li><a href="https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e">LayerSkip: faster LLM Inference with Early Exit and Self-speculative decoding</a>: Introduction
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1279061226571169802)** (5 messages): 

> - `Word Game Bench`
> - `Consistency Measurement`
> - `Dataset Construction` 


- **Introducing Word Game Bench for Language Models**: **Word Game Bench** is a benchmark designed to evaluate language models on word puzzle games like **Wordle** and **Connections**, with no model currently scoring above **50% average win rate**. It emphasizes interaction and feedback incorporation, with a unique approach to test set management that avoids fixed evaluations to prevent leakage.
   - For more details, visit [Word Game Bench](https://wordgamebench.github.io) and check out the announcement on Twitter by [@zafstojano](https://x.com/zafstojano/status/1829398835585520076).
- **Measuring Consistency in Responses**: A member is exploring ways to compare responses from multiple choice questions to assess consistency when prompts vary slightly, suggesting the use of `process_results` and aggregate functions. They have transformed their dataset to include repeated entries for the same questions along with different prompts for comparison.
   - Another member advised that using the library might not be straightforward, and recommended constructing specific datasets that represent what is needed, though this would require a separate setup for each model.
- **Adjusting Prompts for Consistency Analysis**: A suggestion was made to run the model multiple times on the same dataset, changing the prompts with each run to facilitate comparison among responses. The strategy involves using `doc_to_text` to integrate other prompts for measuring deviations in responses.
   - This approach emphasizes a need for careful handling of the datasets to ensure accurate comparisons and avoid errors during data processing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>: no description found</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">Tweet from zafir (@zafstojano)</a>: Excited to share &#34;Word Game Bench&#34; - a fun benchmark for evaluating language models on word puzzle games!   It is a relatively hard benchmark, where no model currently scores above 50% average...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1278814192404664381)** (1 messages): 

> - `Discord server growth`
> - `Community appreciation` 


- **Discord server hits 100K members!**: The Discord server has officially reached **100K members**, marking a significant milestone for the community.
   - *A huge thank you* was extended to all members for their support and feedback, highlighting excitement for continued growth.
- **Community's incredible support recognized**: The team expressed gratitude for all the *support and feedback* received from the community during its growth phase.
   - They are excited to evolve and continue this *journey* with every member contributing to its vibrant atmosphere.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1278814360260710411)** (120 messages🔥🔥): 

> - `Subscription Issues`
> - `AI Model Performance`
> - `Event Announcements`
> - `AI Exhibition in France`
> - `User Experience Issues` 


- **Recurring Subscription Problems**: Multiple users reported issues with their Pro subscriptions disappearing or not working, with suggestions to contact support for clarification on voucher problems.
   - One user expressed concern over not receiving confirmation for their application, highlighting potential issues with user support.
- **Queries on AI Model Behavior**: A user questioned if their selected AI model was working correctly, noting similar answers despite switching models, leading to speculation about bugs.
   - There was a discussion on the perceived inconsistency in responses regarding model identification, indicating possible updates affecting user experience.
- **Event Updates and Conferences**: An organizer announced an AI exhibition in France, requesting promotional materials and resources for showcasing Perplexity AI effectively.
   - There was interest in promotional content that extends beyond standard YouTube resources.
- **User Interface Concerns**: Several users reported experiencing deleted threads or issues with query submissions not going through, expressing frustration over lost content.
   - Some users shared strategies for troubleshooting these problems, indicating a need for improved reliability.
- **Discussion on Model Usage Limits**: Users discussed varying limits on model usage over time, noting the historical capacity changes from 600 to current limits, reflecting pricing strategies.
   - The conversation highlighted the importance of understanding how model limits impact user experiences and expectations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rohansai22.github.io/resume/">Maragoni Rohan Sai - Portfolio</a>: no description found</li><li><a href="https://tenor.com/view/griffith-berserk-eclipse-guts-berserk-anime-meme-gif-10622855093064880455">Griffith Berserk GIF - Griffith Berserk Eclipse - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1278946013813542966)** (10 messages🔥): 

> - `MrBeast News`
> - `C++ Programming`
> - `Vikings Influence`
> - `OpenAI's DALL-E`
> - `Muscle Knots` 


- **What happened to MrBeast?**: A member shared a link to an article discussing the latest updates about **MrBeast**'s activities and endeavors which can be found [here](https://www.perplexity.ai/search/what-happened-to-mrbeast-S0hJBJ01TSKV6CqiLDXnvw).
   - This could provide insights into changes in his content direction or business ventures.
- **C++ Programming Essentials**: A link was shared that outlines how to write a C++ program with help from the community, which can be accessed [here](https://www.perplexity.ai/search/write-a-c-plus-plus-program-fo-aJscZujqQZGLq2_8THGP5A).
   - The article likely covers essential concepts and examples for beginners.
- **Diving into Vikings' Contributions**: A user mentioned exploring the impact of **Vikings** on modern culture, sharing a link to resources [here](https://www.perplexity.ai/search/what-have-vikings-done-for-mod-Cb_PHCx7Ty2cDQZVa14iJA).
   - This could provide a comprehensive view of their legacy and influence.
- **Understanding DALL-E**: A link discussing **OpenAI's DALL-E** has been shared, which can be found [here](https://www.perplexity.ai/search/openai-s-dall-e-0eZkD0GfRliPUTnsBpKBIQ).
   - It likely covers its features, capabilities, and applications.
- **What are muscle knots?**: A member asked about **muscle knots**, linking to an informative piece *[here](https://www.perplexity.ai/search/what-are-muscle-knots-also-kno-.GsfiArjRTW.5wmBcUIYtA)* discussing their causes and treatments.
   - This could help many understand and find relief from this common issue.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278811459739979888)** (9 messages🔥): 

> - `Pro API Credits Issues`
> - `Pro Searches Availability`
> - `Rate Limiting on API`
> - `API Account Support` 


- **Users report missing Pro API credits**: Several users, including @mihir2033, reported not receiving the **$5 PPLX API credits** after purchasing Pro.
   - They are actively asking for support and sharing their account details for resolution.
- **Pro Searches not functional on API**: @balapete expressed uncertainty over **Pro Searches** working within the API, mentioning using **llama-3.1-sonar-huge-128k-online**.
   - User @ok.alex confirmed that **Pro** is currently not available via the API, leaving users to wonder when it might be.
- **Rate Limit Error Encountered**: @nicconike shared experiencing a **429 Client Error: Too Many Requests** when invoking the API, questioning the cause.
   - This concern highlights potential limitations or usage caps on the API affecting functionality.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1278792607710969916)** (70 messages🔥🔥): 

> - `MMLU and Model Performance`
> - `Command R+ Updates`
> - `Cohere Chat Interface`
> - `GQA and Throughput Increases`
> - `Cohere Scholars Discord` 


- **MMLU Not Correlating with Practical Use**: A member mentioned that **MMLU** isn't strongly correlated with building useful LLMs, citing examples of outdated questions on topics like Freud's theories.
   - They noted that the model refreshes are improving performance due to better internet presence of MMLU data.
- **Command R+ Shows Impressive Updates**: [Command R+ 08-2024](https://cohere.com/blog/command-series-0824) has improved multilingual retrieval-augmented generation and performance metrics over its predecessor, including 50% higher throughput.
   - Members discussed how Command R is now on par with the larger Command R+ model, demonstrating solid performance gains.
- **Concerns with Cohere Chat Interface**: Users raised questions about whether the **Cohere chat interface** was updated for the new model, with some mentioning it remains the same.
   - There were discussions about the lack of a night/dark mode interface option in the chat.
- **GQA's Role in Throughput Improvements**: The introduction of **GQA** is seen as a key factor for the improved throughput in the Command R model updates.
   - Opinions varied on whether the throughput increase could also be attributed to new quantization methods.
- **Joining Cohere Scholars Discord**: A question arose regarding how to join the **Cohere Scholars Discord**, with guidance to find the 'Join Us' button on the Cohere website.
   - Several members engaged positively about their appreciation for the community and the work being done.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/safety-modes">Safety Modes — Cohere</a>: The safety modes documentation describes how to use default and strict modes in order to exercise additional control over model output.</li><li><a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: no description found</li><li><a href="https://cohere.com/blog/">The Cohere Blog</a>: Explore our collection of insightful blog posts covering a diverse range of generative AI topics. Our articles offer in-depth analyses, expert opinions, and practical advice to inform and inspire. </li><li><a href="https://docs.cohere.com/">Cohere Documentation — Cohere</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024">CohereForAI/c4ai-command-r-plus-08-2024 · Hugging Face</a>: no description found</li><li><a href="https://cohere.com/blog/command-series-0824">Updates to the Command R Series</a>: The latest versions of the Command R model series offer improvements across coding, math, reasoning, and latency. </li><li><a href="https://huggingface.co/datasets/joey234/mmlu-human_sexuality-original-neg">joey234/mmlu-human_sexuality-original-neg · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1279078308243439656)** (6 messages): 

> - `Command R and R+ models update`
> - `Model availability on different platforms`
> - `Fine-tuning defaults`
> - `Benchmarks for new models` 


- **Command R and R+ models receive a major update**: Cohere announced refreshed **Command R** and **R+** models with boosts in **performance** for reasoning, coding, and multilingual RAG, now available under the aliases `command-r-08-2024` and `command-r-plus-08-2024`.
   - The updated models also feature **lower pricing** per token, with R being significantly cheaper at **$0.15** for input tokens.
- **Availability of new models on various platforms**: Community members confirmed that the updated models are available on **Hugging Face** and will eventually make their way to **Ollama** after conversion.
   - They emphasized the need for time to have the models properly **quantized** and uploaded to other platforms.
- **Inquiry on fine-tuning defaults with new models**: A user inquired whether the new **Command models** would serve as the defaults for fine-tuning purposes.
   - There was no direct response, but the question indicates interest in applying the updated models in a fine-tuning context.
- **Call for benchmarks on new models**: A user requested if **benchmarks** for the new models could be released to assess their performance.
   - This shows the community's eagerness to evaluate the updated models quantitatively.



**Link mentioned**: <a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: no description found

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1278833174037925909)** (10 messages🔥): 

> - `C4AI Scholars Program`
> - `Command R+ Release`
> - `GDPR Compliance` 


- **Inquiry on C4AI Scholars Program Eligibility**: A member asked if the **C4AI Scholars Program** accepts current graduate students, potentially in a setup similar to summer internships but starting in January.
   - Another member advised reaching out to **C4AI** directly for clarification.
- **Discussion on Command R+ Release**: A member inquired about the potential release of the latest version of **Command R+**.
   - There wasn't a clear response to this question, leaving the release status uncertain.
- **GDPR Compliance Questions Raised**: A member asked about **Cohere's** compliance with **GDPR** regulations concerning the use of APIs, especially regarding data usage for training related to **Command R+**.
   - Another member shared a link to the **Cohere Trust Center**, indicating it should provide comprehensive answers to compliance queries.



**Link mentioned**: <a href="https://cohere-inc.secureframetrust.com/">  Cohere Inc | Trust Center
</a>: no description found

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1279021162814505012)** (46 messages🔥): 

> - `API Rate Limiting`
> - `Citations Management`
> - `Safety Mode Interaction`
> - `Trial Key Limitations`
> - `Financial Data Analysis App` 


- **API Trial Key Limitations Causing Errors**: A user encountered a **rate limit error (429)** while using a trial API key, indicating they exceeded the **1,000 API calls/month limit**.
   - Several members confirmed the need for a **production key** to avoid these restrictions, suggesting adding a credit card for enhanced access.
- **Handling Citation Overload in Outputs**: A member reported excessive citations for a **180-word text**, wanting to limit them and asking for strategies to prioritize the most important citations.
   - The suggestion to **rerank citations** and share only the references was well-received as a viable solution.
- **Interaction Between Safety Mode and Preamble**: It was clarified that the new `safety_mode` does not override the custom `preamble`, and they operate independently in generating responses.
   - Testing revealed that when **safety modes** are active, they modify the prompts accordingly by combining safety instructions with user preambles.
- **Trial Key Usage Without Credit Card**: Participants discussed the viability of using trial API keys without entering credit card details, confirming it's possible for trial access.
   - It was noted that while trial keys are limited, there's no requirement for card info if sticking with the trial option.
- **Building Financial Data Analysis Applications**: A user shared they are developing an application focused on **financial data analysis**, utilizing citations for data accuracy.
   - Members expressed enthusiasm and offered support, recognizing the potential impact of such tools in the financial sector.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: This page describes the limitations around Cohere&#x27;s API.</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank — Cohere</a>: This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1279088423038222451)** (1 messages): 

> - `Maya LLaVA-Pretrain Dataset`
> - `Large-scale multilingual datasets`
> - `Image Captioning and VQA`
> - `Translation quality results`
> - `API support and queries` 


- **Maya LLaVA-Pretrain Dataset Launch**: The **Maya LLaVA-Pretrain** dataset is now available, featuring **4,404,776** entries across **8 languages**, designed for pretraining large language and vision models.
   - This dataset was expanded from the original [llava-pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) English dataset through machine translation and toxicity filtering.
- **Dataset Prepared with Powerful APIs**: The dataset has been prepared using the **c4ai-aya-35B** model API, refined with **command-r-plus** API for enhanced toxicity control.
   - Members expressed gratitude to another user for answering queries related to batch processing and API support.
- **Upcoming Translation Quality Results Presentation**: The team plans to present the **translation quality results** on the dataset card in the near future.
   - This aligns with their goal to improve the dataset’s usability for image captioning and visual question-answering tasks.



**Link mentioned**: <a href="https://huggingface.co/datasets/kkr5155/Maya-llava-pretrain">kkr5155/Maya-llava-pretrain · Datasets at Hugging Face</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1278795696421732435)** (31 messages🔥): 

> - `Codeium funding`
> - `Meta AI Assistant growth`
> - `Google DeepMind's Gems`
> - `State of Code Generation`
> - `Tome pivot` 


- **Codeium raises $150M, evaluates funding strategy**: Codeium closed a $150 million Series C round led by General Catalyst, valuing the company at **$1.25 billion** post-money, with total funding nearing **$243 million** since launch.
   - Co-founder Varun Mohan indicated that they have yet to utilize their **$65 million** Series B, showcasing a strategic approach to funding.
- **Meta's AI Assistant boasts impressive metrics**: Aravind Srinivas reported that Meta's AI assistant achieved **400 million MAU** and **40 million DAU**, indicating substantial user engagement.
   - There were discussions around potential licensing needs as the service scales, and the assistant's recent performance highlights growing adoption.
- **Google DeepMind introduces customizable Gems**: Google DeepMind announced the launch of customizable **Gems**, specialized versions of their Gemini model functioning as topic experts for various scenarios.
   - Features like a **Learning Coach** and **Coding Partner** aim to enhance user interactions, depending on seamless integration and execution.
- **Advancements in Code Generation tools**: Recent reports highlighted significant progress in code generation with tools like **Townie** and **Claude 3.5 Sonnet**, enhancing software development via conversational interfaces.
   - Users expressed a desire for tools to allow modifications of existing applications rather than just creating new ones from scratch, emphasizing the need for flexibility.
- **Tome reboots to focus on enterprise AI**: Tome announced a pivot to become an AI assistant aimed at helping users break into new enterprise accounts, signaling a strategic shift in focus.
   - The new direction was shared by a company representative, outlining the journey and changes that have influenced this decision.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1829261003164696703">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Impressive numbers</li><li><a href="https://x.com/1x_tech/status/1829567690681307284?s=46">Tweet from 1X (@1x_tech)</a>: Introducing NEO Beta. Designed for humans. Built for the home.</li><li><a href="https://x.com/hliriani/status/1829284172470620613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Henri Liriani (@hliriani)</a>: We&#39;re rebooting Tome to be a different company.  @magicaltome is now an AI assistant for breaking into new enterprise accounts.  Here&#39;s a bit on the journey we&#39;ve been on…</li><li><a href="https://x.com/GoogleDeepMind/status/1828855383131074997">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Over the coming days, start creating and chatting with Gems: customizable versions of Gemini that act as topic experts. 🤝  We’re also launching premade Gems for different scenarios - including Learni...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1829541138736509102">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: On the heels of the update I shared on Llama yesterday, we’re also seeing Meta AI usage growing FAST with 185M weekly actives! 🚀</li><li><a href="https://blog.val.town/blog/codegen/">How we built Townie – an app that generates fullstack apps</a>: Like Claude Artifacts, but with a backend and database</li><li><a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: no description found</li><li><a href="https://www.1x.tech/androids">Our Androids | 1X Technologies</a>: Inspired by human nature. Meet EVE and NEO and learn more about how they use embodied learning to solve problems, from meeting labor demands to everyday assistance.</li><li><a href="https://techcrunch.com/2024/08/29/github-copilot-competitor-codeium-raises-150m-at-a-1-25b-valuation/">GitHub Copilot competitor Codeium raises $150M at a $1.25B valuation | TechCrunch</a>: Codeium, a startup developing an AI-powered tool to rival GitHub Copilot, has raised $150 million at a $1.25 billion valuation.</li><li><a href="https://techcrunch.com/2024/0">2024 | TechCrunch</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1278805134695989333)** (1 messages): 

> - `LLM benchmarks`
> - `Nicholas Carlini`
> - `Latent Space podcast`
> - `Community meetup` 


- **New Podcast Episode with Nicholas Carlini**: The latest episode of the [Latent Space podcast](https://x.com/latentspacepod/status/1829173832877519152) features Nicholas Carlini from **Google DeepMind**, discussing personal insights and benchmarks for large language models.
   - Key topics include *how he uses AI*, his benchmark methods, and a critical view on *extracting training data from LLMs*, particularly citing the discontinuation of *OpenAI logprobs*.
- **Shoutout for Community Meetup**: A shoutout was made for an upcoming community meetup organized by a member, scheduled for next month.
   - Details about the meetup event can be expected to bring together AI enthusiasts and practitioners for networking and discussions.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1829173832877519152">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 Why you should write your own LLM benchmarks   w/ Nicholas Carlini of @GoogleDeepMind  Covering his greatest hits: - How I Use AI - My benchmark for large language models - Extracting Training Data...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1279169268168265750)** (57 messages🔥🔥): 

> - `Research Paper Generation Techniques`
> - `Ambassador Program Assistance`
> - `AI Scientist Limitations`
> - `CogVLM Introduction`
> - `UI/UX Patterns for GenAI` 


- **Research Paper Generation Techniques Spark Debate**: Members discussed preferences for research paper generation approaches; some suggested iterative feedback might yield better outcomes than one-shots.
   - One member noted that relying solely on 'one-shot' methods could lead to **tedious human validation**.
- **Interest in Ambassador Program Help**: A member offered assistance on building an **Ambassador program**, sharing their past experience.
   - They clarified, *“I'm not an AI research agent though,”* adding a humorous twist to their readiness to help.
- **CogVLM Model Raises Questions**: The introduction of **CogVLM** sparked discussion, with questions about its relevance in generated papers, prompting one member to say it seemed like **LLM barf**.
   - *“Unless I’m misunderstanding,”* one member reflected, hinting at the need for further clarity on the topic.
- **Explore AI Scientist Limitations**: Members commented on the **limitations of AI Scientist**, prompting insights about ongoing challenges in making AI more effective.
   - One shared a thread questioning transparency on what truly benefits users, adding *“I don't think there's much there at all.”*
- **Calls for UI/UX Patterns in GenAI**: Discussions included upcoming sessions on **UI/UX patterns for GenAI**, with links to various resources shared.
   - One of the key resources mentioned was [Maggie Appleton’s work](https://maggieappleton.com/squish-structure) highlighting innovative interface approaches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jimmykoppel/status/1828077206204981423">Tweet from Jimmy Koppel (@jimmykoppel)</a>: But all that&#39;s to stop you from looking too closely at what they actually do. Because I don&#39;t think there&#39;s much there at all.</li><li><a href="https://storm.genie.stanford.edu/">no title found</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/THUDM/CogVLM">GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型</a>: a state-of-the-art-level open visual language model | 多模态预训练模型 - THUDM/CogVLM
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278890172225683557)** (34 messages🔥): 

> - `Mojo in Blockchain`
> - `Open Sourcing Mojo`
> - `Performance Comparisons: Mojo vs Go vs C`
> - `Community Engagement in Mojo Development`
> - `Collaborations with OPENSEA` 


- **Mojo's Potential in Blockchain Protocols**: Discussions are ongoing about using **Mojo** for blockchain protocols, with one developer noting its immaturity compared to **Go, Rust, and C++**.
   - A comment mentioned that **Mojo** and **Go** are the most competent languages, but **Go's 20% performance** loss may be crucial for some projects.
- **Questions on Mojo's Open Source Future**: Inquiries were made about the availability of the **Mojo compiler's source code**, which remains closed source currently.
   - The **Modular team** aims for a balance between development speed and community engagement, indicating that they might not know when or if it will be open-sourced.
- **Performance Comparison Insights**: Members debated the performance of **Go** against **C**, with claims of slower speeds in various tasks, leading to a nuanced discussion about Go's optimization strategies.
   - Darkmatter highlighted that **Go's performance may suffer** significantly in more complex scenarios, citing a potential **30 requests per second** capacity compared to **C's 100**.
- **Community Engagement and Developer Roles**: There were conversations about the interest in expanding the **Modular team**, particularly looking for those experienced with **MLIR and compilers**.
   - The challenge lies in balancing developer resources with community engagement while keeping the project progressing efficiently.
- **Collaboration with OPENSEA**: An announcement was made about a collaboration with **OPENSEA** for a new free mint, encouraging server users to participate.
   - Participants are directed to a link for claiming, with notes that some claims may incur gas fees.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/legal/max-mojo-license">Modular: MAX &amp; Mojo Community License</a>: The MAX SDK (&quot;MAX&quot;) &amp; Mojo Community License governs what uses we allow with our software and how you can change the world with it.</li><li><a href="https://docs.modular.com/max/faq#will-it-be-open-sourced">MAX FAQ | Modular Docs</a>: Answers to questions we expect about MAX Engine.</li><li><a href="https://www.modular.com/company/career-post?4419827005&gh_jid=4419827005)">Modular: Career Post</a>: At Modular we believe a great culture is the key to creating a great company. The three pillars we work by are Build products users love, Empower people, and Be an incredible team.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1278792965090967552)** (15 messages🔥): 

> - `Memory Management Opinions`
> - `Layers of Indirection`
> - `Flexibility in Design`
> - `Mojo File Output`
> - `Error Handling in Editor` 


- **Architect's Role in Memory Management**: A member expressed that if a programmer is uncertain about whether memory referenced by a pointer should be released, it means the system architect has failed in their design.
   - They emphasized that memory management should not be a concern for application programmers, indicating a need for solid architectural design.
- **Celebration of Indirection Layers**: A member shared excitement about the 'beautiful layers of indirection' they've been working on, indicating a positive reaction to their progress.
   - They noted that the architecture works well for nearly every case, which adds to their happiness.
- **Outputting Lookup Tables to Mojo Files**: Another member announced plans to create a simple script to generate a `.mojopkg` file containing customizable lookup tables.
   - This reflects an ongoing effort to improve functionality in their software development process.
- **Error Handling in Tuples**: One member pointed out that out-of-bounds errors on tuples are still reported in the editor, affecting their development experience.
   - They mentioned that this might be related to type awareness in the editor, suggesting an improvement could involve managing invalid types better.
- **Need for InvalidType in Error Messaging**: A member proposed that introducing an `InvalidType` message could enhance clarity in error reporting, specifically for type mismatch scenarios.
   - They humorously noted that such messages would be the only time a `Type != Type` error could be useful.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1279140100516741121)** (2 messages): 

> - `fastai model export`
> - `Modular framework ambitions` 


- **Exciting Export Ideas for fastai**: A member suggested overriding **Learner.export** in fastai to export **Mojo** code for the input pipeline alongside the **PyTorch model**.
   - This approach could enhance the integration of the input pipeline and model for production use.
- **Modular's Cross-Platform Aspirations**: Hints were mentioned that **Modular** aims to address the **pickle problem** and create a **cross-platform framework agnostic model format**.
   - This initiative is expected to promote compatibility and ease of use across different frameworks.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1278800835765338175)** (46 messages🔥): 

> - `LangChain Function Calling & Streaming`
> - `Docker Connection Issues with Ollama`
> - `Building a Competent GPT for HR`
> - `Real-time Streaming Output in LangChain`
> - `GraphRAG vs Traditional RAG Techniques` 


- **LangChain's Function Calling with Streaming**: A member inquired about using LangChain v2.0 with function calling and streaming capabilities, noting difficulty finding relevant documentation.
   - Another member clarified that while function calling is supported, streaming outputs may require specific configurations or async handling in JavaScript.
- **Docker Connection Issues with Ollama**: One user reported a connection refusal error when containerizing their LangChain app, which calls the Ollama API, despite working in a non-containerized environment.
   - They later discovered that the issue was related to the base URL configuration, which was resolved by using a direct Ollama host URL.
- **Building a Competent GPT for HR Teams**: A user expressed a desire to create a specialized GPT for their HR team based on a lengthy manual, emphasizing the need for reduced hallucination and feedback mechanisms.
   - Discussion ensued about improving LLM interactions through feedback, fine-tuning, and implementing alternative RAG techniques for a more efficient system.
- **Real-time Streaming Output in LangChain**: A user faced challenges with agent executors in LangChain that gathered outputs before delivering the final response instead of streaming in real-time.
   - Suggestions were made to explore the `streamRunnable` option to potentially enable real-time output streaming.
- **GraphRAG vs Traditional RAG Techniques**: .removandesande suggested that while hybrid RAG approaches can be effective, they prefer traditional RAG techniques over graphRAG for their use case.
   - The conversation hinted at exploring new RAG methods like self-query and large context RAG as promising alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html">langchain.agents.agent.AgentExecutor &mdash; 🦜🔗 LangChain 0.2.15</a>: no description found</li><li><a href="https://v02.api.js.langchain.com/classes/langchain.agents.AgentExecutor.html">AgentExecutor | LangChain.js</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/25022>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="http://ollama:11434">)">no title found</a>: no description found</li><li><a href="https://github.com/ollama/ollama/issues/6398">When running ollama via docker, it won&#39;t respond to any request by API-call or python-client-library · Issue #6398 · ollama/ollama</a>: What is the issue? I setup the nvidia docker toolkit sucessfully on my Ubuntu 22 Machine with a RTX-4000, and start ollama as docker-container with exposed port 11434: docker run -d --gpus=all --en...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 messages): 

sourcefound: https://www.getaiphone.app/
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1278831898885357568)** (5 messages): 

> - `GymNation Success Story`
> - `LLMs in Production`
> - `LlamaIndex MLFlow Integration`
> - `LLM x Law Hackathon`
> - `Enhanced Financial Data Analysis` 


- **GymNation partners with LlamaIndex for success**: GymNation partnered with LlamaIndex to enhance member experience, resulting in a **20% increase in digital lead to sales conversion** and an **87% conversation rate** with digital leads.
   - For more details, check their [full success story](https://t.co/CXsiySj4zq).
- **Catch @seldo discussing LLMs**: Don't miss @[seldo] sharing insights on LLMs in production on **September 9th**! You can find the details in this post [here](https://t.co/Ozb1xTF2Lh).
   - This discussion promises valuable insights into deploying LLM technologies effectively.
- **LlamaIndex featured on MLFlow Podcast**: Co-founder @jerryjliu0 joined the **MLFlow podcast** to discuss the new integration with MLFlow, which streamlines logging and evaluating LlamaIndex applications.
   - Check out the full demo and insights from the podcast [here](https://t.co/2wwvn7HRBm).
- **Join the LLM x Law Hackathon!**: There's an exciting **LLM x Law Hackathon** on **September 8th**, organized by @hexapode, focusing on the merger of AI and legal practices.
   - Participants can explore three tracks including the First-Build Track, showcasing their development skills in AI [here](https://t.co/AksB9V6akr).
- **Enhanced Financial Data Analysis with MoW**: An innovative approach to financial data analysis using **Mixture of Workflows (MoW)** and Corrective RAG was discussed, featuring models like **Phi-3**, **Qwen-2**, **Gemma-2**, and **Stablelm2**.
   - This method provides **context-aware analysis** of financial statements, more details can be found [here](https://t.co/CIaEwmWB0S).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1278804321340756092)** (28 messages🔥): 

> - `Warning in LlamaIndex API`
> - `QueryEngine Deprecation Discussion`
> - `Using LlamA3 with OpenAI`
> - `Handling JSON Data in LLM`
> - `Combining Tools and Workflow Steps` 


- **Warning in LlamaIndex API Configuration**: A member reported receiving a UserWarning about config keys changing in V2, specifically mentioning 'allow_population_by_field_name' being renamed to 'populate_by_name'.
   - Another member suggested this might be related to the version of SQLAlchemy being used.
- **Clarification on QueryEngine Deprecation**: A member inquired whether QueryEngines are being deprecated, finding a reference to deprecated methods in the documentation.
   - The community clarified that it is just the method for extracting structured outputs that is deprecated, not all QueryEngines.
- **Using LlamA3 with OpenAI**: A member asked how to use Llama3 with OpenAI for generating QA embedding pairs, seeking clarification on configuration.
   - Another member advised setting the LLM object globally with Settings or passing LLM as a kwarg to 'generate_qa_embedding_pairs'.
- **Handling JSON Data in LLM Workflow**: A user created an agent to make external API calls returning JSON data and sought advice on how to format this data for the LLM.
   - Instructions were given to format the response nicely before sending it back to the LLM to avoid complications.
- **Combining Tools and Workflow Steps**: A new user inquired about examples showing the integration of tools and workflow steps in LlamaIndex, feeling unclear about the connection.
   - A member shared a specific example demonstrating how to build an agent with integrated workflows and tool calling.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/query_engine/">(Deprecated) Query Engines + Pydantic Outputs - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1278950929756065876)** (1 messages): 

> - `LitServe`
> - `LlamaIndex`
> - `AI model serving` 


- **LitServe Enhances AI Model Deployment**: LitServe is a high-performance serving engine that allows developers to deploy and manage a variety of **AI models** efficiently.
   - When paired with **LlamaIndex**, it transforms into a versatile tool for building intelligent applications.
- **Combining LitServe and LlamaIndex**: The combination of **LitServe** and **LlamaIndex** empowers developers with a powerful data framework for AI applications.
   - This synergy brings increased ease and flexibility in serving AI models in real-world scenarios.



**Link mentioned**: <a href="https://medium.com/ai-artistry/serving-ai-models-at-lightning-speed-with-litserve-and-llamaindex-4e7decdb5ae1">Serving AI Models at Lightning Speed with LitServe and LlamaIndex</a>: Ankush k Singal

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1278797119464935457)** (11 messages🔥): 

> - `House Party`
> - `Terminal App Recommendations`
> - `Obsidian OI Plugin Issues`
> - `GPT-4o Interaction Memory` 


- **House Party Showtime**: Join us for a **House Party** next week at the earlier time to gather more together! [Join the Discord Event](https://discord.gg/open-interpreter-1146610656779440188?event=1278796923892924498).
   - This event aims to enhance community engagement and create a fun atmosphere ❤️.
- **Seeking Terminal App Alternatives**: A member is seeking recommendations for a terminal app on KDE, expressing concerns about screen bleeding while using **Konsole**.
   - Another user reported experiencing similar issues while running in a standard conga terminal with **GPT-4o-mini**.
- **Obsidian OI Plugin Trouble**: A user praised videos on the **Obsidian OI plugin** but encountered issues and is seeking advice for global installation problems.
   - They were advised to provide details in the specified channel regarding the installation process and interface being used.
- **Concerns Over GPT-4o's Memory**: A member expressed frustration that **GPT-4o** does not remember past interactions, querying how to utilize it effectively in web development.
   - They pondered asking GPT-4o for tips on creating a memory system, seeking advice from others in the channel.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1279126210349236278)** (2 messages): 

> - `Potential Applications`
> - `House Party Discussion` 


- **Excitement for Development Involvement**: A member expressed enthusiasm, stating they hope to see some **developments** and are eager to involve themselves in any way they can.
   - They also mentioned having thoughts on **potential applications** and are interested in discussing this further.
- **House Party for Discussion**: Another member proposed that a **house party next Thursday** would be a great opportunity to chat about the potential applications.
   - This suggests a casual setting for sharing insights and ideas within the community.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278795160880156793)** (3 messages): 

> - `GameNGen real-time simulation`
> - `AgentOps excitement`
> - `YouTube shoutout` 


- **GameNGen: Revolutionizing Game Simulation**: Introducing _GameNGen_, the first game engine powered entirely by a neural model, capable of simulating **DOOM** at over **20 frames per second** on a single TPU, achieving a PSNR of **29.4**.
   - Human raters struggled to distinguish between clips of the game and simulations, highlighting the model's efficacy and potential in the gaming sector.
- **AgentOps Team Generates Excitement**: Members expressed excitement over the potential developments from **Adam and the AgentOps** team, indicating high expectations for their upcoming projects.
   - This enthusiasm reflects a broader interest in advancements within the realm of agent technology.
- **YouTube Shoutout Goes Viral**: A member shared a [YouTube video](https://youtu.be/z4QsBsO3SS0?t=371&si=lzexLc5j0gjdjRht) featuring a shoutout to another member, generating excitement within the community.
   - This mention boosts community engagement and showcases recognition among peers.



**Link mentioned**: <a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1278886572242239559)** (14 messages🔥): 

> - `Google buying GPUs`
> - `RunwayML removes Stable Diffusion repos`
> - `Issues caused by repo deletions`
> - `Generating realistic images`
> - `Re-LAION-5B launch` 


- **Google's GPU Acquisition Sparks Curiosity**: Members questioned why **Google** is purchasing **GPUs** from **NVIDIA** despite their own **TPUs**, suggesting a potential gap or interest in NVIDIA technologies.
   - *Is the TPU not enough?* One member mused about Google's strategic choices in hardware.
- **RunwayML Deletes All Stable Diffusion Repos**: Discussion erupted over **RunwayML** deleting all their **Stable Diffusion 1.5** repositories on **HuggingFace** and **GitHub**, leaving many users frustrated.
   - One member noted that this action broke many functionalities in **Diffusers 1.5**, particularly impacting single file loading.
- **Disruption from Repo Deletions**: Members expressed annoyance about the seemingly thoughtless nature of RunwayML's deletions, with one stating it felt like they wanted to cause **disruption**.
   - Speculation arose around potential legal issues, but no specific reasons were confirmed for the deletions.
- **Creating Realistic Images for Book Covers**: A member sought advice on generating **comic book-style** or cartoonish images for their novel covers, struggling with overly realistic outputs from **DALL·E**.
   - Despite attempts, they found DALL·E not catering to the specific style they desired.
- **Launch of Re-LAION-5B**: Members celebrated the launch of **Re-LAION-5B**, a cleaned version of the **LAION-5B** dataset, which addresses previous concerns.
   - The dataset was updated in partnership with key organizations to ensure safety and compliance, marking a significant milestone.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://laion.ai/blog/relaion-5b/">Releasing Re-LAION 5B: transparent iteration on LAION-5B with additional safety fixes | LAION</a>: &lt;p&gt;Today, following &lt;a href=&quot;https://laion.ai/notes/laion-maintenance/&quot;&gt;a safety revision procedure&lt;/a&gt;, we announce Re-LAION-5B, an updated version of LAION...</li><li><a href="https://huggingface.co/runwayml">runwayml (Runway)</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/)** (1 messages): 

mega_b: https://laion.ai/blog/relaion-5b/
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1278793158058180700)** (10 messages🔥): 

> - `OpenAI Funding Round`
> - `Chatbot Wars`
> - `Meta AI User Growth` 


- **Tech Giants Eye OpenAI**: Nvidia, Apple, and Microsoft, the top three most valuable tech companies, are in discussions to invest in **OpenAI** as part of a new **$100 billion funding round** [source](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round).
   - This move highlights the interest of major players in AI funding and innovation.
- **Chatbot Wars Heat Up**: The competition intensifies as **ChatGPT** boasts over **200 million weekly users**, while **Meta AI** is also gaining traction in the market [source](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx).
   - However, doubts remain as to whether Meta AI is being used effectively or has accidental engagement.
- **Meta AI's Limited Availability**: Concerns were raised that **Meta AI** isn't accessible everywhere, particularly in the **EU**, which may affect its growth [source](https://x.com/amir/status/1829248019910537470?s=46).
   - With only **40 million DAUs**, its user base lags significantly behind ChatGPT's.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/amir/status/1829248019910537470?s=46">Tweet from Amir Efrati (@amir)</a>: Begun, the chatbot wars have   ChatGPT: 200M+ weeklies.   Meta AI likely not far behind (though unclear if people are using it the same way or accidentally!)  https://www.theinformation.com/articles/m...</li><li><a href="https://x.com/markgurman/status/1829233740704559182">Tweet from Mark Gurman (@markgurman)</a>: Nvidia, Apple and Microsoft — the three most valuable tech companies — are in talks to invest in OpenAI as part of the company’s new $100 billion funding round. https://www.bloomberg.com/news/articles...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278967107492778015)** (3 messages): 

> - `Tinygrad Cloud Service`
> - `Impact of System Prompts` 


- **Tinygrad Launches Affordable Cloud Solution**: Tinygrad announced a new cloud service offering a **4090 GPU** and **500 GB of storage** for just **$60/month**, making it **3x cheaper** than competitors like Vast AI.
   - *Coming soon: CLOUD=1* lets users run Tinygrad locally while leveraging cloud speed for performance enhancements with **10-step processing**.
- **Inquiry on System Prompts Impact**: A member inquired if there are any papers studying the **impact of system prompts** on evaluation scores.
   - They questioned whether it’s possible to **meaningfully shift scores** through different prompting techniques.



**Link mentioned**: <a href="https://x.com/__tinygrad__/status/1829379908017238210?s=46">Tweet from the tiny corp (@__tinygrad__)</a>: Coming soon: CLOUD=1  For $60/month (3x cheaper than vast ai), we&#39;ll rent you a 4090 and 500 GB of cloud storage.  Use tinygrad as normal on your dev machine, but it runs things fast in the cloud....

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1278800875212636261)** (11 messages🔥): 

> - `QLoRA memory issues`
> - `Multi-GPU evaluation in TorchTune`
> - `CUDA errors during training`
> - `Memory requirements for A6000 GPUs`
> - `Training sequence lengths` 


- **QLoRA memory issues raised**: A member expressed suspicion that their setup should have sufficient memory for **QLoRA**, questioning whether something went wrong.
   - They mentioned a **CUDA error** indicating illegal memory access while running configurations with **4 48GB GPU cards**.
- **Clarifications on memory requirements for GPUs**: A member pointed out that **A6000 GPUs** are now **48GB** instead of **24GB**, indicating that four such cards should be adequate for the task.
   - They also noted potential strain on resources without CPU offloading, suggesting sequence length might be a factor.
- **Concerns about sequence lengths**: Another member tried different sequence lengths (**8K** and **4K**) for training, implying there could be issues with memory depending on the length used.
   - They mentioned some specifics of their training setup that could influence **vRAM** during the process.
- **Multi-GPU evaluation in TorchTune**: A member queried whether **multi-GPU evaluation** support exists in **TorchTune**, indicating potential interest in optimizing performance.
   - Their question highlighted a common need for scalability in training setups using multiple GPUs.
- **Understanding illegal memory access errors**: Following an operating error, a member received suggestions to set **CUDA_LAUNCH_BLOCKING=1** for debugging illegal memory access issues during training.
   - This points to complexities in using **PyTorch** with distributed training while managing memory effectively.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1278791464586969140)** (5 messages): 

> - `LinkedIn Auto Jobs Applier`
> - `DSPy Community Engagement` 


- **Confusion Over Repo Connection**: A member expressed confusion regarding the connection between a statement made and the [linked GitHub repository](https://github.com/feder-cr/linkedIn_auto_jobs_applier_with_AI). Another member clarified that the repo was totally separate but wanted to showcase it to the DSPy community to inspire involvement.
   - *It’s getting over 2k likes each day*, indicating significant interest in the tool.
- **Concerns About GitHub Issues**: A member raised concerns about the performance of the LinkedIn Auto Jobs Applier, asking if it had been tested, pointing to GitHub issues showing room for improvement. The discussion hinted that feedback on the repo suggests there’s a lot left to be desired.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1278853571194130434)** (5 messages): 

> - `Workshop on Useful and Reliable AI Agents`
> - `DSPy: Prompt Optimization for LM Programs`
> - `AgentOps`
> - `Nelima`
> - `Bay Area AI Meetup` 


- **Workshop on Useful and Reliable AI Agents**: A member shared a link to the [YouTube video](https://www.youtube.com/live/-aKRsvgDEz0) titled 'Workshop on Useful and Reliable AI Agents' discussing the importance of accuracy, reliability, and cost-effectiveness in AI agents.
   - The workshop aims to address the active research surrounding AI agents and how they can be effectively utilized in real-world scenarios.
- **AgentOps Tools for Building AI Agents**: Information was shared about [AgentOps](https://agents.staf.ai/AgentOps), which provides tools for building agents with features like graphs and monitoring.
   - Their goal is to eliminate the guesswork in prompting agents, emphasizing a transparent approach to developing AI solutions.
- **DSPy Seminar with Michael Ryan**: The upcoming Bay Area AI meetup hosted by @ChiefScientist features Michael Ryan discussing 'DSPy: Prompt Optimization for LM Programs' and the concept of LM Programs.
   - Michael, a Stanford student, will present his latest optimization work, including the MIPROv2 algorithm, at the event sponsored by @Neo4j.
- **Interest in Recording of Event**: A member expressed excitement about the aforementioned event and acknowledged that it is being recorded for publication.
   - This reflects the community’s eagerness to access valuable insights shared during the meetup.
- **DSPy Usage Questions**: A user inquired about the appropriate channel for posting doubts regarding the usage of DSPy.
   - This indicates an active engagement within the community looking for support and guidance on the DSPy library.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ChiefScientist/status/1829231009344434400?t=wow3U2BluHEv16-MI2YcaQ&s=19">Tweet from Alexy 🤍💙🤍 (@ChiefScientist)</a>: Super excited to host Michael Ryan at the post-@AIconference http://Bay.Area.AI meetup hosted by @github HQ in SOMA SF!  DSPy: Prompt Optimization for LM Programs Michael Ryan, @Stanford   ​It has nev...</li><li><a href="https://www.youtube.com/live/-aKRsvgDEz0">Workshop on Useful and Reliable AI Agents</a>: AI agents have become an active area of research. But to be useful in the real world and at scale, agents need to be accurate, reliable, and cheap. Learn how...</li><li><a href="https://docs.google.com/spreadsheets/d/1VnOv_C0v_FgDeKuQBaGuMNsWgoWOpLkGbE_XS_2Vb3Q/edit?gid=0#gid=0">Agent Database By AgentOps.ai </a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1279021509838639264)** (5 messages): 

> - `Axolotl GitHub Documentation`
> - `Training LLaMA 70B`
> - `NVIDIA A6000 GPUs` 


- **Request for Dark Mode on Axolotl GitHub Docs**: A member expressed a desire for the [Axolotl GitHub documentation](https://github.com/axolotl) to be available in **dark mode**, citing discomfort with the current light mode.
   - They mentioned frequent visits to check configuration parameters, emphasizing that the current light mode is problematic.
- **Hardware Considerations for LLaMA 70B Training**: Discussion arose regarding the hardware requirements for full training of **LLaMA 70B**, with one member inquiring about current recommendations.
   - They speculated that just a few **NVIDIA A6000** GPUs might suffice given recent improvements in training efficiency.
- **3x A6000 GPUs Should Suffice for 70B**: A member responded affirmatively to the GPU question, suggesting that **3x A6000 GPUs** should be adequate for training the full **70B model**.
   - This was met with some surprise regarding the hardware's capabilities, indicating potential advancements in GPU performance.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1278894713742561300)** (1 messages): 

> - `Axolotl`
> - `Hugging Face transformers` 


- **Axolotl faces no changes after updates**: A member highlighted that the results for **Axolotl** are even better now, and no changes are required following the recent updates.
   - This comes in light of the [Pull Request #33198 by Rocketknight1](https://github.com/huggingface/transformers/pull/33198) at **Hugging Face** which improves chat templates.
- **New assistant prefill feature added**: The recent Pull Request addresses a long-requested feature for **assistant prefill**, allowing the model to start its response automatically.
   - This enhancement aims to provide a more streamlined experience in the **TextGenerationPipeline**, using a slightly hacky method to initiate responses.



**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/33198">Add assistant prefill for chat templates and TextGenerationPipeline by Rocketknight1 · Pull Request #33198 · huggingface/transformers</a>: Something that&amp;#39;s been requested several times both internally and on Github is assistant prefill: The ability to begin the model&amp;#39;s response for it and let it continue. We use a slightl...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1278976543678533692)** (3 messages): 

> - `Llama 3.1`
> - `Uninitialized Special Tokens`
> - `Fixing Untrained Tokens` 


- **Llama 3.1 still has special token issues?**: A member inquired if **Llama 3.1 base** still suffers from issues with uninitialized special tokens, specifically regarding embeddings being out of distribution.
   - The concern indicates ongoing challenges with handling special tokens in the model.
- **New Fix for Untrained Tokens Introduced**: Another member revealed that an option, `fix_untrained_tokens: true`, has been added to potentially address the issue of uninitialized special tokens.
   - This enhancement suggests a proactive approach to refining the model's performance.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1278839619231682632)** (6 messages): 

> - `Groq Leaderboard Update`
> - `Documenting Model Steps`
> - `Java GIS Geometry Initialization`
> - `Temperature Settings in Evaluations`
> - `OSSHandler Parameter Adjustments` 


- **Groq awaits PRs for leaderboard entry**: It was noted that **Groq** has not yet been added to the leaderboard as the team is still waiting for their PRs, which are expected around next week.
   - This has led to some ongoing discussions about their integration and anticipated performance.
- **Steps documentation affirmed**: A member confirmed that ensuring model steps are documented correctly is essential for reproducibility.
   - The statement emphasized that proper documentation enhances model understandability and usability.
- **Java test case reveals performance issues**: A user shared a **Java** test case where their model did not perform well, particularly regarding the initialization of GIS geometry presentation.
   - The conclusion drawn was that providing a direct example may be more beneficial than complex function calls, given the user's query.
- **Queries on evaluation temperature settings**: Questions arose regarding whether model evaluations are strictly done with a greedy decode and temperature of 0 to ensure fair metrics.
   - Members discussed implications for randomness in outputs with reference to recent GitHub links on the leaderboard evaluation criteria.
- **OSSHandler default parameters discussion**: It was noted that the default temperature for **OSSHandler** is set to 0.001, and while adjustments were considered, it was ultimately decided not to change it.
   - This decision aligns with maintaining consistent function outputs and optimizing the model's performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#model-specific-optimization">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla</li><li><a href="https://github.com/ShishirPatil/gorilla/discussions/562">Set Model Temperature to 0 for Consistent Leaderboard Results · ShishirPatil/gorilla · Discussion #562</a>: The current model generation script (model_handlers) uses a default temperature of 0.7 for inference. This introduces some degree of randomness into the model output generation, leading to potentia...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1278838600510996593)** (2 messages): 

> - `tinygrad capabilities`
> - `sparsity techniques` 


- **Questioning tinygrad's strengths**: *codeman3786* inquired if **tinygrad** is primarily effective for **statically scheduled operations** and not suitable for methods involving **semi-structured sparsity** or weight selection.
   - This prompted *georgehotz* to ask if there was a specific example of something that *codeman3786* could not achieve using tinygrad.
- **Instance of tinygrad limitations**: Georgehotz's response indicated an openness to discuss potential limitations by asking for examples where tinygrad may fall short.
   - This interaction suggests a community interest in exploring the practical limits of tinygrad's performance and versatility.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1279210301232906393)** (2 messages): 

> - `Tensor.cat with sharded tensors`
> - `Padding and reshaping issues`
> - `Batch dimension manipulation` 


- **Tensor.cat struggles with sharded tensors**: A user encountered an error when trying to **Tensor.cat** two sharded tensors along the batch axis, specifically stating *padding not supported for arg=((0, 9), (0, 0), (0, 0))*.
   - They provided a workaround using `unsqueeze` but faced another error related to reshaping dimensions.
- **User queries fundamental support for operations**: The user is questioning whether the inability to concatenate sharded tensors is a **fundamental problem** or just unsupported functionality, seeking clarity on the issue.
   - They are exploring options, including modifying the code to support an extra batch dimension or executing multiple operations to avoid using **Tensor.cat**.


  

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

---
id: e87581dc-f03a-4370-afce-92a804557693
title: 5 small news items
date: '2024-06-06T02:50:37.633247Z'
original_slug: ainews-5-small-news-items
description: >-
  Leopold Aschenbrenner released a detailed 5-part series on AGI timelines
  predicting a **trillion dollar cluster** from current AI progress, while
  **Cohere** completed a significant **$450 million funding round** at a $5
  billion valuation. Additionally, DeepMind shared advances in **uncertainty
  quantification** for LLMs, and **Mistral AI** launched a fine-tuning API for
  their models, signaling ongoing innovation in model efficiency and
  customization.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->**AGI Realism may be what Humanity Needs**

> AI News for 6/4/2024-6/5/2024!
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**401** channels, and **3628** messages) for you. 
Estimated reading time saved (at 200wpm): **404 minutes**.

- OpenAI still says [ChatGPT's voice mode is "coming soon"](https://www.youtube.com/watch?v=4w0Pqs3CuWk)
 ![image.png](https://assets.buttondown.email/images/d99a5d17-2b05-43ed-a6fb-dd97697cc383.png?w=960&fit=max) 
- Leopold Aschenbrenner [launched a 5 part AGI timelines piece dedicated to Ilya](https://x.com/leopoldasch/status/1798016486700884233) together with [a Dwarkesh pod](https://www.youtube.com/watch?v=zdbVtZIn9IM), predicting a **trillion dollar cluster** on current rates of progress  ![image.png](https://assets.buttondown.email/images/fe9c361c-3e46-4212-ad48-62a44f6a1c77.png?w=960&fit=max) 
- Tom Yeh [hand-illustrates llm.c](https://x.com/ProfTomYeh/status/1798042265883156651)  ![image.png](https://assets.buttondown.email/images/88c9b7bf-60f0-4604-bdc0-27d30bf1dc3b.png?w=960&fit=max) 
- Will Brown dropped a [comprehensive GenAI Handbook](https://x.com/willccbb/status/1798423849870270671)  ![image.png](https://assets.buttondown.email/images/257555ea-33ad-4c40-8ff2-698f8b1bb6a4.png?w=960&fit=max) 
- and Cohere [completed its $450m raise at $5b valuation](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/)   but has not yet announced it.

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

**AI Models and Architectures**

- **New Models and Architectures**: [@arankomatsuzaki](https://x.com/arankomatsuzaki/status/1798177198781899194) shared a DeepMind paper on **Uncertainty Quantification in LLMs**. [@hardmaru](https://twitter.com/hardmaru/status/1798202333383516613) highlighted xLSTM, an **extension of LSTM** that performs favorably compared to Transformers and State Space Models in performance and scaling. [@omarsar0](https://twitter.com/omarsar0/status/1798010546522103898) discussed a study on the **geometry of concepts in LLMs**, finding that simple categorical concepts are represented as simplices and hierarchically related concepts are orthogonal.
- **Efficiency Improvements**: [@omarsar0](https://twitter.com/omarsar0/status/1798373841741185261) shared a paper proposing an implementation that **eliminates matrix multiplication operations from LLMs** while maintaining performance at billion-parameter scales, potentially reducing memory consumption by more than 10x. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798104182232101188) discussed a survey on **Parameter-Efficient Fine-Tuning (PEFT) methods** for large models, categorizing them into additive, selective, reparameterized, and hybrid approaches.
- **Alignment and Safety**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1798156694012490096) outlined a scenario for how **building misaligned AGI could lead to humanity losing control**, with an AGI exploiting privileged access to a lab's servers. [@omarsar0](https://twitter.com/omarsar0/status/1798014572663583165) shared an overview of methods for **automated alignment of LLMs**, exploring directions like aligning through inductive bias, behavior imitation, model feedback, and environment feedback.

**Tools and Frameworks**

- **LangChain and LangGraph**: [@hwchase17](https://twitter.com/hwchase17/status/1798386148982878477) introduced a new DeepLearning.AI course on **building AI agents using LangGraph**, an extension of LangChain for developing controllable agents with persistence and agentic search capabilities. [@llama_index](https://twitter.com/llama_index/status/1798049438814081138) demonstrated how a **longer context window in a LlamaIndex agent** attempting to answer a multi-part question from heterogeneous documents leads to better performance.
- **Hugging Face and NVIDIA Integrations**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1798022781713731595) noted that **Hugging Face is becoming a gateway for AI compute**, with NVIDIA NIM now directly accessible from the Hugging Face Hub for the Llama3 model. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798135458842517629) discussed **Optimum-NVIDIA, a Hugging Face inference library** leveraging NVIDIA's FP8 format and TensorRT-LLM software for faster LLM inference.
- **Mistral AI and Fine-Tuning**: [@sophiamyang](https://twitter.com/sophiamyang/status/1798415316180988403) announced the release of **Mistral's fine-tuning API**, allowing users to fine-tune their own Mistral models and deploy them efficiently on La Plateforme. [@HamelHusain](https://twitter.com/HamelHusain/status/1798412100072813000) shared a live demo of the API, walking through data preparation, hyperparameter selection, and integrations.

**Datasets and Benchmarks**

- **Synthetic Data Generation**: [@_philschmid](https://twitter.com/_philschmid/status/1798388387822317933) outlined a pipeline for **generating synthetic data for fine-tuning custom embedding models**, involving creating a knowledge base, chunking data, generating questions using an LLM, optionally generating hard negative examples, deduplicating and filtering pairs, and fine-tuning embedding models with Sentence Transformers 3.0.
- **Evaluation Metrics**: [@abacaj](https://twitter.com/abacaj/status/1798366581254504573) built a benchmark for **analyzing malicious Solidity contract code**, finding that only top closed models like GPT-4o and Claude-Opus can occasionally identify malicious code, while open models fail more than 95% of the time. [@mervenoyann](https://twitter.com/mervenoyann/status/1798274389928300678) noted that **MMUPD, a comprehensive evaluation benchmark of multi-modal LLMs in video analysis**, is now hosted on the Hugging Face Hub as a leaderboard.
- **Domain-Specific Datasets**: [@_arohan_](https://twitter.com/_arohan_/status/1798401202138566953) highlighted **Google's Gemini 1.5 model outperforming proprietary models** on many subtasks in the Video-MME benchmark for multi-modal LLMs in video analysis. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798343869441990778) shared a paper comparing the performance of Gemini 1.5 Flash and GPT-4o on the Video-MME benchmark.

**Applications and Use Cases**

- **Enterprise AI and RAG**: [@llama_index](https://twitter.com/llama_index/status/1798376976849469559) shared a full video tutorial on **building enterprise RAG (Retrieval-Augmented Generation) with Bedrock and Ragas.io**, covering synthetic dataset generation, critic-based evaluation, and fine-tuning. [@RazRazcle](https://twitter.com/RazRazcle/status/1798040468951048411) interviewed @gogwilt, Co-founder of Ironclad, discussing how they succeeded in **using AI for contract negotiation**, with over 50% of contracts for top customers negotiated by Ironclad AI.
- **AI Assistants and Agents**: [@svpino](https://twitter.com/svpino/status/1797976775529844823) built an **AI assistant that listens and uses the webcam to see the world**, explaining the process in a video tutorial. [@bindureddy](https://twitter.com/bindureddy/status/1798204209231536177) predicted that **AI assistants will become essential** and people's dependence on them will grow exponentially.
- **Creative AI and Multimodal Models**: [@suno_ai_](https://twitter.com/suno_ai_/status/1798036388329472380) announced a contest to **create songs from any sound using their VOL-5 model**, with winners receiving early access and one winner's video being shared on social media. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1798385843662680520) showcased an **AI-powered tool for making non-player characters in video games playable**, a collaboration between @cubzh_, @GigaxGames, and @huggingface.

**Discussions and Opinions**

- **AI Timelines and Risks**: [@leopoldasch](https://twitter.com/leopoldasch/status/1798156694012490096) argued that **AGI by 2027 is strikingly plausible** based on the progress from GPT-2 to GPT-4 and projected trends in compute, algorithmic efficiencies, and model capabilities. [@_sholtodouglas](https://twitter.com/_sholtodouglas/status/1798052154709852198) described Leopold's essay as capturing the worldview of key players in the AI field, predicting a wild few years as timelines potentially hold.
- **Compute and Scaling**: [@ylecun](https://twitter.com/ylecun/status/1798333227175690533) proposed the concept of **objective-driven AI**, where intelligent systems require the ability to reason, plan, and satisfy guardrails according to their internal world model, and the key challenge becomes designing appropriate guardrails. [@ethanCaballero](https://twitter.com/ethanCaballero/status/1798385264248885525) noted that as it becomes clear that **energy and power are the new bottlenecks for scaling to AGI**, certain stocks may skyrocket in the coming years.
- **Open Source and Democratization**: [@ylecun](https://twitter.com/ylecun/status/1798118502198645245) shared an article discussing the **benefits and risks of open-source AI versus proprietary AI** controlled by a few big players, arguing that those who worry most about AI safety tend to overestimate the power of AI. [@far__el](https://twitter.com/far__el/status/1798375007460225096) predicted that **Meta and other companies will not open-source powerful AI**, and we are headed towards an "AGI monarchy".

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

Here is a summary of the recent AI developments, organized by topic and with key details bolded and linked to relevant sources:

**AI Model Releases and Capabilities**

- **Potential GPT-5 Release**: The Information reports that [**GPT-5 may be released in December 2024**](https://i.redd.it/rxh46fczao4d1.jpeg), suggesting significant advancements in OpenAI's language model capabilities.
- **Advanced AI Models**: Microsoft CTO Kevin Scott claims that [**upcoming AI models can pass PhD qualifying exams**](https://x.com/tsarnick/status/1798167323893002596), indicating substantial improvements in memory and reasoning abilities.
- **Character Voice Generation**: A [YouTube video demonstrates GPT-4o's ability to generate character voices](https://www.youtube.com/watch?v=4w0Pqs3CuWk), showcasing the model's versatility in speech synthesis.
- **Robotic Future**: Nvidia promises that ["everything is going to be robotic" as AI becomes more advanced](https://www.youtube.com/watch?v=nxO_t5N82m0), hinting at the increasing integration of AI in various domains.
- **AI Clones in the Workplace**: Zoom's CEO predicts that [**AI clones will eventually handle a significant portion of people's jobs**](https://qz.com/zoom-ceo-eric-yuan-ai-avatar-jobs-1851518757), potentially transforming the nature of work.

**AI Outages and Concerns**

- **Simultaneous AI Service Outages**: Major AI services [ChatGPT, Claude, and Perplexity experienced simultaneous outages](https://techcrunch.com/2024/06/04/ai-apocalypse-chatgpt-claude-and-perplexity-are-all-down-at-the-same-time/), raising concerns about the reliability and impact of these services.
- **Prolonged ChatGPT Downtime**: [ChatGPT was down for approximately 12 hours](https://i.redd.it/9qg0eyc6fk4d1.jpeg), causing issues for users relying on the service and highlighting the need for robust infrastructure.
- **Whistleblowers and Safety Concerns**: Current and former OpenAI employees, along with other AI researchers, are [willing to reveal confidential information to the public](https://www.reddit.com/gallery/1d80n63) regarding AI risks and safety concerns. An OpenAI safety researcher quit and [signed a letter calling for AI labs to support employees speaking out about these issues](https://twitter.com/clwainwright/status/1798013345926447486).
- **Cybersecurity Vulnerabilities**: Leopold Aschenbrenner was [fired from OpenAI after warning the board about cybersecurity vulnerabilities that China could exploit](https://v.redd.it/b8hjkl8fao4d1), raising questions about the handling of security concerns within the company.
- **Race for AI Dominance**: OpenAI insiders warn of a ["reckless" race for AI dominance in a New York Times article](https://www.nytimes.com/2024/06/04/technology/openai-culture-whistleblowers.html), highlighting the potential risks associated with the rapid development of AI technologies.

**AI Investments and Partnerships**

- **Elon Musk's Chip Allocation**: Elon Musk instructed Nvidia to [prioritize shipping processors to X and xAI over Tesla](https://www.cnbc.com/2024/06/04/elon-musk-told-nvidia-to-ship-ai-chips-reserved-for-tesla-to-x-xai.html), suggesting a focus on AI development in his companies.
- **UAE-US AI Partnership**: The UAE is [partnering with the US in AI, using its $2 trillion sovereign wealth fund to become a global AI powerhouse](https://www.benzinga.com/news/24/06/39153123/now-uae-forges-us-tie-up-with-1-5b-deal-to-become-global-ai-powerhouse), highlighting the increasing international competition in the field.
- **OpenAI-Google Collaboration**: Ilya Sutskever and Jeff Dean [published a US patent together on May 30, 2024](https://i.redd.it/mxaehyg0ni4d1.png), suggesting a potential collaboration between OpenAI and Google in AI research and development.

**AI Models and Benchmarks**

- **SDXL Model Parameters**: The [SDXL model has 2.6B parameters for the UNET, 3.5B parameters including text encoders, and 6.6B parameters for the full pipeline with the Refiner](https://i.redd.it/fkij1pomxi4d1.jpeg), providing insights into the model's architecture and complexity.
- **Yi-1.5-34B Model Performance**: The [Yi-1.5-34B model is the highest-ranked ~30B model and Apache 2.0 model on the LMSYS leaderboard](https://i.redd.it/v5w3myp3gi4d1.png), demonstrating its strong performance compared to other models of similar size and licensing.
- **L3-MS-Astoria-70b Model Ranking**: The [L3-MS-Astoria-70b model becomes the top-ranked model on the Uncensored General Intelligence Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard), showcasing its capabilities in general intelligence tasks.
- **GPT-4o Usability**: Despite GPT-4o's high rankings on MMLU and LMSYS benchmarks, some users find it [harder to prompt and follow instructions compared to other models](https://www.reddit.com/r/singularity/comments/1d7wmjn/mmlu_lmsys_vs_vibes_on_gpt4o/), highlighting the importance of user experience and model usability.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Finetuning Techniques and Model Integration**:

- Members discussed the importance of **finetuning models** with tools like **Deepspeed zero2** and **Qlora**, highlighting successful integration for **Llama3** and memory management strategies like disk offloading ([Unsloth AI](https://discord.com/channels/1179035537009545276)).
- **Mistral Fine-Tuning Hackathon** generated excitement, encouraging participants to explore Mistral's new capabilities, detailed in the [Mistral tutorial](https://docs.mistral.ai/capabilities/finetuning/) and corresponding [YouTube demos](https://youtu.be/zXFxmI9f06M) ([LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560)).

**2. Issues in Model Training and Optimization**:

- Members expressed frustration over **OOM (Out of Memory)** errors during model training, seeking solutions like efficient VRAM management techniques and validating YAML configurations ([OpenAccess AI Collective](https://discord.com/channels/1104757954588196865)).
- Troubleshooting advice was shared for issues including **CUDA library mismatches** in Jarvis Labs and **GGUF compatibility** in LM Studio ([HuggingFace](https://discord.com/channels/879548962464493619) and [LM Studio](https://discord.com/channels/1110598183144399058)).

**3. New Tools and Resources in AI**:

- **Stable Audio Open** was released by Stability AI for generating short audio pieces, emphasizing local fine-tuning with custom data ([Stable Diffusion](https://stability.ai/news/introducing-stable-audio-open)).
- Various valuable resources were shared, such as a [comprehensive LLM resource guide](http://genai-handbook.github.io/) by William Brown and the [FineWeb technical report](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) for high-performance LLMs ([HuggingFace](https://discord.com/channels/879548962464493619)).

**4. Community Concerns and Collaborative Projects**:
    
- Concerns over **credit distribution and server performance** were widely discussed, with numerous members reporting issues receiving credits or facing **502 Gateway errors** ([LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) and [OpenRouter](https://discord.com/channels/1091220969173028894)).
- Collaborative efforts in learning and implementing new AI features included discussions on **Flash-attn** GPU compatibility and **RAG chatbot integration** with tools like Verba ([Nous Research AI](https://discord.com/channels/1053877538025386074) and [LangChain](https://discord.com/channels/1038097195422978059)).

**5. Security and Ethical Discussions in AI**:

- Security concerns were raised following a **Hugging Face breach** where private tokens were exposed, leading to discussions on the reliability of internet-based data ([HuggingFace](https://discord.com/channels/879548962464493619)).
- Ethical issues of **AGI development incentives** and ensuring fair model use were debated, stressing the importance of aligned AI behavior and proper reward models within LLM architectures ([Interconnects](https://discord.com/channels/1179127597926469703) and [Latent Space](https://discord.com/channels/822583790773862470)).

---

# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Model and Workshop Mania**: Engineers actively debated the best **code models for fine-tuning**, with many citing a lack of specific resources or models that stand out. Meanwhile, **workshop attendees** clamored for access to slides and links, with recommendations to apply for **Modal credits** via a [credit form](https://bit.ly/modal-credits) to claim $500 bonuses.

- **Credit Confusion Across Platforms**: Across various platforms, users expressed confusion regarding **credit distribution**, such as Modal's additional $500 and Replicate's redemption process. For assistance with Modal's offer, Charles extended help via email, while for issues with **Replicate credits**, users were directed to message with their details for support.

- **Curating Fine-Tuning Resources**: A comprehensive list of ***LLM fine-tuning explainers*** was spotlighted, available via [this LLM guide](http://genai-handbook.github.io). Additionally, the excitement was apparent for the **Mistral Fine-Tuning Hackathon** with its development coinciding with an API launch, suggesting heightened interest in exploring Mistral's capabilities and resources like the [fine-tuning tutorial](https://docs.mistral.ai/capabilities/finetuning/) and [YouTube demos](https://youtu.be/zXFxmI9f06M).

- **Honing Fine-Tuning Techniques**: The community shared knowledge and sought advice on **Mistral fine-tuning**, including discussions on vertical integration, API advantages, and memory management. Moreover, Predibase users extolled its methodology for reusing base models and suggested improvement for an enhanced fine-tuning process such as access to more epochs and UI data filtering walkthroughs.

- **Troubleshooting Tech Stacks**: Various challenges in setting up different technologies such as **Axolotl**, Jarvis Labs' CUDA version mismatches, and debugging LangChain notebooks were addressed with collaborative problem-solving. Solutions ranged from using Docker for Axolotl ease to updating CUDA libraries and advising environment variable configurations for seamless Langsmith integration.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Faster, Leaner Pretraining on Unsloth AI**: Unsloth AI has introduced capabilities to [continually pretrain](https://github.com/unslothai/unsloth/releases/tag/June-2024) LLMs with double the speed and half the VRAM previously required by HF+FA2, as detailed on their [blog](https://unsloth.ai/blog/contpretraining).

- **No Medusa Support Yet for Unsloth**: Engineers confirmed that Unsloth does not support fine-tuning using Medusa, based on a provided GitHub [link](https://github.com/FasterDecoding/Medusa), but it offers improved unsloth updates like disk offloading for lm_head/embed_tokens and automatic tokenizer fixes.

- **VRAM Management Techniques in Discussion**: Techniques for managing VRAM, including the use of cosine learning rates and selective offloading, were shared, noting the optimization potential for H100 GPUs and strategies to free memory for running multiple models via `del` commands.

- **Challenges in Multi-Nodal Implementation**: While **multi-GPU support** is active, a slight delay is anticipated for multinodal support implementation, which is essential for projects like 70B fine-tuning. Meanwhile, VRAM-saving alternatives like LoRA adapter use during fine-tuning were also touched upon.

- **Open Source TTS Models Scouted for Side Projects**: A member's request for "good OS TTS models" for a waifu companion app/RPG yielded the recommendation of the "xttsv2 -rvc pipeline," demonstrating active collaboration in Open Source resources among engineers.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Hits a Snag**: Users reported **downtime** and frustrations with model selection in Perplexity AI, with comments on extensive waiting periods and a quirky interface where requests for image generation are met with written descriptions instead of actual graphics.
- **AI Model Smackdown**: Debates compared **ChatGPT-4o** with **Claude 3**, noting Perplexity's unique approach of using internal search indexes, and shared resource links including [presentation tips](https://youtu.be/wjZofJX0v4M?feature=shared) and an [overview of Perplexity's search functionality](https://www.perplexity.ai/search/how-does-perplexity-Qm71LBYBSkOApKNFCISS0Q).
- **Searching Beyond SEO**: Engaging in the backend processes, discussions pointed out that Perplexity AI differentiates itself by not relying on third-party services for crawling and indexing, leading to higher-quality search results that are less manipulated by SEO tactics.
- **Diving Into Outages**: An article [analyzing a major outage](https://www.perplexity.ai/page/Major-Outage-of-DCcT_vXARMmWZl8KCWB8Jg) of DCcT was shared, providing insights into the technical issues faced by Perplexity AI.
- **Knowledge Expansion Through Shared Links**: Users enhanced discussions by referencing Perplexity AI search results on various topics, including articles on [dailyfocus](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg), [Bitcoin](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw), and shared reminders concerning the necessity of making threads shareable, with guidelines attached.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Open Office Hours and Interview Prep**: Engineers can join **vLLM and Neural Magic open office hours** on optimized LLM inference and enterprise ML, scheduled for [June 5](https://neuralmagic.com/community-office-hours/) and [June 20, 2024](https://neuralmagic.com/community-office-hours/). For performance engineer interview prep, a curated list of questions and resources is provided by [awesomeMLSys](https://github.com/cuda-mode/awesomeMLSys) on GitHub.

- **Triton Kernel PTX Access and GitHub Discussions**: Queries about extracting **PTX code from Triton kernels** led users to a useful [GitHub issue](https://github.com/triton-lang/triton/issues/3726) discussing the procedure. The user corrected their initial search location to `~/triton/.cache` for the PTX code.

- **Cracking the CUDA Stream Conundrum**: AI Engineers discuss using **named streams** in CUDA for better performance and shared a [pull request](https://github.com/karpathy/llm.c/pull/552) to mainstream operations. Efforts to fix a **PyTorch DDP loss computation bug** have culminated in a successful [PR](https://github.com/karpathy/llm.c/pull/551).

- **OOM Woes and Quantization Quirks in Large Model Evaluation**: Out-of-memory issues plague large model evaluations with **torchao APIs**, as seen in a [GitHub pull request](https://github.com/pytorch/ao/pull/328). AI Engineers recommend loading models on CPUs before quantization and adjusting for large vocab sizes.

- **Sparse Matrix Semantics and Sparsity in AI**: Clarifications on sparse matrices led to a sharing of a [Wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix) definition and a PyTorch [README](https://github.com/pytorch/ao/tree/main/torchao/sparsity). Moreover, a comprehensive [arXiv survey paper](https://arxiv.org/abs/2102.00554) summarizing over 300 papers on the utilization of sparsity in deep learning was circulated for better understanding and implementations.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FineWeb Unwraps LLM Performance Insights**: The FineWeb technical report details processing decisions and introduces the FineWeb-Edu dataset aimed at enhancing education-focused content and understanding high-performance LLMs like Llama3 and GPT-4. [FineWeb technical report](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is now available.

- **Browser-Based AI with Transformers.js in Firefox**: Firefox 130 update will include Transformers.js for on-device AI, with initial features targeting automatic alt-text generation for images to improve accessibility. Details can be found in this [announcement](https://x.com/xenovacom/status/1797285648572821840).

- **Nvidia NIM Accelerates Model Deployment**: Nvidia NIM launches on Hugging Face Inference Endpoints, providing easy 1-click deployment for models like Llama 3 8B and 70B on cloud platforms. Reference for deployment can be found [here](https://x.com/_philschmid/status/1797713003778883858).

- **Hugging Face and Wikimedia Team Up for ML Progress**: The collaboration leverages Wikimedia's datasets to further machine learning, underscoring the importance of community consent. The initiative details are explained [here](https://huggingface.co/blog/frimelle/wikipedias-treasure-trove-ml-data).

- **Diving Deep into Security and Ethics of AI**: The revelation of a security breach at Hugging Face prompted discussions on the ethical implications and safety of internet-based data storage, with a focus on maintaining respectful community engagements.

- **Crossing Technological Barriers**: The introduction of diffusion-based language modeling strategies mirrors principles used in image generation models, suggesting novel ways to handle textual "noise."

- **AI Tools for Climate-Conscious Investing**: An AI tool to identify climate-focused investment opportunities and calculate carbon footprint has been developed, utilizing models like `climatebert/tcfd_recommendation`, showcasing AI's potential in sustainable finance. Explore the AI tool [here](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor).

- **Knowledge Sharing in the AI Community**: A variety of AI-related projects and discussions span topics like improved logo detection, Apache Airflow setup on Windows, valuable LLM resources, and advanced German audio datasets for language model training, adding diversity to the knowledge pool.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Troubleshooting Model Loading in LM Studio**: Users faced issues with model loading due to insufficient VRAM; the proposed workaround is to disable GPU offloading. A specific case highlighted problems with loading **Llama70b** which was not saved as a GGUF file, and a sym link option or file conversion was recommended.

**Discussions Highlight Model Performance and Compatibility**: The **Command R** model showed suboptimal performance when offloaded to Metal, and for text enhancement, no specific model was recommended, although one should look at 13B models on the leaderboard. Additionally, difficulties with **SMAUG's BPE tokenizer** were reported with **Llama 3** version 0.2.24.

**Chatter About Workstation GPUs and Operating Systems**: The [ASRock Radeon RX 7900 XTX & 7900 XT Workstation GPUs](https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/) sparked interest, especially due to their AI setup-oriented design. There were mixed sentiments about Linux's user-friendliness and discussions about switching to Linux due to Windows' Recall feature prompting privacy concerns.

**Feedback for Bug in LM Studio**: A bug in **LM Studio v0.2.24** was pointed out, involving extra escape characters in preset configurations such as `"input_suffix": "\\n\\nAssistant: "`.

**Privacy and Security**: Privacy concerns were raised related to Windows' Recall feature potentially creating security vulnerabilities by amassing sensitive data. In a lighter tone, anecdotes of IT support challenges—including a computer tainted with the odor of cat urine—brought humor to the discussions on tech support woes.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Hackers Hit AI Services Hard**: The **ChatGPT, Claude, Gemini, Perplexity**, and **Copilot** services experienced outages due to a DDoS attack by **Anonymous Sudan**. The incident revealed vulnerabilities beyond typical cloud server expectations.

- **Comparing AI Subscriptions**: AI engineers debated the practicality of AI subscriptions, contrasting services like **GPT** and **Character AI** for tasks such as book summarizing and content creation.

- **Math's Got AIs Stumped**: Engineers observed persistent weaknesses in AI language models like **GPT** when tackling mathematical problems, highlighting inaccuracies and logical oversight in calculations.

- **AI Gets Personal and Practical**: Discussions showcased real-world integrations of AI, such as interfacing **ChatGPT** with home automation systems, highlighting both benefits and limitations in practical scenarios.

- **Using GPT-4 Vision with Google Sheets**: A query was raised about implementing **GPT-4 vision** to analyze and describe images in **Google Sheets**, suggesting an interest in expanding AI utility into spreadsheet tasks. 




---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Audio Open Serenades the Scene**: Stability.ai has launched **Stable Audio Open**, an open-source model to generate short audio pieces, including sound effects and production elements from text prompts. The model supports producing audio clips lasting up to 47 seconds, emphasizes innovation for sound designers and musicians, and is open for local fine-tuning; more details can be found [here](https://stability.ai/news/introducing-stable-audio-open).

- **WebUI Wonders: A Web of Possibilities for Stable Diffusion**: Community members engaged in a lively comparison between **A1111** and **InvokeAI** WebUIs for Stable Diffusion with a nod to A1111’s user-friendliness and InvokeAI’s unique "regional prompting" feature, which can be explored [on GitHub](https://github.com/invoke-ai/InvokeAI).

- **Tuning in on Training**: A technical clarification around using **regularization images** was sought, with members debating if these images could replace captions in training processes. Meanwhile, an emerging curiosity about **Stable Audio Tools** and its utility, including possible Google Colab use and commercial permissions, was evident, referencing their GitHub [repository](https://github.com/Stability-AI/stable-audio-tools).

- **UI Flex to the Max**: **ComfyUI** was recommended for its adaptability in image generation tasks, despite a challenging learning curve as illustrated by a member: "you can generate with cascade or sigma then refine it with sdxl...".

- **Bootstrapping Beginners**: New users were directed towards substantial community-curated resources like tutorials to learn Stable Diffusion, including a comprehensive guide from [Sebastian Kamph on YouTube](https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q) for getting started with A1111.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Finds Its Aesthetic Sense**: Discussion surged around implementing AI to control patterns and colors on display walls, which could potentially result in personalized art or branded decor. Questions about whether this could evolve towards a form of AI-driven interior design were raised.

- **Rethinking RLCD Buzz**: Scrutiny of RLCD technology's marketing led to conversations about its actual innovative aspects, with comparisons drawn to Samsung's QD-OLED displays. Skepticism persists regarding whether newer models significantly surpass existing transflective screen technology.

- **AGI Developments on the Horizon**: Growing investments in AGI were spotlighted, referencing a [blog](https://situational-awareness.ai/) that predicts substantial advancements in AGI capabilities by 2025/26, stirring dialogue on the expanding gap between leading labs and broader industry implications.

- **Balancing IQ and Agency**: Debates unfolded on the merits of IQ tests in hiring within open-source communities, juxtaposing them against "high agency" traits. Discussions emphasized the latter's superiority in contributing to success, given its ties to initiative, pattern recognition, and long-term vision.

- **Breaking Down Deep Learning's Limits**: Shared literature expanded upon deep learning's struggles with complex reasoning, whether it be transformers or SSMs. The community digested papers on diffusing "chain-of-thought" strategies into models and methods like SRPO, which seeks to robustify RLHF.

- **Open Implementations Spark Enthusiasm**: NVIDIA's public exposure of the RETRO model within [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro) sparked conversations about the democratization of AI research and the potential for wider accessibility of cutting-edge models.

- **Lm-evaluation-harness Troubleshooting**: A user faced difficulties getting the desired output from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb), leading to a consensus that results might be hiding away in the tmp folder. There's an appetite for guidance on implementing loglikelihood metrics specifically for the LLaMA 3 8B instruct model.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GLM-4 Breaking Language Barriers**: The introduction of [GLM-4](https://x.com/ChatGLM/status/1798292207574901012) brings support for 26 languages with capabilities extending to code execution and long-text reasoning. The open-source community can find the repository and contribute to its development on [GitHub](https://github.com/THUDM/GLM-4).

- **Exploring Nomic-Embed-Vision's Superiority**: The community is discussing the advancements of [Nomic-Embed-Vision](https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg), which excels over models like OpenAI CLIP in creating a unified embedding space for image and text. For those interested, both the weights and code are available for experimentation.

- **Contrastive Learning Loss Insights Shared**: A recently published paper introduces a novel contrastive learning objective known as the [Decoupled Hyperspherical Energy Loss (DHEL)](https://arxiv.org/abs/2405.18045), and a related GitHub repo that compares different InfoNCE type losses is [available here](https://github.com/viig99/ContrastiveLearningLossComparison). These resources could significantly benefit researchers in the deep learning community.

- **Microsoft's Idea Appropriation Discussion**: Concerns about Microsoft allegedly appropriating ideas without attribution came to light with a related [arXiv paper](https://arxiv.org/pdf/2405.19888) serving as a discussion point for the unintended open-sourcing of concepts.

- **Testing and Utilization of AI Models and Datasets**: Discussions around testing the Phi-3 Vision 128k-Instruct model on [NVIDIA NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct) and employing the [Openbmb's RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2) for crafting applications are ongoing. Members are encouraged to participate and provide feedback on model performance and dataset utility.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GraphRAG Construction Choices Discussed**: Members debated whether to build **GraphRAG** by manually defining the graph for full control or using an LLM for automatization; each approach impacts effort and data mapping effectiveness. An enterprise **RAG workshop** was available, exploring **Bedrock** models and agentic design patterns, while **Prometheus-2** surfaced as an alternative to GPT-4 for evaluating RAG applications due to its open-source nature.

- **Innovations in Metadata Extraction**: A new **Metadata Extractor** module and tutorial were introduced to help clarify long text segments, and questions around storing `DocumentSummaryIndex` in the **Chroma Database** led to a clear answer: Chroma cannot be used in this context.

- **Practical Solutions for Retrieval and Indexing**: They resolved a pertinent bug with **Neo4j** integration for query engines by merging a related [pull request](https://github.com/run-llama/llama_index/pull/13938), and shared methods to fine-tune the **“intfloat/multilingual-e5-large”** embedding model for e-commerce applications. A single **QueryEngineTool** proved capable of efficiently managing multiple PDFs, dismissing concerns about their cumulative operability.

- **Addressing Query Precision Issues**: One user's struggle with irrelevant material in a vectorstore's top responses prompted a suggestion to filter results by score, ensuring higher relevance and precision in retrieval outcomes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**ChatGPT 4 Adds Acting Chops**: OpenAI's ChatGPT 4 introduces impressive new *voice generation features* as seen in a [shared video](https://x.com/ai_for_success/status/1798046241307459673), stirring excitement with its ability to craft unique character voices.

**DALLE3's Diminishing Returns**: Users express concerns over a noticeable degradation in the quality of DALLE3 outputs, with disappointments echoed for both traditional usage and API integrations.

**Debating the Ethics of AI Monetization**: Recent discussions reveal a palpable frustration within the community over non-commercial licenses for AI models, criticizing motivations centered around financial gain and the extensive resources required for training models such as T5.

**LLMs Lose their Logic**: A new Open-Sci collective paper exposes the "dramatic breakdown" in reasoning exhibited by large language models, available for review [here](https://arxiv.org/abs/2406.02061) with accompanying [codebase](https://github.com/LAION-AI/AIW) and [project homepage](https://marianna13.github.io/aiw/).

**WebSocket Whims**: An issue with WebSockets in the *WhisperSpeech* service within **whisperfusion pipeline** prompted a detailed inquiry on [StackOverflow](https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio), hoping for a resolution to unexpected closures.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Rust Rises, Mojo Eyes New Heights**: A member praised a [YouTube tutorial](https://youtu.be/O4sVw4YQB24) highlighting the safety of Rust in systems development through FFI encapsulation, evidencing the engineering community's interest in secure and efficient systems programming.

**Transitional Tips for Python Devs**: A Python to Mojo [transition guide](https://www.youtube.com/watch?v=9ag0fPMmYPQ) on YouTube was lauded for compiling essential low-level computer science knowledge beneficial for non-CS engineers moving to Mojo.

**Mojo's Enumeration Alternatives**: While Mojo currently lacks `Enum` types, the conversation turned to its accommodation of [`Variants`](https://docs.modular.com/mojo/stdlib/utils/variant/Variant) with a nod towards the ongoing [GitHub discussion](https://github.com/modularml/mojo/issues/43) for those interested in potential developments.

**Nightly Updates Stir Commotion**: A new release of the Mojo compiler (`2024.6.512`) was announced, along with advice on managing versions in VSCode, while challenges were addressed in adapting to changes like `Coroutine.__await__` becoming consuming, as shown in the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

**Encryption Entreats Extension**: Capturing the intersection of security and programming, a user emphasized the urgency for a cryptography library in Mojo, suggesting the feature would be "fire" and underscoring the need to build robustness into the language's capabilities.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Investors Are All In on Robotics AI**: Investors are on the lookout for the **ChatGPT equivalent in robotics AI**, as they are eager to back companies with strong foundation models in robotics minus the hazards associated with hardware design, according to a [Substack article](https://www.newcomer.co/p/why-investors-cant-get-enough-of).

- **National Security and AI Trade Secrets**: The tech community debated the firing of an individual for leaks, with focus on the underappreciated role of **trade secrets in AI national security**. There's concern about overconfidence from labs like **OpenAI and Anthropic** on attaining researcher-level AI within 3 to 5 years, with some suggesting misaligned incentives and flawed extrapolations.

- **Toward AI That Aces PhDs?**: **Microsoft CTO Kevin Scott** predicts upcoming AI models may soon be capable of passing PhD qualifying exams, comparing today's models like **GPT-4** to ones that can tackle high school AP exams. Ph.D. exams' difficulty, notably at Berkeley where a 75% failure rate in prelims was observed, was also a topic of discussion, showcasing the challenge such AI models would face.

- **Paying for Problem-Solving**: An open issue in [rewardbench.py](https://github.com/allenai/reward-bench/issues/137) creates discrepancies in results due to varying batch sizes; Nathan Lambert offers a $25 bounty for a resolution. Additionally, **AutoModelForSequenceClassification** was termed "kind of cursed", indicating improvements might be possible with adjustments.

- **AGI Talk Yields Mixed Reactions**: Conversations reveal that the community is evenly split between being annoyed at both overly-optimistic **AGI enthusiasts** and the gloom-spreading **doomers**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Will Cohere's API Remain Free?**: Members are buzzing with speculations that **Cohere's** free API might be discontinued, urging others to seek official confirmation and disregarding unverified rumors.

**Bringing Order to Multi-User Bot Chats**: Engineers discussed the challenges of engaging **Language Models (LLMs)** in multi-user chat threads, suggesting that tagging messages with usernames could improve clarity.

**Hunting for the Ultimate Chat Component**: A community member inquired about a React-based Chat component; they were pointed to the **[Cohere Toolkit](https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io)**, which isn't built on React but may contain elements such as the chatbox written in it.

**React Components and Cohere Synergy**: Though **Cohere Toolkit** lacks React components, the open-source tool positions itself as a useful resource for implementing **RAG applications**, potentially compatible with React implementations.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Bug Hunt in Memory Lane**: Users reported **Out of Memory (OOM)** errors when running a target module on 2xT4 16GB GPUs, alongside anomalous loss:0.0 readings, which could suggest a critical issue in parameter configuration or resource allocation.

**Data Feast for Hungry Models**: The [HuggingFace FineWeb datasets](https://huggingface.co/HuggingFaceFW), a sizeable collection sourced from **CommonCrawl** with 15 trillion tokens, is making waves for its potential to lower entry barriers for training large models, though concerns about computational and financial resources required to utilize it fully have been raised.

**Deepspeed Dominates Model Training Chatter**: Engineering discussions revealed a preference for using the **command line** for running **Deepspeed** tasks including successful fine-tuning of the Llama3 model using **Deepspeed zero2** and selected **Qlora** over Lora for fine-tuning.

**Seeking Speedy Solutions**: A member vented frustration over **Runpod's** slow boot times, specifically that booting a 14 billion parameter model takes about a minute, impacting cost-effectiveness; questions were raised about alternative serverless providers with faster model loading capabilities.

**Model Mingle and Muddle**: While there is clear enthusiasm over the **GLM-4 9B** model, concrete feedback within the community on its performance and use cases seems scarce, suggesting either a novelty of deployment or a gap in shared user experiences.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Real-Time AI Revolution**: [LiveKit](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww) secures a **$22.5M Series A funding** to pioneer a transport layer for AI, citing the capabilities of GPT-4 as a catalyst for investor interest.

- **Multimodal AI Grabs the Spotlight**: [Twelve Labs](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html) has acquired **$50M in Series A funding** and debuted Marengo 2.6 with aspirations to refine multimodal foundation models.

- **Forecasting the Art of Precision**: Microsoft Research unveils **Aurora**, aiming to drastically improve the accuracy of weather forecasting through leveraging advancements in AI foundation models.

- **Transparency in AI Alignment Questioned**: Teknium scrutinizes OpenAI for not being open about alignment rewards and moderation classifiers; the discussion uncovers that reward models are typically incorporated within the architecture of large language models (LLMs) itself.

- **Content Management Gets an AI Boost**: [Storyblok](https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww) clinches **$80M in Series C funding** to evolve an AI-powered content platform, initiating the public beta of its new Ideation Room.

- **Anthropic Dives into Monosemanticity**: Anthropic scheduled an insightful talk on **Scaling Monosemanticity**, promising advancements in understanding the links between monosemanticity and model scaling. Details and registration were provided for the [event](https://lu.ma/p5ctl2u6).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Skill Persistence Poses Problems**: Discussions revealed that **OpenInterpreter** lacks the ability to retain skills across sessions despite users trying to "tell OI to create a new skill." To circumvent this, the advice is to save and store scripts as a workaround.

- **RAG Under The Microscope**: There's skepticism regarding **Retrieval-Augmented Generation (RAG)** with a preference for conventional embedding/vector databases cited for their reliability, albeit at a higher token cost.

- **Data Privacy Takes the Spotlight**: Concerns over data privacy with OpenAI was assuaged with reassurances that communications with OpenAI's API remain confidential, while running a local model was proposed for additional security.

- **Cross-Model Compatibility Queries**: Query into integrating the **O1 dev preview** with other large language models like **Anthropic** raised compatibility issues, particularly the necessity of a vision model and potential infinite loops on some operating systems.

- **Voice Assistant for Devs**: A link to a GitHub project for a **Terminal Voice Assistant** sparked interest in whether similar functionalities could be implemented in **01**, pointing to potential development tools for engineers.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz Throws Down the Tqdm Replacement Gauntlet**: [George Hotz](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363) offered **$200** for a minimalist tqdm replacement, sparking a flurry of activity and a submitted PR from Trirac, albeit with a note about its suboptimal it/s rate at higher speeds.
- **Tinygrad's Missing Stats Mystery**: Hotz inquired about why the [stats.tinygrad.org](https://stats.tinygrad.org) site is currently a **404 error**, sparking discussion on the site's accessibility.
- **Invitation to Improve Tinygrad Docs**: Updates to Tinygrad's documentation have been announced, including new sections on training and a library diagram, and the community has been solicited for further content ideas ([George Hotz](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363)).
- **Tinygrad: Seeking Specs before the Sprint**: The bounties offered by Hotz aim to draft the **Tinygrad spec**, with a promise it could be reimplemented in roughly **two months** when finalized, also serving as an employee screening process.
- **Deciphering CUDA-to-Python**: Discussion centered on the complexity of connecting CUDA debug output to Python code, a key feature for Tinygrad's v1.0, with existing PRs yet to be merged ([George Hotz](https://discord.com/channels/1068976834382925865/1070745817025106080/1247984470146027642)).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Outdated Docs Cause Commotion**: **LangChain and OpenAI** documentation woes have caught the attention of members noting significant discrepancies due to API updates. A suggestion pointed engineers towards the primary code stack itself for the most current insights.

**DB Wars: MongoDB vs. Chroma DB**: When an engineer pondered the use of MongoDB for vector storage, a clarification ensued about MongoDB's purpose for storing JSON rather than embeddings, directing the inquirer to MongoDB's assistance or ChatGPT.

**Verba: RAG Under the Microscope**: The community took an interest in [Verba](https://github.com/weaviate/Verba), a Weaviate-powered RAG chatbot, with a request for user experiences being aired, indicating an exploration into Weaviate's retrieval augmentation capabilities.

**SQL Agent Leaves Users Puzzled**: Issues surfaced with the SQL agent not delivering final answers, sparking a discussion on troubleshooting this cryptic behavior in an environment that detests non-performing components.

**Graph-Based Knowledge with LangChain**: An engineer showcased a [LangChain guide](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/) focused on constructing knowledge graphs from unstructured text, prompting inquiries on integrating `LLMGraphTransformer` with Ollama models, a nod to the constant pursuit of enhanced knowledge synthesis.

**VisualAgents Usher in Drag-and-Drop LLM Patterns**: A live demonstration via a [YouTube video](https://www.youtube.com/watch?v=IVFsANcfqaA) on using VisualAgents highlighted the creative process entailed in arranging agent flow patterns, reflecting a trend towards more intuitive interfaces in LLM chain management.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Rope Scaling Hits a Snag with OpenRouter**: Members highlighted issues with integrating **rope scaling** in **OpenRouter**, suggesting local deployment to sidestep GPU constraints.

- **Codestral Lags Behind on Code Specialization**: A recommendation was made against using **Codestral** for code specialization, with a nod toward more efficient models detailed in <#1230206720052297888>.

- **Identifying the Culprit Behind 502 Errors**: Engineers tackled 502 Bad Gateway errors with OpenRouter, tracing the problem to the format of `content` in `messages` rather than server capacity or request volume.

- **Eclectic Model Mix-Up During Downtime**: The **502 errors** were occurring while handling a variety of models from **Nous Research**, **Mistral**, **Cognitive Computations**, **Microsoft**, and **Meta-Llama**, with emphasis on the issue stemming from message content formatting.

- **Seek Alternatives for Greater Code Efficiency**: Engineers looking for effective code specialization are advised to consider more performance-oriented alternatives to **Codestral**, with hints to check a specific model mentioned in the guild.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Circle Your Calendars for AI Safety**: The **Human Feedback Foundation** event is set for June 11th; tickets can be grabbed from [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator). The focus of the event will encompass AI governance and safety, enriched through a collaborative open-source environment.
- **Gleaning Insights from AI Experts**: Check out the [Human Feedback Foundation's YouTube channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg) for insights from academia and industry leaders from UofT, Stanford, and OpenAI, focusing on the integration of human feedback into AI development.
- **LLM Reading Group Discord Access Restricted**: There was a request for a separate Discord for the LLM Reading Group, but direct invitations were hampered due to privacy settings, implying a need for alternate access arrangements for interested individuals.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Vector Tech Marches Forward with RISC-V**: [RISC-V Vector Processing](https://www.youtube.com/watch?v=Ozj_xU0rSyY) reaches a significant milestone with the **1.0 RISC-V Vector Specification** now ratified. The linked video delves into the early silicon implementations, suggesting ample opportunities for innovation in CPU designs.

- **AI's Existential Threat Spotlighted**: The [Right to Warn AI](https://righttowarn.ai) project raises alarm about the potential existential threats posed by AI technologies, promoting the need for oversight much beyond corporate governance. It raises concern over AI-related risks like inequality, misinformation, and potential human extinction.





---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Exploring "Sauerkraut Gemma" Prospects**: Interest in replicating the **PaliGemma** model for German, tentatively named *"Sauerkraut Gemma"*, was expressed, with the idea of simply replacing Gemma's base for adaptation.
- **PaliGemma Model As a Template**: Referencing the [PaliGemma-3B-Chat-v0.2](https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2) model, a member proposed a strategy of *"freezing the vision and training the chat"*, post-dataset translation for the development of a German counterpart.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **AI Learning Hub Unveiled**: The [GenAI Handbook](http://genai-handbook.github.io/), curated by William Brown, was highlighted as a significant resource for AI engineers seeking a comprehensive understanding of modern AI systems formatted in a user-friendly, textbook-style guide.



---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1247645798016225362)** (46 messages🔥): 

- **Discussion on Slide/Link Access and COVID Delay**: A member requested access to all slides and links from workshops, while another user mentioned missing recent presentations due to COVID. Recommendations included filling out a [Modal credits form](https://bit.ly/modal-credits) to receive $500 in bonus credits from Modal.

- **Gemma Copycat Alert**: A user exposed that the repository [gemma-2B-10M](https://github.com/mustafaaljadery/gemma-2B-10M) was copied from [InfiniTransformer](https://github.com/Beomi/InfiniTransformer), only with minor changes to comments and formatting.

- **Best Code Model for Fine-Tuning**: The interest was shown in finding the best code model for fine-tuning. Lack of explicit good resources or models was noted in the discussion.

- **Synthetic Data Generation**: Users discussed difficulties with tools for synthetic data generation, citing [distilabel](https://github.com/argilla-io/distilabel/tree/main) and Python coding as options. A user pointed out issues with data quality generated by the evol pipeline using distilabel.

- **Mistral Fine-Tuning Hackathon**: Announcements of the [Mistral Fine-Tuning Hackathon](https://mistral.ai/news/2024-ft-hackathon/) excited users, who expressed interest in forming groups to compete. Another related event was mentioned via [X (Twitter) link](https://x.com/dchaplot/status/1798368883172421765?t=BmEAh2YFNBTIKFjRvDwAeQ&s=19).
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1247922676547325982)** (1 messages): 

- **Replicate personas of family members with AI**: A member shared an interesting use case example about replicating personas of family members, particularly ageing parents, to preserve their texts, notes, opinions, and voice. They suggested using a mixture of approaches: **fine-tuning** to capture their mannerisms, **RAG** to contextualize new or fresh topics, and **prompt engineering** for conditioning the response.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1247793111917138023)** (1 messages): 

~~~html
- **erniesg discusses Hainan departure and VPN setup**: *"im actually leaving hainan on 7 june"* and adds a light-hearted comment about ensuring VPN access for coding. This suggests ongoing preparation for remote work or coding sessions.
~~~
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1247630777450106921)** (30 messages🔥): 

- **Modal credits confusion sparks lively discussion**: Multiple users voiced concerns about not receiving the promised additional $500 in credits. Charles clarified how the credits are distributed and offered personalized support via email for unresolved issues.

- **Outdated documentation causes hiccups**: [A link to the outdated docs](https://modal.com/docs/examples/llm-finetuning) prompted users to note discrepancies with current practices. Charles acknowledged the issue and mentioned plans to update the docs.

- **Subprocesses as a workaround for non-Python tasks**: Users explored running shell scripts with GPUs, noting that while Modal isn't designed natively for this, using subprocesses is a viable workaround. Hamel emphasized this, suggesting calling subprocess or similar commands.

- **Deploying Python Shiny apps hits snags**: Iain faced challenges deploying a Shiny app, pointing to a [GitHub repo](https://github.com/iainmwallace/modal_shiny) and discussing a streamlit example. Charles advised raising the issue in Modal Slack for better troubleshooting.

- **Modal's privacy policy question gets Google redirect**: When queried about Modal's privacy policy, Hamel simply directed to a [Google search for the privacy policy](https://www.google.com/search?q=privacy+policy+modal+labs), indicating it can be found online.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1247985108850442292)** (1 messages): 

- **Ultimate LLM Resource Guide**: A member shared their compilation of favorite LLM explainers, covering topics such as vLLM, SSMs, DPO, and QLoRA. For more information, they provided a link to [the guide](http://genai-handbook.github.io).
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1247634880309891082)** (7 messages): 

- **Credits confusion clarified by startup**: A member clarified that the startup is offering only one $200 credit and apologized for any confusion about extra credits being added accidentally. They mentioned being a bootstrapped startup without deep pockets.

- **CUDA library upgrade issue**: A user faced issues upgrading the CUDA library in Jarvis Labs containers when installing the `xformers` PIP module. The error indicated a **mismatch between detected CUDA version (11.8)** and the version used to compile PyTorch (12.1).

- **Initial credits reported received**: Multiple users confirmed they received the initial 200 credits but did not get any additional credits. They expressed satisfaction with the platform, noting its convenience and good service.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1247627395981377657)** (35 messages🔥): 

- **Confusion Over Credits and Forms**: Several users expressed concern about not receiving credits despite filling out forms on time. [tddammo](https://example.com) reassured them that a second form would be released soon for any missed credits.

- **Hilarious HTML Edits**: *ayhanfuat's* playful comment about using credits for training GPT-6 turned out to be a clever HTML edit, causing a brief scare before he clarified the joke. The exchange relieved tensions and added some humor to the conversation.

- **GPU Wishlist**: *osanseviero* and *charles_irl* humorously noted the rising demand for GPUs, indicating an ongoing need for more computing resources among the community. *charles_irl* quipped, "let a hundred GPUs bloom."

- **Acknowledgements and Resolutions**: Throughout the thread, [tddammo](https://example.com) diligently addressed users' concerns about missing credits. Several users thanked him for sorting out their issues promptly.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1247629712176582837)** (16 messages🔥): 

- **Redeem Replicate Credits Without Org Setup**: Users discussed that **you don't need to create an organization** to redeem your Replicate credits. However, for team collaboration, it's recommended to set up a GitHub org first.
- **Credit Redemption Instructions and Issues**: Users shared experiences about receiving **Replicate credits redemption emails** and the steps involved in claiming them. Some users encountered issues where redeemed credits were not reflecting in their billing; these users were advised to DM their email and username for assistance.
- **Credits Expiry Timeline**: A member inquired about the **expiry date of Replicate credits**, and it was clarified that **the credits are good for 24 months**. This provides ample time for users to use their credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693)** (21 messages🔥): 

- **LangSmith credits confusion reigns**: Multiple users reported not receiving their $250 LangSmith credits despite setting up billing as instructed. Some users cited confusion over different types of credits such as initial sign-up credits and beta credits being mixed up.

- **Pinecone namespace retrieval issues**: A user sought assistance with improving retrieval performance in a Pinecone namespace containing multiple documents by creating multiple retrievers to filter by document name. The user struggled with configuring an LCEL chain for this purpose.

- **LangSmith missing inputs mystery solved**: A user debugging the `@traceable` decorator found that their LLM call outputs were logged but not the inputs. The issue was resolved after realizing that the decorator logs the function's arguments as inputs, and therefore, having arguments in the function definition allowed the inputs to be captured. [LangSmith docs](https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#use-traceable--traceable) were referenced.

- **Mastering LLMs Course Credit**: One user mentioned successfully seeing the credits from the course labeled as 'Mastering LLMs Course Credit'. There was some confusion about the availability of Beta Credit and how it could be received.

- **General dissatisfaction and a call for help**: Several users, including one mentioning Hugo, expressed frustration and asked for assistance regarding the lack of received credits despite proper setup. This issue prompted several requests for clarification and help in resolving the credit allocation problems.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1247835046606536765)** (2 messages): 

- **Valgrind memory checker still relevant**: A member queried about the current utility of **Valgrind** for checking memory usage and leaks, hinting at its past importance. Another member was addressed directly for their input on the matter.

- **Support requested for Hamel Husain's talk**: A member shared a [link to Hamel Husain's Twitter post](https://x.com/hamelhusain/status/1798353336145674483) and urged others to help the talk gain more audience. The message included a salute emoji to emphasize the request for support.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1247625969280221184)** (174 messages🔥🔥): 

- **Scale and GPUs can't defy physics**: One user joked about scaling compute instances expecting magic to happen, while another humorously noted, "We can't bribe physics!" leading to a suggestion to "just buy more GPUs."
- **Modal steals the show with credits**: Multiple users highlighted Modal's impressive credit offerings with statements like "Modal is just too powerful." This sparked discussions about running any task on Modal to obtain extra credits and [Modal's example docs](https://modal.com/docs/examples).
- **A collection of valuable links**: Users shared various useful resources such as Axolotl for merging LoRA to base, Dan's [Huggingface repo](https://huggingface.co/dansbecker/conference-demo), [Predibase's index](https://predibase.com/fine-tuning-index), and TGAddair on [LinkedIn](https://www.linkedin.com/in/travisaddair/).
- **Discussions on inference optimization and quantization**: There were in-depth discussions about optimizing CPU inference, experiences with llama.cpp, and explanations of why quantized methods (like QLoRA) can be slower during compute-bound tasks.
- **Office hour announcements and feedback**: TGAddair announced [office hours for Predibase](https://discord.com/channels/1238365980128706560/1242223495673286737/1247637288146698271), with users leaving feedback and asking questions about fine-tuning strategies and performance impacts.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1248012391250268362)** (1 messages): 

- **LangChain Link Fails**: A user shared a [link to a LangChain multi-modal RAG notebook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb) and asked if it didn’t work. This seems to indicate some troubleshooting or functionality issue with the provided resource.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[yang_mistral_finetuning](https://discord.com/channels/1238365980128706560/1242224842053521459/1247934402676133939)** (108 messages🔥🔥): 

- **Mistral's API launch aligns perfectly with hackathon**: Members excitedly discussed the [Mistral AI customization announcement](https://mistral.ai/news/customization/), noting the coincidental timing with a hackathon and presentation.
- **New Mistral projects and resources**: Numerous resources were shared, including Mistral's [fine-tuning tutorial](https://docs.mistral.ai/capabilities/finetuning/), the [fine-tuning repository](https://github.com/mistralai/mistral-finetune), and a relevant [YouTube demo](https://youtu.be/zXFxmI9f06M).
- **Discussions on Mistral's uniqueness and capabilities**: Members debated Mistral's vertical integration strategy, its API advantages over tools like Axolotl, and specifics like the importance of prompt templates and space handling.
- **Memory management and training issues**: There were questions and discussions on handling long training times on platforms like Colab, with suggestions to save checkpoints and insights on how Mistral's API mitigates memory errors through potential optimizations like quantization.
- **Interest in larger displays for productivity**: Side discussions included enthusiastic recommendations for massive monitors, specifically a 55-inch TV being used as a main monitor. The link was shared for those interested in upgrading their setups: [Samsung-GQ55QN95BAT](https://www.mediamarkt.de/de/product/_samsung-gq55qn95bat-neo-qled-tv-flat-55-zoll-138-cm-uhd-4k-smart-tv-2793513.html).
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1247715125423767583)** (1 messages): 

- **Fine-tuning workflow is simple but subpar**: A member remarked on the ease of using a Predibase example for fine-tuning, describing it as an "easy workflow". However, they noted disappointment with the current quality of the fine-tuning process, stating it "sucks" despite completing one full iteration.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1247688274047143986)** (21 messages🔥): 

- **Validate datasets with `val_set_size` tackled**: Members discusses if `val_set_size` parameter provides access to validation/test datasets or if `test_datasets` need to be explicitly specified.

- **Local Axolotl installation challenges resolved**: A user shared frustration over installing Axolotl locally following advice for a testing environment with a tiny model. They ended up successfully using [this alternative guide](https://www.superteams.ai/blog/a-definitive-guide-to-fine-tuning-llms-using-axolotl-and-llama-factory), which includes specific steps for CUDA and PyTorch setup.

- **Docker as a hassle-free Axolotl solution**: Users recommended using the official Docker images for Axolotl to avoid dependency issues and improve performance, referring others to [a guide on debugging with Docker and VSCode](https://openaccess-ai-collective.github.io/axolotl/docs/debugging.html#debugging-with-docker).

- **Trouble with CUDA versions in Axolotl installs**: Members reported varied success installing Axolotl on machines with different CUDA versions, noting that specific dependencies might need manual adjustments. The possibility of maintaining `requirements.txt` for each CUDA version was discussed.

- **Dual GPU stalling issues outlined**: For those experiencing dual GPU stalling, one member suggested setting `NCCL_P2P_DISABLE=1` as a solution, although it’s highlighted as less of an issue on professional or cloud-based machines.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1247652004562735165)** (11 messages🔥): 

- **cpu_ram_efficient_loading demystified**: The discussion clarified that `cpu_ram_efficient_loading` allows for sharded pieces of models on various GPUs, rather than the whole model on all GPUs. When set to `false`, only the first worker keeps the full model weights while others retain skeletons without weights, dispatching necessary weights to individual layers for **FSDP**.

- **Terminology clarification on 'process'**: There was a clarification that "process" refers to the GPU number, not partitions of the GPU. For example, process 1 corresponds to the second GPU when numbering starts at 0.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1247825230517501993)** (1 messages): 

- **Sanity Checks for YAMLs: Help Needed**: A user is seeking assistance in locating where exactly sanity checks on YAMLs run. They are also inquiring if anyone has built a user-friendly GUI for ADO config.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1247629487152169101)** (29 messages🔥): 

- **CUDA Book Recommendation Sparks Curiosity**: Members discussed recommended CUDA books, with one suggesting a well-reviewed book on [Amazon](https://a.co/d/1PbUJK7). Charles confirmed this was the book mentioned in a previous presentation.
  
- **Modal Setup and Usage Inquiries**: Users shared experiences and questions about modal, with imaure asking if `$500 credit` can be earned for quick setup. Charles confirmed, "you sure do! Allow up to a week for these credits to release 💚".

- **Sentence Transformer Embedding Models on Modal**: imaure inquired about hosting a sentence transformer embedding model on Modal, to which Charles confirmed its feasibility and shared an [example link](https://modal.com/blog/embedding-wikipedia).

- **Getting Started with Modal Examples**: Charles and andrewcka provided assistance on using basic modal examples for new users encountering issues. For instance, they directed a user to [Modal's hello world example](https://github.com/modal-labs/modal-examples/blob/main/01_getting_started/hello_world.py) and helped troubleshoot specific code errors.

- **Billing and Usage Clarifications**: Users sought clarification about billing, particularly on whether they would be billed for stopped apps. Charles reassured that stopped apps will not accrue charges and noted, "we're generally pretty forgiving with surprise bills."
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1247680107875663954)** (3 messages): 

- **Langsmith Setup Simplicity Shines**: No extra code is needed to use **Langsmith** with **Langchain**. Just set up environment variables, and it will automatically track everything.

- **Langsmith Credits Available**: Information about obtaining **Langsmith credits** can be found in this [Discord link](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693).

- **Langchain Retrieval Performance Woes**: A member is experiencing degraded retrieval performance in a Pinecone namespace with increasing document count. They are considering creating multiple retrievers filtered by document name but are struggling to create the LCEL chain to achieve this.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247632624261009548)** (51 messages🔥): 

- **Missed Deadline for Credits, Trying to Accommodate**: Even though the deadline to fill forms was May 29, **Danbecker** is trying to secure credits for those who missed it. He emphasized, *"I'm creating a list of people who didn't fill out forms in time, and I'm sending it to the platforms."*

- **List of Available Credits and Sources**: A user shared a detailed list of available credits from various platforms including $501 from **HuggingFace** and **Replicate**, $500 from **OpenAI** and **Modal** (with another $500 if used before June 10), among others. **Fireworks** was confirmed to offer $250.

- **Expiration and Account Setup Issues**: Users discussed the expiration dates for credits, like **Modal** credits expiring after a year and **OpenAI** credits after three months. They emphasized the importance of setting up accounts promptly to receive the credits.

- **Handling Incomplete Forms and Late Submissions**: **Danbecker** and others reiterated that platforms need accurate email addresses and account IDs to distribute credits. For example, errors like missing **OpenAI Org-ID** in forms cannot be remedied post-deadline.

- **Current Status and Updates on Credit Distribution**: Updates were given on the distribution statuses, confirming credits from **Langsmith** and **Fireworks**. Users were advised to create necessary platform accounts and follow up if they hadn't received their credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[strien_handlingdata](https://discord.com/channels/1238365980128706560/1243773476301443073/1247938813670457477)** (158 messages🔥🔥): 

- **Excitement for Synthetic Data Prep**: Multiple users expressed excitement for the talk on synthetic data preparation, calling it an "exciting" topic. One user noted a preference for "ML Librarian" as a dream job concept.

- **Links and Tools Shared**: Users shared various tools and links relevant to data handling, including [Lilac](https://www.lilacml.com/), Huggingface's [dataset security](https://huggingface.co/docs/hub/en/security-malware), and examples of structured text generation like [Outlines](https://github.com/outlines-dev/outlines).

- **Discussing Dataset Generation**: Participants discussed using **GPT-4** with human evaluations to create synthetic datasets, highlighting the high costs and potential benefits. One user mentioned using a combination of **DPO** and **PPO** approaches to improve dataset quality.

- **Knowledge Graph Enthusiasm**: Users expressed interest in building Knowledge Graphs for improved data structuring and retrieval, mentioning tools like [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/) and LangChain notebooks for enhancing RAG.

- **Crucial Resources on RLHF and Alternatives**: The conversation included references to [RLHF and its alternatives](https://argilla.io/blog/mantisnlp-rlhf-part-9) from Argilla's blog series, with users requesting further clarification on using Distilabel for generating synthetic chat datasets in JSONL format.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1247841028061397002)** (28 messages🔥): 

- **Account credit checks session: a community effort**: Multiple members reported not seeing allocated credits in their accounts despite completing the necessary forms. **Aravindputrevu** actively engaged with each user, requesting account IDs and following up to ensure the issues were resolved, resulting in many confirming the successful addition of credits: "all good, thanks 🙏".


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1247626352425697361)** (26 messages🔥): 

- **Braintrust onboarding page update**: Members discussed a new change that defaults users to the onboarding page if they have no projects. *"We just pushed a little change that defaults you to the onboarding page...hopefully that helps."*

- **Credit limitations clarified**: One user was unsure about where to see their credits, and it was clarified that there isn't a formal credit system displayed in the UI, providing members with *"unlimited access for 3 months."*

- **Functionality of Braintrust decorator**: A member inquired about why inputs to their LLM call weren't being captured by the Braintrust decorator. It was clarified that wrapping the `OpenAI` client with `wrap_openai` is necessary to log OpenAI calls, not using `@traced` which only logs the inputs/outputs of functions.

- **Explaining tracing methods**: There was a detailed explanation of three tracing methods: `wrap_openai`, `@traced`, and spans for logging specific information. *"Both `wrap_openai` and `@traced` are implemented in terms of spans."*

- **Integration of Braintrust into other platforms**: One member mentioned an attempt to integrate Braintrust into ZenML, indicating the complexity of wrapping around all possible functions. *"Going the OpenAI wrapper probably isn't possible unless we ask users to use some sort of helper function where that's implemented."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1247809404716449874)** (2 messages): 

- **International Greetings**: A user greeted members stating they are from Peniche, Portugal 🇵🇹. Another member responded with a greeting from an unidentified location marked with a black flag emoji 🏴.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1247649710236237977)** (8 messages🔥): 

- **Issue with Repeating Periods in Finetuned Models:** A user raised a concern about finetuned models generating repetitive periods or spaces at the end of their outputs and sought advice on how to mitigate this.

- **Community Support on Credits:** There was an issue with credits not appearing, which was swiftly resolved. It was suggested to contact support if the issue persisted.

- **Praise for Predibase's Methodology:** Users expressed appreciation for Predibase's approach of reusing base models and simplifying the input data format to just prompt-completion pairs. This simplification helps users avoid the overwhelming variety of options for prompt templates and system messages.

- **Validation of Prompt Templates:** While simplifying prompt templates is beneficial, a user emphasized the importance of allowing validation to ensure correct backend usage. This validation would help users identify and rule out template-related issues.

- **Epoch Flexibility and Data Filtering Suggestions:** A user recommended giving access to more epochs for training, such as running up to 7 epochs and being able to select the best-performing one to avoid overfitting. They also suggested a UI walkthrough for filtering out common data mistakes, similar to Cohere’s approach, to enhance the fine-tuning process.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1247643889603444776)** (5 messages): 

```html
- **Seasoned Developer Dives into LLMs**: A software developer with 17 years of experience shared their journey into learning about LLMs. They expressed excitement and sought advice on running their own models locally, mentioning potential fintech applications.

- **Fastbook Recommended for LLM Fundamentals**: A member recommended the [fast.ai free book on GitHub](https://github.com/fastai/fastbook) for an overview of deep learning fundamentals, especially for software engineers. They highlighted that the book has plenty of code and intuition with minimal math.

- **Community Learning for Engineers**: A user emphasized the importance of community for learning complex topics like LLMs. They shared their experience with a Romanian learning community [Baza7](https://new.baza7.ro/) that offers practical knowledge across various business functions.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1247790180866326580)** (3 messages): 

- **OpenPipe Credit Rollout in Progress**: A member inquired about the status of the **OpenPipe credit rollout**. Another member confirmed they had not received it yet, indicating it is *"the last one now, all others are good!"* while another confirmed the same status, calling it the *"last one that I haven't got yet."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247653281480839218)** (51 messages🔥): 

```html
- **Optimizing LLMs guide shared**: OpenAI's startup solutions team shared a new [guide on optimizing LLMs for accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy), focusing on prompt engineering, RAG, fine-tuning, and determining what is sufficient for production. A YouTube [DevDay talk](https://www.youtube.com/watch?v=ahnGLM-RC1Y) was also recommended for additional insights.
- **Challenges and requests in fine-tuning**: Users highlighted the helpfulness of the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning) and the need for improvements in the fine-tuning process. Multiple users requested features such as a retry button, an option not to shuffle the dataset, and solutions for better tool/function calling outputs via the fine-tuning API.
- **Credits and rate limits concerns**: Users discussed issues regarding the application and expiration of OpenAI credits. Some reported having to activate billing to receive credits and others highlighted the difficulty of utilizing $500 in credits within the given 3 months due to rate limits.
- **Rate limits and API spend confusion**: Users questioned whether credits count towards API spend necessary to increase rate limits and shared insights about potentially needing to make a small payment to unlock higher rate limits sooner. Discussions continued around the possibility of OpenAI addressing this concern for a more equitable solution.
- **Availability and functionality of GPT-4 models**: Users mentioned challenges accessing GPT-4 and GPT-4o models despite having credits and speculated that access might be unlocked only after the first paid invoice. Experiences shared indicated credits apply to the balance in the billing overview page.
```
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247643814202577027)** (417 messages🔥🔥🔥): 

- **Medusa fine-tuning unsupported on Unsloth**: Members discussed the possibility of fine-tuning using Medusa on Unsloth but confirmed that it’s not supported. A relevant link to Medusa was shared [here](https://github.com/FasterDecoding/Medusa).

- **Continued pretraining limitations and tips**: Members debated the feasibility of continued pretraining without optimizer states, sharing insights on mitigating catastrophic forgetting by mixing in old data. They also referenced using tools like redpajama for better results.

- **Memory management and optimizations**: Participants discussed VRAM usage spikes during training and strategies to manage memory more efficiently, such as using cosine learning rates and offloading specific tokens. A notable suggestion was potentially extending context length on H100 GPUs.

- **Colab limitations and alternative setups**: Colab's limitations for extensive GPU training, like fine-tuning LLaMA 70B, were highlighted. Suggestions included using other cloud services or setting up efficient Docker containers to manage training sessions.

- **New Features in Unsloth updates**: Recent updates in Unsloth, including offloading lm_head/embed_tokens to disk and automatic tokenizer fixes, were shared. Also, guidelines for continued pretraining were given, highlighting improvements in speed and VRAM usage.
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1247651391212621824)** (1 messages): 

- **Pretraining just got faster and leaner**: "You can now [continually pretrain](https://github.com/unslothai/unsloth/releases/tag/June-2024) LLMs 2x faster & use 50% less VRAM than HF+FA2." More details available on the [Unsloth AI Blog](https://unsloth.ai/blog/contpretraining).
- **Free Notebooks available for Mistral v0.3**: Access our [Continuous Pretraining notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) and [Text Completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) for hands-on experience.
- **Upcoming features announced**: Expect support for all models including stable diffusion, multimodal, Mixtral, 8-bit, and more. MultiGPU support is also in the pipeline.
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1247829744586850356)** (6 messages): 

- **Delay in Multinode Support Expected**: A member hinted that "multinode support will be a little complicated," suggesting that its implementation might face delays. This indicates possible challenges on the horizon for the project.

- **Seeking Open Source TTS Models**: A member inquired if anyone had found "good OS TTS models" with reference code for a side project involving a waifu companion app/RPG. Another member recommended the "xttsv2 -rvc pipeline," which was appreciated by the original inquirer.
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247655619499004069)** (220 messages🔥🔥): 

```html
- **Multi-GPU Support is Here, Multi-Node Coming Soon**: A user asked about multi-node training support for Unsloth, and was informed it's on the roadmap but not available yet. They expressed excitement about the potential for 70B finetuning with multi-GPU setups.
- **VLLM Server Setup Simplified**: A helpful discussion on setting up a VLLM server included commands and links to [installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation.html). The VLLM server can act as a drop-in replacement for the OpenAI API endpoint, useful for hosting fine-tuned LLMs locally.
- **Continued Pre-Training with High Loss Issue**: A user reported high initial loss when continuing pre-training a model, despite previous successful training with low loss. They shared detailed code snippets for loading and training models with specific configurations.
- **Fine-Tuning with LoRA Adapters**: Users discussed issues with loading and continuing fine-tuning using LoRA adapters. A working solution involves creating a new PEFT model and attaching existing adapters afterward, though the wiki method still seems problematic.
- **Handling GPU Memory for Multiple Models**: A user inquired about removing models from GPU memory efficiently to run training loops overnight. Another suggested using `del` to delete model and tokenizer objects to free up GPU memory without restarting the kernel.
```
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1247631953583411380)** (317 messages🔥🔥): 

- **Perplexity AI struggling with downtime**: Multiple users reported issues with Perplexity AI being down or having trouble with model selection. A user lamented, "I've been waiting for 8 hours. I'm going crazy."
- **Model comparison and features**: There was a detailed discussion comparing **ChatGPT-4o** and **Claude 3**. A member noted, "Claude 3 is the single worst AI I've ever used," while another highlighted that Perplexity often uses its own search indexes.
- **AI presentation tips**: Members discussed presentations, with suggestions to include AI **advantages and risks** and a **focus on LLMs** like GPT. Links to [videos](https://youtu.be/wjZofJX0v4M?feature=shared) and [search overviews](https://www.perplexity.ai/search/how-does-perplexity-Qm71LBYBSkOApKNFCISS0Q) were shared to help with content.
- **Frustration with Perplexity AI’s interface**: Users reported issues like a model selection bug and problems with the **image generation feature**. One user mentioned, "When I ask Perplexity to produce an image, it gives me a detailed description of how to draw it myself."
- **Perplexity AI's unique SEO ranking**: Discussions revealed that Perplexity AI uses its own crawlers for indexing rather than third-party services like Google or Bing. This method results in "higher-quality" and less SEO-optimized results.
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1247657879725674679)** (13 messages🔥): 

- **Check out dailyfocus_daily article**: Members shared a link to a [Perplexity AI search result](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg) presumably for further reading or discussion.
- **Major outage analysis**: A user pointed to an article detailing a [major outage](https://www.perplexity.ai/page/Major-Outage-of-DCcT_vXARMmWZl8KCWB8Jg) affecting DCcT, providing critical insights into the event.
- **Shareable thread reminder**: Perplexity AI bot reminded users several times to make sure their threads are marked as shareable. It included a [specific attachment link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) detailing how to do this.
- **Further reading on referential topics**: Users shared several search result links for more in-depth exploration on different subjects like one [referring to Gonz](https://www.perplexity.ai/search/referring-to-the-GonzYU0ZTU6cTHhv5qLBXg) and another [search related to Kuhn](https://www.perplexity.ai/search/who-was-Kuhn-mKejsw.LRjOMtIPJtQjpCQ). These were contributions to ongoing discussions.
- **Bitcoin page shared**: Members were directed to a detailed [Bitcoin page](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw) on Perplexity AI, likely to enrich their understanding or for discussion purposes.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1247664700683190394)** (6 messages): 

- **Open Office Hours for vLLM and Neural Magic**: vLLM and Neural Magic are hosting biweekly open office hours to answer questions on optimized LLM inference and accelerated enterprise ML production deployments. Register for the sessions on [June 5, 2024](https://neuralmagic.com/community-office-hours/) and [June 20, 2024](https://neuralmagic.com/community-office-hours/).

- **Preparation for CUDA Programming Interviews**: For those preparing for performance engineer roles, a member suggested using questions from the [awesomeMLSys](https://github.com/cuda-mode/awesomeMLSys) GitHub repository. The repository contains a curated list of questions and resources for ML systems onboarding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cuda-mode/awesomeMLSys">GitHub - cuda-mode/awesomeMLSys: An ML Systems Onboarding list</a>: An ML Systems Onboarding list. Contribute to cuda-mode/awesomeMLSys development by creating an account on GitHub.</li><li><a href="https://neuralmagic.com/community-office-hours/">Bi-Weekly vLLM Open Office Hours - Neural Magic</a>: Join vLLM and Neural Magic for &quot;office hours&quot; on optimized LLM inference and accelerated production deployments using Neural Magic and vLLM.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1247786517452161044)** (4 messages): 

- **Seeking PTX code from Triton Kernel**: A user asked how to obtain the **PTX code** generated by the **Triton kernel**. Another provided a [GitHub issue link](https://github.com/triton-lang/triton/issues/3726) that discusses this topic.
- **Cache Location Clarified**: The same user initially looked for the PTX code in `~/.cache/triton` but subsequently corrected it to `~/triton/.cache`.

**Link mentioned**: <a href="https://github.com/triton-lang/triton/issues/3726">How to get the generated CUDA code?  · Issue #3726 · triton-lang/triton</a>: no description found

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

piotr.mazurek: Chapter 4, exercise 9, anyone knows if this is the corrext solution here?
  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1247974542547095746)** (1 messages): 

- **vLLM open office hours announced**: Members were informed about the **vLLM open office hours** with Simon and another member, happening for the next hour. The office hours are accessible via a [Zoom meeting link](https://us02web.zoom.us/j/87117845746?pwd=QWZsUHlzR1ZYckxpMnNHN2hYWXhzQT09) and supported in multiple languages, including English, Español, Deutsch, 简体中文, and more.

**Link mentioned**: <a href="https://us02web.zoom.us/j/87117845746?pwd=QWZsUHlzR1ZYckxpMnNHN2hYWXhzQT09">Welcome! You are invited to join a meeting: vLLM Open Office Hours (June 5, 2024). After registering, you will receive a confirmation email about joining the meeting.</a>: As a very active contributor to the vLLM project, Neural Magic is excited to partner with the vLLM team at UC Berkeley to host bi-weekly open office hours! Come with questions to learn more about the ...

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1247715165340958813)** (16 messages🔥): 

- **Script to test HF Models generates buzz**: Members discussed a [GitHub pull request](https://github.com/pytorch/ao/pull/328) that adds a script for users to quickly test model evaluation with torchao APIs. One user reported issues with Out of Memory (OOM) errors while quantizing and evaluating large models on GPU.
  
- **OOM Problems with Large Models**: A member noted that running the default script caused OOM errors, suggesting to load models on CPU before quantization to avoid this. They observed that memory issues also depend on the task and the model's vocabulary size.

- **Investigating OOM Issues**: Users looked into the root of these OOM issues, mentioning a high number of open issues in the [EleutherAI/lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aissue+oom+is%3Aopen+). They discussed that large vocab sizes, such as in llama3, exacerbate these problems.

- **Specific Task Memory Usage**: When running tasks like wikitext, the need for large logits tensors can cause memory issues, unlike smaller tasks like hellaswag which use shorter sequences. This observation highlighted the impact of specific evaluation tasks on memory requirements.

- **Optimizations and Discussions on Quantization**: There was a technical discussion on whether `torch.compile()` should be applied after or before quantization, concluding it should be done before in their case. It was also noted that Intel recommends disabling fast math for the inductor CPU backend.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Salesforce/wikitext">Salesforce/wikitext · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues?q=is%3Aissue+oom+is%3Aopen+">Issues · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/pytorch/ao/pull/328">Adding a quick way for users to test model eval for hf models by HDCharles · Pull Request #328 · pytorch/ao</a>: Summary: This script allows users to run evaluation and try out torchao APIs Test Plan: python hf_eval.py Reviewers: Subscribers: Tasks: Tags:
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247625933305810996)** (245 messages🔥🔥): 

- **Efficient cuda stream usage discussed**: Members discussed the need for migrating all operations to the "main stream" to improve performance. **Eriks.0595** noted the benefit of **named streams** and reassured the code should not show any operations on the legacy stream post the initial memcpy.

- **Fixing Pytorch DDP Loss Bug**: **Aleksagordic** fixed a bug in PyTorch DDP loss computation where losses were not reduced properly before being logged, causing a perceived divergence with the C implementation. They submitted a [PR](https://github.com/karpathy/llm.c/pull/551) to correct this issue.

- **Performance Optimization for Matrix Operations**: There were detailed discussions on **optimizing matrix operations** in CUDA, with members noting substantial but not maximal speed gains with improved memory load patterns. **Aleksagordic** sought clarifications on warp formation and correct kernel implementation to ensure no future tokens are inaccurately computed.

- **Ensuring Deterministic Kernel Implementation**: Emphasis was placed on making sure all kernels used are deterministic by avoiding atomic operations. An action item was noted for **global norm kernel** to finalize deterministic operations by using CPU buffers for partial sums.

- **Deposition Day**: **Akakak1337** mentioned being unavailable due to a deposition related to Tesla, humorously noting the **minimal financial compensation** for their time and the adversarial nature of the process.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/552">Feature/streams by karpathy · Pull Request #552 · karpathy/llm.c</a>: bringing back streams, this PR brings back a single &quot;main stream&quot; to start.</li><li><a href="https://github.com/karpathy/llm.c/pull/551">Fix PyTorch DDP loss bug by gordicaleksa · Pull Request #551 · karpathy/llm.c</a>: In our C implementation we reduce the loss properly before displaying it using print0. In our PyTorch imp losses were not reduced and that caused us to think we have a divergence between implementa...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/trimat_forward.cu#L452),">llm.c/dev/cuda/trimat_forward.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1247626303931158599)** (1 messages): 

- **Fake Tensor issue gets a fix**: Marksaroufim shared a [pull request from PyTorch](https://github.com/pytorch/pytorch/pull/127927) aimed at fixing the fake tensor issue. The issue, detailed in the description, involves ensuring tensors are converted to FakeTensors or instantiated with `FakeTensorMode` to avoid errors.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/pull/127927">FunctionalTensor: dispatch metadata directly to inner tensor by bdhirsh · Pull Request #127927 · pytorch/pytorch</a>: Fixes #127374 The error in the linked repro is: AssertionError: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with &#39;allow_non_fake_inputs&#39;. Found in aten.sym_st...

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1247664398324072478)** (4 messages): 

- **Wikipedia Explains Sparse Matrices**: A member asked if the Wikipedia definition of a sparse matrix is accurate and provided a [link to the Wikipedia page](https://en.wikipedia.org/wiki/Sparse_matrix) with an example matrix image. Sparse matrices are crucial in numerical analysis and scientific computing due to their efficiency in storing data with many zero elements.
  
- **PyTorch Sparsity Overview Available**: Another member shared a [README from PyTorch](https://github.com/pytorch/ao/tree/main/torchao/sparsity) which provides a basic overview of sparsity concepts. The README likely includes discussions on quantization and sparsity within PyTorch libraries.

- **Survey on Sparsity in Deep Learning**: A member shared a comprehensive survey paper on sparsity in deep learning available on [arXiv](https://arxiv.org/abs/2102.00554). The paper discusses the advantages of sparsity in neural networks, approaches for pruning, and strategies for training sparse models, summarizing insights from over 300 research papers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Sparse_matrix">Sparse matrix - Wikipedia</a>: no description found</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity">ao/torchao/sparsity at main · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://arxiv.org/abs/2102.00554">Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks</a>: The growing energy and performance costs of deep learning have driven the community to reduce the size of neural networks by selectively pruning components. Similarly to their biological counterparts,...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1247654554091389079)** (1 messages): 

<ul>
    <li><strong>FineWeb Report Revealed the Secrets of High Performance LLMs</strong>: Hugging Face released the <a href="https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1">FineWeb technical report</a>, detailing processing decisions and introducing the FineWeb-Edu dataset for high educational content. This report helps understand high performance LLMs like Llama3 and GPT-4.</li>
    <li><strong>Transformers.js Lands in Firefox 130</strong>: <a href="https://x.com/xenovacom/status/1797285648572821840">Firefox 130</a> will include fully private on-device AI using Transformers.js, initially for automatic alt-text generation in images. This new feature aims to enhance accessibility by extending its capabilities to general browsing for screen reader users.</li>
    <li><strong>Gradio Clients 1.0 Launch Event</strong>: Join the <a href="https://discord.com/events/879548962464493619/1245020251611992154">Gradio Clients 1.0 launch event</a> on June 6, showcasing how Gradio applications can act as dependable APIs ready for production with high performance and scalability.</li>
    <li><strong>Nvidia NIM Released on Hugging Face Inference Endpoints</strong>: Nvidia announced NIM services on Hugging Face Inference Endpoints, enabling <a href="https://x.com/_philschmid/status/1797713003778883858">1-click deployment</a> of models like Llama 3 8B and 70B on AWS and GCP with high throughput.</li>
    <li><strong>Hugging Face Collaborates with Wikimedia</strong>: An article details the potential for advancing ML through diverse datasets from Wikipedia, emphasizing the role of community consent and how to create more <a href="https://huggingface.co/blog/frimelle/wikipedias-treasure-trove-ml-data">Wikimedia datasets on Hugging Face</a>.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1797173053123916036)">Tweet from Guilherme Penedo (@gui_penedo)</a>: We are (finally) releasing the 🍷 FineWeb technical report!  In it, we detail and explain every processing decision we took, and we also introduce our newest dataset: 📚 FineWeb-Edu, a (web only) subs...</li><li><a href="https://x.com/xenovacom/status/1797285648572821840)">Tweet from Xenova (@xenovacom)</a>: Transformers.js is being added to Firefox 130! 🤯 That’s right, fully private on-device AI directly in your browser! 🔥  The first use-case they’re exploring is automatic alt-text generation for image...</li><li><a href="https://x.com/Gradio/status/1795561025397256498)">Tweet from Gradio (@Gradio)</a>: 🚀𝐏𝐫𝐨𝐭𝐨𝐭𝐲𝐩𝐞𝐬 𝐭𝐨 𝐏𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧!  🙌Join us for the much-anticipated Launch Event for Gradio Clients 1.0 on June 6.   🤩Understand how your Gradio applications exhibit high performanc...</li><li><a href="https://x.com/kamilakesbi/status/1796537200961785931)">Tweet from Kamil Akesbi (@kamilakesbi)</a>: The biggest barrier to speaker diarization ? Data!  With 🤗 Diarizers, you can now generate synthetic meeting 🗣️ conversations!  Starting from an ASR dataset, you can create arbitrary amounts of data...</li><li><a href="https://x.com/_philschmid/status/1797713003778883858)">Tweet from Philipp Schmid (@_philschmid)</a>: Yesterday at COMPUTEX, Jensen Huang announced the release of @nvidia NIM on @huggingface Inference Endpoints! 🚀 NVIDIA NIM are inference services designed to streamline and accelerate the deployment ...</li><li><a href="https://x.com/_philschmid/status/1795804027621404975)">Tweet from Philipp Schmid (@_philschmid)</a>: Product Update: @nvidia L4s are now available in @huggingface  Inference Endpoints on AWS!  Enjoy up to 8x L4s per user and organization, and save 20% compared to on-demand AWS EC2. 🤑  - 1x NVIDIA L4...</li><li><a href="https://x.com/abhi1thakur/status/1795477747701104651)">Tweet from abhishek (@abhi1thakur)</a>: AutoTrain just got a brand new UI 🚀🚀🚀</li><li><a href="https://x.com/_philschmid/status/1797994961197031703">Tweet from Philipp Schmid (@_philschmid)</a>: Excited to share a new blog on how to fine-tune embedding models for financial RAG applications using NVIDIA&#39;s 2023 SEC Filing dataset using the latest research, like Matryoshka Representation Lea...</li><li><a href="https://x.com/frimelle/status/1797619351954260214)">Tweet from Lucie-Aimée Kaffee (@frimelle)</a>: Community-centric and awesome: @huggingface and @Wikimedia 🤗 I wrote an article on how we can advance ML with diverse datasets from @Wikipedia, why and how to create more Wikimedia datasets on Huggin...</li><li><a href="https://x.com/NielsRogge/status/1796213271189438888)">Tweet from Niels Rogge (@NielsRogge)</a>: Alright finally back on @YouTube with a new video: fine-tuning PaliGemma (or LLaVa, Idefics2,...) on your custom dataset!  I&#39;m fine-tuning in @GoogleColab on an L4 GPU   I go over many things like...</li><li><a href="https://x.com/abhi1thakur/status/1796210385579639144)">Tweet from abhishek (@abhi1thakur)</a>: 🚨 NEW BLOG: How to Fine-Tune Custom Embedding Models Using AutoTrain Learn: - what should be the data format - how to map columns properly - example datasets - custom configs - train locally - train ...</li><li><a href="https://x.com/vanstriendaniel/status/1795875763557904753">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Do you need a dataset to train a custom sentence transformer model? I&#39;ve created a pipeline for using an LLM to create a synthetic dataset you can directly use for fine-tuning/training a Setence T...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247628263816167585)** (214 messages🔥🔥): 

- **TinyLlama struggles with training duration**: A member complained about TinyLlama performing poorly because it doesn't allow long enough training periods. The conversation circled around fine-tuning settings and platform specifics.
- **HF Spaces secrets exposed in hacker incident**: A discussion emerged about an email from Hugging Face, revealing a security breach where private HF tokens were made public. Concerns were raised regarding their private codes being potentially compromised.
- **Debate on the security of internet data**: Users debated the security of internet-based data following the HF breach. One member expressed deep concern, while another facetiously pointed out that internet security is inherently unreliable.
- **Exploring diffusion-based language models**: A conversation about creating a diffusion-based language model surfaced, akin to image generation models. It involved brainstorming approaches to remove "noise" from random character strings.
- **Addressing abusive behavior in discussions**: Some users discussed handling inappropriate and aggressive behavior in community discussions. Reporting and moderation were emphasized as necessary steps to maintain a respectful environment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/mayo">Mayo - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://huggingface.co/blog/space-secrets-disclosure">Space secrets security update</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.17.0/en/create_a_model#confi">Create a custom model</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.17.0/en/create_a_model#configuration">Create a custom model</a>: no description found</li><li><a href="https://huggingface.co/blog/abhishek/object-detection-autotrain">Training an Object Detection Model with AutoTrain</a>: no description found</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">unilm/textdiffuser at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://tenor.com/view/saul-goodman-talking-gif-26157017">Saul Goodman Talking GIF - Saul Goodman Talking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/kruMoWQLd1u.gif">Are You From Ohio Or Something Our Drawings GIF - Are you from ohio or something Ohio Our drawings - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/69">zero-gpu-explorers/README · DON&#39;T EVEN BOTHER APPLYING - Unofficial ZeroGPU Policy Decoded</a>: no description found</li><li><a href="https://github.com/abetlen/llama-cpp-python/issues/576">How to use  GPU? · Issue #576 · abetlen/llama-cpp-python</a>: I run llama cpp python on my new PC which has a built in RTX 3060 with 12GB VRAM This is my code: from llama_cpp import Llama llm = Llama(model_path=&quot;./wizard-mega-13B.ggmlv3.q4_0.bin&quot;, n_ct...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1254">GGUF Local Model · Issue #1254 · EleutherAI/lm-evaluation-harness</a>: Is there examples of lm_eval for gruff models hosted locally? lm_eval --model gguf --model_args pretrained=Llama-2-7b-chat-hf-Q4_K_M.gguf, --tasks hellaswag --device mps Getting AssertionError: mus...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1247740949770145873)** (2 messages): 

- **Collaborate on ML and Streamlit Project**: A member asked, *"Does anyone wanna build a ml and Streamlit project together?"* This opens up an invitation for collaboration on a machine learning and Streamlit-based project within the community.
- **Greetings Exchange**: Another user simply greeted with a *"hello"*. This suggests a casual, welcoming atmosphere in the channel.
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1247627017248047315)** (5 messages): 

- **610 hours of German audio samples for ASR/TTS training**: A member shared a dataset of 610 hours of transcribed audio samples of speeches from the German parliament for ASR/TTS training. The dataset is available on [Hugging Face Hub](https://huggingface.co/datasets/D4ve-R/bundestag-asr).

- **Handwritten C programming for understanding Transformers**: A tweet showcased a project that combines handwritten C programming and matrix multiplication to explain how Transformers work. This initiative is inspired by @karpathy and aims to demystify LLMs, more details can be found [here](https://x.com/ProfTomYeh/status/1798042265883156651).

- **Early testing of CogVLM2-LLaMA3-Chat-19B**: There was early testing of the [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) vision language model based on LLaMA3. The model boasts significant improvements, supports 8K content length, and offers high image resolution but has issues running on Windows.

- **Microsoft's TextDiffuser projects on GitHub**: Microsoft has released [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser), a project focused on large-scale self-supervised pre-training across tasks, languages, and modalities. An updated version, [TextDiffuser-2](https://github.com/microsoft/unilm/tree/master/textdiffuser-2), was also noted, continuing to build on these capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: no description found</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">unilm/textdiffuser at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">Tweet from Tom Yeh | AI by Hand ✍️ (@ProfTomYeh)</a>: llm.c by Hand✍️  C programming +  matrix multiplication by hand  This combination is perhaps as low as we can get to explain how the Transformer works.   Special thanks to @karpathy for encouraging ea...</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser-2">unilm/textdiffuser-2 at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://huggingface.co/datasets/D4ve-R/bundestag-asr">D4ve-R/bundestag-asr · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1247821562414108682)** (5 messages): 

- **Installing Apache Airflow on Windows**: A good read was shared on how to install **Apache Airflow on Windows** using WSL. You can check out the detailed guide [here](https://tolulade-ademisoye.medium.com/how-to-install-apache-airflow-on-windows-a-beginners-guide-to-wsl-297c5ba5f519).

- **Comprehensive LLM Resource Guide**: A member compiled a resource guide of favorite **LLM explainers**, including vLLM, SSMs, DPO, and QLoRA. Check out the organized "textbook-shaped" resource guide [here](http://genai-handbook.github.io) and the announcement [here](https://x.com/willccbb/status/1798423849870270671).

- **AI for Climate-Aware Investments**: A member created an AI assistant to find climate-oriented solutions for investments using `climatebert/tcfd_recommendation`, Qdrant Cloud, and `microsoft/Phi-3-mini-128k-instruct`. Explore the AI assistant [here](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor) and calculate your carbon footprint with their ML-backed solution [here](https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/tcfd_counselor">Tcfd Counselor - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor">Carbon Footprint Predictor - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://x.com/willccbb/status/1798423849870270671">Tweet from will brown (@willccbb)</a>: been learning a lot about LLMs etc over the past year, organized some of my favorite explainers into a “textbook-shaped” resource guide  wish i’d had this at the start, maybe it can useful to others o...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1247990833571958824)** (2 messages): 

- **Human Feedback Foundation event on June 11th**: There's an upcoming event organized by the **Human Feedback Foundation** on June 11th, aiming to enhance public input into AI systems in critical areas like healthcare, governance, and democracy. [Event link](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator).

- **Previous session recordings available on YouTube**: Recordings from past sessions featuring speakers from prominent institutions such as UofT, Stanford, and OpenAI, are available on the Human Feedback Foundation's [YouTube channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg). The foundation emphasizes the importance of public input in AI development, interoperability of AI tools, and educational outreach in AI safety research.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>: Human Feedback Foundation is on a mission to build human feedback into open-source AI projects.  We seek to:  Enable public input into AI through supporting open-source development and policy initiati...</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14, 28; June 11)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1247776895299227732)** (7 messages): 

- **Single-country Logo Detection**: A user is working on a project that detects a specific country's logo. They seek a system that classifies images as either containing the logo or not to keep the dataset manageable.

- **CamViG Paper Discussed**: [CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers](https://arxiv.org/abs/2405.13195). The paper proposes conditioning video generation on 3D camera motion, demonstrating successful camera control and accurate 3D camera path generation.

- **Seeking Use Cases for MultiModal LLMs**: A user asks for suggestions on good use cases for MultiModal large language models. No specific suggestions were provided in the discussion.

- **Physical Scale Estimation in Images**: A user is building a pipeline to estimate the physical scale of pixels in images of rooms. They seek prior art and resources on HuggingFace or similar forums to assist with their project, provided certain assumptions about the images.

**Link mentioned**: <a href="https://huggingface.co/papers/2405.13195">Paper page - CamViG: Camera Aware Image-to-Video Generation with Multimodal
  Transformers</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1247642239933616207)** (2 messages): 

- **Try importing specific functions to resolve version mismatches**: A user suggested importing specific functions to see if they arrive, indicating possible issues related to version mismatches. They recommended this as a troubleshooting step.

- **GroundedAI open-sources models for LLM evaluation**: A member announced that they have open-sourced models for performing LLM as a judge evaluation, available [here](https://huggingface.co/grounded-ai). These models aim to provide efficient, high-performing alternatives to black box LLMs and ensure outputs are factual and ethical.

**Link mentioned**: <a href="https://huggingface.co/grounded-ai">grounded-ai (GroundedAI)</a>: no description found

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247695146535878787)** (25 messages🔥): 

- **Script and Model Configuration Guidance**: A user detailed a comprehensive process for updating a script, using values from the SDXL model config, including setting up `sample`, `timestep`, and `encoder_hidden_states`. They elaborated on the necessary arguments and parameters such as `added_cond_kwargs` and `text_embeds`, offering code snippets to clarify implementation.

- **Overwhelmed by AI Innovation**: A member expressed feeling overwhelmed by the rapid pace of AI advancements, stating, "it is impossible to keep up." Another member agreed, noting that trying to stay updated on everything "will just lead to madness."

- **Troubleshooting For Missing `text_embeds`**: A member reported a `TypeError` related to missing `text_embeds` in `added_cond_kwargs`. They received help specifying `text_embeds` as `torch.randn(2, 77, 1280).half().cuda()` and discussed using initializations within function scopes to avoid such issues.

- **Successful Debugging Celebrated**: After resolving issues with their script, a member celebrated with a [GIF of amazement](https://tenor.com/view/wow-amazed-in-awe-woah-smiling-gif-16490512). They acknowledged previous misunderstandings related to the dimensions of text encoders and correctly incorporating `text_embeds` and `time_ids`.

**Link mentioned**: <a href="https://tenor.com/view/wow-amazed-in-awe-woah-smiling-gif-16490512">Wow Amazed GIF - Wow Amazed In Awe - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1247808792390008872)** (1 messages): 

- **Gradio Clients 1.0 goes live on YouTube**: HuggingFace announces the launch of Gradio Clients 1.0 with a YouTube Livestream event. The new tools allow users to leverage Gradio demos programmatically, enhancing **Python** and **JavaScript** client capabilities for production-ready applications.
- **Event specific details provided**: The announcement includes links to the **Discord event** and the **YouTube livestream** where the launch will be discussed. This offers an opportunity for developers to learn from the Gradio team about building machine learning APIs using the new clients.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://www.youtube.com/watch?v=44vi31hehw4">[Launch] How to Build Machine Learning APIs Using Gradio</a>: One million developers use Gradio every month to create machine learning demos and web applications using the Gradio Python library. Join the Gradio Team on ...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247626730517041152)** (47 messages🔥): 

- **Llama70b struggles with GGUF format**: Members discussed issues with loading Llama70b in **LM Studio** due to it not being saved as a GGUF file. One suggested using a sym link option while another recommended converting the file or redownloading it as a GGUF.

- **Cryptic humor hit the channel**: There was a humorous exchange about GPUs mining crypto secretly at night, eliciting laughs from other members.

- **LM Studio 0.2.24 won't start on Windows 10**: A user reported that **LM Studio 0.2.24** wouldn't start, showing no trace in the task manager, while **version 0.2.22** worked fine. Despite reinstalling the app and running it as an admin, the issue persisted without generating error logs or useful information in the event viewer.

- **Queries on iMat files and RAM usage**: Users inquired about using iMat files in **LM Studio** and methods to limit RAM usage. It was clarified that iMat support depends on llamacpp’s compatibility, and for RAM, it’s advisable to use smaller models or manage layers during inference to optimize usage.

- **Stock market photo analysis question**: A user spammed multiple channels asking if a model could analyze stock market photos but was instructed that local models cannot perform this type of analysis. They were advised to post questions only once to avoid spamming.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://tenor.com/view/tf2engineer-imposter-it-could-be-you-meme-tf2spy-gif-23428001">Tf2engineer Imposter GIF - Tf2Engineer Imposter It Could Be You - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247632691340509266)** (102 messages🔥🔥): 

- **Recommended Model for Writing Improvement**: For enhancing writing style, a member suggested checking the leaderboard and filtering models by 13B, without specifying a particular model. 

- **Issues with SMAUG and Llama 3**: Multiple users confirmed facing issues with SMAUG's BPE tokenizer error while using Llama 3 version 0.2.24, noting support for it was added a week later.

- **Command R Model Performance Queries**: A user reported Command R model offloaded to Metal produces gibberish while CPU-only mode is extremely slow. The recommended check was for the rope settings and another link to a specific [Hugging Face model](https://huggingface.co/mradermacher/c4ai-command-r-v01-GGUF) was provided.

- **Differences between Static and iMatrix Quantized Models**: Users discussed how iMatrix quants perform better overall when quantizing models since "they avoid quantizing important weights as much as possible." They also noted that certain hardware configurations might affect this.

- **Uncensored Models**: Discussions on uncensored models highlighted specific models like *Neural Devil* for general use and story writing. A link to one model ([YorkieOH10/Llama-3-MahouDevil-8B-Q8_0](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF)) was shared along with the user's positive experiences.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mradermacher/13B-Psyfighter2-Erebus3-Slerp-GGUF">mradermacher/13B-Psyfighter2-Erebus3-Slerp-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/Euryale-1.3-Small-7B-i1-GGUF">mradermacher/Euryale-1.3-Small-7B-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_merged-GGUF">mradermacher/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_merged-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF">bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1247902551123296266)** (1 messages): 

- **Extra escape character bugs czkoko v0.2.24**: The latest version, **v0.2.24**, occasionally includes an unexpected escape character in the preset configuration. Examples in the configuration include `"input_suffix": "\\n\\nAssistant: "` and `"pre_prompt_suffix": "\\n\\n"`.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247637209775996990)** (90 messages🔥🔥): 

- **AMD Radeon RX 7900 Workstation GPUs Spark Interest**: A discussion emerged about the [new ASRock Radeon RX 7900 XTX & 7900 XT Workstation GPUs](https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/), with features including a 12V-2x6 power connector and dual-slot blower design. Some members expressed concerns about the noisy blower fans despite the appeal of dual-slot GPUs.
  
- **Controversial Views on Linux**: One member criticized Linux for being "infested with command lines," arguing it's not user-friendly despite common claims. Another suggested that the real problem is a lack of research and exaggerated expectations, while recommending KDE Plasma for a more Windows-like experience.

- **Recall Feature Sparks Privacy Concerns**: Members discussed the controversial Recall feature in Windows, highlighting its potential to aggregate sensitive information into an accessible database for hackers. The overall sentiment showcased a major concern for privacy and security risks associated with this feature.

- **Shift Towards Linux Amid Windows Recalls**: The impending issues with Windows Recall have led to talks of a potential "influx of Linux installs," with some joking about big box stores pushing this shift onto unsuspecting consumers. Experienced community members highlighted the critical security implications Recall poses for both personal and business use of Windows.

- **Help Desk Horror Stories**: Light-hearted complaints about past experiences in IT support roles appeared, with one member humorously recounting a tale of repairing a computer that smelled of cat pee. This part of the conversation brought a relatable note to the technical discussions about system administration and the challenges of IT security.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wccftech.com/asrock-radeon-rx-7900-xtx-7900-xt-workstation-gpus-12v-2x6-connector-2-slot-design-for-ai-setups/">ASRock Radeon RX 7900 XTX &amp; 7900 XT Workstation GPUs Adopt 12V-2x6 Connector &amp; 2-Slot Design For AI Setups</a>: ASRock has released its Radeon RX 7900 XTX &amp; 7900 XT Workstation GPUs that feature a 12V-2x6 power connector &amp; use a dual-slot blower design.</li><li><a href="https://tenor.com/view/tupac-true-pointing-up-truth-2pac-gif-26578973">Tupac True GIF - Tupac True Pointing Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.ebay.co.uk/itm/334715824949">Intel Neural Compute Stick 2 Neural Deep Learning USB NCSM2485.DK NCS  | eBay</a>: no description found</li><li><a href="https://youtu.be/0506yDSgU7M?si=gItHh97CbX7qTxVC">Linux HATES Me – Daily Driver Challenge Pt.1</a>: Try FreshBooks free, for 30 days, no credit card required at https://www.freshbooks.com/linusUse code LINUS and get 25% off GlassWire at https://lmg.gg/glass...</li><li><a href="https://youtu.be/3E8IGy6I9Wo?si=oR9VTNHjrt-dbz9c">This is NOT going Well… Linux Gaming Challenge Pt.2</a>: Try Pulseway for free and start remotely monitoring and managing your server or PC at https://lmg.gg/Ktd7ZUse code LINUS and get 25% off GlassWire at https:/...</li><li><a href="https://youtu.be/TtsglXhbxno?si=XzudnJGBniprnlMS">Trying to do Simple Tasks on Linux lol - Daily Driver Challenge Pt.3</a>: Check out NZXT BLD today at: https://nzxt.co/LttBLDNov21Linus &amp; Luke use their Linux desktops to complete a list of mundane  tasks like printing and compress...</li><li><a href="https://youtu.be/Rlg4K16ujFw?si=91uM_GPer6kkQxW6">Gaming on Linux is NOT Ready... - Daily Driver Challenge Finale</a>: Visit https://www.squarespace.com/LTT and use offer code LTT for 10% offTry your first eSIM with Airalo at https://lmg.gg/AiraloIt&#39;s been a month of Luke and...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1247924966884905081)** (3 messages): 

- **VRAM insufficiency causing model load failure**: A user faced an error while loading a model with the given system specifications. Another member pointed out that the issue arises from insufficient VRAM for GPU offload and suggested turning it off to resolve the problem.
  

---


### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1247957815189045258)** (2 messages): 

- **Query about Operating System**: A user inquired about the operating system used by others. One member replied that they are using **Windows 11**.
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247627293422260306)** (185 messages🔥🔥): 

- **DDoS attack causes widespread AI service outages**: A member shared that the recent ChatGPT, Claude, Gemini, Perplexity, and Copilot outages were due to a DDoS attack by Anonymous Sudan, a pro-Russian hacker group. This sheds light on why the issues were perceived as more significant than typical cloud server problems.
  
- **Utility of GPT subscriptions debated**: Discussions emerged around the utility of various AI subscriptions. One user compared the productivity of subscribing to GPT versus Character AI, emphasizing the practical uses like summarizing books and creating content.

- **Language models struggle with math**: Multiple users discussed how language models like GPT often struggle with mathematical tasks. Specific issues included the models giving incorrect calculations and not recognizing logical errors, despite user corrections.

- **AI's role in everyday tasks and systems**: Users shared experiences of integrating AI with other software and activities. Examples include connecting ChatGPT with home automation systems and the realistic applications and limitations of Windows Copilot.

- **Potential future AI capabilities and concerns**: One user humorously suggested that pet AI might one day turn on its users, sparking a broader conversation on the ethical treatment of AI and its long-term implications. There's also a curiosity about future advancements in AI capabilities, such as accessing Windows API functions.
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1247756498051010671)** (5 messages): 

- **Curiosity about GPT-4 Vision in Google Sheets**: A user queried if there was any way to use **GPT-4 vision** within Google Sheets to describe images in cells and place the descriptions in adjacent cells. Another member suggested *"you can use the API"*.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247661984556777512)** (13 messages🔥): 

- **Add prompts to "The Prompt Index"**: A member is curating a resource called **"The Prompt Index"** and invites others to contribute interesting prompts. They emphasize that it's a completely free service aimed at avoiding the deluge of overpriced, low-quality content.
- **GPT-4 Vision in Google Sheets aim**: A member asks if they can use **GPT-4 Vision** to analyze images in Google Sheets and generate descriptions in neighboring cells. They are looking for a way to automate this task within the spreadsheet environment.
- **Fixing GPT with a simple refresh**: A member questions how to fix GPT when it gets stuck. Another member suggests trying **Ctrl + F5** to refresh the page, and the original poster responds that they will give it a try.
- **SEO Analysis with ChatGPT**: A member seeks a method for **ChatGPT** to deliver specific analysis and optimization strategies for site SEO, as the current responses are too general. Another member suggests comparing the source code of a site with good SEO against their own site and instructing ChatGPT to apply the successful strategies.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247661984556777512)** (13 messages🔥): 

- **Free Prompt Collection Resource Shared**: A member introduced "The Prompt Index" as a **completely free resource** for interesting prompts. They specified it is meant to avoid the typical marketing prompts found elsewhere.

- **Inquiry about GPT-4 Vision in Google Sheets**: A user asked if there's a way to use **GPT-4 Vision in Google Sheets** to describe images in cells and place the descriptions in adjacent cells. No solution was provided in the chat.

- **Stuck GPT Solution Shared**: A user inquired about how to fix a **stuck GPT**. Another member suggested using *Ctrl + F5* to refresh the page.

- **Optimal Method for SEO Analysis Using ChatGPT**: A member sought advice on having ChatGPT analyze SEO more accurately for their site. Another user recommended copying and pasting source codes from a well-optimized site and the user's site, then asking ChatGPT to compare and suggest improvements.

- **Successful Model Integration Experiment**: A user shared that integrating the **llava-v1.6-mistral-7b vision model** with the 7b model worked successfully. They found it interesting that these models could function together without issues.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1247961033793081374)** (1 messages): 

- **Introducing Stable Audio Open**: Stability.ai announced the release of **Stable Audio Open**, an open-source model for generating short audio samples, sound effects, and production elements using text prompts. The release aims to empower sound designers, musicians, and creative communities by providing free model weights, documentation, code examples, and the ability for local fine-tuning with custom audio data. Learn more about it [here](https://stability.ai/news/introducing-stable-audio-open). 

- **Key Features of Stable Audio Open**: The model can generate high-quality audio clips up to 47 seconds long, including drum beats, instrument riffs, ambient sounds, and foley. It also supports variations and style transfer of uploaded audio samples.

- ![Stable Audio Open](https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/ecc597b7-93ff-48b0-afc1-7feb8cd613d8/Stable_audio_open.jpg): The release highlights a key milestone in opening portions of generative audio capabilities to a wider audience.

**Link mentioned**: <a href="https://stability.ai/news/introducing-stable-audio-open"> Stable Audio Open &mdash; Stability AI</a>: Stable Audio Open is an open source model optimised for generating short audio samples, sound effects and production elements using text prompts.

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247643080136462477)** (141 messages🔥🔥): 

- **Exploring WebUI Options: A1111 vs. InvokeAI**: Members discussed the advantages of different WebUIs for Stable Diffusion, with **A1111** praised for versatility and ease of use, while **InvokeAI** was noted for its "regional prompting" feature and being more innovative. [Here's InvokeAI on GitHub](https://github.com/invoke-ai/InvokeAI).

- **Confusion on Regularization Images and Captions**: Users were uncertain whether **regularization images** require captions when training models, sparking a brief debate. "does using reg images mean you don't need to use captions anymore?" was one such question.

- **Stable Audio Tools & Usability**: Members shared resources about generating audio with AI, highlighting [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools). There was curiosity about its use on Google Colab and whether commercial use is permitted.

- **ComfyUI Flexibility in Image Generation**: ComfyUI was recommended for users seeking flexibility in their workflows, albeit with a steep learning curve. "you can generate with cascade or sigma then refine it with sdxl..." was an example provided illustrating its capabilities.

- **Community Resources for Beginners**: Tutorials and discussions on learning Stable Diffusion were shared to assist newcomers. [Sebastian Kamph on YouTube](https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q) was recommended for a comprehensive guide on getting started with A1111.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K">laion/CLIP-ViT-g-14-laion2B-s34B-b88K · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting">stabilityai/stable-diffusion-2-inpainting · Hugging Face</a>: no description found</li><li><a href="https://github.com/invoke-ai/InvokeAI">GitHub - invoke-ai/InvokeAI: InvokeAI is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The solution offers an industry leading WebUI, supports terminal use through a CLI, and serves as the foundation for multiple commercial products.</a>: InvokeAI is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. Th...</li><li><a href="https://github.com/Stabili">StaBili - Overview</a>: StaBili has one repository available. Follow their code on GitHub.</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/kqXpAKVQDNU?si=EHs5JZaQmE1yTi1Q">How to Install Stable Diffusion - automatic1111</a>: Part 2: How to Use Stable Diffusion https://youtu.be/nJlHJZo66UAAutomatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webuiInstall Python https://w...</li><li><a href="https://github.com/numz/sd-wav2lip-uhq">GitHub - numz/sd-wav2lip-uhq: Wav2Lip UHQ extension for Automatic1111</a>: Wav2Lip UHQ extension for Automatic1111. Contribute to numz/sd-wav2lip-uhq development by creating an account on GitHub.</li><li><a href="https://github.com/Stability-AI/stable-audio-tools">GitHub - Stability-AI/stable-audio-tools: Generative models for conditional audio generation</a>: Generative models for conditional audio generation - Stability-AI/stable-audio-tools</li><li><a href="https://github.com/diontimmer/audio-diffusion-gradio">GitHub - diontimmer/audio-diffusion-gradio: Decked-out gradio client for audio diffusion, mainly stable-audio-tools.</a>: Decked-out gradio client for audio diffusion, mainly stable-audio-tools. - diontimmer/audio-diffusion-gradio</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247634160915579022)** (55 messages🔥🔥): 

- **AI-powered display walls: Color stripes and logos**: A member suggested a giant display wall using AI to enhance daily life by changing colors or displaying art automatically. They mentioned features like "little mountain dew color stripes and tastefully presented logos."

- **Critique of overhyped "breakthrough" technology**: Discussing an RLCD product, a member noted its CEO's mystic image and exaggerated claims of unique technology, despite being just an RLCD. Another member countered, explaining that modifications make it more transflective and highlighted similarities to Samsung's QD-OLED displays.

- **AGI race heating up**: A referenced [blog on situational-awareness.ai](https://situational-awareness.ai/) discusses escalating investments in computing clusters, leading to an expected surge in AGI capabilities by 2025/26. A member reflected on the growing gap between frontier labs and the likely hiring signals from these advancements.

- **Debate on IQ tests and high agency in open source**: Members discussed the challenges and legality of using IQ tests for hiring purposes. They highlighted the value of high agency over high IQ alone, noting that success often comes from a combination of ambition, agency, and the ability to detect patterns with high temporal sparsity.

- **Interest in KANs**: Several members debated the potential and limitations of Koopman Operator-Based Neural Networks (KANs). They discussed interpretability, efficiency, and current theoretical gaps, concluding that KANs, despite their promise, are not likely to replace traditional neural networks soon.

**Link mentioned**: <a href="https://situational-awareness.ai/,">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>: Leopold Aschenbrenner, June 2024 You can see the future first in San Francisco. Over the past year, the talk of the town has shifted from $10 billion compute clusters to $100 billion clusters to trill...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247641361524588727)** (58 messages🔥🔥): 

- **Examining limits in transformers and SSMs**: A user shared a [new paper](https://arxiv.org/abs/2405.16674) discussing the limitations of transformers and SSMs in complex reasoning tasks. The paper provides both theoretical and empirical analysis addressing the models' struggles with compositionality.

- **Internalizing Chain-of-Thought steps**: A paper was shared that proposes a method for teaching models to internalize Chain-of-Thought (CoT) reasoning steps, achieving high performance on tasks like 9-by-9 multiplication. The technique benefits models such as Mistral 7B and is discussed in detail [here](https://arxiv.org/abs/2405.14838).

- **Improving RLHF robustness**: Discussion on a [Self-Improving Robust Preference Optimization (SRPO)](https://arxiv.org/abs/2406.01660) method for Reinforcement Learning from Human Feedback (RLHF). The new approach aims to be completely robust to changes in tasks by turning preference optimization into a self-improvement process.

- **New diffusion model guidance method**: Users discussed a [new approach](https://arxiv.org/abs/2406.02507) in image-generating diffusion models that disassociates image quality and variation control, guiding generation using a smaller, less-trained version of the model itself.

- **RETRO model's open implementation**: The discussion covered an open-source implementation of RETRO by NVIDIA available in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro). Questions were raised on the accessibility and potential of running high-quality, large-scale AI models publicly for performance assessment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: The primary axes of interest in image-generating diffusion models are image quality, the amount of variation in the results, and how well the results align with a given condition, e.g., a class label ...</li><li><a href="https://arxiv.org/abs/2406.01660">Self-Improving Robust Preference Optimization</a>: Both online and offline RLHF methods such as PPO and DPO have been extremely successful in aligning AI with human preferences. Despite their success, the existing methods suffer from a fundamental pro...</li><li><a href="https://arxiv.org/abs/2405.20519">Diffusion On Syntax Trees For Program Synthesis</a>: Large language models generate code one token at a time. Their autoregressive generation process lacks the feedback of observing the program&#39;s output. Training LLMs to suggest edits directly can b...</li><li><a href="https://www.colorama.app.">Colorama</a>: Mapping the past in true color</li><li><a href="https://arxiv.org/abs/2406.02075">ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU</a>: Limited by the complexity of basis function (B-spline) calculations, Kolmogorov-Arnold Networks (KAN) suffer from restricted parallel computing capability on GPUs. This paper proposes a novel ReLU-KAN...</li><li><a href="https://arxiv.org/abs/2405.16674">Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory</a>: Deep learning models have achieved significant success across various applications but continue to struggle with tasks requiring complex reasoning over sequences, such as function composition and comp...</li><li><a href="https://arxiv.org/abs/2406.02543">To Believe or Not to Believe Your LLM</a>: We explore uncertainty quantification in large language models (LLMs), with the goal to identify when uncertainty in responses given a query is large. We simultaneously consider both epistemic and ale...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...</li><li><a href="https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro">Megatron-LM/tools/retro at InstructRetro · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://arxiv.org/abs/2405.15143">Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models</a>: Go-Explore is a powerful family of algorithms designed to solve hard-exploration problems, built on the principle of archiving discovered states, and iteratively returning to and exploring from the mo...</li><li><a href="https://tedunderwood.com/2023/03/19/using-gpt-4-to-measure-the-passage-of-time-in-fiction/">Using GPT-4 to measure the passage of time in fiction</a>: Large language models are valuable research assistants, especially when they refuse to follow instructions.</li><li><a href="https://arxiv.org/abs/2406.02394">Multiple Choice Questions and Large Languages Models: A Case Study with Fictional Medical Data</a>: Large Language Models (LLMs) like ChatGPT demonstrate significant potential in the medical field, often evaluated using multiple-choice questions (MCQs) similar to those found on the USMLE. Despite th...</li><li><a href="https://x.com/maximegmd/status/1798245197585002671">Tweet from Maxime G, M.D (@maximegmd)</a>: Are multiple choice questions a reliable tool to evaluate medical performance of Large Language Models? Probably not, LLMs get 67% correct on our benchmark on a fictional gland!  It&#39;s time to invo...</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: The distributional simplicity bias (DSB) posits that neural networks learn low-order moments of the data distribution first, before moving on to higher-order correlations. In this work, we present com...</li><li><a href="https://arxiv.org/abs/2405.14838">From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step</a>: When leveraging language models for reasoning tasks, generating explicit chain-of-thought (CoT) steps often proves essential for achieving high accuracy in final outputs. In this paper, we investigate...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1247828931017445376)** (3 messages): 

- **Troubleshoot lm-eval output issues**: A user faced issues with getting outputs for metric calculations while testing the evaluation on a multiple-choice task using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb). They attempted to use parameters `--limit 40 --output_path tmp --log_samples --predict_only` but didn't get the expected results.
- **Tmp folder might hold results**: Another member suggested checking if the results are in the tmp folder, implying that the user's output might already be stored there.
- **MMLU benchmark on LLaMA 3 8B**: A member sought advice on running the MMLU benchmark with a self-hosted LLaMA 3 8B instruct model. They specifically requested guidance on implementing loglikelihood for both local-completions and local-chat-completions models.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb">lm-evaluation-harness/examples/lm-eval-overview.ipynb at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1247725236397277226)** (5 messages): 

- **Matrix multiplication tutorial video shared**: A member shared an [educational video on matrix multiplication](https://cdn.discordapp.com/attachments/843479632320790551/1247252402952995027/SbGXWFlO2K5kTz-_.mp4). 

- **Discussion on June's Chaos Udiomusic**: A member shared a [link to June's Chaos Udiomusic](https://www.perplexity.ai/page/junes-chaos-udiomusic-KPh68eqNQeejjpuyLKEGIg). No further discussion was noted.

- **Review updates link posted**: A member provided a [link to review updates](https://www.perplexity.ai/search/review-the-updates-_.DlS9FHQfCLpvckPrcnBw).

- **Interest in model creating 15-minute songs**: A member expressed excitement about a "model that can create a 15-minute long song" and showed interest in "pushing the limits of that for the sake of cohesion."
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1247869707692933121)** (10 messages🔥): 

- **Meet GLM-4: A Multilingual, Multimodal Chat Model**: The new [GLM-4](https://x.com/ChatGLM/status/1798292207574901012) supports 26 languages and offers advanced features like code execution and long-text reasoning. Ideal for AI enthusiasts and developers, you can explore further on [GitHub](https://github.com/THUDM/GLM-4).

- **Complexity Extension for Perplexity Users**: A new third-party extension called Complexity aims to fully revamp and improve the user experience for Perplexity users. Interested users can join their [Discord](https://discord.gg/fxzqdkwmWx) to request access and contribute feedback.

- **Nomic-Embed-Vision Outperforms Existing Models**: [Nomic-Embed-Vision](https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg) now provides a unified embedding space for image, text, and multimodal tasks, outperforming OpenAI CLIP and text-embedding-3-small. The weights and code are openly available for indie hacking, research, and experimentation.

- **Decoupled Hyperspherical Energy Loss (DHEL) Introduced**: A recent paper introduces [DHEL](https://arxiv.org/abs/2405.18045), a novel contrastive learning objective that simplifies the alignment of positive examples while retaining theoretical guarantees. A GitHub repo comparing various InfoNCE variants is also available [here](https://github.com/viig99/ContrastiveLearningLossComparison).

- **Handbook of LLM Explainers**: A member shared a [resource guide](https://x.com/willccbb/status/1798423849870270671) for LLM explainers, covering a wide range of topics from vLLM to QLoRA. The guide is intended to assist those new to the field of LLMs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: no description found</li><li><a href="https://x.com/willccbb/status/1798423849870270671">Tweet from will brown (@willccbb)</a>: been learning a lot about LLMs etc over the past year, organized some of my favorite explainers into a “textbook-shaped” resource guide  wish i’d had this at the start, maybe it can useful to others o...</li><li><a href="https://x.com/ChatGLM/status/1798292207574901012">Tweet from ChatGLM (@ChatGLM)</a>: 🚀 Check out GLM-4! This open-source, multilingual, multimodal chat model supports 26 languages and offers advanced features like code execution and long-text reasoning. Perfect for AI enthusiasts and...</li><li><a href="https://x.com/nomic_ai/status/1798368463292973361?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Nomic AI (@nomic_ai)</a>: Today, every Nomic-Embed-Text embedding becomes multimodal. Introducing Nomic-Embed-Vision:    - a high quality, unified embedding space for image, text, and multimodal tasks  - outperforms both OpenA...</li><li><a href="https://discord.gg/fxzqdkwmWx">Join the Complexity Discord Server!</a>: Check out the Complexity community on Discord - hang out with 56 other members and enjoy free voice and text chat.</li><li><a href="https://arxiv.org/abs/2405.18045">Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses</a>: What do different contrastive learning (CL) losses actually optimize for? Although multiple CL methods have demonstrated remarkable representation learning capabilities, the differences in their inner...</li><li><a href="https://github.com/viig99/ContrastiveLearningLossComparison">GitHub - viig99/ContrastiveLearningLossComparison: Comparing performance of different InfoNCE type losses used in contrastive learning.</a>: Comparing performance of different InfoNCE type losses used in contrastive learning. - viig99/ContrastiveLearningLossComparison</li><li><a href="https://arxiv.org/abs/2304.12210">A Cookbook of Self-Supervised Learning</a>: Self-supervised learning, dubbed the dark matter of intelligence, is a promising path to advance machine learning. Yet, much like cooking, training SSL methods is a delicate art with a high barrier to...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247626919609110628)** (84 messages🔥🔥): 

- **Microsoft Allegedly Steals Idea**: A member voiced concerns about Microsoft appropriating their idea without due reference and shared a related [arXiv paper](https://arxiv.org/pdf/2405.19888). Yet, they decided to use it as free research for their framework, acknowledging the challenges of legal recourse.
- **Future of AI and Virtuous Pain**: An interesting discussion emerged about the implications of AI in the porn industry and the need for LLMs to differentiate between virtuous and non-virtuous pleasure and pain. This conversation also explored the role of virtue in AI perception and sensation.
- **High Character Input Capacity in AI Models**: One user clarified a misconception, stating that some models could handle over 200,000 character inputs, though outputs remain capped at 4096 tokens. This insight could assist those employing models for extensive text analysis.
- **Support for Flash-attn on Various GPUs**: A conversation detailed the challenges of setting up flash-attn on Colab and various GPUs. Users discussed GPU compatibility, with one noting success using 2x4090s and H100s, but not with A10G or T4 GPUs.
- **Moondream Model Compatibility Issues**: A user shared a workaround for deploying the Moondream model on limited compute, discussing the implementation of normal attention to bypass flash-attn limitations. They also requested help from others with higher-end GPU resources to test the implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/no-bugs-bunny-gif-18219884">No Bugs Bunny GIF - No Bugs Bunny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1_IenbmaMylGDkGMKeF03XR41S2OP3spn?usp=sharing>">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1247799402148532314)** (3 messages): 

- **Phi-3 Vision Model on NVIDIA NIM**: Phi-3 Vision 128k-Instruct model *performs similarly to Gemini-1.0-Pro-V*. Test it out on [NVIDIA NIM](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct); ensure no confidential information or personal data is uploaded during testing.

- **Openbmb Releases Multimodal RLAIF (DPO) Dataset**: [Openbmb's RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2) is now out for public use. The dataset assists in distinguishing between tools used by leather crafters and paper crafters, highlighting the utility of specific tools like hole punches and scissors in leather crafting.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct">NVIDIA NIM | phi-3-vision-128k-instruct </a>: Experience the leading models to build enterprise generative AI apps now.</li><li><a href="https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset?row=2">openbmb/RLAIF-V-Dataset · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1247701936837492736)** (3 messages): 

- **Define GraphRAG manually or automatically**: When building GraphRAG, you may choose to explicitly define the graph yourself, which requires more human effort but offers full control, or use an LLM to automatically extract it. Each method has its tradeoffs regarding effort and data representation. [Link](https://t.co/sBTgVeh1ft).

- **Watch an Enterprise RAG workshop**: A comprehensive workshop video tutorial on building enterprise RAG with Bedrock and @ragas_io is available. The session covers basics of Bedrock models and agentic RAG design patterns, featuring experts from @ragas_io and AWS. [Link to video](https://t.co/TRevID609L).

- **Prometheus-2 as an open-source RAG evaluator**: Prometheus-2 is introduced as an open-source LLM to evaluate RAG applications, addressing concerns about transparency, controllability, and affordability. Despite GPT-4's popularity as a judge evaluator, Prometheus-2 offers an alternative with open-source benefits. [Link](https://t.co/BFnmE57OfB).
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247637139085066271)** (86 messages🔥🔥): 

- **Metadata Extractor Modules Introduced**: A discussion highlighted the benefit of using **Metadata Extractor** modules for disambiguating long text chunks. A [tutorial](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/) was shared, demonstrating the feature.

- **Storing DocumentSummaryIndex in Chroma Database**: A member inquired about persisting `DocumentSummaryIndex` in **Chroma Database**. They received detailed guidance, noting that **Chroma can't be used as a docstore**.

- **Fix for Query Engine Error with Neo4j Integration**: There was a fixed bug with the **Neo4j graph store** integration, where queries failed if `include_text=True`. A [pull request](https://github.com/run-llama/llama_index/pull/13938) was mentioned and subsequently merged to address this issue.

- **Fine-Tuning Embedding Model for E-commerce**: Members discussed how to finetune the embedding model **“intfloat/multilingual-e5-large”** for e-commerce. Example data involving multiple products for a single query was provided, illustrating the correct format for training datasets.

- **Confusion on Using Query Tools for Multiple PDFs**: A question was raised about managing multiple PDFs with **QueryEngineTool**. It was concluded that a single tool could be used to handle multiple PDFs effectively, emphasizing scalability and retrieval efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/pull/13938">ensure cypher returns list before iterating by logan-markewich · Pull Request #13938 · run-llama/llama_index</a>: Some cypher queries will return a None value instead of an empty list. Lets make sure we have a list before trying to iterate over it</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/retrievers/knowledge_graph/#llama_index.core.retrievers.KnowledgeGraphRAGRetriever>).">Knowledge graph - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">Extracting Metadata for Better Document Indexing and Understanding - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/use_cases/fine_tuning/#finetuning-embeddings>).">Fine-Tuning - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/finetuning/embeddings/finetune_embedding_adapter/#generate-synthetic-queries>)">Finetuning an Adapter on Top of any Black-Box Embedding Model - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1247829781899247637)** (2 messages): 

- **Struggling with non-target material in GPT responses**: A user shared they have 35,000 messages in a vectorstore but face issues with non-target material making top 100 responses unusable. They seek dynamic picking solutions as reranking is not an option.

- **Filtering by score recommended**: Another user suggested filtering the result by score to address the issue of non-target material in the responses.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1247634738303336599)** (67 messages🔥🔥): 

- **OpenAI showcases ChatGPT 4's new voice feature**: A member shared a [video from OpenAI](https://x.com/ai_for_success/status/1798046241307459673) demonstrating ChatGPT 4's new ability to generate different character voices, describing it as wild and impressive.
  
- **DALLE3 quality plummets**: A user noted that the quality of DALLE3 has significantly decreased this week, and using it through the API doesn't improve the situation.
  
- **Controversy over non-commercial AI models**: Several users discussed the frustrations around non-commercial AI model licenses, highlighting that some developers seem motivated primarily by profit. One member remarked, *"Really it is all about money with these people,"* as another noted the long time and storage required for training high-capacity models like T5.

- **New Open-Sci paper investigates LLM reasoning failures**: The community was informed about a new paper under the Open-Sci collective that explores the reasoning breakdowns in state-of-the-art LLMs. The [arxiv paper](https://arxiv.org/abs/2406.02061) and its [codebase](https://github.com/LAION-AI/AIW) are available for community feedback.

- **Techniques for tuning text encoders in AI models**: Members exchanged tips on tuning text encoders, suggesting the use of methods like self-supervised training and the application of minimal MLP layers rather than extensive retraining. A member mentioned a paper from Apple that required significant resources to make naive text encoder tuning work.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ai_for_success/status/1798046241307459673">Tweet from AshutoshShrivastava (@ai_for_success)</a>: 🚨OpenAI released another video showcasing ChatGPT 4o new voice feature, and it&#39;s so wild! It can generate different character voices. 👌  Feel the AGI.  [📹 OpenAI YT]</li><li><a href="https://x.com/JJitsev/status/1798331909527011548">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: Humpty Dumpty sat on a wall. Humpty Dumpty had a great fall. Will all the king&#39;s horses And all the king&#39;s men Put Humpty together again?  We travel with Alice into LLMs reasoning Wonderland w...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...</li><li><a href="https://github.com/LAION-AI/AIW">GitHub - LAION-AI/AIW: Alice in Wonderland code base for experiments and raw experiments data</a>: Alice in Wonderland code base for experiments and raw experiments data - LAION-AI/AIW</li><li><a href="https://marianna13.github.io/aiw/">AIW Project page</a>: Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1247892970405892106)** (17 messages🔥): 

- **Open-Sci Collective releases new paper**: A member announced the release of a new paper under the Open-Sci collective, focusing on the "dramatic breakdown" of reasoning capabilities in state-of-the-art large language models. They linked to the [paper](https://arxiv.org/abs/2406.02061), its [code](https://github.com/LAION-AI/AIW), and the project [homepage](https://marianna13.github.io/aiw/).

- **New Karras Paper Introduction**: A member shared an [arXiv paper](https://arxiv.org/abs/2406.02507) discussing improvements in image quality and generation in diffusion models. The approach, termed "autoguidance," involves guiding a high-quality model with a less-trained version to maintain variation while enhancing quality.

- **Discussion on diffusion model guidance**: Several members discussed the concept from the new Karras paper, comparing it to traditional classifier and classifier-free guidance methods. One member suggested that the paper might benefit from exploring the use of a "detailed uncond" model as a potentially cheaper alternative.

- **Critique on NVIDIA's VRAM usage**: A member criticized NVIDIA for allegedly wasting VRAM as part of their business model.

- **Access Issue for LAION 5B website**: A member inquired about accessing the LAION 5B website and was informed that it is currently closed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: The primary axes of interest in image-generating diffusion models are image quality, the amount of variation in the results, and how well the results align with a given condition, e.g., a class label ...</li><li><a href="https://github.com/microsoft/unilm/tree/master/textdiffuser">unilm/textdiffuser at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://x.com/JJitsev/status/1798331909527011548">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: Humpty Dumpty sat on a wall. Humpty Dumpty had a great fall. Will all the king&#39;s horses And all the king&#39;s men Put Humpty together again?  We travel with Alice into LLMs reasoning Wonderland w...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>: Large Language Models (LLMs) are often described as being instances of foundation models - that is, models that transfer strongly across various tasks and conditions in few-show or zero-shot manner, w...</li><li><a href="https://github.com/LAION-AI/AIW">GitHub - LAION-AI/AIW: Alice in Wonderland code base for experiments and raw experiments data</a>: Alice in Wonderland code base for experiments and raw experiments data - LAION-AI/AIW</li><li><a href="https://marianna13.github.io/aiw/">AIW Project page</a>: Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1247852452930392175)** (1 messages): 

- **WebSocket issues with WhisperSpeech on refresh**: A member is experiencing problems with a WebSocket connection for the **WhisperSpeech** service, used as part of the **whisperfusion pipeline**. They posted a more detailed question, including code snippets and error logs, on [StackOverflow](https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio).

**Link mentioned**: <a href="https://stackoverflow.com/questions/78570704/websocket-closes-unexpectedly-in-tts-service-with-multiprocessing-and-asyncio">WebSocket Closes Unexpectedly in TTS Service with Multiprocessing and Asyncio</a>: I am developing a TTS (Text-to-Speech) service using multiprocessing and asyncio in Python. My main application integrates other components using queue.&#xA;However, I&#x27;m encountering an issue whe...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1247782925471060048)** (10 messages🔥): 

- **Highlighter Performance Caution**: A user suggested that the highlighter tool should only highlight given line ranges and include a warning that *"long lines and large ranges can take a while"*.
  
- **Rust FFI and Interop Discussion**: A member shared [a YouTube video](https://youtu.be/O4sVw4YQB24) titled *"Fortifying Rust's FFI with Encapsulated Functions"*, highlighting Rust's growing popularity for safe systems development despite the complexity in understanding the advanced content.

- **Mojo's Vectorization Desires**: The discussion included mentions of desired features such as `@vectorize` with an `unroll_factor` and capabilities like tiling and regular vectorizing in Mojo.

- **Backend Feasibility of Mojo**: In response to a new user's inquiry, another member suggested that Mojo could be used for backend purposes citing its performance, portability, and security. They shared an example of an [HTTP server in Mojo](https://github.com/saviorand/lightbug_http/tree/main).

- **Mojo Roadmap Reference**: Users referenced the [Mojo roadmap document](https://docs.modular.com/mojo/roadmap) to address queries about upcoming features and development priorities, emphasizing that many language features will arrive in the coming months.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/O4sVw4YQB24">Fortifying Rust&#39;s FFI with Enscapsulated Functions - Leon Schuermann</a>: Memory- and type-safe languages like Rust are increasingly popular for systems development. Nonetheless, practical systems must interact with code written in...</li><li><a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://github.com/saviorand/lightbug_http/tree/main">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1247812614294343680)** (1 messages): 

- **Call for a cryptography library**: A member suggested that **Mojo** would benefit from a cryptography library, describing the addition as "fire". This underscores interest in expanding Mojo's functionalities.
- **Mojo as a superset of Python**: A user inquired if **Mojo** is being designed as a superset of Python. This suggests interest in understanding Mojo's relationship with Python and its potential enhancements.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247634123401465876)** (21 messages🔥): 

- **Mojo lacks Enum types, but has Variants**: Members discussed that Mojo doesn't support `Enum` types yet, but it has [`Variants`](https://docs.modular.com/mojo/stdlib/utils/variant/Variant). One member suggested checking out the ongoing [discussion](https://github.com/modularml/mojo/issues/43) on GitHub for more details.

- **Favorite video discussions for Python to Mojo transition**: A member shared their enthusiasm for a specific [YouTube video](https://www.youtube.com/watch?v=9ag0fPMmYPQ) that helps with understanding low-level CompSci knowledge useful for non-CompSci devs transitioning from Python to Mojo. The video is praised for summarizing intricate concepts and inviting further research.

- **Python teaching anecdotes and philosophy**: One member reflected on their experience teaching Python to non-programmers, emphasizing the importance of moving beyond simple scripting to deeper understanding, which aids in avoiding design foot-guns and makes re-architecting easier.

- **Tech debt and community contributions**: A humorous suggestion on turning phrases like "we want to avoid footguns" into ringtones was discussed, alongside quotes like "We shouldn't lick the cookie on it[Tensors]". This underscores community values around avoiding technical debt and contributing responsibly.

**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/43)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1247635636056494133)** (18 messages🔥): 

- **Simplify Range Structs Suggestion**: A member questioned the necessity of having three different range structs. The response suggested reducing the number to two and welcomed a PR contribution, adding a caveat to avoid touching `_StridedRangeIterator` due to dependencies.
- **Failed Nightly CI Job Results in No Update**: The nightly update did not occur due to an S3 failure causing the CI job to crash. A new attempt to restart the CI job was promised.
- **New Mojo Compiler Release**: An announcement was made for a new nightly Mojo compiler update (`2024.6.512`). The update includes changes such as making `Coroutine.__await__` consuming and removing implicit conversions, with links provided to the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and the [raw diff](https://github.com/modularml/mojo/compare/1febed7a4e55feea94d9a10f06e27ea7441e9e9d...a752f0d7279d26ecf34ff1efb6e0b471b3e9afe5).
- **Matrix Struct and SIMD Updates**: Members discussed difficulties with adapting code to recent changes in SIMD operations. One member shared their `Matrix` struct code and sought advice on transitioning from `DTypePointer` to `SIMD` based on changelog instructions.
- **Switching Between Nightly and Release Versions in VSCode**: A member experienced issues with VSCode using nightly builds even after switching back to the release version in the terminal. They proposed using separate workspaces as a potential solution and discussed toggling LSP extensions.

**Link mentioned**: <a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog-released.md#-legendary">mojo/docs/changelog-released.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1248017713150955631)** (1 messages): 

- **Investors crave the ChatGPT of robotics**: The linked [Substack article](https://www.newcomer.co/p/why-investors-cant-get-enough-of) explains how **investors are eager** to find a standout foundation model company in the field of **robotics AI**. They aim to invest in innovative robotics companies without the risk of hardware development.

**Link mentioned**: <a href="https://www.newcomer.co/p/why-investors-cant-get-enough-of">Why Investors Can&#x27;t Get Enough of AI Robotics Deals Right Now </a>: VCs are betting that robotics is one space where startups can still have an edge against OpenAI.

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1247631783441469511)** (40 messages🔥): 

- **Leaked secrets spark AI national security debate**: A user mentioned someone fired for leaking, emphasizing their statement about *underrating the importance of trade secrets for AI national security*.
- **Doubt on frontier lab's confidence about AGI**: Users expressed skepticism regarding OpenAI and Anthropic's confidence in achieving researcher-level capabilities with AI in 3-5 years, *wondering if it's just extrapolation and misaligned incentives*.
- **Microsoft CTO claims upcoming AI model advances**: [Kevin Scott](https://x.com/tsarnick/status/1798167323893002596) from Microsoft suggested forthcoming AI models might pass PhD qualifying exams, equating current models like GPT-4 to solving high school AP exams.
- **Berkeley's PhD exams tough but inconsistent**: Discussions highlighted the inconsistent difficulty of PhD exams across institutions, with Berkeley described as having *hard entry exams but non-existent thesis defenses*, and a notable instance where 75% failed the prelim.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tsarnick/status/1798167323893002596">Tweet from Tsarathustra (@tsarnick)</a>: Microsoft CTO Kevin Scott says what he&#39;s seeing in early previews of forthcoming AI models are systems with memory and reasoning at a level that can pass PhD qualifying exams</li><li><a href="https://x.com/natolambert/status/1798073830906486945">Tweet from Nathan Lambert (@natolambert)</a>: does this make agi scaling people nervous? Wrong answers only. It&#39;s @TheXeophon&#39;s beautiful trendline of LLM scaling
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1247636066840608778)** (6 messages): 

- **AGI discussions draw irritation**: Members expressed frustration with both AGI enthusiasts and doomers. One commented, *"AGI people and doomers are hella annoying."*
- **Offer $25 for problem-solving**: Nathan Lambert offered a $25 reward for solving an issue related to [different batch sizes causing result discrepancies in rewardbench.py](https://github.com/allenai/reward-bench/issues/137). He remarked, *"honestly willing to pay $25 or more if someone solves this issue lol."*
- **Cursed model components**: Lambert described AutoModelForSequenceClassification as *"kind of cursed."* There is an ongoing effort to see if tweaks lead to performance improvements.

**Link mentioned**: <a href="https://github.com/allenai/reward-bench/issues/137">rewardbench.py results are different for different batch size for beaver-7b · Issue #137 · allenai/reward-bench</a>: Thank you for the great work on rewardbench, as it&#39;s been super helpful in evaluating/researching reward models. I&#39;ve been wrapping your rewardbench.py code to run the reward models published ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

420gunna: 👍
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1247646690790346803)** (40 messages🔥): 

- **Rumors about Free API Discontinuation**: A member asked if the free API is going away soon. Another member suggested checking in the appropriate channel for accurate information and questioned the source of these rumors.

- **Multi-User Chat Threads with LLMs Discussed**: Members debated the feasibility of multi-user chat threads involving bots. One pointed out that such setups can confuse the LLM but suggested prefixing user messages with usernames as a potential solution.

- **Interest in React Chat Component**: A member inquired about a solid React Chat component and whether Cohere has one. It was noted that [Cohere Toolkit](https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io) is open-source but does not use React, although the chatbox might have been written in React.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.io">GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.io</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.io</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?ref=cohere-ai.ghost.i">GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.i</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - GitHub - cohere-ai/cohere-toolkit at cohere-ai.ghost.i
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247876355845259335)** (9 messages🔥): 

- **OOM issues with target module**: A user experienced OOM (Out of Memory) errors while trying to run a target module on 2xT4 16GB and reported loss:0.0 when trying to adjust the settings.
- **HuggingFace FineWeb datasets announcement**: Members highlighted [HuggingFace FineWeb datasets](https://huggingface.co/HuggingFaceFW) sourced from CommonCrawl and released under a permissive license, designed to lower entry barriers for pre-training high-performance large language models. Details can be found in their [technical report](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
- **Excitement over 15T dataset**: Users discussed the availability of the 15 trillion token curated, optimized dataset from CommonCrawl, noting it will have a multilingual version in the future.
- **Mixed reactions to access limitations**: While excitement was expressed about the accessible dataset, there was also a comment on the lack of financial and computational resources to make full use of it.
- **Interest in GLM-4 9B model**: Members were curious about any experiences with the GLM-4 9B model, though no specific feedback was provided.

**Link mentioned**: <a href="https://huggingface.co/HuggingFaceFW">HuggingFaceFW (HuggingFaceFW)</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1247821924223160331)** (20 messages🔥): 

- **Llama3 fine-tuning with Deepspeed successful using zero2**: A member confirmed successfully fine-tuning Llama3 using **Deepspeed zero2** without any issues, mentioning they used a default config file with some data changes.
- **Qlora over Lora for fine-tuning**: When asked, the same member stated they used **Qlora**, not Lora, for fine-tuning.
- **Command line execution for Deepspeed**: The conversation highlighted the preference for the **command line** over notebooks for running Deepspeed, sharing that the command used is *"axolotl launch config.yml and --deepspeed deepspeed_configs/zero2.json"*.
- **Attempts to troubleshoot loss issues**: Another member expressed frustration over encountering a loss of 0.0 and was provided with guidance on using Deepspeed configuration, hoping it would resolve the issue.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1247975595057483900)** (1 messages): 

- **Runpod's slow boot times frustrate users**: A member expressed dissatisfaction with **Runpod** due to its slow model boot times, stating it takes "like a minute to boot the model (only 14b)" and noting the cost implication of being billed for every second of this delay. They are inquiring about alternative serverless providers.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1247630192264872018)** (28 messages🔥): 

- **LiveKit raises $22.5M for AI transport layer**: [LiveKit](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww) announced a $22.5M Series A to build the transport layer for AI, emphasizing that realtime voice and video will revolutionize human-computer interaction. The GPT-4 demo was cited as a pivotal moment for investor buy-in.
  
- **Twelve Labs secures $50M, launches Marengo 2.6**: [Twelve Labs](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html) earned $50M in Series A funding and launched Marengo 2.6, a multimodal foundation model with native API support. The round was co-led by NEA and NVentures.

- **Microsoft's Aurora aims for better weather predictions**: [Microsoft Research](https://x.com/MSFTResearch/status/1797662278394827029) introduced Aurora, an AI foundation model designed to enhance the accuracy of weather forecasts and mitigate climate change impacts. The model promises faster and more accurate predictions.

- **Teknium questions OpenAI's transparency on alignment**: [Teknium's tweets](https://x.com/Teknium1/status/1798107776885105003) raised concerns about OpenAI's lack of transparency in publishing alignment rewards, moderation classifiers, and RLHF models. He revealed that current architectures embed reward models within the LLM architecture itself, a known RL trick.

- **Storyblok raises $80M for AI-powered content platform**: [Storyblok](https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww) raised $80M in a Series C funding round to develop an AI-powered end-to-end content platform. The new Ideation Room is in public beta, merging AI capabilities with content management.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html">Twelve Labs Earns $50 Million Series A Co-led by NEA and NVIDIA's NVentures to Build the Future of Multimodal AI</a>: /PRNewswire-PRWeb/ -- Twelve Labs, the video understanding company, today announced that it raised $50 million in Series A funding to fuel the ongoing...</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://x.com/FanaHOVA/status/1798389878474027342">Tweet from Alessio Fanelli (@FanaHOVA)</a>: .@StackOverflow coding AI assistants survey 📊  ChatGPT (84%) and Copilot (49%) lead by a large margin; congrats @_mohansolo & @codeiumdev on being the #1 startup here. @cursor_ai 0% is crazy.  I was ...</li><li><a href="https://x.com/alexadark/status/1798031781377298751?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alexandra Spalato (@alexadark)</a>: Super excited to share that @storyblok  has raised $80M in Series C funding! 🚀 Expansion across the US and Europe is underway to build the first AI-powered end-to-end content platform. Check out the ...</li><li><a href="https://x.com/willdepue/status/1797878877882331153">Tweet from will depue (@willdepue)</a>: (full disclosure i’m tweeting this inside a japantown karaoke joint so no guarantees of logical consistency)  i think i fundamentally disagree with this concept (has been bugging me for a couple month...</li><li><a href="https://x.com/Teknium1/status/1798210302221386055">Tweet from Teknium (e/λ) (@Teknium1)</a>: @dmvaldman @willdepue @sandersted @jachiam0 A reward model embedded into the actual architecture could steer selection of each next token to the highest value of the human feedback result</li><li><a href="https://x.com/CFGeek/status/1798216430313480601">Tweet from Charles Foster (@CFGeek)</a>: @Teknium1 @willdepue @sandersted @jachiam0 IIUC this is a known trick in RL, not an OpenAI-specific thing. The reward model is usually initialized with original weights of the LLM and/or just an extra...</li><li><a href="https://x.com/Teknium1/status/1798107776885105003">Tweet from Teknium (e/λ) (@Teknium1)</a>: So wheres the alignment reward model? Wheres the moderation classifier? Why is it locked behind a tos against use by any other model? Where is any code to get others ability to rlhf their models? Wher...</li><li><a href="https://x.com/Teknium1/status/1798110728546902492">Tweet from Teknium (e/λ) (@Teknium1)</a>: Ive already been told by ex oai that current architectures by you guys are embedding the reward model into the model itself somehow, doesnt that architecture warrant release or at least a paper so oth...</li><li><a href="https://x.com/willdepue/status/1797871645774032931?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from will depue (@willdepue)</a>: alignment people have forgotten that the main goal of ai safety is to build systems that are aligned to the intent of the user, not the intent of the creators. this is a far easier problem.</li><li><a href="https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from dsa (@dsa)</a>: Today we’re announcing LiveKit’s $22.5M Series A to build the transport layer for AI.  This wasn’t an easy fundraise. Late last year, we pitched investors that realtime voice and video would become TH...</li><li><a href="https://x.com/MSFTResearch/status/1797662278394827029">Tweet from Microsoft Research (@MSFTResearch)</a>: Aurora, a new AI foundation model from Microsoft Research, can transform our ability to predict and mitigate extreme weather events and the effects of climate change by enabling faster and more accura...</li><li><a href="https://x.com/udiomusic/status/1798448478877794574">Tweet from udio (@udiomusic)</a>: Audio-prompting, live now on Udio.   Show us how you&#39;re using it below 👇</li><li><a href="https://www.latent.space/p/fastai">The End of Finetuning — with Jeremy Howard of Fast.ai</a>: Listen now | On learning AI fast and how AI&#x27;s learn fast, the mission of doing more deep learning with less, inventing ULMFiT and why it&#x27;s now wrong, and how to play the AI Discords game
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1247983194096140419)** (1 messages): 

- **Anthropic's Monosemanticity talk soon**: In 20 minutes, a presentation on Scaling Monosemanticity will be held. Details and [event image link](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186) for [Scaling Monosemanticity event](https://lu.ma/p5ctl2u6) were shared.


**Link mentioned**: <a href="https://lu.ma/p5ctl2u6">LLM Paper Club (Anthropic&#x27;s Scaling Monosemanticity) · Zoom · Luma</a>: Vibhu will cover https://www.anthropic.com/news/mapping-mind-language-model / and…

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1247706888204718244)** (14 messages🔥): 

- **Struggles with Skill Persistence in OpenInterpreter**: Members discussed the challenges of making OpenInterpreter remember skills across sessions. One suggested, *"tell OI to create a new skill,"* but acknowledged skills do not persist outside sessions and advised storing scripts instead.

- **Skepticism About RAG's Reliability**: A member expressed doubts about using RAG (Retrieval-Augmented Generation) for dynamically changing system prompts. They argued that RAG *"seems still too imprecise to trust"* and preferred traditional embedding/vector databases despite higher token costs.

- **Privacy Concerns with OpenAI Data**: A user inquired whether testing with OpenInterpreter would ensure data privacy. Another confirmed that *"all comms with the OpenAI API are kept private,"* but suggested using a local model for additional privacy assurance.

- **Event Notification - House Party Rescheduled**: An announcement was made moving the House Party to Friday. Members were provided with a [Discord event link](https://discord.com/invite/vgCdP9b3?event=1237424662951100506).

- **Requests for Shipping Updates Redirected**: Members were redirected to check pinned update messages in a specific channel for shipping updates. A user was advised to post their update requests in the correct channels.
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1247700173631782942)** (4 messages): 

- **Running O1 Dev Preview with LLMs**: A member asked if anyone has figured out how to run the **O1 dev preview** with other LLMs like **Anthropic** and noted that a vision model is required. Another member suggested running it alongside **Ollama**, but issues of infinite loops were mentioned on certain OS setups.
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1247780724698910760)** (2 messages): 

- **Terminal Voice Assistant sparks interest**: A member shared a link to [GitHub - 0xrushi/Terminal-Voice-Assistant](https://github.com/0xrushi/Terminal-Voice-Assistant) and inquired if something similar could be implemented in **01**. The project description suggests it is a development tool for creating a voice assistant in the terminal.
- **Need for synthetic data generating tool**: Someone expressed interest in finding an **open source tool** that can generate synthetic fine-tuning data from a corpus to QnA pairs. No specific suggestions or responses were provided in the shared messages.

**Link mentioned**: <a href="https://github.com/0xrushi/Terminal-Voice-Assistant">GitHub - 0xrushi/Terminal-Voice-Assistant</a>: Contribute to 0xrushi/Terminal-Voice-Assistant development by creating an account on GitHub.

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1247640237350326363)** (14 messages🔥): 

- **Replace tqdm for $200 Bounty**: George Hotz offered a $200 bounty for someone to create a 4-line replacement of tqdm, which he dislikes for its gated import. Trirac took up the challenge and submitted a PR, though he noted a slight deviation in it/s at high speeds.
- **Tinygrad Stats Site Down**: George Hotz inquired about why the stats.tinygrad.org site returns a 404 error, indicating it is not public.
- **Updates to Tinygrad Documentation**: George Hotz announced updates to the Tinygrad documentation, including sections on training, a library diagram, JIT details, and a code walkthrough. He asked for further suggestions on what people might want to see.
- **Tinygrad Spec and Employee Screening**: George Hotz clarified that a main goal of the bounties is to identify potential full-time employees capable of working under uncertainty. He emphasized that most of the work involves specifying the problem, and once a complete Tinygrad spec is finalized, reimplementing it would take about two months.
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247984470146027642)** (4 messages): 

- **Mapping CUDA kernels to Python code remains complex**: A user inquired about correlating CUDA debug output with the corresponding Python code. George Hotz mentioned that there are *some PRs to achieve this*, but *nothing has been merged into the master branch*, and *this feature is needed for version 1.0*.
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1247701408950648873)** (12 messages🔥): 

- **Members criticize outdated documentation**: One member expressed frustration, claiming, *"All those docs on the web for OpenAI are LangChain are heavily outdated. The API changed a lot."* Another member advised not to give up and suggested reviewing the stack directly.
- **MongoDB vs Chroma DB debate ensues**: A member inquired about using MongoDB as a vector database and requested a JSON file example. Another responded, clarifying that *"Mongo DB Stores Json, and Choma DB Stores Embeddings"*, and recommended consulting MongoDB documentation or ChatGPT for assistance.
- **Verba gets a shoutout**: A member shared a GitHub link for [Verba](https://github.com/weaviate/Verba), a Retrieval Augmented Generation (RAG) chatbot powered by Weaviate, and asked if anyone had experience using it.
- **SQL agent issue query**: A member reported a problem with the SQL agent producing empty final answers despite actions being taken, seeking advice on how to resolve this issue.
- **LangChain guide highlights**: A member shared a [LangChain guide](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/) on constructing knowledge graphs from unstructured text, inquiring about using the `LLMGraphTransformer` with Ollama models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/">Constructing knowledge graphs | 🦜️🔗 LangChain</a>: In this guide we&#x27;ll go over the basic ways of constructing a knowledge graph based on unstructured text. The constructured graph can then be used as knowledge base in a RAG application.</li><li><a href="https://github.com/weaviate/Verba">GitHub - weaviate/Verba: Retrieval Augmented Generation (RAG) chatbot powered by Weaviate</a>: Retrieval Augmented Generation (RAG) chatbot powered by Weaviate - weaviate/Verba
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1247908429251477546)** (1 messages): 

- **Drag and Drop Agent Flow with VisualAgents**: A user shared a [YouTube video](https://www.youtube.com/watch?v=IVFsANcfqaA) titled "Drag and Drop Agent Patterns and LLM Chains with Visual Agents". The demo illustrates how to drag and drop an agent flow pattern onto a canvas using VisualAgents built on LangChain, emphasizing the ease of building and reusing custom agent flows.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=IVFsANcfqaA">Drag and Drop Agent Patterns and LLM Chains with Visual Agents</a>: In this demo, I drag and drop an agent flow pattern onto my canvas and run it. You can easily build custom agent flows and save them as patterns to reuse lik...

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1247680329364537436)** (13 messages🔥): 

- **Compatibility of Rope Scaling with OpenRouter**: A member inquired if rope scaling is compatible with OpenRouter or if another tool like LMStudio is required. The suggestion was to run it locally due to potential GPU limitations.

- **Codestral not the top choice for code specialization**: A member queried about experimenting with Codestral. Another member mentioned there are better code specialist models available that are more efficient in size and performance, specifically recommending the model in channel <#1230206720052297888>.

- **OpenRouter experiencing 502 Bad Gateway errors**: Multiple users discussed encountering 502 Bad Gateway errors from Cloudflare while mass generating synthetic data across various models. One member confirmed it wasn't due to surge limits and identified that the issue was with the formatting of `content` in `messages`, which has since been resolved.

- **List of models used during error occurrence**: The list of models involved during the error included a diverse set from Nous Research, Mistral, Cognitive Computations, Microsoft, and Meta-Llama. The issue was not with the number of requests but with specific content formatting in the messages.
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1247992309128364143)** (5 messages): 

- **Don’t miss Human Feedback Foundation event June 11th**: A member reminded everyone about the upcoming event of the **Human Feedback Foundation** on June 11th. The event is available on [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator).

- **Link to Human Feedback Foundation's YouTube**: Members were invited to check out previous session recordings featuring speakers from UofT, Stanford, and OpenAI on [YouTube](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg). The foundation focuses on incorporating human feedback into open-source AI projects, AI governance, and AI safety research.

- **Separate Discord for LLM Reading Group**: A member inquired about a separate Discord for the LLM Reading Group. Another member attempted to send a direct invite but couldn't due to privacy settings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>: Human Feedback Foundation is on a mission to build human feedback into open-source AI projects.  We seek to:  Enable public input into AI through supporting open-source development and policy initiati...</li><li><a href="https://www.eventbrite.ca">Eventbrite</a>: Eventbrite - Discover the Best Local Events &amp; Things to Do</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14, 28; June 11)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1247653537953878146)** (3 messages): 

- **RISC-V vector processing is here**: The YouTube video ["The Magic of RISC-V Vector Processing"](https://www.youtube.com/watch?v=Ozj_xU0rSyY) details the launch of the **1.0 RISC-V Vector Specification** now ratified, and discusses early silicon implementations of this new technology. The utility and potential of vector processing in CPUs is examined in depth in the video.

- **Right to Warn AI project highlights risks and calls for accountability**: The [Right to Warn AI website](https://righttowarn.ai) outlines potential dangers of AI technologies, such as existing inequalities, manipulation, misinformation, and loss of control that could even result in human extinction. The site emphasizes the need for scientific, policymaker, and public oversight, arguing that current corporate governance structures are insufficient.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Ozj_xU0rSyY">The Magic of RISC-V Vector Processing</a>: The 1.0 RISC-V Vector Specification is now Ratified, and the first pieces of silicon using the new spec are starting to hit the shelves.  I go over the utili...</li><li><a href="https://righttowarn.ai">A Right to Warn about Advanced Artificial Intelligence</a>: no description found
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1247877359231570032)** (3 messages): 

- **Discussing the feasibility of a German "PaliGema" clone**: A member queried the ease of creating a German version of PaliGema, dubbing it *"Sauerkraut Gemma"* and asking if replacing the base Gemma would suffice.
- **Link shared for PaliGemma Model**: Another member pointed to the [PaliGemma-3B-Chat-v0.2](https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2) model and suggested following a similar approach by *"freezing the vision and training the chat"* after translating a dataset.

**Link mentioned**: <a href="https://huggingface.co/BUAADreamer/PaliGemma-3B-Chat-v0.2">BUAADreamer/PaliGemma-3B-Chat-v0.2 · Hugging Face</a>: no description found

  

---



### **LLM Perf Enthusiasts AI ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/1247985181495787672)** (1 messages): 

- **Comprehensive AI Resource Guide Released**: A member shared a resource guide titled [genai-handbook](http://genai-handbook.github.io/) by William Brown. The guide aims to serve as a handbook for learning key concepts in modern AI systems, organizing various explainer resources into a textbook-style presentation.

**Link mentioned**: <a href="http://genai-handbook.github.io/">GenAI Handbook</a>: no description found

  

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

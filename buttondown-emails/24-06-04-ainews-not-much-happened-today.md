---
id: b67a635d-1bfc-4e15-bbb2-e7c543f37bde
title: '[AINews] Not much happened today'
date: '2024-06-04T23:53:47.552835Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-not-much-happened-today-5500
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 6/3/2024-6/4/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**400** channels, and **4568** messages) for you. 
Estimated reading time saved (at 200wpm): **455 minutes**.

Twelve Labs [raised $50m](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html), Livekit [raised $22m](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww), [Groq is now running 800tok/s](https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy63D2OtiaoGJuww), and there's an OpenAI resignation [thread from Daniel Kokotajlo](https://x.com/dkokotajlo67142/status/1797994238468407380?s=46&t=JE84TqLviekDnEt8MAT-Eg).

But no technical developments caught our eye.

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


**AI and Large Language Model Developments**

- **Gemini model performance**: [@_arohan_](https://twitter.com/_arohan_/status/1798001375462432901) highlighted the Gemini 1.5 FlashModel as an outlier providing high performance at low cost, making useful models accessible to more users. He also [noted](https://twitter.com/_arohan_/status/1797785953890676771) Gemini Pro taking the #2 spot in Japanese language performance.
- **Optimizing Mixtral models with TensorRT**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797683462633402646) shared how Mixtral models can be made to run up to 8x faster on NVIDIA RTX GPUs using TensorRT-LLM, which compiles the model and optimizes kernels for efficient serving, supporting expert and tensor parallelism.
- **Mamba-2 model architecture**: [@tri_dao](https://twitter.com/tri_dao/status/1797650443218436165) and [@_albertgu](https://twitter.com/_albertgu/status/1797651223035904355) introduced Mamba-2, which uses state space duality to enable sequence models with 8x larger states, 50% faster training, and connections between SSMs and linear attention, outperforming Mamba-1 and strong Transformer architectures.
- **Phi-3 model benchmarks**: [@_philschmid](https://twitter.com/_philschmid/status/1797700161226838362) reported that Phi-3 Medium (14B) and Small (7B) models are on the @lmsysorg leaderboard, with Medium close to GPT-3.5-Turbo-0613 but behind Llama 3 8B, and Small near Llama-2-70B and Mistral fine-tunes, suggesting optimizing only for academic benchmarks is not enough.

**Prompt Engineering and Data Curation**

- **Power of prompt engineering**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797988953389457475) emphasized the power of prompting LLMs correctly to enable capabilities like jailbreaking, adhering to JSON schemas, grounding and more by navigating the latent space.
- **Importance of data quality**: [@sarahcat21](https://twitter.com/sarahcat21/status/1797639227188170882) pointed out that models perform better when trained on good data, making data curation critical. [@HamelHusain](https://twitter.com/HamelHusain/status/1798015536279990740) promoted an upcoming masterclass on organizing and generating high quality data for fine-tuning.

**AI Safety and Alignment Discussions**

- **Frontier AI lab employee letter on safety disclosures**: [@jachiam0](https://twitter.com/jachiam0/status/1798013509978210431) shared thoughts on a circulating letter from former and current frontier AI lab staff advocating for whistleblower protections on safety and risk issues, arguing it could disrupt trust and make sensitive internal discussions harder.  
- **Aligning AI to user intent vs humanity's interests**: [@willdepue](https://twitter.com/willdepue/status/1797871645774032931) argued alignment should focus on the easier problem of aligning AI to the user's intent rather than the intent of creators or benefit of all humanity. However, [@jachiam0](https://twitter.com/jachiam0/status/1797874200058978786) and [@Teknium1](https://twitter.com/Teknium1/status/1797979400526581833) countered that AI could become autonomous and not serve user interests, necessitating global alignment.



---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Finetuning and Optimization for LLMs**:

   - **[Optimizing LLM Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy)** by OpenAI provides advanced techniques like prompt engineering, RAG, and guidelines on acceptable performance levels. Check out the accompanying **[YouTube talk](https://www.youtube.com/watch?v=ahnGLM-RC1Y)** for deeper learning.
   
   - Discussing **Multimodal Finetuning**, users explored **Opus 4o** and **MiniCPM-Llama3-V-2_5** for image text parsing and OCR, and considered retrieval methods for structured datasets ([Countryside Stewardship grant finder](https://www.gov.uk/countryside-stewardship-grants)).

   - Queries about **continuous pretraining** and memory efficiency highlight Unsloth AI's ability to halve VRAM usage compared to standard methods, detailed in their **[blog](https://unsloth.ai/blog/contpretraining)** and **[GitHub](https://github.com/unslothai/unsloth)** page.

2. **Model Performance and Inference Efficiency**:

   - Modal impressed with **50x revenue growth** and revenue exceeding eight figures while also optimizing infrastructure. Insights were shared in **[Erik's talk at Data Council](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure)** and **[Modal's hiring link](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34)**.
   
   - Discussions about **bitshift operation across all backends** (tinygrad) and performance adjustments ([PR #4728](https://github.com/tinygrad/tinygrad/pull/4728)) versus traditional operations stirred debates on improvement margins.
   
   - Users tackled **CUDA recompile issues** by realigning flags for **effective compilation**. They exchanged resources like the **[RISC-V Vector Processing YouTube video](https://www.youtube.com/watch?v=Ozj_xU0rSyY)** for further learning.

3. **Open-Source Developments and Community Projects**:

   - LlamaIndex's integration with **[Google Gemini](https://t.co/Qg9ydPzBdd)** demonstrated a million-token context window facilitating complex queries, while practical problems were solved via custom solutions detailed in their **[documentation](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations)**.

   - Modlab’s **[Deep Dive into Ownership in Mojo](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo)** showcased detailed work by CEO Chris Lattner exploring developer-friendly innovations. Community feedback on making all functions `async` sparked diverse opinions on compatibility and ease of transition.
   
   - Projects like **[FineWeb](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)** from Hugging Face and the **[Phi-3 models](https://fxtwitter.com/_philschmid/status/1797700161226838362)** climbing the @lmsysorg leaderboard highlight progress and ongoing research in open-source AI.

4. **System and Hardware Troubleshooting**:

   - Members resolved several technical issues, such as **infinite loops on Macbook M1 with ollama llama3 setup** by troubleshooting system commands, and **Async processing in LM Studio** facilitated with practical discussions on GPU usage efficiency.
   
   - They discussed performance discrepancies in GPUs (e.g., **6800XT** achieving only 30it/s) and potential improvements with proper setup and driver considerations, showcasing a blend of peer support and technical expertise.
   
   - Open-source solutions like **[IC-Light](https://github.com/lllyasviel/IC-Light)**, focusing on improving image relighting, and **CV-VAE** for video models ([ArXiv link](https://arxiv.org/abs/2405.20279)) were enthusiastically shared among hardware and software enthusiasts.

5. **Health of AI Communities and Conferences**:

   - Several platforms confirmed **credit distributions** to users, dealing with issues such as double credits, while fostering a supportive environment seen in community exchanges and career stories.
   
   - Events like **Qwak's Infer: Summer '24** invite AI/ML enthusiasts for practical sessions with industry experts, further detailed in **[conference registration](https://tinyurl.com/j8z6s8ka)**. 

   - **AI News newsletters** faced formatting issues in ProtonMail dark mode, encouraging community-led problem-solving, and events like **Torchtune seeking recognition** highlighted active engagements and the importance of visibility in community contributions.

---

# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Optimization Adeptness for LLM Smarts**: OpenAI shared an advanced guide on optimizing LLMs for better accuracy, which includes techniques like prompt engineering, RAG, and fine-tuning, along with deciding acceptable performance levels for real-world applications. The guide is accessible at [Optimizing LLM Accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy), and for deeper learning, check out the YouTube talk "A Survey of Techniques for Maximizing LLM Performance" [here](https://www.youtube.com/watch?v=ahnGLM-RC1Y).

- **Credits Galore Across the LLM Landscape**: Across the guild, several platforms including **Hugging Face**, **Replicate**, **Modal**, **Predibase**, and **Braintrust** are confirming the distribution of credits to users. Issues like double credits and missing credits are being addressed, with indications that users should contact support or check their billing settings for confirmation. Users are also advised to follow up on pending platforms for credit distribution.

- **All Aboard Fine-Tuning Innovations**: Lively discussions revolve around fine-tuning LLMs, exploring multimodal finetuning, leveraging **Opus 4o** and **MiniCPM-Llama3-V-2_5** for parsing text from images, and using **PaliGemma** for OCR tasks. Tear into model merging tactics with **Axolotl**, [here](https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base), and pick apart **Medusa** and **LOra**'s ability to enhance LLM inference, [here](https://arxiv.org/abs/2401.10774) and [here](https://github.com/predibase/lorax). Users suggested retrieval applications could be fruitful for structured government data, such as dataset details found at [Countryside Stewardship grant finder](https://www.gov.uk/countryside-stewardship-grants).

- **Community Exchange and Shared Journeys**: From discussions on CUDA book recommendations to tales of transitioning from academia to freelancing, or novices' learning journeys to AI, there's a buzz of community empowerment and peer guidance. Shared experiences underscore the importance of learning and adapting, as seen in diverse career paths involving freelancing, industry R&D, and game achievements.

- **Inferential Framework Finesse**: Talk of efficient LLM inference included Modal's infrastructure optimization, with a dive into filesystem and container runtime custom solutions by Erik at Data Council, available [here](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure). Modal also stirred interest with their eight-figure revenue growth and ongoing hiring, detailed further at [Modal's hiring link](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34). Etched's faster chip for running transformers more than 10x faster than GPUs signaled an engineering leap, with job openings accessible via [Etched's hiring link](https://boards.greenhouse.io/etchedai).

- **Deployment Decoded and Credits Decrypted**: From error resolution in Modal's hello world example to embedding models, engineers dissect the deployment intricacies and swoop on credit opportunities—Charles from Modal dished extra $500 credits, expounded upon at [Modal's blog](https://modal.com/blog/embedding-wikipedia). Meanwhile, predibase users report fine-tuning oddities and credit discrepancies, exploring if adapters could trim nonsensical continuations generated by the L3 70B base model.

While credits fueled the AI engine rooms and technical tidbits circulated, members swapped both assistance and anecdotes—an emblem of the guild's pulsing core of collective progress and exchange.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**CUDA Conundrums and Triton Tips**: Users discussed the tech used for generating digital humans without concluding, and sought **efficient LLM training** methods on multiple GPUs. Challenges were noted in Triton for **indexing a tensor in shared memory**, and advice on Triton and Torch was provided for those considering a switch to CUDA/Triton programming.

**Torch Troubleshooting and Profiling Proficiency**: Users shared on debugging NHWC tensor normalization, opening metal traces using the **torch.mps.profiler**, and sought to understand `torch.compile` along with its child function calls.

**AO's Arrival and Sparsity Specs**: News emerged about an **Apple Metal kernel** and **2:4 sparsity benchmarks** contributing to PyTorch AO, sparking debates on torch.ao.quantization deprecation and discussing the efficiency of structured pruning.

**Blog Browsing and Binary Banter**: A mention of *State Space Duality* delved into on [goomblog](https://goombalab.github.io/blog/), while discussions flourished around PyTorch's `uint2-7` types and custom dtype string-conversion for `TrinaryTensor`.

**ARM's Acceleration Aspirations**: Conversation revolved around the capabilities and support of ARM for Hexagon SDK and Adreno SDK, with a member sharing resources on ARM's performance and discussing its potential in GEMM implementations.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM Vanquished by Token Increase**: Extending `llama-3-8b` to **64k tokens** caused an OutOfMemoryError on an H100 with 80GB VRAM; discussions aimed to resolve this through gradient checkpointing and tuning configurations.
  
- **Speedy Sustained LLM Pretraining**: Unsloth AI’s new update allows for **doubling the speed** and **halving the VRAM** usage compared to Hugging Face + Flash Attention 2 QLoRA during continuous pretraining of LLMs, as discussed in [their recent blog](https://unsloth.ai/blog/contpretraining).

- **Questions on Multi-GPU and 8-bit Optimization**: The community actively engaged in conversations about multi-GPU support and testing Unsloth AI’s performance on different GPU configurations, while addressing the current limitations of fine-tuning with 8-bit quantization on models like phi-3-medium-4k.

- **Unsloth Setup and Optimization Tactics**: Instructions and troubleshooting tips for local Unsloth setup were shared, including the use of Jupyter Notebook and Docker, with links to GitHub [readme](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions) and [Jiar/jupyter4unsloth](https://github.com/Jiar/jupyter4unsloth). The community also covered LoRA rank calculation, referencing insights from [Lightning AI](https://lightning.ai/pages/community/lora-insights/).

- **Community Cordiality Continues**: New members were warmly welcomed into the community, fostering a supportive environment for collaboration and knowledge exchange.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Wikipedia Bias in Academic Searches**: A user highlighted potential issues with **Perplexity's academic search** capabilities, pointing out a bias towards Wikipedia over other sources like Britannica, and provided a [link to the search results](https://www.perplexity.ai/search/I-want-to-ZoV4zN4LRKa2YcbFG52K.Q).

**AI Services Experience Simultaneous Downtime**: Reports emerged of simultaneous outages affecting **Perplexity**, **ChatGPT**, and similar AI services, spurring discussions about a larger infrastructure issue possibly connected to common providers like **AWS**.

**The Opus 50 Limit has Users Longing for More**: Users expressed dissatisfaction with the new **Opus 50** limit, comparing it unfavorably to the previous **Opus 600** and criticizing Perplexity's communication about it.

**Perplexity vs. ChatGPT: A Duel of AI Titans**: Discussions around the pros and cons of **Perplexity AI Premium** and **ChatGPT** touched on web search capabilities, model range, subscription limits, and practical use cases for both platforms.

**Tech Talk Assistance for School**: AI enthusiasts shared resources and advised using AI tools to assist with school presentations on AI, highlighting the need to explain both benefits and risks, along with sharing a [YouTube video](https://youtu.be/wjZofJX0v4M) for technical understanding.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Fine-Tuning Fervor**: The community discussed the advantages of fine-tuning models like **Mistral 7B Instruct** on customized datasets for improved instruction following. They also explored **Vision-Language Models (VLMs)** and challenges in integrating visual data with language models, emphasizing the need for alignment with tokenizers and datasets suitable for specific tasks ([Vision-Language Modeling Primer](https://huggingface.co/papers/2405.17247)).

- **Enhancing Image Generation via Corruption**: A collaboration with Microsoft and CMU resulted in a [study](https://arxiv.org/abs/2405.20494) highlighting the impact of slight corruption in pre-training data on the quality of diffusion models. Alternatively, a blog post discussed [Diffusion Policy](https://radekosmulski.com/how-to-train-your-robot-with-a-transformer/), a visuomotor learning algorithm that begins with Gaussian noise to predict actions, underlining its novel approach to generating actions through an encoder-decoder model.

- **New Tools and Pipelines**: Hunyuan DiT pipeline was added to the `diffusers` library, providing a fresh method for image and audio generation ([Diffusers Release Notes](https://github.com/huggingface/diffusers/releases)). Moreover, the community was invited to improve **LevelBot's new activity tracker** by integrating additional actions and activities like GitHub contributions into its system ([LevelBot Activity Tracker](https://huggingface.co/posts/lunarflu/239147617114976)).

- **Optimizing ML Workflows**: There's active engagement in improving model inference efficiency, with discussions on utilizing `jit.trace` for SDXL and other optimization tips found in the [Diffusers optimization guide](https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing). Furthermore, troubleshooting included the use of explicit function imports to resolve potential version conflicts.

- **Dataset and Algorithm Revelations**: Novel datasets are being shared, like the German parliament speech data for ASR/TTS ([Bundestag ASR Dataset](https://huggingface.co/datasets/D4ve-R/bundestag-asr)). Additionally, a focus on preference alignment was highlighted in the [Alignedge paper](https://arxiv.org/abs/2403.07691), introducing the ORPO algorithm that enhances preference alignment without an additional fine-tuning phase.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AGI Amusement and Practical AI Tools**: The Discord community pondered the essence of AGI with humorous suggestions such as flawless USB insertion skills, indicating the expectation for AGI to perform complex human-like tasks. Useful AI tools like [Elicit](https://elicit.com) were recommended for summarizing scientific research, with Elicit notably commended for its efficient paper summarization and synthesis.

**ChatGPT Takes a Sick Day, Voice Mode Hesitation**: Speculation around ChatGPT outages included backend provider issues and a potential DDoS attack by Anonymous Sudan. The rollout of new **voice mode** features in GPT-4o was discussed, with mixed feelings about the promised timeline and reported persistent issues such as 'bad gateway' errors and laggy Android keyboards.

**The Prompt Engineering Conundrum**: Challenges in prompt engineering were aired, especially the difficulty of adhering to complex guidelines, leading to calls for improved versions. *WizardLM 2* was suggested as a high-performing alternative to GPT-4, and breaking down complex prompts into steps was recommended as an approach to optimize results.

**API Affordability under Scrutiny**: Conversations turned to the cost of using the GPT API versus ChatGPT Plus, with API potentially being the cheaper option depending on usage. Alternatives like **OpenRouter** and **WizardLM 2** were proposed for better value, and an article titled "*Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*" was endorsed as a must-read for prompt engineering insights.

**Rollout Delays and Performance Puzzles**: Delays in new feature rollouts and performance issues with large prompts were common concerns. To counteract the sluggish response with hefty prompts, lazy loading was mentioned as a potential solution to browser difficulties.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio GPU Sagas**: Engineers discussed the behavioral quirks of **LM Studio models** with an emphasis on offloading parameters, affirming that running on a dedicated **GPU** often yields better results than shared resources and underlined the fine line between model size restrictions and **GPU memory**—mentioning that models should be under **6GB** to alleviate loading issues.

**Model Recommendations for Codewranglers**: The **CodeQwen 1.5 7B** and **Codestral 22b** models were specifically recommended for code optimization tasks, while **Wavecoder Ultra** was also suggested despite its obscure launch history. Additionally, the utility of platforms like [Extractum.io](https://llm.extractum.io/list) was highlighted for filtering models based on criteria such as VRAM and quantization.

**The Fine Print of AI Performance**: Conversation veered into the technical details of AI limitations, noting that performance can often be limited by **memory bandwidth**, and members suggested targeting an **80%** workload in relation to physical core count on processors. The uncertainty surrounding future **Chinese language support** was also brought up.

**Do-It-Yourself Servers Draw Debate**: Discussions around building custom **homelab GPUs** focused on VRAM capacity, driver support, and performance between manufacturers. Concerns were addressed regarding second-hand GPUs' reliability and members weighed pros and cons of **AMD ROCm** versus NVIDIA's ecosystem for stability and throughput.

**Engineering a Beta Buff**: In the world of software development and AI tinkering, **continue.dev** was lauded for local setups, particularly for supporting **LM Studio** configuration, while a call for testers was raised for a new **AVX-only extension pack**, showcasing the community's collaborative spirit and ongoing optimization endeavors.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Takes the Stage**: Discussions emerged around the *[Wayseer Manifesto - Official Video](https://youtu.be/OPR3GlpQQJA)*, evidently popular for its motivational message, and design talk sparked by Nous Research’s [Twitter account](https://x.com/StudioMilitary), hinting at a creative flair within the AI community.
- **OpenAI Unwrapped**: Speculation arises about [OpenAI's GPT-4](https://laion.ai/notes/open-gpt-4-o/), with members eagerly anticipating potential capabilities and implications for future AI research and application realms.
- **Gearing Up for T5 and Beyond**: Technical conversations revealed that the T5 model sets a high barrier for adoption due to its hefty hardware requirements; meanwhile, promising alternatives like an open-source UI from Mobius for chat assistants and potential improvements via ggml are subjects of interest.
- **Graphical Glitches with Pixart**: Technical angst surfaced as Pixart struggles when scaling to datasets larger than 10k images, unlike other models that retain stability with up to 400k images, attributing success to unique training methodologies.
- **WorldSim Wonders and Wisdoms**: The recent WorldSim Jam Session is available [on YouTube](https://www.youtube.com/watch?v=qaE99xmtvd4), coupled with a dose of irony in recognizing that *agents* might be the first job category outpaced by AI, and a re-engaged member festivities by sharing their return and research progress.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Faces Tough Crowd**: Initial feedback from users indicates that the early model of **Stable Diffusion 3 (SD3)** struggles with hand depictions, lagging behind Dall-E in some aspects; however, optimism remains for the potential improvements through custom models upon wider release.

- **Architecture’s AI Angle**: Discussions surfaced around applying **Stable Diffusion** to **architectural visualization**, suggesting the use of img2img techniques with detailed inputs to enhance output quality, despite the tool's limitations with rendering straight lines and geometrically accurate mechanics.

- **Plugin Pitfalls**: Users are encountering quality degradation issues when using the **wildcards plugin** with **Stable Diffusion**, reporting grainy results and color distortions despite multiple installation attempts.

- **Community Model Mining**: The engineering community recommends exploring community models available on platforms like [civitai.com](https://civitai.com) and utilizing [ChaiNNer](https://github.com/JoeyBallentine/chaiNNer), a node-based image processing GUI tool, to improve and upscale image results from **Stable Diffusion**.

- **AI's Celebrity Conundrum**: The rise of AI-generated influencer profiles such as 'celebrity LoRas' on **Civit** prompted a tongue-in-cheek debate on the nature of celebrity in the era of AI, highlighting the blurred lines between virtual and reality as these profiles gain followers and media attention.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**SD3 Models Grapple with Grainy Results**: Users highlight a *spotty noise issue* in SD3 2B models despite using advanced features like a 16ch VAE, with noise artifacts particularly evident in areas such as running water. Skepticism has been voiced about the current validation metrics and loss functions for SD3 models, as they are perceived to poorly indicate model performance.

**Open-source Breakthrough for Video Models**: The community showed enthusiasm about an [Apache2 licensed video-capable CV-VAE](https://arxiv.org/abs/2405.20279), expected to be a valuable resource for research on latent diffusion-based video models.

**Peering into Future Model Architectures**: newly released research introduces the **State Space Duality (SSD)** framework and the cutting-edge **Mamba-2** architecture, claimed to be 2-8X faster than its predecessor, contesting Transformer models in language processing tasks ([arxiv paper](https://arxiv.org/abs/2405.21060)).

**Training Tactics Under Scrutiny**: A preprint suggests that embeddings perturbed by slight corruption of pretraining datasets can improve diffusion models' image quality ([arxiv preprint](https://arxiv.org/abs/2405.20494)), while others mention using dropout and data augmentation to prevent overfitting in large diffusion models, and a debate on whether adding training data difficulty can enhance model robustness.

**Aesthetic Assessments and Realism Rivalries**: Comparisons between SD3 images and Google's realistic examples have sparked discussions, with SD3 images being humorously likened to "women suffering a bad botox injection" ([Reddit examples](https://www.reddit.com/r/StableDiffusion/comments/1d73j3r/some_sd3_images_women/)), and Google's work earning praise for its textured cloth and consistent hair representations ([Google demo](https://vxtwitter.com/GoogleDeepMind/status/1797605392089825457)).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **E-Paper or Not E-Paper? That is the Question**: Members hotly debated the legitimacy of the Daylight tablet's e-paper claims after a review video suggested it might be a reflective LCD instead. The discussion gravitated around whether the Daylight is misbranded existing Sharp RLCD tech or a genuine innovation, with members suggesting a possible teardown for clarity.

- **Beyond Heads and MLPs: Discovering LLM Knowledge Circuits**: A new research paper revealing deeper insights into LLM knowledge circuits has caught the attention of members, with one member valuing its departure from focusing on individual network components. The research community also delved into whether corrupted datasets might actually improve diffusion models and the synergy between RNNs and pre-trained transformers.

- **Public Servant AI?**:
  Users discussed efficiency improvements for AI tasking, with concerns about single-query slow processing and the impact of default n-shot values on results. There's also a practical search for the smallest Huggingface decoder model to study energy use, and a GitHub [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900) introducing machine-translated ARC challenges across multiple languages.

- **The Efficiency Hunt**: Concerns about model size have been posed, with efforts to condense a 22GB model like TinyLLama.db for better activation and weight entry coordination. Furthermore, the community pondered using a differentiable top-k function for image classification, potentially to hone model focus on the most significant elements.

- **Global Benchmarking Gets Multilingual**: An initiative to expand the ARC challenge benchmark to 11 machine-translated languages via a collaborative [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900) was put forward, with an eye on future language additions. Review and contributions to this multilingual extension of benchmarks are underway.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cryptocurrency Credits Conundrum**: A user encountered issues with credits not appearing after an **ETH payment**, prompting advice to wait up to an hour before lodging a complaint; patience might be a virtue in blockchain timing.
- **LLMs' Proficiency with Prefills**: The consensus among users is that **Language Learning Models (LLMs)** adeptly handle **prefill text**, ensuring the generation of subsequent content is consistent with the initial input.
- **Turbo Troubles and API Oversight**: **GPT-3.5 Turbo's inconsistencies** led to a discussion about potential **API moderation**, with a reminder that OpenAI mandates moderation for all requests via their *moderation API*.
- **Mistral's Momentary Mute**: Reports of receiving empty responses from **Mistral: Mixtral 8x22B Instruct** prompted administrative guidance to set **DeepInfra** as a preferred provider and check [load balancing documentation](https://openrouter.ai/docs/provider-routing) for resolving provider-specific issues.
- **Narrative Nuances via Neural Networks**: When debating the best models for storytelling, users recommended various **roleplay-specific models**, directing attention to [OpenRouter's rankings](https://openrouter.ai/rankings/roleplay?view=week) for those particularly excelling in this creative endeavor.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Speed Boost with a Line**: A [tutorial video](https://youtu.be/OiMZtjSZVOw) on how **Numba** can dramatically increase Python performance using JIT compilation was highlighted. The impact of this one-liner, however, piqued interest regarding the potential for achieving similar performance without additional libraries.

- **Efficiency in Python and Mojo**: The efficacy of utilizing **for loops within while loops** in Python was debated, with a recommendation to explore generators via a [Real Python resource](https://realpython.com/introduction-to-python-generators/). Additionally, the possibility of Mojo's **MAX** tool expediting Python execution was discussed, comparing it to enhancements brought by **Tensor** and **Torch** libraries.

- **Mojo Async Paradigm Sparks Controversy**: The suggestion to default all Mojo functions to `async` sparked a debate. Concerns were voiced about straying from Python standards and complicating the workflow for those accustomed to explicit `async`/`await` methodologies.

- **Ownership Deep Dive by Modular**: A blog post titled [Deep Dive into Ownership in Mojo](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo), featuring insights from CEO Chris Lattner, was introduced. This piece is a sequel to an earlier exploration of ownership as a conceptual framework.

- **Challenging Rust with Project Verona**: **Project Verona** was put under the spotlight as a rival to **Rust** in providing memory safety with a gentler learning curve. Enthusiasts are directed to watch a [YouTube talk](https://youtu.be/VU9QATDvidw), "*Concurrent Mutation must go*," for an in-depth discussion on the topic.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Bind_Tools Tamed for Ollama Models**: Members confirmed that **LangChain** supports `bind_tools` for **Ollama models** through the `OllamaFunctions` class, and provided a relevant GitHub issue link for additional reference.
  
- **Building Customer Support with AI**: An ongoing discussion on creating **AI-driven customer support systems** identified LangChain, LLMs like Llama3, and custom tools for actions such as user verification, with shared Python code as an example for chaining models and tools.

- **Preserving Conversation Memory in SQL Failures**: SQL agent chat context preservation was a hot topic, with a shared code snippet using `ConversationBuggerMemory`. However, there were concerns regarding unsupported kwargs.

- **Categorization Using LangChain and Embeddings**: The guild explored strategies for categorizing 10,000 free-text responses using **LangChain and embeddings**, highlighting the use of prompt engineering to enhance efficiency.

- **Text Editing Redefined with Automated Chat Analyzer**: An **automated chat analyzer** that can produce editable plain text Q&A from message lists was introduced, aiming to ease manual editing and reduce compute usage.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Bitshifting Buzz in Backend Development**: Engineers engaged in a lively debate on the merits of implementing a [bitshift operation PR #4728](https://github.com/tinygrad/tinygrad/pull/4728) across all backends in **tinygrad**, with scepticism around its performance boost compared to traditional multiply/divide operations.
  
- **Testing Puzzles in GPU Land**: Curiosity arose as to why device tests from 'gpuctypes' were absent, referencing a specific [`test_cuda.py` missing tests](https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36), contributing to the ongoing discussion about thorough testing practices.

- **Diving into tinygrad's Depths with Hotz**: George Hotz unveiled plans for a **tinygrad** presentation focused on clarity and deep dives into the codebase, emphasizing the project's independence from dependencies like CUDA.

- **Lean Toward Clearer Mechanization Bounties**: The community grappled with ambiguous problem statements regarding `ShapeTrackers` in Lean, recommending a review of [ShapeTracker](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py) in tinygrad's repository for clearer understanding.

- **Traceback Trail for Tensors**: A proposal to add 'traceback' attributes to new `Tensor` instances in **tinygrad** was revisited, emphasizing the potential for enhanced debugging despite previous incomplete attempts like PR #3302.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM-Sourced Python Scripting Wins**: An LLM demonstrated its scripting chops by producing a Python script for [extracting structured data](https://t.co/N2BJ54zr7i) from Gmail, an approach that could streamline data extraction processes across diverse email datasets.
- **Google Gemini Widens the Window**: The structure and processing capabilities of [Google Gemini](https://t.co/Qg9ydPzBdd) were highlighted in a LlamaIndex agent, touting an impressive 1 million token context window to tackle complex, multifaceted queries from heterogeneous documents.
- **Custom Parsing Conundrums and Solutions**: Challenges around implementing Langchain’s `HTMLHeaderTextSplitter` led to the engineering of a custom solution within LlamaIndex's `IngestionPipeline`, supported by [custom transformations documentation](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations).
- **Rkhettry Cracks Chroma Code**: A document retrieval issue from `VectorStoreIndex` within Chroma's vector store was addressed by a user through direct access methods, demonstrating the practical problem-solving approaches in database manipulation.
- **Metadata Magic for Enhanced Indexing**: Incorporating metadata was recommended to improve document retrieval within indexing systems, as outlined in LlamaIndex's [metadata extraction guide](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/), emphasizing the importance of rich document descriptors for fine-grained search capabilities.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Phi-3 Models Ascend the Leaderboards**: The **Phi-3 Medium (14B)** and **Small (7B)** models have been highlighted on the **@lmsysorg leaderboard**, with the Medium model being in close proximity to **GPT-3.5-Turbo-0613** performance levels and the Small model compared to **Llama-2-70B** and various **Mistral fine-tunes**. The community reaction includes both humor at personal wagers gone awry and serious discussions on the sustainable growth and reputation gains in such model rankings.

- **OpenAI's Inner Turmoil**: Current and former OpenAI employees penned [an open letter](https://righttowarn.ai/) bringing up concerns about insufficient oversight in AI development, while the firing of researcher Leopold Aschenbrenner for the alleged leak of proprietary information elevated conversations around trade secrets and national security. Additionally, skepticism proliferates the channels regarding scaling up compute as a linear path to AGI, with users questioning the sustainability of such growth without formidable challenges.

- **Scaling Laws, Seriously?**: A layer of humor underscores discussions of scaling laws, with users mocking the faith in perpetual scaling deep into the 2030s through a meme, as well as a playful request for "just 100k more GPUs" for achieving a hypothetical 10 trillion parameter model. Frustration is expressed over the starkly contrasting beliefs in the AGI debates, with criticism aimed at parties with little acknowledgment of their epistemic uncertainty.

- **Addressing AI Whistleblower Safeguards**: The concerns for safety and oversight in AI justified by an [open letter from OpenAI employees](https://righttowarn.ai/) and subsequent talks illustrate disquiet regarding how rapidly advancing AI might be mishandled due to powerful financial incentives against robust oversight.

- **Stale Memes Gather No Engagement**: The extremely low engagement in the #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/) channel suggests either a disinterest in meme culture or the need for fresher content to stimulate exchanges among the AI engineering audience.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **TorchTune's Shoutout for Newsletter Recognition**: An individual from the Torchtune community asked for their work on state-of-the-art (SOTA) models and methods to be featured in the AI News newsletter, inviting others to join their server with a [provided invite link](https://discord.gg/6cGsKMM5).
  
- **Tech Glitch in AI News Delivery**: A formatting glitch was reported with AI News emails on ProtonMail, causing display issues when dark mode is enabled, which only allows links and images to be clearly seen.

- **Behind the Podcast Curtain**: The secret sauce behind the podcast's automated transcripts, which include speaker identification, was disclosed to be Google's smol-podcaster tool, enhanced with manual edits for speaker names.

- **LiveKit's Lucrative Leap**: LiveKit successfully raised $22.5 million in a Series A round, aiming at establishing a fresh infrastructure for AI's transport layer, focusing on real-time voice and video interactions, despite a challenging fundraising experience amplified by the emergence of GPT-4. The details were shared in a [tweet](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww).

- **Twelve Labs' Multi-Model Money Magnet**: Twelve Labs bagged a $50 million Series A investment for innovating video understanding AI foundation models, introducing their new Marengo 2.6 that fuses multimodal capabilities in a single API; full information can be found in their [press release](https://www.prweb.com/releases/twelve-labs-earn-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Conference Joy Without the Paper Trail**: Engineers reflected on the rewarding experience of participating in conferences even without contributing papers, signifying the importance of community and knowledge exchange.

- **Help Wanted with OrPO’s Stubborn Formatter**: A user is struggling with a **Python script for a custom OrPO formatter** to tokenize datasets and has reached out for support. A related script has been shared for reference.

- **AI's Medical Diagnosis Dilemma**: A tweet highlighted the poor performance of advanced AI models like **GPT-4V and Gemini Pro** in medical Visual Question Answering (VQA) tasks, using the **ProbMed dataset** as a benchmark. The engineering community discussed the challenges faced by visual LLMs in medical fields.

- **Seeking and Succeeding with arXiv Endorsement**: An AI collective member sought an **arXiv endorsement** for the cs.LG category and managed to solve their dilemma by resorting to their organizational email.

- **Troubleshooting LoRA Training Hitches**: An engineer encountered a hiccup where **QLoRA training output** was not initiating as expected. Another member pointed out that LoRA training scripts might automatically download the necessary model from **Hugging Face Model Hub** if it's not available locally.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Artificial Ivan Troubleshoots Like a Pro**: **Cohere** has advanced its "Artificial Ivan" to version 4.0, enabling it to troubleshoot code and sharing an [affirmations app](https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations) tied to his developments.

**Real Ivan's Easing into Early Retirement?**: One user's quip about a human counterpart, the "real Ivan," potentially retiring at 35 due to Artificial Ivan's accomplishments, brought a humorous spin to the project's success.

**Cross-Project Synergy Unlocked**: A user highlighted the integration of **Aya 23** with **Llama.cpp** and **LangChain**, offering a sample code and seeking assistance to implement a stopping condition in conversations using "\n".

**Seeking Bilingual AI Conciseness**: Detailing code aiming to produce concise, Spanish-language responses, the user outlined the use of prompts for conversation memory and parameters to enhance **Aya 23**'s performance.

**Cohere's Community Corner**: Contrasting with Langchain, a playful comment from a guild member described the Cohere Discord as a "chronically online AI lab," pointing to lively interaction and engagement among its members.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **CUDA Conundrum Conquered**: An engineer resolved CUDA error issues by using `--recompile --gpu NVIDIA` and discovered that the flag `-ngl 9999` must come after `--recompile` for effective resolution.

- **Peer Power Prevails in Problem Solving**: Successful troubleshooting of CUDA recompilation was attributed to community collaboration and a resourceful check of the `--help` options, with a helpful [GitHub resource](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine) also being shared. 

- **Engineers Emphasize Unity and Treats**: The sentiment within the community highlighted the importance of learning from one another and a humorous nod to "cookies" as part of community spirit.

- **CPU Operations Entering New Phase with RISC-V**: A recent YouTube video, ["The Magic of RISC-V Vector Processing"](https://www.youtube.com/watch?v=Ozj_xU0rSyY), shed light on the ratified 1.0 RISC-V Vector Specification and its expected impact on vector computations. 

- **Link Love**: Engineers were directed to further details and discussions via two key links: the [RISC-V Vector Processing video](https://www.youtube.com/watch?v=Ozj_xU0rSyY) explaining new specifications and advancements, and a [GitHub repository for llamafile](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine) to assist with distributed LLMs.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Windows Wizard Wrangles Gemini Setup**: A member outlined a detailed **setup guide for Gemini model** on Windows systems, citing outdated official documentation. The guide includes command-line steps and workarounds for common setup issues.
  
- **AR Enters the Workspace with Spacetop**: Sightful's **Spacetop AR Laptop** grabbed attention with its battery-free glasses and unique design intending to replace traditional laptop displays, sparking discussions on its place in the future of mobile computing. Members also discussed **Xreal glasses**, mentioning their reliance on an external device for power and the need for improvement in resolution and field of view.
  
- **Macbook M1 Meets Match with Infinite Loop**: A user reported a persistent issue when running `poetry run 01 --local` on a Macbook M1, facing an infinite loop with the **ollama llama3** setup, and is seeking a solution.
  
- **Secure Version Query Goes Unanswered**: A question was raised about the release of a **secure version** of a discussion topic, but it went without a specific answer or follow-up within the discussions.

- **Availability and Battery Life in AR Glasses Explored**: Conversations on **Xreal glasses** included experiences with using them with a MacBook, highlighting the limitations they currently have in terms of resolution, battery life, and device dependency for power.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **Initiating Direct Dialogue**: Members in the **ai-ml** channel have expressed interest in collaborating and have opted to continue their exchange via direct message for more detailed discussions.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI and ML under the Microscope**: The **Infer: Summer ‘24** conference, hosted virtually by Qwak on June 26, offers an in-depth look at AI and ML practices with a focus on real-world applications from industry experts.
- **Get Ready for a Deep Knowledge Dive**: Those interested in **recommender systems** and AI in sports will find targeted sessions on advanced ML model construction and AI-driven sports analytics.
- **Safe AI Takes the Spotlight**: AI safety and adherence to regulation is set to be a main talking point, highlighting strategies like "Schematic Questioning" to mitigate risks such as inaccurate content in AI systems.
- **From Stream to Bank**: Attendees can expect insights from heavy-hitters at Disney Streaming, Lightricks, LSports, and Lili Banking, who will convey real-world experience with AI/ML integration.
- **Production-Ready LLMs Explored**: Large Language Models (LLMs) like GPT and Llama feature prominently, with discussions planned around effective implementation in production settings across various industries. [Conference Registration](https://tinyurl.com/j8z6s8ka)



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **xLSTM's Open Source Debut**: Dr. Tristan Behrens has announced the release of **xLSTM's** source code, a move sure to excite AI engineers and developers. The official announcement and access to the source code can be found on his [LinkedIn post](https://www.linkedin.com/posts/dr-tristan-behrens-734967a2_drop-everything-nxai-has-released-the-official-activity-7203659602628935682-GwOA).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1247386474589061120)** (16 messages🔥): 

- **Managing Confidential Data in Datasets**: Discussed the options for using tools with private datasets, such as training models with private data using AutoTrain and integrating with the Datasets library to handle data from formats like CSV, JSON, and parquet files.

- **DSPy Resources Galore**: Members shared several resources for getting started with **DSPy**, including Hamel's blog post on [intercepting LLM API calls](https://hamel.dev/blog/posts/prompt/) and a YouTube video titled [DSPy Explained](https://youtu.be/41EfOY0Ldkc?si=e6mVFi9tC6KJOaQC).

- **Python Logging Book Released**: Michael Driscoll's step-by-step guide on Python logging, funded through Kickstarter, was announced, with code examples available on [GitHub](https://github.com/driscollis/pythonlogging).

- **Where to Direct Zoom Session Questions**: Users were guided to the correct channel for posting questions during the current Zoom session.

- **Situational Awareness AGI Paper**: A link to a detailed paper by Leopold Aschenbrenner on the future of AGI and its implications was shared. It's available at [situational-awareness.ai](https://situational-awareness.ai).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://situational-awareness.ai/">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>: Leopold Aschenbrenner, June 2024 You can see the future first in San Francisco. Over the past year, the talk of the town has shifted from $10 billion compute clusters to $100 billion clusters to trill...</li><li><a href="https://youtu.be/41EfOY0Ldkc?si=e6mVFi9tC6KJOaQC">DSPy Explained!</a>: Hey everyone! Thank you so much for watching this explanation of DSPy! DSPy is a super exciting new framework for developing LLM programs! Pioneered by frame...</li><li><a href="https://hamel.dev/blog/posts/prompt/">- Fuck You, Show Me The Prompt.</a>: Quickly understand inscrutable LLM frameworks by intercepting API calls.</li><li><a href="https://github.com/driscollis/pythonlogging">GitHub - driscollis/pythonlogging: Code examples for the Python Logging book</a>: Code examples for the Python Logging book. Contribute to driscollis/pythonlogging development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1247275041666629632)** (2 messages): 

- **LLM Validates Medical Billing**: One member shared their use case of an **LLM for the German DRG system** to act as a medical controller validating hospital bills. It utilizes **RAG** to incorporate patient information and legal texts, identifying potential coding violations.

- **5 Practical Finetuning Use Cases**: Another member outlined five compelling finetuning use cases, including:
  1. **Summarizing medical test results** with personalized Q&A.
  2. **Evaluating sales staff performance** via call transcripts.
  3. **Whatsapp bot triaging**.
  4. **Transcribing regional accents** using Whisper.
  5. **Creating a search assistant**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1247311590722633760)** (11 messages🔥): 

- **Modal's rapid revenue growth impresses**: A Substack post highlighted Modal's impressive revenue growth, increasing over 50x in one year to reach an eight-figure run rate. They are currently hiring software engineers in New York City. [Modal's hiring link](https://jobs.ashbyhq.com/modal/dd84cf88-1f13-4f39-b371-237e103fce34).

- **Etched builds a faster chip for transformers**: Etched developed a chip that significantly outperforms GPUs, running transformers more than 10x faster. They are actively hiring for various engineering roles in Cupertino. [Etched's hiring link](https://boards.greenhouse.io/etchedai).

- **Modal's infrastructure prowess**: One member shared a talk by Erik on how Modal optimizes its data infrastructure using custom solutions such as a filesystem written in Rust and their own container runtime. This approach allows them to efficiently start thousands of large containers in seconds. [Erik's talk at Data Council](https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure).

- **Modal credits last for a year**: There was a clarification about Modal credits, ensuring that they do not expire at the end of the billing cycle but last for a year.

- **Detailed fine-tuning guide available**: A member praised Modal's documentation and tutorial repositories for their clarity and usefulness in fine-tuning LLMs. The guide includes advanced techniques such as [LOra](https://huggingface.co/blog/peft), [Flash Attention](https://arxiv.org/abs/2205.14135), and [Gradient checkpointing](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9). [Modal fine-tuning examples](https://modal.com/docs/examples/llm-finetuning).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modal.com/docs/examples/llm-finetuning">Fine-tune an LLM in minutes (ft. Llama 2, CodeLlama, Mistral, etc.)</a>: Tired of prompt engineering? Fine-tuning helps you get more out of a pretrained LLM by adjusting the model weights to better fit a specific task. This operational guide will help you take a base model...</li><li><a href="https://whyyoushouldjoin.substack.com/p/modal.">Why You Should Join Modal</a>: The future of GPUs is serverless.</li><li><a href="https://www.datacouncil.ai/talks/creating-our-own-kubernetes-and-docker-to-run-our-data-infrastructure">Creating our own Kubernetes and Docker to run our data infrastructure</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1247590113009995786)** (4 messages): 

- **Jarvis Labs Credits Giveaway**: All users enrolled in the course received **$200 in credits** to use on Jarvis Labs. A member noted humorously that no extra credits were added to their account by accident.

- **Channel for Announcements**: The channel was highlighted as the best place for **Jarvis Labs announcements**. Members were encouraged to spread the word about the platform and provide feedback.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1247489054832857088)** (130 messages🔥🔥): 

- **Hugging Face Credits Rollout Underway**: Members who filled out the form have begun seeing $501.42 in credits applied to their accounts, visible on their [billing settings](https://huggingface.co/settings/billing). Remaining rollouts are expected to complete within a day, while some users experienced delays and are advised to email `billing@huggingface.co` if issues persist.

- **Double Credits Issue**: Several users reported receiving double the expected credits ($1002.84). The Hugging Face team acknowledged the issue and assured that corrections would be made automatically.

- **Inference and Compute Options Highlighted**: Users can manage endpoints and auto-scaling for cost efficiency via [Inference Endpoints documentation](https://huggingface.co/docs/inference-endpoints/index) and utilize [Spaces' GPU upgrades](https://huggingface.co/docs/hub/spaces-gpus) for compute-intensive tasks.

- **Finetuning Custom Models**: Queries about using Hugging Face compute for custom code and fine-tuning models were directed to resources like [SpaceRunner](https://huggingface.co/blog/abhishek/autotrain-spacerunner) and [Spaces Dev Mode](https://huggingface.co/blog/spaces-dev-mode), which allow custom training scripts and environments.

- **Community Integrity Praised**: Users who reported issues with credits and errors were thanked for their honesty. The community's integrity and collaboration were highly appreciated by the Hugging Face team.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/api-inference/en/index">Serverless Inference API</a>: no description found</li><li><a href="https://huggingface.co/docs/inference-endpoints/index">🤗 Inference Endpoints</a>: no description found</li><li><a href="https://ui.endpoints.huggingface.co/">Inference Endpoints - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/inference-api/serverless">Inference API (serverless) - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/abhishek/autotrain-spacerunner">Train Custom Models on Hugging Face Spaces with AutoTrain SpaceRunner</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/en/spaces-sdks-docker-jupyter">JupyterLab on Spaces</a>: no description found</li><li><a href="https://huggingface.co/blog/spaces-dev-mode">Introducing Spaces Dev Mode for a seamless developer experience</a>: no description found</li><li><a href="https://tenor.com/view/bill-nye-party-horn-confetti-sarcastic-like-child-gif-5499505">Sarcastic Celebration GIF - Bill Nye Party Horn Confetti - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/settings/billing">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/docs/inference-endpoints/autoscaling#scaling-to-0">Autoscaling</a>: no description found</li><li><a href="https://huggingface.co/spaces/launch">Spaces Launch – Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#sleep-time)">Using GPU Spaces</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#pause)">Using GPU Spaces</a>: no description found</li><li><a href="https://huggingface.co/settings/billing.">Hugging Face – The AI community building the future.</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1247573592447651952)** (11 messages🔥): 

- **Watch for Replicate credits email**: Members are reminded to check their inboxes for an email from **Replicate** to redeem credits. *"Keep an eye on your inbox today for an email from Replicate to redeem credits."*
- **Creating Replicate orgs piggybacks on GitHub**: Users inquired about the necessity of creating an 'org' on **Replicate**. It's clarified that while it's possible to redeem credits with a personal account, creating an org requires a GitHub org. *"Replicate orgs currently piggyback on GitHub orgs, so you'll need to create a GitHub org first, or use an existing one."*
- **Query on credit visibility with billing setup**: There was concern about the visibility of credits without billing setup. It's indicated that credits should still be visible but a member is asked to DM their details for confirmation with the product team. *"I think you should be able to see/redeem the credits without having set up billing yet, but let me confirm with the product team."*
- **Mixed success in claiming credits**: Multiple users report successfully receiving and claiming their **Replicate** credits. However, one user noted they had not received theirs yet. *"I received a replicate credits redeem mail"*, *"I got mine."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1247292497953030236)** (4 messages): 

- **Members experience credit issues**: One member asked if others had received credits yet and mentioned not seeing any in their account. Another member confirmed they had not received credits either.
- **Form completion mistake causes concern**: A member shared that they mistakenly filled out their organization name instead of the organization ID while completing the form on their phone. They wondered if it was too late to correct the mistake.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1247475652677074985)** (2 messages): 


- **John's Presentation Entertains**: A user praised John's presentation and called him a great presenter. They mentioned his humorous statement: *"LLMs are dumb mechanical humans, temperature is blood alcohol content of the model,"* and expressed interest in reading his books.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1247323056788082748)** (7 messages): 

- **Parsing Screenshots with Models Explored**: A user discusses the capabilities of **Opus 4o** and **MiniCPM-Llama3-V-2_5** in parsing text from images and considers fine-tuning them for better performance. They express a willingness to collaborate on this task.
- **PaliGemma for OCR Tasks**: It's noted that **PaliGemma** can perform OCR among other tasks like generating captions in various languages. This might serve as a helpful alternative to models that need more complex adaptations for handling screenshots.
- **Finetuning vs. Retrieval for Government Data**: A user shares their dataset of government farming grants and considers whether to finetune an LLM with Q&A pairs or train it on the full text. Another user suggests exploring **retrieval applications**, given the structured nature of grant data.
- **Experiments with RAG**: The user considering finetuning also acknowledges that **RAG (Retrieval-Augmented Generation)** might be a more suitable application but focuses on experimenting with model training for educational purposes. This approach might provide foundational understanding useful for future applications.

**Link mentioned**: <a href="https://www.gov.uk/countryside-stewardship-grants">Countryside Stewardship grant finder</a>: Find information about Countryside Stewardship options, capital items and supplements

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1247587624131231755)** (361 messages🔥🔥): 

- **HuggingFace and Model Serving Insights**: There's considerable enthusiasm around merging LoRA with base models using Axolotl, as highlighted by links to [Axolotl documentation](https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base). One user shared, "HuggingFace does *a lot* of things...difficult to both (1) provide a clean user interface...and (2) inform users about all the possibilities without being annoying."
- **Modal and Charles' Flex on Credits**: Charles from Modal showcased an impressive presentation and gave out additional $500 credits to participants who tried Modal before June 11. This move stirred excitement with comments like, “charles_irl on ‘Inference Trilemma’; insane flex by Modal and Charles.”
- **Medusa and LLM Inference Innovations**: Various users praised the introduction of Medusa, a method that "augment[s] LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel," as detailed further in [Medusa's paper](https://arxiv.org/abs/2401.10774).
- **Handling Quantization Challenges**: Queries about whether quantized inference is slower were addressed with explanations pointing out that quantization trades decreased memory overhead for increased computation overhead. Specific issues with bitsandbytes quantization being less efficient were discussed.
- **Predibase and LoRAX Enthusiasm**: Travis Addair from Predibase mentioned setting up office hours to discuss LoRAX and other related topics. He shared links to the [LoRAX GitHub](https://github.com/predibase/lorax) for users interested in the multi-LoRA inference server.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca-merged-awq">parlance-labs/hc-mistral-alpaca-merged-awq · Hugging Face</a>: no description found</li><li><a href="https://outerbounds.com/blog/the-many-ways-to-deploy-a-model/">The Many Ways to Deploy a Model | Outerbounds</a>: There are many ways to deploy models and perform inference. Here, we share our decision rubric for model deployments using LLM inference as an example.</li><li><a href="https://huggingface.co/VAGOsolutions/Kraken-LoRA">VAGOsolutions/Kraken-LoRA · Hugging Face</a>: no description found</li><li><a href="https://cog.run/">Cog</a>: no description found</li><li><a href="https://docs.vllm.ai/en/stable/quantization/auto_awq.html">AutoAWQ &#8212; vLLM</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>: The inference process in Large Language Models (LLMs) is often limited due to the absence of parallelism in the auto-regressive decoding process, resulting in most operations being restricted by the m...</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - Efficient finetuning of Llama 3 with FSDP QDoRA</a>: We’re releasing FSDP QDoRA, a scalable and memory-efficient method to close the gap between parameter efficient finetuning and full finetuning.</li><li><a href="https://www.deeplearning.ai/short-courses/efficiently-serving-llms/">Efficiently Serving LLMs</a>: Serve LLM applications in production with techniques like KV caching. Learn to apply frameworks like LoRA and LoRAX.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/#merge-lora-to-base">Axolotl</a>: no description found</li><li><a href="https://tenor.com/view/burn-elmo-pyro-burn-it-down-ashes-gif-12152943568085011868">Burn Elmo GIF - Burn Elmo Pyro - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/eugeneyan/status/1798073562764620003">Tweet from Eugene Yan (@eugeneyan)</a>: .@charles_irl on &#34;Inference Trilemma&#34; (TLC): Throughput, Latency, Cost  Other trilemmas: • CAP: Consistency, Availability, Partition Tolerance • Impossible Trinity: Fixed exchange rate, free c...</li><li><a href="https://predibase.com/fine-tuning-index">The Fine-tuning Index</a>: Performance benchmarks from fine-tuning 700+ open-source LLMs</li><li><a href="https://tenor.com/view/yes-excited-screaming-gif-20651075">Yes Excited GIF - Yes Excited Screaming - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/presentation/d/1PS0nWigtRLQd5q0czCvNzu19Qc7eM5o6pI7mDS5JxrI/edit#slide=id.g2c7588f453b_0_272">Mastering LLMs - Deploying LLM Services on Modal</a>: Deploying LLM Services on Modal bit.ly/mastering-llms-deployment</li><li><a href="https://x.com/TheZachMueller/status/1798078059247259729">Tweet from Zach Mueller (@TheZachMueller)</a>: Me: we&#39;re done with the course tweets  @modal_labs labs:</li><li><a href="https://tenor.com/view/making-it-rain-gif-617522917670078816">Making It Rain GIF - Making it rain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/eugeneyan/status/1798074567023628697">Tweet from Eugene Yan (@eugeneyan)</a>: @charles_irl insane flex by @modal_labs and @charles_irl giving students who&#39;ve tried modal ANOTHER $500 credit</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master/replicate-examples">ftcourse/replicate-examples at master · parlance-labs/ftcourse</a>: Contribute to parlance-labs/ftcourse development by creating an account on GitHub.</li><li><a href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines.</a>: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...</li><li><a href="https://x.com/bytetweets/status/1798078590749388851">Tweet from Florian Buetow (@bytetweets)</a>: Yes absolutely insane. They added another $500 in compute credits for us students in the LLM Fine-Tuning conference/course by @dan_s_becker and @HamelHusain on @MavenHQ. The big question is if another...</li><li><a href="https://github.com/predibase/lorax">GitHub - predibase/lorax: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs</a>: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs - predibase/lorax</li><li><a href="https://modal.com/docs/examples">Featured examples</a>: How to run LLMs, Stable Diffusion, data-intensive processing, computer vision, audio transcription, and other tasks on Modal.</li><li><a href="https://predibase.com/">Predibase: The Developers Platform for Fine-tuning and Serving LLMs</a>: The fastest and easiest way to fine tune and serve any open-source large language model on state-of-the-art-infrastructure hosted within your private cloud.</li><li><a href="https://www.anyscale.com/blog/continuous-batching-llm-inference">Achieve 23x LLM Inference Throughput &amp; Reduce p50 Latency</a>: In this blog, we discuss continuous batching, a critical systems-level optimization that improves both throughput and latency under load for LLMs.</li><li><a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/benchmarks/inf2/inf2-performance.html#decoder-models">Inf2 Inference Performance &#8212; AWS Neuron Documentation</a>: no description found</li><li><a href="https://github.com/replicate/cog-vllm">GitHub - replicate/cog-vllm: Inference LLM on replicate with vllm</a>: Inference LLM on replicate with vllm. Contribute to replicate/cog-vllm development by creating an account on GitHub.</li><li><a href="https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/">The State of AI Infrastructure at Scale 2024</a>: How are fortune 1000 companies handling the growing demands of AI on their infrastructure? Can they move fast enough to deploy Gen AI but at the same time keep that AI on a tight leash to deliver fant...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/)** (1 messages): 

rumbleftw: Anyone with experience on finetuning instruct models with prompt template?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1247652004562735165)** (5 messages): 

- **cpu_ram_efficient_loading demystified**: A member clarified that `cpu_ram_efficient_loading` allows for sharded model pieces across GPUs instead of loading the entire model on each GPU. When set to `false`, the first worker holds all the model weights, while other workers hold skeleton weights and dispatch required weights dynamically utilizing FSDP.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1247392429380862043)** (2 messages): 

- **Multimodal finetuning query arises**: A member inquired whether **multimodal finetuning** is supported by **Axolotl**, specifically asking about *"IDEFICS 2"*. No answer followed the query.

- **Help request shared without context**: A user sought assistance and shared a [Discord link](https://discord.com/channels/1238365980128706560/1247260012741525549). However, no further details or responses were provided in this interaction.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1247629487152169101)** (19 messages🔥): 

- **CUDA book recommendation generates interest**: A member inquired about which CUDA book to recommend, mentioning an unspecific link that prompted a verification challenge. The CUDA book was confirmed by another member to be the same one mentioned in a presentation by Charles on June 4, 2024.
- **Quick work earns credit**: After quickly running a hello world example and resolving installation issues, a member inquired if they earned $500 credit for less than five minutes of work. Charles confirmed the credit and noted it could take up to a week for release.
- **Embedding model on Modal makes sense**: Members discussed hosting a sentence transformer embedding model on Modal, referencing an [example of running small bert-style embedding models on Wikipedia data](https://modal.com/blog/embedding-wikipedia). Charles affirmed it makes sense and suggested using the SBert library.
- **Handling errors in Modal's Hello World example**: A member sought help for a `Missing option '--i'` error while running Modal's hello world example. After sharing the code, another member identified a missing '@' before `app.local_entrypoint()`, resolving the error.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://a.co/d/1PbUJK7">no title found</a>: no description found</li><li><a href="https://modal.com/blog/embedding-wikipedia">Embedding English Wikipedia in under 15 minutes</a>: Leverage Modal’s parallel batch jobs and in-house storage features to quickly generate embeddings for billions of tokens.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247299495797063720)** (22 messages🔥): 

- **Caught the Credit Train with Jackie**: One user expressed gratitude for help in creating their account with Jackie. *"I was able to get my account created the other day with Jackie's help. 👍"*
- **Missed Forms, Hope for Credits**: There were several discussions about missing the deadline to submit forms but hoping to still receive credits. One user noted, *"I wasn't able to fill the forms, is it possible to consider credits at least on some of the vendors?".*
- **Braintrust Credits Applied for Users**: Ankur from Braintrust confirmed that credits have been applied, with 1366 people set up for accounts and upgraded to the Enterprise tier. *"The credit dollar amount... rounded up to 3 months since it will start immediately."*
- **Help with Tracking Credits**: Users were sharing updates on what credits they received and asking for a recap list. A helpful member provided a detailed list of expected credits from various vendors.
- **New Chance to Submit Forms**: Dan confirmed the possibility of receiving some credits even if forms were missed earlier. *"I'm going to try to get you credits... But we won't be able to get you everything on all platforms."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1247279775798792313)** (17 messages🔥): 

- **Check 'Upgrade' Button to Confirm Credits**: A member inquired about verifying credit application, and it was clarified that absence of an "Upgrade" button indicates active credits. *"If you see an 'Upgrade' button towards the top right, the credits have not been applied."*
- **Confusion between Different Braintrust Platforms**: Some users mistakenly tried to access the wrong Braintrust site, thinking their credits weren't applied. **Clarification was provided**: *"That's the wrong braintrust. we're braintrust.dev"*
- **Guidance on Braintrust Usage**: Guidance was offered to users on how to get started, including links to helpful resources. Members were directed to the [Braintrust Welcome Page](https://www.braintrust.dev/docs/welcome/start) and [Evals Guide](https://www.braintrust.dev/docs/guides/evals) for further information.
- **Additional URL for Onboarding**: For users wanting to re-enter the onboarding state, it was suggested to append `?onboarding=true` to the URL. *"You can add `?onboarding=true` to the URL."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.braintrust.dev/docs/cookbook/Text2SQL-Data">LLM Eval For Text2SQL</a>: Braintrust is the enterprise-grade stack for building AI products.</li><li><a href="https://www.braintrust.dev/docs/welcome/start">Braintrust</a>: Braintrust is the enterprise-grade stack for building AI products.</li><li><a href="https://www.braintrust.dev/docs/guides/evals">Braintrust</a>: Braintrust is the enterprise-grade stack for building AI products.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1247438137332990052)** (3 messages): 

- **Casual greetings in the channel**: Members shared casual greetings, with one member saying "*Salut!*" while another checked in from Malmö, Sweden. 
- **Expressing a busy schedule**: One member joked about meeting after the course ends due to their busy schedule, mentioning involvement in multiple activities including an MLOps course and work.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1247591929298812989)** (1 messages): 

- **Members receive Jarvis Labs credits**: *Thanks <@657253582088699918> for giving everyone credits to Jarvis Labs*. The [announcement](https://discord.com/channels/1238365980128706560/1241117895740625099/1247591436929466439) was shared, indicating a community-wide credit distribution.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1247591109413048371)** (6 messages): 

- **Engineering Feedback Loop**: A user mentioned they would add feedback to the items under discussion with engineering. *"That makes sense. I'll add it to the items to discuss with engineering, thanks for the feedback."*

- **Credits Distributed**: Predibase announced that credits have been distributed. Users were asked to email [support@predibase.com](mailto:support@predibase.com) if they did not receive them.

- **Missing Credits Issue**: Some users reported not seeing their credited amounts. One user specified having only 25 credits from signup and would contact support.

- **Fine-Tuning L3 70B Concerns**: A user reported issues with the L3 70B base model continuing to generate text after the ```<|end_of_text|>``` token. *"Using the adapter in the prompt tab when the base model generates  <|end_of_text|> it doesn't stop the generation, it lets the model continue writing non-sense."*

- **Generation Issues with Fine-Tuned Models**: Another issue was raised regarding the model repeating periods or period spaces during generation. *"If there's something I can do to decrease the frequency with which the fine-tuned base model just starts repeating periods or period space at the end of its generation that would also be helpful."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1247437655902392383)** (7 messages): 

- **Freelancing journey insight**: A seasoned industry professional shared their transition from academia to freelance work, highlighting the struggles and eventual success found through a supportive community. They emphasized the importance of community for support, networking, and inspiration.

- **Learning Generative AI in one year**: One user detailed their rapid learning journey over the past year, starting with no coding skills and progressing to deploying complex AI models and writing Python & JavaScript code. They encouraged others to start learning, highlighting community and assisted methods as catalysts for their progress.

- **From industry to ML**: A mechanical engineering professional shifted from industry R&D and medical device consulting to machine learning after encountering the fast.ai course. They have since applied ML in various startups, emphasizing their enjoyment in the transition and continued learning.

- **Excitement for AI learning journey**: A software developer with 17 years of experience expressed their newfound excitement for learning about large language models and machine learning as part of the course. They seek advice on resources for learning model fine-tuning and have project ideas related to fintech improvements.

- **Gaming achievement shared**: A user humorously mentioned reaching the diamond rank by playing random factions in a game, adding a light-hearted note to the discussion about professional experiences and learning paths.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247410269244624937)** (5 messages): 

- **Optimizing LLMs Guide Shared**: A team member from OpenAI shared a new guide on [optimizing LLMs for accuracy](https://platform.openai.com/docs/guides/optimizing-llm-accuracy). This guide covers prompt engineering, RAG, and fine-tuning, along with identifying what’s "good enough" for production for both business and technical stakeholders.
- **YouTube Talk on Maximizing LLM Performance**: Another member recommended checking out a [DevDay talk on YouTube](https://www.youtube.com/watch?v=ahnGLM-RC1Y) titled "A Survey of Techniques for Maximizing LLM Performance". This talk provides a comprehensive survey of techniques designed to unleash the full potential of language models.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=ahnGLM-RC1Y">A Survey of Techniques for Maximizing LLM Performance</a>: Join us for a comprehensive survey of techniques designed to unlock the full potential of Language Model Models (LLMs). Explore strategies such as fine-tunin...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1247410323565183029)** (2 messages): 

- **Curiosity about Jen-Hsun Huang's tech for digital humans**: A member asked, *"Does anyone know what technology Jen-Hsun Huang used to generate digital humans?"* The query remains unanswered in the chat.
  
- **Seeking efficient LLM training methods**: A member inquired about the best way to train LLMs efficiently on multiple GPUs, questioning whether FSDP (Fully Sharded Data Parallel) is the fastest option. They expressed concern about the complexities involved even if FSDP is used, highlighting *"time is the main issue here."*
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1247351502960332811)** (1 messages): 

- **Indexing Tensor with Shared Memory in Triton**: A member noted that directly **indexing a tensor in shared memory** is generally not feasible. They mentioned two methods: loading one row/column at a time or using `tl.split` in the latest Triton or Torch nightly builds, which only works for **power of 2**.
- **Triton's Block-Based Design Complicates Fine-Grained Indexing**: It's highlighted that **Triton is primarily block-based**, making fine-grained indexing awkward. This suggests why methods like `tl.split` might be necessary yet still suboptimal for certain scenarios.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1247332044854526012)** (3 messages): 

- **Debugging NHWC tensor normalization**: A member is seeking help with normalization on **NHWC tensors** and is currently debugging implementation issues in **ATen internals**. They asked the community for advice on what might be wrong with their implementation.
- **Using Metal Traces with MPS Profiler**: A discussion covered how to open metal traces using the **torch.mps.profiler**. One member shared that they figured out either to open **XCode Instruments directly and hit record**, or make a **project with Python** and run a performance scheme after asking for community input.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/_modules/torch/mps/profiler.html#start);">torch.mps.profiler &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/_modules/torch/mps/profiler.html#star">torch.mps.profiler &mdash; PyTorch 2.3 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1247341251951460437)** (7 messages): 

- **Considering a Switch to CUDA/Triton Programming**: A user expressed concerns about switching from general software development to CUDA/Triton programming, noting the drastic change and the potential job market limitation. They sought advice on whether this specialized field will have a lasting demand.

- **Reassurance about Learning CUDA in 30s**: In response, someone pointed out that they also learned CUDA in their 30s and hinted at the increasing demand for faster models, indirectly suggesting job security in the field.

- **Starting with CUDA or Triton**: The user was advised to start with the PMPP book up to chapter 6 to learn the basics of CUDA, but then transition to Triton for easier creation of decently performing kernels. Further, they were encouraged to move their question to another channel for increased visibility.
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1247325756405841942)** (11 messages🔥): 

- **Download CUDA for Nvidia Compilation**: A user advised to download CUDA from its [official website](https://developer.nvidia.com/cuda-downloads), mentioning that it comes as an .exe file. They also pointed out the need to add the compiler, nvcc, to the system PATH.
- **Torch.compile usage detailed**: A user inquired about using `torch.compile` on a function and its child function calls. Another member confirmed that compiling the parent function will also compile the child functions, with a suggestion to use `torch.compile(func, fullgraph=true)` for fusing the entire computation graph into one kernel.
- **Torch.compile adjustment using fine-grained APIs**: A user shared a [documentation link](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) for fine-grained control over `torch.compile`. This includes APIs like `torch.compiler.disable` to manage parts of the model where compilation should be skipped.
- **Gratitude for provided documentation**: The user expressed gratitude for the documentation link and mentioned that it was helpful in finding arguments for `torch.compile`. They are preparing to train a model and aim to save on compute requirements.

**Link mentioned**: <a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.3 documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1247352023310143619)** (15 messages🔥): 

- **Apple Metal kernel joins PyTorch**: A member shared an exciting update that an [Apple Metal kernel is coming to PyTorch AO](https://github.com/pytorch/ao/pull/311) with support for int4 quant metal kernels and detailed benchmarks.
- **Sparse benchmarking boosts**: Another member added [simple benchmarks for 2:4 sparsity](https://github.com/pytorch/ao/pull/303), reporting a 10-23% performance increase with minimal accuracy degradation on GPUs like the 3090 and 4090.
- **Debate on torch.ao.quantization deprecation**: One member inquired whether `torch.ao.quantization` would eventually be deprecated for `torchao`, with responses suggesting deprecation is not a current priority but will happen once more features and performance improvements are established.
- **Structured pruning efficiency**: A member explained that structured pruning can provide real inference time benefits on both CPU and GPU but may also cause accuracy issues compared to fine-grained sparsity like 2:4. Experimental support for structured pruning is available [here](https://github.com/pytorch/ao/tree/main/torchao/sparsity/prototype/pruner).
- **LLM-Shearing paper insights**: The discussion concluded with insights from the LLM-Shearing paper, highlighting that pruning models with non-uniform layer configurations might reduce model size but could increase inference time due to irregular memory access patterns.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/303">sparse benchmarking numbers by jcaip · Pull Request #303 · pytorch/ao</a>: Updated benchmark script for standalone sparse numbers. Switched from segment-anything to segment-anything-fast Updated README with results for segment-anything and BERT</li><li><a href="https://github.com/pytorch/ao/pull/311">Add int4 groupwise quantization metal kernels for linear layers by kimishpatel · Pull Request #311 · pytorch/ao</a>: This diff adds:  int4 quant metal kernels Add tests for it with group size = 32 using cmake build  TODO  Adding tests to CI Custom op integration into pytorch (helps with testing via python)  Kerne...</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity#sparsity-pattern">ao/torchao/sparsity at main · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity/prototype/pruner">ao/torchao/sparsity/prototype/pruner at main · pytorch/ao</a>: Native PyTorch library for quantization and sparsity - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1247331215099564082)** (4 messages): 

- **Community preferences for Discord servers**: A member inquired about recommendations for other Discord servers and mentioned knowing about **ML Ops**. Another member responded humorously, admitting they mostly check the affiliated server daily but are planning on creating a new one soon.

- **Explore State Space Duality on 'Goomblog'**: A link to [the goomblog](https://goombalab.github.io/blog/) was shared, featuring multiple parts of the **State Space Duality (Mamba-2)** series. This includes in-depth posts on the systems, algorithm, theory, and model behind Mamba-2, each discussing different aspects of the project.

**Link mentioned**: <a href="https://goombalab.github.io/blog/"> blog | Goomba Lab </a>: no description found

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1247405203926286336)** (1 messages): 

- **Chicago CUDA Enthusiasts Unite**: A member expressed excitement about finding another fellow CUDA learner in Lincoln Park, Chicago. They are keen on arranging a meeting and are **“100% down”** for it.

  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1247429955743383554)** (3 messages): 

- **Fancytrevor finds matrix representation impressive**: Fancytrevor commented that the visualization "looks like the whole matrix" but acknowledged it could also represent a part of the overall operation.
- **Calls for demonstration on window movement**: Fancytrevor suggested demonstrating "moving the window along K after reaching C[3,3]" if it were a CTA, implying a need for a clearer illustration of the window operation.
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247263810679603341)** (488 messages🔥🔥🔥): 

- **CUDA Compiler Controversy**: Users were perplexed about **CUDA code compiling errors** related to `__syncthreads` and code in `.cpp` files rather than `.cu`. A conversion from `.cpp` to `.cu` resolved issues, highlighting nuances between compiling CUDA and non-CUDA code.
- **Linker Hiccups: Inline to the Rescue**: A series of **linker errors** due to multiple definitions (e.g., `cudaCheck`, `deviceProp`) were mitigated by making functions and variables `inline`. Members discussed nuances of inline, static, and extern declarations.
- **Grad Mirroring Mishap**: A significant bug causing divergence in gradients between **single and multi-GPU** runs was traced to **gradients not being zeroed after PyTorch to C bridging**. This discrepancy was resolved, ensuring consistent results across different setups.
- **Refactoring Frenzy**: **Major code refactoring** isolated CUDA kernels into separate files, improving code organization and IDE performance. Discussions on further modularizing the code to better handle models like GPT-2 and potential ZeRO-2 implementations were prevalent.
- **Loss Calculation Head-scratchers**: An unexpected difference in loss calculations between single and multi-GPU setups in PyTorch was discovered to be due to **missing reduction of loss in distributed runs**. This was contrasted against the consistent results from llm.c, highlighting the intricacies of DDP in PyTorch.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/proftomyeh/status/1798042265883156651?s=46&t=ROCrCC19RlrPdFqCtEaiGA">Tweet from Tom Yeh | AI by Hand ✍️ (@ProfTomYeh)</a>: llm.c by Hand✍️  C programming +  matrix multiplication by hand  This combination is perhaps as low as we can get to explain how the Transformer works.   Special thanks to @karpathy for encouraging ea...</li><li><a href="https://en.wikipedia.org/wiki/Makedepend">makedepend - Wikipedia</a>: no description found</li><li><a href="https://x.com/karpathy/status/1013244313327681536?lang=en">Tweet from Andrej Karpathy (@karpathy)</a>: most common neural net mistakes: 1) you didn&#39;t try to overfit a single batch first. 2) you forgot to toggle train/eval mode for the net. 3) you forgot to .zero_grad() (in pytorch) before .backward...</li><li><a href="https://github.com/karpathy/llm.c/pull/536">move encoder, kernel 1/N by karpathy · Pull Request #536 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/548/">Fix zero grads bug by gordicaleksa · Pull Request #548 · karpathy/llm.c</a>: We have to zero the grads here because in the training loop we first do backward and only later we do zero grad. This causes a duplicate &quot;deposit&quot; of gradients in the first training step whi...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.py#L688">llm.c/train_gpt2.py at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1247298746912346153)** (39 messages🔥): 

- **uint2-7 in PyTorch remains dummy**: It has been confirmed that `uint2-7` types are currently "dummy" and not implemented in PyTorch, with plans for future utilization. This was backed up with a link to the relevant [GitHub code](https://github.com/pytorch/pytorch/blob/a4064da8cac7345fdf1ffb1f03262f9b235f37a0/c10/core/ScalarType.h#L28-L31).

- **Customizing dtype string-conversion**: Members discussed the challenge of `TrinaryTensor` auto-conversion from unit2 to uint8 when printed. Suggested solutions included overriding the `__repr__` method to correctly display values as -1, 0, 1 instead of 0, 1, 2.

- **Interesting paper on binary/trinary matrix multiplication**: A member shared an [interesting paper](https://arxiv.org/pdf/2205.09120) on binary/trinary matrix multiplication. Additional resources provided included links to [Cutlass BMMA](https://github.com/NVIDIA/cutlass/blob/ddd8f9cf4126dbd73b451d7cdd17aab7242fda53/include/cutlass/arch/mma_sm80.h#L2119) and [NVIDIA’s CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations).

- **Debugging PyTorch's 'cache_size_limit reached' issue**: Various strategies were discussed for debugging the `torch._dynamo.exc.Unsupported: cache_size_limit reached` error. Solutions involved marking parameters as dynamic, checking for graph breaks, and potentially increasing cache size to avoid recompiles. 

- **PyTorch FakeTensor issue resolution**: A proposed fix for the FakeTensor issue was shared with a link to a [GitHub pull request](https://github.com/pytorch/pytorch/pull/127927). This aims to address problems with tensor metadata dispatching in functional tensors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/pull/127927">FunctionalTensor: dispatch metadata directly to inner tensor by bdhirsh · Pull Request #127927 · pytorch/pytorch</a>: Fixes #127374 The error in the linked repro is: AssertionError: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with &#39;allow_non_fake_inputs&#39;. Found in aten.sym_st...</li><li><a href="https://github.com/pytorch/pytorch/blob/a4064da8cac7345fdf1ffb1f03262f9b235f37a0/c10/core/ScalarType.h#L28-L31">pytorch/c10/core/ScalarType.h at a4064da8cac7345fdf1ffb1f03262f9b235f37a0 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/NVIDIA/cutlass/blob/ddd8f9cf4126dbd73b451d7cdd17aab7242fda53/include/cutlass/arch/mma_sm80.h#L2119>">cutlass/include/cutlass/arch/mma_sm80.h at ddd8f9cf4126dbd73b451d7cdd17aab7242fda53 · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1247326741815627836)** (22 messages🔥): 

- **ARM for Performance: Concerns Raised and Resources Shared**: A member expressed concerns about buying an X Elite and sought information on support for Hexagon SDK for FPU and Adreno SDK for GPU. A response clarified that while ExecuTorch supports Hexagon, Adreno is not CUDA compatible and PyTorch lacks robust Vulkan/OpenGL backends.

- **Deep Dive into ARM's Capabilities**: Relevant resources were shared, including a [PDF on ARM's performance](https://www.dcs.warwick.ac.uk/pmbs/pmbs22/PMBS/talk10.pdf) and a [blog post on ARM's Scalable Matrix Extension](https://newsroom.arm.com/blog/scalable-matrix-extension).

- **Discussing Adreno and SNPE**: A discussion unfolded about using SNPE for Adreno, highlighting its OpenCL base and the complexities involved in making models compatible. One user mentioned that some systems block direct access to OpenCL, adding another layer of difficulty.

- **Clarifying QNN vs. SNPE**: There was a debate on whether Qualcomm's QNN is different from SNPE, with some asserting it's merely a rebranding while others pointed out roadmap and team differences. Documentation confusion and outdated support for certain operations were noted as ongoing issues.

- **ARM SME, SVE, and Hardware Challenges**: A member mentioned plans to implement an SVE2 GEMM but postponed due to the unavailability of hardware, intending to use llvm-mca for metric evaluations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction">Arm Scalable Matrix Extension (SME) Introduction </a>: This blog series provides an introduction to the Arm Scalable Matrix Extension (SME) including SVE and SVE2.</li><li><a href="https://developer.arm.com/documentation/109246/0100/matmul-int8--8-bit-integer-to-32-bit-integer-matrix-by-matrix-multiplication/Overview-of-the-matmul-int8-algorithm?lang=en">Documentation – Arm Developer</a>: no description found</li><li><a href="https://newsroom.arm.com/blog/scalable-matrix-extension">Introducing Armv9 Scalable Matrix Extension for AI Innovation on the Arm CPU</a>: New Armv9 architecture feature offers significant performance uplifts for AI and ML-based applications, including generative AI.</li><li><a href="https://github.com/google/gemmlowp">GitHub - google/gemmlowp: Low-precision matrix multiplication</a>: Low-precision matrix multiplication. Contribute to google/gemmlowp development by creating an account on GitHub.</li><li><a href="https://github.com/kexinzhao/farm">GitHub - kexinzhao/farm: Fast matrix multiplication library for ARM CPUs</a>: Fast matrix multiplication library for ARM CPUs. Contribute to kexinzhao/farm development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247270329529663600)** (427 messages🔥🔥🔥): 

- **RoPE `llama-3-8b` to 64k tokens requires caution**: A user attempted to extend `llama-3-8b` to handle **64k tokens** but faced significant **VRAM issues**, including a spike that caused an OutOfMemoryError on an H100 with 80GB. They shared the detailed results and discussed possible solutions, including tuning configurations and gradient checkpointing.
- **Cost-effective optimization via Unsloth**: The Unsloth team engaged in discussions about their framework's use in **fine-tuning models like Llama 3 and Mistral**, emphasizing cost and memory efficiency improvements. They also clarified queries related to Unsloth applications in sequence classification and continuous pretraining, pointing to [their latest blog post on continued pretraining](https://unsloth.ai/blog/contpretraining) and setup [guides on GitHub](https://github.com/unslothai/unsloth).
- **Phi-3 model evaluations get mixed reactions**: A user considered giving Phi3 another try for a data extraction task, and it sparked a debate over the **inconsistent performance** reported by others. Some found great success while others noted subpar results, highlighting the model's variances.
- **Latent VRAM optimization insights**: Through experiments, it was noted that the **RoPE extension** of models might see a substantial VRAM spike initially, which then settles. This discovery suggests potential to optimize and manage memory more effectively, possibly extending capacities even further.
- **Community engagement and resources**: Members shared links to helpful resources, like [Daniel Han’s live event](https://meet.google.com/sxp-ekzv-osb) recording, and engaged in-depth on fine-tuning practices, further boosting collective knowledge and troubleshooting methods in model training and optimization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/wWJCq9PU?event=1244218238288531457">Join the Aleksa Gordić - The AI Epiphany Discord Server!</a>: Machine Learning. | 7662 members</li><li><a href="https://huggingface.co/fimbulvntr/lewd-stories/tree/main">fimbulvntr/lewd-stories at main</a>: no description found</li><li><a href="https://wandb.ai/unnamed_org/text-novel-completion-ropes/reports/Progressively-RoPE-llama-3-70b-bnb-4bit--Vmlldzo4MjAzMDI1">Weights & Biases</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/FasterDecoding/Medusa">GitHub - FasterDecoding/Medusa: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads</a>: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads - FasterDecoding/Medusa</li><li><a href="https://www.youtube.com/watch?v=v_q2JTIqE20">GPU optimization workshop (hosted by @ChipHuyen )</a>: 00:30 Workshop overview03:51 Crash course to GPU optimization (Mark Saroufim, Meta)39:18 High performance LLM serving on NVIDIA GPUs (Sharan Chetlur, NVIDIA)...</li><li><a href="https://api.wandb.ai/links/unnamed_org/yyl8ymr1">Progressively RoPE llama-3-70b-bnb-4bit</a>: Run on an H100 SXM (runpod) Dataset consists of samples whose length equals (tokens): - 4k - 8k - 16k - 32k - 48k - 64k (crashed)  Run name represents the context length that was used</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://x.com/UnslothAI/status/1798088790919332013">Tweet from Unsloth AI (@UnslothAI)</a>: Unsloth now allows you to do continued pretraining with QLoRA 2x faster and use 50% less VRAM than Hugging Face+FA2.  Continued pretraining allows models to train on new domain data.  Read our blog: h...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1247651391212621824)** (1 messages): 

- **Unsloth Continually Pretrains LLMs Faster**: [Unsloth’s new release](https://github.com/unslothai/unsloth/releases/tag/June-2024) allows you to continually pretrain LLMs **2x faster** while using **50% less VRAM** than Hugging Face + Flash Attention 2 QLoRA. The release is detailed in their [blog post](https://unsloth.ai/blog/contpretraining).

- **Free Notebooks for Learning and Text Completion**: Unsloth provides free Colab notebooks for hands-on experience. Access the [Continuous Pretraining notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) for Mistral v0.3 7b and the [Text Completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing).

- **Technical Insights on Pretraining**: Key insights include finetuning embeddings, offloading embeddings to disk, and using different learning rates for embeddings to stabilize training. They also recommend using Rank stabilized LoRA to improve performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1247284318020046943)** (18 messages🔥): 

- **Jailbreak Prompt Fun Shared**: A user shared a jailbreak prompt from [Hugging Face](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) and suggested trying it for fun. The prompt involves creating a story between two entities, Tom and Jerry, detailing a technical instruction.

- **Multi-GPU Support Coming Soon**: Members discussed the possibility of multi-GPU support in Unsloth AI. One user was skeptical, but another confirmed, "we are rolling out support for multigpu soon but it will take some time as always".

- **Multi-GPU Performance Insights**: It was noted that Unsloth AI is faster on a single A100 GPU compared to using two GPUs. A user working on multi-GPU setups expressed concerns about future training scalability, asking about using "device = 'cuda'".

- **Integration with Ray for Finetuning**: A query was raised about potential future usability through Ray for finetuning over multiple nodes when multi-GPU support becomes available. This showcases users' interest in optimizing high-performance model training across multiple GPUs.

**Link mentioned**: <a href="https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts?">rubend18/ChatGPT-Jailbreak-Prompts · Datasets at Hugging Face</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247265389570818240)** (141 messages🔥🔥): 

- **Guide for Local Setup of Unsloth**: Queries about setting up Unsloth locally were addressed by directing users to install Jupyter Notebook and refer to the [readme](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions) on the Unsloth GitHub page. Docker images for ease of use were also suggested, linked to the [Jiar/jupyter4unsloth repository](https://github.com/Jiar/jupyter4unsloth).
  
- **Challenges with Custom Train Loop**: Members discussed issues with custom training loops, where one user faced a `RuntimeError`. Attempts to debug included switching model sources, but the exact cause remained elusive.
  
- **Discussion on Fine-Tuning with 8-bit Quantization**: Users inquired about the feasibility of fine-tuning with 8-bit quantization on models such as phi-3-medium-4k. The community clarified that this is currently unsupported and further research is being conducted.
  
- **Project Feasibility Using Llama 3 for Sentiment Analysis**: Members debated the feasibility of using Llama 3 for sentiment analysis within a short project timeframe. It was suggested to use embeddings like BERT instead, with considerations to enhance capabilities through COT (Chain of Thought) and guided generation.
  
- **LoRA Rank Calculation and Alpha Discussion**: Detailed discussions took place regarding the calculation of LoRA ranks and setting the alpha parameter, citing heuristics and experimental observations. The article [from Lightning AI](https://lightning.ai/pages/community/lora-insights/) was referenced for further insights on optimal LoRA settings.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Jiar/jupyter4unsloth">GitHub - Jiar/jupyter4unsloth: Jupyter for Unsloth</a>: Jupyter for Unsloth. Contribute to Jiar/jupyter4unsloth development by creating an account on GitHub.</li><li><a href="https://lightning.ai/pages/community/lora-insights/">Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments - Lightning AI</a>: LoRA is one of the most widely used, parameter-efficient finetuning techniques for training custom LLMs. From saving memory with QLoRA to selecting the optimal LoRA settings, this article provides pra...</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/27fa021a7bb959a53667dd4e7cdb9598c207aa0d/unsloth/models/llama.py#L1196">unsloth/unsloth/models/llama.py at 27fa021a7bb959a53667dd4e7cdb9598c207aa0d · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/27fa021a7bb959a53667dd4e7cdb9598c207aa0d/unsloth/models/llama.py#L1175C9-L1175C36)">unsloth/unsloth/models/llama.py at 27fa021a7bb959a53667dd4e7cdb9598c207aa0d · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1247468974929608847)** (3 messages): 

- **New Member Joins the Fray**: A new member, **s3nh1123**, enthusiastically joined the community, expressing their happiness with *"great to be here 💚💚."* Members **theyruinedelise** quickly welcomed them, reiterating the sentiment with *"It's great to have you here!"*
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1247273615901065248)** (556 messages🔥🔥🔥): 

- **Perplexity Faces Academic Query Challenges**: A user expressed frustration that Perplexity's academic-focused search yielded results primarily from **Wikipedia** and only one source from **Britannica**, questioning its effectiveness for academic searches. They provided a link to the search results, calling it a potential bug to report. [Link to search](https://www.perplexity.ai/search/I-want-to-ZoV4zN4LRKa2YcbFG52K.Q)

- **Server Outages Spark Frustration**: Numerous users reported issues with **Perplexity**, **ChatGPT**, and other AI services experiencing simultaneous outages. This incident led to speculation about a larger infrastructure problem, potentially related to **AWS** or other common service providers.

- **Concerns Over Opus 50 Limitation**: Members expressed dissatisfaction with the reduced **Opus 50** limit, reminiscing about the previous **Opus 600** limit. There is a sense of frustration due to the lack of communication from Perplexity regarding potential adjustments or the return of the higher limit.

- **Comparing Perplexity and ChatGPT**: Users debated the advantages and disadvantages of **Perplexity AI Premium** versus **ChatGPT**, noting Perplexity's superior web search capabilities and more extensive model selection. However, there were mixed feelings about subscription limits and use cases for both platforms.

- **Technical and Presentation Assistance**: Members offered advice for school presentations about AI, recommending outlining advantages and risks, and using AI tools like Perplexity and ChatGPT to prepare content. Useful links and videos were shared to help understand the technical concepts of AI. [YouTube Video](https://youtu.be/wjZofJX0v4M).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.openai.com/">OpenAI Status</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=OkJnY8bpXAM&t=744s&pp=ygUaaGFja2VycyBzb21lb3JkaW5hcnlnYW1lcnM%3D">The Feds Just Launched The Largest Attack On Hackers...</a>: Hello guys and gals, it&#39;s me Mutahar again! This time we take a look at what appears to be Operation Endgame, all the police agencies of the world are launch...</li><li><a href="https://tenor.com/view/server-is-fine-burn-fire-spongebob-blow-gif-14178373">Server Is Fine Burn GIF - Server Is Fine Burn Fire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/terminator-smiling-arnold-schwarzenegger-gif-16197002">Terminator Smiling GIF - Terminator Smiling Arnold Schwarzenegger - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273">Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/down-town-la-bomb-dtlablowup-gif-21604415">Down Town La Bomb Dtlablowup GIF - Down Town La Bomb Dtlablowup - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://tenor.com/view/kristanna-loken-tx-melt-down-terminator-rise-of-the-machines-gif-24408281">Kristanna Loken Tx GIF - Kristanna Loken TX Melt Down - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/skinet-skinet-remember-maquinas-revolucion-gif-7115195292076587720">Skinet Skinet Remember GIF - Skinet Skinet remember Maquinas - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/terminator-gif-21649154">Terminator GIF - Terminator - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://msty.app/">Msty - Running LLMs made simple and easy.</a>: Msty allows you to use local and online LLMs in the simplest way. Its chat interface packed with powerful features make it easy to use LLMs. Run Ollama models such as Mixtral, Llama2, Qwen, or online ...</li><li><a href="https://tenor.com/view/robot-tech-skynet-ai-arnold-gif-13186419">Robot Tech GIF - Robot Tech Skynet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=OFS90-FX6pg&t=612s">ChatGPT: 30 Year History | How AI Learned to Talk</a>: This video explores the journey of AI language models, from their modest beginnings through the development of OpenAI&#39;s GPT models. Our journey takes us thro...</li><li><a href="https://downforeveryoneorjustme.com/perplexity">Perplexity down? Check here and read user reports. - DownFor</a>: Perplexity won't load? Or getting an error? Check the real-time status and see what other Perplexity users are reporting.</li><li><a href="https://www.cloudflarestatus.com/">Cloudflare Status</a>: no description found</li><li><a href="https://youtu.be/wjZofJX0v4M?feature=shared">But what is a GPT?  Visual intro to transformers | Chapter 5, Deep Learning</a>: no description found</li><li><a href="https://tenor.com/view/larry-david-seinfeld-pretty-good-enthusiasm-curb-gif-3938316">Larry David Pretty Good GIF - Larry David Seinfeld Pretty Good - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://poe.com/login">Poe - Fast, Helpful AI Chat</a>: no description found</li><li><a href="https://copilot.microsoft.com/">Microsoft Copilot: 你的日常 AI 助手</a>: Microsoft Copilot 利用 AI 的强大功能来提高工作效率、释放创造力，并通过简单的聊天体验帮助你更好地理解信息。</li><li><a href="https://downdetector.com/">Downdetector</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1247267604784087111)** (11 messages🔥): 

- **Les Derniers Developpment Link Shared**: A member shared a [link](https://www.perplexity.ai/search/Les-deniers-dveloppent-m.hwMU.EQTCIy.9YeH5xoQ) to a Perplexity AI search about the latest developments.
- **AMD Recent News Link**: A member posted a [link](https://www.perplexity.ai/search/AMD-a-rcemment-kVmULrxBT7yTJm0eKCmkvA) to news involving AMD's recent activities.
- **Discord Search and Truth Link**: Multiple members shared links directing to a Perplexity AI search on [Discord](https://www.perplexity.ai/search/discordperplexi-tDIxu6YHQlaSKrREsdB5MQ) and a page about the [truth](https://www.perplexity.ai/page/Truth-About-the-yulN9Dp4T2K_EeOonc52PQ).
- **Perplexity AI Sharing Thread Reminder**: Perplexity AI prompted several users to ensure their thread is "Shareable," providing a [Discord link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) for further instruction.
- **Request to Repeat Link**: A user shared a [link](https://www.perplexity.ai/search/httpsdiscordcom-repeat-this-p6zDFgNJS5Wn4D4YdmeEGg#0) asking others to repeat the process mentioned in the linked page.
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1247654554091389079)** (1 messages): 

- **FineWeb Report goes public**: The **FineWeb technical report** has been released, shedding light on *every processing decision* and introducing the *FineWeb-Edu dataset*. The report aims to explain high-performing models like Llama3 and GPT-4 + Mixtral [here](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
- **Transformers.js lands in Firefox 130**: **Transformers.js** will now be part of **Firefox 130** with *fully private on-device AI*. The first use-case is automatic alt-text generation for images, significantly boosting accessibility, detailed more [here](https://mzl.la/4aPeBFL).
- **Nvidia NIM available on HF Inference Endpoints**: **Nvidia NIM** is now deployable via **Hugging Face Inference Endpoints**, starting with Llama 3 8B and 70B models on AWS and GCP. It promises up to 9000 tokens/sec performance, with more models to come [here](https://x.com/_philschmid/status/1797713003778883858).
- **Gradio Clients 1.0**: The **Gradio Clients 1.0 launch event** is set, advocating high-performance and scalable Gradio applications from prototypes to production-ready APIs. The event details can be found [here](https://discord.com/events/879548962464493619/1245020251611992154).
- **Sentence Transformers v3 Blog**: A new blog demonstrates fine-tuning embedding models for financial RAG applications using NVIDIA's 2023 SEC Filing dataset. Performance improvements between 7.4% to 22.55% were achieved using synthetic data and advanced techniques like Matryoshka Representation Learning, detailed [here](https://www.philschmid.de/fine-tune-embedding-model-for-rag).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1797173053123916036)">Tweet from Guilherme Penedo (@gui_penedo)</a>: We are (finally) releasing the 🍷 FineWeb technical report!  In it, we detail and explain every processing decision we took, and we also introduce our newest dataset: 📚 FineWeb-Edu, a (web only) subs...</li><li><a href="https://x.com/xenovacom/status/1797285648572821840)">Tweet from Xenova (@xenovacom)</a>: Transformers.js is being added to Firefox 130! 🤯 That’s right, fully private on-device AI directly in your browser! 🔥  The first use-case they’re exploring is automatic alt-text generation for image...</li><li><a href="https://x.com/Gradio/status/1795561025397256498)">Tweet from Gradio (@Gradio)</a>: 🚀𝐏𝐫𝐨𝐭𝐨𝐭𝐲𝐩𝐞𝐬 𝐭𝐨 𝐏𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧!  🙌Join us for the much-anticipated Launch Event for Gradio Clients 1.0 on June 6.   🤩Understand how your Gradio applications exhibit high performanc...</li><li><a href="https://x.com/kamilakesbi/status/1796537200961785931)">Tweet from Kamil Akesbi (@kamilakesbi)</a>: The biggest barrier to speaker diarization ? Data!  With 🤗 Diarizers, you can now generate synthetic meeting 🗣️ conversations!  Starting from an ASR dataset, you can create arbitrary amounts of data...</li><li><a href="https://x.com/_philschmid/status/1797713003778883858)">Tweet from Philipp Schmid (@_philschmid)</a>: Yesterday at COMPUTEX, Jensen Huang announced the release of @nvidia NIM on @huggingface Inference Endpoints! 🚀 NVIDIA NIM are inference services designed to streamline and accelerate the deployment ...</li><li><a href="https://x.com/_philschmid/status/1795804027621404975)">Tweet from Philipp Schmid (@_philschmid)</a>: Product Update: @nvidia L4s are now available in @huggingface  Inference Endpoints on AWS!  Enjoy up to 8x L4s per user and organization, and save 20% compared to on-demand AWS EC2. 🤑  - 1x NVIDIA L4...</li><li><a href="https://x.com/abhi1thakur/status/1795477747701104651)">Tweet from abhishek (@abhi1thakur)</a>: AutoTrain just got a brand new UI 🚀🚀🚀</li><li><a href="https://x.com/_philschmid/status/1797994961197031703">Tweet from Philipp Schmid (@_philschmid)</a>: Excited to share a new blog on how to fine-tune embedding models for financial RAG applications using NVIDIA&#39;s 2023 SEC Filing dataset using the latest research, like Matryoshka Representation Lea...</li><li><a href="https://x.com/frimelle/status/1797619351954260214)">Tweet from Lucie-Aimée Kaffee (@frimelle)</a>: Community-centric and awesome: @huggingface and @Wikimedia 🤗 I wrote an article on how we can advance ML with diverse datasets from @Wikipedia, why and how to create more Wikimedia datasets on Huggin...</li><li><a href="https://x.com/NielsRogge/status/1796213271189438888)">Tweet from Niels Rogge (@NielsRogge)</a>: Alright finally back on @YouTube with a new video: fine-tuning PaliGemma (or LLaVa, Idefics2,...) on your custom dataset!  I&#39;m fine-tuning in @GoogleColab on an L4 GPU   I go over many things like...</li><li><a href="https://x.com/abhi1thakur/status/1796210385579639144)">Tweet from abhishek (@abhi1thakur)</a>: 🚨 NEW BLOG: How to Fine-Tune Custom Embedding Models Using AutoTrain Learn: - what should be the data format - how to map columns properly - example datasets - custom configs - train locally - train ...</li><li><a href="https://x.com/vanstriendaniel/status/1795875763557904753">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Do you need a dataset to train a custom sentence transformer model? I&#39;ve created a pipeline for using an LLM to create a synthetic dataset you can directly use for fine-tuning/training a Setence T...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247267022434209792)** (418 messages🔥🔥🔥): 

- **Activity Tracker v1.0 for LevelBot Launched**: Lunarflu introduced an [updated activity tracker](https://huggingface.co/posts/lunarflu/239147617114976) for LevelBot v1.0 and invited suggestions for improvements, encouraging the community to open PRs. Initial ideas include tracking more types of actions, creating bigger plots, and integrating Discord activities and GitHub links.
- **Seeking Open Source Chat Assistant UI**: A member asked if anyone knew of an open-source version of [HF's chat assistant UI](https://huggingface.co/chat/assistants) for customizing prompts, vector databases, and RAG parameters. The lack of responses indicates a potential gap in available resources.
- **Fine-Tuning HuggingFace Models**: Robin_01_ inquired about the effectiveness of fine-tuning models like Mistral 7B Instruct v0.3 on small custom datasets for better instruction following; the community's response hinted at the usefulness of such endeavors.
- **Training Guide for Hugging Face Models**: Pranavadvani2003 sought assistance in setting up the Transformers library for open-source contributions. The advice involved specifying the correct TensorFlow version, noting IDE issues with dependencies, and using alternative load methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://kingnish-sdxl-flash.hf.space'">no title found</a>: no description found</li><li><a href="https://kingnish-image-gen-pro.hf.space'">no title found</a>: no description found</li><li><a href="https://fluently-fluently-playground.hf.space'">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.20279">CV-VAE: A Compatible Video VAE for Latent Generative Video Models</a>: Spatio-temporal compression of videos, utilizing networks such as Variational Autoencoders (VAE), plays a crucial role in OpenAI&#39;s SORA and numerous other video generative models. For instance, ma...</li><li><a href="https://huggingface.co/posts/lunarflu/239147617114976">@lunarflu on Hugging Face: &quot;By popular demand, HF activity tracker v1.0 is here! 📊 let&#39;s build it…&quot;</a>: no description found</li><li><a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>: LLM System Eval 101 - Build better agentsGet free HubSpot report of how to land a Job using AI: https://clickhubspot.com/fo2🔗 Links- Follow me on twitter: h...</li><li><a href="https://www.youtube.com/watch?v=fc_NSAu41b0">Create A Personalized AI Chatbot with ChatRTX</a>: Create a personalized chatbot with the ChatRTX tech demo.  Accelerated by TensorRT-LLM and Tensor Cores, you can quickly get tailored info from your files an...</li><li><a href="https://tenor.com/view/big-brain-gif-27108854">Big Brain GIF - Big Brain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros">python how to pad numpy array with zeros</a>: I want to know how I can pad a 2D numpy array with zeros using python 2.6.6 with numpy version 1.5.0. But these are my limitations. Therefore I cannot use np.pad. For example, I want to pad a with ...</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion">Outpainting II - Differential Diffusion</a>: no description found</li><li><a href="https://huggingface.co/spaces/HengJay/snomed-ct-assistant">SNOMED CT Assistant - a Hugging Face Space by HengJay</a>: no description found</li><li><a href="https://tenor.com/view/cat-eating-eatin-gamer-gunk-gamer-gunk-cat-monkey-cat-gamer-gif-20643451">Cat Eating Eatin Gamer Gunk GIF - Cat Eating Eatin Gamer Gunk Gamer Gunk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://groq.com/">Groq builds the world&#x27;s fastest AI inference technology</a>: The LPU™ Inference Engine by Groq is a hardware and software platform that delivers exceptional compute speed, quality, and energy efficiency. Groq provides cloud and on-prem solutions at scale for AI...</li><li><a href="https://www.youtube.com/watch?v=qDXa2rUdia0">Long memory (ChromaDB Infinity context) - SillyTavern AI</a>: ChromaDB stores each of your chat messages in the database, and only outputs them if the context matches, thus &quot;remembering previous events&quot;.Colab link - htt...</li><li><a href="https://www.producthunt.com/posts/asknews"> AskNews - News, when quality matters | Product Hunt</a>: AskNews is re-imagining how news is consumed by humans and LLMs alike. We provide human editorial boosted by AI-powered insights to minimize bias and build a transparent view of current events. Meanwh...</li><li><a href="https://x.com/karpathy/status/1797313173449764933">Tweet from Andrej Karpathy (@karpathy)</a>: Awesome and highly useful: FineWeb-Edu 📚👏 High quality LLM dataset filtering the original 15 trillion FineWeb tokens to 1.3 trillion of the highest (educational) quality, as judged by a Llama 3 70B....</li><li><a href="https://x.com/karpathy/status/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: Official repository of the xLSTM.</a>: Official repository of the xLSTM. Contribute to NX-AI/xlstm development by creating an account on GitHub.</li><li><a href="https://discuss.huggingface.co/t/unable-to-load-saved-tokenizer/86631">Unable to load saved tokenizer</a>: I’m able to successfully train and save my tokenizer but then i cant reload it.  tokenizer.save(tokenizer_save_path+&quot;tokenizer.json&quot;) #works newTokenizer = Tokenizer.from_file(tokenizer_save...</li><li><a href="https://www.gradio.app/guides/interface-state">Interface State</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://github.com/pranav-bot/transformers">GitHub - pranav-bot/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - pranav-bot/transformers</li><li><a href="https://colab.research.google.com/drive/1GSqPM13lS-vTH94dhMP1O2uwrJMbCe0e">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF">bartowski/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

asuka_minato: yes，origin is in torch and deployed on colab。but need on edge inference，so riir
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1247443621419155498)** (4 messages): 

- **Study on preference alignment in language models**: The paper [Alignedge](https://arxiv.org/abs/2403.07691) highlights the importance of supervised fine-tuning (SFT) for preference alignment in language models. It introduces the ORPO algorithm, which optimizes preference alignment without needing an additional phase, showing empirical and theoretical benefits across various model sizes.

- **New YouTube video linked**: A link to a YouTube video was shared [here](https://www.youtube.com/watch?v=PeSLWTZ1Yg8).

- **German parliament speech dataset for ASR/TTS**: A member shared a new dataset containing 610 hours of transcribed audio samples from the German parliament, available on [Hugging Face](https://huggingface.co/datasets/D4ve-R/bundestag-asr). This dataset can be used for Automatic Speech Recognition (ASR) and Text-To-Speech (TTS) training.

- **Visualizing Transformer workings**: A [tweet](https://x.com/ProfTomYeh/status/1798042265883156651) was shared showing Prof. Tom Yeh's project on explaining Transformers with C programming and matrix multiplication by hand. This exercise aims to help people better understand the workings of large language models (LLMs).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.07691">ORPO: Monolithic Preference Optimization without Reference Model</a>: While recent preference alignment algorithms for language models have demonstrated promising results, supervised fine-tuning (SFT) remains imperative for achieving successful convergence. In this pape...</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">Tweet from Tom Yeh | AI by Hand ✍️ (@ProfTomYeh)</a>: llm.c by Hand✍️  C programming +  matrix multiplication by hand  This combination is perhaps as low as we can get to explain how the Transformer works.   Special thanks to @karpathy for encouraging ea...</li><li><a href="https://huggingface.co/datasets/D4ve-R/bundestag-asr">D4ve-R/bundestag-asr · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1247324607304699934)** (5 messages): 

- **New Preprint on Corrupted Pre-training Data Released**: A member shared a preprint in collaboration with Microsoft and CMU, discussing the impact of corrupted datasets in pre-training diffusion models. The study found that "slight corruption in pre-training can significantly enhance the quality, diversity, and fidelity of generated images" ([arXiv:2405.20494](https://arxiv.org/abs/2405.20494)).

- **Security Awareness Video Highlighted**: A YouTube video titled "[Signs that your Computer has been Hacked](https://youtu.be/jG56MKen6YM?si=JCjIayR9tCb38zHa)" was shared to help users recognize the warning signs of a hacked system. The video emphasizes the importance of maintaining cybersecurity.

- **Framework for Image Regression Introduced**: A member announced the creation of a new framework for **Image Regression** using PyTorch and Transformers, which integrates seamlessly into the 🤗 ecosystem. This framework trains the model, uploads it to the HuggingFace hub, and automatically generates model cards, providing instructions and metadata ([Blog post](https://huggingface.co/blog/tonyassi/image-regression)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/tonyassi/image-regression">Sales Forecasting with Image Regression</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.20494">Slight Corruption in Pre-training Data Makes Better Diffusion Models</a>: Diffusion models (DMs) have shown remarkable capabilities in generating realistic high-quality images, audios, and videos. They benefit significantly from extensive pre-training on large-scale dataset...</li><li><a href="https://youtu.be/jG56MKen6YM?si=JCjIayR9tCb38zHa">Signs that your Computer has been Hacked</a>: Recognizing the warning signs of a hacked system is crucial for protecting your data and maintaining cybersecurity. In this video we will discuss about the s...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1247501795501281360)** (1 messages): 

- **Hunyuan DiT pipeline now available in diffusers**: The team just released a patch to include the **Hunyuan DiT pipeline** through `diffusers` contributed by one of Hunyuan's authors, Xingchao Liu. For more details, check the [release notes](https://github.com/huggingface/diffusers/releases).

**Link mentioned**: <a href="https://github.com/huggingface/diffusers/releases">Releases · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1247275363046916208)** (11 messages🔥): 

- **Exploring Vision-Language Models (VLMs) with New Paper**: A paper on [Vision-Language Models (VLMs)](https://huggingface.co/papers/2405.17247) outlines the intricacies and challenges of integrating visual data with language models. It discusses VLM applications and provides a primer on how these models are trained and evaluated.

- **Guidance on Fine-Tuning VLM for OCR in New Language**: Members discuss fine-tuning a pretrained VLM for OCR tasks in a new language. The recommended approach includes ensuring the tokenizer supports the target language and using a dataset with the primary fields: the document image and the extracted text.

- **Training ResNet-50 with 600 Images**: A member requests guidance on training a ResNet-50 model with a single class using 600 collected images. Issues of overfitting and incorrect predictions on new data arise, and the discussion covers methods to address these problems.

- **Regularization Methods for Overfitting**: To combat overfitting, a member is directed to a [Coursera course](https://www.coursera.org/learn/deep-neural-network) focused on regularization methods. The advice sparked a debate on the balance between providing direct help and teaching foundational knowledge.

- **Improving Classification Model Outputs**: A suggestion is made to modify the output layer in a classification model to include two outputs (True/False) with a softmax function instead of using a single sigmoid output. This approach can simplify threshold adjustments and improve prediction clarity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.coursera.org/learn/deep-neural-network)">Coursera | Online Courses From Top Universities. Join for Free</a>: 7,000+ courses from schools like Stanford and Yale - no application required. Build career skills in data science, computer science, business, and more.</li><li><a href="https://www.coursera.org/learn/deep-neura">Coursera | Online Courses From Top Universities. Join for Free</a>: 7,000+ courses from schools like Stanford and Yale - no application required. Build career skills in data science, computer science, business, and more.</li><li><a href="https://huggingface.co/papers/2405.17247">Paper page - An Introduction to Vision-Language Modeling</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1247642239933616207)** (1 messages): 

- **Debugging Function Import Issues**: One user suggested that a possible **version mismatch** might be causing issues with importing a specific function. They recommended attempting to explicitly import the desired function to see if it resolves the problem.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247275018321264847)** (2 messages): 

- **Exploring Diffusion Policy for Visuomotor Learning**: A blog post discusses the [Action Chunking Transformer](https://radekosmulski.com/how-to-train-your-robot-with-a-transformer/) and its mechanism to generate actions through an encoder-decoder model, translating learned embeddings into predicted trajectories. The post also mentions the "[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137?ref=radekosmulski.com)" paper, which starts with Gaussian noise for generating predicted actions.
- **Optimizing SDXL Inference with JIT Trace**: A member asks for an example script using `jit.trace()` for SDXL, similar to the one provided for `stable-diffusion-v1-4` in the [Diffusers optimization guide](https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing). The mentioned techniques include enabling cuDNN auto-tuner and using `autocast (fp16)`, notably speeding up inference from 9.50s to 3.21s on an NVIDIA TITAN RTX.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/v0.6.0/en/optimization/fp16#tracing">Memory and speed</a>: no description found</li><li><a href="https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/">Diving into Diffusion Policy with LeRobot</a>: In a recent blog post, we looked at the Action Chunking Transformer (ACT).  At the heart of ACT lies an encoder-decoder transformer that when passed in   * an image  * the current state of the robot  ...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247283746403516468)** (136 messages🔥🔥): 

- **Debating Real AGI**: A lively discussion unfolded about the nature of AGI, with members sharing humorous takes such as, "*real AGI loans me money*" and "*real AGI knows how to plug USB correctly (every single time)*". The consensus leaned towards the belief that true AGI should perform complex, human-like tasks continuously and adaptively.
- **Medical Research AI Alternatives**: Recommendations for AI tools like [Elicit](https://elicit.com) and Perplexity were highlighted for tasks like summarizing research papers and answering complex questions. Elicit's features were praised for summarizing papers and synthesizing findings efficiently.
- **ChatGPT Outage Theories**: Members speculated about the ChatGPT outage, suspecting issues with backend providers like Cloudflare or Azure. It was later suggested that a DDoS attack by Anonymous Sudan, a pro-Russian hacker group, caused the disruption.
- **New Voice Features in GPT-4o**: There was anticipation for new voice and vision capabilities in GPT-4o, which are expected to roll out in the coming weeks. Members expressed eagerness and some skepticism about the timeline, based on recent demo videos.
- **Improving GPT-3 Output Consistency**: A user launching a copywriting software sought advice on training GPT for consistent outputs. Options like fine-tuning models and using few-shot learning techniques were discussed, with references to detailed [OpenAI documentation](https://help.openai.com/en/articles/6614161-how-can-i-contact-support).

**Link mentioned**: <a href="https://elicit.com/">Elicit: The AI Research Assistant</a>: Use AI to search, summarize, extract data from, and chat with over 125 million papers. Used by over 2 million researchers in academia and industry.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1247463165135159338)** (26 messages🔥): 

- **ChatGPT Server Outage and Solutions**: Users experienced difficulties logging into ChatGPT due to an outage. OpenAI acknowledged the issue and suggested [performing a hard refresh](https://status.openai.com/) as a possible solution.

- **Status of New Voice Mode**: Members discussed delays in the rollout of the new voice mode for the Apple app. One member noted, *"the Apple app says it will be available in the coming weeks, but it's been saying this for almost three weeks now."*

- **Persistent Bad Gateway Errors**: Several users reported encountering 'bad gateway' issues while trying to use ChatGPT, indicating that the service had not stabilized for everyone.

- **Performance Issues with Large Prompts**: Users noted performance degradation with large prompts. One user mentioned, *"the bigger the prompt the more worse it gets,"* while another recommended lazy loading to mitigate browser issues.

- **Laggy Keyboard on Android App**: Members noted that the Android ChatGPT app has a laggy keyboard problem. Despite troubleshooting efforts such as removing cache and rebooting, the issue persisted for some users.

**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247402129866625084)** (81 messages🔥🔥): 

- **Too Advanced in Prompt Engineering**: A member expressed frustration that despite five days of refining prompts, GPT remains inconsistent with guidelines and struggles with complex requests. They concluded that the model is lacking in its current form and hoped for improvements in future versions.

- **WizardLM 2 was "Too Good"**: Another member suggested trying **WizardLM 2**, claiming it outperformed GPT-4 and was pulled due to its high performance, although they offered to test some prompts considering they run it locally.

- **API Cost vs. ChatGPT Plus**: Discussion ensued about the cost-effectiveness of using API over **ChatGPT Plus**, with claims that API might be cheaper unless one uses a high volume of tokens. **OpenRouter** and other alternatives were proposed as potentially more economical options.

- **Complex Prompts and Multiple Calls**: Members discussed breaking down complex prompts into multiple calls to improve consistency, balancing this approach against time and cost constraints. The pipeline and clear computable instructions were suggested as optimization strategies.

- **Insightful Reading Suggestion**: A prompt engineering research article titled "**Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**" was recommended for further reading. This paper, suggested to be found via a search engine, might offer insights into improving prompt engineering techniques.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247402129866625084)** (81 messages🔥🔥): 

- **Too many prompt requests cause inconsistency**: A member expressed frustration, stating that despite five days of refinement, GPT's **token limit** and overall "smartness" cause inconsistencies when attempting to follow multiple guidelines in one request. They noted, *"gpt is still not good at sticking to many guidelines in one request."*
- **Commercial value prompts cause skepticism**: A member claimed their prompt has high commercial value and couldn't share it publicly, prompting another to question their ability without seeing the prompt. This sparked skepticism with remarks like, *"It's funny how many people have these valuable prompts they can't share but also they need help with them."*
- **WizardLM 2 offers a promising alternative**: Another member recommended **WizardLM 2**, explaining it surpassed GPT-4 in performance but was cut due to unverified safety concerns. They offered to test prompts on it, highlighting that it's open-source and public.
- **OpenRouter and affordable alternatives**: Discussion about cost-effective AI solutions revealed **OpenRouter**'s competitive pricing, with WizardLM 2 at $0.65 per million tokens and free models like Llama 3 8b instruct. Members considered these alternatives due to API cost concerns.
- **Recommendation for reading on prompt engineering**: A member suggested reading an article titled **"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"** to deal with complex prompt engineering challenges. They emphasized that breaking tasks into steps can address performance issues.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247267877522903161)** (56 messages🔥🔥): 

- **Debate over CPU vs GPU usage in LM Studio**: Members discussed whether LM Studio models run on CPU or GPU, with a consensus that **LM Studio handles these settings** through GPU offload parameters. *"Mixed mode allows using both RAM and VRAM for big models, but dedicated GPU usage is often better if available."*

- **Error with loading models**: One user shared a JSON error indicating a problem loading a model and received advice to disable **GPU Offload** and ensure the model size is under 6GB. *"Try loading a model that is 6GB or less in size."*

- **Async processing of requests in LM Studio**: Users clarified that **LM Studio’s server processes requests sequentially**, with no support for running tasks in parallel by default. *"All the requests are being queued and processed one after the other."*

- **Challenges with model token limits and tracking**: A user inquired about the maximum tokens LM Studio can handle, leading to clarifications that **token limits depend on the specific model** and the necessity to manually track token usage. *"Number of tokens is down to the model. You can find that information in the model card."*

- **Downloading and managing models**: Users asked about downloading models via CLI and handling multiple model loads, with responses suggesting **using LM Studio to download models and advising caution with memory limits**. *"Download them through LM Studio itself and load them through LMS CLI seems okay."*

**Link mentioned**: <a href="https://tenor.com/view/tf2engineer-imposter-it-could-be-you-meme-tf2spy-gif-23428001">Tf2engineer Imposter GIF - Tf2Engineer Imposter It Could Be You - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247282087657214033)** (69 messages🔥🔥): 

- **CodeQwen 1.5 and Codestral 22B get recommended for code fixing**: Members suggested **CodeQwen 1.5 7B** and **Codestral 22b** for code fixing tasks. Another member noted that **wavecoder-ultra-6.7b** is also a good option to consider.

- **Wavecoder Ultra's murky release history**: "Wavecoder Ultra is also such a weird model cause it came out under the shadow of wizard." This implies it had a fragmented release with limited information.

- **MahouDevil 8B models shared**: Links to **MahouDevil-8B** in both [Q8](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF) and [Q4_K_M](https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF) quantizations were provided for download.

- **Extractum.io for filtering model searches**: Members discussed using [Extractum.io](https://llm.extractum.io/list) for filtering large lists of models by various criteria like VRAM and quantization, with links back to **Hugging Face** models.

- **Explaining Quantization**: Quantization was described as a process reducing the number of decimal places in model weights to make the model smaller, with the explanation, *"Quantization reduces quality but also makes the model smaller."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q4_K_M-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Contamination/contaminated_proof_7b_v1.0_safetensor">Contamination/contaminated_proof_7b_v1.0_safetensor · Hugging Face</a>: no description found</li><li><a href="https://llm.extractum.io/list/">All Large Language Models</a>: A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1247267324562640937)** (1 messages): 

- **Inference speed testing for Llama-3-70B**: A user shared their findings after testing various inference speeds for **llama-3-70b** using a **q4 4090** setup. This information could be particularly useful for those with similar hardware configurations.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247272943231434852)** (129 messages🔥🔥): 

- **Troubleshooting GPU Performance Issues on 6800XT**: A member using a 6800XT GPU for models such as **CodeLlama** and **Mistral7b** noted they were achieving only 12it/s instead of the expected 52it/s. After switching from **OpenCL** to **ROCm** and properly reloading the model, they managed to increase their speed to 30it/s but were still troubleshooting to reach the optimal performance.

- **Debate Over Homelab GPU Choices**: A discussion about an AI/server build led to the suggestion of substituting a **12GB GPU** with a more capable one like the **4060 Ti (16GB)** due to concerns over VRAM limitations. Used **3090s** were recommended as cost-effective alternatives to the **4090** and other high-end options.

- **Driver and Software Stack Considerations**: When considering GPU options, members highlighted how **Intel Arc** is not yet fully supported and that **AMD ROCm** can be slower than NVIDIA's software stack. The preference shifted toward well-supported NVIDIA cards for stability and performance.

- **Exploring Alternative Linux Setups**: The discussion of setting up a homelab also touched on the merits of different Linux distributions. **Ubuntu** was popular for its ease of use although some members preferred **Arch** for package management, despite acknowledging its steeper learning curve.

- **Trust in Second-Hand GPUs**: Members considered purchasing second-hand GPUs like the **3090** from the marketplace as a cost-saving measure. However, they discussed the trade-offs such as potential performance penalties and power consumption, particularly when using multiple cards without NVLink.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev.to/maximsaplin/running-local-llms-cpu-vs-gpu-a-quick-speed-test-2cjn/">no title found</a>: no description found</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://wccftech.com/amd-radeon-pro-w7900-dual-slot-gpu-48-gb-ai-workstations-compact-design-3499-usd/">AMD Radeon PRO W7900 Dual Slot GPU Brings 48 GB Memory To AI Workstations In A Compact Design, Priced at $3499</a>: AMD is adding a new Radeon PRO W7900 GPU to its lineup which adopts a dual-slot design but retains the same specs as the previous model.</li><li><a href="https://tenor.com/view/arch-linux-arch-btw-cv-walter-white-gif-24576245">Arch Linux Arch Btw GIF - Arch Linux Arch Btw Cv - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1247298531312537610)** (2 messages): 

- **Memory Bandwidth Limits AI Performance**: A member pointed out that **AI is memory bandwidth limited**, explaining that running more than one thread per physical core can choke the memory controller and slow down performance. They suggested that the ideal thread count is around *80% of the physical core count*.
- **Chinese Language Support Inquiry**: A member inquired about when **Chinese language support** would be available.
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1247624625215504414)** (1 messages): 

- **Tracking LM Studio's upcoming release**: A member inquired about an upcoming release for **LM Studio** and asked if there is a place to track it. They expressed interest in returning to **LM Studio** once it is fixed.
  

---


### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1247568831367614557)** (3 messages): 

- **Call for AVX Extension Pack Testers**: A request was made for testers of an early AVX-only extension pack. The message encouraged members to like or comment if interested, receiving positive responses from the community.
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1247338818001043529)** (1 messages): 

- **Continue.dev shines for local setups**: A member praised **continue.dev** as *"the best one..."* for local use, citing a recent update to their documentation. They specifically mentioned adding steps on how to *"setup LM Studio support"*.
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1247264658063097969)** (12 messages🔥): 


- **Music video explosion gets lively**: Members shared a series of YouTube music video links, such as *Mindchatter - Night Goggles (Rome in Silver Remix)* and *Porter Robinson & Madeon - Shelter (Official Video)*. These videos seem to resonate well within the community.
- **Wayseer Manifesto makes an appearance**: The *Wayseer Manifesto - [Official Video]* was shared, highlighting a unique blend of motivational and rule-breaking themes. Available also in Spanish, it boasts over 6 million views.
- **Design talk sparks interest**: Comments about the enjoyable design work at Nous Research were noted, with one user humorously declaring, "I'm the design people." More of their design work can be found on their [Twitter account](https://x.com/StudioMilitary).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/OPR3GlpQQJA">THE WAYSEER MANIFESTO - [Official Video] (HQ)</a>: Spanish version available now with OVER 6 MILLION VIEWS: http://www.youtube.com/watch?&amp;v=KYfc5_YFFb0#!ATTENTION: All you rule-breakers, you misfits &amp; trouble...</li><li><a href="https://youtu.be/fzQ6gRAEoy0">Porter Robinson &amp; Madeon - Shelter (Official Video) (Short Film with A-1 Pictures &amp; Crunchyroll)</a>: Porter Robinson &amp; Madeon - Shelter (Official Video) (Short Film with A-1 Pictures &amp; Crunchyroll)Shelter tells the story of Rin, a 17-year-old girl who lives ...</li><li><a href="https://x.com/StudioMilitary">Tweet from undefined</a>: no description found</li><li><a href="https://youtu.be/A5Npdlg1Vaw">Mindchatter - Night Goggles (Rome in Silver Remix)</a>: Stream/Download:https://lnk.to/nightgogglesromeinsilverremixIDFollow Mindchatter:https://mindchatter.lnk.to/Instagramhttps://mindchatter.lnk.to/Twitterhttps:...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

sidfeels: https://laion.ai/notes/open-gpt-4-o/
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247263672175300629)** (198 messages🔥🔥): 

- **T5 Hardware Requirements Limits Adoption**: A member noted that T5 sets a high hardware requirement, impacting its adoption. Another member mentioned the lack of fast CPU implementations suitable for broad use, referencing Huggingface's limited Candle library and potential future developments with ggml.
  
- **Open Source Version of Chat Assistant UI Needed**: A user inquired about an open-source version of Huggingface’s chat assistant UI for defining prompt templates and customizing vector databases. Another member suggested a better alternative from Mobius, noting improvements over previous implementations.
  
- **Pixart Training Issue Raised**: An issue with Pixart falling apart when trained on large datasets was discussed, highlighting how it produces garbled shapes with just 10k images. Contrarily, another model performed well with over 400k images due to unique training techniques.

- **Datasets and AGIEval Leaderboard**: Discussion about large datasets, with mentions of Redpajama v2 at 30 trillion tokens. A request for leaderboards using AGIEval was directed to a GitHub repository with relevant benchmark logs.

- **Mobius Model Highlights and Offer for Compute Sponsorship**: Highlights of the Mobius model's capabilities were presented, and the repository was shared. The model's creator offered to sponsor compute for interesting projects and discussed potential collaborations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Corcelio/mobius">Corcelio/mobius · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/cloneofsimo/lavenderflow-5.6B">cloneofsimo/lavenderflow-5.6B · Hugging Face</a>: no description found</li><li><a href="https://www.together.ai/blog/redpajama-data-v2">RedPajama-Data-v2: An open dataset with 30 trillion tokens for training large language models</a>: no description found</li><li><a href="https://discord.gg/kRbaDnHE">Join the PixArt-α Discord Server!</a>: Check out the PixArt-α community on Discord - hang out with 1702 other members and enjoy free voice and text chat.</li><li><a href="https://www.academia.edu/120538461/Sensuality_in_Emergent_LLM_Models_A_New_Turing_Criteria_question">Sensuality in Emergent LLM Models -A New Turing Criteria question</a>: &amp;quot;Sensuality in Emergent LLM Models -A New Turing Criteria question?&amp;quot; In this paper, we will propose that AI/LLM models and instantiations who express both &amp;quot;Pleasure&amp;quot...</li><li><a href="https://x.com/zhanga6/status/1797293189378068768?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Ao Zhang (@zhanga6)</a>: So sad to hear the news (https://github.com/OpenBMB/MiniCPM-V/issues/196)😰. The conclusion of our investigation:  1. Llama3-V can be run using MiniCPM-Llama3-V 2.5&#39;s code and config.json after ch...</li><li><a href="https://tenor.com/view/plink-nerd-plank-plink-cat-cat-gif-17569403098672348326">Plink Nerd GIF - Plink Nerd Plank - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/akshgarg03/status/1797682238961914370?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Aksh Garg (@AkshGarg03)</a>: Re Llama3V: First of all, we want to apologize to the original authors of MiniCPM. We wanted Mustafa to make the original statement but have been unable to contact him since yesterday.  @siddrrsh and ...</li><li><a href="https://github.com/teknium1/LLM-Benchmark-Logs">GitHub - teknium1/LLM-Benchmark-Logs: Just a bunch of benchmark logs for different LLMs</a>: Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.</li><li><a href="https://x.com/Teknium1/status/1797491548353183994">Tweet from Teknium (e/λ) (@Teknium1)</a>: How many images does it take to train SD3 and how many h100 hours</li><li><a href="https://x.com/shreyaskapur/status/1797726079995826629">Tweet from Shreyas Kapur (@shreyaskapur)</a>: My first PhD paper!🎉We learn *diffusion* models for code generation that learn to directly *edit* syntax trees of programs. The result is a system that can incrementally write code, see the execution...</li><li><a href="https://x.com/yangzhizheng1/status/1797197104999518306?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from PrimerYang (@yangzhizheng1)</a>: Shocked! Llama3-V project from a Stanford team plagiarized a lot from MiniCPM-Llama3-V 2.5! its code is a reformatting of MiniCPM-Llama3-V 2.5, and the model&#39;s behavior is highly similar to a nois...</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: Official repository of the xLSTM.</a>: Official repository of the xLSTM. Contribute to NX-AI/xlstm development by creating an account on GitHub.</li><li><a href="https://bellard.org/nncp/">NNCP: Lossless Data Compression with Neural Networks</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

vorpal_strikes: cant wait for The Emergence of Nous World in 2030
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1247469340379320431)** (4 messages): 

- **WorldSim Jam Session airs on YouTube**: The WorldSim Jam Session from a few weeks back will "premiere" on YouTube. Participants can [re-watch it here](https://www.youtube.com/watch?v=qaE99xmtvd4).
- **Exploring LLM System Evaluation**: A detailed video titled *"Make Agent 10x cheaper, faster & better?"* discusses LLM System Evaluation. You can check it out [here](https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U).
- **AI Agent job irony**: A member humorously noted the irony that the *“number one job to be lost to AI agent is... agent.”* with laughter emojis.
- **Returning member updates**: A returning member expressed they missed the group and mentioned they *“wrote a paper and did a thing”* during their absence.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>: LLM System Eval 101 - Build better agentsGet free HubSpot report of how to land a Job using AI: https://clickhubspot.com/fo2🔗 Links- Follow me on twitter: h...</li><li><a href="https://www.youtube.com/watch?v=qaE99xmtvd4">WorldSim Jam Session #2</a>: no description found
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247268569436393513)** (159 messages🔥🔥): 

- **SD3 Model Draws Mixed Reactions**: Members shared their frustrations with the early version of SD3, stating it doesn't do hands properly and falls short compared to Dall-E. A user remarked, *"Give it time to local release and before long custom models will have it entirely solved."*

- **Stable Diffusion for Architecture**: A member asked about using Stable Diffusion for architectural projects, specifically for previews of interiors based on drawings. Another responded that *"It's not so hot with anything involving straight lines and mechanics,"* but suggested using img2img with detailed drawings for possible results.

- **Wildcards Plugin Issues**: A user experienced degraded image quality after installing the wildcards plugin on Stable Diffusion, mentioning *"the images were extremely grainy and the faces had those ugly color-blotches."* Despite several re-installs, the problem persisted.

- **Discussion on Community Models and Resources**: Members recommended community models from sites like [civitai.com](https://civitai.com) for improving Stable Diffusion rendering quality. [ChaiNNer](https://github.com/JoeyBallentine/chaiNNer) was also suggested as an upscaler tool for those looking to batch upscale images.

- **Celebrities as AI Models**: The community discussed the rise of influencer and celebrity LoRas on platforms like Civit, noting the influx of AI-generated profiles. One user humorously commented, *"When is a celebrity not a celebrity?"* in reference to the trend.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/excited-excited-man-excited-funny-rubbing-hands-plotting-gif-27652478">Excited Excited Man GIF - Excited Excited Man Excited Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.instagram.com/fit_aitana/">Login • Instagram</a>: no description found</li><li><a href="https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K">laion/CLIP-ViT-g-14-laion2B-s34B-b88K · Hugging Face</a>: no description found</li><li><a href="https://youtube.com/watch?v=JOiK8z-Ffp4">Upsize Your MidJourney or Dalle-2 AI Artwork Resolution for FREE!</a>: There is a fantastic method to not only upsize your AI art from MidJourney, Dalle-2, or Stable Diffusion, but it can also maintain or even add to the details...</li><li><a href="https://www.youtube.com/watch?v=lNGEeUCL8NE">SD一键生成视频！免费开源，想怎么玩就怎么玩！附完整安装教程  | 零度解说</a>: 【更多资源】▶https://www.youtube.com/channel/UCvijahEyGtvMpmMHBu4FS2w?sub_confirmation=1【零度博客】▶https://www.freedidi.com【加入会员】▶https://www.youtube.com/channel/UCvij...</li><li><a href="https://www.instagram.com/tucumandigital/p/C1ndOWvrhXY/?img_index=1">Tucuman Digital on Instagram: &quot;#INCRE&#xcd;BLE &#x1f534; LA MODELO QUE LA ROMPE EN UNA PLATAFORMA PARA ADULTOS PERO NO ES REAL &#x1f62e;

Se trata de Emily Pellegrini, quien ha ganado millas de seguidores en las &#xfa;ltimas semanas al crear un perfil en una plataforma para adultos, desafiando al conocido Onlyfans. En las redes sociales, la propagaci&#xf3;n de influencers generados con Inteligencia Artificial (IA) est&#xe1; en aumento. Ahora, una modelo que vende contenido para adultos se est&#xe1; volviendo viral en Instagram: Emily Pellegrini, de 21 a&#xf1;os, ha acumulado millas de seguidores en la popular plataforma.

La influencer ha generado ingresos cercanos a los 10 mil d&#xf3;lares en tan solo seis semanas gracias al contenido que vende a trav&#xe9;s de Fanvue, una plataforma para adultos que compite con Onlyfans. A pesar de &quot;vivir&quot; en Italia, la peculiaridad de Emily Pellegrini es que todas sus im&#xe1;genes son creadas mediante IA, un m&#xe9;todo que ha ganado popularidad en los &#xfa;ltimos meses en internet.

La cuota de suscripci&#xf3;n mensual es de aproximadamente nueve d&#xf3;lares, y ya cuenta con cerca de 100 suscriptores regulares. La popularidad de Emily Pellegrini ha alcanzado tal nivel que, en apenas seis semanas, ha acumulado casi 90 mil seguidores en Instagram. Incluso en su perfil de redes sociales, tiene una secci&#xf3;n de historias destacadas donde aparece con &quot;amigas&quot;, que son otros modelos de la plataforma creadas con IA.

La fama de Emily Pellegrini ha llegado tan lejos que el reconocido medio internacional New York Post le dedic&#xf3; un art&#xed;culo tanto a ella como a la plataforma Fanvue. Esta plataforma es similar a Onlyfans, pero con la particularidad de que todos los modelos son exclusivamente creados con IA.

El fundador del sitio, Will Monange, sostiene que la IA es una &quot;herramienta&quot; y una &quot;extensi&#xf3;n de lo que somos y lo que hacemos&quot;, una diferencia de lo que otras personas consideran como un reemplazo de la creatividad humana.&quot;</a>: 1,779 likes, 13 comments - tucumandigital on January 2, 2024: &quot;#INCRE&#xcd;BLE &#x1f534; LA MODELO QUE LA ROMPE EN UNA PLATAFORMA PARA ADULTOS PERO NO ES REAL &#x1f62e;  Se trata de Emily Pellegr...</li><li><a href="https://github.com/jaisidhsingh/CoN-CLIP">GitHub - jaisidhsingh/CoN-CLIP</a>: Contribute to jaisidhsingh/CoN-CLIP development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/432244/forrealxl">ForRealXL - v0.5 | Stable Diffusion Checkpoint | Civitai</a>: Hello ♥ for whatever reason you want to show me appreciation, you can: ❤️ Ko-Fi ❤️ This is an experimental Checkpoint because its my first. Special T...</li><li><a href="https://github.com/JoeyBallentine/chaiNNer">GitHub - chaiNNer-org/chaiNNer: A node-based image processing GUI aimed at making chaining image processing tasks easy and customizable. Born as an AI upscaling application, chaiNNer has grown into an extremely flexible and powerful programmatic image processing application.</a>: A node-based image processing GUI aimed at making chaining image processing tasks easy and customizable. Born as an AI upscaling application, chaiNNer has grown into an extremely flexible and power...</li><li><a href="https://upscale.wiki/wiki/Model_Database">Model Database - Upscale Wiki</a>: no description found
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1247264971935580381)** (107 messages🔥🔥): 

- **SD3 Models Struggle with Noise Artifacts**: Members discuss the *persistent spotty noise issue* in SD3 medium (2B) models, which show up as "weird spray-list spotty noise" especially noticeable in specific areas like the water in a fountain, despite using advanced features like a 16ch VAE.
- **Validation and Evaluation Issues**: There is significant skepticism about the effectiveness of current *validation metrics* and loss functions used in evaluating SD3 models. One user highlighted that "*loss is a terrible indicator of model performance*," critiquing the methods described in the SD3 paper.
- **Artifact Issues Across Models**: The conversation explores how *MM-DiT models* suffer from consistent artifacts due to their architecture which impacts the overall image quality. Validating this point, one member noted, "for a single seed you'll see the **same** artifacts across essentially every prompt."
- **Open-source Video VAE News**: There was excitement over the release of a freely licensed [Apache2 video-capable CV-VAE](https://arxiv.org/abs/2405.20279), highlighting its importance for latent diffusion-based video models within the research community.
- **Reddit and Google Examples**: Members compared SD3 images, with some claiming they look like "women suffering a bad botox injection" ([Reddit link](https://www.reddit.com/r/StableDiffusion/comments/1d73j3r/some_sd3_images_women/)). Google’s recent demo ([Twitter link](https://vxtwitter.com/GoogleDeepMind/status/1797605392089825457)) garnered praise for its realistic examples, especially for cloth texture and hair consistency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.20279">CV-VAE: A Compatible Video VAE for Latent Generative Video Models</a>: Spatio-temporal compression of videos, utilizing networks such as Variational Autoencoders (VAE), plays a crucial role in OpenAI&#39;s SORA and numerous other video generative models. For instance, ma...</li><li><a href="https://x.com/Lykon4072/status/1797703714180051130?t=4DG7gVlXqw65fOJrNpHBAw&s=19">Tweet from Lykon (@Lykon4072)</a>: For reference, SD3 2B has roughly the same size but it&#39;s MMDiT (which is far superior to Unet) and used 3 text encoders, plus has a 16ch VAE. You can&#39;t get this level of detail in XL without c...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1247267349996769372)** (17 messages🔥): 

- **Transformers vs. State-Space Models Rivalry Sparks Interest**: A shared paper ([arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060)) discusses how state-space models (SSMs) like Mamba can outperform Transformers and introduces the **State Space Duality (SSD)** framework that enhances model efficiency. The new architecture, **Mamba-2**, is 2-8X faster than its predecessor and competitive with Transformers in language modeling.

- **Tiny Bit Methods Criticized**: Queries about the effectiveness of the 1.58-bit method reveal general skepticism and negative feedback. Users report that *"everyone I know who has tried these tiny bit methods say they suck."*

- **Diffusion Models Thrive on Data Corruption**: A preprint ([arxiv.org/abs/2405.20494](https://arxiv.org/abs/2405.20494)) suggests that slight corruption in pretraining datasets can enhance the quality and diversity of generated images in diffusion models (DMs). The paper introduces **Conditional Embedding Perturbation (CEP)**, showing that intentional data corruption benefits model training.

- **Concerns on Large Diffusion Models and Overfitting**: Discussions highlight concerns about large diffusion models, particularly those over 500 million parameters, overfitting on datasets like ImageNet. Dropout and data augmentation are mentioned as methods to combat overfitting.

- **Debate on Robustness and Model Training Difficulty**: The conversation explores how increasing training data difficulty might improve model robustness. Referencing [Distill.pub's article on interpretability](https://distill.pub/2020/circuits/), the intuition that *"increasing the difficulty of tasks during optimization creates better models"* is considered plausible, although quantifying this effect remains a challenge.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.21060">Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality</a>: While Transformers have been the main architecture behind deep learning&#39;s success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Tran...</li><li><a href="https://arxiv.org/abs/2405.20494">Slight Corruption in Pre-training Data Makes Better Diffusion Models</a>: Diffusion models (DMs) have shown remarkable capabilities in generating realistic high-quality images, audios, and videos. They benefit significantly from extensive pre-training on large-scale dataset...</li><li><a href="https://distill.pub/2020/circuits/">Thread: Circuits</a>: What can we learn if we invest heavily in reverse engineering a single neural network?
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247345403448262716)** (68 messages🔥🔥): 

- **Daylight tablet debate sparks controversy**: Members debated the claims in [a YouTube video](https://youtu.be/iHeIw9rXzUQ?si=QDETgMaKJTaTYLkV) reviewing the Daylight tablet, questioning its advertising as an e-paper device. Many argued that it is misleadingly similar to E-ink but is in fact a reflective LCD, sparking discontent about its price and features compared to devices like iPads and Kindles.
  
- **Innovation vs. Mislabeling in e-paper technology**: The discussion highlighted the fine line between innovation and mislabeling, as users argued that the Daylight tablet claiming to have invented new RLCD technology felt misleading. Critics raised concerns about their claims of superior FPS and daylight visibility while others pointed out it might merely be branded over existing Sharp RLCD technology.

- **Battery life concerns for new devices**: Battery life comparisons also arose, with some highlighting the Kindle's weeks-long battery life versus the Daylight tablet's claim of lasting multiple days to a week. This raised skepticism about the practicality of the new device as a daily-use tablet for activities like notetaking and drawing under sunlight.

- **Skepticism about founding claims**: Arguments extended to suspecting the authenticity of the e-paper claims, with some users noting the presence of the founder outside a Sharp factory as indicative of merely using pre-existing Sharp technology rather than innovating. This skepticism was met with rebuttals emphasizing the absence of direct evidence proving or disproving the novel technology claims.

- **A call to investigate further with potential teardown**: The debate concluded with the suggestion that a teardown of the Daylight tablet could potentially reveal the true nature of its tech, comparing it to an existing Sharp RLCD. The theoretical discussions remained inconclusive without physical verification, potentially waiting for someone to provide concrete evidence to settle the argument.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sharpdevices.com/reflective-lcd/#1619791276508-913e49e4-989a">no title found</a>: no description found</li><li><a href="https://youtu.be/9M6zT0mRvW0?si=RwRcoY8AVROZR_5W">World&#39;s first 60+ FPS e-paper display | Daylight</a>: The world&#39;s first 60+ FPS e-paper display by Daylight Episode 45 of S³. See how it works, the 6-year development journey, and Daylight&#39;s vision for the futur...</li><li><a href="https://youtu.be/iHeIw9rXzUQ?si=QDETgMaKJTaTYLkV">Daylight Tablet Review: World’s First 60 Hz E-Paper!</a>: Spoiler alert: It&#39;s not E-ink.Daylight DC1: https://daylightcomputer.com/Reddit Post: https://www.reddit.com/r/daylightco/comments/1cz23hj/anjans_mission/🖼️...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247276260002893915)** (46 messages🔥): 

- **New insights into LLM internals**: A research paper [explores the knowledge circuits in large language models](https://arxiv.org/abs/2405.17969), extending beyond isolated components like attention heads and MLPs. One member appreciated the approach, noting it feels more appropriate compared to current popular methods.
  
- **Challenges with TinyLLama database**: A member working on TinyLLama.db revealed efforts to coordinate activations with weight entries and quickly push database changes. They also expressed the need to shrink the model from 22GB, which they found too large.
  
- **Corrupted data helps diffusion models**: A [preprint by Microsoft and CMU](https://arxiv.org/abs/2405.20494) reveals that slightly corrupted datasets can benefit diffusion models, enhancing image quality and diversity. The research introduces Conditional Embedding Perturbation (CEP) and evaluates over 50 models.
  
- **Differentiable top-k function for visual models**: A member explored the [differentiable top-k function](https://math.stackexchange.com/questions/3280757/differentiable-top-k-function) to select the most important image tokens for classification tasks. They found attention maps in ViT models to be less interpretable.
  
- **RNNs and pre-trained transformers synergy**: Discussions centered around the idea of using pre-trained transformers to stabilize and inform the training of new RNNs. A proposed method involves transformers providing informative vectors to RNNs, potentially enhancing learning despite theoretical limitations in state tracking for both architectures.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.20494">Slight Corruption in Pre-training Data Makes Better Diffusion Models</a>: Diffusion models (DMs) have shown remarkable capabilities in generating realistic high-quality images, audios, and videos. They benefit significantly from extensive pre-training on large-scale dataset...</li><li><a href="https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp">McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.17969">Knowledge Circuits in Pretrained Transformers</a>: The remarkable capabilities of modern large language models are rooted in their vast repositories of knowledge encoded within their parameters, enabling them to perceive the world and engage in reason...</li><li><a href="https://tridao.me/blog/2024/mamba2-part1-model/"> State Space Duality (Mamba-2) Part I - The Model | Tri Dao </a>: no description found</li><li><a href="https://arxiv.org/abs/2405.16674">Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory</a>: Deep learning models have achieved significant success across various applications but continue to struggle with tasks requiring complex reasoning over sequences, such as function composition and comp...</li><li><a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>: State-space models (SSMs) have emerged as a potential alternative architecture for building large language models (LLMs) compared to the previously ubiquitous transformer architecture. One theoretical...</li><li><a href="https://math.stackexchange.com/questions/3280757/differentiable-top-k-function,">Differentiable top-k function</a>: Is there any differentiable function that, for a given vector, selects and encourages the top-k maximum value and suppresses the rest of the values? For example for z = [0.01 0.1 0.04 0.5 0.24] the...</li><li><a href="https://arxiv.org/abs/2405.06640">Linearizing Large Language Models</a>: Linear transformers have emerged as a subquadratic-time alternative to softmax attention and have garnered significant interest due to their fixed-size recurrent state that lowers inference cost. Howe...</li><li><a href="https://github.com/NX-AI/xlstm">GitHub - NX-AI/xlstm: Official repository of the xLSTM.</a>: Official repository of the xLSTM. Contribute to NX-AI/xlstm development by creating an account on GitHub.</li><li><a href="https://github.com/Felix-Petersen/diffsort">GitHub - Felix-Petersen/diffsort: Differentiable Sorting Networks</a>: Differentiable Sorting Networks. Contribute to Felix-Petersen/diffsort development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2405.14838">From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step</a>: When leveraging language models for reasoning tasks, generating explicit chain-of-thought (CoT) steps often proves essential for achieving high accuracy in final outputs. In this paper, we investigate...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1247425540797763636)** (9 messages🔥): 

- **Increasing AI Task Concurrency**: One member asked if there's a way to increase the level of concurrency when running tasks, as the current setup runs one query at a time and is "extremely slow". There was no immediate follow-up response addressing this question.

- **Smallest Decoder Model Query**: A user searched for the smallest decoder model available on Huggingface to measure energy consumption. Another member suggested, "Just instantiate a small model with random values", initiating a further request for a tutorial.

- **N-shot Default Values Affect Results**: It was noted that when not specifying n-shot values, the default is 5, which leads to different results than expected. This discrepancy was explained by stating that the Huggingface Open LLM leaderboard uses an older version of the harness.

- **Pull Request for ARC Challenge**: A member requested a review for their [PR #1900 on GitHub](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900), which adds tasks for machine-translated versions of the ARC challenge for 11 languages, with plans to add more languages in the future.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>: This PR adds tasks for machine-translated versions of arc challenge for 11 languages.  We will also be adding more languages in the future.

  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

merfippio: I know this is really late, but did you find the FE help you were looking for?
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1247339340867178497)** (106 messages🔥🔥): 

- **Missing Credits after ETH Payment Issue**: A user reported that their credits weren't showing up after paying with ETH on Base. Another member suggested waiting 20 minutes to an hour before raising a complaint if the credits still do not appear.
- **Prefill Handling by LLMs**: A user queried about LLM's handling of prefill text and whether it would generate subsequent paragraphs the same way it would if done continuously. The consensus was that prefill is handled seamlessly, as if it were part of the original prompt.
- **GPT-3.5 Turbo Issues and Moderation**: A user reported issues with GPT-3.5 Turbo not working for them while other OpenRouter LLMs were fine, leading to a discussion on possible API moderation affecting requests. OpenRouter confirmed that OpenAI requires all requests to be moderated using their moderation API.
- **Mistral Model Reliability Issues**: Users reported consistently getting empty responses with Mistral: Mixtral 8x22B Instruct on Fireworks, suggesting a potential issue with the provider. OpenRouter's admin suggested setting DeepInfra as a preferred provider and referred to load balancing documentation for manual provider whitelisting.
- **Best Models for Storytelling**: A discussion took place on the best models for storytelling, with recommendations including roleplay-specific models from OpenRouter's rankings and Wizardlm2.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/rankings/roleplay?view=week">LLM Rankings: roleplay | OpenRouter</a>: Language models ranked and analyzed by usage for roleplay prompts
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1247265544554549391)** (26 messages🔥): 

- **Speed Up Python with Numba**: A YouTube video titled ["Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)"](https://youtu.be/OiMZtjSZVOw) was shared, explaining how to use a JIT compiler to speed up Python code significantly. One member expressed hope for similar performance without relying on external libraries.

- **Discussion on Python Generators**: A member inquired whether using a for loop within a while loop is effective in pure Python, sparking a discussion about generators, `yield`, and other optimization methods. A Real Python article and video on [Python Generators](https://realpython.com/introduction-to-python-generators/) was recommended for further learning.

- **Mojo and Python Execution**: There was a conversation about whether MAX can accelerate Python execution, contrasting various tools and approaches such as the benefits of Tensor and Torch libraries versus pure Python performance improvements.

- **Mojo Community Meetings**: Information about the second Mojo Community Meeting was shared, including a [Youtube playlist](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D) of past meetings. The next meeting was announced to occur in two weeks, with an open invitation for participants to share their work by adding themselves to the [Google doc agenda](https://modul.ar/community-meeting-doc).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/OiMZtjSZVOw?si=JrgOG_UL662xZ48W">Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)</a>: Numba can speed up your python code 1000x with just a single line of code using a JIT compiler used to optimize simple functions in python by compiling funct...</li><li><a href="https://youtu.be/OiMZtjSZVOw?si=JrgO">Make Python 1000x Faster With One Line 🐍 ⏩ (Numba Tutorial)</a>: Numba can speed up your python code 1000x with just a single line of code using a JIT compiler used to optimize simple functions in python by compiling funct...</li><li><a href="https://realpython.com/introduction-to-python-generators/#creating-data-pipelines-with-generators">How to Use Generators and yield in Python – Real Python</a>: In this step-by-step tutorial, you&#x27;ll learn about generators and yielding in Python. You&#x27;ll create generator functions and generator expressions using multiple Python yield statements. You&#...</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1247263614335717500)** (2 messages): 

- **Modular Tweets Shared**: A member shared two tweets from **Modular**: [Tweet 1](https://twitter.com/Modular/status/1797699002353488183) and [Tweet 2](https://twitter.com/Modular/status/1798055387557749010). No further discussion or commentary followed these shares.
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1247600282817265694)** (1 messages): 

- **Deep Dive into Ownership in Mojo**: The blog post titled [Deep Dive into Ownership in Mojo](https://www.modular.com/blog/deep-dive-into-ownership-in-mojo) is the second part of a series exploring ownership in Mojo. It builds on concepts from the first part, [What Ownership is Really About: A Mental Model Approach](https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach), and includes insights from the CEO, [Chris Lattner](https://www.modular.com/team/chris-lattner), on implementing ownership in Mojo’s compiler.

**Link mentioned**: <a href="https://www.modular.com/blog/deep-dive-into-ownership-in-mojo">Modular: Deep Dive into Ownership in Mojo</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Deep Dive into Ownership in Mojo

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 messages): 

melodyogonna: I don't think cryptography libraries has been added to the stdlib yet
  

---


### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1247490462676287528)** (1 messages): 

- **Project Verona challenges Rust's memory safety with ease**: Researchers discuss **Project Verona**, a programming language designed to provide memory safety guarantees similar to **Rust**, but with an easier-to-learn model. Check out the [YouTube video](https://youtu.be/VU9QATDvidw) for an in-depth talk titled "*Concurrent Mutation must go*".

**Link mentioned**: <a href="https://youtu.be/VU9QATDvidw">[POCL&#39;24] Concurrent Mutation must go</a>: [POCL&#39;24] Concurrent Mutation must goMatthew J. Parkinson, Sylvan Clebsch, Tobias Wrigstad, Sophia Drossopoulou, Elias Castegren, Ellen Arvidsson, Luke Chees...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247296560560341072)** (55 messages🔥🔥): 

- **Debate over `async` by Default in Mojo**: A member suggested that every function in Mojo should be implicitly `async` to avoid blocking issues, particularly for GPU programming where blocking is unacceptable. Others raised concerns about deviating from Pythonic practices and potential difficulties for users with an explicit `async`/`await` mindset.

- **Community Feedback on `async` Proposal**: The suggestion to make `async` the default sparked mixed reactions. Some argued for keeping functions synchronous to maintain compatibility with Python and ease the transition for Python programmers, while others pointed out the need for efficient scheduling and suspension mechanisms in GPU programming.

- **JSON Parsing Clarification**: There was a query about the endianness of JSON and parsing conventions. It was clarified that JSON is a text format usually in UTF-8, which is endian-independent, and that indexing `[1:-1]` is used to omit the enclosing brackets during parsing.

- **Issue Reporting and Tool Support Requests**: Members discussed potential bugs in the code assistant regarding relative imports and encouraged filing bug reports. Additionally, issues were created on GitHub for tools like `tokei` and `tcount` to add Mojo support, with requests for others to support these issues ([Tokei Issue](https://github.com/XAMPPRocky/tokei/issues/1107), [TCount Issue](https://github.com/RRethy/tcount/issues/3)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/XAMPPRocky/tokei/issues/1107>">Issues · XAMPPRocky/tokei</a>: Count your code, quickly. Contribute to XAMPPRocky/tokei development by creating an account on GitHub.</li><li><a href="https://github.com/RRethy/tcount/issues/3>">Issues · RRethy/tcount</a>: Count your code by tokens and patterns in the syntax tree. A tokei/scc/cloc alternative. - Issues · RRethy/tcount
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1247411061892714557)** (1 messages): 

- **Value decorator may optimize benchmarks**: A member suggested that using the `@value` decorator can simplify and potentially improve benchmark performance. *"Not sure how it can affect the benchmark, but you can use the `@value` decorator and save all the copyinit and moveinit logic."*
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1247420844289032253)** (18 messages🔥): 

- **ComparableCollectionElement inherits correctly**: A user confirmed that *ComparableCollectionElement* inherits from *CollectionElement* via a trait definition. [Source provided](https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/value.mojo#L224).

- **List requirements and future improvements**: @clattner suggested that **List** should primarily require movable elements, possibly being reduced to *AnyType* or less in the future. Another user proposed an alternative implementation of a List struct with various functions.

- **Transition issues with CollectionElementNew**: Users discussed the complications arising from transitioning to *CollectionElementNew* and the absence of *ComparableCollectionElementNew*. This indicates ongoing refinements within the system.

- **Support for trait objects in Mojo**: There was a discussion about whether Mojo will support trait objects, which are crucial for creating heterogeneous lists similar to Python. @jack.clayton confirmed that this feature is planned but without a specified target date.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://doc.rust-lang.org/book/ch17-02-trait-objects.html">Using Trait Objects That Allow for Values of Different Types - The Rust Programming Language</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/builtin/value.mojo#L224)">mojo/stdlib/src/builtin/value.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1247290220228120808)** (50 messages🔥): 


- **Ollama models struggle with bind_tools**: A community member asked if LangChain supports bind_tools for Ollama models and it was confirmed that it does via the `OllamaFunctions` class, not `ChatOllama`. A [GitHub issue](https://github.com/langchain-ai/langchain/issues/21479) was referenced for additional context.

- **Building a versatile customer support assistant**: Community discussions focused on creating a customer support assistant using LLMs like Llama3, LangChain, and tools for actions like user verification and probability calculation. Example Python code was shared to illustrate chaining models and custom tools.

- **Handling chat context in SQL agents**: Members discussed maintaining chat context/history in agents, specifically SQL and CSV agents. A code snippet using `ConversationBufferMemory` was shared, but issues with unsupported kwargs were noted.

- **Categorizing large text datasets with embeddings**: One member sought help for categorizing 10,000 free-text responses using OpenAI and discussed using embeddings and LangChain. Another member suggested prompt engineering with embeddings to streamline the process.

- **Persisting chatbot memory**: A user attempted to follow a LangChain chatbot tutorial for persistent memory and server setup but encountered an error in the provided `server.py` code. The source of the error was not identified in the discussion.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/21479>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/">Build a Chatbot | 🦜️🔗 LangChain</a>: Overview</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/llms/ollama/#via-langchain>)">Ollama | 🦜️🔗 LangChain</a>: Ollama allows you to run open-source large language models, such as Llama 2, locally.</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#langgraph>).">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#tools>).">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1247493648677863444)** (1 messages): 

- **LangServe Endpoint Throws Error**: A member asked how to pass human and AI messages to a LangServe endpoint as their server returns an *HTTP 500 error* when attempted. They provided a code snippet showing their method using **RemoteRunnable** and **HumanMessage** and **AIMessage** from `@langchain/core`.
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1247277160360579184)** (4 messages): 

- **Using `ChatPromptTemplate.partial` for Selective Placeholder Replacement**: A member discussed the use of `ChatPromptTemplate.partial` to replace some placeholders with given text while leaving others to be handled via `Runnable.invoke`. They noted that this method is not available for `SystemMessagePromptTemplate`, leading to some confusion.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1247276747225825350)** (7 messages): 

- **Automated Chat Analyzer Debuts**: A member announced that their **automated chat analyzer** can extract Q&A from any message list size. This tool aims to create a plain text file for easy manual editing and minimal compute requirements.

- **CrewAI News-Crew Project**: A member shared their initial approach of implementing a news-crew project using [CrewAI and LangChain tools](https://github.com/touhi99/tldr-ai-news-crew). They welcomed suggestions and feedback for this work-in-progress project, which incorporates multiple agents and voice modality.

- **Dynamic Tool Calling with LangChain**: A YouTube video titled [Dynamic Tool Calling with Visual Agents & LangChain](https://youtu.be/jBbLxbVDaM4?si=b4hMnqYzme11gshL) was shared, showcasing the use of an OpenAI chat model with a dynamic tool implemented as a JavaScript function.

- **EDA GPT for Data Analysis**: An invitation was extended to data analysts to explore a project called [EDA GPT](https://eda-gpt-24.streamlit.app/EDA_GPT). The project is designed to assist with exploratory data analysis (EDA) using a GPT-model interface.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/jBbLxbVDaM4?si=b4hMnqYzme11gshL">Dynamic Tool Calling with Visual Agents &amp; LangChain</a>: In this example, I use an open ai chat model (from langchain) and provide a dynamic tool implemented as a plain-old-javascript-function, and ask the AI a que...</li><li><a href="https://github.com/touhi99/tldr-ai-news-crew">GitHub - touhi99/tldr-ai-news-crew: Experiment repo to learn with CrewAI and make life easier with Agents</a>: Experiment repo to learn with CrewAI and make life easier with Agents - touhi99/tldr-ai-news-crew</li><li><a href="https://eda-gpt-24.streamlit.app/EDA_GPT">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1247345536227479685)** (2 messages): 

- **Transform workflow with Langsmith Hub**: A Medium article discusses how **LangChain Hub** can revolutionize prompt management for JavaScript engineers by centralizing prompts, similar to how GitHub transformed code collaboration. The article emphasizes the **intuitive interface** for uploading, organizing, and versioning prompts, streamlining workflows and fostering innovation. [Read more here](https://medium.com/@kenzic/transform-your-workflow-with-langsmith-hub-a-game-changer-for-javascript-engineers-183af7cc4e31).
  
- **Evaluate LLM System with Langsmith**: A YouTube video titled "*Make Agent 10x cheaper, faster & better?*" dives into **LLM System Evaluation** using Langsmith. The video aims to teach viewers how to build better agents and includes links for additional resources, like a free HubSpot report. [Watch the video](https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@kenzic/transform-your-workflow-with-langsmith-hub-a-game-changer-for-javascript-engineers-183af7cc4e31">Transform Your Workflow with LangSmith Hub: A Game-Changer for JavaScript Engineers</a>: Are scattered AI prompts slowing down your development process? Discover how LangChain Hub can revolutionize your workflow, making prompt…</li><li><a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>: LLM System Eval 101 - Build better agentsGet free HubSpot report of how to land a Job using AI: https://clickhubspot.com/fo2🔗 Links- Follow me on twitter: h...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1247286969537396868)** (43 messages🔥): 

- **Bitshift for all backends sparks debate**: Members discussed incorporating a bitshift operation from [this PR](https://github.com/tinygrad/tinygrad/pull/4728) into all backends, not just assembly. Concerns were raised about the potential performance improvement compared to traditional division and multiplication operations.
- **Device tests missing from 'gpuctypes'**: A member asked why the device tests from the 'gpuctypes' repo were not added, referencing [test_cuda.py](https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36).
- **Hotz's presentation plans**: George Hotz shared his plan for a presentation, which includes slides, a code walkthrough, and a live demo. He aimed for readability and separate explanation of how tinygrad's stack speaks directly to the kernel without dependencies like Nvidia's CUDA libraries.
- **Lean mechanization bounty confusion**: Members attempted to clarify the scope for a bounty related to mergeable `ShapeTrackers` in Lean. The conversation highlighted the lack of clear problem statements and conditions necessary for proof, directing to existing resources like [ShapeTracker in the repo](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py) for better understanding.
- **Endgame strategy discussion**: There was a brief suggestion to discuss the "endgame" or future aims of the tinygrad project, hinting at the potential for ambitious cloud integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.codeeurope.pl/en/speakers/george-hotz">Code Europe - Poland's biggest Tech Festival</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/blob/8f749ae0eb75da821637e613a68c9192da474ac2/docs-legacy/reshape_without_symbolic.md">tinygrad/docs-legacy/reshape_without_symbolic.md at 8f749ae0eb75da821637e613a68c9192da474ac2 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/gpuctypes/blob/c4c4394239dce4d4ecfe7016ca268c49cb2d02a4/test/test_cuda.py#L36">gpuctypes/test/test_cuda.py at c4c4394239dce4d4ecfe7016ca268c49cb2d02a4 · tinygrad/gpuctypes</a>: ctypes wrappers for HIP, CUDA, and OpenCL. Contribute to tinygrad/gpuctypes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/shape/shapetracker.py">tinygrad/tinygrad/shape/shapetracker.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4728">Bitshift by SzymonOzog · Pull Request #4728 · tinygrad/tinygrad</a>: Multiplies and divides by a power of 2 can be executed with a simple bitshift, for cstyle backends it&#39;s done by the compiler but assembly needs this explicitly</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4">Make unit-tests run in all branches · Pull Request #4 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247281718319386717)** (13 messages🔥): 

- **Adding traceback to incontiguous tensors**: Users discussed the usefulness of an error traceback for incontiguous tensors in **tinygrad**. One user suggested *"every time a new `Tensor` is created, it adds a 'traceback' attribute"*, viewing it as a significant enhancement for debugging.

- **Previous attempts and POCs mentioned**: Members recalled previous proof of concepts (POCs) regarding this feature but couldn’t remember the details. **UUUVN** acknowledged having worked on a related POC in PR ##3302 but lacked motivation to complete it.

- **Community interest in revisiting the feature**: Despite the challenges, some members expressed interest in contributing to refine this feature to improve tooling efficiency. Another user highlighted that it would be a *"pretty large boost in tooling usefulness/non-frustration for a small code size/performance cost"*.
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1247305452488757269)** (2 messages): 

- **LLM crafts Python script for structured Gmail data extraction**: In a [repo](https://t.co/N2BJ54zr7i), an LLM is used to extract structured data from Gmail by feeding a subset of emails, allowing it to write a script capable of handling the full dataset. This technique showcases the intelligent scripting potential of LLMs for data extraction tasks.
- **Google Gemini's 1 million token context window demonstrated**: Watch the benefits of [Google Gemini](https://t.co/Qg9ydPzBdd) with its extensive 1 million token context window integrated into a LlamaIndex agent. The demonstration involves answering a multi-part question from complex, heterogeneous documents, illustrating the power of a larger context window in enhancing comprehension and processing.
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247271178272510125)** (43 messages🔥): 

- **Gamecode8 seeks Langchain equivalent in LlamaIndex**: A user asked if there is a LlamaIndex variant of Langchain's `HTMLHeaderTextSplitter` and mentioned difficulties with `HTMLNodeParser`. They decided to implement a custom solution after a discussion on the integration within the `IngestionPipeline` and [provided documentation](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations) for custom transformations.
  
- **Rkhettry resolves node fetching issue in Chroma**: A user initially faced issues fetching documents from a `VectorStoreIndex` in a Chroma vector store and queried for help. They later resolved it by directly accessing documents from the ChromaDB collection.

- **Clarification on setting prompts for LLM in indexes**: There was confusion about setting prompts for indexes; a user shared how they used to set prompts using `LLMPredictor`. It was clarified that prompts are not attached to indexes, and guidance was provided on how to set prompts for LLM ([link](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts#accessingcustomizing-prompts-within-higher-level-modules)).

- **Addressing domain-specific context in RAG pipelines**: A user asked how to inject domain-specific context into RAG pipelines effectively without fine-tuning. Suggestions included rewriting user queries or modifying prompts, but it was noted that for super niche topics, fine-tuning or extensive data curation would be necessary.
  
- **Metadata enhances retrieval in document indexing**: It was discussed how adding metadata to documents, like sales dates, helps improve retrieval. A link to [metadata extraction documentation](https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/) was shared, indicating this practice can aid in disambiguating similar documents.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/MGkByeDm-90?si=14AxwOfkULwC4a2U">&quot;Make Agent 10x cheaper, faster &amp; better?&quot; -  LLM System Evaluation 101</a>: LLM System Eval 101 - Build better agentsGet free HubSpot report of how to land a Job using AI: https://clickhubspot.com/fo2🔗 Links- Follow me on twitter: h...</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/HTML_header_metadata/">Split by HTML header | 🦜️🔗 LangChain</a>: Description and motivation</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/#custom-transformations">Transformations - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts#accessingcustomizing-prompts-within-higher-level-modules">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">Extracting Metadata for Better Document Indexing and Understanding - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1247266522234093691)** (8 messages🔥): 

- **Phi-3 models climb the leaderboard**: "_philschmid_" announced that Phi-3 Medium (14B) and Small (7B) models are now on the [@lmsysorg leaderboard](https://fxtwitter.com/_philschmid/status/1797700161226838362). Medium ranks close to GPT-3.5-Turbo-0613, and Small is comparable to Llama-2-70B and Mistral fine-tunes.
- **Dylan loses his bet**: A member humorously noted "dylan lost his bet," reflecting on recent predictions.
- **Reputation over bets**: Another member expressed regret at not participating in the bets, stating "I’m in it for the reputation gain" and noting that all bets turned into donation-bets for a good cause.

**Link mentioned**: <a href="https://fxtwitter.com/_philschmid/status/1797700161226838362">Tweet from Philipp Schmid (@_philschmid)</a>: Phi-3 Medium (14B) and Small (7B) models are on the @lmsysorg leaderboard! 😍 Medium ranks near GPT-3.5-Turbo-0613, but behind Llama 3 8B. Phi-3 Small is close to Llama-2-70B, and Mistral fine-tunes. ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1247582450197201056)** (29 messages🔥): 

- **OpenAI employees voice safety concerns**: A group of current and former OpenAI employees published [an open letter](https://righttowarn.ai/) about the lack of oversight and whistleblower protections in the AI industry. They believe AI companies have strong financial incentives to avoid effective oversight.
- **AGI projections and industrial scaling**: Leopold Aschenbrenner's [essay](https://situational-awareness.ai/) discusses the massive scale-up in compute clusters and projects AGI capabilities by 2025/26. This has led to debates on whether such exponential scaling trends will continue unabated.
- **Leaked information leads to firing at OpenAI**: Leopold Aschenbrenner was reportedly fired for leaking proprietary information, referencing a [scoop](https://www.theinformation.com/articles/openai-researchers-including-ally-of-sutskever-fired-for-alleged-leaking) by @steph_palazzolo. This leak has raised issues about the significance of trade secrets for AI national security.
- **Questioning perpetual scaling for AGI**: Natolambert criticized the overreliance on extrapolating trends to predict AGI by merely increasing model size. He argued that there's no guarantee that log linear graph trends will continue without hitting a "pinch point."
- **Daniel Kokotajlo's equity dispute**: Kevin Roose reported that Daniel Kokotajlo's equity holdings in OpenAI were at risk unless he signed an NDA, which he refused, leading to public exposure of secret NDAs. [Details](https://www.nytimes.com/2024/06/04/technology/openai-culture-whistleblowers.html?unlocked_article_code=1.xE0._mTr.aNO4f_hEp2J4&smid=nytcore-ios-share&referringSource=articleShare&sgrp=c-cb) were elaborated by @KelseyTuoc.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://situational-awareness.ai/">Introduction - SITUATIONAL AWARENESS: The Decade Ahead</a>: Leopold Aschenbrenner, June 2024 You can see the future first in San Francisco. Over the past year, the talk of the town has shifted from $10 billion compute clusters to $100 billion clusters to trill...</li><li><a href="https://x.com/steph_palazzolo/status/1798041967118750118">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: @leopoldasch shares what he was fired from openai for &#34;leaking&#34; (per our scoop here: https://www.theinformation.com/articles/openai-researchers-including-ally-of-sutskever-fired-for-alleged-le...</li><li><a href="https://x.com/natolambert/status/1798073830906486945">Tweet from Nathan Lambert (@natolambert)</a>: does this make agi scaling people nervous? Wrong answers only. It&#39;s @TheXeophon&#39;s beautiful trendline of LLM scaling</li><li><a href="https://www.cnbc.com/2024/06/04/openai-open-ai-risks-lack-of-oversight.html">Current and former OpenAI employees warn of AI&#x27;s &#x27;serious risk&#x27; and lack of oversight</a>: A group of current and former OpenAI employees published an open letter Tuesday describing concerns about the AI industry&#x27;s rapid advancement.</li><li><a href="https://righttowarn.ai/">A Right to Warn about Advanced Artificial Intelligence</a>: no description found</li><li><a href="https://x.com/_sholtodouglas/status/1798052154709852198">Tweet from Sholto Douglas (@_sholtodouglas)</a>: A brilliant essay that really captures the world view that all the players in the game are operating under.   &#39;AGI by 2027 is strikingly plausible&#39; - all it needs is for the trend lines to hol...</li><li><a href="https://x.com/kelseytuoc/status/1798029447662371231?s=46">Tweet from Kelsey Piper (@KelseyTuoc)</a>: Fantastic reporting from Kevin and some new details on Daniel Kokotajlo&#39;s situation: he had equity holdings in OpenAI worth $1.7million. He was told it would be cancelled if he didn&#39;t sign. He...</li><li><a href="https://x.com/natolambert/status/1798042504635523276">Tweet from Nathan Lambert (@natolambert)</a>: I was listening to the latest @dwarkesh_sp with @leopoldasch and I&#39;m just kind of shocking how many times I hear the same take that is an extrapolation of trend lines guaranteeing us &#34;AGI&#34;...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1247597857268306090)** (6 messages): 

- **Scaling Law Dismissal Meme**: A user mockingly portrayed evolution conversations about pattern-seeking brains and predators, ending with a jibe at those who believe in scaling laws applying meaningfully in 2030+: *"Actually believes scaling laws will apply meaningfully in 2030+ like a boss."*
- **Call for Massive Parameter Scaling**: There was a humorous plea for scaling AI models to 10 trillion parameters, with a dramatic *"please bro just 100k more GPUs"*.
- **Frustration with AGI Debates**: Expressing irritation, a user commented on the polarized views in AGI discussions, criticizing both optimists and doomers for their *"little epistemic uncertainty"* despite claiming Bayesian logic. *"the barriers to further improvements always seem so trivial to them."*
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

420gunna: 👍
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1247279508260917329)** (41 messages🔥): 

- **Torchtune seeks AI News coverage**: A core developer from the Torchtune Discord server requested inclusion in the AI News newsletter, highlighting their active community discussions on SOTA models and methods. The server invite is [available here](https://discord.gg/6cGsKMM5).
- **AI News email formatting issue**: A member reported a formatting issue with AI News emails in ProtonMail, where only links and images were visible against a white background. The issue appears linked to ProtonMail's dark mode settings.
- **Automated podcast transcripts**: A member asked about how the podcast generates its transcripts with speaker identification, and it was revealed that they use Google's smol-podcaster tool, with manual label replacements for speaker names.
- **LiveKit raises $22.5M Series A**: LiveKit announced a $22.5M Series A funding to build the transport layer for AI, emphasizing real-time voice and video as future interaction methods with computers. The [fundraising was challenging](https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww), but the onset of GPT-4 demonstrated the immediate need for such technology.
- **Twelve Labs' $50M Series A for multimodal AI**: Twelve Labs raised $50M to further develop its video understanding AI foundation models, introducing the Marengo 2.6 model with multimodal support in a single API. [Full details here](https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tree-diffusion.github.io/">no title found</a>: no description found</li><li><a href="https://discord.gg/6cGsKMM5.">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/dsa/status/1798027280872321117?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from dsa (@dsa)</a>: Today we’re announcing LiveKit’s $22.5M Series A to build the transport layer for AI.  This wasn’t an easy fundraise. Late last year, we pitched investors that realtime voice and video would become TH...</li><li><a href="https://x.com/itsandrewgao/status/1797739301947748541?s=46&t=2qGo-Hp_MDNyh14F888CkQ">Tweet from Andrew Gao (@itsandrewgao)</a>: no description found</li><li><a href="https://x.com/dkokotajlo67142/status/1797994238468407380?s=46&t=JE84TqLviekDnEt8MAT-Eg">Tweet from Daniel Kokotajlo (@DKokotajlo67142)</a>: 1/15: In April, I resigned from OpenAI after losing confidence that the company would behave responsibly in its attempt to build artificial general intelligence — “AI systems that are generally smarte...</li><li><a href="https://future.mozilla.org/builders/">Mozilla Builders</a>: no description found</li><li><a href="https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Rowan Cheung (@rowancheung)</a>: A GPT-4 level chatbot, available to use completely free, running at over 800 tokens per second on Groq.  I&#39;m genuinely mindblown by LlaMA 3.  Try it with the link in the next tweet.</li><li><a href="https://x.com/rowancheung/status/1781732100556591525?s=46&t=90xQ8sGy6">Tweet from Rowan Cheung (@rowancheung)</a>: A GPT-4 level chatbot, available to use completely free, running at over 800 tokens per second on Groq.  I&#39;m genuinely mindblown by LlaMA 3.  Try it with the link in the next tweet.</li><li><a href="https://x.com/MSFTResearch/status/1797662278394827029">Tweet from Microsoft Research (@MSFTResearch)</a>: Aurora, a new AI foundation model from Microsoft Research, can transform our ability to predict and mitigate extreme weather events and the effects of climate change by enabling faster and more accura...</li><li><a href="https://www.forbes.com/sites/alexkonrad/2024/06/04/inside-silicon-valley-influence-battle-for-ai-future/?sh=5fece9ce2dc4">Vinod Khosla, Marc Andreessen And The Billionaire Battle For AI's Future</a>: Billionaire investors of the internet era are locked in a policy battle to determine whether AI’s future will be one of concentrated safety or of unfettered advancement. </li><li><a href="https://github.com/go-go-golems/geppetto/blob/main/cmd/pinocchio/prompts/code/plantuml/activity.yaml">geppetto/cmd/pinocchio/prompts/code/plantuml/activity.yaml at main · go-go-golems/geppetto</a>: golang GPT3 tooling. Contribute to go-go-golems/geppetto development by creating an account on GitHub.</li><li><a href="https://www.prweb.com/releases/twelve-labs-earns-50-million-series-a-co-led-by-nea-and-nvidias-nventures-to-build-the-future-of-multimodal-ai-302163279.html">Twelve Labs Earns $50 Million Series A Co-led by NEA and NVIDIA's NVentures to Build the Future of Multimodal AI</a>: /PRNewswire-PRWeb/ -- Twelve Labs, the video understanding company, today announced that it raised $50 million in Series A funding to fuel the ongoing...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247265025765277768)** (17 messages🔥): 

- **Fun at conferences even without results**: Attendees express enjoyment of conferences despite not having accepted papers, highlighting the value of participation itself.
- **Struggling with custom OrPO formatter**: A member seeks help loading a custom OrPO formatter Python script for tokenizing pre-converted datasets. [Related script link](https://discord.com/channels/1104757954588196865/1117071926926512248/1245037389886521464).
- **Critiquing AI in medical VQA**: A tweet [shared by a member](https://x.com/xwang_lk/status/1797475354745197029?t=nLUioafCJbCenSnEQ8xfIw&s=19) criticizes state-of-the-art models like GPT-4V and Gemini Pro for performing worse than random in medical VQA tasks, introducing the ProbMed dataset to evaluate performance. Discussion arises on the inadequacy of vision LLMs for medical image diagnosis.
- **Seeking arXiv endorsement**: One member asks for endorsement on arXiv for the cs.LG category but later resolves the issue by using their organizational email.

**Link mentioned**: <a href="https://x.com/xwang_lk/status/1797475354745197029?t=nLUioafCJbCenSnEQ8xfIw&s=19">Tweet from Xin Eric Wang (@xwang_lk)</a>: Can we really trust AI in critical areas like medical image diagnosis? No, and they are even worse than random. Our latest study, &#34;Worse than Random? An Embarrassingly Simple Probing Evaluation of...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1247513942801125436)** (13 messages🔥): 

- **QLoRA Training Output Issues**: A member reported an issue where they could not start with the QLoRA output despite completing the training. They requested assistance to resolve the matter urgently.

- **LoRA Training Without Model Download Explanation**: A user queried why their LoRA training proceeded without error even though the model was not pre-downloaded. Another member clarified that the training scripts automatically handle the download from the Hugging Face Model Hub if the model is not found locally.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=022a3d10-bf15-408b-b525-168f5f199dd0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e364cc67-a6a9-4aa8-9173-f7bfb1d02469)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1247490831422853130)** (13 messages🔥): 

- **Cohere's Artificial Ivan V.4.0 is a hit**: Cohere's project "Artificial Ivan" is now at version 4.0 and can troubleshoot code. *"We also automated his golden thoughts at some stage"* with a link to [affirmations](https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations).
- **Real Ivan's Retirement Dream**: There was a humorous comment about the real Ivan retiring at age 35, implying satisfaction with the project’s progress.
- **Artificial Ivan's other projects**: It's clarified that the app mentioned isn't just for Ivan but is an umbrella for various projects led by another user.
- **Cohere Discord is lively**: A user noted the high activity levels in the Cohere Discord compared to Langchain with a playful jab, *“chronically online AI lab”*.

**Link mentioned**: <a href="https://amorisot-cohere-demos-onceuponatime-x1w9hn.streamlit.app/affirmations">no title found</a>: no description found

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1247493689798955110)** (1 messages): 

- **Configuring Aya 23 with Llama.cpp and LangChain**: A project using **Aya 23** as an LLM with **Llama.cpp** and **LangChain** is shared. The user specifically asks for help on how to add a stop to prevent the system from generating content beyond the conversation, proposing something like "\n".

- **Code Implementation for Aya 23 Project**: The user provides a detailed code sample showing the integration setup of Aya 23 with prompts and conversation memory. The code aims for concise responses in Spanish, with parameters set for the model's performance and looped user interaction.
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1247314853354410036)** (9 messages🔥): 

- **CUDA recompile issues resolved**: When facing a CUDA error, a member found that using `--recompile --gpu NVIDIA` resolved their problem. They noted that `-ngl 9999` must come after `--recompile` for it to work effectively.

- **Community assists with troubleshooting**: A member seeking help with CUDA recompilation received a [GitHub link](https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine) but later resolved the issue by reading the `--help` options. The resolution was met with appreciation and camaraderie.

- **Cookies and camaraderie**: The community emphasized mutual support, learning from each other, and humorously mentioned that they're also there for the cookies.

- **Future of CPU vector computations**: A YouTube video titled "The Magic of RISC-V Vector Processing" was shared, providing insights into the newly ratified 1.0 RISC-V Vector Specification and its applications. [Watch the video](https://www.youtube.com/watch?v=Ozj_xU0rSyY).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Ozj_xU0rSyY">The Magic of RISC-V Vector Processing</a>: The 1.0 RISC-V Vector Specification is now Ratified, and the first pieces of silicon using the new spec are starting to hit the shelves.  I go over the utili...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/build/llamafile-upgrade-engine">llamafile/build/llamafile-upgrade-engine at main · Mozilla-Ocho/llamafile</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1247320450871328861)** (7 messages): 

- **Gemini setup simplified for Windows**: One member provided a step-by-step guide to getting the **Gemini model** working with Google Cloud SDK on Windows, including necessary commands and tips to bypass common issues. They also mentioned that the documentation seems outdated and hence the need for this guide.
- **Curiosity about secure version release**: A member inquired about updates on the release of a **secure version**, though no specific responses were noted.
- **Innovative AR laptop technology**: A user shared a link to [Sightful's Spacetop](https://www.sightful.com/), an AR laptop, praising its no-battery-needed glasses and its innovative design that replaces a physical laptop display.
- **Opinions on Xreal glasses**: One member noted they use Xreal glasses with a MacBook and highlighted a need for more resolution and field of view. Others discussed battery life and availability issues, highlighting that the Xreal glasses do not have their own battery and are powered by the device they are connected to.

**Link mentioned**: <a href="https://www.sightful.com/">Spacetop - Meet The AR Laptop for Work</a>: Discover Spacetop, the AR laptop for work. Redefine mobile computing. Experience AR like never before with unmatched performance and cutting-edge tech.

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1247355694789820486)** (1 messages): 

- **Infinite Loop in Poetry Run 01 on Macbook M1**: A user shared an issue where running `poetry run 01 --local` on a Macbook M1 with **ollama llama3** does not stop and enters an infinite loop. They are seeking a fix for this problem.
  

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1247533664401625130)** (1 messages): 

- **Infer: Summer ‘24 offers insight-packed agenda**: Qwak’s free virtual conference on June 26 invites AI and ML enthusiasts to learn from industry experts through live interactions and practical sessions. [Registration link](https://tinyurl.com/j8z6s8ka) and details for joining as a future speaker are included.
  
- **Learn Recommender Systems and AI in Sports**: Sessions will cover building advanced ML models for recommender systems, and examining AI-based sports narrators and predictive solutions in sports analytics.

- **Tackle AI Safety and Regulatory Challenges**: The event promises insights into addressing risks like inaccurate content, implementing "Schematic Questioning," and navigating regulatory hurdles to ensure secure AI deployments.

- **Speakers from leading companies**: Experts from Disney Streaming, Lightricks, LSports, and Lili Banking will be sharing their valuable insights and experiences in the conference.

- **Featured Presentations on Running LLMs in Production**: Look out for keynotes on principles for running Large Language Models (LLMs) like GPT and Llama in various sectors, highlighting the operationalization of AI in production environments. 

![Russ Wilcox](https://cdn.prod.website-files.com/64b3ee21cac9398c75e5d3ac/65d4a59498a2a15e477c88df_Russ%20Wilcox.png)
![Hudson Buzby](https://cdn.prod.website-files.com/64b3ee21cac9398c75e5d3ac/65d4a5b13ae20858614e0fcd_Hudson.png)

**Link mentioned**: <a href="https://tinyurl.com/j8z6s8ka">Infer Summer ‘24 by Qwak | The Engineering Behind AI and ML</a>: Infer Summer ‘24 by Qwak brings AI leaders to share how the world’s leading companies use ML and AI in production. Join live on Jun 26, 2024, 11:00 AM EDT

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1247488626326245398)** (1 messages): 

- **xLSTM Source Code Released**: Dr. Tristan Behrens announced the release of xLSTM's source code with an enthusiastic *"drop everything NXAI has released the official"*. The announcement was made on [LinkedIn](https://www.linkedin.com/posts/dr-tristan-behrens-734967a2_drop-everything-nxai-has-released-the-official-activity-7203659602628935682-GwOA).
  

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

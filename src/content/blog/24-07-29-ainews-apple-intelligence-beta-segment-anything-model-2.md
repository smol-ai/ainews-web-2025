---
id: 010a9b07-042e-4425-9fef-2806762d8754
title: Apple Intelligence Beta + Segment Anything Model 2
date: '2024-07-30T02:45:55.827150Z'
original_slug: ainews-apple-intelligence
description: >-
  Apple has delayed the official release of its Apple Intelligence to iOS 18.1
  in October but launched a developer preview on MacOS Sequoia, iOS 18, and
  iPadOS 18, accompanied by a detailed 47-page paper revealing extensive
  pretraining and hardware details. Meanwhile, Meta advanced its computer vision
  capabilities with a new version of its Segment Anything Model that uses memory
  attention to scale image segmentation to video efficiently.
tags:
  - apple
  - apple-intelligence
  - ios-18
  - macos-sequoia
  - meta
  - segment-anything
  - image-segmentation
---


<!-- buttondown-editor-mode: plaintext -->**The second largest LLM deployment of 2024 is so delayed/so here.**

> AI News for 7/26/2024-7/29/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**325** channels, and **6654** messages) for you. Estimated reading time saved (at 200wpm): **716 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Meta continued its open source AI roll with a worthy sequel to [last year's Segment Anything Model](https://www.latent.space/p/segment-anything-roboflow). Most notably, ontop of being a [better image model than SAM1](https://x.com/josephofiowa/status/1818087114208342527), it now uses [memory attention to scale up image segmentation to apply to video, using remarkably little data and compute](https://x.com/swyx/status/1818085925714620449).  

But the computer vision news was overshadowed by **Apple Intelligence**, which both [delayed the official release to iOS 18.1 in October](https://archive.ph/Wm85S) and released in Developer preview on [MacOS Sequoia (taking a 5GB download)](https://x.com/zaidmukaddam/status/1818000899098231091), [iOS 18](https://x.com/sparkhi/status/1818006585324761098), and [iPadOS 18](https://x.com/TyleTweets/status/1818061615545114704) today (with a very short [waitlist](https://x.com/BrandonButch/status/1817974792282325374), [Siri 2.0 not included](https://x.com/VadimYuryev/status/1818036274223284557), Europeans not included), together with a surprise [47 page paper](https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models.pdf) going into further detail than their June keynote ([our coverage here](https://buttondown.email/ainews/archive/ainews-talaria-apples-new-mlops-superweapon-4066/)).

Cue widespread demos of:

Notifications screening

 ![image.png](https://assets.buttondown.email/images/0427becf-2742-4f17-aaab-97d34c6176d0.png?w=960&fit=max) 

Rewriting in arbitrary apps with low power
 ![image.png](https://assets.buttondown.email/images/82c4c542-7322-4e5f-93e2-831c94ef966d.png?w=960&fit=max) 

Writing tools

 ![image.png](https://assets.buttondown.email/images/fabff175-36c2-4e16-a122-9478388ff56f.png?w=960&fit=max) 

and more. 

As for the paper, the best recap threads from [Bindu](https://x.com/bindureddy/status/1818068830570299521) and [Maxime](https://x.com/maximelabonne/status/1818017461738102823) and  
 [VB](https://x.com/reach_vb/status/1818014366555586611) probably cover it all. Our highlight is the amount of pretrain detail contained on page 6 and 7:

 ![image.png](https://assets.buttondown.email/images/721a24de-ad6a-4dbb-b4c7-cf48112218bd.png?w=960&fit=max) 

- **Data**: fresh dataset scrape of ? Applebot web crawl, ? licensed datasets, ? code, 3+14b math tokens, ? "public datasets" leading to a final 6.3T tokens for CORE pretraining, 1T tokens with higher code/math mix for CONTINUED pretraining, and 100B tokens for context lengthening (to 32k)
- **Hardware**: AFM was trained with v4 and v5p Cloud TPUs, not Apple Silicon!! AFM-server: 8192 TPUv4, AFM-on-device: 2048 TPUv5p
- **Post Training**: "While Apple Intelligence features are powered through adapters on top of the base model, empirically we found that **improving the general-purpose post-training lifts
the performance of all features**, as the models have stronger capabilities on instruction following, reasoning, and writing.
- [Extensive use of synthetic data for Math, Tool Use, Code, Context Length, Summarization (on Device), automatic redteaming, and committee distillation](https://x.com/swyx/status/1818109227090825261)

Also notable, they disclose their industry standard benchmarks, which we have [taken the liberty of extracting and comparing with Llama 3](https://x.com/swyx/status/1818114253523656946):

 ![image.png](https://assets.buttondown.email/images/d0c3d901-1d18-4d8c-b0d5-c92e06866bc3.png?w=960&fit=max) 

Yes they are notably, remarkably, significantly lower than Llama 3, but we wouldn't worry too much about that as we trust Apple's Human Evaluations.

 ![image.png](https://assets.buttondown.email/images/f44fe5f9-5c2a-4eb1-835e-1fb6fb92c3c8.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/3ac6a2e7-6435-4f34-97c6-5084a17a14ee.png?w=960&fit=max) 


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

**AI Model Developments and Industry Updates**

- **Llama 3.1 Release**: Meta released Llama 3.1, including a 405B parameter model, the first open-sourced frontier model on par with top closed models. [@adcock_brett](https://twitter.com/adcock_brett/status/1817591648764801399) noted it\'s "open source and free weights and code, with a license enabling fine-tuning, distillation into other models and deployment." The model supports eight languages and extends the context window to **128K tokens**.

- **Mistral AI\'s Large 2**: [@adcock_brett](https://twitter.com/adcock_brett/status/1817591702665695549) reported that Mistral released Large 2, its flagship AI model scoring close to Llama 3.1 405b and even surpassing it on coding benchmarks while being much smaller at 123b. This marks the release of "two GPT-4 level open models within a week."

- **OpenAI Developments**: OpenAI announced SearchGPT, an AI search engine prototype that combines AI models with web information. [@adcock_brett](https://twitter.com/adcock_brett/status/1817591625918423453) mentioned it "organizes search results into summaries with source links and will be initially available to 10,000 test users." Additionally, [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1817563907113763280) shared insights on OpenAI\'s potential impact on call centers, suggesting AI agents could replace human operators within two years.

- **Google DeepMind\'s Achievements**: [@adcock_brett](https://twitter.com/adcock_brett/status/1817591671703445839) highlighted that "Google DeepMind\'s AlphaProof and AlphaGeometry 2 achieved a significant milestone in AI math reasoning capabilities," attaining a silver medal-equivalent score at this year\'s IMO.

**AI Research and Technical Advancements**

- **GPTZip**: [@jxmnop](https://twitter.com/jxmnop/status/1817660589797486860) introduced gptzip, a project for compressing strings with language models, achieving "5x better rates than gzip" using Hugging Face transformers.

- **RAG Developments**: [@LangChainAI](https://twitter.com/LangChainAI/status/1817623555946709323) shared RAG Me Up, a generic framework for doing RAG on custom datasets easily. It includes a lightweight server and UIs for communication.

- **Model Training Insights**: [@abacaj](https://twitter.com/abacaj/status/1817694944774987951) discussed the importance of low learning rates during fine-tuning, suggesting that weights have "settled" into near-optimal points due to an annealing phase.

- **Hardware Utilization**: [@tri_dao](https://twitter.com/tri_dao/status/1817711223879971179) clarified that "nvidia-smi showing \'GPU-Util 100%\' doesn\'t mean you\'re using 100% of the GPU," an important distinction for AI engineers optimizing resource usage.

**Industry Trends and Discussions**

- **AI in Business**: There\'s ongoing debate about the capabilities of LLMs in building businesses. [@svpino](https://twitter.com/svpino/status/1817546892395393325) expressed skepticism about non-technical founders building entire SaaS businesses using LLMs alone, highlighting the need for capable human oversight.

- **AI Ethics and Societal Impact**: [@fchollet](https://twitter.com/fchollet/status/1817651235341861256) raised concerns about cancel culture and its potential impact on art and comedy, while [@bindureddy](https://twitter.com/bindureddy/status/1817635971388862531) shared insights on LLMs\' reasoning capabilities, noting they perform better than humans on real-world reasoning problems.

- **Open Source Contributions**: The open-source community continues to drive innovation, with projects like [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1817706514792013895) sharing a local voice chatbot powered by Ollama, Hugging Face Transformers, and Coqui TTS Toolkit using local Llama.

**Memes and Humor**

- [@willdepue](https://twitter.com/willdepue/status/1817640842598727697) joked about OpenAI receiving "$19 trillion in tips" in the last month.

- [@vikhyatk](https://twitter.com/vikhyatk/status/1817618257266040983) shared a humorous anecdote about getting hit with a $5k network transfer bill for using S3 as a staging environment.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Ultra-Compact LLMs: Lite-Oute-1 300M and 65M Models**

- **Lite-Oute-1: New 300M and 65M parameter models, available in both instruct and base versions.** ([Score: 59, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ee5lzo/liteoute1_new_300m_and_65m_parameter_models/)): **Lite-Oute-1** has released new **300M** and **65M** parameter models in both instruct and base versions, available on Hugging Face. The **300M** model, built on the **Mistral** architecture with a **4096** context length, aims to improve upon the previous 150M version by processing **30 billion** tokens, while the **65M** model, based on **LLaMA** with a **2048** context length, is an experimental ultra-compact version processing **8 billion** tokens, both trained on a single **NVIDIA RTX 4090**.
    - /u/hapliniste: "As much as I'd like nano models so we can finetune easily on specific tasks, isn't the benchmark radom level? 25% on mmlu is the same as random choice right?I wonder if it still has some value for autocompletion or things like that."

**Theme 2. AI Hardware Investment Challenges: A100 GPU Collection**


- **[The A100 Collection and the Why](https://www.reddit.com/gallery/1eecrnp)** ([Score: 51, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1eecrnp/the_a100_collection_and_the_why/)): The post describes a personal investment in **23 NVIDIA A100 GPUs**, including **15 80GB PCIe water-cooled**, **5 40GB SXM4 passive-cooled**, and **8 additional 80GB PCIe water-cooled** units not pictured. The author expresses regret over this decision, citing difficulties in selling the water-cooled units and spending their entire savings, while advising others to be cautious about letting hobbies override common sense.


**Theme 4. New Magnum 32B: Mid-Range GPU Optimized LLM**

- **"The Mid Range Is The Win Range" - Magnum 32B** ([Score: 147, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eenk1e/the_mid_range_is_the_win_range_magnum_32b/)): Anthracite has released **Magnum 32B v1**, a **Qwen finetune** model targeting **mid-range GPUs** with **16-24GB** of memory. The release includes **full weights in BF16** format, as well as **GGUF** and **EXL2** versions, all available on [Hugging Face](https://huggingface.co/anthracite-org/magnum-32b-v1).
    - Users discussed creating a **roleplay benchmark**, with suggestions for a community-driven "hot or not" interface to evaluate model performance on writing style, censorship levels, and character adherence.
    - The profile pictures in Magnum releases feature **Claude Shannon**, the father of information theory, and **Tsukasa** from Touhou. Users appreciated this unique combination of historical and fictional characters.
    - A user shared a **500-token story** generated by Magnum 32B, featuring two cyborgs in Elon Musk's factory uncovering a corporate conspiracy. The story showcased the model's creative writing capabilities.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Industry Discussion**

- **Criticism of misleading AI articles**: In /r/singularity, a [post questions the subreddit's apparent bias against AI](https://www.reddit.com/r/singularity/comments/1ee7q64/i_dont_get_why_the_sub_despises_ai_so_much_that/), specifically referencing a potentially misleading article about OpenAI's financial situation. The post title suggests that the community may be overly critical of AI advancements.

  Key points:
  - The post links to an article claiming OpenAI could face bankruptcy within 12 months, projecting $5 billion in losses.
  - The post received a relatively high score of 104.5, indicating significant engagement from the community.
  - With 192 comments, there appears to be substantial discussion around this topic, though the content of these comments was not provided.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Model Releases and Performance**

- **Llama 3 Shakes Up the Leaderboards**: **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** from Meta has quickly risen to the top of leaderboards like **ChatbotArena**, outperforming models like **GPT-4-Turbo** and **Claude 3 Opus** in over 50,000 matchups.
   - The **Llama 405B Instruct** model achieved an average accuracy of **0.861** across multiple subjects during the MMLU evaluation, with notably strong performances in biology and geography. The evaluation was completed in about **two hours**, demonstrating efficient processing.
- **DeepSeek V2 Challenges GPT-4**: **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** with **236B parameters** has shown impressive performance, surpassing **GPT-4** in some areas on benchmarks like **AlignBench** and **MT-Bench**.
   - The model's strong showing across various benchmarks has sparked discussions about its potential to compete with leading proprietary models, highlighting the rapid progress in open-source AI development.
  


**2. AI Development Tools and Frameworks**

- **LlamaIndex Launches New Course with Andrew Ng**: **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** has announced a new course on building agentic RAG systems in collaboration with Andrew Ng's DeepLearning.ai, aiming to enhance developers' skills in creating advanced AI applications.
   - This collaboration highlights the growing importance of Retrieval-Augmented Generation (RAG) in AI development and showcases LlamaIndex's commitment to educating the community on cutting-edge techniques.
- **Axolotl Expands Dataset Format Support**: **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** has expanded its support for diverse dataset formats, enhancing its capabilities for instruction tuning and pre-training LLMs.
   - This update allows developers to more easily integrate various data sources into their model training pipelines, potentially improving the quality and diversity of trained models.
  

**3. AI Infrastructure and Optimization**

- **vAttention Revolutionizes KV Caching**: The **[vAttention](https://arxiv.org/abs/2405.04437)** system dynamically manages KV-cache memory for efficient LLM inference without relying on PagedAttention, offering a new approach to memory management in AI models.
   - This innovation addresses one of the key bottlenecks in LLM inference, potentially enabling faster and more efficient deployment of large language models in production environments.
  


**4. Multimodal AI Advancements**

- **Meta Unveils Segment Anything Model 2**: Meta has released **[Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2)**, a unified model for real-time, promptable object segmentation in both images and videos, available under an Apache 2.0 license.
   - SAM 2 represents a significant leap in multimodal AI, trained on a new dataset of approximately **51,000 videos**. This release includes model inference code, checkpoints, and example notebooks to assist users in implementing the model effectively.

**5. LLM Advancements**

- **Llama 405B Instruct Shines in MMLU**: **Llama 405B Instruct** achieved an average accuracy of **0.861** in the MMLU evaluation, excelling in subjects like biology and geography, completing the evaluation in about **two hours**.
   - This performance has sparked discussions on the robustness of the evaluation process and the model's efficiency.
- **Quantization Concerns in Llama 3.1**: Members raised concerns about **Llama 3.1**'s performance drop due to quantization, with better results noted using **bf16** ([X.com post](https://x.com/_xjdr/status/1816892492580814856)).
   - Discussions suggest that the quantization impacts might be tied to the total data volume rather than just the parameter count.


---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Synthetic Data Generation Tools Under Scrutiny**: Members discussed tools for generating synthetic data, highlighting **Argila** and **Distillabel** while gathering resources for a holistic overview.
   - A [Twitter thread](https://twitter.com/natolambert/status/1817964740611817926) was shared, though its specific relevance to synthetic data tools remains ambiguous.
- **Moondream's Video Analysis Potential**: **Moondream** was considered for identifying **criminal activity** by analyzing selective frames in videos, aiming for effective detection of dangerous actions.
   - Productivity tips emphasized the necessity of quality images and robust prompting strategies for optimal performance.
- **Llama 405B Instruct Shines in MMLU**: The **Llama 405B Instruct** model achieved an average accuracy of **0.861** during the MMLU evaluation, with prominent results in biology and geography.
   - The evaluation process was executed efficiently, wrapping up in about **two hours**.
- **RAG Production Challenges and Solutions**: A recent post detailed common issues faced in RAG production, showcasing potential solutions and best practices in a [LinkedIn post](https://www.linkedin.com/posts/joannastoffregen_rag-in-production-issues-and-solutions-ugcPost-7219283564062785536-H2rk).
   - Community members emphasized the importance of shared knowledge for overcoming obstacles in RAG implementation.
- **JSON-MD Integration for Task Management**: Discussions focused on using **JSON** for task organization while leveraging **Markdown** for readability, paving the way for a synchronized contribution process.
   - The **Operation Athena** website is poised to serve as a dynamic frontend for task management, designed for collaborative interaction.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Fine-tuning w2v2-bert on Ukrainian Achieves 400k Samples**: A project demonstrated fine-tuning of [w2v2-bert](https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo) on Ukrainian with the YODAS2 dataset, totaling **400k samples** to enhance model accuracy in the language.
   - This initiative expands model capabilities for Ukrainian, addressing language processing needs effectively.
- **Meta Llama 3.1 Performance Insights**: In-depth evaluation of **Meta Llama 3.1** models compared **GPU and CPU performance**, documented in a detailed [blog post](https://vandu.tech/meta-llama-3-1-405b-gpu-vs-cpu-performance-evaluation-and-ram-considerations/), revealing notable findings.
   - The evaluation included performance insights along with video demonstrations of test scenarios, shedding light on computational efficiency.
- **Issues with Hugging Face Tokenizer Implementation**: A member highlighted that `tokenizer.apply_chat_template` is broken in the recent Hugging Face Transformers, with `add_generation_prompt = False` not functioning correctly.
   - This issue has sparked conversations about potential workarounds and the implications for ongoing projects integration.
- **Research Collaboration Opportunities at the Hackathon**: Steve Watts Frey announced a new **6-day ultra-hackathon** aimed at advancing open-source benchmarks, featuring substantial computing resources for participants and outlining collaboration chances.
   - Teams are encouraged to take advantage of this chance to drive research efforts forward, boosting community engagement.
- **User Experiences Highlighting Challenges with Data Management**: Members shared experiences on dataset management, noting that organizing training data in order of increasing difficulty led to improved model performance.
   - Additionally, discussions surfaced around enhancing breed classification models on [Kaggle](https://www.kaggle.com/code/root31415/breed-classification), tackling concerns about learning efficiency.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio performance varies by GPU**: Users noticed significant variations in performance metrics across models, particularly with **Llama 3.1**, as GPU configurations impacted speed and context length settings.
   - Some users reported different tokens per second rates, emphasizing the role of their **GPU type** and **RAM specifications** on inference efficiency.
- **Model loading issues require updates**: Several users faced model loading errors with **Llama 3.1**, citing tensor-related issues and recommending updating LM Studio or reducing context size.
   - Guidelines were shared on troubleshooting, focusing on **GPU compatibility** and proper model directory structures.
- **Fine-tuning vs embeddings debate**: Discussion centered on the effectiveness of fine-tuning versus embeddings, highlighting the necessity for well-prepared examples for model operation.
   - Participants emphasized that inadequate context or tutorial content could impede models' performance.
- **Snapdragon X Elite ARM CPU generates buzz**: The performance of the new **Snapdragon X Elite ARM CPU** in Windows 11 sparked conversation, with a review video titled ["Mac Fanboy Tries ARM Windows Laptops"](https://youtu.be/3GZ4sqB3juQ) generating user interest.
   - Members speculated on real-world usability and shared personal experiences with ARM CPU setups.
- **GPU preferences for model training**: Consensus emerged that the **4090 GPU** is optimal for model training, outperforming older models such as the **K80** or **P40**.
   - Members highlighted the importance of modern hardware for effective **CUDA** support, especially when handling large models.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AI Tools Face-Off: ComfyUI Takes the Crown**: In discussions around **Stable Diffusion**, users compared **ComfyUI**, **A1111**, and **Forge**, revealing that ComfyUI offers superior control and model flexibility, enhancing speed.
   - Concerns arose regarding Forge's performance after its latest update, prompting users to consider A1111 as a viable alternative.
- **Frustration Over Inpainting Quality**: User Modusprimax reported continual **blurry outputs** with Forge's new inpainting feature, despite multiple configuration attempts.
   - The community suggested reverting to **ComfyUI** or trying earlier Forge versions for potentially better inpainting outcomes.
- **Strategies for Character Consistency Revealed**: Participants shared techniques using specific models and **IP adapters** to maintain character consistency in AI-generated images, particularly recommending the **'Mad Scientist'** model.
   - This approach is noted to yield better results in character anatomy, helping to refine user outputs.
- **Censorship Concerns with AMD's Amuse 2.0**: Discussion ensued around **AMD’s Amuse 2.0 model**, criticized for heavy censorship affecting its ability to render certain body curves accurately.
   - This sparked broader conversations about the implications of censorship on creativity within AI applications.
- **Community Emphasizes Learning Resources**: Several users highlighted the necessity of utilizing video tutorials and community forums to improve understanding of **Stable Diffusion** prompts and operations.
   - Crystalwizard encouraged diligence in exploring **ComfyUI** features while clarifying misconceptions about various AI generation tools.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SearchGPT Performance Highlights**: Users shared positive feedback about **SearchGPT**, noting its ability to search through credible sources and utilize **Chain of Thought (CoT)** reasoning during inquiries.
   - One user demonstrated its practicality by showcasing a calculation of trip costs while retrieving relevant car model information.
- **ChatGPT Connectivity Frustrations**: Multiple reports emerged regarding ongoing access issues with **ChatGPT**, with users experiencing significant loading delays.
   - One user expressed particular frustration over being unable to log in for weeks and receiving no assistance from OpenAI support.
- **AI Assists Coding Efficiency**: Users eagerly discussed their experiences using AI tools for coding, highlighting successful Python scripts created to launch Chrome and other tasks.
   - One user praised the feedback loop enabled by **ChatGPT** on their server, enhancing collaboration and code quality.
- **Voice Mode Excitement**: Anticipation grew around the rollout of **voice mode** in ChatGPT, expected to launch this week for a select group of users.
   - Speculation arose regarding how users would be chosen to access this feature, generating excitement within the community.
- **Cultural Exchanges in the Community**: A user identified as Russian engaged with another who identified as Ukrainian, fostering a sharing of cultural backgrounds.
   - This brief interaction highlighted the diverse community and encouraged inclusivity among members.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Best Practices for Using Unsloth AI**: Users discussed the effectiveness of various system messages for **Llama 3.1**, with the default from Unsloth notebooks sufficing for tasks. Some opted to remove the system message to save context length without loss in performance.
   - Conversations highlighted how flexible modeling aligns well with task-specific needs, especially when optimizing GPU memory usage.
- **Fine-tuning with LoRa Adapters**: Members confirmed that **LoRa adapters** from fine-tuning can be applied to original **Llama** models, granted the base model remains unchanged. Uncertainties remain about the compatibility across model versions, necessitating attention.
   - Apple's usage of **LoRA** for fine-tuning demonstrated effective balance between capacity and inference performance, particularly for task-specific applications.
- **Quantization Trade-offs**: Discussions addressed the performance vs VRAM costs of 4-bit versus 16-bit models, urging experimentation as users find varying results in efficacy. Notably, **16-bit models** deliver superior performance despite demanding four times the VRAM.
   - Members emphasized the application of these quantization strategies based on unique workloads, reinforcing the necessity for hands-on metrics.
- **Hugging Face Inference Endpoint Security**: Clarifications around **Hugging Face** endpoints emphasized that 'protected' status applies only to one's token; sharing could lead to unauthorized access. It was stressed that safeguarding your token is paramount.
   - Overall, members cautioned against potential security risks, underscoring vigilance in managing sensitive credentials.
- **Efficiency in ORPO Dataset Creation**: A member raised concerns about the manual nature of creating **ORPO datasets**, exploring the feasibility of a UI to streamline this process. Suggestions included leveraging smarter models for efficiently producing responses.
   - The discourse stressed the need for automation tools to overcome repetitive tasks, potentially enhancing productivity and focus on model optimization.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Mojo Community Meeting Scheduled**: The next **Mojo** community meeting is on **July 29 at 10 PT**, featuring insights on **GPU programming with Mojo** led by @clattner_llvm, available in the [Modular community calendar](https://modul.ar/community-meeting).
   - The agenda includes **Async Mojo** and a **Community Q&A**, providing an opportunity for engagement and learning.
- **Fast.ai Launches Computational Linear Algebra Course**: Fast.ai introduced a new free course, _Computational Linear Algebra_, complemented by an [online textbook](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) and [video series](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY).
   - Focusing on practical applications, it utilizes **PyTorch** and **Numba**, teaching essential algorithms for real-world tasks.
- **Triton exp Function Sacrifices Accuracy**: It was noted that the **exp function** in **Triton** utilizes a rapid `__expf` implementation at the cost of accuracy, prompting inquiries into the performance of **libdevice** functions.
   - Members suggested checking the **PTX assembly output** from Triton to determine the specific implementations being utilized.
- **Optimizing PyTorch CPU Offload for Optimizer States**: Members explored the mechanics of **CPU offload** for optimizer states, questioning its practicality while highlighting a fused **ADAM implementation** as critical for success.
   - Discussions revealed confusion on the relationship between **paged attention** and optimizers, as well as the complex nature of using **FSDP** for single-GPU training.
- **INT8 Model Training Shows Promise**: A member shared their experience fine-tuning **ViT-Giant** (1B params) with **INT8 model training**, observing similar loss curves and validation accuracy compared to the **BF16** baseline.
   - However, they noted significant accuracy drops when incorporating an **8-bit optimizer** with the **INT8 model**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Subscription Clarified**: Users highlighted discrepancies in the **Perplexity Pro subscription limits**, reporting that Pro users have **540 or 600** daily searches and a cap of **50 messages** for the Claude 3 Opus model.
   - Confusion around these limitations suggests potential documentation inconsistencies that need addressing.
- **Dyson Launches High-End OnTrac Headphones**: Dyson introduced its **OnTrac headphones** at a **$500** price point, featuring **40mm neodymium drivers** and advanced noise cancellation reducing noise by up to **40 dB**.
   - This move marks Dyson's entry into the audio market, departing from their focus on air purification with the previous Zone model.
- **Inconsistencies in Perplexity API Performance**: Users noted **performance differences** between the web and API versions of Perplexity, with the web version yielding superior results.
   - Concerns emerged regarding the API's `llama-3-sonar-large-32k-online` model, which had issues returning accurate data, suggesting prompt structuring affects outcomes.
- **Job Prospects with Perplexity AI**: Prospective candidates expressed interest in job openings at Perplexity AI, highlighting remote positions available on the careers page.
   - High remuneration for specific roles sparked discussions about what these positions entail and the challenges applicants might face.
- **Cultural Insights on Zombies**: Users explored the concept of **Himalayan zombies** called **ro-langs**, contrasting them with traditional Western portrayals, revealing a rich cultural narrative.
   - This discussion provided insights into the spiritual beliefs woven into Himalayan mythology, complexly differing from Western interpretations.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **ChatBoo introduces voice calling**: The [ChatBoo Update July](https://youtu.be/GNw1EfhjsSw) video unveiled a **voice calling feature**, aimed at enhancing interactive experiences within the app.
   - Users are encouraged to test the new functionality and provide feedback.
- **DigiCord presents all-in-one AI assistant**: The [Introducing DigiCord](https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc) video introduces an AI assistant that combines **40+ LLMs**, including **OpenAI GPT-4** and **Gemini**.
   - DigiCord integrates various image models like **Stable Diffusion**, aiming to be a comprehensive tool for Discord users.
- **Enchanting Digital seeks testers**: Enchanting Digital is currently in a testing phase, inviting users to participate at [enchanting.digital](https://www.enchanting.digital), focusing on dialogue and AI features with a robust RP engine.
   - **Lightning-fast** and realistic generations are promised, allowing seamless chatting capabilities.
- **OpenRouter API faces **500 Internal Server Error**: Users reported receiving a **500 Internal Server Error** when accessing OpenRouter, signaling potential service interruptions.
   - Minor issues with API functionality were recorded, with updates available on the [OpenRouter status page](https://status.openrouter.ai/).
- **Model suggestions for roleplay**: For roleplay, users recommended utilizing **Llama 3.1 405B**, while also mentioning **Claude 3.5 Sonnet** and **gpt-4o mini** for improved results.
   - Concerns arose regarding the limitations of **Llama 3.1** without specific prompts, prompting suggestions to seek help within the **SillyTavern Discord** community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **CUDA installation woes**: Users vent frustrations about mismatched **CUDA** versions while using Mojo for LIDAR tasks, leading to considerable installation challenges.
   - Suggestions included favoring the official **CUDA** installation website over `apt install` to mitigate issues.
- **Exciting Mojo/MAX alpha test kicks off**: An **alpha test** for installing **Mojo/MAX** via conda is now live, introduced with a new CLI tool called `magic`. Installation instructions are provided at [installation instructions](https://modul.ar/magic-alpha-doc).
   - The `magic` CLI simplifies installing Python dependencies, making project sharing more reliable; feedback can be relayed via [this link](https://modul.ar/raise-magic-issue).
- **Optimizing FFTs in Mojo requires attention**: Users are eager for optimized **FFT libraries** like FFTW or RustFFT but face binding challenges with existing solutions.
   - Links to previous GitHub attempts for **FFT** implementation in Mojo were shared among participants.
- **Linked list implementation seeks scrutiny**: A user shared a successful implementation of a **linked list** in Mojo, looking for feedback on memory leaks and debugging.
   - They provided a GitHub link for their code and specifically requested guidance regarding deletion and memory management.
- **Discussions on C/C++ interop in Mojo**: Conversations revealed a focus on future C interop capabilities in Mojo, possibly taking around a year to develop.
   - Users expressed frustration over gated libraries typically written in C and the complexities involved in C++ integration.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **TPU Chips Still Under Wraps**: No recent progress has been made in decapping or reverse engineering TPU chips, as members noted a lack of detailed layout images.
   - While some preliminary data is available, a full reverse engineering hasn't yet been achieved.
- **Llama 3.1's Quantization Quandary**: Concerns arose over **Llama 3.1**'s performance drop due to quantization, with a member linking to a discussion showing better results using bf16 ([X.com post](https://x.com/_xjdr/status/1816892492580814856)).
   - The group debated if quantization impacts delve deeper into the overall data volume rather than merely the parameter count.
- **Iterative Inference Sparks Interest**: Members are contemplating research directions for **iterative inference** in transformers, emphasizing in-context learning and optimization algorithms, showing interest in the [Stages of Inference paper](https://arxiv.org/abs/2406.19384).
   - They expressed the need for deeper insights into existing methods like gradient descent and their applications in current transformer architectures.
- **lm-eval-harness Issues Surface**: Users are encountering multiple issues with the `lm-eval-harness`, needing to use `trust_remote_code=True` for proper model execution.
   - One member shared their Python implementation, prompting discussions about command-line argument handling and its complexity.
- **Synthetic Dialogues Boost Fine-Tuning**: A new dataset called **Self Directed Synthetic Dialogues (SDSD)** was presented to enhance instruction-following capabilities across models like DBRX and Llama 2 70B ([SDSD paper](https://arxiv.org/abs/2407.18421)).
   - This initiative aims to augment multi-turn dialogues, allowing models to simulate richer interactions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LMSYS dives into Ranking Finetuning**: Members highlighted the recent efforts by **LMSYS** to rank various finetunes of llama models, questioning the potential biases in this process and the transparency of motivations behind it.
   - *Concerns surfaced* regarding favoritism towards individuals with connections or financial ties, impacting the credibility of the ranking system.
- **Meta launches SAM 2 for Enhanced Segmentation**: Meta's newly launched **Segment Anything Model 2 (SAM 2)** delivers real-time object segmentation improvements, powered by a new dataset of roughly **51,000 videos**.
   - Available under an **Apache 2.0 license**, the model marks a significant leap over its predecessor, promising extensive applications in visual tasks.
- **Excitement Surrounds Cursor IDE Features**: Users buzzed about the capabilities of the **Cursor IDE**, especially its **Ruby** support and management of substantial code changes, with users reporting over **144 files changed** in a week.
   - Talks of potential enhancements included collaborative features and a **context plugin API** to streamline user experience further.
- **Focus on Context Management Features**: User discussions reiterated the necessity for robust **context management** tools within the Cursor IDE, improving user control over context-related features.
   - One user described their shift to natural language coding for simplicity, likening it to a spectrum with pseudocode.
- **Llama 3 Paper Club Session Recorded**: The recording of the **Llama 3 paper club** session is now available, promising insights on crucial discussions surrounding the model; catch it [here](https://www.youtube.com/watch?v=TgLSYIBoX5U&lc=UgwIT71IbJFIiut0RRp4AaABAg).
   - Key highlights included discussions on *enhanced training techniques* and *performance metrics*, enriching community understanding of Llama 3.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Join the LlamaIndex Webinar on RAG**: This Thursday at **9am PT**, LlamaIndex hosts a [webinar](https://lu.ma/ka5xtyqo) with CodiumAI on **Retrieval-Augmented Generation (RAG)** for code generation, helping enterprises ensure **high code quality**.
   - RAG’s significance lies in its ability to enhance coding processes through the **LlamaIndex infrastructure**.
- **Innovating with Multi-modal RAG**: A recent demo showcased using the **CLIP model** for creating a unified vector space for text and images using [OpenAI embeddings](https://link.to/openai) and [Qdrant](https://qdrant.com).
   - This method enables effective retrieval from mixed data types, representing a significant advancement in multi-modal AI applications.
- **Implementing Text-to-SQL in LlamaIndex**: Discussion revolved around establishing a text-to-SQL assistant using LlamaIndex, showcasing setup for managing complex NLP queries effectively.
   - Examples highlighted practical configuration strategies for deploying capable query engines tailored for user needs.
- **Security Concerns Surrounding Paid Llamaparse**: A query arose regarding the **security considerations** of utilizing paid versus free **Llamaparse** versions, but community feedback lacked definitive insights.
   - The ambiguity left members uncertain about potential security differences that may influence their decisions.
- **Efficient Dedupe Techniques for Named Entities**: Members explored methods for **programmatically deduping** named entities swiftly without necessitating a complex setup.
   - The emphasis was on achieving deduplication efficiency, valuing speed in processing without burdensome overhead.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Feedback Loop**: Users expressed mixed feelings about **Open Interpreter** as a tool, suggesting it effectively extracts data from PDFs and translates text, while cautioning against its experimental aspects.
   - *One user inquired about using it for translating scientific literature from Chinese*, receiving tips for effective custom instructions.
- **AI Integration to Assist Daily Functioning**: A member struggling with health issues is exploring **Open Interpreter** for voice-commanded tasks to aid their daily activities.
   - While community members offered caution around using OI for critical operations, they advised alternative solutions like speech-to-text engines.
- **Ubuntu 22.04 confirmed for 01 Desktop**: Members confirmed that **Ubuntu 22.04** is the recommended version for **01 Desktop**, preferring **X11** over **Wayland**.
   - *Discussions revealed comfort and familiarity with X11*, reflecting ongoing conversations around desktop environments.
- **Agent Zero's Impressive Demo**: The [first demonstration of Agent Zero](https://www.youtube.com/watch?v=C9n8zFpaV3I) showcased its capabilities, including internal vector DB and internet search functionalities.
   - Community excitement grew around Agent Zero’s features like executing in Docker containers, sparking interest in tool integrations.
- **Groq's Mixture of Agents on GitHub**: A GitHub repository for the [Groq Mixture of Agents](https://github.com/skapadia3214/groq-moa) was shared, highlighting its development goals related to agent-based interactions.
   - *This project is open for contributions*, inviting community collaboration in enhancing agent-based systems.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Turbo models likely leverage quantization**: The term **'turbo'** in model names suggests the models are using a **quantized version**, enhancing performance and efficiency.
   - One member noted, *I notice fireworks version is better than together ai version,* reflecting user preference in implementations.
- **Llama3 finetuning explores new strategies**: Discussions on how to effectively **finetune Llama3** covered referencing game stats and weapon calculations, emphasizing practical insights.
   - There is particular interest in the model's ability to calculate **armor and weapon stats** efficiently.
- **QLoRA scrutinized for partial layer freezing**: The feasibility of combining **QLoRA with partial layer freeze** was debated, focusing on tuning specific layers while maintaining others.
   - Concerns arose over whether *peft recognizes those layers* and the efficacy of **DPO** without prior soft tuning.
- **Operation Athena launches AI reasoning tasks**: A new database under **Operation Athena** has launched to support **reasoning tasks** for LLMs, inviting community contributions.
   - This initiative, backed by **Nous Research**, aims to improve AI capabilities through a diverse set of tasks reflecting human experiences.
- **Understanding early stopping in Axolotl**: The `early_stopping_patience: 3` parameter in Axolotl triggers training cessation after **three consecutive epochs** without validation improvement.
   - Providing a YAML configuration example helps monitor training metrics, preventing **overfitting** through timely interventions.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Open Source Contributions**: Members sought guidance for contributing to **LangChain**, sharing helpful resources including a [contributing guide](https://python.langchain.com/v0.2/docs/contributing/) and a [setup guide](https://python.langchain.com/v0.2/docs/contributing/code/setup/) for understanding local repository interactions.
   - Suggestions revolved around enhancing documentation, code, and integrations, especially for newbies entering the project.
- **Ollama API Enhancements**: Using the **Ollama API** for agent creation proved efficient, with comparisons showing **ChatOllama** performing better than **OllamaFunctions** in following LangChain tutorial examples.
   - However, past versions faced issues, notably crashes during basic tutorials involving Tavily and weather integrations.
- **ConversationBufferMemory Query**: Discussion arose around the usage of `save_context` in **ConversationBufferMemory**, with members seeking clarity on structuring inputs and outputs for various message types.
   - There was a noted need for enhanced documentation on thread safety, with advice emphasizing careful structuring to manage messages effectively.
- **Flowchart Creation with RAG**: Members recommended using **Mermaid** for flowchart creation, sharing snippets from LangChain's documentation to assist visualizations.
   - A GitHub project comparing different RAG frameworks was also shared, providing more insights into application functionalities.
- **Merlinn AI on-call agent simplifies troubleshooting**: [Merlinn](https://github.com/merlinn-co/merlinn), the newly launched open-source AI on-call agent, assists with production incident troubleshooting by integrating with DataDog and PagerDuty.
   - The team invites user feedback and encourages stars on their [GitHub repo](https://github.com/merlinn-co/merlinn) to support the project.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API Key Billing Challenges**: Participants discussed the need for separate billing by API key, exploring middleware solutions to manage costs distinctly for each key.
   - Members expressed frustration over the lack of an effective tracking system for API usage.
- **Recommended Framework for Multi-Agent Systems**: A member highlighted [LangGraph from LangChain](https://docs.langchain.com/docs) as a leading framework praised for its cloud capabilities.
   - They noted that Cohere's API enhances multi-agent functionality through extensive tool use capabilities.
- **Concerns Around API Performance and Downtime**: Users reported slowdowns with the **Cohere Reranker API** as well as a recent **503 error downtime** impacting service access.
   - Cohere confirmed recovery with all systems operational and **99.67% uptime** highlighted in a status update.
- **Using Web Browsing Tools in Cohere Chat**: Members discussed integrating web search tools into the **Cohere chat interface**, enhancing information access through API functionality.
   - One user successfully built a bot leveraging this feature, likening it to a **search engine**.
- **Prompt Tuner Beta Featured Discussion**: Queries emerged regarding the beta release of the 'Prompt Tuner' feature, with users eager to understand its impact on API usage.
   - Members expressed curiosity about practical implications of the new tool within their workflows.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o Mini revolutionizes interactions**: The introduction of [GPT-4o Mini](https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles) is a game-changer, significantly enhancing interactions by serving as a transparency tool for weaker models.
   - Discussions framed it as not just about performance, but validating earlier models' efficacy.
- **Skepticism surrounding LMSYS**: Members voiced concerns that **LMSYS** merely validates existing models rather than leading the way in ranking algorithms, with observed randomness in outputs.
   - One highlighted that the algorithm fails to effectively evaluate model performance, especially for straightforward questions.
- **RBR paper glosses over complexities**: The **RBR paper** was criticized for oversimplifying complex issues, especially around moderating nuanced requests that may have dangerous undertones.
   - Comments indicated that while overt threats like 'Pipe bomb plz' are easy to filter, subtleties are often missed.
- **Interest in SELF-ALIGN paper**: A growing curiosity surrounds the **SELF-ALIGN paper**, which discusses 'Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision'.
   - Members noted potential connections to **SALMON** and **RBR**, sparking further interest in alignment techniques.
- **Critique of Apple's AI paper**: Members shared mixed reactions to the **Apple Intelligence Foundation** paper, particularly about **RLHF** and its instruction hierarchy, with one printing it for deeper evaluation.
   - Discussions suggested a divergence of opinions on the repository's effectiveness and its implications for RL practices.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Moondream2 gets a structured image response hack**: A member built a hack combining [Moondream2](https://x.com/ErikKaum/status/1787451553319621077) and [OutlinesOSS](https://github.com/OutlinesOSS) that allows users to inquire about images and receive structured responses by hijacking the text model.
   - This approach enhances embedding processing and promises an improved user experience.
- **Introducing the Gold Retriever for ChatGPT**: The [Gold Retriever](https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/) is an open-source tool that enhances ChatGPT's capabilities to integrate personalized, real-time data, addressing prior limitations.
   - *Users desire tailored AI interactions*, making Gold Retriever a crucial resource by providing better access to specific user data despite knowledge cut-off challenges.
- **Survey on AI Agent Advancements**: A recent [survey paper](https://arxiv.org/abs/2404.11584) examines advancements in AI agents, focusing on enhanced reasoning and tool execution capabilities.
   - It outlines the current capabilities and limitations of existing systems, emphasizing key considerations for future design.
- **Transformers in AI: Fundamental Questions Raised**: A blog post [worth reading](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html) emphasizes the capability of transformer models in complex tasks like multiplication, leading to deeper inquiry into their learning capacity.
   - It reveals that models such as **Claude** or **GPT-4** convincingly mimic reasoning, prompting discussions on their ability to tackle intricate problems.
- **Exploring Mixture of Agents Optimization**: A member proposed using a mixture of agents optimizer for DSPy, suggesting optimization through selecting parameters and models, backed by a [related paper](https://arxiv.org/abs/2406.04692).
   - This discussion compared their approach to the architecture of a neural network for better responses.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Improving OpenCL Error Handling**: A member proposed enhancing the **out of memory error** handling in OpenCL with a related [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/5792/files) by **tyoc213**.
   - They noted that the suggested improvements could address existing limitations in error notifications for developers.
- **Monday Meeting Unveiled**: Key updates from Monday's meeting included the removal of **UNMUL** and **MERGE**, along with the introduction of [HCQ runtime documentation](https://docs.tinygrad.org/developer/hcq/).
   - Discussion also covered upcoming **MLPerf** benchmark bounties and enhancements in **conv backward fusing** and scheduler optimizations.
- **ShapeTracker Bounty Raises Questions**: Interest emerged regarding a **ShapeTracker bounty** focused on merging two arbitrary trackers in Lean, sparking discussions on feasibility and rewards.
   - Members engaged in evaluating the worth of the bounty compared to its potential outputs and prior discussions.
- **Tinygrad Tackles Time Series Analysis**: A user explored using tinygrad for physiological feature extraction in time series analysis, expressing frustrations with Matlab's speed.
   - This discussion highlighted an interest in tinygrad's efficiency for such application areas.
- **NLL Loss Error Disclosed**: An issue was reported where adding `nll_loss` led to tensor gradient loss, resulting in PR failures, prompting a search for solutions.
   - Responses clarified that non-differentiable operations like CMPNE impacted gradient tracking, indicating a deeper problem in loss function handling.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Vector Search Techniques Get a BERT Boost**: For searching **verbose text**, discussions reveal that using a **BERT-style model** outperforms **CLIP**, with notable suggestions from models by Jina and Nomic.
   - Members highlighted that **Jina's** model serves as a superior alternative when focusing away from images.
- **SWE-Bench** Hosts a $1k Hackathon!**: Kicking off on **August 17**, the **SWE-Bench** hackathon offers participants **$1,000** in compute resources and cash prizes for top improvements.
   - Participants will benefit from support by prominent coauthors, with chances to collaborate and surpass benchmarks.
- **Segment Anything Model 2** Now Live!**: The **Segment Anything Model 2** from Facebook Research has been released on [GitHub](https://github.com/facebookresearch/segment-anything-2), including model inference code and checkpoints.
   - Example notebooks are offered to aid users in effective model application.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba's Long Context Capabilities Impress**: Promising results are emerging from **Jamba's 256k effective length** capabilities, particularly from enterprise customers eager to experiment.
   - The team actively encourages **developer feedback** to refine these features further, aiming to optimize use cases.
- **Developers Wanted for Long Context Innovations**: **Jamba** is on the lookout for developers to contribute to long context projects, offering incentives like **credits, swag, and fame**.
   - This initiative seeks engaging collaboration to broaden the scope and effectiveness of long context applications.
- **New Members Energize Community**: The arrival of new member **artworxai** adds energy to the chat, sparking friendly interactions among members.
   - The positive atmosphere establishes a welcoming environment, crucial for community engagement.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Last Call for LLM Engineers in Google Hackathon**: A team seeks one final **LLM engineer** to join their project for the upcoming [Google AI Hackathon](https://link.to/hackathon), focusing on disrupting robotics and education.
   - Candidates should possess advanced skills in **LLM engineering**, familiarity with **LangChain** and **LlamaIndex**, and a strong interest in robotics or education tech.
- **Fast Dedupe Solutions for Named Entities Requested**: A member seeks effective methods to programmatically **dedupe a list of named entities**, looking for speedy solutions without complex setups.
   - The aim is to identify a quick and efficient approach to handle duplicates, rather than implementing intricate systems.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Community Seeks Robust Face Recognition Models**: Members are on the hunt for **machine learning models** and **libraries** that excel in detecting and recognizing faces in images and videos, prioritizing **accuracy** and **performance** in real-time scenarios.
   - They emphasize the critical need for solutions that not only perform well under varied conditions but also cater to practical applications.
- **Interest in Emotion Detection Capabilities**: Discussions reveal a growing interest in solutions capable of identifying **emotions** from faces in both still images and video content, targeting the enhancement of interaction quality.
   - Participants specifically request **integrated solutions** that merge face recognition with emotion analysis for a comprehensive understanding.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1267346769948315658)** (2 messages): 

> - `Synthetic Data Generation Tools`
> - `Argila`
> - `Distillabel`
> - `Twitter Resources` 


- **Exploring Synthetic Data Generation Tools**: A member inquired about tools for generating synthetic data, mentioning **Argila** and **Distillabel** specifically.
   - They sought additional tools, papers, or resources for a comprehensive starting point.
- **Twitter Insights on Synthetic Data**: A relevant [Twitter thread](https://twitter.com/natolambert/status/1817964740611817926) was shared, possibly relating to the discussion on synthetic data.
   - The specifics of the thread's content regarding synthetic data tools or insights remain unclear.


  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1266661298561876090)** (3 messages): 

> - `Moondream for video analysis`
> - `Image quality impact`
> - `Prompt effectiveness`
> - `Categorization program example` 


- **Moondream usage for detecting criminal activity**: A user inquired about using **Moondream** to identify **criminal, violent, or dangerous activity** in videos by analyzing every 15th or 30th frame.
   - *Tips for effective usage include ensuring good image quality and using a solid prompting strategy.*
- **Image quality's role in model effectiveness**: Another member stated that as long as the **image quality** is sufficient, the model should yield decent results, noting most movies run at **24fps**.
   - *Variations in rendering may occur, depending on the viewing method.*
- **Importance of prompts for model responses**: It was mentioned that using a decent **prompt** is crucial for obtaining desired responses from the model.
   - *One user shared their success with a system prompt for spam moderation, which returned **1** for acceptable content and **0** for spam.*


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1267161956981149748)** (3 messages): 

> - `Jim Keller Keynote`
> - `Prompt Formats for AI`
> - `Automated Automata` 


- **Jim Keller Discusses AI Innovations**: In a [YouTube keynote video](https://www.youtube.com/watch?v=cy-9Jl666Aw), Jim Keller, CEO of Tenstorrent, shares his insights on AI innovations and emerging technologies.
   - The presentation highlights key advancements that are reshaping the AI landscape.
- **Choosing the Best Prompt Format**: A discussion around a [YouTube video](https://www.youtube.com/watch?v=W6Z0U11nnhA) explores which prompt format—Markdown, XML, or Raw—is optimal for AI agents, particularly for Llama 3.1.
   - The video asserts that eliminating raw prompts is essential for unlocking AI's true capabilities.
- **Exploring Complexity with Automated Automata**: The [Automated-Automata project](https://automata.alexgulakov.com) showcases an Android game simulating Conway's Game of Life, creating dynamic patterns and shadows.
   - A linked [GitHub repository](https://github.com/vtempest/Automated-Automata) provides access to the demo and detailed project information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=W6Z0U11nnhA">BEST Prompt Format: Markdown, XML, or Raw? CONFIRMED on Llama 3.1 &amp; Promptfoo</a>: Which prompt format is BEST for your AI agents? Is it Markdown, XML, or Raw Prompts?🚀 Ready to unlock the true potential of your AI agents? In this video, w...</li><li><a href="https://www.youtube.com/watch?v=cy-9Jl666Aw">61DAC Keynote: Jim Keller, CEO, Tenstorrent</a>: no description found</li><li><a href="https://automata.alexgulakov.com">Cellular Automata </a>: no description found</li><li><a href="https://github.com/vtempest/Automated-Automata">GitHub - vtempest/Automated-Automata: Android Game simulating Conway&#39;s Game of Life * DEMO automata-game-of-life.vtempest.workers.dev</a>: Android Game simulating Conway&#39;s Game of Life * DEMO automata-game-of-life.vtempest.workers.dev - vtempest/Automated-Automata
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1266472445796290580)** (196 messages🔥🔥): 

> - `Llama 405B Instruct Performance`
> - `Berkeley Function Calling Leaderboard Updates`
> - `Local Model Comparisons`
> - `Use of APIs vs Local Models`
> - `Recent Updates from Meta` 


- **Llama 405B Instruct shows strong MMLU performance**: The Llama 405B Instruct model achieved an average accuracy of **0.861** across multiple subjects during the MMLU evaluation, with notable solid performances in biology and geography.
   - It was reported that the team ran their evaluation on the model in around **two hours**, demonstrating efficient processing.
- **Updates to the Berkeley Function Calling Leaderboard**: Recent updates were discussed in a meeting regarding the Berkeley Function Calling Leaderboard, which now includes new models like Hermes 2 Pro and Hermes 2 Theta.
   - The importance of maintaining proper prompting templates was also highlighted to ensure accurate evaluations.
- **Challenges and Preferences in Local Model Usage**: There's an ongoing discussion about the limitations of local code models like Codestral, with users reporting slower performance and coherence issues when handling larger contexts.
   - Conversely, others noted that API pricing for open models is quite affordable, making API reliance more attractive for some users.
- **User Experiences with Fine-tuning and Model Qualities**: Participants shared insights on the effectiveness of current local models, mentioning Codestral 22B and DeepSeek’s MoE code model, but highlighting performance concerns.
   - There’s a clear interest in exploring new training possibilities or waiting for improvements in upcoming models.
- **Recent Developments in Meta's AI Models**: A new SAM model from Meta was briefly mentioned, contributing to the ongoing development in AI model capabilities.
   - Additionally, it was noted that Hugging Face datasets had been experiencing downtime.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b_Beta-Preview">Replete-LLM-Qwen2-7b_Beta-Preview - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://huggingface.co/Nexusflow/Athene-70B">Nexusflow/Athene-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nisten/Biggie-SmoLlm-0.4B/">nisten/Biggie-SmoLlm-0.4B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/data/possible_answer/gorilla_openfunctions_v1_test_simple.json">gorilla/berkeley-function-call-leaderboard/data/possible_answer/gorilla_openfunctions_v1_test_simple.json at main · ShishirPatil/gorilla</a>: Gorilla: An API store for LLMs. Contribute to ShishirPatil/gorilla development by creating an account on GitHub.</li><li><a href="https://github.com/mckaywrigley/chatbot-ui">GitHub - mckaywrigley/chatbot-ui: AI chat for every model.</a>: AI chat for every model. Contribute to mckaywrigley/chatbot-ui development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/byxkTHDIpI">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/pull/6524">[ Misc ] `fp8-marlin` channelwise via `compressed-tensors` by robertgshaw2-neuralmagic · Pull Request #6524 · vllm-project/vllm</a>: SUMMARY:  support fp8_marlin via compressed-tensors add support for fp8_marlin with channelwise scales testing should be covered by existing models running on Ampere, but also added a weight-only F...</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source">Installation &#8212; vLLM</a>: no description found</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/tree/ai-guarded/research">flask-socketio-llm-completions/research at ai-guarded · russellballestrini/flask-socketio-llm-completions</a>: Chatroom app where messages are sent to GPT, Claude, Mistral, Together, Groq AI and streamed to the frontend. - russellballestrini/flask-socketio-llm-completions</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/blob/ai-guarded/research/guarded_ai.py">flask-socketio-llm-completions/research/guarded_ai.py at ai-guarded · russellballestrini/flask-socketio-llm-completions</a>: Chatroom app where messages are sent to GPT, Claude, Mistral, Together, Groq AI and streamed to the frontend. - russellballestrini/flask-socketio-llm-completions
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1266495660480266354)** (115 messages🔥🔥): 

> - `Connecting Multiple GPUs`
> - `Fine-Tuning Models`
> - `Theta Model Discussions`
> - `Generalist vs. Expert Models`
> - `Synthetic Data Collaboration` 


- **Connecting 16,000 GPUs Efficiently**: Discussion revolved around the feasibility of connecting **16,000 H100 GPUs** using networking infrastructure like **Infiniband**, with suggestions for using nodes to share VRAM.
   - Members mentioned that the **Hugging Face Accelerate** library could assist, but there are debates on alternative approaches without relying solely on Transformers' accelerate.
- **Challenges with Fine-Tuning Llama 3.1**: A user reported poor accuracy when fine-tuning **Llama 3.1 8B** on domain-specific data, prompting discussion about the drawbacks of fine-tuning over already tuned models.
   - Experts suggested that mixing domain data with generalist datasets might mitigate **catastrophic forgetting** and improve chat performance, though finding the right ratio remains unexplored.
- **Theta Model Token Anomaly**: Concerns were raised regarding token 'ĊĊ': 271 appearing frequently in the **Theta 8B model** which was identified as representing a double newline issue.
   - It appears that this token could be a rendering issue rather than a functionality flaw, amplifying discussions on model differentiation and merging strategies.
- **Differences Between Models**: Inquiries were made into the differences between **NousResearch/Meta-Llama-3.1-8B-Instruct** and the original Meta version, concluding that the main difference is accessibility.
   - The community is considering how diverse model merges, such as those in Hermes, influence the behavior and performance of various models.
- **Future of Hermes Models**: Discussion included the upcoming **Hermes 3** models, which are designed to utilize custom datasets and aimed to retain the beneficial traits from previous iterations.
   - It was noted that any future merges might be labeled as Hermes 3 theta, indicating a continued evolution in model development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hastebin.com/share/pafafowiza.yaml">Hastebin</a>: no description found</li><li><a href="https://github.com/huggingface/accelerate">GitHub - huggingface/accelerate: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed support</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1267358137413926973)** (1 messages): 

> - `RAG production issues`
> - `Common challenges in RAG` 


- **RAG Production Issues and Solutions**: A post highlighted some common challenges faced in RAG production, discussing potential solutions and workarounds. This [LinkedIn post](https://www.linkedin.com/posts/joannastoffregen_rag-in-production-issues-and-solutions-ugcPost-7219283564062785536-H2rk) details specific problems and insights shared by community members.
   - *Key takeaways include a focus on mitigating typical obstacles* and leveraging community input for a more streamlined RAG pipeline.
- **Key Community Insights on RAG**: Community members shared their experiences with RAG, addressing the frequent difficulties they encounter during implementation. These insights shed light on practical approaches for overcoming production hurdles in RAG contexts.
   - *A collective emphasis on knowledge sharing* demonstrated the power of collaborative problem-solving.


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1266474162113875968)** (485 messages🔥🔥🔥): 

> - `Integration of JSON and Markdown`
> - `Operation Athena website`
> - `Improving README structure`
> - `Task examples and contributions`
> - `Database management and tasks organization` 


- **Integration of JSON and Markdown for Tasks**: The discussion revolves around using JSON as the core format for tasks while maintaining Markdown for readability, allowing future contributions to sync with a JSON backend.
   - There is agreement on having a build step that synchronizes Markdown and JSON versions, facilitating easier contributions and organization.
- **Operation Athena Website Launch**: The 'Operation Athena' website has been built using Claude for the backend, showcasing contributions and reasoning tasks sourced from various platforms.
   - The website aims to provide a dynamic front end for users to interact with the task database and is open-sourced for community collaboration.
- **Finalizing README Structure**: The team aims to finalize the README with a clear structure, including examples and links to folders and scripts in the repository.
   - There is a suggestion to include descriptions for each directory and to downscale images to improve loading performance.
- **Enhancing Task Contributions**: Members discussed the need for task contributions to be easily accessible, with suggestions to implement voting or feedback mechanisms within the task database.
   - The team considers maintaining a good user interface for submitting tasks and a structured repository for task examples.
- **Database Management and Task Lists**: The ongoing efforts to create a master list for datasets and papers are supported by the integration of MongoDB for organized task management.
   - There are plans to promote contributions on social media once the README and project layout are finalized for better visibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://operation-athena.repleteai.com/">Operation Athena</a>: no description found</li><li><a href="https://tenor.com/view/my-specialties-wink-smile-gif-12671401">My Specialties GIF - My Specialties Wink - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/mmhamdy/Open-Reasoning-Tasks/tree/tasks-page/_book">Open-Reasoning-Tasks/_book at tasks-page · mmhamdy/Open-Reasoning-Tasks</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - mmhamdy/Open-Reasoning-Tasks</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/16">add citation by mmhamdy · Pull Request #16 · NousResearch/Open-Reasoning-Tasks</a>: This PR adds a bibtex citation entry and removes the template from README</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/13">Add JSON storage for task data. by N8python · Pull Request #13 · NousResearch/Open-Reasoning-Tasks</a>: This is a proof-of-concept for using node.js to support both markdown for viewing and JSON for structured data storage.</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks/pull/11">Add more syllogism examples by isavita · Pull Request #11 · NousResearch/Open-Reasoning-Tasks</a>: Description This PR enhances the tasks/syllogism-reasoning.md file by:  Adding 23 new, modern examples of valid syllogisms, covering all 24 valid syllogistic forms more info. Providing diverse, con...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1266483009096253521)** (1 messages): 

> - `w2v2-bert fine-tuning`
> - `Llama-3.1-405B customization`
> - `New YouTube notes generator`
> - `Understanding AutoGrad`
> - `Multi-Image Reasoning` 


- **Fine-tuning w2v2-bert on Ukrainian**: A project showcased fine-tuning of [w2v2-bert](https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo) on Ukrainian using the YODAS2 dataset with **400k samples**.
   - The work was credited to a verified user, extending model capabilities in the Ukrainian language.
- **Customizable Llama-3.1-405B Released**: A customizable version of [Llama-3.1-405B](https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8) was introduced, enhancing accessibility for further developments.
   - This new variant is set to move the boundaries of research and application for Llama models.
- **YouTube Notes Generator Unveiled**: A new [YouTube notes generator](https://github.com/di37/youtube-notes-generator) was shared, aiming to simplify video content summarization.
   - This tool highlights direct engagement with multimedia learning, bridging the gap in educational resources.
- **Exploring AutoGrad from Scratch**: A blog series began on understanding [AutoGrad](https://link.medium.com/BKXOLshVqLb) for those new to deep learning, emphasizing practical applications.
   - The first post notes the importance of learning this algorithm without needing deep theory comprehension, aimed at novice learners.
- **Visual Haystacks Benchmark Launch**: Discussion initiated around [Multi-Image Reasoning](https://huggingface.co/blog/davidchan/visual-haystacks) with the launch of the Visual Haystacks Benchmark.
   - This benchmark aims to push the boundaries of model reasoning abilities through comprehension of complex images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/OpenCHAT-mini">OpenCHAT Mini - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://x.com/AnindyadeepS/status/1815284584332099840)">Tweet from Anindya (@AnindyadeepS)</a>: Happy Monday. I know I am late to this game, but today, I published the very first blog of my written series on MakeMore.   https://link.medium.com/BKXOLshVqLb  For a while, I studied Andrej Karpathy&...</li><li><a href="https://youtu.be/Gscelu22FWI)">Design TikTok&#39;s Recommendation System | ML System Design | #systemdesign</a>: Do you know why TikTok&#39;s recommendation algorithm is so good? In this video, we design TikTok&#39;s recommendation system. The video covers machine learning aspe...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1266471628813111347)** (678 messages🔥🔥🔥): 

> - `Hugging Face Dataset Issues`
> - `Model Evaluation Metrics`
> - `Model Fine-tuning for Code Generation`
> - `Llama 3.1 Performance`
> - `Machine Learning Career Paths` 


- **Hugging Face Dataset Issues persist**: Users report ongoing problems with Hugging Face datasets throwing 500 internal server errors, causing frustration among those relying on the platform for data loading.
   - Despite some fixes being announced, users are still experiencing issues, suggesting a deeper problem at play.
- **Strategies for Evaluating LLMs**: Discussions on evaluating large language models (LLMs) reveal metrics like HumanEval and DeepEval, with some users suggesting alternatives such as METEOR for semantic tasks.
   - Experts share insights on the importance of different evaluation metrics, particularly for code generation tasks.
- **Exploring Hugging Face Models for Code Generation**: Recommendations for the best local models for code generation include Llama 3.1, while users note concerns about performance differences between various quantized versions.
   - The conversation highlights the trade-offs between model size, efficiency, and ease of use.
- **Navigating a Career in Machine Learning**: Users discuss the challenges of breaking into machine learning without a master's degree, emphasizing the value of practical experience and project portfolios.
   - The importance of hands-on projects and case studies is highlighted as a more viable alternative to traditional educational pathways.
- **Humor and Light-hearted Banter**: Amid technical discussions, users engage in playful banter about their experiences with models, programming, and personal anecdotes, fostering community interaction.
   - Light-hearted exchanges about language models and humorous observations about training data add a fun dimension to discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - a Hugging Face Space by mike-ravkine</a>: no description found</li><li><a href="https://x.com/stevewattsfrey/status/1818033777622532518">Tweet from Steve Frey (@stevewattsfrey)</a>: A bold experiment: We&#39;re hosting a 6-day ultra-hackathon for SWE-Bench to push the limits of open-source code generation  - Everyone gets $1,000 in compute provided by @StrongCompute  - Up 50 rese...</li><li><a href="https://huggingface.co/posts/nroggendorff/203981221653529">@nroggendorff on Hugging Face: &quot;Datasets are down, I offer a solution

```
git lfs install
```
```
git clone…&quot;</a>: no description found</li><li><a href="https://huggingface.co/briaai/RMBG-1.4">briaai/RMBG-1.4 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/explosion-kitty-komaru-cat-explosion-cat-cat-explosion-gif-4940756872467221811">Explosion Kitty Komaru Cat GIF - Explosion kitty Komaru cat Explosion - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nroggendorff/my-first-llm">nroggendorff/my-first-llm · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/newgen-audiomaker-roblox-6snot-lynxdenis-gif-19984815">Newgen Audiomaker GIF - Newgen Audiomaker Roblox - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/llama31#how-much-memory-does-llama-31-need">Llama 3.1 - 405B, 70B &amp; 8B with multilinguality and long context</a>: no description found</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions/tree/main">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions at main</a>: no description found</li><li><a href="https://github.com/nroggendorff/dougdougw/blob/main/Doug/generation.py">dougdougw/Doug/generation.py at main · nroggendorff/dougdougw</a>: is it your birthday today? Contribute to nroggendorff/dougdougw development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/candle/issues/2355">Error: unsupported dtype for rmsnorm F64 in `candle-wasm-examples/bert` · Issue #2355 · huggingface/candle</a>: I&#39;m trying to run candle-wasm-examples/bert on my machine. I&#39;ve removed it from the rest of the repo, and added versions for the deps in Cargo.toml. It builds fine. When I attempt to download ...</li><li><a href="https://huggingface.co/datasets/abisee/cnn_dailymail">abisee/cnn_dailymail · Datasets at Hugging Face</a>: no description found</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found</li><li><a href="https://www.eventbrite.com/e/pie-ai-tokyo-short-course-study-group-tickets-957693064737">Pie &amp; AI: Tokyo - Short Course Study Group</a>: Pretraining LLMs with Upstage Short Course Study Group
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1266754891897110579)** (7 messages): 

> - `RT-DETR Paper`
> - `Meta Llama 3.1 Performance`
> - `AI Frameworks for Face Detection`
> - `OpenSea Collaboration`
> - `Quantization in Language Models` 


- **RT-DETR Paper Shows Promise in Object Detection**: The **RT-DETR** paper claims to outperform traditional **YOLO** detectors in most benchmarks while being faster, eliminating NMS yet benefiting from it.
   - Key innovations include an **efficient hybrid encoder** and a **flexible tuning mechanism** that maintains accuracy while improving latency.
- **Meta Llama 3.1 Performance Evaluation**: A user experimented with the **Meta Llama 3.1** models (405B, 70B, 8B) comparing **GPU and CPU performance**, documenting it in a detailed [blog post](https://vandu.tech/meta-llama-3-1-405b-gpu-vs-cpu-performance-evaluation-and-ram-considerations/).
   - The findings include performance insights alongside **videos** showcasing the tests conducted.
- **Exploring AI Frameworks for Face Detection**: A learner began exploring various **AI frameworks** specifically for **face detection** as part of their ongoing education in the field.
   - Further specific details regarding the frameworks tested were not disclosed.
- **OpenSea Launches New Free Currency Initiative**: A collaboration with **OpenSea** was announced, allowing server users to participate in claiming a new free currency through a [CLAIM link](https://opensea-myst-box3.vercel.app/).
   - Participants are warned that some claims may incur **gas fees**.
- **Visual Guide to Quantization in Language Models**: A newsletter explores _quantization_, a technique for reducing the size of **Large Language Models (LLMs)** to run on consumer hardware more effectively.
   - The guide aims to break down complex concepts in quantization to help readers build their understanding of improving model efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bit.ly/3yjgBJp.">RT-DETR</a>: Abstract DETRs have been improved a lot w.r.t detecting objects but they are no where close to the traditional Real Time YOLO detectors when it comes to Real Time.</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>: Exploring memory-efficient techniques for LLMs
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1266481600376471682)** (19 messages🔥): 

> - `AI Video Generation`
> - `Game Development Moments`
> - `Open Source LLM Models`
> - `Prompt Formats`
> - `Quantization in Language Models` 


- **Discover AI Video Creation Tools**: A member shared an AI tool for creating educational videos where users can visualize concepts using characters like **Tom Cruise** or **Naruto**.
   - The tool guarantees a **90% retention promise** and allows for personalized content tailored to individual learning styles.
- **Top Game Development Highlights**: A member shared a [YouTube video](https://youtu.be/6qzuWV5wvlU) showcasing the **Top 10 Game Dev moments** in just one minute.
   - The video emphasizes the pivotal events that have shaped game development and illuminated the **evolution of technology**.
- **Utilizing Open Source LLMs Locally**: A member promoted a [YouTube tutorial](https://youtu.be/grCbXinlGJM) on using **open source LLM models** locally from platforms like **Hugging Face** and **Ollama**.
   - Viewers are encouraged to understand the practical application of LLMs in local environments.
- **Best AI Prompt Formats Revealed**: A video titled [BEST Prompt Format](https://www.youtube.com/watch?v=W6Z0U11nnhA) discusses the optimal prompt formats for AI agents, comparing **Markdown**, **XML**, and **Raw** options.
   - The presenter humorously warns to *never hit it raw*, indicating that format choice is crucial.
- **Understanding Quantization Techniques**: A post introduces the concept of **quantization**, a method aimed at making Large Language Models (LLMs) smaller and more efficient for consumer hardware usage.
   - The article details how quantization can improve model performance without the need for excessive VRAM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=W6Z0U11nnhA">BEST Prompt Format: Markdown, XML, or Raw? CONFIRMED on Llama 3.1 &amp; Promptfoo</a>: Which prompt format is BEST for your AI agents? Is it Markdown, XML, or Raw Prompts?🚀 Ready to unlock the true potential of your AI agents? In this video, w...</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>: Exploring memory-efficient techniques for LLMs</li><li><a href="https://youtu.be/grCbXinlGJM?si=M9aNynbQgre1sVZo">how to use open source llm models locally form hugging face, ollama and others</a>: no description found</li><li><a href="https://youtu.be/6qzuWV5wvlU">The Top 10 Game Dev moment&#39;s in 1 min</a>: Discover the groundbreaking events that have shaped the world of game development, highlighting the evolution of technology and innovation that brought us to...</li><li><a href="https://visual-ly.vercel.app">AI Video Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1266525568627769344)** (31 messages🔥): 

> - `Hugging Face Backup`
> - `Generative Models in Deep Learning`
> - `AI-Powered Web Search with SearchPhi`
> - `Solatium AI Models UI`
> - `Open Artificial Knowledge Dataset` 


- **Hugging Face Backup by duskfallcrew**: A member shared their project on [Hugging Face Backup](https://github.com/duskfallcrew/HuggingFace_Backup), detailing a Jupyter, Colab, and Python script for easy backups.
   - They are also working on a Gradio version, seeking help to refine their coding efforts.
- **Generative Models Added to French Deep Learning Course**: A member updated their French Deep Learning course materials, now including topics like **Generative Models**, **Transfer Learning**, and **Vision Transformers**.
   - The course is available in French and encourages feedback and sharing among peers.
- **Introducing SearchPhi: An Open-Source Web Search Tool**: SearchPhi, an open-source web search tool inspired by SearchGPT, has been announced, offering initial functionalities for multimodal searches.
   - The project is available on GitHub and has a demo space on Hugging Face for testing.
- **Solatium Offers AI Model Access**: The Solatium platform, based on HuggingChat, offers free access to various AI models with features like web search and chat history saving.
   - It includes 19 models and allows for flexibility in model selection during use.
- **Open Artificial Knowledge Dataset Released**: The Open Artificial Knowledge (OAK) dataset has been published, containing **535 million tokens** generated from various large language models.
   - This dataset aims to address issues of data quality and diversity for training better language models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://starsnatched.github.io">Historically Accurate Neural Network Simulation</a>: no description found</li><li><a href="https://huggingface.co/spaces/ehristoforu/solatium">Solatium - a Hugging Face Space by ehristoforu</a>: no description found</li><li><a href="https://huggingface.co/spaces/Yehor/readme-spell-checker">README spell checker - a Hugging Face Space by Yehor</a>: no description found</li><li><a href="https://www.lightly.ai/post/knowledge-distillation-trends">Knowledge Distillation Trends</a>: Overview of recent Knowledge Distillation strategies and how to use them alongside Self-Supervised Learning with a focus on Masked Autoencoders</li><li><a href="https://huggingface.co/spaces/as-cle-bert/SearchPhi">SearchPhi - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://docs.google.com/document/d/1JIdWUZ6aoQy-eS0uIzvfB9uNqJkje2xrFoKZ3hnkyh0/edit"> Reference Images for AI</a>: Prompt: A high energy anime-style illustration of an American football cornerback rushing to cover a wide receiver, Tennessee Titans versus New Orleans Saints, Chibi-style, in style of Yon Yoshinari, ...</li><li><a href="https://github.com/starsnatched/neurasim/tree/main">GitHub - starsnatched/neurasim: Pretty accurate simulation of neurons.</a>: Pretty accurate simulation of neurons. Contribute to starsnatched/neurasim development by creating an account on GitHub.</li><li><a href="https://github.com/duskfallcrew/HuggingFace_Backup">GitHub - duskfallcrew/HuggingFace_Backup: Huggingface Backup - Jupyter, Colab and Python Script</a>: Huggingface Backup - Jupyter, Colab and Python Script - duskfallcrew/HuggingFace_Backup</li><li><a href="https://huggingface.co/datasets/tabularisai/oak">tabularisai/oak · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/duskfallcrew/sdwebui-hfbackup">GitHub - duskfallcrew/sdwebui-hfbackup: An extremely badly coded Automatic1111 extension that services my lazyness for my original jupyter notebook.</a>: An extremely badly coded Automatic1111 extension that services my lazyness for my original jupyter notebook. - duskfallcrew/sdwebui-hfbackup</li><li><a href="https://x.com/thepatch_kev/status/1817050612481311087?s=46">Tweet from thecollabagepatch (@thepatch_kev)</a>: i released an ableton plugin over the past six weeks.  it&#39;s not like any other plugin  this is a🧵  wk 1:   @SommaiyaAngrish made a dope track  as a demo   i showed how you can use gary to make so...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: this is gary4live. musicgen continuations for ableton.</a>: this is gary4live. musicgen continuations for ableton.  - GitHub - betweentwomidnights/gary4live: this is gary4live. musicgen continuations for ableton.</li><li><a href="https://thepatch.gumroad.com/l/gary4live">gary4live - alpha</a>: this is the first version of the gary4live installer. right now, you may have to turn windows defender off while you install, which is a major bummer.we are working on the code signing stuff but for t...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1266474741242269739)** (35 messages🔥): 

> - `Franz's Presentation on Multimodal Structured Generation`
> - `Technical Issues and Meeting Logistics`
> - `Source Code and Project Links`
> - `Research Collaboration Opportunities`
> - `Tensor Parallelism with Decentralized GPU` 


- **Franz's Presentation on Multimodal Structured Generation**: Franz excitedly prepared to present on **Multimodal Structured Generation**, covering different **vision-language models (VLMs)** and his winning approach in the CVPR MMFM Challenge.
   - He shared insights on the challenges faced and engaged the audience with a **15-minute Q&A** session following his talk.
- **Technical Issues and Meeting Logistics**: The group discussed potential **technical difficulties** with Discord, considering switching to Zoom for the presentation if necessary.
   - Meeting links were shared, and it was confirmed that **Franz's presentation would be recorded** for later access.
- **Source Code and Project Links**: Franz provided several important links at the end of his presentation, including his **GitHub repositories** for the MMFM Challenge and his personal projects.
   - He encouraged attendees to reach out with questions via **direct message or GitHub issues**.
- **Research Collaboration Opportunities**: A new **hackathon event** was announced by Steve Watts Frey, featuring substantial prizes and collaboration opportunities for open-source benchmark improvements.
   - The event will allow researchers to team up for effective use of provided computing resources, highlighting the importance of community efforts in advancing research.
- **Tensor Parallelism with Decentralized GPU**: A user inquired about **tensor parallelism techniques** specifically related to distributed GPU operations, looking for suggestions.
   - Conversations continued around challenges and opportunities in leveraging decentralized systems for **AI processing**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">Tweet from Steve Frey (@stevewattsfrey)</a>: A bold experiment: We&#39;re hosting a 6-day ultra-hackathon for SWE-Bench to push the limits of open-source code generation  - Everyone gets $1,000 in compute provided by @StrongCompute  - Up 50 rese...</li><li><a href="https://arxiv.org/abs/2406.11403">Multimodal Structured Generation: CVPR&#39;s 2nd MMFM Challenge Technical Report</a>: Multimodal Foundation Models (MMFMs) have shown remarkable performance on various computer vision and natural language processing tasks. However, their performance on particular tasks such as document...</li><li><a href="https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/">AI achieves silver-medal standard solving International Mathematical Olympiad problems</a>: Breakthrough models AlphaProof and AlphaGeometry 2 solve advanced reasoning problems in mathematics</li><li><a href="https://docs.google.com/presentation/d/1UuPaLxesik7zEjVDP3V8tnjidX68iH4ZCUr_MpBY5Aw/edit?usp=sharing">Multimodal Structured Generation</a>: Multimodal Structured Generation &amp; CVPR’s 2nd MMFM Challenge By Franz Louis Cesista franzlouiscesista@gmail.com leloykun.github.io</li><li><a href="https://github.com/leloykun/MMFM-Challenge">GitHub - leloykun/MMFM-Challenge: Official repository for the MMFM challenge</a>: Official repository for the MMFM challenge. Contribute to leloykun/MMFM-Challenge development by creating an account on GitHub.</li><li><a href="https://github.com/leloykun/mmsg">GitHub - leloykun/mmsg: Generate interleaved text and image content in a structured format you can directly pass to downstream APIs.</a>: Generate interleaved text and image content in a structured format you can directly pass to downstream APIs. - leloykun/mmsg</li><li><a href="https://leloykun.github.io/">Franz Louis Cesista</a>: Mathematician | Machine Learning (AI) Research Scientist
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1266810118998331422)** (4 messages): 

> - `ONNX models for TensorRT`
> - `Scrabble fonts resource`
> - `Model learning improvement strategies`
> - `Custom model implementation with ViT`
> - `Connecting vision encoders to language decoders` 


- **Searching for ONNX depth model for TensorRT**: A user inquired about the availability of a **depth ONNX model** that can be transferred to **TensorRT** for the **Jetson Orin Nano**.
   - No specific resources were provided in response.
- **Scrabble fonts might be helpful**: A member mentioned that there are **scrabble fonts** available which could assist, as they also include **numbers** in the corners.
   - This could be utilized in various applications requiring digit recognition or formatting.
- **Improving breed classification model**: A user directed attention to their **breed classification model** on [Kaggle](https://www.kaggle.com/code/root31415/breed-classification) and expressed concerns about it not learning effectively.
   - They sought suggestions for potential improvements to enhance model performance.
- **Implementing a custom model with ViT**: A user expressed interest in utilizing a **Vision Transformer** (ViT) as the encoder while planning to use either **LLaMA 3.1** or **Mistral** as the decoder.
   - They requested guidance on the steps to integrate the vision encoder with the language decoder, particularly concerning input compatibility.



**Link mentioned**: <a href="https://www.kaggle.com/code/root31415/breed-classification">breed_classification</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from Cat Breeds Dataset 

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1266475654958809130)** (24 messages🔥): 

> - `Tokenizer Issues`
> - `LLM Internship Opportunities`
> - `Unstructured Text Processing for RAG`
> - `Multiple Neighbors Ranking Loss`
> - `Dataset Management and Training Improvements` 


- **Tokenizer.apply_chat_template appears broken**: A member reported that `tokenizer.apply_chat_template` is broken in the latest Hugging Face Transformers, specifically mentioning that `add_generation_prompt = False` doesn't work.
- **Seeking LLM internship opportunities**: A newcomer expressed their eagerness to learn about LLMs and requested suggestions for finding internships, including unpaid positions.
- **Guide needed for unstructured text processing**: A member inquired about guides for processing unstructured text for Retrieval-Augmented Generation (RAG), emphasizing the need to clean papers containing various data types.
   - Another member confirmed that querying structured fields is possible and suggested using embedding models for such tasks.
- **Insights on Multiple Negatives Ranking Loss**: A member provided insights on using `MultipleNegativesRankingLoss`, explaining that in-batch negatives are less useful compared to more related negatives for training.
   - They shared their experience that adding multiple negatives per anchor only marginally improved performance, while discussing dataset efficiency and its arrangement.
- **Training improvements when ordering datasets**: A member reported that organizing their training dataset in increasing order of difficulty (i.e., harder negatives towards the end) led to significant improvements in model performance.
   - They noted that this method allows the model to focus more effectively on refining features rather than toggling between different learning focuses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/">Knowledge Graphs for RAG</a>: Build and use knowledge graph systems to improve your retrieval augmented generation applications. Enhance RAG apps with structured data.</li><li><a href="https://huggingface.co/datasets/tomaarsen/gooaq-hard-negatives">tomaarsen/gooaq-hard-negatives · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1266896089307680778)** (1 messages): 

> - `News on AI Use Cases`
> - `People to Follow in AI`
> - `Innovative AI Websites`
> - `AI Channels for Creative Ideas` 


- **Searching for innovative AI news sources**: A member inquired about where to find news or posts regarding **innovative** and **creative AI use cases**.
   - They expressed interest in recommendations for **people to follow**, websites, channels, or any resources.
- **Request for AI community recommendations**: A member requested insights on influential **personalities** or **websites** in the AI sector.
   - Suggestions for specific **channels** or platforms where creative AI use cases are discussed would be greatly appreciated.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1266471068705624094)** (661 messages🔥🔥🔥): 

> - `LM Studio performance`
> - `Model loading issues`
> - `Fine-tuning with embeddings`
> - `RoPE settings in Llama 3.1`
> - `Running LLMs in virtual environments` 


- **Performance variations across models**: Users discussed performance metrics for different models, with observations on speed varying significantly based on GPU configurations and context length settings, especially with Llama 3.1 models.
   - Some users reported varying tokens per second rates, highlighting the impact of GPU type and RAM specifications on inference efficiency.
- **Model loading errors and resolutions**: Several users encountered issues loading models, particularly with errors related to the number of tensors in Llama 3.1, and were advised to update LM Studio or reduce context size.
   - Users were guided on troubleshooting steps, including checking their GPU compatibility and ensuring proper model directory structure.
- **Fine-tuning and embeddings usage**: The conversation included discussions about the effectiveness of fine-tuning versus embeddings for specific libraries, emphasizing that models may need well-prepared examples to function correctly.
   - Participants noted the limitations of models' understanding and the necessity of providing context or tutorial-like content to improve performance.
- **Updates and presets in LM Studio**: Discussions highlighted the importance of using the correct presets for Llama 3.1, with strong recommendations to use the Llama 3 V2 preset for compatibility.
   - Users asked about the functionality of different presets, confirming that outdated versions may not utilize new features effectively.
- **Using LM Studio within a virtual environment**: Users expressed interest in running LM Studio in a Docker or virtual machine setup for better resource management and isolation, with considerations on GUI requirements.
   - Suggestions included leveraging the LMS-CLI tool for headless operation, but users were cautioned about running applications in virtualized environments without proper GPU passthrough.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: no description found</li><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://docs.useanything.com/setup/llm-configuration/local/lmstudio">LMStudio LLM ~ AnythingLLM</a>: LMStudio is a popular user-interface, API, and LLM engine that allows you to download any GGUF model from HuggingFace and run it on CPU or GPU.</li><li><a href="https://www.itpro.com/security/python-developers-beware-this-info-stealing-malware-campaign-is-targeting-thousands-of-github-accounts">Python developers beware: This info stealing malware campaign is targeting thousands of GitHub accounts</a>: Python developers should be wary of an information stealing malware disguised in the popular Colorama python package, which has already compromised a community of over 170,000 users</li><li><a href="https://huggingface.co/gbueno86/Meta-Llama-3.1-70B-Instruct.Q4_0.gguf/tree/main">gbueno86/Meta-Llama-3.1-70B-Instruct.Q4_0.gguf at main</a>: no description found</li><li><a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>: You can use LLMs you load within LM Studio via an API server running on localhost.</li><li><a href="https://huggingface.co/TencentARC/PhotoMaker-V2">TencentARC/PhotoMaker-V2 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/were-not-worthy-waynes-world-worship-not-worthy-gif-15600896">Were Not Worthy Waynes World GIF - Were Not Worthy Waynes World Worship - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/magic-eight-ball-signs-no-gif-12565031">Magic Eight GIF - Magic Eight Ball - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md">llama-models/models/llama3_1/MODEL_CARD.md at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://www.newegg.com/pny-vcnrtx6000ada-pb/p/N82E16814133886?Item=9SIA1K6JZ31513">PNY RTX 6000 Ada VCNRTX6000ADA-PB 48GB 384-bit GDDR6 PCI Express 4.0 x16 Workstation Video Card - Newegg.com</a>: Buy PNY RTX 6000 Ada VCNRTX6000ADA-PB 48GB 384-bit GDDR6 PCI Express 4.0 x16 Workstation Video Card with fast shipping and top-rated customer service. Once you know, you Newegg!</li><li><a href="https://www.newegg.com/p/1VK-0066-00022?Item=9SIATRNJT14908">NVIDIA H100 80GB HBM2e PCIE Express GPU Graphics Card New - Newegg.com</a>: Buy NVIDIA H100 80GB HBM2e PCIE Express GPU Graphics Card New with fast shipping and top-rated customer service. Once you know, you Newegg!</li><li><a href="https://arxiv.org/html/2407.18219v1">Recursive Introspection: Teaching Language Model Agents How to Self-Improve</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8676">Add llama 3.1 rope scaling factors to llama conversion and inference by jmorganca · Pull Request #8676 · ggerganov/llama.cpp</a>: Hi all, this commit generates the rope factors on conversion and adds them to the resulting model as a tensor. At inference time, these factors are passed to the ggml_rope_ext rope operation. From ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1266473863529627688)** (53 messages🔥): 

> - `Snapdragon X Elite ARM CPU Review`
> - `Tesla P40 Cooling Solutions`
> - `GPU Comparison for Model Training`
> - `Llama.cpp Development Updates`
> - `Inference Speed with Multiple GPUs` 


- **Snapdragon X Elite ARM CPU Review sparks discussion**: Members discussed the performance of the new **Snapdragon X Elite ARM CPU** in Windows 11, questioning its usability.
   - A review video titled ["Mac Fanboy Tries ARM Windows Laptops"](https://youtu.be/3GZ4sqB3juQ) was mentioned, generating interest in user experiences.
- **Krypt Lynx experiments with Tesla P40 cooling**: **Krypt Lynx** shared updates on a custom cooling solution for the **Tesla P40**, which requires additional adjustments to hide seams.
   - Discussion included comments on **fan speeds**, performance under load, and plans for testing temperature readings.
- **Choosing GPUs for training models**: There was a consensus that using a **4090** is preferable for model training over older GPUs like the **K80** or **P40**, which are considered outdated.
   - Members highlighted the importance of purchasing modern hardware for **CUDA** support and performance, especially for large models.
- **Llama.cpp development insights**: Members were directed to the **issues tab** on the **Llama.cpp GitHub** to find updates on the **Snapdragon Elite X NPU**'s development.
   - One member confirmed that GPU setups typically allow larger models to be loaded but do not necessarily improve inference speed.
- **Inference speed challenges with multiple GPUs**: Discussion revealed that splitting models over multiple GPUs does not actually improve inference speed, despite allowing for larger model sizes.
   - Members noted that utilizing modern GPUs is generally more efficient than older models, promoting a focus on current technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/quick-maths-mans-not-hot-gif-11163522">Quick Maths Mans Not Hot GIF - Quick Maths Mans Not Hot - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/3GZ4sqB3juQ">Mac Fanboy Tries ARM Windows Laptops</a>: I went to the store and bought a new ARM laptop to see if it competes with the MacBook Air.Hohem Official Store: https://bit.ly/45TOdKf (20% off code: YT20)H...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/1499#issuecomment-2248528025">[Feature request] Any plans for AMD XDNA AI Engine support on Ryzen 7x40 processors? · Issue #1499 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1266475142003818580)** (690 messages🔥🔥🔥): 

> - `Stable Diffusion Tools`
> - `ComfyUI vs A1111 and Forge`
> - `Image Inpainting Issues`
> - `Character Consistency in AI Generations`
> - `AMD's Amuse 2.0 Censorship` 


- **Comparison of AI Tools for Image Generation**: Users discussed the differences between ComfyUI, A1111, and Forge, highlighting that ComfyUI allows for more control and has various advantages in terms of model use and speed.
   - Several users also noted that Forge has faced issues with its recent update, making A1111 a potential alternative for those experiencing problems.
- **Issues with Inpainting and Output Quality**: Modusprimax encountered persistent blurry outputs from the new Forge inpainting feature, leading to frustration despite various configurations tried.
   - Others suggested exploring ComfyUI or older versions of Forge for potentially better results.
- **Maintaining Character Consistency in Generative AI**: Users shared tips on using specific models and IP adapters with checkpoints to achieve character consistency, noting that some models better serve this purpose than others.
   - Neonninjaastro recommended using the 'Mad Scientist' model for stronger output in terms of character anatomy.
- **Discussion on AMD's Amuse 2.0 App**: Gitiyasix mentioned that AMD's Amuse 2.0 model for Stable Diffusion is heavily censored, impacting its ability to render certain body curves.
   - The conversation transitioned into concerns about censorship in AI applications and the implications for user creativity.
- **Learning Resources and Community Support**: Several users emphasized the importance of engaging with video tutorials and community forums to deepen understanding of Stable Diffusion prompts and workflows.
   - Crystalwizard encouraged users to explore ComfyUI features and clarified common misconceptions about various tools used in AI generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/search/models?modelType=LORA&sortBy=models_v9&query=rifle">Civitai | Share your models</a>: no description found</li><li><a href="https://fyrean.itch.io/bgbye-background-remover">BGBye - Background Remover by Fyrean</a>: Free background remover, 10 methods!</li><li><a href="https://microsoft.github.io/DirectML/">DirectML</a>: Learn about DirectML, a high-performance ML API that lets developers power AI experiences on almost every Microsoft device.</li><li><a href="https://www.youtube.com/@sedetweiler">Scott Detweiler</a>: Quality Assurance Guy at Stability.ai &amp; PPA Master Professional Photographer  Greetings!  I am the lead QA at Stability.ai as well as a professional photographer and retoucher based near Milwaukee...</li><li><a href="https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview">Stable Diffusion pipelines</a>: no description found</li><li><a href="https://stability.ai/news/license-update">Community License &mdash; Stability AI</a>: Our new Community License is now free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in...</li><li><a href="https://www.cdiscount.com/informatique/ordinateurs-pc-portables/pc-portable-msi-creator-m16-a11uc-850fr-16-qh/f-1070992-9s7158242850.html#mpos=0|mp">Cdiscount.com</a>: Cdiscount : Meuble, Déco, High Tech, Bricolage, Jardin, Sport | Livraison gratuite à partir de 10€ | Paiement sécurisé | 4x possible | Retour simple et rapide | E-commerçant français, des produits et ...</li><li><a href="https://huggingface.co/docs/diffusers/en/index">Diffusers</a>: no description found</li><li><a href="https://www.youtube.com/channel/UCissXBD6LDXmf2WNxE_F15A">Jônathas Aquino Melo</a>: Esta é uma jornada multidisciplinar entre arquitetura e tecnologia por meio da IA Stable Diffusion, desvendando o conceito de jogo conforme desenvolvido por Huizinga e explorando a sua relação com o a...</li><li><a href="https://stability.ai/stable-artisan">Stable Artisan &mdash; Stability AI</a>: Stable Artisan is a fun multimodal generative AI Discord bot that utilizes the products on the Stability AI Platform API within the Discord ecosystem.</li><li><a href="https://www.youtube.com/watch?v=4YLHGEVeVpk">RTX 4090 vs 3090 ti stable diffusion test. (UPDATE) This video is now out of date!</a>: I reran the test without recording and the 4090 completed the run in 10.46 seconds and the 3090 ti completed the run in 16.62 seconds. Which makes the 4090 4...</li><li><a href="https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436">ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com</a>: Buy ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/16030">Stable Diffusion 3 support by AUTOMATIC1111 · Pull Request #16030 · AUTOMATIC1111/stable-diffusion-webui</a>: Description  initial SD3 support can load sd3_medium.safetensors from https://huggingface.co/stabilityai/stable-diffusion-3-medium will download CLIP models from huggingface into models/CLIP direct...
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1266491770133549181)** (602 messages🔥🔥🔥): 

> - `SearchGPT Access`
> - `Technical Issues with ChatGPT`
> - `Coding Assistance with AI`
> - `User Experiences with AI Models` 


- **SearchGPT: A New Tool for Users**: Users shared positive experiences with **SearchGPT**, highlighting its ability to search through multiple credible sources and sometimes utilize **Chain of Thought (CoT)** reasoning during queries.
   - One user noted the tool's capability to calculate specific trip costs while retrieving relevant car model information, indicating its practical application.
- **Ongoing Technical Issues with ChatGPT**: Multiple users reported difficulties accessing the **ChatGPT** website, with suggestions to clear the cache and check browser extensions to resolve connectivity issues.
   - One user was particularly frustrated with the inability to log in after two weeks, emphasizing a lack of response from OpenAI support.
- **AI as a Coding Assistant**: Users discussed their experiences using AI for coding tasks, with one user successfully creating a Python script to download and launch Chrome, showcasing the efficiency of AI assistance.
   - Another user shared their workflow using **ChatGPT** to directly write code on their server, enhancing collaboration through feedback and iteration.
- **Voice Mode Release Update**: Anticipation around the release of **voice mode** in ChatGPT was expressed, with insights that it is rolling out this week for a limited number of users.
   - The discussion included speculation about the selection criteria for users receiving access to this new feature.
- **User Queries about AI Capabilities**: One user inquired about developing an AI specifically for coding that surpasses current capabilities, leading to discussions on the complexity and investment required for such a model.
   - Others emphasized the importance of using AI to enhance coding efficiency rather than completely offloading responsibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2207.14255">Efficient Training of Language Models to Fill in the Middle</a>: We show that autoregressive language models can learn to infill text after we apply a straightforward transformation to the dataset, which simply moves a span of text from the middle of a document to ...</li><li><a href="https://www.micro1.ai/gpt-vetting-staffing">AI Interviewer for High Volume Recruitment Agencies | micro1</a>: Interview 100x more candidates async using AI</li><li><a href="https://fxtwitter.com/testingcatalog/status/1817099943905046727?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: A real life preview example of SearchGPT and how fast it is 👀👀👀  Quoting Kesku (@yoimnotkesku)   SearchGPT is pretty fast</li><li><a href="https://x.com/ai_for_success/status/1817037780733898911?s=46">Tweet from AshutoshShrivastava (@ai_for_success)</a>: OpenAI new SearchGPT access is rolling out. If you are lucky, you might get to access it very soon. Did anyone else get access other than Alex?  Quoting Alex Volkov (Thursd/AI) (@altryne)   Wasn&#39;t...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1266523283663228969)** (13 messages🔥): 

> - `API response size limits`
> - `Site access issues`
> - `Image editing feature removal`
> - `GPT-4o model parameters`
> - `Function call for network utility settings` 


- **API response size limit for custom actions**: A user inquired about the maximum size of API responses on custom actions before encountering errors.
   - This discussion doesn't seem to have a definitive answer yet.
- **Ongoing site access problems**: Multiple users reported issues with accessing the site, experiencing prolonged loading times.
   - One user noted that they had been facing this problem for several days.
- **Disappearance of image editing features**: A member expressed concern about the removal of an option to edit specific parts of images with a brush.
   - Another user suggested it might be a temporary bug, as the feature still works on mobile.
- **GPT-4o model parameter inquiry**: A user asked how many parameters the GPT-4o and mini models have, indicating a lack of clarity on this topic.
   - No response or information was provided in the discussion regarding the parameters.
- **Function call assistance for network utilities**: A user sought help on configuring functions for a network utility based on OpenAI but reported only partial functionality.
   - They specifically indicated a need for assistance from professionals on the matter.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1266737921344540703)** (11 messages🔥): 

> - `Function Calls for Network Utility`
> - `Self-Improvement Book Query`
> - `Russian Language Discussions`
> - `Profile and Posture Check Functions` 


- **Queries on Uploading Books for AI Answers**: A user inquired whether they could upload a self-improvement book to receive answers via prompts, to which another member replied that it is highly dependent on the content of the book.
   - The conversation indicated that for self-improvement materials, engaging with the content for accurate responses is likely achievable.
- **Sharing Cultural Backgrounds**: A user identified themselves as Russian, and another member followed up by stating they are Ukrainian, highlighting a cultural exchange in the chat.
   - This brief interaction showed the diverse backgrounds of the members participating.
- **Challenges in Function Calls for Network Utility**: A user detailed troubles they faced while trying to write a function call for configuring settings in a network utility, specifically needing two functions to be called simultaneously.
   - Another member suggested that including a clear system message could potentially aid in resolving the issue, while noting it worked correctly with a different approach.
- **Testing Function Calls**: A member mentioned their intent to test the long tools shared in the conversation, showing engagement with the coding challenge presented.
   - They further indicated that with a 2-shot approach, both methods called were functioning correctly, a detail that offers insight into troubleshooting processes.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1266737921344540703)** (11 messages🔥): 

> - `Book Upload Functionality`
> - `Network Utility Configuration`
> - `Language Backgrounds` 


- **Book Uploading for Queries**: @thrallboy inquired about uploading a book and using prompts to ask questions about it, to which @darthgustav. responded that it is highly dependent on the book.
   - The discussion varied towards self-improvement books, with @darthgustav. confirming that this type is likely more manageable.
- **Configurations for Network Utility**: @polarisjrex0406 sought help on writing a function call for configuring a network utility based on OpenAI but faced issues with function calls not executing correctly.
   - @neural_nova_28405 suggested adding a clear system message might improve functionality, noting that even with smaller models, both required methods were being called.
- **Cultural Exchange in the Community**: A cultural exchange was noted with members identifying as Russian and Ukrainian, sharing their backgrounds.
   - This interaction highlighted the diversity within the community, fostering inclusivity and dialogue among members.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1266470487354114048)** (292 messages🔥🔥): 

> - `Unsloth Usage`
> - `Llama 3.1 Model Discussion`
> - `Fine-tuning Techniques`
> - `Model Quantization`
> - `Inference Settings` 


- **Best Practices for Using Unsloth**: Users have discussed the effectiveness of various system messages for Llama 3.1, with some noting that the default from Unsloth notebooks works sufficiently well for their tasks.
   - Some participants have even opted to remove the system message to conserve context length without observing any significant change in performance.
- **Fine-tuning and LoRa Adapters**: Several users confirmed that LoRa adapters created through fine-tuning with Unsloth can be successfully applied to the original Llama models, as long as the base model is the same.
   - There remains some uncertainty about the compatibility of these adapters across different model versions, emphasizing the importance of using the correct model.
- **Quantization and VRAM Considerations**: Discussions included the trade-offs between using 4-bit and 16-bit models, with participants noting that while 16-bit requires 4x the VRAM, it yields better performance.
   - Users are encouraged to experiment with both bit levels, as individual experiences vary based on specific use cases.
- **Inference Settings for Llama 3.1**: Participants noted that inference with Llama 3.1 requires significant VRAM, suggesting that 48 GB is needed for full capabilities, particularly for larger models.
   - They discussed the process of handling inference requests while maximizing GPU utilization, especially when using libraries like vLLM.
- **Resources for Understanding LLMs**: Users shared resources, including videos by Andrej Karpathy and articles focusing on the understanding of large language models and their pre-training mechanisms.
   - A variety of guides and articles were recommended, making it easier for newcomers to navigate the complexities of LLM training and fine-tuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/FhBnfFP">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b_Beta-Preview">Replete-LLM-Qwen2-7b_Beta-Preview - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=Zt9CHJqO6p30">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4#scrollTo=MKX_XKs_BNZR">Google Colab</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu118">no title found</a>: no description found</li><li><a href="https://magazine.sebastianraschka.com/p/understanding-large-language-models">Understanding Large Language Models</a>: A Cross-Section of the Most Relevant Literature To Get Up to Speed</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored">Orenguteng/Llama-3.1-8B-Lexi-Uncensored · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: Fine-tune and run Meta&#x27;s updated Llama 3.1 model with 6x longer context lengths via Unsloth!</li><li><a href="https://x.com/maximelabonne/status/1817850208094486770">Tweet from Maxime Labonne (@maximelabonne)</a>: 🦥 Fine-tune Llama 3.1 Ultra-Efficiently with @UnslothAI   New comprehensive guide about supervised fine-tuning on @huggingface.   Over the last year, I&#39;ve done a lot of fine-tuning and blogging. ...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=3eq84KrdTWY">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: Learn how to easily fine-tune Meta&#39;s powerful new Llama 3 language model using Unsloth in this step-by-step tutorial. We cover:* Overview of Llama 3&#39;s 8B and...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/byxkTHDIpI">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/pacozaa/mistral-sharegpt90k-merged_16bit">pacozaa/mistral-sharegpt90k-merged_16bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://github.com/kvcache-ai/ktransformers">GitHub - kvcache-ai/ktransformers: A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations</a>: A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations - kvcache-ai/ktransformers</li><li><a href="https://github.com/huggingface/peft/releases/tag/v0.12.0">Release v0.12.0: New methods OLoRA, X-LoRA, FourierFT, HRA, and much more · huggingface/peft</a>: Highlights  New methods OLoRA @tokenizer-decode added support for a new LoRA initialization strategy called OLoRA (#1828). With this initialization option, the LoRA weights are initialized to be or...</li><li><a href="https://github.com/huggingface/peft/tree/main/examples/olora_finetuning">peft/examples/olora_finetuning at main · huggingface/peft</a>: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8676#issuecomment-2252855962">Add llama 3.1 rope scaling factors to llama conversion and inference by jmorganca · Pull Request #8676 · ggerganov/llama.cpp</a>: Hi all, this commit generates the rope factors on conversion and adds them to the resulting model as a tensor. At inference time, these factors are passed to the ggml_rope_ext rope operation. From ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1266677858542161930)** (29 messages🔥): 

> - `Hugging Face Inference Endpoints`
> - `Hardware Requirements for Fast Training`
> - `Ollama Agent Roll Cage Video`
> - `Applied LLMs Resources` 


- **Clarification on Hugging Face Inference Endpoints**: A member inquired whether 'protected' on Hugging Face means anyone with a token or just the owner, to which it was clarified that *it's your own token* and *do not share with anyone*.
   - Sharing your token could allow others to upload on your page, making it crucial to keep it secure.
- **Fast Training Hardware Needs**: A discussion emerged around which operations need hardware acceleration for fast training, with suggestions that *RTX 2080 or newer GPUs* are minimum requirements.
   - Members also pondered the viability of using theoretical hardware like embedded GPUs or TPUs for training needs.
- **Ollama Agent Roll Cage Video Release**: A member shared a YouTube video titled *Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, & Agent Library*, showcasing new optimizations for speech and vision agents and including a customizable agent library.
   - They expressed excitement about the updates and encouraged members to check it out, linking to the demo video.
- **Free Resources for Applied LLMs Course**: A member announced that several resources from the *Applied LLMs course* are now available for free, enhancing accessible learning materials with added tracks and notes for better comprehension.
   - This release aims to maximize learning opportunities for everyone interested in the subject.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/HamelHusain/status/1817935895246635362">Tweet from Hamel Husain (@HamelHusain)</a>: If you remember our Applied LLMs course, you&#39;ll love this.  Today, we are making all these resources available for free to everyone! 📚   We did extra work to add learning tracks, resources, and n...</li><li><a href="https://www.youtube.com/watch?v=W7TusPTnNXA">Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, &amp; Agent Library</a>: Welcome to the demo of Ollama Agent Roll Cage (OARC) for V0.28.0! This video showcases the latest advancements in my speech-to-speech and image recognition c...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1266529301918908536)** (306 messages🔥🔥): 

> - `Orpo Dataset Creation`
> - `Llama Model Fine-tuning`
> - `Accuracy Metrics in Training`
> - `Dynamic Rope Scaling in Models`
> - `LoRA Adapters Usage` 


- **Creating ORPO Datasets Efficiently**: A member discussed the tedious process of creating ORPO datasets without writing everything manually and inquired if a UI could automate parts of this process.
   - Suggestions included utilizing a smarter model for positive responses and using the fine-tuned model to generate responses.
- **Challenges with Llama 3.1 and Python Packages**: Users faced issues with the Llama 3.1 model in Colab, receiving errors related to missing tensor files and required package versions.
   - After troubleshooting, it was found that installing specific versions of Python packages resolved the tensor mismatch errors.
- **Evaluating Model Performance with Accuracy Metrics**: A user questioned how to incorporate accuracy metrics in training workflows, as traditional metrics like loss may not provide enough insight into model performance.
   - It was emphasized that tracking both loss and accuracy on validation datasets is crucial to avoid overfitting.
- **Understanding Dynamic Rope Scaling in Models**: Users inquired about the validity and implementation of dynamic rope scaling in their models, particularly when facing errors related to unsupported configurations.
   - Clarifications were given about setting the rope_scaling parameter to null or 'none' to resolve issues when fine-tuning.
- **Apple’s Usage of LoRA in Adapters**: Discussion on how Apple utilizes LoRA adapters for fine-tuning foundation models highlighted the use of task-specific adapters initialized from accuracy-recovery adapters.
   - Rank 16 adapters were noted for striking a balance between model capacity and inference performance in Apple’s on-device applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rohanpaul_ai/status/1818063089872384348?s=46">Tweet from Rohan Paul (@rohanpaul_ai)</a>: 📌 LoRA adapters fine-tune the foundation models for specific tasks.  📌 Adapters are applied to all linear projection matrices in self-attention layers and fully connected layers in feedforward netwo...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth? Start here!</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit">unsloth/Meta-Llama-3.1-70B-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://www.unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://x.com/maximelabonne/status/1817850208094486770">Tweet from Maxime Labonne (@maximelabonne)</a>: 🦥 Fine-tune Llama 3.1 Ultra-Efficiently with @UnslothAI   New comprehensive guide about supervised fine-tuning on @huggingface.   Over the last year, I&#39;ve done a lot of fine-tuning and blogging. ...</li><li><a href="https://docs.unsloth.ai/basics/chat-templates,">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/734"> Add support for InternLM2.5 model · Issue #734 · unslothai/unsloth</a>: Hello unsloth team, I&#39;m trying to use the InternLM2.5 model (specifically internlm/internlm2_5-7b-chat) with unsloth, but I&#39;m encountering a NotImplementedError. Could you please add support f...</li><li><a href="https://huggingface.co/docs/transformers/v4.43.3/en/model_doc/llama#transformers.LlamaConfig.max_position_embeddings">LLaMA</a>: no description found</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/453">pip install flash-attn always happens ModuleNotFoundError: No module named &#39;packaging&#39;,but actually i have pip install packaging · Issue #453 · Dao-AILab/flash-attention</a>: Collecting flash-attn Using cached flash_attn-2.0.7.tar.gz (2.2 MB) Installing build dependencies ... done Getting requirements to build wheel ... error error: subprocess-exited-with-error × Gettin...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1329-L1352">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1266546819178037248)** (2 messages): 

> - `Unsloth AI`
> - `Grant Proposal Resources` 


- **Clarification on Unsloth AI**: A member noted, *'It's Unsloth....'* to clarify a point in the discussion.
   - This prompted others to engage further on the topic of Unsloth AI.
- **Discussion on Grant Proposals**: A member offered to prepare a grant proposal, asking for interest and needed resources.
   - They expressed readiness to assist the team in getting organized for potential funding.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1267275320474603613)** (4 messages): 

> - `[PAD] Token in Models`
> - `Finetuning Phi-3 Models`
> - `Word Removal for GPT Training` 


- **[PAD] Token might be a class**: One member speculated that the **[PAD]** token is treated as a class in certain models, though their certainty is in question.
   - This discussion linked to the [finetuning script](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/c1358f8a35e6d2af81890deffbbfa575b978c62f/sample_finetune.py#L141) for models, showcasing how **PAD** may function within that context.
- **Finetuning methods for Phi-3**: A member provided an overview of how to finetune **Phi-3** models, mentioning the use of **DeepSpeed ZeRO3** for memory efficiency.
   - The instructions included steps like reducing batch size and setting appropriate parameters to manage resource consumption effectively.
- **Removing unimportant words for GPT training**: Another member posed a question about tactics for eliminating non-essential words in rough text to prepare data for GPT training.
   - No direct solutions were offered in the messages, leaving the query open for suggestions and insights.



**Link mentioned**: <a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/c1358f8a35e6d2af81890deffbbfa575b978c62f/sample_finetune.py#L141">sample_finetune.py · microsoft/Phi-3-mini-4k-instruct at c1358f8a35e6d2af81890deffbbfa575b978c62f</a>: no description found

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1266802933316583534)** (4 messages): 

> - `Mojo Community Meeting`
> - `Computational Linear Algebra Course`
> - `Predicate Computation`
> - `Predicate Registers` 


- **Mojo Community Meeting Scheduled**: On Monday, **July 29 at 10 PT**, the **Mojo** community is having their next meeting, featuring @clattner_llvm on **GPU programming with Mojo**.
   - The agenda includes **Async Mojo** with **10 Simple Rules** and a **Community Q&A**, with full details available in the [Modular community calendar](https://modul.ar/community-meeting).
- **Fast.ai Offers New Course on Computational Linear Algebra**: Fast.ai has launched a new free course, _Computational Linear Algebra_, which includes an [online textbook](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) and a [video series](https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY).
   - This course is the first of its kind focused on **practical applications**, using tools like **PyTorch** and **Numba**, and teaches algorithms for tasks such as identifying foregrounds in videos and reconstructing images from CT scans.
- **Understanding Predicate Computation and Registers**: A user inquired about **predicate computation** and **predicate registers**, which track which warps participate in current instructions for conditional code executing different branches.
   - Another member clarified that these registers are essential for managing branching in threads within a **warp** during execution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.fast.ai/posts/2017-07-17-num-lin-alg.html">fast.ai - New fast.ai course: Computational Linear Algebra</a>: Making neural nets uncool again</li><li><a href="https://x.com/modular/status/1816957129879879684?s=46">Tweet from Modular (@Modular)</a>: 📆 Monday, July 29 at 10 PT, join the Mojo 🔥 community for its next meeting! On the agenda:  🔢 @clattner_llvm on GPU programming with Mojo 🔥 🔀 Async Mojo 🔥 - 10 Simple Rules ❓ Community Q&A  Full...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1267515764768903421)** (2 messages): 

> - `Triton exp function`
> - `libdevice exp implementation`
> - `PTX assembly inspection` 


- **Triton exp function may sacrifice accuracy**: A member noted that the **exp function** in **Triton** seems to utilize a fast `__expf` implementation, which compromises **accuracy for speed**.
   - This raises questions about whether the **exp version in libdevice** follows the same pattern or uses `expf` instead.
- **Inspecting PTX for implementation details**: Another member suggested that the discrepancy can be checked by examining the **PTX assembly** output from Triton.
   - *Looking at the output should clarify which implementation is actually being used*.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1266470560318230672)** (17 messages🔥): 

> - `Optimizing CPU Offload for Optimizer States`
> - `Activation Offloading Strategies`
> - `Paged Optimizers Discussion`
> - `Challenges with FSDP and Single-GPU Training`
> - `Future of Contributions to PyTorch Repositories` 


- **Optimizing CPU Offload for Optimizer States Stirs Interest**: Members discussed the mechanics of **CPU offload** for optimizer states, questioning if it involves storing optimizer states in CPU memory and transferring parameters during each optimization step.
   - One member noted, *'fused ADAM implementation is key to making CPU offloading viable'* and shared insights on its current proof of concept.
- **Confusion Surrounds Paged Optimizer versus Attention**: The conversation revealed confusion regarding a link to **paged attention**, with questions raised about what exactly is being paged, possibly the **KV cache**.
   - A member brought attention to a discussion on GitHub concerning **paged optimizers**, citing it requires CUDA/C++ code, which they prefer to avoid.
- **Concerns About FSDP for Single-GPU Training**: Members expressed frustrations that utilizing **FSDP** with single-GPU training is overly complicated and largely impractical.
   - In response, one user mentioned they were focusing on simpler **CPU offload** methods for their ongoing project.
- **Exploring APIs for Memory Management Without C/C++ Extensions**: Suggestions were made to leverage CUDA APIs such as **cudaMemPrefetchAsync** and **cudaMallocManaged** for memory handling without a C/C++ extension.
   - Despite these recommendations, one member reinforced their focus on **optimizer CPU offload** rather than the paged optimizer approach.
- **Future Contributions and Experimentation Noted**: Users discussed the temporary storage of experimental work in separate repositories due to the non-research nature of their projects, leading to faster iteration.
   - One participant expressed intent to contribute their validated ideas to **torchao** once experiments yield promising results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/425/files#diff-0e7c18a93422c6127f5ee71124d6c5381e8dcf5b1cc9143189a5d8fc6f2e1015R10">Paged attention by liangan1 · Pull Request #425 · pytorch/ao</a>: Related RFC</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/issues/962#issuecomment-1894348159.">paged optimizers doc? · Issue #962 · bitsandbytes-foundation/bitsandbytes</a>: Feature request Hi Tim, I have just accidentally discovered that you added paged optimizers to this library - so very awesome! But there is absolutely zero documentation - would you consider adding...</li><li><a href="https://github.com/pytorch/ao/pull/425/files#diff-0">Paged attention by liangan1 · Pull Request #425 · pytorch/ao</a>: Related RFC</li><li><a href="https://github.com/pytorch/torchtitan/pull/467">[Do not review] Activation offloading by awgu · Pull Request #467 · pytorch/torchtitan</a>: Stack from ghstack (oldest at bottom):  -&gt; #467  Current UX  We use a saved_tensors_hooks context manager, which should be wrapped around module.forward. The context lets us override pack and unpac...</li><li><a href="https://github.com/pytorch/ao/issues/519">Paged Low Bit Optimizers · Issue #519 · pytorch/ao</a>: Right now our Optimizers are low bit so they save a bunch of memory but considering optimizers can also spike memory it&#39;s common to page them out to CPU RAM. There&#39;s a prototype of this here #...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1266795044946579477)** (18 messages🔥): 

> - `INT8 model training`
> - `BF16 model performance`
> - `Stochastic rounding challenges`
> - `Quantization aware training`
> - `8-bit optimizer results` 


- **INT8 Model Training Shows Promise**: Inspired by Q-GaLore work, a member is exploring **INT8 model training** while fine-tuning **ViT-Giant** (1B params) and sharing promising results with similar loss curves and validation accuracy to the **BF16** baseline.
   - They noted that accuracy dropped significantly when using an **8-bit optimizer** with the **INT8 model**, indicating further testing is needed.
- **BF16 Model + 8-bit Optimizer Maintains Accuracy**: After re-running experiments, it was found that the **BF16 model + 8-bit optimizer** maintains accuracy well compared to the **INT8 model + 8-bit optimizer**, which shows large drops.
   - Randomness in results led to discussions about ensuring consistency across setups by using `torch.manual_seed()` for data sequence and augmentation.
- **Stochastic Rounding Complexity**: Handling **denormal numbers** in implementing stochastic rounding for **FP8** poses challenges, with one member sharing experiences from their own work on mapping **BF16 values** to subnormals in **FP8**.
   - Concerns were expressed over the lack of pre-emptive testing for bias introduced by rounding approaches, highlighting the importance of writing comprehensive tests.
- **Discussion on Random Seed Implementation**: A member raised questions about the effectiveness of setting **random seeds** in **torch**, suspecting that **torch.compile** could affect randomness during code translation.
   - Clarification was provided about the intent behind setting the random seed for ensuring data sequence consistency, leading to fruitful discussions on how random number generation is handled.
- **Hyped for Viable Quantized Optimizer**: Excitement surrounded the potential of a viable **quantized optimizer**, with multiple members expressing eagerness for further updates and results as progress continues.
   - The shared journey into effective quantization strategies reflects a broader enthusiasm for enhancements in model training efficiency.



**Link mentioned**: <a href="https://github.com/gau-nernst/quantized-training">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>: Explore training for quantized models. Contribute to gau-nernst/quantized-training development by creating an account on GitHub.

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1267067570117414933)** (4 messages): 

> - `CUDA Kernel Compilation Error`
> - `Multimodal Hypograph Database`
> - `Understanding VRAM and GPU Memory` 


- **CUDA kernel compilation error resolved**: A user encountered an error when running `load_inline` in PyTorch with CUDA setup, attributed to incorrect installations with missing channel labels. After creating a new environment and ensuring the correct version of CUDA and PyTorch were installed, the issue was resolved successfully.
- **Building a multimodal hypograph database**: For building a multimodal hypograph database using a specific library, one might start by organizing raw data from chaotic sources, like the collection of 2600 books. Leveraging OpenAI's API can help categorize and analyze these resources efficiently.
- **Clarifying VRAM and GPU memory types**: A developer tries to connect consumer knowledge about VRAM with a deeper understanding of GPU memory types learned in lectures. They seek clarification on whether VRAM refers only to global memory or encompasses all memory types, as online searches yielded insufficient answers.



**Link mentioned**: <a href="https://lablab.ai/event/open-interpreter-hackathon/2600-books-files-sorted/2600-books-sorted-for-multi-agent-creation">2600 Books Sorted for Multi-Agent Creation</a>: In the neon-lit digital underworld, we wielded the code, Bash, and the terminal like a switchblade in a dark alley. With 2600 books jumbled in a main folder, chaos reigned. But then, we summoned OpenA...

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1267139444079722557)** (3 messages): 

> - `CUDA Cores vs FP32 Units`
> - `Thread Processing Capability`
> - `Integer vs FP32 Units`
> - `Hopper Architecture Insights`
> - `Nvidia's Implementation Secrets` 


- **CUDA Cores represent FP32 capabilities**: The term **CUDA Cores** primarily indicates the number of **FP32** operations that can execute in parallel, which is crucial when considering **FP64** performance due to the limited number of FP64 units compared to FP32 in big accelerators like **H100** and **A100**.
   - *It is important to remember* that the FP32 unit count does not convey the total computational potential of the GPU, especially when factoring in different data types.
- **Thread processing limited to CUDA Core count**: Each **Streaming Multiprocessor (SM)** can handle **as many threads** as there are CUDA cores at any one time, indicating that one thread per CUDA core is the maximum processing capacity.
   - *Yes*, one thread per core means that processing efficiency depends heavily on optimizing thread usage relative to core availability.
- **Integer units function separately from FP32 units**: It was explained that **integer units** operate independently alongside FP32 units, allowing for parallel execution of **integer** calculations even when a GPU features a limited number of CUDA cores.
   - For instance, a GPU with **64 CUDA Cores** can support 64 threads for FP32 compute while another set of 64 can manage integer operations.
- **Hopper has fewer integer units than expected**: It's noted that **Hopper architecture** actually contains only half as many integer units compared to FP32 units, contrary to previous architectures like **A100** and **V100**, which had matching counts.
   - This insight reflects evolving design considerations in modern GPUs for optimizing specific computational tasks.
- **Nvidia keeps architectural details under wraps**: Despite sharing schematics and diagrams, some aspects of Nvidia's in-silicon implementation remain tightly guarded business secrets.
   - As such, even detailed diagrams like those discussing **arithmetic units** can be seen as abstractions from actual architectural realities.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1266495497124839577)** (38 messages🔥): 

> - `AWQ Implementation`
> - `AQT Changes and Updates`
> - `Quantization Performance Issues`
> - `2-bit Llama3 Model`
> - `Tensor Packing Issues` 


- **AWQ Implementation Simplified**: The most basic form of **AWQ** involves creating an **AQT layout class** that stores activation channel scales, allowing for a straightforward implementation.
   - Members discussed potential improvements and scaling methods, considering **block_size** and group-based approaches for effective implementation.
- **Recent Changes in AQT are Confusing**: The recent modification in **AQT** requires input weights to be pre-packed before using `_convert_weight_to_int4pack`, resulting in unexpected performance metrics.
   - Discussion included a **runtime error** arising from adjusted input weight requirements, confirming that the tests in **ao** are currently disabled for nightly builds.
- **Noteworthy Performance Observations**: When comparing **torchao** with **bitblas**, the performance is close, with torchao delivering 94 tokens/sec and bitblas 97 tokens/sec for 4-bit models with batch size 1.
   - Further comparisons indicate that as batch size increases, performance significantly drops, especially for 2-bit models using **HQQ+** with **Llama3**.
- **Challenges in 2-bit Llama3 Quantization**: The **2-bit Llama3 model** faces quantization challenges, particularly at lower bits, impacting speed and quality, even with techniques like **low-rank adapters**.
   - Despite difficulties, it runs efficiently with the **BitBlas** framework, achieving speeds of **95-120 tokens/sec**, similar to its 4-bit counterparts.
- **Future Directions for uint4 Tensor Subclass**: There are plans to establish a **default packing format** of `[n][k/2]` for the upcoming **uint4 tensor subclass**, transitioning it to an optimized layout as needed.
   - This change could streamline processes and potentially merge layout functionalities from **AffineQuantizedTensor** to the uint4 tensor design.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq">mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq · Hugging Face</a>: no description found</li><li><a href="https://github.com/pytorch/ao/issues/530">Add AWQ support · Issue #530 · pytorch/ao</a>: AWQ seems popular: 3000 appearances in huggingface models: (https://huggingface.co/models?sort=trending&amp;search=AWQ), similar to GPTQ. Maybe we can add this to torchao as well. Overview At the high...</li><li><a href="https://github.com/pytorch/ao/blob/afde1755d906ad644e04835675e7856d72c3c87b/torchao/quantization/smoothquant.py#L150-L153">ao/torchao/quantization/smoothquant.py at afde1755d906ad644e04835675e7856d72c3c87b · pytorch/ao</a>: Custom data types and layouts for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/458">[WIP] Int4Tensor refactor to implements pattern by melvinebenezer · Pull Request #458 · pytorch/ao</a>: Refactoring UInt4Tensor to have implements pattern similar to nf4tensor  and UInt2Tensor ToDo   Create implements for UInt4Tensor and PerChannelSymmetricWeight  Test Cases  Move uint4i to uint4.py</li><li><a href="https://github.com/pytorch/ao/pull/517">Fix int4pack_mm error by yanbing-j · Pull Request #517 · pytorch/ao</a>: Need update meta shape in PyTorch first pytorch/pytorch#130915.</li><li><a href="https://github.com/pytorch/pytorch/commit/6f662e95756333284450ff9c3c6e78c796aa6e77">update the input `weight` of `_convert_weight_to_int4pack` to `[n][k … · pytorch/pytorch@6f662e9</a>: …/ 2] uint8` (#129940)  This PR is to update the input `weight` of `_convert_weight_to_int4pack` from `[n][k] int32` to `[n][k / 2] uint8`, both for CPU, CUDA and MPS, which can help decouple int4 ...</li><li><a href="https://github.com/pytorch/pytorch/blob/6f662e95756333284450ff9c3c6e78c796aa6e77/torch/testing/_internal/common_quantization.py#L478-L479">pytorch/torch/testing/_internal/common_quantization.py at 6f662e95756333284450ff9c3c6e78c796aa6e77 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1267255492795957290)** (3 messages): 

> - `LinkedIn Post on Token Pricing`
> - `Twitter Discussions` 


- **LinkedIn Post on Token Pricing**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/yangqing-jia_people-often-ask-why-prices-like-28m-token-activity-7222871364666376192-toMF?utm_source=share&utm_medium=member_ios) discussing the peculiarities of **token pricing**, specifically referencing a price of **28M tokens**.
   - This post sparked interest as another member noted seeing the same information on **Twitter**.
- **Whimsical 1000-Dimensional Orange**: One member humorously remarked that a **1000-dimensional orange** is essentially **100% peel**, plus or minus a rounding error.
   - This comment lightened the mood, blending humor with complex dimensional concepts.


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1266806610219503758)** (2 messages): 

> - `Lakeview Meetup` 


- **Totaldev is in Lakeview**: A member noted that they are currently in **Lakeview**.
   - *Sweet*.
- **Totaldev expresses excitement about location**: Another member expressed enthusiasm by stating, *Sweet, I’m in Lakeview*.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1266479322869272708)** (401 messages🔥🔥): 

> - `RoPE performance`
> - `Gradient clipping challenges`
> - `Training stability`
> - `SwiGLU implementation`
> - `CUDA-related issues` 


- **RoPE improves model performance**: RoPE has shown significant improvements in model performance during training, with users noting better stability compared to baseline methods without it.
   - Manual testing confirmed that the kernel meets expected output, leading to a successful implementation of RoPE.
- **Concerns over gradient clipping**: There are ongoing discussions about how gradient clipping may interfere with Adam optimizer's performance, especially regarding updating the second moment based on potentially reduced gradients.
   - Many contributors expressed skepticism about the efficacy of gradient clipping in Adam, suggesting that it might not be beneficial for training stability.
- **Training stability issues**: Users highlighted the concerning instability in training runs, particularly regarding the use of different GPUs and their configurations, which may yield different results.
   - Investigations into model convergence and performance indicated that fine-tuning approaches might not always lead to optimal results depending on the hardware used.
- **Complications with SwiGLU implementation**: The implementation of SwiGLU turned out to be more complicated than initially anticipated, requiring substantial changes throughout the codebase.
   - Developers are moving forward with the SwiGLU integration while also addressing related architecture improvements.
- **CUDA and cuDNN compatibility challenges**: There are concerns regarding the operation and performance of CUDA and cuDNN functions, especially in relation to FP8 performance and GPU utilization.
   - Discussions pointed out that different CUDA architectures might yield different training dynamics, leading to unexpected results in model behavior.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cloneofsimo/status/1815750768299037153">Tweet from Simo Ryu (@cloneofsimo)</a>: As a gpu-poor I hope this works at scale. 1.8k $ can you believe this?  https://arxiv.org/abs/2407.15811</li><li><a href="https://arxiv.org/abs/2407.17465">u-$μ$P: The Unit-Scaled Maximal Update Parametrization</a>: The Maximal Update Parametrization ($μ$P) aims to make the optimal hyperparameters (HPs) of a model independent of its size, allowing them to be swept using a cheap proxy model rather than the full-si...</li><li><a href="https://arxiv.org/abs/2407.15892">MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training</a>: We introduce Mini-Sequence Transformer (MsT), a simple and effective methodology for highly efficient and accurate LLM training with extremely long sequences. MsT partitions input sequences and iterat...</li><li><a href="https://arxiv.org/abs/2406.04267">Transformers need glasses! Information over-squashing in language tasks</a>: We study how information propagates in decoder-only Transformers, which are the architectural backbone of most existing frontier large language models (LLMs). We rely on a theoretical signal propagati...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>: Teams that have trained large Transformer-based models have reported training instabilities at large scale that did not appear when training with the same hyperparameters at smaller scales. Although t...</li><li><a href="https://arxiv.org/abs/2302.06675">Symbolic Discovery of Optimization Algorithms</a>: We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training. We leverage efficient search techniques to ex...</li><li><a href="https://arxiv.org/abs/2305.19466">The Impact of Positional Encoding on Length Generalization in Transformers</a>: Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding ...</li><li><a href="https://huggingface.co/jrahn/gpt3_125M_edu_pr711">jrahn/gpt3_125M_edu_pr711 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2203.16634">Transformer Language Models without Positional Encodings Still Learn Positional Information</a>: Causal transformer language models (LMs), such as GPT-3, typically require some form of positional encoding, such as positional embeddings. However, we show that LMs without any explicit positional en...</li><li><a href="https://x.com/austinvhuang/status/1816141044540739642">Tweet from Austin Huang (@austinvhuang)</a>: Announcing: The initial release of my 1st project since joining the amazing team here at @answerdotai   gpu.cpp Portable C++ GPU compute using WebGPU  Links + info + a few demos below 👇</li><li><a href="https://x.com/Yuchenj_UW/status/1817223820589375752">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: After training the GPT-2 (2.7B), I delved even &#34;deeper&#34; into the scaling law by training a 7.3B model with @karpathy&#39;s llm.c 🌠  Scaling the model was straightforward, primarily was just s...</li><li><a href="https://github.com/mosaicml/llm-foundry/blob/5e07a05dd2f727928729ad23c26ce68ec8349286/llmfoundry/optim/adaptive_lion.py">llm-foundry/llmfoundry/optim/adaptive_lion.py at 5e07a05dd2f727928729ad23c26ce68ec8349286 · mosaicml/llm-foundry</a>: LLM training code for Databricks foundation models - mosaicml/llm-foundry</li><li><a href="https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/optim/outlier_detection.py">llm-foundry/llmfoundry/optim/outlier_detection.py at main · mosaicml/llm-foundry</a>: LLM training code for Databricks foundation models - mosaicml/llm-foundry</li><li><a href="https://github.com/karpathy/llm.c/pull/711">Outlier detection: catch more outliers by not updating moving average with skipped updates by ademeure · Pull Request #711 · karpathy/llm.c</a>: This is an improvement to the znorm/zgrad update skipping mechanisms (-sl and -sg) to avoid skipping updates for outliers. Note that znorm will still be updated if zgrad is an outlier that causes t...</li><li><a href="https://github.com/karpathy/llm.c/pull/654">Set RNG seed manually with &#39;-rg&#39; parameter by ademeure · Pull Request #654 · karpathy/llm.c</a>: This adds a &#39;-rg&#39; parameter to manually set the RNG seed. This is useful to see if a change is beneficial or not when the difference is potentially real but smaller than the noise threshold. A...</li><li><a href="https://github.com/karpathy/llm.c/pull/714">Add RoPE positional encoding by gordicaleksa · Pull Request #714 · karpathy/llm.c</a>: Implemented RoPE - rotary position embedding from the RoFormer paper. Note:  I do not conditionally remove the allocation of our learnable position embedding buffer (wpe) as that would require touc...</li><li><a href="https://github.com/karpathy/llm.c/pull/715">Feature/restore from master by karpathy · Pull Request #715 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/compare/master...YuchenJin:llm.c:integer-overflow">Comparing karpathy:master...YuchenJin:integer-overflow · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/702">Restore from master weights (&amp; allow restoring from a checkpoint of different precision) by ademeure · Pull Request #702 · karpathy/llm.c</a>: This is fully deterministic for new checkpoints where the new rng_state_last_update is saved, so that stochastic rounding from master weights is done with the exact same seeds (while restoring the ...</li><li><a href="https://github.com/karpathy/llm.c/blob/dec7f767a269bbcd7c8ac8e767b83d549b539f49/train_gpt2.cu">llm.c/train_gpt2.cu at dec7f767a269bbcd7c8ac8e767b83d549b539f49 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1266472633873203361)** (435 messages🔥🔥🔥): 

> - `Perplexity Pro Subscription Limits`
> - `User Experience with Perplexity and AI Models`
> - `Coding and AI for Blogs`
> - `Keyword Research with AI`
> - `Job Opportunities with Perplexity AI` 


- **Clarification on Perplexity Pro Limits**: Users discussed the limits of the Perplexity Pro subscription, noting that Pro users currently have either 540 or 600 daily searches, alongside a limit of 50 messages for the Claude 3 Opus model.
   - The confusion around these limits was acknowledged, indicating potential discrepancies in official documentation.
- **User Experiences with Perplexity AI**: Several users shared their experiences with Perplexity AI, noting its effectiveness for fact-checking and generating accurate blog posts, especially compared to models that may hallucinate information.
   - The consensus is that Perplexity provides a more reliable tool for ensuring the credibility of written content.
- **Using AI Models for Coding**: Discussions highlighted that while AI can assist with coding tasks and provide explanations, it should not solely be relied upon for learning coding.
   - Users recommended utilizing various resources such as YouTube videos along with AI for a better learning experience.
- **Keyword Research Capabilities of Perplexity**: A user inquired about Perplexity's ability to handle keyword research effectively, comparing its output to other AI models like Claude 4o.
   - Responses indicated that both Perplexity and other models can provide satisfactory results for keyword research depending on the prompt used.
- **Job Opportunities at Perplexity AI**: Prospective candidates expressed interest in working with Perplexity AI and discovered a variety of remote job opportunities on the company’s careers page.
   - The high remuneration for certain positions was noted, prompting discussions about the challenges and requirements of these roles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo-results-arent-actually-that-helpful">OpenAI’s SearchGPT demo results aren’t actually that helpful.</a>: The trend of hallucinations showing up in public AI demos continues. As noted by a couple of reporters already, OpenAI’s demo of its new SearchGPT engine shows results that are mostly either wrong or ...</li><li><a href="https://x.com/AravSrinivas/status/1817022053452755143">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Query autocomplete is slowly rolling out to users with a Llama 3.1-based model.</li><li><a href="https://www.windowscentral.com/software-apps/openai-could-be-on-the-brink-of-bankruptcy-in-under-12-months-with-projections-of-dollar5-billion-in-losses">OpenAI could be on the brink of bankruptcy in under 12 months, with projections of $5 billion in losses</a>: OpenAI might need another round of funding to remain afloat.</li><li><a href="https://x.com/altryne/status/1817307423251546347">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: SearchGPT (@OpenAI) vs @perplexity_ai   Ok I heard you guys, here&#39;s a teaser of a comparison I&#39;m cooking up for @thursdai_pod  (will be sent to subscribers first then to everyone, link in firs...</li><li><a href="https://felo.ai/search">Felo Search - Your Free AI Search Engine</a>: The multilingual AI search engine optimized for discovering and understanding world knowledge. Leverage the power of ChatGPT and AI Agent to break language barriers and access global information with ...</li><li><a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo">OpenAI’s SearchGPT demo results aren’t actually that helpful.</a>: The trend of hallucinations showing up in public AI demos continues. As noted by a couple of reporters already, OpenAI’s demo of its new SearchGPT engine shows results that are mostly either wrong or ...</li><li><a href="https://x.com/apostraphi/status/1816869114645807466?s=46&t=JsxhFTRLBknd8RUv1f73bA">Tweet from Phi Hoang (@apostraphi)</a>: Keep your eyes on your own path. That&#39;s where the magic happens.</li><li><a href="https://www.perplexity.ai/search/generate-an-image-that-shows-c-kyD2BVGoQuaGlU1ysFz9KA">Generate an image that shows “chill beach vibe”</a>: I apologize, but I do not have the capability to generate, create, edit, manipulate or produce images. I can only perceive and analyze existing images. I...</li><li><a href="https://www.perplexity.ai/hub/faq/what-is-perplexity-pro">What is Perplexity Pro?</a>: Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-3gIrXAlxSmCaySfyep1WdQ">please respond exactly with whole complete input file wrapped in a markdown...</a>: Here&#x27;s the complete input file wrapped in a markdown code block: const Koa = require(&#x27;koa&#x27;); const multer = require(&#x27;@koa/multer&#x27;); const Router =...</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-FeTeHpErT4WDH_0Env9nVA">please respond exactly with whole complete input files wrapped in a markdown...</a>: Here are the complete input files wrapped in markdown code blocks, as requested:  First, part1: const Koa = require(&#x27;koa&#x27;); const multer =...</li><li><a href="https://www.perplexity.ai/search/please-respond-exactly-with-wh-yd1cfTcxTQmkMlOVUtkAaA">please respond exactly with whole complete input file wrapped in a markdown...</a>: Here&#x27;s the complete input file wrapped in a markdown code block: const Koa = require(&#x27;koa&#x27;); const multer = require(&#x27;@koa/multer&#x27;); const Router =...</li><li><a href="https://www.perplexity.ai/search/do-some-napkin-math-confirm-as-HfKTtQjXSoyDW75d9yV6Yg">do some napkin math:
confirm assumption that 0.75 word roughly equals 1 token...</a>: Let&#x27;s do some napkin math to answer your questions:  1. Confirming the assumption that 0.75 words roughly equals 1 token in modern LLMs:  This assumption...</li><li><a href="https://www.perplexity.ai/search/create-plan-how-germany-would-AZEtxrTwTeaYcG.yfGtP_w">create plan how germany would handle a kaiju invasion. i am making a fiction...</a>: Here&#x27;s a detailed, bureaucratic plan for how Germany might handle a kaiju invasion in your fictional novel, based on the information provided:  1. The kaiju...</li><li><a href="https://www.perplexity.ai/search/create-plan-how-germany-would-AZEtxrTwTea">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/respond-in-very-pessimistic-an-JwtojmgNQWS6bxqh6LQyDQ">Respond in very pessimistic and snarky way, use language creatively.
Write...</a>: Oh joy, let me just contain my excitement as I regale you with the utterly thrilling tale of Venus, the most mind-numbingly dull planet in our solar system....</li><li><a href="https://www.perplexity.ai/search/analyze-and-verify-biggest-cla-lR2cAPV6QPuGSS2uPA44tg">analyze and verify biggest claims from this blog post, always add sources</a>: Based on the information provided and additional sources, I&#x27;ll analyze and verify some of the key claims from the blog post about using Haskell in production...</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-3mf5YSf3QDKpMosaFBBQ5Q">bureaucracy v3</a>: This is the reply #1  Based on the instructions AND search results, to respond to the query about creating a plan for how Germany would handle a kaiju...</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-2MKinl_3TwihXocb73IUsw">mocking bureaucracy</a>: This is the reply #1   Based on the instructions AND search results, to reply to the query &quot;create plan how germany would handle a kaiju invasion. i am making...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1266520903731384420)** (15 messages🔥): 

> - `Dyson OnTrac Headphones`
> - `Himalayas Zombies`
> - `Cultural Insults`
> - `RMS and LUFS Calculation`
> - `German Bureaucracy in Kaiju Invasion` 


- **Dyson enters the audio market**: Dyson has launched its new OnTrac headphones, priced at **$500** and designed for high-end audio performance, moving away from the previous Zone model focused on air purification.
   - The headphones feature **40mm neodymium drivers** and advanced noise cancellation technology that reduces ambient noise by up to **40 dB**.
- **Himalayan Zombies vs Western Zombies**: In Tibetan culture, **ro-langs** are the unique portrayal of zombies, diverging significantly from typical Western interpretations, reflecting local beliefs.
   - This cultural understanding offers insights into the spiritual narratives present in Himalayan mythology.
- **Origins of the terms 'khokhol' and 'katsap'**: The derogatory terms **'khokhol'** and **'katsap'** have historical roots, with **'khokhol'** originally describing a traditional Cossack hairstyle, whereas **'katsap'** may derive from various linguistic origins suggesting a negative connotation towards Russians.
   - These terms have evolved into insults in contemporary Ukrainian and Russian discourse, often reflecting historical animosities.
- **Calculating RMS and LUFS**: The Root Mean Square (RMS) is calculated by taking the square root of the average of squared amplitude samples, whereas LUFS also involves specific normalization related to loudness.
   - The understanding of these calculations is essential for audio professionals in mastering and engineering sound.
- **Germany's Kaiju Invasion Plan**: A detailed plan to handle a **200m tall kaiju invasion** in Germany is being crafted, with an emphasis on the bureaucratic processes involved.
   - The scenario humorously highlights the complexities and challenges of response protocols inherent in German governance during an unexpected disaster.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/_vBz9ntptbs">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/create-an-image-of-mountain-pa-Lps5fAboQUm3VLPK6xDtkQ">Create an image of Mountain panda who is wandering in Himalayas</a>: I&#x27;m unable to create images directly. However, I can help you visualize a scene or provide a description that you could use to create or commission an...</li><li><a href="https://www.perplexity.ai/search/what-is-the-main-caveat-of-hav-5Vb1MqG5RL.SOZPLOewTYQ">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-QNPzayqVSqm3d8H8pyPJVQ">what is the difference between Perplexity, ChatGPT, Copilot, Gemini and Google?</a>: The differences between Perplexity, ChatGPT, Copilot, Gemini, and Google can be understood through their functionalities, integrations, and target use cases....</li><li><a href="https://www.perplexity.ai/search/what-do-companies-need-from-an-8mZVNFpZSPO9hDtdoxYfJw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/how-much-is-the-world-wide-deb-_BpkaqplRZ.mLmVHWECeAw#0">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/tensorflow-ztkJ7ClVTfmoSPjSxeNYIA">tensorflow</a>: TensorFlow is an end-to-end open source platform for machine learning developed by the Google Brain team. It has a comprehensive, flexible ecosystem of tools,...</li><li><a href="https://www.perplexity.ai/search/instructions-always-trigger-se-Qko5CS6NS4GMPZExzBuXIg">memoryBarrierShared(); vs barrier();</a>: Based on the instructions AND search results, I will carefully fact-check the statement &quot;memoryBarrierShared(); vs barrier();&quot; and provide a detailed...</li><li><a href="https://www.perplexity.ai/search/zvidki-vzialis-obrazlivi-slova-9YD8YR51SQutrHpuDiBPjw">Звідки взялись образливі слова &quot;хохол&quot; і &quot;кацап&quot;?</a>: Слова &quot;хохол&quot; і &quot;кацап&quot; мають досить давнє походження, але їх значення з часом змінювалось і набувало негативного відтінку.  Слово &quot;хохол&quot; походить від...</li><li><a href="https://www.perplexity.ai/search/what-is-the-difference-between-QNPzayqVSqm3d8H8pyPJVQ,">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/new-dyson-s-customizable-headp-UuXhBbEjTaCyrYbDDdgPsQ">Dyson Launches Customizable Headphones</a>: Dyson, known for its innovative home appliances, has entered the premium audio market with its new OnTrac headphones, offering extensive customization options...</li><li><a href="https://www.perplexity.ai/search/what-is-the-exact-formula-for-08XP4LY6Tyy2ArgUfyovHA">What is the exact formula for calculating RMS and LUFS?</a>: The exact formulas for calculating RMS (Root Mean Square) and LUFS (Loudness Units Full Scale) are as follows:  1. Calculate the Absolute Values:    - Take...</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-2MKinl_3TwihXocb73IUsw">mocking bureaucracy</a>: This is the reply #1   Based on the instructions AND search results, to reply to the query &quot;create plan how germany would handle a kaiju invasion. i am making...</li><li><a href="https://www.perplexity.ai/search/to-response-to-query-create-pl-3mf5YSf3QDKpMosaFBBQ5Q">bureaucracy v3</a>: This is the reply #1  Based on the instructions AND search results, to respond to the query about creating a plan for how Germany would handle a kaiju...</li><li><a href="https://www.perplexity.ai/search/himalaya-zombies-zombies-in-hi-C0Znuz9HTwuWwnt7lYdlTQ">Himalaya Zombies - zombies in Himalayan cultures, distinct from the typical...</a>: The concept of zombies exists in various cultures around the world, each with its unique characteristics and folklore. In the Himalayan region, particularly...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1266503952866742322)** (18 messages🔥): 

> - `Perplexity API Performance`
> - `Model Comparison`
> - `Web Search Feature`
> - `Automatic Top-Up Feature`
> - `Model Update Issues` 


- **Perplexity API shows inconsistencies**: Users reported performance differences between the **web** and **API** versions of Perplexity, with the web version providing better results.
   - A member noted that the API's `llama-3-sonar-large-32k-online` model had issues returning accurate addresses, suggesting prompt structure affects results.
- **Questions about model usage for accurate results**: To get the most similar up-to-date answers as in Perplexity's UI, users suggest using the `llama-3-sonar-large-32k-online` model.
   - Participants also discussed the expected differences between **large** and **small** models for performance when handling requests.
- **Automatic Top-Up Feature for API Requests**: A user inquired about the **automatic top-up** feature, questioning if a minimal balance would prevent low balance responses during high-volume requests.
   - Another participant confirmed a **20 requests per minute** rate for online models, advising the use of a rate limiter for API interaction.
- **API model performance changes**: A user noticed a drop in quality, suspecting the API model changed the previous week, leading to hallucinations and incorrect citations.
   - Feedback indicated a consistently high performance until this suspected update, raising concerns over model reliability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://www.locksmith-boulder-co.com/">Locksmith Boulder CO - Fast Local Service - Call (720) 961-5060</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1266810854742163518)** (3 messages): 

> - `ChatBoo voice calling feature`
> - `DigiCord AI Assistant launch`
> - `Enchanting Digital testing phase` 


- **ChatBoo showcases voice calling**: The [ChatBoo Update July](https://youtu.be/GNw1EfhjsSw) video reveals an exciting new voice calling feature for the app, aimed at enhancing user interaction.
   - The team encourages users to reach out and try the app functionalities.
- **DigiCord offers all-in-one AI solution**: The [Introducing DigiCord](https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc) video presents an AI assistant in Discord featuring 40+ LLMs including **OpenAI GPT-4**, **Gemini**, and **Claude**.
   - DigiCord is described as a comprehensive tool that also includes image models like **Stable Diffusion**.
- **Enchanting Digital invites testers**: Enchanting Digital invites users to join their testing phase at [enchanting.digital](https://www.enchanting.digital), focusing on quality chat and AI built around a solid RP engine.
   - They promise **lightning fast** and realistic generations with the ability to chat with anyone seamlessly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/GNw1EfhjsSw">ChatBoo Update July</a>: Quick update for July, we are really excited to show off our new voice calling. Please feel free to reach out or to try the app.</li><li><a href="https://youtu.be/e4TIdoWksiQ?si=XihTYJsH0s1MiCRc">Introducing DigiCord - The most useful ALL-IN-ONE AI Assistant in Discord</a>: ✨http://DigiCord.Site - an ALL-IN-ONE AI assistant in Discord - with 40+ LLMs (OpenAI GPT-4, Gemini, Claude, Meta, etc), vision models, stable diffusion,...N...</li><li><a href="https://www.enchanting.digital">Enchanting Digital - Uncensored AI Chat Companion And Digital Art</a>: Imagine a cutting-edge AI chat companion website that offers an unparalleled level of customization, allowing you to create uncensored characters and Digital Art as realistic or fantastical as you des...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1266486769323343994)** (342 messages🔥🔥): 

> - `OpenRouter API issues`
> - `Model recommendations`
> - `Image generation services`
> - `Roleplay model prompts`
> - `Integration inquiries` 


- **OpenRouter API encountering errors**: Users reported receiving a **500 Internal Server Error** while interacting with OpenRouter, highlighting current service issues.
   - Minor hiccups in API functionality were noted, with incidents tracked on the OpenRouter status page.
- **Recommendations for AI models**: For roleplay purposes, users discussed trying **Llama 3.1 405B** and suggested alternatives like **Claude 3.5 Sonnet** or **gpt-4o mini** for better performance.
   - While **DeepSeek Coder V2** was mentioned for coding tasks, concerns about its slower speed compared to other models were raised.
- **Image generation alternatives to OpenRouter**: Users inquired about services similar to OpenRouter for image generation, leading to recommendations like **fal.ai** for text-to-image and video generation.
   - The lack of integration with **ComfyUI** on these platforms was highlighted as a drawback.
- **Challenges with roleplay models**: Concerns were expressed regarding the limitations of **Llama 3.1** for roleplay without a magical 'assistant prefill' prompt, and the need for specialized fine-tuned models like **Lumimaid**.
   - Users were advised to manually add prompts or seek assistance from the **SillyTavern Discord** community.
- **Seeking development opportunities**: A user reached out to inquire if anyone was looking for a developer, indicating interest in potential collaborations.
   - This opens a discussion about project needs and potential contributions from developers in the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1091220969173028894/1195014798837043240/1266852701686202389">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://discordapp.com/channels/1091220969173028894/1195014798837043240/1266847346105520129">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/nani__ooo/status/1816935034399735851">Tweet from NANI ⌘ (@nani__ooo)</a>: Today we are releasing an uncensored fork of the @Meta frontier AI: 𝚖𝚎𝚝𝚊-𝚕𝚕𝚊𝚖𝚊/𝙼𝚎𝚝𝚊-𝙻𝚕𝚊𝚖𝚊-𝟹.𝟷  This work builds on breakthrough research on AI alignment or &#34;jailbreaking&#34; m...</li><li><a href="https://deepinfra.com/chat">LLaMa Chat | Text Generation Machine Learning Model | Deep Infra</a>: Discover the LLaMa Chat demonstration that lets you chat with llama 70b, llama 13b, llama 7b, codellama 34b, airoboros 30b, mistral 7b, and more! </li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>: Manage your keys or create new ones</li><li><a href="https://openrouter.ai/docs/integrations">Integrations (Beta) | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://openrouter.ai/docs/parameters-api">Parameters API | OpenRouter</a>: API for managing request parameters</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: Manage your credits and payment history</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Meta: Llama 3.1 8B Instruct (free) by meta-llama</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This 8B instruct-tuned version is fast and efficient.  It has demonstrated strong performance compared to ...</li><li><a href="https://www.perplexity.ai/">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://huggingface.co/NeverSleep/Lumimaid-v0.2-123B">NeverSleep/Lumimaid-v0.2-123B · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e6bceq/new_geminitest_in_chatbot_arena_is_good/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/docs/requests">Requests | OpenRouter</a>: Handle incoming and outgoing requests</li><li><a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>: Configure parameters for requests</li><li><a href="https://github.com/search?q=repo%3AMintplex-Labs%2Fanything-llm%20max_tokens&type=code">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1266475547610054779)** (70 messages🔥🔥): 

> - `CUDA installation issues`
> - `Mojo playground feedback`
> - `New hardware support for Mojo`
> - `Mojo Community Meeting`
> - `Mojo version management` 


- **Frustrations with CUDA Installation**: A member expressed their struggles with mismatched CUDA versions while trying to use Mojo for LIDAR data tasks, leading to installation headaches and frustrations.
   - Others suggested using the official website for CUDA installations over `apt install` to reduce issues.
- **Mojo Playground Needs Improvements**: Feedback was provided regarding the Mojo playground, specifically requesting better auto-indentation especially for control structures and the addition of a dark mode.
   - Another user noted that dark mode is already available by clicking the sun or moon icon.
- **Interest in Mojo Support for New Hardware**: Discussions emerged around how to add Mojo support for hardware like Tensor Torrent chips, with mention of MLIR and developer kits as starting points.
   - Links to existing guides and documentation for interfacing with MLIR were shared to assist those interested in targeting new architectures.
- **Overview of Mojo Community Meeting**: The Mojo Community Meeting featured presentations on GPU programming and async Mojo, with recordings available on YouTube for those who missed it.
   - Participants eagerly engaged with topics related to Mojo's development and future enhancements.
- **Mojo Version Management Suggestions**: A user suggested that the Mojo CLI should allow switching between stable and nightly versions more easily, ideally using different names or paths.
   - Concerns were raised about the current user experience with managing installations, especially when configuration files aren't very accommodating.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>: Learn how to use low-level primitives to define your own boolean type in Mojo.</li><li><a href="https://github.com/modularml/mojo/issues/3308">[Docs] Mojo URL leads to 404 · Issue #3308 · modularml/mojo</a>: Where is the problem? https://github.com/modularml/mojo What can we do better? The URL displayed on GitHub for Mojo in the upper right is no longer valid. Please replace this link with something be...</li><li><a href="https://modul.ar/community-meeting-zoom">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1266516037701210112)** (1 messages): 

> - `Mojo/MAX Alpha Test`
> - `Magic CLI`
> - `MAX Tutorials Page` 


- **Mojo/MAX Alpha Test Begins**: An alpha test for installing **Mojo/MAX** via the conda ecosystem has commenced, introduced alongside a new CLI tool called `magic`.
   - Installation instructions are available at [installation instructions](https://modul.ar/magic-alpha-doc).
- **Introducing Magic CLI for Conda**: The `magic` CLI allows users to install Python dependencies and share projects more reliably, marking a significant advancement in the installation process.
   - Feedback and issues can be reported through [this link](https://modul.ar/raise-magic-issue).
- **Launch of MAX Tutorials Page**: A new **MAX tutorials page** has been launched to provide step-by-step guides on using MAX APIs for various deployment strategies.
   - Users can access the tutorials at [MAX Tutorials](https://docs.modular.com/max/tutorials), featuring guides such as deploying with Kubernetes and AWS CloudFormation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/magic-alpha-doc">Magic🪄 + Conda Alpha Release Documentation</a>: Magic🪄 + Conda Alpha Release Documentation Introduction We are excited to announce the alpha release of MAX on Conda along with our new package manager called Magic 🪄, which will supersede Modular C...</li><li><a href="https://modul.ar/raise-magic-issue">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://docs.modular.com/max/tutorials">MAX Tutorials | Modular Docs</a>: Step-by-step programming guides using MAX APIs
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1266475556241936414)** (146 messages🔥🔥): 

> - `FFT library integration`
> - `Linked list implementation`
> - `Mojo interop with C/C++`
> - `Binding C libraries`
> - `Multithreading issues` 


- **Challenges with FFTs in Mojo**: Users are seeking optimized FFT libraries like FFTW or RustFFT for use in Mojo, but face challenges with current bindings.
   - One user shared a GitHub link where another member had previously attempted an FFT implementation in Mojo.
- **Linked List Implementation Feedback**: A user shared their successful implementation of a linked list in Mojo, asking for feedback on potential memory leaks and debug issues.
   - They provided a GitHub link to their code and specifically requested help regarding deletion and memory management.
- **Future of C/C++ Interop in Mojo**: Discussion on the future focus of Modular regarding C interop capabilities indicates a potential development timeline of approximately one year.
   - Users expressed frustration over needing access to gated libraries typically written in C or FORTRAN and highlighted the complexity of C++ interop.
- **Improved C Interop and Pointers API**: The recent changes to the pointers API have improved interop with C, though manual binding remains a time-consuming task.
   - Users noted that while interop has become somewhat easier, issues persist with multithreading, particularly involving the pthread library.
- **Multithreading Difficulties with Mojo**: There are challenges in using pthreads and calling Mojo functions from multiple threads, indicating a need for better support.
   - Users currently struggle with the complexity of these issues, underscoring the demand for enhancements in multithreading capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/benchmark/memory/clobber_memory">clobber_memory | Modular Docs</a>: clobber_memory()</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/commit/6f9170f2b63b3e6cbda4c994c99508f3ccd35309">[Docs] Changelog for `DTypePointer` removal. · modularml/mojo@6f9170f</a>: MODULAR_ORIG_COMMIT_REV_ID: 5bab8ecf43554f3997512d52797fbaa843dbaaab</li><li><a href="https://github.com/modularml/mojo/blob/ffe0ef102f52f06a448e452292863e8d68306d8e/stdlib/src/memory/__init__.mojo">mojo/stdlib/src/memory/__init__.mojo at ffe0ef102f52f06a448e452292863e8d68306d8e · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/MVPavan/mojos/blob/master/learn/dsa/my_linked_list.mojo">mojos/learn/dsa/my_linked_list.mojo at master · MVPavan/mojos</a>: Collection of mojo codes. Contribute to MVPavan/mojos development by creating an account on GitHub.</li><li><a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>: Welcome to the MAX quickstart guide!</li><li><a href="https://docs.modular.com/max/tutorials">MAX Tutorials | Modular Docs</a>: Step-by-step programming guides using MAX APIs</li><li><a href="https://docs.modular.com/mojo/manual/get-started">Get started with Mojo🔥 | Modular Docs</a>: Install Mojo now and start developing</li><li><a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/#contributing)">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/nn/model.mojo#L52)">basalt/basalt/nn/model.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/autograd/graph.mojo#L14).">basalt/basalt/autograd/graph.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://github.com/basalt-org/basalt/blob/44a7b1a19797b795fca2b624faa9f1de7d72968c/basalt/nn/model.mojo#L119)">basalt/basalt/nn/model.mojo at 44a7b1a19797b795fca2b624faa9f1de7d72968c · basalt-org/basalt</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

jack.clayton: Thanks this will be fixed on the next website push
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1266486071634432111)** (50 messages🔥): 

> - `TPU Reverse Engineering`
> - `Decoder-only LLMs`
> - `ACL Conference Attendance`
> - `Llama 3.1 Quantization Impact`
> - `Local Embedding Models` 


- **TPU Chips Not Yet Reverse Engineered**: A member inquired whether any recent TPU or NPU chips have been decapped or reverse engineered, noting a lack of detailed layout shots.
   - Another member indicated that while the first half of the information was available, there hasn't been a reverse engineer yet.
- **Exploring Shared Feedforward Parameters in LLMs**: Discussion arose about a paper that trained a decoder-only LLM while sharing feedforward parameters, which someone suggested might be akin to techniques used in Albert.
   - A link to a related [arXiv paper](https://arxiv.org/abs/2309.01826) was shared, emphasizing the efficiency of reducing model parameters.
- **ACL Conference Socializing Plans**: Members discussed their plans to attend the ACL conference, with several expressing interest in connecting at the event.
   - One member mentioned creating a social thread to facilitate meetups closer to the date of the conference.
- **Quantization Concerns in Llama 3.1**: Concerns were raised regarding the performance degradation of Llama 3.1 due to quantization, with one member sharing an [X.com post](https://x.com/_xjdr/status/1816892492580814856) about better response outcomes with bf16.
   - Discussion also touched on the notion that quantization impacts could stem from the total data amount rather than just the parameter-to-data ratio.
- **Inquiries about Local Embedding Models**: A member sought advice on running a local embedding model and whether fine-tuning those models with synthetic data could be beneficial.
   - This led to discussions on the potential advantages of such fine-tuning approaches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_xjdr/status/1816892492580814856">Tweet from xjdr (@_xjdr)</a>: my personal version of 405B Instruct @ bf16 has pretty different (better) responses than almost all inference providers, especially for long prompts.   I have a feeling L3.1 is going to be very finick...</li><li><a href="https://arxiv.org/abs/2309.01826">One Wide Feedforward is All You Need</a>: The Transformer architecture has two main non-embedding components: Attention and the Feed Forward Network (FFN). Attention captures interdependencies between words regardless of their position, while...</li><li><a href="https://github.com/cameronshinn/tiny-tpu">GitHub - cameronshinn/tiny-tpu: Small-scale Tensor Processing Unit built on an FPGA</a>: Small-scale Tensor Processing Unit built on an FPGA - cameronshinn/tiny-tpu</li><li><a href="https://github.com/wbrown/anthropic">GitHub - wbrown/anthropic: golang interface for Anthropic&#39;s Machine Learning API interfaces</a>: golang interface for Anthropic&#39;s Machine Learning API interfaces - wbrown/anthropic
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1266647923005980722)** (44 messages🔥): 

> - `Iterative Inference in Transformers`
> - `Layer Sharing in Universal Transformers`
> - `Causal Language Models and CoT`
> - `Diffusion Forcing Training Paradigm`
> - `Synthetic Dialogues for Fine-Tuning` 


- **Research Directions on Iterative Inference**: A member inquired about developing research on **iterative inference** in transformers, particularly in relation to **in-context learning** and implicit optimization algorithms, noting familiarity with the [Stages of Inference paper](https://arxiv.org/abs/2406.19384).
   - They expressed interest in how existing methods like gradient descent are used in these contexts but found the papers typically focus on specific algorithms.
- **Challenges of Layer Sharing in Universal Transformers**: A paper discussing **layer-sharing** in Universal Transformers is highlighted, which emphasizes its trade-offs, particularly the reduced parameter count and computational costs ([MoEUT paper](https://arxiv.org/abs/2405.16039)).
   - The authors proposed the **Mixture-of-Experts (MoE)** architecture to combine recent advances for more effective layer sharing in transformer design.
- **Diffusion Forcing: A New Training Approach**: The **Diffusion Forcing** training paradigm, which focuses on denoising tokens with **independent noise levels**, was introduced as a way to improve generative modeling ([Diffusion Forcing paper](https://arxiv.org/abs/2407.01392)).
   - This method uniquely allows for **variable-length generation** and helps manage memory usage throughout training while improving performance.
- **Synthetic Dialogues for Improved Fine-Tuning**: The creation of the **Self Directed Synthetic Dialogues (SDSD)** dataset was announced, comprising guided conversations to enhance instruction following and complex problem solving in language models ([SDSD paper](https://arxiv.org/abs/2407.18421)).
   - The dataset pushes forward the work on multi-turn data by implementing a structure for engaging models like DBRX and Llama 2 70B to simulate more complex interactions.
- **Insights on Reasoning Steps in CoT**: A member pointed out that in **Chain of Thought (CoT)** reasoning, models can produce valid outputs despite incorrect intermediate values, which raises questions about their processing ([source](https://discord.com/channels/729741769192767510/747850033994662000/1258122808752472099)).
   - This discussion reflects on whether models handle **relative scaling** effectively and how modifications can affect reasoning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Hannes">Tweet from undefined</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.15892">MINI-SEQUENCE TRANSFORMER: Optimizing Intermediate Memory for Long Sequences Training</a>: We introduce Mini-Sequence Transformer (MsT), a simple and effective methodology for highly efficient and accurate LLM training with extremely long sequences. MsT partitions input sequences and iterat...</li><li><a href="https://arxiv.org/abs/2407.18421">Self-Directed Synthetic Dialogues and Revisions Technical Report</a>: Synthetic data has become an important tool in the fine-tuning of language models to follow instructions and solve complex problems. Nevertheless, the majority of open data to date is often lacking mu...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>: Previous work on Universal Transformers (UTs) has demonstrated the importance of parameter sharing across layers. By allowing recurrence in depth, UTs have advantages over standard Transformers in lea...</li><li><a href="https://arxiv.org/abs/2407.01392">Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion</a>: This paper presents Diffusion Forcing, a new training paradigm where a diffusion model is trained to denoise a set of tokens with independent per-token noise levels. We apply Diffusion Forcing to sequ...</li><li><a href="https://arxiv.org/abs/2407.14435">Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders</a>: Sparse autoencoders (SAEs) are a promising unsupervised approach for identifying causally relevant and interpretable linear features in a language model&#39;s (LM) activations. To be useful for downst...</li><li><a href="https://x.com/HannesStaerk/status/1817683155903787185">Tweet from Hannes Stärk (@HannesStaerk)</a>: @icmlconf done, more papers to come: tomorrow @BoyuanChen0 and @vincesitzmann join us to discuss their &#34;Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion&#34; https://arxiv.or...</li><li><a href="https://arxiv.org/abs/2407.01178">$\text{Memory}^3$: Language Modeling with Explicit Memory</a>: The training and inference of large language models (LLMs) are together a costly process that transports knowledge from raw data to meaningful computation. Inspired by the memory hierarchy of the huma...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

nullonesix: good observation
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1266648572812726322)** (92 messages🔥🔥): 

> - `lm-eval-harness usage issues`
> - `vllm and HF model comparison`
> - `bigbench task migration`
> - `custom evaluation arguments`
> - `model performance benchmarks` 


- **lm-eval-harness encounters**: Users are experiencing various issues with the `lm-eval-harness`, including needing to pass `trust_remote_code=True` to run models properly.
   - One user shared their Python code to demonstrate how they're invoking models which also prompted questions about handling command-line arguments.
- **Exploring vllm and HF differences**: Discussions highlighted the differences in performance between using `vllm` and older Hugging Face (HF) versions for tasks, particularly in terms of batch size and runtime.
   - While `vllm` may optimize for speed, there seem to be complications with continuous batching affecting overall efficiency in evaluations.
- **Transitioning bigbench tasks**: A migration plan for bigbench tasks was discussed, pointing towards the necessity to specify `bigbench_date_understanding_multiple_choice` instead of just `bigbench_*`.
   - It was suggested that users create grouping configurations for ease of invoking multiple related tasks without listing them individually.
- **Stop words application concerns**: One user raised concerns over stop words in generation kwargs failing to apply correctly, leading to questions about the parsing behavior of the `until` argument.
   - Confirmations were sought regarding the exact stopping behavior expected with multiple stop words, particularly in the context of a `vllm` model.
- **Benchmarking insights**: Users shared links to repositories containing benchmark logs and evaluations for models, emphasizing the practicality of comparing performance across different setups.
   - There was a collective interest in standardizing benchmarks to facilitate direct comparisons between current models and historical performance data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-405B-Instruct-evals">meta-llama/Meta-Llama-3.1-405B-Instruct-evals · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Llama-3.1-8B-Instruct.md">LLM-Benchmark-Logs/benchmark-logs/Llama-3.1-8B-Instruct.md at main · teknium1/LLM-Benchmark-Logs</a>: Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.</li><li><a href="https://github.com/dmahan93/lm-evaluation-harness/blob/add-agieval/lm_eval/tasks/bigbench.py">lm-evaluation-harness/lm_eval/tasks/bigbench.py at add-agieval · dmahan93/lm-evaluation-harness</a>: A framework for few-shot evaluation of autoregressive language models. - dmahan93/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bigbench/multiple_choice">lm-evaluation-harness/lm_eval/tasks/bigbench/multiple_choice at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/zeroshot/_bbh_zeroshot.yaml">lm-evaluation-harness/lm_eval/tasks/bbh/zeroshot/_bbh_zeroshot.yaml at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1266489612797345823)** (53 messages🔥): 

> - `LMSYS Ranking`
> - `Segment Anything Model 2`
> - `Claude Control Vectors`
> - `LLM-Enabled Recommendation Systems`
> - `OpenAI Financials` 


- **LMSYS enters Ranking Finetuning**: Members discussed the recent involvement of **LMSYS** in ranking various finetunes of llama models, questioning the motivations and transparency behind this initiative.
   - Concerns were raised about potential biases, with comments suggesting that ranking might favor those who have connections or offer payments.
- **Launch of SAM 2 for Visual Segmentation**: Meta introduced **SAM 2**, a unified model for real-time object segmentation in images and videos, achieving remarkable performance improvements over its predecessor, **SAM**.
   - The model is open-sourced under an **Apache 2.0 license**, and includes a new dataset for training that comprises approximately **51,000 videos**.
- **Discussion on Claude's Control Vectors**: A bet is ongoing regarding whether **Claude 3.5** uses control vectors, particularly in reference to the 'Golden Gate Bridge' feature's weights in various contexts.
   - The community is actively debating how these potential control vectors impact performance in user interactions.
- **LLM-Enabled Recommendation Systems Insights**: Conversations about **LLM-enabled recommendation systems** highlighted the importance of behavioral data over pure content-based signals for accuracy and personalization.
   - Participants suggested a hierarchy of recommendation signals, placing behavioral insights at the top and emphasizing the role of metadata and reviews.
- **OpenAI's Financial Landscape**: An analysis shared from **The Information** explored **OpenAI's** financial structure, including variable costs associated with free versus paid users.
   - Insights suggested that **OpenAI** needs to account for the significant expenses tied to maintaining a large base of free users who consume substantial variable costs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1816956663821381678">Tweet from lmsys.org (@lmsysorg)</a>: Chatbot Arena update!  @NexusflowX&#39;s Athene-70B, an open-weight model fine-tuned from Llama-3-70B, is now ranked #8 on  the leaderboard with a significant 30+ ELO boost from Llama-3.   We see bala...</li><li><a href="https://x.com/twominutepapers/status/1816951813456998901">Tweet from Two Minute Papers (@twominutepapers)</a>: I made a certain someone from @nvidia hold on to his papers, Two Minute Papers style! 🙌📜  Full video is available here: https://www.youtube.com/watch?v=6Nr0_lZScug</li><li><a href="https://x.com/jiayq/status/1817092427750269348?s=46">Tweet from Yangqing Jia (@jiayq)</a>: People often ask why prices like $2.8/m token for Llama 405B, while being super fast, are still profitable at @LeptonAI. We&#39;ve even been asked by a leading GPU provider!  So, I figured we should s...</li><li><a href="https://x.com/iampatelajeet/status/1817193540407210295?s=46&t=6F">Tweet from Ajeet Patel ✨ (@Iampatelajeet)</a>: 🔴Fun Project Alert  You think you&#39;ve a really great Github profile? c&#39;mon let&#39;s get it roasted..  Created this tiny project which reads your profile, parses to Gemini and returns you a sa...</li><li><a href="https://x.com/swyx/status/1818074658299855262">Tweet from swyx 🌉 back in SF! (@swyx)</a>: Memory Attention: adding object permanence with $50k in compute  @AIatMeta continues to lead Actually Open AI. SAM2 generalizes SAM1 from image segmentation to video, releasing task, model, and datase...</li><li><a href="https://x.com/eugeneyan/status/1796630319745151320),">Tweet from Eugene Yan (@eugeneyan)</a>: Steck senpai suggesting for folks to: • simplify evals to ranking metrics • use small models like 23M BERT</li><li><a href="https://x.com/testingcatalog/status/1817320299991552405?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: &#34;Advanced Voice Mode is on its way!&#34;  This is a new message that most of us will likely start seeing next week.   A couple of other changes from the latest iOS update 👀👀👀 - It might happen ...</li><li><a href="https://x.com/jiayq/status/1817453735444160953?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Tweet from Yangqing Jia (@jiayq)</a>: Accounting report on llama3 tokenomics.  After my initial post, @swyx and @dylan522p had a great follow up question on the llama3 405b profitability. Read the original post here: https://x.com/swyx/st...</li><li><a href="https://x.com/yuchenj_uw/status/1817223820589375752?s=46">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: After training the GPT-2 (2.7B), I delved even &#34;deeper&#34; into the scaling law by training a 7.3B model with @karpathy&#39;s llm.c 🌠  Scaling the model was straightforward, primarily was just s...</li><li><a href="https://share.snipd.com/episode/fd03944b-18e3-49c3-a770-97f3ab11d405">In search of the perfect movie recommendation</a>: In search of the perfect movie recommendation</li><li><a href="https://x.com/iampatelajeet/status/1817193540407210295?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Ajeet Patel ✨ (@Iampatelajeet)</a>: 🔴Fun Project Alert  You think you&#39;ve a really great Github profile? c&#39;mon let&#39;s get it roasted..  Created this tiny project which reads your profile, parses to Gemini and returns you a sa...</li><li><a href="https://x.com/AIatMeta/status/1818055906179105010">Tweet from AI at Meta (@AIatMeta)</a>: Introducing Meta Segment Anything Model 2 (SAM 2) — the first unified model for real-time, promptable object segmentation in images & videos.  SAM 2 is available today under Apache 2.0 so that anyone ...</li><li><a href="https://x.com/mattshumer_/status/1816891950903279827?s=46">Tweet from Matt Shumer (@mattshumer_)</a>: Introducing `llama-405b-to-8b` ✍️  Get the quality of Llama 3.1 405B, at a fraction of the cost and latency.  Give one example of your task, and 405B will teach 8B (~30x cheaper!!) how to do the task ...</li><li><a href="https://x.com/capetorch/status/1816770328334602434?s=46">Tweet from Thomas Capelle (@capetorch)</a>: How are model providers making money serving Llama 405B?  A 8xH100 node costs around 1k per day. It can serve Llama 405B at ~300tok/s (with ten batched requests).  &gt; That&#39;s 26M tokens per day, ...</li><li><a href="https://x.com/helloiamleonie/status/1817455019995578583?s=46">Tweet from Leonie (@helloiamleonie)</a>: Here’s how the LinkedIn Engineering team reduced the error rate from ~10% to ~0.01% when generating structured outputs with LLMs.  Turning natural text into structured outputs is a cool use case for L...</li><li><a href="https://x.com/d_mccar/status/1817681755861651915?s=46">Tweet from Daniel McCarthy (@d_mccar)</a>: A lot of interesting data points from this @theinformation article by @amir.   This gives us a hint as to the contribution margin and CLV of @OpenAI paid subscribers, how much is lost per free user, a...</li><li><a href="https://ai.meta.com/blog/segment-anything-2/">no title found</a>: no description found</li><li><a href="https://manifold.markets/SCS/does-claude-35-have-control-vectors">Does Claude 3.5 have control vector(s) to increase its capabilities?</a>: 37% chance. Yes:  Claude 3.5 (either Haiku, Sonnet, or Opus) has at least one control vector enabled by default, where a &quot;control vector&quot; is the up-regulation of a specific feature&#x27;s we...</li><li><a href="https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1267521922992832533)** (1 messages): 

> - `Llama 3 paper club` 


- **Llama 3 Paper Club Recording Released**: The recording of the **Llama 3 paper club** session is now live! Watch it [here](https://www.youtube.com/watch?v=TgLSYIBoX5U&lc=UgwIT71IbJFIiut0RRp4AaABAg) for insights on the latest discussions.
   - *This session promises to cover key facets of the Llama 3 paper,* so don't miss out on the details discussed.
- **Insights from Llama 3 Discussion**: In the Llama 3 paper club, participants shared valuable insights about the features and improvements of the model.
   - Key highlights from the discussion included *enhanced training techniques* and *performance metrics*.


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1266484963813883908)** (122 messages🔥🔥): 

> - `Cursor IDE`
> - `Context Management`
> - `Generative UI Development`
> - `LLM Project Discussions`
> - `Plugin Development` 


- **Excitement Around Cursor IDE**: Users expressed enthusiasm about the **Cursor IDE**, especially enjoying its capabilities for programming in **Ruby** and managing large changes, with one user noting **144 files changed** during their week of use.
   - There's been talk of potential integrations and improvements, including **collaborative mode** and the desire for a **context plugin API**.
- **Context Management Discussions**: The conversation highlighted the importance of **context management**, with users expressing a strong desire for features that allow better control over context and links in the Cursor IDE.
   - One user mentioned having moved to coding in natural language for ease, comparing it to pseudocode on a spectrum.
- **Generative UI Development Insights**: Participants discussed the concept of **generative UIs**, particularly the idea of building **UIs on the fly** through predefined components, with comparisons made to several models like **Claude** and projects like **websim**.
   - An interest in coding benchmarks, especially between tools like **Sonnet** and **Llama3**, was also noted, suggesting an evolving landscape in AI development.
- **Interest in AI Model Advancements**: The chat mentioned excitement for models like **Llama3** and the implications of innovations like **1-bit quantization** in AI systems for enhancing performance and lowering resource consumption.
   - Participants expressed curiosity about future developments and benchmarks, particularly with larger models like **405B**.
- **Community Engagement and Tools Sharing**: Users have been sharing various tools and plugins they find valuable, including a range of **Neovim** plugins for better interaction and coding experiences.
   - Contributions reflected a collaborative spirit with a focus on enhancing **developer productivity** through shared knowledge and tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cursor.com/blog/problems-2024">Problems for 2024-2025</a>: no description found</li><li><a href="https://sourcegraph.com/blog/the-death-of-the-junior-developer">The Death of the Junior Developer</a>: LLMs are putting pressure on junior tech jobs. Learn how to stay ahead.</li><li><a href="https://github.com/twilwa/crawler.nvim">GitHub - twilwa/crawler.nvim: uses firecrawl, jina, and/or jsondr to render webpages in neovim buffers</a>: uses firecrawl, jina, and/or jsondr to render webpages in neovim buffers - twilwa/crawler.nvim</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1267510570014347294)** (2 messages): 

> - `LlamaIndex Webinar`
> - `LlamaIndex Office Hours`
> - `Retrieval-Augmented Generation (RAG)`
> - `Agentic Strategies` 


- **Join the LlamaIndex Webinar on RAG this Thursday!**: Excited to host a new webinar with CodiumAI this week on **Retrieval-Augmented Generation (RAG)** for code generation, happening this Thursday at **9am PT**. [Register here](https://lu.ma/ka5xtyqo) to learn about enhancing the coding process!
   - RAG is crucial for enterprises adopting code generation to maintain **high code quality and integrity**, building on top of the **LlamaIndex infrastructure**.
- **Sign up for LlamaIndex Office Hours and get free swag!**: LlamaIndex invites users building agents or RAG applications with agentic elements to sign up for **office hours** for a **15-30 minute Zoom chat**. [Fill out the form here](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform) and receive LlamaIndex-branded swag.
   - This is an opportunity for in-depth conversations regarding use-cases where **agentic strategies** apply, not just basic how-to questions which are better served through [official documentation](https://docs.llamaindex.ai/en/stable/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/ka5xtyqo">LlamaIndex Webinar: Using RAG with LlamaIndex for Large-Scale Generative Coding · Zoom · Luma</a>: Retrieval-Augmented Generation (RAG) plays a central role in achieving contextual awareness in AI-generated code, which is crucial for enterprises adopting…</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: Have in-depth questions or feedback for the folks at LlamaIndex? Sign up for our community office hours! We&#39;ll get back to you to set up a 15-30 minute Zoom call to chat. We are particularly inter...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1266509588098711562)** (11 messages🔥): 

> - `Multi-modal RAG`
> - `LlamaIndex Hackathon`
> - `LLM ETL Stack Releases`
> - `AI Data Engineer Role`
> - `RAG Evaluation Methods` 


- **Multi-modal RAG for Text and Images**: In a recent video, it was demonstrated how to use the **CLIP model** to create a unified vector space for text and images, utilizing [OpenAI embeddings](https://link.to.openai) and [Qdrant](https://qdrant.com) as the multimodal vector store.
   - This approach enables effective retrieval of relevant text and is a game-changer for applications that integrate multiple data formats.
- **Join the LlamaIndex Hackathon!**: An exciting opportunity to hack with LlamaIndex for a chance to win **fabulous prizes** is available at [this link](https://link.to/hackathon).
   - Participants can explore innovative solutions and features within the LlamaIndex ecosystem.
- **New Releases in LLM ETL Stack**: LlamaIndex announced two significant releases focused on **structured outputs** and async + streaming capabilities, allowing LLMs to return unstructured data in structured formats like names and dates.
   - The introduction of **LlamaExtract** facilitates efficient schema inference from unstructured files, making data handling simpler for developers.
- **Emerging AI Data Engineer Role**: A new role is emerging in the AI landscape: the **AI data engineer**, essential for bringing context-augmented LLM applications to production by ensuring scalable and reliable data management.
   - This role combines data engineering skills with AI, highlighting its necessity in modern AI implementations.
- **Exploring RAG Evaluation Methods**: A webinar will cover **five different ways** to evaluate RAG systems, demonstrating methods using LLMs as judges, enhancing understanding of system performance.
   - This session aims to equip participants with the skills to effectively assess their RAG applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/Kx2j1B9XOV">Structured Data Extraction - LlamaIndex</a>: no description found</li><li><a href="https://t.co/4Iu6PRfsd3">GitHub - run-llama/llama_extract</a>: Contribute to run-llama/llama_extract development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1266493434953334834)** (93 messages🔥🔥): 

> - `Instrumentation with Custom Span`
> - `Text-to-SQL agent example`
> - `LlamaIndex and RAPTOR usage`
> - `Document conversion to nodes`
> - `Embedding storage in vector databases` 


- **Custom Span Usage in Instrumentation**: A user sought help in creating a custom span with specific properties for a RAG pipeline, highlighting confusion around the usage of span handlers and decorators.
   - They shared their custom span implementation but observed none of the print statements fired, indicating potential issues with event handling.
- **Using Text-to-SQL in LlamaIndex**: Members discussed implementing a text-to-SQL assistant capable of complex queries, with examples provided on configuring an NLP query engine with LlamaIndex.
   - An example showcased how to set up tools and manage query parameters using LlamaIndex's capabilities.
- **RAPTOR Pack in Vector Databases**: A user inquired about saving RAPTOR packs to Pinecone and how to add more documents to an existing pack without losing previous data.
   - The community clarified that each new document could be added in bulk but emphasized the need for periodic re-clustering for data integrity.
- **Converting Documents to Base Nodes**: A member asked how to convert Document objects to base nodes using LlamaIndex and received guidance about leveraging the `get_nodes_from_documents()` method.
   - Examples were shared that included creating a simple node parser and loading documents from a specified directory.
- **Storing Summarization Index Data**: A user sought advice on efficiently storing summary index data similar to vector index data, expressing a need to avoid re-creating indexes on each run.
   - The discussion highlighted the importance of managing pipeline caches to ensure all processed directories are accurately stored.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/">Embeddings - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v1/?h=graphrag),">GraphRAG Implementation with LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: Have in-depth questions or feedback for the folks at LlamaIndex? Sign up for our community office hours! We&#39;ll get back to you to set up a 15-30 minute Zoom call to chat. We are particularly inter...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation">Instrumentation - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/">Hugging Face LLMs - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/agent_runner/query_pipeline_agent/#setup-simple-retry-agent-pipeline-for-text-to-sql>))">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/recurisve_retriever_nodes_braintrust/#load-data-setup>)">Recursive Retriever + Node References + Braintrust - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/bm25_retriever/#load-data>)">BM25 Retriever - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1266652457212182578)** (2 messages): 

> - `Security considerations for paid Llamaparse`
> - `Programmatic deduplication of named entities` 


- **Exploring Security of Paid Llamaparse**: A member inquired about potential **security considerations** when using paid **Llamaparse** compared to the free version.
   - No definitive answers were provided, leaving it unclear whether there are any significant differences in security.
- **Fast Dedupe Techniques for Named Entities**: Another member asked for ways to **programmatically dedupe** a list of named entities without a complex **RAG setup**.
   - The focus was on achieving speed and efficiency in deduplication without the overhead of complicated systems.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1266612620371300402)** (51 messages🔥): 

> - `Open Interpreter Feedback`
> - `AI Integration with Daily Tasks`
> - `Coding and Learning`
> - `Custom Environments in OI`
> - `Agent Zero Discussion` 


- **Open Interpreter Feedback Loop**: Users expressed mixed feelings about **Open Interpreter** as a tool, with some suggesting it works well for extracting data from PDFs and translating text, while others caution about its experimental nature.
   - One user specifically asked about its practicality for tasks like finding and translating scientific literature from Chinese, receiving tips for effective custom instructions.
- **AI Integration to Assist Daily Functioning**: A member outlined their struggle with health issues impacting their ability to use a computer and expressed interest in using **Open Interpreter** for voice-commanded tasks.
   - Community members provided advice on the risks associated with using OI for critical operations and suggested exploring alternatives like speech-to-text engines.
- **Learning to Code for AI Usage**: Several users discussed the value of learning coding skills, with one member feeling discouraged about their coding abilities but wanting to learn more.
   - A suggestion was made that coding knowledge can improve understanding of AI error management and problem-solving approaches.
- **Custom Virtual Environments in OI**: A user presented the idea of implementing custom **venvs** (virtual environments) in **Open Interpreter**, which could enhance functionality for GUI executables.
   - Another user highlighted their progress in this area and the potential need for collaboration to refine the implementation.
- **Agent Zero and OI Discussion**: Interest in **Agent Zero** was shared, referencing a demonstration and its approaches to agentic behavior, showcasing a growing community interest in such projects.
   - Community members expressed their desire to explore how these technologies can work collectively to enhance user capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=t67Sn0RGK54&t=31s">Vimium - The Hacker&#39;s Browser</a>: Vimium is a Google Chrome extension that provides keyboard shortcuts for navigation and control in the spirit of Vim. Check it out here: https://chrome.googl...</li><li><a href="https://tenor.com/view/my-man-rick-and-morty-oh-yeah-mail-man-gif-16450197">My Man Rick And Morty GIF - My Man Rick And Morty Oh Yeah - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/e2b-dev/ai-artifacts">GitHub - e2b-dev/ai-artifacts: Hackable open-source version of Anthropic&#39;s AI Artifacts chat</a>: Hackable open-source version of Anthropic&#39;s AI Artifacts chat - e2b-dev/ai-artifacts</li><li><a href="https://github.com/Soulter/hugging-chat-api">GitHub - Soulter/hugging-chat-api: HuggingChat Python API🤗</a>: HuggingChat Python API🤗. Contribute to Soulter/hugging-chat-api development by creating an account on GitHub.</li><li><a href="https://github.com/blazzbyte/OpenInterpreterUI">GitHub - blazzbyte/OpenInterpreterUI: Simplify code execution with Open Interpreter UI Project with Streamlit. A user-friendly GUI for Python, JavaScript, and more. Pay-as-you-go, no subscriptions. Ideal for beginners.</a>: Simplify code execution with Open Interpreter UI Project with Streamlit. A user-friendly GUI for Python, JavaScript, and more. Pay-as-you-go, no subscriptions. Ideal for beginners. - blazzbyte/Open...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/9124d2c34c444aa897df08befd553cf43c27b803/interpreter/terminal_interface/terminal_interface.py#L155">open-interpreter/interpreter/terminal_interface/terminal_interface.py at 9124d2c34c444aa897df08befd553cf43c27b803 · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1266810766317719723)** (5 messages): 

> - `Ubuntu 22.04 for 01 Desktop`
> - `Wayland vs X11`
> - `Virtual Environments for OpenInterpreter` 


- **Ubuntu 22.04 confirmed for 01 Desktop**: Members confirmed that the recommended Ubuntu version for the **01 Desktop** is indeed **22.04**, with specific instructions for configuration.
   - *X11* is being preferred over *Wayland* for this setup, reflecting user comfort and familiarity.
- **User Preference for X11 Over Wayland**: One member expressed a lack of enthusiasm for *Wayland*, suggesting that it's likely due to not being used to it.
   - The community's current favouring of *X11* highlights an ongoing discussion about desktop environments and user experience.
- **Running OI and 01 in separate virtual environments**: There was a query about whether to run **OpenInterpreter (OI)** in one virtual environment and **01 (desktop version)** in another.
   - Clarification on this practice was sought, indicating a point of confusion regarding setup instructions.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1266866376396767383)** (9 messages🔥): 

> - `Agent Zero`
> - `Groq Mixture of Agents`
> - `Docker Integration` 


- **Agent Zero's Impressive Demo**: The [first demonstration of Agent Zero](https://www.youtube.com/watch?v=C9n8zFpaV3I) showcased capabilities like internal vector DB, internet search, and agent spawning.
   - Community members discussed features like executing in Docker containers and expressed curiosity about potential integration with their tools.
- **Curiosity About Agent Zero's Potential**: Members have shown enthusiasm about Agent Zero’s framework, noting its potential capabilities which include built-in memory management.
   - One member planned to investigate its setup, particularly using VSCode with Docker containers, to replicate some functionalities.
- **Groq's Mixture of Agents on GitHub**: A GitHub repository for the [Groq Mixture of Agents](https://github.com/skapadia3214/groq-moa) was shared, emphasizing its development goals.
   - The project promises contributions in agent-based interactions and is open for collaboration.
- **Debugging with Docker and LLMs**: A member successfully ran a Docker image in debug mode using `chat_llm` and `utility_llm` references for the Ollama models.
   - They highlighted the configurations in the `vscode/launch.json` file that facilitate the debugging process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=C9n8zFpaV3I">Agent Zero 🤖 first demonstration</a>: First public demo of Agent Zero framework.GitHub: https://github.com/frdel/agent-zeroDiscord: https://discord.gg/AQdRvSYX</li><li><a href="https://github.com/skapadia3214/groq-moa">GitHub - skapadia3214/groq-moa: Mixture of Agents using Groq</a>: Mixture of Agents using Groq. Contribute to skapadia3214/groq-moa development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1266734110311710822)** (43 messages🔥): 

> - `Turbo Quantization`
> - `Finetuning Llama3`
> - `Partial Layer Freezing with QLoRA`
> - `Challenges with Tokenization`
> - `Strategies for Embedding Models` 


- **Turbo models likely use quantization**: A member noted that the use of the term **'turbo'** implies the usage of a **quantized version** of the model.
   - *I notice fireworks version is better than together ai version,* indicating a preference for different implementations.
- **Finetuning strategies for Llama3 discussed**: A member expressed interest in how much you can **finetune Llama3**, specifically regarding referencing links and game stats.
   - They aimed for the model to calculate **armor and weapon stats** effectively.
- **Partial layer freezing with QLoRA under scrutiny**: There was discussion regarding the feasibility of using **QLoRA with partial layer freeze**, with suggestions to freeze intermediate layers while tuning others.
   - Concerns were raised about whether *peft recognizes those layers* and if **DPO** can be effective without prior soft tuning.
- **Tokenization issues with ShareGPT datasets**: A member faced challenges with **tokenization** on ShareGPT formatted datasets and needed to adjust the conversation template explicitly.
   - The conversation format using **FastChat template** led to questions about why it isn't set as default for instruction-tuned models.
- **Finetuning strategies for embedding models**: A member sought effective strategies for **finetuning embedding models**, noting that the default chromadb settings yielded poor results.
   - They inquired if anyone has successful methods in enhancing **document selection** quality through tuning.



**Link mentioned**: <a href="https://axolotl-ai-cloud.github.io/axolotl/docs/config.html">Config options – Axolotl</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1267138032038051851)** (7 messages): 

> - `4xH100 FSDP+QLoRA`
> - `CPU RAM Usage`
> - `Math Re-implementation`
> - `Model Weight Distribution Issues` 


- **4xH100 FSDP+QLoRA reduces CPU RAM usage**: The member noted that the integration of **4xH100 FSDP+QLoRA** will significantly reduce CPU RAM usage, making it more efficient. According to the discussion, this is based on comparisons with loading model weights on multiple ranks.
   - Previously, **8 GPUs** required **~1.6TB** of memory due to this setup, but now it is expected to be much more manageable.
- **Clarification on CPU RAM comparison**: There was a query regarding **what the CPU RAM usage** was less compared to, and the member clarified it was about loading model weights across ranks. This relates to using a full node with 8 GPUs vs. just rank 0, indicating a **4x** reduction in memory needs.
   - FSDP+QLoRA aims to optimize the peak system memory requirements regardless of device count.
- **Inquiry on Math Re-implementation**: A member asked if anyone could help re-implement a mathematical function shared in a **Twitter link**. This triggered a discussion regarding aggregation methods if computed in increments of **8k**.
   - Another member questioned whether aggregation should be done by **summation**.
- **Concerns about FSDP handling**: There was confusion about whether **FSDP** adequately manages model weight distribution across GPUs. A member posited that the issue might not be with FSDP but possibly related to **Transformers**.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/1266496987281227828)** (4 messages): 

> - `Atlantis-v0.1-12B`
> - `GGUF uploads`
> - `Nemo 12B finetune`
> - `Hugging Face upload speeds` 


- **Atlantis-v0.1-12B is now available**: [Atlantis-v0.1-12B](https://huggingface.co/invisietch/Atlantis-v0.1-12B) has been released, marked as having sensitive content that may be harmful.
   - This new model is a **Nemo 12B** finetune for RP and creative writing, with GGUFs expected to be uploaded soon.
- **GGUF uploads ongoing but slow**: One user expressed frustration regarding the lack of available GGUFs for the model, stating, *still no ggufs 😦*.
   - In response, a member confirmed that the main model was relinked and that the alternate GGUFs under formats are available.
- **Slow upload speeds causing delays**: The developer shared that they are experiencing slow upload speeds, stating, *HF has decided I need to be uploading this 70GB folder at 120k/s*.
   - This slow speed has led to prolonged upload times, causing visible delays in making the model fully accessible.



**Link mentioned**: <a href="https://huggingface.co/invisietch/Atlantis-v0.1-12B">invisietch/Atlantis-v0.1-12B · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1267283417569951795)** (1 messages): 

> - `Operation Athena`
> - `Collaborative reasoning tasks`
> - `Dataset diversity` 


- **Operation Athena launches reasoning tasks database**: A new database focused on **reasoning tasks** for LLMs has been assembled as part of [Operation Athena](https://operation-athena.repleteai.com/), allowing users to contribute their own tasks.
   - This initiative is supported by **Nous Research**, aiming to enhance AI understanding through diverse datasets that reflect human experiences.
- **Call to action for task contributions**: The initiative encourages community involvement, inviting contributions to the database for a comprehensive list of reasoning tasks in AI.
   - This approach aims to maintain **dataset diversity**, crucial for improving model performance in **real-world applications**.
- **Original concepts from Nous Research**: The foundation for Operation Athena stems from work published by [Nous Research](https://github.com/NousResearch/Open-Reasoning-Tasks/tree/main), which initiated the idea of curating reasoning tasks.
   - The heaviest contributions to the database come from existing resources detailed in their [documentation](https://github.com/NousResearch/Open-Reasoning-Tasks/blob/main/tasks.md) as of **July 28th, 2024**.



**Link mentioned**: <a href="https://operation-athena.repleteai.com/">Operation Athena</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1266934038589866079)** (4 messages): 

> - `early_stopping_patience`
> - `Axolotl configurations` 


- **Understanding early_stopping_patience in Axolotl**: In Axolotl, the `early_stopping_patience: 3` parameter stops training if the validation metric does not improve for **three consecutive epochs**, not sequences.
   - *Early stopping* helps to prevent overfitting by halting training if performance does not improve, making it a crucial part of model training configuration.
- **Configuring early stopping in training**: A YAML configuration example for early stopping in Axolotl is shown with `early_stopping_patience: 3` underlining its role in monitoring defined metrics.
   - This configuration ensures no training occurs for more than **three epochs** if there is no performance improvement on the validation set.



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dbdc3c04-77e2-483b-b215-78fd6732d625)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1266749658936377406)** (48 messages🔥): 

> - `Open Source Contributions to LangChain`
> - `Ollama API for Tool Calling`
> - `ConversationBufferMemory in LangGraph`
> - `Creating Flowcharts for RAG`
> - `LangChain Tutorial Issues` 


- **Open Source Contributions to LangChain**: A member expressed interest in guidance for contributing to LangChain, prompting others to share resources including a [contributing guide](https://python.langchain.com/v0.2/docs/contributing/). Suggestions included improving documentation, code, and integrations as ways to contribute.
   - For beginners, one member recommended reading through the [setup guide](https://python.langchain.com/v0.2/docs/contributing/code/setup/) to understand local repository interactions.
- **Ollama API for Tool Calling**: Members discussed the efficiency of using the Ollama API for creating agents, with one reporting better functionality using `ChatOllama` than `OllamaFunctions`. They noted that it works better for following examples from the LangChain tutorial.
   - There was mention of issues with previous APIs crashing on basic tutorials, specifically involving the Tavily and weather examples.
- **ConversationBufferMemory in LangGraph**: One member sought clarity on how to use the `save_context` method in `ConversationBufferMemory`, querying how to structure inputs and outputs for various message types like `HumanMessage` and `AIMessage`. Others noted the lack of explicit documentation on thread safety in `ConversationBufferMemory`.
   - Advice provided noted that careful structuring of inputs and outputs is necessary to handle different message types effectively.
- **Creating Flowcharts for RAG**: Discussion included recommendations for using Mermaid for flowchart creation, with one member sharing code snippets from LangChain's documentation. It was suggested that this offers good production value for visualizing workflows and processes.
   - A member shared a GitHub project comparing different RAG frameworks, encouraging others to check it out for more insights into RAG applications.
- **LangChain Tutorial Issues**: A beginner user reported encountering a `ConnectError` while trying to follow the LangChain RAG tutorial. Recommendations were made to reproduce official tutorials to better grasp the functionality and troubleshoot issues.
   - Concerns were raised about multiple LLM calls in the JS quickstart, implying potential inefficiencies or misunderstandings in handling LLM interactions within the application.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/contributing/">Welcome Contributors | 🦜️🔗 LangChain</a>: Hi there! Thank you for even being interested in contributing to LangChain.</li><li><a href="https://python.langchain.com/v0.2/docs/contributing/code/setup/">Setup | 🦜️🔗 LangChain</a>: This guide walks through how to run the repository locally and check in your first code.</li><li><a href="https://stackoverflow.com/questions/4974238/javascript-equivalent-of-pythons-format-function">JavaScript equivalent of Python&#x27;s format() function?</a>: Python has this beautiful function to turn this:&#xA;bar1 = &#x27;foobar&#x27;&#xA;bar2 = &#x27;jumped&#x27;&#xA;bar3 = &#x27;dog&#x27;&#xA;&#xA;foo = &#x27;The lazy &#x27; &#x2B; bar3 &#x2B; &#x27; &...</li><li><a href="https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#define-graph">Customer Support</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/">Build an Agent | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/rag/">Build a Retrieval Augmented Generation (RAG) App | 🦜️🔗 LangChain</a>: One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&amp;A) chatbots. These are applications that can answer questions about specific source information. These ...</li><li><a href="https://github.com/oztrkoguz/RAG-Framework-Evaluation">GitHub - oztrkoguz/RAG-Framework-Evaluation: This project aims to compare different Retrieval-Augmented Generation (RAG) frameworks in terms of speed and performance.</a>: This project aims to compare different Retrieval-Augmented Generation (RAG) frameworks in terms of speed and performance. - oztrkoguz/RAG-Framework-Evaluation</li><li><a href="https://github.com/langchain-ai/langchain/issues/16651">Start here: Welcome to LangChain! · Issue #16651 · langchain-ai/langchain</a>: Welcome to the LangChain repo! What&#39;s in this repo Please only open Issues, PRs, and Discussions against this repo for the packages it contains: langchain python package langchain-core python pack...</li><li><a href="https://github.com/langchain-ai/langchain/issues/3536>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2256>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17867>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11734>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/6761>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1266857840342204437)** (7 messages): 

> - `Merlinn AI on-call agent`
> - `AI Copilot for User Acquisition`
> - `Langchain Recipe Bot`
> - `Knowledge Distillation Trends`
> - `AI Analyst Builder Launch` 


- **Merlinn AI on-call agent simplifies troubleshooting**: The team launched [Merlinn](https://github.com/merlinn-co/merlinn), an open-source AI on-call agent that assists in troubleshooting production incidents by integrating with tools like DataDog and PagerDuty.
   - They invite feedback and encourage users to star their [GitHub repo](https://github.com/merlinn-co/merlinn) to support their project.
- **AI Copilot streamlines user promotion**: A new [AI copilot](https://hypespot.pro) has been launched that helps users promote their projects on Twitter effortlessly by suggesting comments for relevant conversations.
   - This tool is aimed at aiding individuals in getting visibility for their products with minimal effort.
- **Conversing with Notion databases made easy**: [Kenzic](https://github.com/kenzic/langchain-recipe-bot) introduced a simple app for having conversations with Notion databases, as detailed in a Medium tutorial.
   - The GitHub repository contains the resources necessary for implementation.
- **Knowledge Distillation advancements discussed**: A recent [blog post by Lightly](https://www.lightly.ai/post/knowledge-distillation-trends) covers trends in Knowledge Distillation, highlighting performance gains from smaller models derived from larger ones.
   - The concept originally introduced by Hinton aims to minimize the KL divergence, enhancing the efficiency of smaller models.
- **AI Analyst Builder goes live on Product Hunt**: Datrics has launched the [AI Analyst Builder](https://bit.ly/4dhK6Km), a no-code tool for creating custom AI analysts, and seeks community support on Product Hunt.
   - Users are encouraged to visit the page and provide feedback to improve the tool continuously.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.lightly.ai/post/knowledge-distillation-trends">Knowledge Distillation Trends</a>: Overview of recent Knowledge Distillation strategies and how to use them alongside Self-Supervised Learning with a focus on Masked Autoencoders</li><li><a href="https://mukullight.github.io/docu/">Home</a>: None</li><li><a href="https://github.com/kenzic/langchain-recipe-bot">GitHub - kenzic/langchain-recipe-bot: Repo for Medium tutorial &quot;Talk to Your Notion Database with LangChain.js&quot;</a>: Repo for Medium tutorial &quot;Talk to Your Notion Database with LangChain.js&quot; - kenzic/langchain-recipe-bot</li><li><a href="https://bit.ly/4dhK6Km"> Datrics AI Analyst Builder - Your custom GenAI solution for analytics and reporting | Product Hunt</a>: AI Analyst Builder enables teams to create custom AI analysts without coding. These analysts answer data questions via a chat interface like ChatGPT. Tailored to specific business processes and data, ...</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️</a>: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️ - merlinn-co/merlinn</li><li><a href="https://hypespot.pro">Hypespot</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1266689548528779295)** (30 messages🔥): 

> - `Billing by API Key`
> - `Multi-Agent Systems Frameworks`
> - `API Issues`
> - `Prompt Tuner Beta Release` 


- **Challenges with Billing by API Key**: A discussion arose regarding the need for separate billing by API key, with members exploring potential solutions like middleware to manage costs distinctly for each key.
   - Participants expressed frustration, noting that there currently isn't a system in place to track this usage effectively.
- **Best Frameworks for Multi-Agent Systems**: Members recommended checking out [LangGraph from LangChain](https://docs.langchain.com/docs), a framework praised for its cloud capabilities and customizability for building multi-agent systems.
   - Furthermore, there was mention of Cohere's API offering extensive multi-step and single-step tool use functionalities that enhance agent capabilities.
- **API Downtime with Error 503**: A user reported an API downtime issue with an error 503 and struggled to check the status due to an inaccessible status page.
   - Another member reassured the community that they were working internally to resolve the issues causing the downtime.
- **Prompt Tuner Beta Feature Release**: Queries were raised about the availability of the 'Prompt Tuner' beta feature on the dashboard, with members acknowledging its recent introduction.
   - Users expressed a general interest in better understanding this feature's implications on API usage.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.cohere.com/>">incident.io - Status pages</a>: no description found</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use">Multi-step Tool Use (Agents)</a>: no description found</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: no description found</li><li><a href="https://docs.cohere.com/docs/implementing-a-multi-step-agent-with-langchain">Implementing a Multi-Step Agent with Langchain</a>: no description found</li><li><a href="https://docs.cohere.com/reference/chat">Chat</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1267031847037177886)** (7 messages): 

> - `Cohere with Oracle APEX`
> - `Cohere API performance issues`
> - `Cohere operational status` 


- **Inquiry on Cohere with Oracle APEX**: A user asked if anyone is using **Cohere** with **Oracle APEX**, seeking insights and experiences related to the integration.
- **Cohere API experiencing slowdowns**: Multiple users reported issues with the **Cohere Reranker API**, noting a sudden slowness and failures.
   - One acknowledged that this was a rare occurrence and shared an [error message](https://status.cohere.com/) indicating that the team is investigating the issue.
- **Cohere service status back to normal**: **Cohere** announced recovery from the previous issues, confirming that all systems are fully operational.
   - A status update highlighted **99.67% uptime** for endpoints and a reassuring message that no ongoing issues are affecting their systems.



**Link mentioned**: <a href="https://status.cohere.com/">Cohere Status Page Status</a>: Latest service status for Cohere Status Page

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1267543982565490741)** (17 messages🔥): 

> - `Cohere's API Benefits`
> - `Innovation in AI Companies`
> - `Web Browsing API Usage`
> - `Search Engine Functionality`
> - `Hype Cycle in AI` 


- **Cohere's API boasts reliability**: A member noted that the **Cohere API** is the only one they've worked with that hasn't experienced downtime, calling it **one of the best enterprise options**.
   - Another member humorously highlighted the implication of favoring Cohere in the face of working for OpenAI.
- **Innovation vs. Similarity in AI Products**: As companies announce new products, a member questioned which ones are truly **innovative** as opposed to just **iterations** of existing offerings.
   - The sentiment reflects the broader industry discussion around whether ongoing announcements are genuinely groundbreaking or just part of the current **hype cycle**.
- **Using Web Browsing Tools in Chat**: Members discussed their ability to utilize web search tools integrated into the **Cohere chat interface** and API for quick access to information.
   - One member successfully created a bot to leverage this capability, indicating it's functionally akin to a **search engine**.
- **Excitement Surrounding New Implementations**: A user expressed enthusiasm for using the new tools during their interview at Cohere, significantly encouraging collaborative testing.
   - The playful tone suggests a light-hearted approach to exploring these new features together, even among non-technical users.
- **Perception of AI's Current Landscape**: One member commented on the industry's **hype cycle** while emphasizing their focus on deriving substantial value from AI models for enterprises.
   - This remark reflects a broader understanding of the challenge to separate effective tools from mere marketing noise in the evolving AI landscape.


  

---



### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1266484549550866605)** (2 messages): 

> - `Blog Posts` 


- **Blogpost Reference**: A member mentioned a **blogpost** but did not provide any specific details or links related to it.
   - Another member made a lighthearted comment by saying *summoned 🪄 ☁️🧍‍♂️☁️*, possibly in response to the mention of the blogpost.
- **User Interaction with the Blogpost**: The interaction surrounding the mention of the **blogpost** included playful engagement from other members.
   - The phrase *summoned 🪄 ☁️🧍‍♂️☁️* suggests a casual or humorous atmosphere in the conversation.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1266505187439280240)** (26 messages🔥): 

> - `GPT-4o Mini`
> - `LMSYS ranking algorithm`
> - `Formatting in Chatbot Arena`
> - `Roleplay and Creative Writing Models`
> - `Zuckerberg's Comments at SIGGRAPH` 


- **GPT-4o Mini takes the lead**: The introduction of [GPT-4o Mini](https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles) is noted as a significant change in the chatbot arena, with claims of enhancing interactions.
   - It's suggested that this model is not just about performance, but also serves as a transparency tool to validate weaker models.
- **LMSYS not a cutting-edge ranking tool**: There's skepticism surrounding LMSYS, with comments stating it merely validates existing models rather than being a leading ranking algorithm.
   - One user emphasized that the examples from the model demonstrate randomness, pointing out that easy questions don't effectively evaluate model performance.
- **Formatting makes an impact**: Discussion highlights the effective use of formatting in chatbot responses, particularly the mastery of list and markdown features that engage users.
   - A member humorously noted their preference for employing hierarchical bullet points, likening it to a widespread preference in human language.
- **Distaste for Roleplay in AI**: A user expressed their reluctance to engage in roleplay or creative writing with AI models, stating they prefer more utilitarian uses.
   - The conversation reflects a divide on application preferences, with some embracing creative use while others resist it.
- **Zuckerberg's candid remarks at SIGGRAPH**: Zuckerberg was noted for making informal remarks, including dropping f-bombs alongside Jensen at SIGGRAPH, signaling a more relaxed atmosphere.
   - The banter included joking requests, showing a light-hearted interaction between the industry leaders.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/gpt-4o-mini_battles">Gpt-4o-mini Battles - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://x.com/tszzl/status/1779608670181171504">Tweet from roon (@tszzl)</a>: the global optimum of human language preference is lists and couplet poetry unfortunately
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1266488321178337412)** (13 messages🔥): 

> - `RBR paper critiques`
> - `SELF-ALIGN paper curiosity`
> - `Apple Intelligence Foundation paper`
> - `RL naming schemes`
> - `iTeC details` 


- **RBR paper glosses over complexities**: A member expressed that the **RBR paper** explains obvious parts while neglecting the more complex issues, particularly with its brief mention of dangerous content within benign requests.
   - They highlighted that while screening out explicit threats like 'Pipe bomb plz' seems straightforward, the nuances are glossed over.
- **Interest in SELF-ALIGN paper**: Another member showed curiosity regarding the **SELF-ALIGN paper**, which is about 'Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision'.
   - They noted it may be related to both the **SALMON** and **RBR** discussions on alignment techniques.
- **Discussion around Apple's AI paper**: Members reacted to the **Apple Intelligence Foundation** paper, indicating it details aspects of **RLHF** and its instruction hierarchy but had mixed feelings about its repository.
   - One member expressed their decision to print it out to evaluate its impact on their opinions about **RLHF**.
- **Critique of RL Naming Schemes**: A member commented on the peculiarities of naming schemes used by **Reinforcement Learning (RL)** researchers, conveying a sense of disbelief.
   - Their reaction underlined a broader sentiment of confusion regarding the terminologies employed within the RL community.
- **iTeC excitement**: There was a brief mention of **iTeC** with one member noting its incredibly detailed write-up.
   - This sparked some excitement in the conversations about the paper's contents and potential implications.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1266571885810942072)** (3 messages): 

> - `Moondream2 hack`
> - `Gold Retriever tool`
> - `Databricks and DSPy` 


- **Moondream2 gets a structured image response hack**: A member revealed they built a hack combining [Moondream2](https://x.com/ErikKaum/status/1787451553319621077) and [OutlinesOSS](https://github.com/OutlinesOSS) that allows users to ask questions about images and receive structured responses.
   - The approach hijacks the text model in Moondream2 while enabling embedding processing through Outlines, promising to streamline the user experience.
- **Introducing the Gold Retriever for ChatGPT**: The [Gold Retriever](https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/) is an open-source tool by Jina that enhances ChatGPT's ability to integrate personalized and real-time data, addressing previous limitations.
   - *Users desire tailored AI interactions*, and Gold Retriever aims to provide improved access to user-specific data while navigating knowledge cut-off challenges.
- **Databricks sees potential with DSPy**: A post shared by Databricks highlights the growing recognition of DSPy, suggesting it's a superior tool for organizations to execute their data strategies.
   - The message invites discussions about various tools, signaling that innovations like DSPy are gaining traction in the industry.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jina.ai/news/gold-retriever-let-chatgpt-talk-to-your-data/">Gold Retriever: Let ChatGPT talk to your data</a>: With Gold Retriever, you can easily enable ChatGPT to store, retrieve, and reason with your data in just a few steps</li><li><a href="https://x.com/ErikKaum/status/1787451553319621077">Tweet from Erik Kaunismäki (@ErikKaum)</a>: image ➡️ json  I&#39;ve built a hack combining @vikhyatk&#39;s moondream2 and @OutlinesOSS   So now you can open up an image, ask something about the image and get a response that is guaranteed to fol...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1266539054036553938)** (2 messages): 

> - `AI Agent Advancements`
> - `Transformers and Compositionality` 


- **Survey on AI Agent Advancements**: A recent [survey paper](https://arxiv.org/abs/2404.11584) examines advancements in AI agent implementations, focusing on enhanced reasoning, planning, and tool execution capabilities.
   - It communicates the current capabilities and limitations of existing systems while suggesting key considerations for future AI agent design, including leadership impact and communication styles.
- **Transformers in AI: Fundamental Questions Raised**: A blog post [worth reading](https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html) emphasizes the study of transformer models' performance on complex tasks, specifically multiplication, linked to deeper questions about their learning capacity.
   - It highlights that models like **Claude** or **GPT-4** produce outputs that convincingly mimic reasoning, raising crucial discussions about their ability to tackle intricate problems across various domains.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.11584">The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey</a>: This survey paper examines the recent advancements in AI agent implementations, with a focus on their ability to achieve complex goals that require enhanced reasoning, planning, and tool execution cap...</li><li><a href="https://www.answer.ai/posts/2024-07-25-transformers-as-matchers.html">Faith and Fate: Transformers as fuzzy pattern matchers – Answer.AI</a>: Are GPT-like models thinking? Unclear. But the Faith and Fate paper (Dziri, 2023) points out they are often “just” pattern matching.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1266811152386756631)** (13 messages🔥): 

> - `Mixture of Agents Optimization`
> - `DSPy without Compiled Pipeline`
> - `Hosting Large Pre-training Data`
> - `Conversational History Aware Agent` 


- **Exploring Mixture of Agents Optimization**: A member proposed the idea of using a mixture of agents optimizer for DSPy, suggesting that one layer of optimization involves selecting parameters and models for a system.
   - They referenced a [related paper](https://arxiv.org/abs/2406.04692) that discusses leveraging multiple LLMs for improved responses and compared their approach to a neural network structure.
- **Using DSPy without Labeled Data**: A member inquired about utilizing DSPy without a compiled pipeline, questioning its usefulness in scenarios without labeled data.
   - Another member confirmed that DSPy can indeed enhance applications, particularly for a RAG system seeking better prompts.
- **Hosting Two Petabytes of Data**: A member sought advice on hosting 2 petabytes of pre-training data for healthcare LLMs, mentioning discussions with the Linux Foundation regarding data usage in low- and middle-income countries.
   - They shared their previous work on ClinicalBERT and linked to a [GitHub issue](https://github.com/ClickHouse/ClickHouse/issues/67296) regarding data hosting solutions.
- **Free Hosting Options for LLM Data**: Responding to a query about hosting large datasets, a member suggested that Hugging Face might allow free hosting without many restrictions.
   - This could be beneficial for projects dealing with significant amounts of health care data.
- **Creating a Conversational History Aware Agent**: A member asked about examples of creating an agent that maintains conversational history.
   - In reply, another member noted that previously users had to manually handle chat history, referencing an example for context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/blob/main/examples/agents/multi_agent.ipynb">dspy/examples/agents/multi_agent.ipynb at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://arxiv.org/abs/2406.04692">Mixture-of-Agents Enhances Large Language Model Capabilities</a>: Recent advances in large language models (LLMs) demonstrate substantial capabilities in natural language understanding and generation tasks. With the growing number of LLMs, how to harness the collect...</li><li><a href="https://docs.google.com/document/d/10j4eTkB_0c6BIMRgUY02L2wWvFWAHuCpkiMZ6dotnkA/edit?usp=sharing">ClinicalBERT Technical Charter Draft 11-29-2022</a>: Technical Charter (the “Charter”) for  ClinicalBERT a Series of LF Projects, LLC  Adopted ___________   This Charter sets forth the responsibilities and procedures for technical contribution to, and o...</li><li><a href="https://github.com/ClickHouse/ClickHouse/issues/67296">Integration with content delivery network for incremental static regeneration · Issue #67296 · ClickHouse/ClickHouse</a>: (you don&#39;t have to strictly follow this form) Company or project name Put your company name or project description here One Fact Foundation Use case A clear and concise description of what is the ...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1266497036883333171)** (17 messages🔥): 

> - `storm.py error`
> - `dspy version update`
> - `Dropbase integration`
> - `Mine Tuning methodology`
> - `DSPy project showcases` 


- **storm.py encounters AttributeError**: A user reported an **AttributeError** when trying to execute **storm.py** due to an undefined attribute in the **dspy** module.
   - Another member suggested updating to a newer version of **dspy** to resolve the issue.
- **Steps to update dspy for Claude support**: Detailed steps were shared to update the **dspy** library, including modifying the **__init__.py** file to include `Claude = dsp.Claude`.
   - This change would enable users to utilize **Claude** functionalities directly in their code.
- **Plans for Dropbase integration**: One member expressed plans to integrate **Dropbase** with **dspy**, aiming to create an extensible GUI for various workflows.
   - Resources were shared on how to leverage **Dropbase** for faster web app development.
- **Showcasing various DSPy projects**: Multiple GitHub links for innovative **DSPy** projects were shared, including methodologies like **Mine Tuning**.
   - Members were encouraged to explore the projects for inspiration and potential improvements in their implementations.
- **Inquiry about DSPy project effectiveness**: Someone inquired about actual projects leveraging **DSPy** that demonstrated significant enhancements in their workflows.
   - The discussion highlighted various projects shared earlier, prompting calls for examples of tangible improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/DropbaseHQ/dropbase">GitHub - DropbaseHQ/dropbase: Dropbase helps developers build and prototype web apps faster with AI. Dropbase is local-first and self hosted.</a>: Dropbase helps developers build and prototype web apps faster with AI. Dropbase is local-first and self hosted. - DropbaseHQ/dropbase</li><li><a href="https://www.dropbase.io/">Dropbase AI | Build Back-Office Software with AI</a>: Dropbase is a prompt-based developer platform for building web apps and back-office operations software, fast and painless. Leave your low-code/no-code frustrations behind.</li><li><a href="https://github.com/stanfordnlp/dsp">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/rawwerks/MineTuning/tree/main">GitHub - rawwerks/MineTuning: Mine-tuning is a methodology for synchronizing human and AI attention.</a>: Mine-tuning is a methodology for synchronizing human and AI attention. - rawwerks/MineTuning</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/jmanhype/dspy-self-discover-framework">GitHub - jmanhype/dspy-self-discover-framework: Leveraging DSPy for AI-driven task understanding and solution generation, the Self-Discover Framework automates problem-solving through reasoning and code generation.</a>: Leveraging DSPy for AI-driven task understanding and solution generation, the Self-Discover Framework automates problem-solving through reasoning and code generation. - jmanhype/dspy-self-discover-...</li><li><a href="https://github.com/chrisammon3000/dspy-neo4j-knowledge-graph">GitHub - chrisammon3000/dspy-neo4j-knowledge-graph: LLM-driven automated knowledge graph construction from text using DSPy and Neo4j.</a>: LLM-driven automated knowledge graph construction from text using DSPy and Neo4j. - chrisammon3000/dspy-neo4j-knowledge-graph</li><li><a href="https://github.com/jmanhype/DSPy-Multi-Document-Agents">GitHub - jmanhype/DSPy-Multi-Document-Agents: An advanced distributed knowledge fabric for intelligent document processing, featuring multi-document agents, optimized query handling, and semantic understanding.</a>: An advanced distributed knowledge fabric for intelligent document processing, featuring multi-document agents, optimized query handling, and semantic understanding. - jmanhype/DSPy-Multi-Document-A...</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI</li><li><a href="https://github.com/RamXX/FSM-Workflow">GitHub - RamXX/FSM-Workflow: Lightweight, async-friendly workflow system with state persistence for Python</a>: Lightweight, async-friendly workflow system with state persistence for Python - RamXX/FSM-Workflow</li><li><a href="https://github.com/jmanhype/Storm">GitHub - jmanhype/Storm</a>: Contribute to jmanhype/Storm development by creating an account on GitHub.</li><li><a href="https://github.com/jmanhype/Storm/blob/main/storm.py">Storm/storm.py at main · jmanhype/Storm</a>: Contribute to jmanhype/Storm development by creating an account on GitHub.</li><li><a href="https://storm.genie.stanford.edu/">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1266477396257800212)** (10 messages🔥): 

> - `OpenCL Out of Memory Error`
> - `Monday Meeting Notes`
> - `ShapeTracker Bounties`
> - `Lean Translation Discussion` 


- **OpenCL Out of Memory Error Improvement**: A member suggested improving the **out of memory error** in OpenCL and linked to a relevant [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/5792/files) by **tyoc213**.
   - *tyoc213* pointed out the potential solution for the error handling related to OpenCL.
- **Highlights from Monday's Meeting**: The Monday meeting discussed various updates, including the removal of **UNMUL** and **MERGE** and the introduction of [HCQ runtime documentation](https://docs.tinygrad.org/developer/hcq/).
   - Other topics included **bounties** related to **MLPerf** benchmarks and advancements in **conv backward fusing** and **scheduler** optimizations.
- **Interest in ShapeTracker Bounty**: A member expressed interest in a bounty focusing on the **mergeability of two arbitrary ShapeTrackers in Lean**, questioning the scope of the task.
   - The member referred to prior discussions about this bounty and engaged regarding its value compared to the reward.
- **Lean Translation of Document**: There was a query about whether the bounty involved translating a document into **Lean**, questioning the compensation for such a task.
   - Another member pointed out previous discussions on **Lean** and suggested that answers might already exist within the chat.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/5792/files">retrieve defined opencl error codes by tyoc213 · Pull Request #5792 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/2218/files)",">view.reshape without symbolic by sahamrit · Pull Request #2218 · tinygrad/tinygrad</a>: @chenyuxyz this is your earlier attempt for reshape without symbolic. I analysed that the increase in time for your change is due to cache misses. Below are some details good part  your change redu...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1266685261618872350)** (21 messages🔥): 

> - `Error with NLL loss during PR`
> - `Understanding nn.Embedding gradients`
> - `Disk device in tinygrad`
> - `PR for better error handling`
> - `Using tinygrad for time series analysis` 


- **Resolving NLL Loss Error in tinygrad PR**: A user shared an error related to adding `nll_loss`, noting that the returned tensor lacks gradients, causing PR failure.
   - Another member mentioned that certain operations used in the loss computation, like CMPNE, are non-differentiable.
- **Clarifying Gradients with nn.Embedding**: A user sought help on `nn.Embedding` gradients, encountering a 'tensor has no grad' error in their model.
   - A reply clarified that `requires_grad=True` is unnecessary for index operations to avoid gradient issues.
- **Explanation of Disk Device Functionality**: A user inquired about the disk device in tinygrad, questioning its role in computations.
   - It was explained that the disk device is utilized for tensor memory mapping, primarily for transferring data, not for computational operations.
- **Proposal for Enhancing Error Handling**: A user suggested that tinygrad should not allow tensors on non-computational backends like disk and seek better error messages.
   - Members discussed the necessity of handling such cases and agreed on contributing a pull request to improve the behavior.
- **Using tinygrad for Time Series Analysis**: A user asked if tinygrad could be applied for time series physiological feature extraction and visualizations, citing slow performance with Matlab.
   - This inquiry indicates interest in leveraging tinygrad's capabilities for more efficient computations in data analysis.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.tinygrad.org/runtime/">Runtime - tinygrad docs</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5769)">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Issues · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/95dda8dadf2970888fc8f494b83a0124eb614aa5/tinygrad/nn/state.py#L13-L58">tinygrad/tinygrad/nn/state.py at 95dda8dadf2970888fc8f494b83a0124eb614aa5 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://gist.github.com/airpods69/2992fc80703f8e15a55d44b3455d9620">Error log when testing cross_entropy loss function</a>: Error log when testing cross_entropy loss function - gist:2992fc80703f8e15a55d44b3455d9620</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5752">Addition of nll_loss and cross_entropy to tensor.py by airpods69 · Pull Request #5752 · tinygrad/tinygrad</a>: tensor: added nll_loss and cross_entropy test_ops: added test for nll_loss and test for cross_entropy   This PR adds negative_log_likelihood and cross_entropy to Tensor. #3891 and #5247 Eg: For neg...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1266719643498844301)** (9 messages🔥): 

> - `Vector Search with Language Models`
> - `SWE-Bench Ultra-Hackathon`
> - `Segment Anything Model 2` 


- **Exploring Vector Search Techniques**: Discussion revealed that for searching verbose text, using a BERT-style model instead of CLIP would be more effective, with suggestions for models from Jina or Nomic.
   - One member noted that CLIP should not be used if images aren't the focus and highlighted Jina's better CLIP-style model as a useful alternative.
- **SWE-Bench Hosts a 6-Day Hackathon Adventure**: A bold experiment is taking place with a 6-day hackathon for **SWE-Bench**, providing participants with **$1,000** in compute resources and opportunities to win cash prizes for improvements.
   - Kickoff is on **August 17** and participants will receive support from notable coauthors, with opportunities for teamwork and prizes for beating benchmarks.
- **Segment Anything Model 2 Released**: The **Segment Anything Model 2** from Facebook Research has been made available on [GitHub](https://github.com/facebookresearch/segment-anything-2), offering code for model inference and links to model checkpoints.
   - Additionally, example notebooks are included to assist users in understanding how to implement the model effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">Tweet from Steve Frey (@stevewattsfrey)</a>: A bold experiment: We&#39;re hosting a 6-day ultra-hackathon for SWE-Bench to push the limits of open-source code generation  - Everyone gets $1,000 in compute provided by @StrongCompute  - Up 50 rese...</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use th...
</li>
</ul>

</div>
  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1267598282008563754)** (1 messages): 

> - `Jamba's long context capabilities`
> - `Developer recruitment`
> - `Enterprise feedback` 


- **Exciting Developments in Long Context Capabilities**: There are a few exciting developments on the way concerning **Jamba's 256k effective length**, with promising results from enterprise customers.
   - The team is eager to engage with developers who are experimenting with long context use cases for further feedback.
- **Developers Wanted for Long Context Projects**: The team is actively seeking **developers** to assist with long context use cases and wants to hear from anyone with feedback on their experiences.
   - In exchange, they promise **credits, swag, and fame** to participating developers.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1267633991289147452)** (2 messages): 

> - `New Members`
> - `Community Engagement` 


- **New Members Join the Chat**: A new member, **artworxai**, announced their arrival, stating 'Just joined!!'.
   - Another member, **akitshudar**, kicked off the conversation with a friendly greeting.
- **Chat Engagement Begins**: The discussion opened with a friendly vibe as members interact with greetings.
   - This welcoming atmosphere sets a positive tone for the community.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1267421734051512411)** (2 messages): 

> - `Google AI Hackathon`
> - `LLM Engineering Opportunities`
> - `Dedupe Methods for Named Entities` 


- **Last Call for LLM Engineers in Google Hackathon**: A team seeks one final LLM engineer to join their innovative project for the upcoming [Google AI Hackathon](https://link.to/hackathon). The project aims to disrupt robotics and education using LLM technology, promising technical complexity and excellent user experience.
   - Candidates should have advanced LLM engineering skills and familiarity with tools like **LangChain** and **LlamaIndex**, with a strong interest in robotics or education tech being a plus.
- **Seeking Fast Dedupe Solutions for Named Entities**: A member inquired about effective methods to programmatically dedupe a list of named entities, seeking fast solutions without a complex RAG setup.
   - The focus is on finding a quick and efficient approach rather than implementing intricate systems to handle duplicates.


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1266748507939930112)** (1 messages): 

> - `Face Recognition Models`
> - `Emotion Detection Libraries` 


- **Discussion on Face Recognition Models**: Members seek recommendations for **machine learning models** and **libraries** suitable for detecting and recognizing faces in videos and images.
   - They emphasize the importance of accuracy and performance in real-time applications.
- **Exploring Emotion Detection Capabilities**: There is an interest in finding solutions that can also identify **emotions** from detected faces in both still images and video content.
   - Participants highlight the need for **integrated solutions** that provide both face recognition and emotion analysis.


  

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

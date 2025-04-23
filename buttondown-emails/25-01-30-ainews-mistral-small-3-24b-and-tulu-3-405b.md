---
id: 43d26a15-9c4e-4391-9331-9f5e01ce3300
title: '[AINews] Mistral Small 3 24B and Tulu 3 405B'
date: '2025-01-31T00:08:47.548368Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-mistral-small-3-24b-and-tulu-3-405b
---

<!-- buttondown-editor-mode: plaintext -->**Open models are all we need.**

> AI News for 1/29/2025-1/30/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **7312** messages) for you. Estimated reading time saved (at 200wpm): **744 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

In a weird twist of fate, the VC backed Mistral ($1.4b raised to date) and the nonprofit AI2 released a small Apache 2 model and a large model today, but they are not in the order that you would expect to go in funding.

First, **Mistral Small 3**, released via their trademark [magnet link](https://x.com/mistralai/status/1884967826215059681?s=46), but thankfully also [blogpost](https://twitter.com/dchaplot/status/1884975427460206649):

![image.png](https://assets.buttondown.email/images/cd4040a3-58ca-4099-aa33-53aa0dea68ab.png?w=960&fit=max)

A very nice 2025 update to Mistral's offering optimized for local inference - though one notices that the x axis of their efficiency chart is changing more quickly than the y axis. Internet sleuths have already [diffed](https://x.com/espadrine/status/1885004488206856638) the architectural differences from Mistral Small 2 (basically scaling up dimensionality but reducing layers and heads for latency):

![image.png](https://assets.buttondown.email/images/eae4fd25-550c-4c9a-9a3d-3a61c7749baa.png?w=960&fit=max)

Their passage on usecases is interesting information as to why they felt this worth releasing:

![image.png](https://assets.buttondown.email/images/b9be05a2-4e6e-48f0-82ce-ae125dc34e31.png?w=960&fit=max)

Next, AI2 released **Tülu 3 405B**, their large finetune of Llama 3 that uses their Reinforcement Learning from Verifiable Rewards (RVLR) recipe (from the [Tulu 3 paper](https://arxiv.org/abs/2411.15124)) to make it competitive with DeepSeek v3 in some dimensions:


![image.png](https://assets.buttondown.email/images/b5a54590-c17c-4025-95c9-821fb1502f6b.png?w=960&fit=max)

Unfortunately there don't seem to be any hosted APIs at launch, so it is hard to try out this [beeg](https://x.com/soldni/status/1885004141564731717) model.




---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Gemini 2.0 Flash

**Model Releases and Updates**

-  **Sakana AI released TinySwallow-1.5B**, a small Japanese language model trained with their new method **TAID (Temporally Adaptive Interpolated Distillation)**, achieving state-of-the-art performance in its size category. The model can run entirely on-device, even in a web browser.  A demo is available to try, as well as [the model](https://twitter.com/SakanaAILabs/status/1884770664353325399) and [GitHub repo](https://twitter.com/SakanaAILabs/status/1884770664353325399). A [self-contained web app](https://twitter.com/SakanaAILabs/status/1884880970790343001) with the model weights is also available for local execution.
- **Mistral AI released Mistral Small 3**, a 24B parameter model, under the Apache 2.0 license, with both base and instruct versions, designed for low latency at **150 tokens/s** and **81% accuracy on MMLU**. It is presented as a competitor to **Llama 3.3 70B**, **Qwen-2.5 32B**, and **GPT4o-mini**.  It is available in [la Plateforme, HF, and other providers](https://twitter.com/omarsar0/status/1884972996575609092), and [blog posts](https://twitter.com/dchaplot/status/1884975427460206649) provide details.  [@ClementDelangue](https://twitter.com/ClementDelangue/status/1884994066129039671) also noted the release and [the availability of base](https://twitter.com/ClementDelangue/status/1884994066129039671) and [instruct models](https://twitter.com/ClementDelangue/status/1884994066129039671). [Ollama](https://twitter.com/ollama/status/1884970144562381165) and [llama.cpp](https://twitter.com/reach_vb/status/1885007847135609224) have released support for it as well.
-   **Alibaba_Qwen released Qwen 2.5 Max**, their largest model yet, achieving performance comparable to **DeepSeek V3**, **Claude 3.5 Sonnet**, and **Gemini 1.5 Pro** with an **Artificial Analysis Quality Index of 79**, trained on **20 trillion tokens**. They also released [Qwen2.5-VL Cookbooks](https://twitter.com/Alibaba_Qwen/status/1884809286288810231), a collection of notebooks showcasing various use cases of Qwen2.5-VL, including computer use, spatial understanding, document parsing, mobile agent, OCR, universal recognition, and video understanding. [The API for the model has been updated](https://twitter.com/Alibaba_Qwen/status/1884995327318782086) to $1.6 / million input tokens and $6.4 / million output tokens.
- **Allen AI released Tülu 3 405B**, an open-source post-training model that surpasses **DeepSeek-V3** in performance, demonstrating that their recipe, which includes Reinforcement Learning from Verifiable Rewards (RVLR) scales to 405B, and performs on par with **GPT-4o**. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1885004067547557987) noted the release as well, highlighting the availability of the models on HF. [@reach_vb](https://twitter.com/reach_vb/status/1884969597473886248) called it a "cooked" release, and noted that it beat DeepSeek V3 while being 40% smaller.
- **DeepSeek-V3 is beaten by Tülu 3**, with  [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1885024960118202538) noting this is achieved with a **405B Llama base**, and that **solid post-training** plays a role. He emphasizes the importance of the fully open-source nature of the recipe.
-   **DeepSeek R1 Distill** is available for free on [Together AI](https://twitter.com/togethercompute/status/1885008866259460474). Together AI also offers a [100% free API endpoint](https://twitter.com/togethercompute/status/1885008864422264997) for this model.

**Tools, Benchmarks, and Evaluations**

-   **LangChain** introduced a [bulk view](https://twitter.com/LangChainAI/status/1885003940661743999) for annotation queues in LangSmith, allowing users to manage large datasets for model training. They also added a [waterfall graph](https://twitter.com/LangChainAI/status/1884987434645041482) to visualize traces, to spot bottlenecks, and optimize response times. A [video was released](https://twitter.com/LangChainAI/status/1885012449352704123) on how to evaluate document extraction pipelines.
-   [@awnihannun](https://twitter.com/awnihannun/status/1884812911572566027) notes that **Qwen 2.5 models can be used to generate or fine-tune code with mlx-lm on a laptop**, reporting the 7B model runs pretty fast on an **M4 Max** using the mlx-lm codebase (16k lines) as context. A [guide on efficiently recomputing the prompt cache](https://twitter.com/awnihannun/status/1884813199347986790) is also available.
-   [@jerryjliu0](https://twitter.com/jerryjliu0/status/1884777070292762723) shared a sneak peek of **LlamaReport**, an agent to create complex, multi-section reports from unstructured data.
-  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1884891721181298847) notes that **sources and reasoning traces** make a massive difference in AI products' UX and trust. He also states that Perplexity will make the **native assistant on phones (android)** accomplish tasks more reliably.  He offered Perplexity Pro for free for one year to all US government employees with a .gov email.
-  [@_akhaliq](https://twitter.com/_akhaliq/status/1884785023569527264) has Perplexity Sonar Reasoning available on ai-gradio with DeepSeek's models. They also released [Atla Selene Mini](https://twitter.com/_akhaliq/status/1884795448139166067), a general purpose evaluation model.
-  [@swyx](https://twitter.com/swyx/status/1884775744917967191) ran their report agent on several models, and concluded that **Gemini 2.0 Flash** was more efficient at abstractive reporting than **O1**, while being **200x cheaper**.
-   [@karpathy](https://twitter.com/karpathy/status/1885026028428681698) explains a textbook analogy for LLMs, comparing **pretraining, supervised finetuning, and reinforcement learning** to textbook exposition, worked problems, and practice problems, respectively.


**AI Infrastructure and Compute**

-   [@draecomino](https://twitter.com/draecomino/status/1885022313260998953) notes that **Cerebras** makes AI instant again with **1 sec time to first token** for **DeepSeek R1 70B**.
-   [@cto_junior](https://twitter.com/cto_junior/status/1884823329477177687) notes that **2000 H100s are good** enough to train a **dense 70B model on 15T tokens in a fiscal year quarter**, costing around **10M$**. He also mentioned that [Yotta has access to 4096 H100s](https://twitter.com/cto_junior/status/1884823329477177687).
-   [@fchollet](https://twitter.com/fchollet/status/1885040378170269889) stated that **$500B number is bogus** for AI, estimating that at most **$150B** is realistic.
-  [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1885042373757198811) argues technology tends to get **cheaper and more efficient**.  He also argues that AI is moving from a world of imitation learning to reward learning.
-   [@teortaxesTex](https://twitter.com/teortaxesTex/status/1884824912818225308) notes that the **R1 drop has led many to conclude "you can just build things."** They state that DeepSeek has done this while **having less compute power** compared to others.
-  [@shaneguML](https://twitter.com/shaneguML/status/1885053092913406236) noted that test-time compute scaling favors fast inference chip startups like **Cerebras and Groq.**

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Mistral Small 3 Released: Competitive with Larger Models**

- **[Mistral Small 3](https://i.redd.it/kj3s0jvr35ge1.png)** ([Score: 643, Comments: 205](https://reddit.com/r/LocalLLaMA/comments/1idny3w/mistral_small_3/)): **Mistral Small 3** is referenced in a tweet by **@MistralAI** dated January 30, 2025, featuring a URL that likely links to resources or details about the release. The tweet has garnered 998 views, indicating interest in the subject.
  - **Mistral Small 3** is a **24B-parameter model** released under the **Apache 2.0 license**, optimized for low latency and efficiency, processing **150 tokens per second**. It's noted for its robust language tasks and instruction-following capabilities, and it's over three times faster than larger models like **Llama 3.3 70B** on the same hardware, achieving over **81% accuracy on MMLU**.
  - Users appreciate the **human evaluation chart** for smaller models, highlighting the importance of aligning models with human perspectives rather than just benchmarks. This model can be fine-tuned for various domains, including legal, medical, and technical support, and is suitable for local inference on devices like **RTX 4090** or **Macbooks with 32GB RAM**.
  - The community is enthusiastic about the **Apache 2.0 licensing**, which allows for wide distribution and modification, and the model's performance compared to others like **Qwen 2.5 32B** and **GPT-4o-mini**. Discussions also include the model's speed and efficiency on different hardware setups, with users reporting speeds of **21.46 tokens/s on RTX 8000** and **24.4 tokens/s on M1 Max 64GB**.


- **[Interview with Deepseek Founder: We won’t go closed-source. We believe that establishing a robust technology ecosystem matters more.](https://thechinaacademy.org/interview-with-deepseek-founder-were-done-following-its-time-to-lead/)** ([Score: 298, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1idtkll/interview_with_deepseek_founder_we_wont_go/)): **Deepseek**'s founder emphasizes their commitment to remaining open-source, prioritizing the development of a robust technology ecosystem over closed-source strategies. The interview suggests this approach is vital for innovation and collaboration in the AI community.
  - **OpenAI and DeepSeek**: Discussions highlight skepticism towards OpenAI's initial open-source intentions, contrasting it with DeepSeek's current open-source strategy. Users express concerns about the potential shift to closed-source once adaptation occurs, as seen with OpenAI.
  - **Hedge Fund Strategy**: There is speculation about DeepSeek's financial strategies, with some users suggesting they operate like a hedge fund by releasing open-source models to influence market valuations, a tactic described as a form of *information-based market manipulation*.
  - **Technical Curiosity**: Interest in DeepSeek's technology is evident, particularly regarding their **FP8 training code**. Users express a desire to access this code to potentially accelerate home-based training, emphasizing the technical community's interest in leveraging open-source advancements for personal projects.


- **[Mistral new open models](https://i.redd.it/5nnsoy4295ge1.png)** ([Score: 128, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1idokcx/mistral_new_open_models/)): **Mistral** has released two new models, **Mistral-Small-24B-Instruct** and **Mistral-Small-24B-Base-2501**, with recent updates and a user interface that includes a search bar and sorting options. The models are part of a collection of 23 available models, with the Instruct model having 50 likes and the Base model having 23 likes.
  - **Mistral Small 3** is highlighted for its competitiveness with larger models like **Llama 3.3 70B** and **Qwen 32B**, being more than **3x faster** on the same hardware and open-source. It's considered an excellent open alternative to proprietary models such as **GPT4o-mini**. More details can be found [here](https://mistral.ai/news/mistral-small-3/).
  - There is curiosity regarding the differences between the **Base** and **Instruct** models, though specifics are not detailed in the comments.


**Theme 2. Nvidia Reduces FP8 Training on RTX 40/50 GPUs**

- **Nvidia cuts FP8 training performance in half on RTX 40 and 50 series GPUs** ([Score: 401, Comments: 93](https://reddit.com/r/LocalLLaMA/comments/1ideaxu/nvidia_cuts_fp8_training_performance_in_half_on/)): **Nvidia** has reportedly reduced **FP8 training performance** by half in the **RTX 40** and **50 series GPUs** according to their new Blackwell GPU architecture whitepaper, with the **4090** model showing a drop from **660.6 TFlops** to **330.3 TFlops** for FP8 with FP32 accumulate. This change may discourage AI/ML training on Geforce GPUs, reflecting a pattern of performance limiting since the Turing architecture while maintaining full performance for Quadro and datacenter GPUs.
  - Many commenters believe the reported halving of **FP8 training performance** in the **RTX 40 and 50 series GPUs** might be a typo in the documentation, referencing the **Ada Lovelace paper** where FP8/FP16 accumulation was confused with FP8/FP32. Some suggest testing with old and new drivers to verify if performance has indeed been altered.
  - There are accusations against **Nvidia** for engaging in anti-consumer practices, with references to **chip etching** and firmware limitations potentially used to restrict performance. Discussions include the possibility of legal actions, comparing this situation to previous cases like Apple's iPhone throttling settlement and Nvidia's GTX 970 false advertising fine.
  - Users highlight the importance of **CUDA** for machine learning tasks, noting difficulties encountered on non-Nvidia hardware like **Apple Silicon**. The discussion also touches on the unhealthy state of the AI/ML GPU market, with comparisons to **Quadro** and datacenter GPUs' full performance capabilities, which are not mirrored in consumer-grade GPUs.


**Theme 3. DeepSeek R1 Performance: Effective on Local Rigs**

- **DeepSeek R1 671B over 2 tok/sec *without* GPU on local gaming rig!** ([Score: 165, Comments: 57](https://reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_671b_over_2_toksec_without_gpu_on/)): The post discusses achieving **2.13 tokens per second** on a **DeepSeek R1 671B** model without using a GPU, instead utilizing a **96GB RAM gaming rig** with a **Gen 5 x4 NVMe SSD** for memory caching. The author suggests that investing in multiple NVMe SSDs could be a cost-effective alternative to expensive GPUs for running large models, as their setup showed minimal CPU and GPU usage, highlighting the potential for better price/performance for home setups.
  - Users discuss the practicality and limitations of using a **2.13 tokens per second** rate, with some expressing that a minimum of **5 tokens per second** is necessary for effective use, and others pointing out that **2k context** is insufficient for certain applications like coding.
  - There is interest in improving performance by stacking **NVMe SSDs** into RAID configurations or using an acceleration card, with a suggestion that for around **$1,000**, one could achieve **60 GBPS** theoretically, enhancing the speed and performance of running large models.
  - Requests for detailed replication instructions and specific command usage indicate community interest in experimenting with similar setups. A user shared a [gist with llama.cpp commands and logs](https://gist.github.com/ubergarm/0681a59c3304ae06ae930ca468d9fba6) to assist others in understanding and replicating the setup.


- **What are you *actually* using R1 for?** ([Score: 106, Comments: 134](https://reddit.com/r/LocalLLaMA/comments/1idgrh4/what_are_you_actually_using_r1_for/)): The author questions the practical utility of **DeepSeek R1 models**, noting their focus on reasoning and generating extensive thought processes, even for simple problems. They express skepticism about the rush to adopt these models for everyday tasks, suggesting they may be more suited for complex problem-solving rather than routine interactions like **GPT-4o**.
  - Users highlight **DeepSeek R1's** utility in various technical tasks, such as coding, math problem-solving, and data analysis. **Loud_Specialist_6574** and **TaroOk7112** find it particularly useful for coding, with **TaroOk7112** noting its ability to convert a script to a newer version without errors on the first try. **No-Statement-0001** describes a complex problem where R1 provided a solution involving a shell script for handling Docker signals.
  - Several users mention the model's effectiveness in **creative and theoretical applications**. **Automatic_Flounder89** and **Acrolith** note its usefulness in theoretical experiments and creative writing, respectively, while **a_beautiful_rhind** appreciates its roleplaying capabilities. **Dysfu** uses it as a teaching assistant for math, enhancing the learning experience by avoiding direct solutions.
  - **AaronFeng47** and **EmbarrassedBiscotti9** discuss challenges with R1, such as logical errors in code and occasional oversight of specifications, but acknowledge its potential for complex tasks. **AaronFeng47** contrasts the experience with other models, finding R1 less reliable than **o1-preview**.


**Theme 4. Mark Zuckerberg on Llama 4 Progress and Strategy**

- **Mark Zuckerberg on Llama 4 Training Progress!** ([Score: 154, Comments: 85](https://reddit.com/r/LocalLLaMA/comments/1id6gcj/mark_zuckerberg_on_llama_4_training_progress/)): **Mark Zuckerberg** emphasizes **Meta's** progress on **Llama 4**, highlighting its potential to lead in AI with its **multimodal capabilities** and upcoming surprises in 2025. He also discusses the success of **Ray-Ban Meta AI glasses** and plans for significant infrastructure investments, expecting **Meta AI** to become a leading personalized assistant used by over **1 billion people**.
  - There is significant interest in **model sizes** and configurations for **Llama 4**. Users express the need for models that fit a range of hardware capabilities, with suggestions for intermediate sizes like **1B**, **3B**, **7B**, and up to **630B** to accommodate various VRAM capacities, avoiding the gap between **7B** and **800B** models.
  - Discussion around **Meta's multimodal capabilities** highlights excitement about native omnimodality, with expectations for models excelling in text, reasoning, visual understanding, and audio. Users are eager for models that support **audio/text**, **image/text**, and **video** capabilities, crucial for applications like vocal assistants and visual synthesis.
  - Comments reflect skepticism about the timeline and strategic decisions of **Meta**. Concerns include the delayed release of **Llama 4**, the focus on fine-tuning post-training, and the potential for a limited range of model sizes. The debate also touches on the broader implications of **Meta's** AI developments in the context of privacy and competition with other tech giants.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. DeepSeek-R1's Impact: Technical and Competitive Analysis**

- **No Hype DeepSeek-R1 [R]eading List** ([Score: 196, Comments: 9](https://reddit.com/r/MachineLearning/comments/1ideupn/no_hype_deepseekr1_reading_list/)): The author shares a **reading list** compiled from their research paper club, focusing on foundational papers in **AI/ML** that lead up to **DeepSeek-R1**. Aimed at providing a deeper understanding of the technology, they invite readers to explore the list on [Oxen.ai's blog](https://www.oxen.ai/blog/no-hype-deepseek-r1-reading-list).
  - **Low rank matrices** approach with attention was discussed, with a question about whether it could be retrofit into existing models using their existing weights.
  - Interest in joining the **research paper club** was expressed, with requests for more information on how to participate.
  - Positive feedback on the **reading list** was shared, with anticipation for the upcoming **Paper Club** meeting.


- **[d] Why is "knowledge distillation" now suddenly being labelled as theft?** ([Score: 256, Comments: 87](https://reddit.com/r/MachineLearning/comments/1idjtta/d_why_is_knowledge_distillation_now_suddenly/)): **Knowledge distillation** is being controversially labeled as theft, despite being a method to approximate transformations by mimicking outputs. The post argues that this label is unfounded since the architecture and training methods differ, and the process does not necessarily replicate the original transformation function.
  - Several commenters highlight the distinction between **copyright law** and **Terms of Service (TOS) violations**, emphasizing that while using outputs from OpenAI models may breach TOS, it does not equate to theft under copyright law. **ResidentPositive4122** notes that OpenAI's documentation clarifies they do not claim copyright on API generations, only that using such data to train other models breaches TOS.
  - Discussion around **OpenAI's reaction** to potential TOS violations suggests a strategic move to maintain their status, with **proto-n** suggesting that OpenAI's claims against DeepSeek are a way to assert their influence and importance in the AI field. **batteries_not_inc** and others argue that OpenAI's response is driven by dissatisfaction rather than legal standing.
  - The debate also touches on broader themes of **regulation and ethics** in AI, with **H4RZ3RK4S3** and others discussing the impact of **EU regulations** and the contrasting perceptions of **US and Chinese tech practices**. **KingsmanVince** and **defaultagi** express skepticism about both US and Chinese approaches, indicating a complex landscape of ethical considerations and public perception.


- **[State of OpenAI & Microsoft: Yesterday vs Today](https://i.redd.it/k79xgml9d6ge1.jpeg)** ([Score: 154, Comments: 27](https://reddit.com/r/OpenAI/comments/1idtwy7/state_of_openai_microsoft_yesterday_vs_today/)): **DeepSeek-R1** is now integrated into **Microsoft Azure services**, marking a shift from previous controversies involving alleged data exfiltration from **OpenAI's API**. The recent launch on **Azure AI Foundry** and **GitHub** highlights the platform's trustworthiness and capabilities, contrasting with earlier security concerns reported by **Reuters**.
  - **DeepSeek-R1** is now available on **Azure**, and users express interest in testing it as an API option. There is skepticism about Microsoft's motives, with some suggesting they are capitalizing on previous controversies.
  - The model is **free and open source**, which is a key reason for its widespread support, despite some users not understanding the distinction between the model and its applications.
  - Discussions include references to Microsoft's historical strategy of **"embrace, extend, and extinguish"**, hinting at concerns about their true intentions behind supporting **DeepSeek-R1**.


**Theme 2. Copilot's AI Model Integration and User Feedback**

- **[o1 now available free of charge in Copilot](https://i.redd.it/r074j9gia1ge1.png)** ([Score: 253, Comments: 56](https://reddit.com/r/OpenAI/comments/1idamb3/o1_now_available_free_of_charge_in_copilot/)): **Copilot** now offers **OpenAI's reasoning model (o1)** free for all users, as announced by Mustafa Suleyman on Twitter. The announcement showcases a conversation about ocean currents, illustrating o1's capability to provide detailed responses, and highlights user engagement metrics.
  - The majority of users express dissatisfaction with **Copilot**, describing it as the "worst" AI for Microsoft products, with several comments highlighting issues related to **wrong answers** and poor integration. There is a sentiment that Copilot's quality has deteriorated, especially since changes made around **August last year**.
  - Some users speculate that the reason for Copilot's perceived decline is due to strategic decisions by **Microsoft and OpenAI** to drive users back to OpenAI subscriptions, or to collect data for future offerings such as "virtual employees." **Microsoft's 49% ownership** of OpenAI is noted as a significant factor in these strategies.
  - Technical issues are blamed on **super long system prompts** and **prompt injections** for "safety reasons," which disrupt model performance. The focus seems to be on corporate users, as **companies** are more comfortable using Copilot with their data, despite the perceived decline in product quality.


**Theme 3. ChatGPT's Latest Updates: User Experience and Technical Changes**

- **[ChatGPT got some nice, incremental updates](https://i.redd.it/gzqfir8465ge1.png)** ([Score: 171, Comments: 61](https://reddit.com/r/OpenAI/comments/1ido8jq/chatgpt_got_some_nice_incremental_updates/)): ChatGPT has received incremental updates in the **GPT-4o model** as of **January 29, 2025**, including an extended training data range for more relevant knowledge, enhanced image analysis capabilities, and improved performance in STEM-related queries. Additionally, the model now responds more enthusiastically to emojis.
  - There is skepticism about the **incremental updates** to **GPT-4o**, with users suggesting that **OpenAI** lifted previous constraints to upsell higher pricing tiers, and some users are noticing a return to the initial quality of responses. The discussion also mentions the anticipation of **o3-mini** as a potential short-term response to current limitations.
  - The use of **emojis** in the new updates has been polarizing, with some users appreciating the enhanced formatting and others finding it excessive and disruptive, especially in professional contexts. One user mentioned the first versions of **Copilot** as a comparison to the current emoji usage.
  - The **"Think" button** feature is discussed, with some users having access to it and noting its potential to add a reasoning chain to **GPT-4o**. However, there is concern about how it might affect message limits, particularly for those with limited quotas.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Exp (gemini-2.0-flash-exp)

**1. DeepSeek's Rise: Speed, Leaks, and OpenAI Rivalry**

- **DeepSeek Defies Expectations, Leaks Data**: DeepSeek models, especially **R1**, show strong reasoning and creative potential, rivaling **OpenAI's o1**, but a database exposure on [Hacker News](https://news.ycombinator.com/item?id=42871371) exposed user data, raising privacy concerns. Despite that, [many see it outpacing OpenAI's performance for creative tasks](https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/) and [code](https://www.codeium.com/changelog).
- **R1 Performance Varies**: **DeepSeek R1 1.58B** runs slow (**3 tokens/s**) on basic hardware, needing **160GB VRAM** or fast storage per [this doc](https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1) for better throughput, but some report **32 TPS** on high-end GPUs. Users also report that the quantized versions can struggle with instruction following.
- **OpenAI and DeepSeek lock horns**: While some note that **OpenAI** [criticized DeepSeek](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA) for its training data, they are also [using DeepSeek internally for data retrieval](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA). This [rivalry has intensified](https://www.youtube.com/watch?v=3QuWqjJ1ZjM), with questions raised about censorship, open access, and data collection practices. 

**2. Small Models Make Big Waves: Mistral and Tülu**

- **Mistral Small 3 Shines Bright**: The new [**Mistral Small 3**](https://mistral.ai/news/mistral-small-3/) (**24B** parameters, **81% MMLU**) is lauded for its low latency and local deployment capabilities, running **3x faster** than competitors per the [official site](https://mistral.ai/news/mistral-small-3/), offering a sweet spot between performance and resource use, and is licensed with Apache 2.0.
- **Tülu 3 Topples Top Dogs**: [**Tülu 3 405B**](https://allenai.org/blog/tulu-3-405B), a **405B** parameter model with open weights, outperformed both **DeepSeek v3** and **GPT-4o** on benchmarks, driven by its **Reinforcement Learning from Verifiable Rewards (RLVR)** approach, with [open post-training recipes](https://allenai.org/blog/tulu-3-405B).
- **Quantization Tradeoffs Discussed**: Developers are experimenting with model quantization, noting that it reduces model size and VRAM usage, but can also **degrade instruction following**, causing users to evaluate its effectiveness on various tasks.

**3. RAG and Tools: LM Studio and Agent Workflow**

- **LM Studio Supports RAG**: [**LM Studio 0.3.9**](https://lmstudio.ai/blog/lmstudio-v0.3.9) now supports **RAG** with local document attachments, described in the [docs](https://lmstudio.ai/docs/basics/rag), allowing documents within context window to be used in chat sessions, and also now supports **Idle TTL** and **auto-update**, which has improved its efficiency.
- **Aider Goes Local With Read-Only Stubs**:  Users are exploring methods to integrate **Aider** with local models like **Ollama** for privacy reasons and the new [YouTube video](https://youtu.be/XE6v_RGe0-U) highlights the use of **read-only stubs** to manage large codebases.
- **LlamaIndex Integrates Advanced Agents**: **LlamaIndex's "Mastering AI Agents Workshop"** has introduced advanced **AgentWorkflow** concepts for multi-agent systems, with robust architectures leveraging LlamaIndex as shown [here](https://t.co/UKIClalkKG). 

**4. Hardware and Performance: GPUs and Optimization**

- **Blackwell's Power Boost**: The new **Blackwell** architecture with **sm_120a** is set to shake up GPU performance, offering stronger compute capability for consumer GPUs, as per [NVIDIA documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md), with discussions highlighting possible **5x** speed boosts in **FP4** tasks on new **RTX 5090**, though some tests show only **2x** gains.
- **PyTorch 2.6 Performance Knobs**: The newly launched [**PyTorch 2.6**](https://pytorch.org/blog/pytorch2-6/) adds `torch.compile` for **Python 3.13**, introduces **FP16** on **X86**, and uses **Manylinux 2.28**, but drops **Conda** support for distribution.
- **GPU Pricing and Availability**: Users note that new **5090** GPUs are very difficult to obtain, selling out rapidly while **Jetson Nano** prices have surged to **$500-$700**, as opposed to listings at around **$250**.

**5. Funding, Ethics, and Community Buzz**

- **Dario Amodei’s AI Safety Investment Criticized**: Community members express skepticism about **Dario Amodei's** bold **$1B** push toward **AI Safety**, with some labeling his claims as *fraudulent marketing*, and [questioning large-scale AI fundraising efforts](https://www.nature.com/articles/s41593-023-01514-1).
-  **SoftBank's Billion-Dollar Bet on OpenAI**:  **SoftBank** is reportedly planning a massive **$15-25 billion** investment in **OpenAI**, as another major bet on AI and its future potential, [adding to its existing commitments](https://x.com/firstadopter/status/1884794211091759444).
- **Community Engages Across Platforms**: Members actively share findings and ask questions, with strong engagements about various AI models, frameworks, and tooling, including discussions in many Discords on how different methods are influencing the field.


---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 Speeds and Snags**: **DeepSeek R1 1.58B** runs at about **3 tokens/s** on limited hardware, with [this official doc](https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1) suggesting **160GB VRAM** or fast storage for higher throughput.
   - Community members flagged potential issues on **Windows** and recommended **Linux** for improved quantization performance.
- **Mistral-Small 24B Breezes In**: The newly shared [Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF) offers an **Apache 2.0 license**, features closed weights, and promises reduced latency.
   - Contributors referenced [Mistral's site](https://mistral.ai/news/mistral-small-3/) citing **81% MMLU**, seeing it as a compelling addition to open-source options.
- **Online DPO Spark with Unsloth**: A user confirmed **online DPO** worked using [Unsloth repos](https://github.com/unslothai/unsloth/issues/1494) after applying partial hard coding to handle memory constraints.
   - They included a [LinkedIn post](https://www.linkedin.com/posts/keith-truongcao-7bb84a23b_reduce-online-dpo-memory-consumption-with-activity-7290108099607097344-jzaO) about lowering DPO memory usage and asked for real-world feedback.
- **MusicGen Fine-Tune Foray**: A newcomer aims to fine-tune **facebook/musicgen-medium** or **musicgen-small** with `.WAV` and `.TXT` files, focusing on epoch and batch size as seen in [this guide](https://github.com/volcengine/verl).
   - They considered leveraging **vllm** for generation but also examined **Unsloth** and **GRPOTrainer**, seeking a stable fine-tuning path.
- **vllm vs. Unsloth: Shared Goals or Splitting Paths?**: Community members compare **vllm**'s neural magic benefits with **Unsloth**'s quantization approach, uncertain about future alignment under **Red Hat**.
   - Some floated partial integration to curb GPU downtime, while others viewed each approach as distinct due to speed differences.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 vs. R1 Rivalry & Perplexity's Toggle Troubles**: Users questioned **O1 vs. R1** reliability in Perplexity Pro, noting default switches to R1 despite choosing O1. Many felt **O1** offers better reasoning, but recent reliability issues prompted concerns.
   - In [a tweet from Aravind Srinivas](https://x.com/aravsrinivas/status/1884801300027589007), **Pro users** were promised **500 daily DeepSeek R1** queries, yet confusion remains on its consistency, with some users calling it *annoyingly unstable*.
- **Alibaba's Competition Chaser Model**: Alibaba introduced a **new model** to strengthen its competitive position, possibly realigning market dynamics. More details appear in [this link](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg), highlighting advanced algorithms for faster user experiences.
   - Community members anticipate further enhancements, with some hinting at *possible synergy with existing open-source frameworks*, though no official statement has been made.
- **DeepSeek Gains Traction & Shakes Up Data Retrieval**: [OpenAI clarified its usage of DeepSeek](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA), praising its query handling for complex datasets. Many praised **DeepSeek's** stable privacy features, even as they noted occasional downtimes.
   - Deepinfra’s [DeepSeek-R1 Demo](https://deepinfra.com/deepseek-ai/DeepSeek-R1) was cited for fulfilling similar tasks as **OpenAI-O1**, sparking lively debate over *token usage* and performance benefits.
- **Sonar-Reasoning's Surprising Shortfalls**: Testers of the **sonar-reasoning** model API questioned its real-world performance, seeking details on improvements over other models. Some reported *lengthy, repeated answers* that wasted tokens and ignored new prompts.
   - Others argued it still outperforms in certain tasks, but direct side-by-side comparisons in the *playground* indicated the model’s *thinking* might be diminished in API responses.
- **GPT-4, Sonnet, and Gemini Showdown**: In an ongoing debate, users covered **GPT-4**, **Sonnet**, and **Gemini 2.0** for advanced queries, including calculus and coding tasks. **Sonnet** earned acclaim for more natural-sounding text, while GPT-4 and Gemini remain powerhouses for raw accuracy.
   - Some highlighted that pairing **Sonnet** with O1 yields *clearer outputs for complex tasks*, motivating a shift away from partial Claude subscriptions and rethinking paywalls.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek's Dynamic Duo**: Windsurf introduced [DeepSeek R1](https://x.com/windsurf_ai/status/1885077046663217230) and **DeepSeek V3** for Pro-level accounts, each requiring distinct credits per message.
   - Developers highlighted R1's first-ever coding agent usage, referencing the [changelog](https://www.codeium.com/changelog) for more updates.
- **Cascade's Quick Fixes**: Community members reported **input lag reductions** and fixes to stop the Cascade panel from reopening on reload.
   - They also discussed new web search capabilities via `@web` and `@docs`, pointing to URL-based context handling.
- **DeepSeek vs. Sonnet Showdown**: Users compared cost-efficiency and performance between **DeepSeek** and **Claude 3.5 Sonnet**, with many testers preferring R1.
   - Others described **Sonnet** perpetually editing files, while R1 demonstrated steady behavior in coding tasks.
- **Credit Confusion Clarified**: Members debated whether **DeepSeek R1** uses 0.25 or 0.5 credits per message, citing variable documentation.
   - They pointed to the [Codeium Docs](https://docs.codeium.com/windsurf/usage) and [support page](https://codeium.com/support) for precise details.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek Dares to Duel with OpenAI**: In guild talk, participants highlighted that **DeepSeek R1** outshines **OpenAI's** o1 for creative tasks, referencing [a Raspberry Pi demo](https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/) and potential rivalry from **Gemini Pro** and **Grok**.
   - Someone claimed *'DeepSeek censors results'* in [a YouTube critique](https://www.youtube.com/watch?v=3QuWqjJ1ZjM), setting off speculation about **data collection** and open access.
- **OneClickPrompts for Swift Setup**: A new tool named **OneClickPrompts** was introduced for constructing personalizable multi-part prompts, with a shared GIF highlighting simplified usage for repeated tasks.
   - Users praised the extension's **modular approach** but noted *'smart prompt combos'* are still essential to achieve deeper results.
- **Fine-Tuning Ollama Gains Ground**: A user sought methods to **fine-tune Ollama** for domain-specific tasks, raising hopes for future expansions or official workflows.
   - Others pointed to scattered references on GitHub, adding that streamlined procedures could unlock *'next-level adaptability'* in **Ollama**.
- **GPT's Memory & Context Windows Collide**: Members criticized **GPT**'s memory for losing crucial details over lengthy chats, sparking renewed interest in bigger context windows from open source projects like **DeepSeek**.
   - They argued that **inconsistent recollection** hinders production usage, with calls for *'stable context retention'* as a must-have feature going forward.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.9 Gains Momentum**: **LM Studio 0.3.9** added **Idle TTL**, separate **reasoning_content** in API responses, and **auto-update** for runtimes, with official installers [here](https://lmstudio.ai/download).
   - Community members recognized improved memory management and cited simpler auto-update processes for **Hugging Face** model downloads, referencing the [docs](https://lmstudio.ai/docs/api/ttl-and-auto-evict).
- **RAG Rolls Out in LM Studio**: **LM Studio** now supports **RAG** with attached local documents in chat sessions, described in the [docs](https://lmstudio.ai/docs/basics/rag).
   - Users observed that if a document fits within the model’s context, it can be included in full, sparking interest in leveraging local references.
- **DeepSeek's GPU Performance Surges**: Discussions revealed **6-7 tokens/sec** on a **GTX 1080** and **Ryzen 5 3600** for **DeepSeek** models, with a focus on VRAM management to prevent slowdowns.
   - Others reported **i9-14900KF**, **128GB RAM**, and dual **RTX 4090** setups reaching **30-40 tokens/sec** on **70B** models, emphasizing the significance of fitting the entire model into GPU memory.
- **Jetson Nano Pricing Raises Eyebrows**: Members noted the **Jetson Nano** hitting **$500-$700** or being backordered, making it less appealing compared to standard GPUs.
   - A few found listings around **$250**, but many leaned toward more conventional hardware for superior performance.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek R1 Soars & Database Spills**: Members reported **DeepSeek R1** hitting around **32 TPS** on a 4090 GPU, praising its performance while also noting issues with quantized variants. [A leak on Hacker News](https://news.ycombinator.com/item?id=42871371) revealed a **DeepSeek** database exposure that raised user privacy alarms.
   - Some participants voiced skepticism about relying on a service with a potential data breach, referencing *privacy nightmares* as a reason to explore local solutions.
- **O3 Mini Hype & Quantization Quirks**: Many expressed interest in **O3 Mini** as a potentially faster and smaller alternative, anticipating improved experiences over existing large models. They discussed how **quantization** can hamper performance and instruction-following, with some calling it a tricky trade-off.
   - A few joked about waiting impatiently for **O3 Mini** to address their model woes, while others shared varied results with prior quantized releases, highlighting the unpredictability of sizing models down.
- **Aider Gets Local & Read-Only Stubs**: Users explored integrating **Aider** with local models like **Ollama** for privacy reasons, expecting a solution that avoids sending data to third parties. A new [YouTube video](https://youtu.be/XE6v_RGe0-U) showcased **read-only stubs** designed to handle large codebases more efficiently.
   - Some encountered confusion using multiple endpoints (e.g., Azure AI) but found references to [advanced model settings](https://aider.chat/docs/config/adv-model-settings.html) helpful, with others praising *stubs* as a welcome step to keep code modifications under tighter control.
- **O1 Pro Debates Spark Pricing Talk**: Several devs championed **O1 Pro** for coding tasks, but they criticized its cost and usage constraints. They weighed these factors against local open-source models, noting that censorship concerns occasionally hinder productivity.
   - A few participants described **O1 Pro** as a strong coding ally despite the price tag, while some remain committed to local models for freedom from potential policy shifts.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 Surfs the West**: Windsurf announced that **DeepSeek R1** and V3 are now live with [tool calling capabilities](https://x.com/windsurf_ai/status/1885077046663217230), enabling R1 to run in coding agent mode for the first time.
   - Users noted that it's fully hosted on Western servers and referenced [Cursor's community forum](https://forum.cursor.com/latest) for ongoing discussion.
- **Token Tangles in Chat & Composer**: Some users expressed confusion over the **10k token context** setting, reporting difficulties tracking usage in chat and composer.
   - They questioned whether the beta settings genuinely provide extended contexts or if messages get truncated without warning.
- **MCP Setup Gathers Steam**: A bash script approach lets people add **MCP server** configurations quickly, as shown in [this GitHub repo](https://github.com/daniel-lxs/mcp-server-starter).
   - Developers shared the [MCP Servers site](https://www.mcpservers.ai/) to encourage trying different servers in tandem with Cursor.
- **Model Security Storm Warnings**: Concerns arose about potential hidden code execution in ML models, referencing [a post on silent backdoors in Hugging Face models](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/).
   - Some recommended using [protectai/modelscan](https://github.com/protectai/modelscan) for scanning local setups to unearth any suspicious payloads.
- **Local vs Hosted Showdown**: A lively debate broke out over self-hosting compared to relying on solutions like **DeepSeek R1**, citing privacy and cost trade-offs.
   - While local enthusiasts hope for better offline models, others point to the performance benefits of hosted servers as they evolve.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous x Solana Sunset Soirée**: The upcoming **Nous x Solana** event in NYC is brimming with attendance requests, focusing on discussions around distributed training in AI models.
   - Participants anticipate **in-person demos** and specialized Q&A, hoping for synergy with the new **Psyche** approach.
- **Mistral & Tülu Tussle**: Community members shared excitement over **Mistral-Small-24B-Instruct-2501** on [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) and **Tülu 3 405B** from [this tweet](https://x.com/allen_ai/status/1884966600039915809), both positioned for top performance in smaller-scale LLMs.
   - Several pointed to [R1-Zero’s blog analysis](https://arcprize.org/blog/r1-zero-r1-results-analysis) for benchmark comparisons, fueling debate on which model truly excels.
- **Psyche Paves Paths for Distributed Gains**: The **Psyche** distributed training framework aims to handle large-scale RL with a modular system, drawing praise for its ability to scale model training.
   - A [tweet](https://x.com/Teknium1/status/1884740956911718853) showcased excitement for open sourcing this framework, with focus on GitHub accessibility and a possible consensus algorithm roadmap.
- **China's Ten Titans Tower Over Europe's Models**: A chat revealed **China** has TEN top-tier AI models rivaling Europe’s biggest, including **Mistral**, per [this tweet](https://x.com/deedydas/status/1884786839913111931).
   - Participants noted the **US** boasts only five major AI labs—**OpenAI**, **Anthropic**, **Google**, **Meta**, and **xAI**—highlighting a fierce global race.
- **CLIP-Driven Generation Gains Ground**: A member inquired about **autoregressive generation** on **CLIP** embeddings, typically employed for guiding **Stable Diffusion**.
   - They stressed a gap in references for direct CLIP-driven generative processes, indicating interest in merging multimodal inputs with decoding tasks.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dario’s Daring $1B Venture**: Community members discussed **Dario Amodei** and his **$1B** push toward **AI Safety**, raising questions about financial transparency and ambitious claims in his blog post. They highlighted unease over what some labeled *fraudulent marketing*, reflecting deeper skepticism toward large-scale AI fundraising efforts.
   - Several technologists argued that funneling such large sums into sweeping safety initiatives may neglect other pressing AI research, while others insisted it could catalyze more responsible AI development.
- **Mistral’s Middling-Sized Marvel**: The newly unveiled [**Mistral Small 3**](https://mistral.ai/news/mistral-small-3/) packs **24B parameters**, nets **81%** on MMLU, and runs **3x faster** than bigger competitors. Developers praised its local deployment capability, citing a sweet spot between performance and resource efficiency.
   - Enthusiasts contrasted it with models like **Llama 3.3 (70B)**, suggesting Mistral’s tight design could spur more accessible, specialized solutions.
- **Tülu 3 405B Triumph**: Researchers at AI2 released [**Tülu 3 405B**](https://allenai.org/blog/tulu-3-405B), boasting an enormous **405B** parameters and defeating both **DeepSeek v3** and **GPT-4o** on multiple benchmarks. Its **Reinforcement Learning from Verifiable Rewards (RLVR)** approach propelled the model’s accuracy and consistency in test environments.
   - Participants noted the model’s training recipes and open-weight policy, citing potential momentum for even bolder open-research collaborations.
- **Framework Face-Off: LlamaIndex vs PydanticAI vs LangChain**: Developers reported **PydanticAI**’s neat interface and internal temperature settings but lamented its frequent broken JSON outputs. [**LlamaIndex**](https://x.com/llama_index) yielded more consistent structured data, while **LangChain** drew criticism for complicating error tracing with its pipe-based architecture.
   - Others highlighted high CPU or GPU usage in certain UIs as a sticking point, fueling calls for streamlined agent tooling with robust logging and performance metrics.
- **Prospective Config’s Bold Brainchild**: A [**Nature Neuroscience** paper](https://www.nature.com/articles/s41593-023-01514-1) introduced **prospective configuration** as a foundation for learning **beyond backpropagation**, sparking fresh speculation on next-gen neural training. The method claims improved efficiency and better alignment with biological processes.
   - Community conversation suggested potential synergy with **RL** approaches, while some questioned if the approach might overpromise, given the field’s rapid pace of technical leaps.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tülu 3 Topples Titans**: The **Tülu 3 405B** launch shows superior performance compared to **DeepSeek v3** and **GPT-4o**, as described in [their blog](https://allenai.org/blog/tulu-3-405B).
   - Enthusiasts highlighted **open post-training recipes**, with excitement swirling over its scalability and massive 405B-parameter footprint.
- **Mistral Small 3 Masters Minimal Latency**: **Mistral Small 3** debuted as a 24B-parameter model at low latency, claimed to run comfortably on typical hardware ([details here](https://mistral.ai/news/mistral-small-3)).
   - Community feedback praised its **knowledge-dense** architecture, positioning it as a strong competitor for local generative AI tasks.
- **DeepSeek Leak Sparks Security Fears**: [Wiz Research revealed](https://x.com/wiz_io/status/1884707816935391703) a publicly accessible DeepSeek database, exposing secret keys and chat logs.
   - Discussions centered on **privacy concerns**, prompting calls for stricter control measures in AI infrastructure.
- **SoftBank Showers OpenAI with Billions**: Reports emerged of **SoftBank** planning to invest **$15-25 billion** into OpenAI, supplementing its existing pledge of over $15 billion.
   - Analysts see this as yet another massive bet on AI, raising the stakes in an already fierce funding race.
- **DeepSeek v3 Experts Go Parallel**: New **Mixture-of-Experts** design in DeepSeek v3 uses **sigmoid gating** and **dropless load balancing**, letting multiple experts respond without direct contention ([paper](https://arxiv.org/abs/2401.06066)).
   - Contributors discussed fine-tuning those expert layers and applying **MTP** to forecast two tokens at once, fueling speculation on inference acceleration.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepseek Dilemma: OpenAI's Double-Edged Ethics**: Community members noted that **OpenAI** criticized **Deepseek training** while ironically using data from similar sources, raising questions about their motivations. They suspected **OpenAI** used legal claims to bolster a confident image in a crowded field.
   - Some participants felt the **Deepseek** debate highlights potential hypocrisy, fueling doubts about whether **OpenAI** truly safeguards collaborators' interests.
- **RL Revelation: Less Tools, More Talent**: LLM enthusiasts discovered that using **reinforcement learning (RL)** can reduce the size of tool usage instructions, letting models pick up essential skills with minimal guidance. They worried that overreliance on specific tools could undermine core problem-solving abilities.
   - By balancing **RL** with selective tool exposure, they hope to preserve a model’s reasoning prowess without letting it drift into rote tool dependency.
- **Hyperfitting Hype: Big Gains from Tiny Data**: New results showed that **hyperfitting** on a tiny dataset can catapult open-ended text generation, climbing from **4.9%** to **34.3%** in human preference scores. A [paper](https://openreview.net/forum?id=Ij9ilPh36h) confirmed these dramatic improvements, prompting a reexamination of traditional overfitting fears.
   - Critics debated whether such narrow training jeopardizes broader generalization, but many welcomed these surprising boosts in text quality.
- **Critique Craze: Fine-Tuning Beats Blind Imitation**: Researchers proposed **Critique Fine-Tuning (CFT)**, teaching models to spot and correct noisy responses rather than merely imitating correct solutions. They reported a **4–10%** performance jump across six math benchmarks, as documented in [this paper](https://arxiv.org/abs/2501.17703).
   - The community expressed optimism that teaching models to critique mistakes might produce more robust reasoning than standard supervised fine-tuning.
- **Backdoor Buzz & Llama2 Config Confusion**: New warnings about undetectable **backdoored models** arose in [this paper](https://arxiv.org/abs/2409.03077), casting doubt on conventional loss-based detection strategies. Meanwhile, developers questioned the significance of **32768** in [Llama2’s config](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26) when setting gated MLP dimensions.
   - Some pointed out that this number isn’t divisible by **3**, leading to a reset toward **11008** and stirring further discussion on how to export model configurations cleanly.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek's Double-Dose Distills**: OpenRouter introduced [DeepSeek R1 Distill Qwen 32B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b) and [DeepSeek R1 Distill Qwen 14B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b), each promising near-larger-model performance at **$0.7–$0.75** per million tokens.
   - The 14B version reportedly scored **69.7** on AIME 2024, with both models accessible via the OpenRouter **Discord**.
- **Subconscious AI & Beamlit Big Moves**: Subconscious AI showcased **causal inference** and **market simulation** potential on their [website](https://www.subconscious.ai), stressing 'guaranteed human-level reliability.'
   - Meanwhile, [Beamlit](https://beamlit.com) launched a free alpha that accelerates shipping AI agents up to **10×**, offering GitHub workflows and observability tools.
- **OpenRouter Pricing Tiffs & Rate Limit Rants**: Users debated the **5%** fee for OpenRouter, attributing it partly to underlying **Stripe** costs.
   - Others reported frequent **429 RESOURCE_EXHAUSTED** errors with Google's Gemini, advising personal API keys to avoid timeouts.
- **Mistral's Small 3 & Tülu 3 Teasers**: Announced via [tweets](https://x.com/allen_ai/status/1884966600039915809), **Mistral's Small 3** (24B, 81% MMLU) and **Tülu 3** (405B) both promise expanded training and faster throughput.
   - Community chatter suggests these new releases may pair well with DeepSeek for bigger gains in speed and accuracy.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt’s Big Binary Break**: Bolt stops generating binary assets, significantly reducing token usage by **hundreds of thousands** and improving output quality, according to a [tweet from bolt.new](https://x.com/boltdotnew/status/1885019780840653183).
   - Members praised the shift to **external assets**, noting faster execution and celebrating it as “a major performance leap” in community talk.
- **Community System Prompt Surprises**: Dev discussions turned to the **Project and Global System Prompt**, with one user employing it for changelog updates and hoping to see expanded creative uses.
   - A tip emerged to share specific files and confirm correct views, showcasing deeper usage potential beyond everyday tasks.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI Gains Crisp Control for Inpainting**: Some participants shared manual approaches to inpainting, referencing [examples on Streamable](https://streamable.com/d3ww4l) for advanced ControlNet setups in **ComfyUI** with accurate touches.
   - They praised flexibility for specific adjustments instead of relying solely on automated methods.
- **Hardware Hustle: GPU Chatter Heats Up**: Users debated their GPU options, with the **Intel Arc A770 LE** pitched as comparable to a **3060** for gaming and AI tasks.
   - Others swapped tips on **3080** and **3090** usage, focusing on VRAM requirements for Stable Diffusion.
- **Face Swap Reactor Reemerges with Filters**: Participants noted the removal of **Reactor** due to lacking NSFW checks, before reuploading a safer version on [GitHub](https://github.com/Gourieff/sd-webui-reactor-sfw).
   - They also pointed to the [ComfyUI extension](https://github.com/Gourieff/ComfyUI-ReActor) for streamlined face swap functionalities.
- **Lora Training Twists for Stable Diffusion**: Members dissected the steps for building **Loras**, emphasizing style integration and precise facial matching.
   - They discussed combining multiple references, highlighting challenges in synchronizing style and features.
- **5090 GPUs Vanish in a Flash**: New **5090** GPUs were snapped up instantly, prompting frustration over shortage and steep demand.
   - People mulled financing choices to afford fresh hardware, disappointed by the minimal inventory.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell & The Brash sm_120a Breakthrough**: New **Blackwell** architecture with **sm_120a** overshadowed prior **sm_90a** features, as detailed in [cutlass/media/docs/blackwell_functionality.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md), promising stronger compute capability for consumer GPUs.
   - Community members debated **RTX 5090** gains vs **RTX 4090**, citing a possible **5x** speedup in **FP4** tasks but only **2x** in other tests, raising concerns about *inconsistent documentation*.
- **PyTorch 2.6 Packs a Punch**: Recently launched **PyTorch 2.6** adds `torch.compile` support for **Python 3.13**, introduces **FP16 on X86**, and uses **Manylinux 2.28**, described in [PyTorch 2.6 Release Blog](https://pytorch.org/blog/pytorch2-6/).
   - Enthusiasts noted **Conda** deprecation while praising new performance knobs like `torch.compiler.set_stance`, with some calling it *'a big shift'* in distribution strategy.
- **Reasoning Gym’s Rapid Expansion**: The **Reasoning Gym** soared to **33** datasets, included in a new gallery at [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md), showcasing a wide range of reinforcement learning tasks.
   - Contributors praised cooperative challenges and proposed *multi-agent negotiation* setups, fueling conversation on **explanatory** and **logic-based** tasks.
- **Mistral’s Mischief at the AIx Jam**: The **Mistral AIx** entry landed #2 in the 🤗 Game Jam, inviting folks to test **ParentalControl** in [this HF Space](https://huggingface.co/spaces/Mistral-AI-Game-Jam/ParentalControl), blending **AI** with game dev for comedic horror.
   - They also showcased **Llama3-8B R1** with a claimed **14%** improvement in GSM8K, as detailed in [this blogpost](https://mobiusml.github.io/r1_redistill_blogpost/), sparking excitement about cost-efficient training.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **DeepSeek Models Spark LaTeX Talk**: Members eagerly await the **DeepSeek** release, touting strong math and **LaTeX** capabilities for complex tasks.
   - They discussed VRAM constraints, stressing careful context-size management for heavier computations.
- **Ollama & GPT4All Connect for Local Gains**: Some confirmed hooking **GPT4All** to **Ollama** by running Ollama as a server and tapping the **OpenAI** API from GPT4All.
   - They pointed to the [GPT4All Docs](https://docs.gpt4all.io/gpt4all_api_server/home.html) for a step-by-step approach.
- **Remote LLMs Step into GPT4All**: Users tested loading remote LLMs into **GPT4All**, highlighting the need to set correct API keys and environment variables.
   - They recommended improved guidance in the [GitHub wiki](https://github.com/nomic-ai/gpt4all/wiki/Local-API-Server) to help newcomers.
- **AI Education Initiative Hits Offline Mode**: A user showcased a plan to build an AI-driven tool for children in Africa, referencing [Funda AI](https://emmanuelsibanda.hashnode.dev/funda-ai-building-a-laptop-powered-by-ai-to-help-students-in-africa-learn).
   - They plan to use small-footprint models and curated data to allow self-study without internet, bridging resource gaps.
- **Model Suffix Mystery -I1-**: One member asked about **-I1-** in some model names but no official explanation was confirmed.
   - Others requested clearer labeling, indicating a demand for more open model documentation.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Cursor's Constrained MCP Capabilities**: The new **Cursor** adds partial **MCP** support, but **environment variables** remain a gap, prompting command-line workarounds like `FOO=bar npx some-server` as noted in [env invocation](https://www.gnu.org/software/coreutils/manual/html_node/env-invocation.html).
   - Community members seek better alignment between **MCP** and **LSP** config structures, describing this mismatch as a stumbling block for broader adoption.
- **Web Client Wizardry for MCP**: A self-hosted web client now coordinates multiple **MCP** servers and agents, enabling smooth hand-offs for local or cloud setups.
   - Its flexible approach fuels interest, although some lament the lack of dynamic agent prompt functionality for **MCP**.
- **Function-Calling Frustrations for 8b Model**: An **8b** model in **MCP** struggles with function calling and tool usage, confounding testers who rely on robust agent interactions.
   - Several contributors suggest deeper community input on forums like *Reddit*, hoping to address the model's reliability concerns.
- **Hataraku Hits #1 on ShowHN**: The **Hataraku** project soared to the top on ShowHN, sparking momentum for its [TypeScript SDK proposal](https://github.com/turlockmike/hataraku/blob/main/docs/sdk-proposal.md) and CLI features.
   - Community members are pitching in with collaboration and trial runs, aiming to refine the interface and improve broader user experiences.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's February Feedback Fandango**: NotebookLM is hosting remote chat sessions on **February 6th, 2025** for user feedback, offering **$75** to participants.
   - They require a [screener form submission](https://forms.gle/HJmCwNepsfPSdC7g7), a stable Internet connection, and a device with video capabilities.
- **Transcribing Trading Tactics with League of Legends Lingo**: A user converted trading course videos to audio, then transcribed them with AI and used [NotebookLM](https://link.to.notebooklm) to clarify advanced material.
   - They introduced **Big Players** using LoL references, demonstrating AI's flexible approach to explaining complex ideas.
- **Executive Order Exposé in 24 Hours**: NotebookLM summarized a new Executive Order on public education privacy in under a day, with an in-depth [YouTube review](https://youtu.be/8RFYmgYn7P4).
   - This demonstration sparked conversation on applying the tool for policy briefs and thorough analysis.
- **DeepSeek R1 Dissected: GRPO & MoE**: A NotebookLM Podcast covered **DeepSeek R1**, highlighting **GRPO** and **Mixture of Experts** to explain its architecture.
   - Listeners viewed the [full discussion](https://youtube.com/watch?v=zVDmKv3hWzk) with benchmarks and a quick demo, fueling questions on performance gains.
- **Sluggish Study Times and Language Lapses**: Some users faced 10–30 minute delays generating study guides, even with a single source.
   - Others lamented poor multilingual handling (e.g., **Korean** and **Japanese**) and brief **Gemini 2.0 Flash** glitches, while seeking stricter source usage rules.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Branch Bump & Retarget Roundup**: The **branch changes** are completed with all pull requests retargeted, ensuring a smooth code integration process.
   - Team members can ask questions if they're unsure, highlighting the project's emphasis on open communication.
- **NeoVim Nudges for Mojo LSP**: Developers discussed enabling **Mojo LSP** with [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig), encountering some quirks during setup.
   - A few reported only partial success, suggesting deeper debugging is needed for a stable workflow.
- **Mojo 1.0: Speed vs. Stability Showdown**: Chris Lattner stressed that **Mojo 1.0** should blend GPU optimization with direct execution to maximize speed.
   - Participants argued that immediate reliability must balance the race for top performance metrics.
- **Backward Compatibility Shakedown**: Members worried that **breaking changes** in new Mojo releases could deter users from upgrading.
   - They emphasized support for older libraries to maintain momentum and cultivate a steady user base.
- **Reflection & Performance in the Mojo Mix**: Conversation centered on **reflection** for data serialization and noted that some reflection is partially implemented.
   - Attendees also pushed for large-cluster **benchmarking**, mentioning the [Mojo🔥: a deep dive on ownership](https://youtu.be/9ag0fPMmYPQ) video with Chris Lattner.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Small but Speedy: Mistral 3**: **Mistral Small 3** was introduced as a 24B-parameter model under an **Apache 2.0 license** with **81% MMLU** and **150 tokens/sec** performance, according to [official details](https://mistral.ai/news/mistral-small-3/).
   - It features fewer layers and a bigger vocabulary, sparking community interest in its unorthodox **FF-dim/model-dim ratio** on social media.
- **DeepSeek Database Leak Exposes Secrets**: A misconfigured **ClickHouse database** at **DeepSeek** led to a major data exposure, including chat histories and secret keys, as reported by [Wiz Research](https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak).
   - They quickly secured the leak after the disclosure, prompting concerns about overall safety in **AI** data handling.
- **FUZZ Frenzy at Riffusion**: Riffusion introduced **FUZZ**, a new generative music model that aims for high-quality output for free, [shared here](https://x.com/riffusionai/status/1884984941081198954).
   - Early adopters praised the melodic results, noting the service is only free while GPU resources hold up.
- **OpenAI API Lag Under Scrutiny**: Discussions mentioned **OpenRouter** and [Artificial Analysis](https://artificialanalysis.ai/providers/openai) as ways to track possible **latency** surges in OpenAI’s API.
   - Some saw normal response rates, but community members recommended caution and continuous checks.
- **ElevenLabs' $180M Funding Feat**: **ElevenLabs** raised **$180M** in a Series C round led by **a16z & ICONIQ**, a milestone [announced here](https://x.com/matistanis/status/1885011065018163224).
   - Observers see it as a strong endorsement for the future of **AI voice** technologies and their bigger market potential.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Track Teasers Tempt LLM Agents Crowd**: Participants await more details about the **application** and **research** tracks for the LLM Agents MOOC, which organizers promised to share soon.
   - Community members repeated *Stay tuned!* messages, eager to hear official announcements.
- **Sign-Up Snafus Stall Confirmation**: Several people noted they submitted the Google Forms sign-up but haven't received replies, particularly those pursuing **PhD** opportunities.
   - They asked for final acceptance details and faster responses to manage their schedules.
- **Quiz 1 Queries and Private Archives**: Members confirmed **Quiz 1** is live on the course website, referencing the syllabus, with some seeking older quiz solutions from a previous **LLM Agent course**.
   - They shared a [Quizzes Archive](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit), cautioning about hidden answers and an outdated browser prompt.
- **Certificate Confusion Continues**: Many await **certificates** from earlier sessions, with official guidance promised soon.
   - Organizers stated upcoming announcements will clarify the process for this semester's awards.
- **Lecture Launches and Accessibility Aims**: Members pressed for the **1st lecture** to be uploaded quickly, suggesting it would only take '5 minutes,' but the editing team cited **Berkeley's** captioning requirements.
   - They noted the livestream is watchable via the [course website](https://llmagents-learning.org/sp25), with a polished version pending completion of accessibility measures.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agents in Action: Mastering AI with LlamaIndex**: The **Mastering AI Agents Workshop** introduced advanced **AgentWorkflow** concepts for multi-agent frameworks, as shown in [this link](https://t.co/UKIClalkKG).
   - Attendees explored robust architectural approaches with **LlamaIndex**, fueling new conversations about best practices.
- **BlueSky Boost: LlamaIndex Spreads Its Wings**: The **LlamaIndex** team officially landed on **BlueSky**, highlighting new visibility at [this link](https://t.co/GK4L8Sb2N6).
   - Contributors anticipate expanded engagement with the platform, sparking more activity around AI developments.
- **O1’s Quirky Support: Partial Streaming and Debates**: Members noted **LlamaIndex** added `o1` compatibility with `pip install -U llama-index-llms-openai`, though some functionality remains unfulfilled.
   - They cited an [OpenAI community thread](https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043) which confirmed **OpenAI** has not fully enabled streaming, fueling user frustration.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GPU Tango: P2P Patch vs Proxmox**: In #general, participants discussed using a **P2P patch** with multiple **NVIDIA GPUs** and weighed **Proxmox** vs baremetal setups for optimal IOMMU support.
   - Some users prefer going baremetal to bypass perceived **hypervisor** constraints, while others reported that **Proxmox** can handle the job if configured precisely.
- **Tiny Boxes Team Up for VRAM Dreams**: Members explored how many **Tiny Boxes** can be interconnected and wondered about sharing **VRAM** for HPC-level inference across them.
   - They noted the lack of a direct VRAM pooling mechanism, suggesting a fast **NIC** for network-based scaling to achieve distributed performance.
- **Token Throughput: 15/sec to 100 Requests**: Estimations indicated a **15 tokens/sec** capacity per model, with potential scaling to **100 requests** if each ran at **14 tokens/sec**.
   - This illustrated how distributing requests can maintain near-peak speeds under controlled conditions, fueling HPC design discussions.
- **Server Shopping for On-Prem LLMs**: A user asked for recommended **physical servers** to host LLMs in an enterprise context, highlighting broader interest in on-prem solutions.
   - Community members discussed cost-effectiveness, power draw, and room for GPU expansion to handle large-scale deployments.
- **Block/Fused Code in Tinygrad**: In #learn-tinygrad, someone requested **sample code** for blocked/fused programs demonstrating how to load and write **tensor blocks**.
   - Others explained that performing operations in blocks can significantly **boost performance** by reducing overhead and merging steps.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R Cranks Up the Context**: A user shared struggles integrating **command-r7b** with **distillation frameworks**, citing *ollama* for synthetic data generation and noting gaps in existing support for these tools. They highlighted **Command R** as a large language model with **128,000-token** context length and retrieval-augmented generation, directing others to the [Models Overview](https://docs.cohere.com/v1/docs/models), [The Command R Model](https://docs.cohere.com/v1/docs/command-r), and the [Command R Changelog](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale).
   - Contributors focused on **Command R**’s upcoming release, emphasizing enhanced decision-making and data analysis capabilities. They also discussed bridging integration gaps for frameworks, hoping for smoother synthetic data workflows in future iterations.
- **AI’s Blanket Debate**: Some members described **AI models** as cold, joking that a *blanket* could bring them warmth. They believed this reflected a playful attempt to humanize emotionless machines.
   - Others insisted **AI** doesn't require warmth or feelings, sparking a quick back-and-forth on what defines genuine empathy in artificial systems. The banter highlighted ongoing curiosity about **AI’s emotional perception**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Proxy Patch & DSPy Debates**: One user asked about adding a **proxy** to `dspy.LM` adapter, referencing a [GitHub PR #1331](https://github.com/stanfordnlp/dspy/pull/1331) that integrated `http_client` in `gpt3.py`. They can't use **dspy 2.6** without proxy support for their hosted endpoints.
   - Another user highlighted how proxy usage aligns with the [dspy/clients/lm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53) code references. They also questioned whether **SSL context** configuration is possible within `litellm` for stable connections.
- **LiteLLM & DSPy: The Supported Squad**: A newcomer asked which **LLMs** are supported by DSPy, prompting a mention of the [LiteLLM documentation](https://docs.litellm.ai/docs/providers). The doc references **OpenAI**, **Azure**, and **VertexAI** offerings.
   - The conversation also addressed the challenge of specifying an **SSL context** with `http_client` for advanced configurations. Participants noted that these parameter settings are not fully explained in the default DSPy docs.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **KTO vs Axolotl: The Urgent Showdown**: Members flagged challenges in integrating **Axolotl** for **KTO** tasks, citing an urgent need to confirm feasibility and solution pathways.
   - They expressed readiness to help review code and finalize tasks, emphasizing a desire to keep projects on schedule.
- **Mistral Rolls Out 24B Model**: A new [Mistral-Small-24B-Base-2501 model](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) with **24B parameters** sparked excitement among members aiming for advanced small-size LLM performance.
   - This launch underscores **Mistral AI's open source commitment**, with additional commercial variants hinted to fill specialized needs.
- **Mistral Performance Mystery**: A member admitted lacking current hands-on experience with the **new Mistral model**, leaving performance claims unconfirmed.
   - The conversation suggested future user testing to gather real-world results and insights into how the model behaves in practice.
- **Winter Semester Overload**: A busy **winter semester** schedule was described as stuffed, impacting a member’s ability to contribute.
   - This may delay collaborative tasks, prompting others to coordinate timelines and share responsibilities.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Farm Friend Mystery**: A user voiced fondness for **Farm Friend** from last year, noting its current absence in discussions.
   - Community members remain curious about its fate, as no further updates were revealed in the thread.
- **Cliché Reviews Spark Amusement**: A lighthearted mention of **cliché reviews** caused playful banter and an accompanying image highlighted the joke.
   - Though no deeper context was provided, the exchange added a fun moment within the community.
- **Decoding '01'**: A user explained that '01' was unrelated to **OpenAI**, clarifying prior confusion in the dialogue.
   - The remark quelled speculation and confirmed the miscommunication was purely coincidental.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Boost Checkpoints with DCP Toggle**: Members clarified that **DCP checkpointing** is off by default but can be activated by setting `enable_async_checkpointing=True` in the config, enabling asynchronous writes.
   - They noted that this functionality, for now, is restricted to **full_finetune_distributed**, which may limit usage for other configurations.
- **Push for Wider Checkpoint Coverage**: Some wondered why **async checkpointing** isn't supported across all configurations, hinting at a needed future update.
   - No firm timeline was provided, leaving members hoping for broader integration to simplify large-scale finetuning processes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Local Img2Vid Craze**: A user asked about the best local **img2vid** tool, prompting conversation around performance needs and GPU utilization.
   - Others weighed in on their experiences, emphasizing quick setup and *clear documentation* for AI engineering workflows.
- **ltxv Gains Favor**: Another member promoted **ltxv** as the top choice for local img2vid tasks, citing its straightforward usage.
   - They hinted at future testing and refinements, hoping for more community-driven benchmarks and expanded model support.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Simba Sparks a Databricks Feature Frenzy**: Simba Khadder launched an **MLOps Workshop** for building **feature pipelines** on **Databricks** on **January 30th at 8 AM PT**, providing a direct sign-up link [here](https://buff.ly/40Ej4Z6).
   - Attendees can glean best practices from **Unity Catalog** integration and direct **Q&A**, with the event being **free** for **Data Engineers**, **Data Scientists**, and **Machine Learning Engineers**.
- **Databricks Embraces Geospatial Analytics**: On **January 30, 2025 at 1:00 PM EST**, Databricks is hosting a free session on advanced **geospatial analytics**, with sign-up available on [Eventbrite](https://www.eventbrite.com/e/doi-geospatial-analytics-with-databricks-tickets-1111902653769).
   - Attendees will see how **spatial data** is processed on Databricks, continuing the momentum from the earlier workshop for those seeking deeper data engineering insights.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL Data Rallies for HF Datasets**: One participant asked about the steps needed to make **BFCL data** align with the **Hugging Face** dataset guidelines, seeking a blueprint to ensure compliance.
   - No examples or documentation were provided, leaving the conversation open-ended on how to adjust the *metadata schema* or format.
- **No Additional Topics Appear**: The conversation was limited to the single inquiry on achieving **Hugging Face** compliance for BFCL data.
   - No further details surfaced, with silence from others on potential solutions.



---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1334252515176222791)** (1053 messages🔥🔥🔥): 

> `DeepSeek R1 performance, Mistral Small 24B, Fine-tuning strategies, Quantization, Unsloth capabilities` 


- **DeepSeek R1 1.58B is slow but coherent**: Users report that while the DeepSeek R1 1.58B model runs correctly, it is very slow, typically achieving about 3 tokens per second due to hardware limitations.
   - For optimal performance, greater VRAM and faster storage solutions are recommended.
- **Release of Mistral Small 24B**: Mistral Small 24B has been uploaded to Hugging Face, offering competitive performance against larger models while being latency-optimized.
   - It is open source under the Apache 2.0 license, although its weights are closed source, generating interest among developers.
- **Fine-tuning multiple tasks without forgetting**: Experts advise against sequential fine-tuning (A -> B -> C) as it often leads to catastrophic forgetting of previous tasks.
   - Instead, it's suggested to combine all desired tasks in a single fine-tuning phase to maintain learned characteristics.
- **Quantization and Model Size**: The conversation discusses the use of dynamic quantization and its potential to reduce memory usage while maintaining performance in large models.
   - Quantization techniques can help allow for larger models to run on smaller hardware, although they may require careful implementation.
- **Best model for limited hardware**: For a PC with 16GB of RAM and an RTX 4060 with 8GB VRAM, users inquire about the best models suitable for their setup.
   - They are directed towards the possibility of running reduced versions of DeepSeek or Mistral models based on compatibility with their hardware limitations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead">DeepSeek's AI breakthrough bypasses industry-standard CUDA for some functions, uses Nvidia's assembly-like PTX programming instead</a>: Dramatic optimizations do not come easy.</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-unsloth-bnb-4bit">unsloth/Llama-3.2-3B-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF">unsloth/Mistral-Small-24B-Instruct-2501-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit">unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit">unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://x.com/OpenWebUI/status/1884719609552752801">Tweet from Open WebUI (@OpenWebUI)</a>: 🚀 You can now run 1.58-bit DeepSeek-R1 (non-distilled version) on Open WebUI with llama.cpp, thanks to @UnslothAI! 💻⚡️ (Tested on M4 Max, 128GB RAM)  📝 Dive into the details in their blog post: htt...</li><li><a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: Clean, accessible reproduction of DeepSeek R1-Zero</a>: Clean, accessible reproduction of DeepSeek R1-Zero - Jiayi-Pan/TinyZero</li><li><a href="https://huggingface.co/docs/trl/main/en/grpo_trainer">GRPO Trainer</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models>">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1494">Changes made in Unsloth and openInstruct to get a successful Online DPO run · Issue #1494 · unslothai/unsloth</a>: Alright so as promised from the unsloth reddit post https://www.reddit.com/r/LocalLLaMA/comments/1hqkeyn/comment/m4rbtto/?utm_source=share&amp;utm_medium=web3x&amp;utm_name=web3xcss&amp;utm_term=1&amp...</li><li><a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal">GitHub - EvolvingLMMs-Lab/open-r1-multimodal: A fork to add multimodal model training to open-r1</a>: A fork to add multimodal model training to open-r1 - EvolvingLMMs-Lab/open-r1-multimodal</li><li><a href="https://forum.devtalk.com/t/deepseek-671b-running-on-a-cluster-of-8-mac-mini-pros-with-64gb-ram-each/185709">DeepSeek (671B) running on a cluster of 8 Mac Mini Pros with 64GB RAM each</a>: This is cool!   DEEPSEEK-V3 ON M4 MAC: BLAZING FAST INFERENCE ON APPLE SILICON We just witnessed something incredible: the largest open-source language model flexing its muscles on Apple Silicon. We’r...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1334272372538081452)** (16 messages🔥): 

> `Text-to-Image Servers, Model Training Issues, Frontend Imperfections, Sensitive Topic Adjustments` 


- **Text-to-Image Server Recommendations**: A user suggested utilizing dedicated text-to-image servers like **Stable Diffusion** or **Midjourney**, stating there are thousands available for such tasks.
   - Responses indicated a consensus on using proper platforms for image generation.
- **Dataset Concerns for Model Training**: A member noted that **R1** isn't trained on **cot traces**, highlighting the limitations in the current model's dataset.
   - This raised awareness about the importance of the right training data for effective model performance.
- **Frontend Imperfections Affecting Output**: A user observed a potential imperfection in the frontend, questioning why changes occur after tabbing out.
   - Another member affirmed the likelihood of model detection systems altering outputs based on frontend behavior.
- **Sensitivity Around Certain Topics**: Discussion emerged around sensitive topics, particularly in relation to changes that occur in different contexts.
   - Members commented that the correctness of topics may depend on the nuances of sensitive issues, particularly concerning **China**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1334259246765965443)** (202 messages🔥🔥): 

> `DeepSeek R1 Models, Fine-tuning Challenges, Quantization Techniques, System Requirements for Models, Inference and Performance` 


- **DeepSeek R1 Dynamic Model Configuration**: Users discussed running the DeepSeek R1 Dynamic 1.58-bit model on dedicated servers, with recommendations for suitable hardware like 160GB VRAM for optimal performance.
   - There was concern about using it on Windows and suggestions to switch to Linux due to better performance and compatibility.
- **Challenges with Fine-Tuning Mistral**: A user reported strange repetitive outputs after fine-tuning a Mistral 7B model, prompting questions about possible overfitting or dataset quality issues.
   - Another user suggested checking the chat template used for fine-tuning as a potential cause.
- **Questions on Model Quantization and Performance**: Discussions included queries about whether the R1 32B model could run on an 8GB RTX 4060, with affirmations of its capability when properly quantized.
   - Users expressed curiosity over the performance comparison between models like DeepSeek R1 8B and GPT-4.
- **User Experiences and Troubleshooting**: Participants shared personal experiences with model installation, highlighting various configurations required for running DeepSeek effectively.
   - Recommendations included using dedicated servers rather than personal hardware and avoiding running heavy models on Windows.
- **Unsloth's Dynamic Quantization Technique**: The dynamic quantization method used by Unsloth was highlighted as a significant factor in reducing size without sacrificing performance, with ongoing discussions about its effectiveness.
   - Participants sought clarification on how many models supported this technique, leading to resource sharing for further learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://ollama.com/Huzderu/deepseek-r1-671b-1.73bit">Huzderu/deepseek-r1-671b-1.73bit</a>: Merged GGUF Unsloth&#39;s DeepSeek-R1 671B 1.73bit dynamic quant</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Nemo_(12B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://www.datacamp.com/blog/llm-evaluation">LLM Evaluation: Metrics, Methodologies, Best Practices</a>: Learn how to evaluate large language models (LLMs) using key metrics, methodologies, and best practices to make informed decisions.</li><li><a href="https://ollama.com/download">Download Ollama on macOS</a>: Download Ollama for macOS</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb#scrollTo=LjY75GoYUCB8">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/html/2404.14047v1">How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings">Advanced settings configuration in WSL</a>: A guide to the wsl.conf and .wslconfig files used for configuring settings when running multiple Linux distributions on Windows Subsystem for Linux.</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7">Unsloth 4-bit Dynamic Quants - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/tree/main">unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt">Training a causal language model from scratch - Hugging Face NLP Course</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/inference">Inference | Unsloth Documentation</a>: Learn how to run your finetuned model.</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb#scrollTo=ekOmTR1hSNcr).">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing">Home</a>: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">Installing + Updating | Unsloth Documentation</a>: Learn to install Unsloth locally or online.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1334285194143858850)** (2 messages): 

> `Online DPO, Memory consumption in AI, Unsloth project` 


- **Successfully Implemented Online DPO**: A user announced that they successfully got **online DPO** working with **Unsloth**, acknowledging the hard coding present in their repos.
   - They requested feedback from the community on any potential issues related to their implementation.
- **LinkedIn Post on Reducing Memory Consumption**: The user shared a [LinkedIn post](https://www.linkedin.com/posts/keith-truongcao-7bb84a23b_reduce-online-dpo-memory-consumption-with-activity-7290108099607097344-jzaO?utm_source=share&utm_medium=member_android) discussing strategies to **reduce online DPO memory consumption**.
   - This post might contain valuable insights for those working on similar problems in AI.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1334527778837758026)** (13 messages🔥): 

> `Fine-tuning MusicGen, RL-based training frameworks, Unsloth and vllm comparison, Neural magic in vllm, Collaboration between vllm and Unsloth` 


- **Seeking Support for Fine-tuning MusicGen**: A member is looking to fine-tune the **facebook/musicgen-medium** or **facebook/musicgen-small** model using their dataset with `.WAV` and `.TXT` files and wants help in creating a training kit focused on parameters like epoch and batch size.
   - They emphasized their novice status and expressed appreciation for any assistance in the training process.
- **Discussion on RL-based Training Frameworks**: Members discussed RL-based frameworks like [verl](https://github.com/volcengine/verl) and Hugging Face's **GRPOTrainer**, noting their inclination towards utilizing **vllm** for generation and Hugging Face Transformers for training.
   - There was curiosity about whether this method could be the best long-term strategy compared to using **Unsloth** for both tasks.
- **Unsloth-Patched Model Speed Concerns**: One member questioned how much slower an **Unsloth-patched model** is at generation compared to **vllm**, considering if efficiencies in hardware utilization could balance the speeds.
   - The discussion highlighted that even if Unsloth's generation does not outperform vllm, close performance could still be beneficial due to reduced GPU idle times.
- **Neural Magic Behind vllm**: **vllm** has drawn attention for its performance, driven by neural magic, especially after its acquisition by **Red Hat**.
   - The community expressed uncertainty about future collaborations and whether both models could work together effectively.
- **Potential Collaboration between vllm and Unsloth**: Members pondered whether it is feasible to have **vllm** and **Unsloth** operate together or if one has to obstruct the other.
   - Questions were raised about the potential benefits and synergies of combining both frameworks rather than choosing one over the other.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1334252988273000459)** (993 messages🔥🔥🔥): 

> `Perplexity Pro features, Model performance comparison, Issues with O1 and R1, DeepSeek functionality, AI usage for academic support` 


- **Confusion Over O1 and R1 Functionality**: Users reported that despite selecting O1 in Perplexity Pro, the system defaults to R1, causing frustration among those wanting consistent performance.
   - Notably, users feel that O1 provides better reasoning capabilities compared to R1, yet it has been unreliable recently.
- **Discussion on AI Models for Learning**: In a conversation about AI models, users debated the effectiveness of GPT-4, Sonnet, and Gemini 2.0 for learning purposes, particularly in calculus and coding.
   - Many users expressed a preference for Sonnet due to its natural-sounding text and partnership with O1 for clarity in complex tasks.
- **DeepSeek Access and Reliability**: Users discussed the reliability of DeepSeek versus Perplexity, highlighting that Perplexity is more stable with better privacy features, while DeepSeek had frequent downtimes.
   - One user indicated that they successfully navigated account setup for Pro services related to `.gov` emails, demonstrating potential uses within organizations.
- **Preferences for AI Platforms**: There was a consensus that while Perplexity offers useful features, users are tempted by the flexibility of using multiple AI platforms, including ChatGPT.
   - Some users transitioned away from Gemini and Claude subscriptions, opting instead for the advantages provided by Perplexity and ChatGPT.
- **Questions on AI Usage**: Users posed questions about which AI models were best suited for specific tasks, such as image generation and document processing, with differing opinions.
   - The community shared experiences using various models, leading to recommendations based on individual use cases and performance expectations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://deepinfra.com/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 - Demo - DeepInfra</a>: We introduce DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. . Try out API on the Web</li><li><a href="https://tenor.com/view/whoa-shaking-head-windy-wind-blown-away-gif-15465608">Whoa Shaking Head GIF - Whoa Shaking Head Windy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/austinpowers-drevil-mikemyers-evillaugh-ohhh-gif-1374158791848398246">Austinpowers Drevil GIF - Austinpowers Drevil MikeMyers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aravsrinivas/status/1884801300027589007?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: All Perplexity Pro users now get 500 daily DeepSeek R1 queries (without censorship and prompts not going to China). Free users get 5 daily queries.Quoting Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">Tweet from Aravind Srinivas (@AravSrinivas)</a>: All Perplexity Pro users now get 500 daily DeepSeek R1 queries (without censorship and prompts not going to China). Free users get 5 daily queries.Quoting Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://inference.cerebras.ai/">Cerebras Inference</a>: no description found</li><li><a href="https://tenor.com/view/synths-and-sounds-ecstacy-ecstatic-climax-peak-gif-24235783">Synths And Sounds Ecstacy GIF - Synths And Sounds Ecstacy Ecstatic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aravsrinivas/status/1884913220734812449?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Perplexity Android App should support DeepSeek R1 on the Pro searches (turn the Pro toggle on to see the option). Update the app from the Play Store before trying.</li><li><a href="https://www.cplx.app/">Complexity</a>: An enhanced version of Perplexity.ai that everyone has ever wanted.</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1idrac5/this_logic_is_unbelievable/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://api-docs.deepseek.com/quick_start/token_usage">Token &amp; Token Usage | DeepSeek API Docs</a>: Tokens are the basic units used by models to represent natural language text, and also the units we use for billing. They can be intuitively understood as &#x27;characters&#x27; or &#x27;words&#x27;. ...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1334275173553143940)** (12 messages🔥): 

> `DeepSeek and OpenAI, Alibaba New Model, Doomsday Clock Update, Nike Snakeskin Red Shoes, Near-Earth Asteroid Discovery` 


- **OpenAI Claims DeepSeek Utilized for Data Retrieval**: OpenAI clarified that their tool **DeepSeek** was used for searching data, emphasizing its efficacy in queries. More information can be found in the [detailed article](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA).
   - The site noted that **DeepSeek** enhances search capabilities significantly when analyzing complex datasets.
- **Alibaba Intros New Model Amidst Market Competition**: Alibaba's recent unveiling of a **new model** aims to enhance its competitive edge in the tech landscape, indicating potential shifts in market dynamics. Full insights are available at [this link](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg).
   - The model incorporates advanced algorithms designed for efficiency, potentially reshaping user experiences.
- **Nike Launches Eye-Catching Snakeskin Red Shoes**: The newly released **Nike Snakeskin Red Shoes** are making waves for their striking design and limited availability, capturing the attention of sneaker enthusiasts. Details about these shoes can be explored [here](https://www.perplexity.ai/search/nike-snakeskin-red-shoes-cnf5iibDQcq18a2Jtv8WIQ#2).
   - Many fans are eager to grab a pair before they sell out, indicating the hype around the release.
- **Near-Earth Asteroid Discovery Sparks Interest**: A recent discovery of a **near-Earth asteroid** is generating excitement among scientists and space enthusiasts alike, as details emerge about its characteristics. Dive deeper into the findings at [this link](https://www.perplexity.ai/page/near-earth-asteroid-discovered-yhHY75OOT4ujnsalxMwelw).
   - The implications of studying such asteroids are vast for understanding the origins of life and planetary formation.



**Link mentioned**: <a href="https://www.youtube.com/embed/mgMh2Kp7Uwo">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1334253531305218048)** (4 messages): 

> `Sonar-Reasoning Model Performance, Response Quality Issues, Repeated Answers Concern` 


- **Sonar-Reasoning Model needs evaluation**: Members are testing the new **sonar-reasoning model API**, questioning its performance compared to other models and where it improves.
   - *Is it really better?* Members seek insights on improvements in specific areas of functionality.
- **Decreased Thinking in Responses**: Members have observed that the model doesn't seem to 'think' as effectively as it does in the playground, leading to frustrations.
   - One user noted that even when instructed, the model returns lengthy reasoning which consumes tokens unnecessarily.
- **Repeated Answers for Similar Questions**: A user reported a bug where the model repeatedly provides the same answer when asked similar questions, ignoring new queries.
   - This has raised concerns about the model's ability to differentiate between similar prompts, leading to a frustrating user experience.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1334636642513715272)** (1 messages): 

> `Cascade Models Update, DeepSeek-R1 and DeepSeek-V3, Input Lag Reductions, Web Search Capabilities, Changelog Insights` 


- **Cascade welcomes DeepSeek-R1 and V3**: Windsurf now supports **DeepSeek-R1** and **DeepSeek-V3** for Pro and Pro Ultimate users, each with specific credit costs per message and tool call.
   - This marks a notable upgrade as R1 can now be utilized in a coding agent for the first time.
- **Notable fixes and improvements in Cascade**: Recent updates include further **input lag reductions** for long Cascade conversations and fixes to prevent the Cascade panel from reopening on reload.
   - Additionally, there are more options in the **@docs** section and improvements to the Tab to Jump Quick Setting configuration.
- **Web search functionalities introduced in Cascade**: Cascade can now conduct **web searches** either automatically or through specific commands like `@web` and `@docs`, enhancing its capabilities.
   - Users can input URLs for context, making it effective for accessing blog posts, documents, and public GitHub files.
- **Changelog insight shared**: A detailed [changelog](https://www.codeium.com/changelog) was released, providing insights into all changes made in the latest version of Windsurf.
   - Users are encouraged to check the full update for a comprehensive understanding of the enhancements.
- **Join the Cascade conversation**: Community members are invited to join the ongoing discussion in the dedicated Discord channel for Cascade-related updates.
   - Engagement is encouraged to provide feedback and share experiences with the new features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1885077046663217230">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek R1 and V3 are now available in Windsurf, fully hosted on Western servers.We implemented tool calling in R1, enabling it to be used in a coding agent for the first time.</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1334285824371593227)** (65 messages🔥🔥): 

> `Codeium Issues, DeepSeek vs Sonnet, Windsurf Feature Requests, Cascade Performance, Android Virtual Device` 


- **Users experiencing issues with Windsurf**: Several users reported problems with Windsurf, mentioning that **Claude 3.5 Sonnet** keeps endlessly editing files and prompting errors.
   - One user was advised to download a diagnostic log and report the issue to support via [this link](https://codeium.com/support).
- **DeepSeek often favored over Sonnet**: Discussion surfaced comparing **DeepSeek** and **Sonnet**, with users mentioning that DeepSeek is cheaper and perceived as better by some.
   - One user noted that after testing **R1**, it seems to be on par with **Sonnet** in performance, sparking debate.
- **Feature requests for Windsurf and Cascade**: A user inquired about an auto-commit message feature similar to **VSCode** and **Cursor**, referencing a feature request already submitted.
   - Another user noted the ability to streamline commit processes within the **Cascade** interface to improve workflow efficiency.
- **Performance improvements in Cascade**: Users discussed the recent enhancements to Cascade, which improved prompt speeds during deep conversations and addressed prior slow response issues.
   - One user confirmed that the updates have resolved previous challenges, encouraging others to check the [changelog](https://codeium.com/changelog).
- **Using Android Virtual Device in WS projects**: A user asked how to utilize **Android Virtual Device** for their Windsurf project, prompting another to recommend the `toroxx.vscode-avdmanager` extension.
   - This suggestion reflects community engagement in seeking solutions for integrating Android development tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://codeium.com/changel">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">Auto commit message | Feature Requests | Codeium</a>: Generate Commit Messages from Committed File Context</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1334251746096185384)** (707 messages🔥🔥🔥): 

> `DeepSeek R1 Implementation, Windsurf Performance and Issues, Comparison of AI Models, Pricing and Credits, User Experiences with Windurf and DeepSeek` 


- **DeepSeek R1 Gains Momentum**: Users celebrated the implementation of DeepSeek R1, noting its affordability at **0.25 credits per message**, which allows for increased usage compared to Claude 3.5.
   - Many users expressed excitement over its capabilities, while others encountered issues such as invalid tool calls and internal errors.
- **Windsurf Performance Concerns**: Some users reported that DeepSeek R1 sometimes failed to apply changes correctly, causing frustration with the model's responses when trying to fix code.
   - There are ongoing discussions about the need for further optimizations and how the model handles certain command executions.
- **AI Model Comparisons: R1 vs. Claude 3.5**: Users engaged in a comparison of DeepSeek R1 and Claude 3.5, noting that R1 generally provides a lower-cost option for similar tasks.
   - There was an emphasis on how R1 could be utilized for project planning, while Sonnet was proposed for coding execution.
- **Pricing and Credits System Explained**: Users discussed the pricing model associated with Windurf and the credit consumption for various models, clarifying that **DeepSeek R1 is priced at 0.5 credits per message**.
   - Confusion arose regarding the credit system, but overall clarity was gained about the benefits of using different models for cost-effectiveness.
- **User Experiences with Windurf and DeepSeek**: Multiple users shared their experiences using Windurf and DeepSeek, highlighting successes and challenges in building projects and handling prompts effectively.
   - Despite some issues, many users are optimistic about the potential improvements and value that DeepSeek could bring to their workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/memories#cascade-auto-generated-memories">Cascade Memories</a>: no description found</li><li><a href="https://gitbutler.com/">GitButler | Git Branching, Refined</a>: A Git client for simultaneous branches on top of your existing workflow..</li><li><a href="https://docs.codeium.com/windsurf/memories#workspace-rules">Cascade Memories</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/memories#cascade-auto-generated-">Cascade Memories</a>: no description found</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering#prompt-engineering">Prompt Engineering - Codeium Docs</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/memories#global-rules">Cascade Memories</a>: no description found</li><li><a href="https://codeium.com/security">Security and Privacy | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://gist.github.com/ykka/31f1059764a4be7d4f2f0e2e700da3f5">Windsurf VSCodeVIM Keyboard Shortcuts</a>: Windsurf VSCodeVIM Keyboard Shortcuts. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://status.codeium.com/">Codeium Status</a>: no description found</li><li><a href="https://x.com/EHuanglu/status/1885024173862613485">Tweet from el.cine (@EHuanglu)</a>: China&#39;s AI wins againAlibaba just secretly dropped Qwen 2.5 Max AI, it takes AI video generation to next level and.. it&#39;s free10 examples:1. woman is quarrelling with a man on a crowded street</li><li><a href="https://x.com/OpenRouterAI/status/1884672717460271176?s=19">Tweet from OpenRouter (@OpenRouterAI)</a>: Our data shows that OpenAI o1 thinks for 3X as many tokens as DeepSeek R1 👀o1-mini thinks 50% moreSee the &#34;avg&#34; column in the timeline below:</li><li><a href="https://x.com/EHuanglu/status/1865475635474493665">Tweet from el.cine (@EHuanglu)</a>: Today, AI changed forever.Hunyuan AI has shattered the uncanny valley. It’s so real and I’m stunned. Can you tell it’s AI? Because I can’t!14 Examples:</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://www.reddit.com/r/Codeium/comments/1idq65e/logging_in_to_windsurf_in_an_enterprise_network/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://codeium.com/live/next-js">Chat with NextJS | Windsurf Editor and Codeium extensions</a>: Chat with next-js using Codeium Live. Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://www.youtube.com/watch?v=cBzc5r-FNW0">Use Obsidian (BEST Markdown editor) for note taking and tech docs!</a>: In this video, I&#39;ll show you my favorite markdown tool Obsidian (a free second brain and knowledge base program). I show you how I write my technical documen...</li><li><a href="https://stratechery.com/2025/deepseek-faq/">DeepSeek FAQ</a>: DeepSeek has completely upended people&#8217;s expectations for AI and competition with China. What is it, and why does it matter?
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1334251723430301778)** (474 messages🔥🔥🔥): 

> `DeepSeek vs. OpenAI Models, AI Detectors and Education Solutions, Creative AI Model Performance, Open Source AI Developments, AI Context Windows and Usability` 


- **DeepSeek holds an edge over OpenAI for creative tasks**: Users noted that DeepSeek R1 is performing better than OpenAI's o1 for creative tasks, which has struggled recently.
   - This shift in performance highlights the rising competition among AI models like Gemini Pro and Grok.
- **Concerns over reliability of AI detectors in academia**: Discussion ensued about the inaccuracy of AI detectors, which have led to unjust consequences for students, including misunderstandings in academic settings.
   - Suggestions were made to utilize tools like Google Docs for tracking drafts instead, which could serve as a more reliable solution.
- **Thoughts on eliminating homework for better learning**: A proposal was made to replace traditional homework with more active in-class learning and quizzes to avoid cheating using AI.
   - This approach suggests a shift in educational strategies to enhance engagement and reduce reliance on AI for assignments.
- **Open source AI and its potential for innovation**: There was a consensus that open-sourced models like DeepSeek could lead to significant advancements in AI by allowing broader access and collaboration.
   - Participants argued that this would encourage more innovation compared to closed systems used by large tech companies.
- **Context window differences among AI models**: Users debated the size and usability of context windows in various AI models, with particular emphasis on DeepSeek's advantages in handling large text inputs effectively.
   - Conversations highlighted how different models handle context and user experience, with many expressing preferences based on their needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat.qwenlm.ai/">Qwen Chat</a>: no description found</li><li><a href="https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/">Running DeepSeek R1 on a Raspberry Pi : Global Nerdy</a>: I’m impressed — it turns out that you can run a local copy of DeepSeek R1 on a Raspberry Pi! The photo above shows the large language model of the moment running on my Raspberry Pi 500, which is simpl...</li><li><a href="https://www.youtube.com/watch?v=3QuWqjJ1ZjM">DeepSeek is Copying Existing Al Data, Censoring Results, and Collecting Your Data for China</a>: What do you get when you train your Al on an existing (error-filled) Al, censor it on behalf of a Communist government, and data mine your users? You get Dee...</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.</li><li><a href="https://bsky.app/profile/jeffgeerling.com/post/3lgt3a76mws2v">Jeff Geerling (@jeffgeerling.com)</a>: Apropos of nothing... there&#39;s a little 16GB Pi 5 off to the left.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1334266029370769409)** (68 messages🔥🔥): 

> `Next Generation AI Instructions, Memory Function in GPT Models, API and Custom GPT Limitations, OpenAI's Model Release Intentions, Fine-tuning Ollama Models` 


- **Next Generation AI hopefully follows instructions**: A user suggested that rather than performing text manipulation, AI models should execute a tool that does the manipulation, ensuring no modifications are made to the responses.
   - By creating a link processor function that formats responses using Markdown, a more consistent result may be achieved through the Custom GPT feature.
- **Discussions on Memory Functionality**: Users shared experiences about memory features in GPT models, noting that while memory is supposed to enhance context awareness, it often fails to do so, especially in lengthy discussions.
   - Concerns were raised that as discussions extend, critical details might scroll out of context, affecting the model's ability to recall important information.
- **API and Custom GPT Scope Challenges**: One user reported challenges with using the API for a project with a railway company, citing inconsistent behavior of the custom GPT in scraping links.
   - Despite trying multiple API solutions, the reliability of link transmission remains a significant issue.
- **Debate on OpenAI Model Release Intentions**: Various opinions emerged regarding whether OpenAI releases models in bad faith, particularly in light of efforts to slow down competitor models.
   - Questions arose about the effectiveness and completeness of models published on GitHub, and whether they were intentionally broken to mislead developers.
- **Interest in Fine-tuning Ollama Models**: A user inquired on how to fine-tune Ollama, possibly seeking guidance on improving model outputs.
   - This aligns with broader interests in customizing AI models for specific applications.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1334314125794344991)** (25 messages🔥): 

> `AI Problem-Solving Limitations, Issues with Visual Recognition, Prompt Construction Tools, Understanding of Math Puzzles` 


- **AI Struggles to Solve Problems**: A member mentioned they are a high IQ individual but still can't solve certain problems, sparking discussions about cognitive tests and their nature.
   - Another member emphasized their competence in creative problem-solving, highlighting the complexity of some AI challenges.
- **Frustrations with AI Behavior**: Concerns were raised about AI not adhering to desired output characteristics, specifically in terms of maintaining proper word count and output length.
   - A member pointed out that low-quality outputs affect future responses, indicating a flaw in the interaction design.
- **Discussion on Math Puzzles**: A user questioned the nature of a specific problem posed involving an owl and palm tree, contemplating its cognitive testing aspect.
   - There was a shared link to a chat discussing this math problem, clarifying it involved solving a system of equations.
- **New Tools for Prompt Construction**: One member introduced 'OneClickPrompts', an extension designed to help construct prompts from multiple parts for quick access to frequently used prompts.
   - An accompanying GIF provided a visual representation of how the tool operates, enhancing usability insights.
- **Vision Model Improvements**: A discussion noted previous limitations in the AI's visual recognition capabilities, where it struggled to distinguish between certain graphical elements.
   - Members shared experiences of training the model with specific problems, indicating that ongoing feedback might have led to improvements.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1334314125794344991)** (25 messages🔥): 

> `Challenges with AI Problem Solving, AI Response Length and Quality, Vision Model Limitations, OneClickPrompts Extension, Algebra Discussion on Social Media` 


- **High IQ but Struggling with Problems**: A member questioned why, despite being a high IQ individual, they face limitations in solving some problems, highlighting their proficiency in creative problem solving.
   - This raised a discussion around specific tasks like the owl/palm tree problem and the nature of cognitive challenges faced.
- **Desire for Consistent AI Output Length**: Members expressed frustration over the AI's inconsistency in output length and quality, particularly when it doesn’t meet the desired criteria.
   - One noted that the AI tends to assume substandard outputs are acceptable, impacting future generations negatively.
- **Vision Models' Distinguishing Challenges**: Concerns were raised about the AI's vision model, which struggled to differentiate elements within images, such as the ground from lines above a figure.
   - Historical discussions indicated that specific issues were diagnosed months prior, indicating potential enhancements in AI learning over time.
- **OneClickPrompts Aids Efficient Prompt Creation**: A new extension called OneClickPrompts was introduced, designed to help users construct prompts from multiple parts for quick access.
   - A GIF demonstrating the functionality was shared, providing a visual understanding of its capabilities.
- **Algebra Puzzle Growing Popular**: A member noted the prevalence of algebra discussions on platforms like TikTok, emphasizing the power of social media in fostering math conversations.
   - Others commented on the perceived effectiveness of the AI in solving relatively simple algebraic problems, despite potential inaccuracies.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1334641669437657189)** (1 messages): 

> `LM Studio 0.3.9 features, Idle TTL functionality, Reasoning content in API responses, Auto-update for LM runtimes, Support for nested folders in Hugging Face` 


- **Exciting Features in LM Studio 0.3.9**: LM Studio 0.3.9 introduces several new features, including **Idle TTL** for managing API model memory efficiently and support for downloading models from **nested folders** in Hugging Face repositories.
   - The update is available via in-app update or from [here](https://lmstudio.ai/download) and comes with various bug fixes for better user experience.
- **Introducing Idle TTL for Smart Memory Management**: With **Idle TTL**, users can set a time-to-live for API models and automatically evict old models, enhancing the management of memory resources in LM Studio.
   - Documentation for the feature is detailed in the [docs](https://lmstudio.ai/docs/api/ttl-and-auto-evict) for users to optimize usage further.
- **Separate Reasoning Content Feature Launched**: The new **reasoning_content** field in API responses allows users to access reasoning details separately, akin to DeepSeek's API, turned on via Settings.
   - This experimental feature enhances the information gleaned during chat completions, aligning closely with developers' needs.
- **New Auto-Update Feature for Runtimes**: LM Studio now supports **auto-update** for runtimes to streamline updating processes, reducing the hassle for users needing to update multiple components.
   - This feature is enabled by default but can be adjusted in App Settings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/api/ttl-and-auto-evict">Idle TTL and Auto-Evict | LM Studio Docs</a>: Optionally auto-unload idle models after a certain amount of time (TTL)</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.9">LM Studio 0.3.9</a>: Idle TTL, auto-update for runtimes, support for nested folders in HF repos, and separate `reasoning_content` in chat completion responses
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1334256243472203778)** (308 messages🔥🔥): 

> `DeepSeek Models, LM Studio Features, RAG in LM Studio, Model Performance and Reasoning, API and UI Discussion` 


- **Discussion on DeepSeek Model Compatibility**: Users reported issues loading DeepSeek models, particularly with error messages related to pre-tokenizer types. Recommendations included updating LM Studio to the latest version and verifying runtime updates via CTRL + SHIFT + R.
   - A user mentioned error messages indicating problems with model vocabulary, prompting others to encourage version updates for resolution.
- **RAG Capabilities in LM Studio**: Users inquired about the ability of LM Studio to facilitate Retrieval-Augmented Generation (RAG) by attaching documents for context. The documentation indicated that LM Studio does support RAG by allowing users to attach document files to chat sessions.
   - Clarifications about RAG indicated that if a document's content fits within the model's context, it could be added in full for conversation enhancement.
- **Model Performance and Reasoning Capability**: Discussion around various models' abilities in reasoning highlighted that specific models support advanced reasoning while others do not. Factors influencing performance included model size and whether they fit entirely in GPU memory.
   - Users requested recommendations for models effective at reasoning, where certain models were identified to offer better logic capabilities in handling tasks, particularly in programming.
- **Customization and UI Tweaks**: Users expressed interest in the potential for customizable themes and CSS for LM Studio to enhance UI flexibility. The LM Studio team acknowledged this as a future feature they plan to implement.
   - Additional discussions emerged regarding the application's structure, with some noting the client is not open-source but the CLI tools are available.
- **General Praise for LM Studio's Progress**: Users expressed overall satisfaction with the advancements in LM Studio, noting improvements in functionality and user experience. Conversations highlighted a strong community interest in improving local LLM applications and integrating advanced features.
   - Amid technical discussions, there was a shared enthusiasm for utilizing powerful models like Qwen-2.5 and others to push the boundaries of what could be accomplished with local LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-small-3">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: no description found</li><li><a href="https://www.anandtech.com/show/21111/amd-unveils-ryzen-7040u-series-with-zen-4c-smaller-cores-bigger-efficiency">AMD Unveils Ryzen Mobile 7040U Series with Zen 4c: Smaller Cores, Bigger Efficiency</a>: no description found</li><li><a href="https://x.com/MistralAI/status/1884967826215059681">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://tenor.com/view/eye-of-sauron-lotr-lord-of-the-rings-gif-16715227">Eye Of Sauron Lotr GIF - Eye Of Sauron Lotr Lord Of The Rings - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/better-call-saul-call-saul-its-showtime-folks-gif-8557719">Better Call Saul Its Showtime Folks GIF - Better Call Saul Call Saul Its Showtime Folks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kryonax-skull-gif-26476587">Kryonax Skull GIF - Kryonax Skull - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents | LM Studio Docs</a>: How to provide local documents to an LLM as additional context</li><li><a href="https://tenor.com/view/fallout-tv-fallout-codsworth-fallout-prime-fallout-amazon-gif-14576962590525720544">Fallout Tv Codsworth GIF - Fallout tv Fallout Codsworth - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.anandtech.com/show/21111/amd-unveils-ryzen-7040u-series-with-zen-4c-smaller-cores-bigger">AMD Unveils Ryzen Mobile 7040U Series with Zen 4c: Smaller Cores, Bigger Efficiency</a>: no description found</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://youtu.be/yFKOOK6qqT8?si=EgnAQF3mVXWcElgH">Deepseek R1 671b Running LOCAL AI LLM is a ChatGPT Killer!</a>: Writeup for Deepseek R1 671b Setup and Running Locally https://digitalspaceport.com/running-deepseek-r1-locally-not-a-distilled-qwen-or-llama/768GB RAM or VR...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/256#issuecomment-2620673643">There was an error fetching results from Hugging Face, please try again in a little bit · Issue #256 · lmstudio-ai/lmstudio-bug-tracker</a>: Use latest version, can&#39;t search and download modules from hugging face, hit me as follow: There was an error fetching results from Hugging Face, please try again in a little bit How to set lm stu...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 👾 LM Studio CLI</a>: 👾 LM Studio CLI. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/256#issuecomment-262">There was an error fetching results from Hugging Face, please try again in a little bit · Issue #256 · lmstudio-ai/lmstudio-bug-tracker</a>: Use latest version, can&#39;t search and download modules from hugging face, hit me as follow: There was an error fetching results from Hugging Face, please try again in a little bit How to set lm stu...</li><li><a href="https://lmstudio.ai/docs/api">LM Studio as a Local LLM API Server | LM Studio Docs</a>: Run an LLM API server on localhost with LM Studio
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1334257962298118235)** (203 messages🔥🔥): 

> `DeepSeek Model Performance, Jetson Nano Discussion, Model Selection for Coding, Hardware Configuration for AI, Temperature Settings for Coding` 


- **DeepSeek Model Performance Insights**: Users reported that running DeepSeek models with configurations like **GTX 1080** and **Ryzen 5 3600** yield about **6-7 tokens per second**, regardless of thread pool size or GPU offload settings.
   - Adjusting model size and ensuring fit within VRAM are crucial, as exceeding VRAM can significantly reduce performance.
- **Discussion on Jetson Nano Pricing**: The **Jetson Nano** was discussed with remarks about its high price range of **$500-$700**, leading many to consider alternatives like real GPUs.
   - Participants highlighted that Jetson Nano appears to be on backorder, but some sellers list it around **$250**.
- **Choosing the Right Model for Coding**: Comparisons were made regarding the performance of smaller models such as **32B** and **70B** models, with remarks that both can handle complex coding tasks effectively.
   - Users indicated that while smaller models perform adequately, they recommend checking benchmarks on platforms like Hugging Face to gauge expected performance.
- **Optimizing AI Workstation Hardware**: Configurations like **i9-14900KF** with **128GB RAM** and dual **RTX 4090 GPUs** can effectively run **DeepSeek 70B** models at **30-40 tokens/sec** with the right quantizations.
   - Users noted the importance of ensuring models fit within available VRAM to maintain optimal performance.
- **Setting Temperature for Better AI Output**: Participants emphasized the importance of setting model temperature, recommending values between **0.5-0.7** to prevent excessive repetition in coding prompts.
   - A lower temperature can enhance output coherence, especially when using models like **DeepSeek** for coding tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gyazo.com/b1903225526b9ea0039cdd6674d8ced8">Gyazo</a>:  </li><li><a href="https://www.theregister.com/2024/07/15/amd_ai_pc_goal/">AMD says future PCs will run 30B parameter models at 100T/s</a>: They&#39;re gonna need a heck of a lot of memory bandwidth – not to mention capacity – to do it</li><li><a href="https://tenor.com/view/whale-swallow-eat-nom-hungry-gif-17097355">Whale Swallow GIF - Whale Swallow Eat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.tomshardware.com/pc-components/cpus/amd-launches-ryzen-ai-300-and-200-series-chips-for-laptops">AMD launches Ryzen AI 300 and 200 series chips for laptops</a>: Ryzen AI spreads its wings.</li><li><a href="https://www.laptopmag.com/laptops/asus-zenbook-s16-um5606-ryzen-ai-9">Asus Zenbook S16 UM5606 (Ryzen AI 9) review: I think AMD just killed the MacBook Air</a>: A laptop with integrated graphics that I can game on? Wild.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1334252552786808965)** (430 messages🔥🔥🔥): 

> `DeepSeek R1 performance, O1 Pro usage, Aider integration with local models, O3 Mini release, Quantization effects on models` 


- **Debate on DeepSeek R1 Speed and Performance**: Users discussed the performance of DeepSeek R1, noting that the model's execution speed varied significantly when run on different hardware setups, such as a 4090 GPU reaching approximately 32 TPS.
   - The quantized versions of models were criticized for slow speeds and poor instruction-following capabilities, raising concerns about their practicality.
- **O1 Pro as a Coding Tool**: Some users expressed that O1 Pro is great for coding both new projects and making modifications to existing codebases, leading to debates about pricing and its overall value for heavy users.
   - Despite its benefits, there were discussions about the limitations posed by usage costs and censorship when compared to local models.
- **Aider Integration with Local Models**: Concerns were raised about sending data to external services like DeepSeek due to data privacy issues, especially when using models hosted in China.
   - Users are exploring ways to leverage local models such as Ollama for Aider applications without compromising sensitive data.
- **Anticipation for O3 Mini Release**: Participants are eagerly awaiting the release of O3 Mini, with some speculating it could enhance their AI model experiences and address some of the performance shortcomings of existing options.
   - There were humorous comments shared about waiting for O3 Mini, seen as a potential game-changer in the ongoing search for better model performance.
- **Effects of Model Quantization**: Discussions revealed that quantization can significantly impact model performance, leading to questions around the balance between size and quality of outputs.
   - Participants shared experiences with different quantized versions of models, noting variability in quality and instruction-following success across setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1131200896827654144/1133060684540813372/1332845238267412480">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">Tweet from Aravind Srinivas (@AravSrinivas)</a>: All Perplexity Pro users now get 500 daily DeepSeek R1 queries (without censorship and prompts not going to China). Free users get 5 daily queries.Quoting Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://one.npr.org/?sharedMediaId=nx-s1-5279550:nx-s1-5343701-1">&#x1f50a; Listen Now: OpenAI touts new government partnership and support for A.I. infrastructure</a>: All Things Considered on NPR One | 7:59</li><li><a href="https://www.youtube.com/channel/UCC0O5FKSMcjzrvOUa08hS_A/videos">Michael Automates</a>: I help thousands learn how to automate their crypto in order to increase their chances or success. To benefit from the upside and protect from the downside, fully automatically.This helps 95% of inves...</li><li><a href="https://tenor.com/view/nalog-gif-25906765">Nalog GIF - Nalog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/wiz_io/status/1884707816935391703">Tweet from Wiz (@wiz_io)</a>: BREAKING: Internal #DeepSeek database publicly exposed 🚨Wiz Research has discovered &#34;DeepLeak&#34; - a publicly accessible ClickHouse database belonging to DeepSeek, exposing highly sensitive inf...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://x.com/allen_ai/status/1884966600039915809">Tweet from Ai2 (@allen_ai)</a>: Here is Tülu 3 405B 🐫 our open-source post-training model that surpasses the performance of DeepSeek-V3! The last member of the Tülu 3 family demonstrates that our recipe, which includes Reinforcemen...</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://x.com/MistralAI/status/1884968836606136636">Tweet from Mistral AI (@MistralAI)</a>: Introducing Small 3, our most efficient and versatile model yet! Pre-trained and instructed version, Apache 2.0, 24B, 81% MMLU, 150 tok/s. No synthetic data so great base for anything reasoning - happ...</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API, Providers, Stats</a>: Sonar Reasoning is a reasoning model provided by Perplexity based on [DeepSeek R1](/deepseek/deepseek-r1).It allows developers to utilize long chain of thought with built-in web search. Run Sonar Reas...</li><li><a href="https://github.com/Aider-AI/aider/pull/3074/files.">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API, Providers, Stats</a>: DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3. Run DeepSeek R1 Distill Llama 70B with API</li><li><a href="https://github.com/marketplace/models/azureml-deepseek/DeepSeek-R1">DeepSeek-R1 · GitHub Models · GitHub</a>: Create AI-powered applications with DeepSeek-R1</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">DeepSeek R1 (free) - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://x.com/jacksonhinklle/status/1884686222356079075">Tweet from Jackson Hinkle 🇺🇸 (@jacksonhinklle)</a>: 🚨🇨🇳🇺🇸 BREAKING: CHINA HAS DEFEATED AMERICA🇨🇳 DeepSeeak BEATS OpenAI 🇺🇸🇨🇳 BYD BEATS Tesla 🇺🇸🇨🇳 Huawei BEATS Apple 🇺🇸🇨🇳 Huawei BEATS US Telecoms 🇺🇸🇨🇳 Alibaba BEATS Amazon 🇺🇸🇨🇳...</li><li><a href="https://aimlapi.com/">Access 200+ AI Models with a Single AI API | AIMLAPI.com</a>: Access over 200 AI models with low latency and high scalability AI APIs. Save up to 80% compared to OpenAI. Fast, cost-efficient, and perfect for advanced machine learning projects. AI Playground.</li><li><a href="https://fourthievesvinegar.org/">Four Thieves Vinegar Collective</a>: Right to Repair–for Your Body. The Four Thieves Vinegar Collective is an anarchist collective dedicated to enabling access to medicines and medical technologies to those who need them but don’t have t...</li><li><a href="https://github.com/bytedance/UI-TARS?">GitHub - bytedance/UI-TARS</a>: Contribute to bytedance/UI-TARS development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1334274075027312777)** (45 messages🔥): 

> `Aider context inclusion, Azure AI deployment issues, Model configuration challenges, Using Aider in different modes, File creation prompts in Aider` 


- **User queries about Aider's context inclusion**: A user asked if there's a feature to automatically include files related to the current editing file within Aider for better prompts.
   - It was mentioned that files can be manually read into the chat and another member discussed modifying files in architect mode.
- **Confusion with Azure AI deployment endpoints**: A member expressed difficulty in confirming which multiple endpoints and keys are required for Azure R1 deployments and faced errors connecting Aider.
   - Suggestions included checking GitHub issues for solutions and trying the alternative 'azure_ai' implementation within Aider for testing.
- **Configuring models in Aider**: Users discussed ways to set different models for various commands in Aider to optimize performance, particularly the default vs specific model usage.
   - One member suggested maintaining a good coding model for general use, while switching to intelligent models only for complex tasks.
- **Operating Aider in chat-only mode**: A user inquired about using Aider solely for chat without involving any code, to focus on project-related discussions.
   - Another member recommended using the '/reset' command to prevent code from being added to prompts.
- **File creation prompts leading to confusion**: Users reported Aider intermittently trying to create files with random names or code snippets, causing frustration.
   - There was commentary on how Aider's editor sometimes misinterpreted inputs, leading to unwanted prompts for file creation.



**Link mentioned**: <a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">Advanced model settings</a>: Configuring advanced settings for LLMs.

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1334313324015521802)** (11 messages🔥): 

> `DeepSeek database leak, Aider Read-Only Stubs, Aider Awesome GitHub Repository, Pull Request Improvements, Bash One-Liners` 


- **DeepSeek Database Exposes Sensitive Info**: The **DeepSeek** database has been reported leaking sensitive information, including user chat histories, with details discussed on [Hacker News](https://news.ycombinator.com/item?id=42871371).
   - *Users expressed concerns about data privacy and security due to the breach.*
- **New YouTube Video on Aider's Features**: A recent [YouTube video](https://youtu.be/XE6v_RGe0-U) titled 'Navigating Large Codebases: Aider's Read-Only Stub Solution' discusses enhancements for AI coding with read-only stubs in Aider.
   - This video focuses on the new draft feature aimed at improving AI interactions with large codebases.
- **Gathering One-Liner and Prompt Suggestions for Aider**: 'Aider Awesome' is a GitHub repository proposed by a member to collect useful one-liner and prompt suggestions specifically for aide-chat, aimed at enhancing user experiences [GitHub - hux/aider-awesome](https://github.com/hux/aider-awesome).
   - Feedback on the repository includes suggestions for making the content more readable.
- **Pull Request Merges for Aider Awesome**: A user pointed out a [pull request](https://github.com/hux/aider-awesome/pull/1) that aimed to improve the readability of the Aider Awesome repository.
   - The pull request was merged successfully, with contributors sharing their satisfaction in the process.
- **Discussion on Bash One-Liners**: A member expressed a preference for using **one-liners** in **bash**, emphasizing their efficiency as a single command.
   - The conversation highlighted the simplicity and effectiveness of using concise command strategies in scripting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/hux/aider-awesome">GitHub - hux/aider-awesome: Prompts and helpers that work well with aider-chat</a>: Prompts and helpers that work well with aider-chat  - GitHub - hux/aider-awesome: Prompts and helpers that work well with aider-chat</li><li><a href="https://youtu.be/XE6v_RGe0-U">Navigating Large Codebases: Aider&#39;s Read-Only Stub Solution (re-upload)</a>: Enhancing AI Coding with Read-Only Stubs in AiderIn this episode, we delve into a new draft-feature for the AI assistant Aider that allows the inclusion of r...</li><li><a href="https://github.com/hux/aider-awesome/pull/1">Update README.md by alexanderkjeldaas · Pull Request #1 · hux/aider-awesome</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1334251977583886399)** (456 messages🔥🔥🔥): 

> `DeepSeek R1, MCP Support, Token Usage in Chat and Composer, Local Models, Security Risks in AI Models` 


- **DeepSeek R1 adds new features**: Windsurf announced that DeepSeek R1 and V3 are now available in their composer feature, fully hosted on Western servers.
   - The update includes tool calling capabilities, allowing R1 to be used in coding agent mode for the first time.
- **Issues with Token Usage Visibility**: Users expressed concern about the lack of visibility regarding token usage in chat and composer, with confusion surrounding the 10k token context limit.
   - There are questions about the effectiveness of beta settings for enabling longer context limits.
- **MCP Server Configuration**: Users discussed using a bash script for configuring MCP servers, enabling the addition of various JSON settings easily with Cursor.
   - This method allows users to run different MCP servers without needing extensive configuration each time.
- **Potential Security Risks in AI Models**: Concerns were raised about potential security risks in machine learning models, including the possibility of hidden code execution within payloads.
   - Tools like modelscan are recommended for checking against serialization attacks to ensure safety when running local models.
- **Local vs. Hosted AI Models**: A discussion highlighted the challenges of running local models compared to hosted options like DeepSeek R1, with privacy considerations impacting user preferences.
   - While some users are hopeful for better local model integrations, others remain skeptical about performance and reliability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1885077046663217230">Tweet from Windsurf (@windsurf_ai)</a>: DeepSeek R1 and V3 are now available in Windsurf, fully hosted on Western servers.We implemented tool calling in R1, enabling it to be used in a coding agent for the first time.</li><li><a href="https://dev.to/krisplatis/auto-add-missing-imports-on-file-save-in-vs-code-1b89">no title found</a>: no description found</li><li><a href="https://www.securityweek.com/unprotected-deepseek-database-leaked-highly-sensitive-information/">Unprotected DeepSeek Database Exposed Chats, Other Sensitive Information</a>: An unprotected database belonging to Chinese AI company DeepSeek exposed highly sensitive information, including chat history, secret keys, and backend data.</li><li><a href="https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/">Data Scientists Targeted by Malicious Hugging Face ML Models with Silent Backdoor</a>: Is Hugging Face the target of model-based attacks? See a detailed explanation of the attack mechanism and what is required to identify real threats &gt;</li><li><a href="https://forum.cursor.com/latest">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet set SOTA on aider’s polyglot benchmark</a>: R1+Sonnet has set a new SOTA on the aider polyglot benchmark. At 14X less cost compared to o1.</li><li><a href="https://forum.cursor.com/t/sonnet-3-5-stops-working/46053">Sonnet 3.5 stops working</a>: When I enable the OpenAI API key but not Anthropic, it still tries to do a custom API call to the server. I expect it to only do OpenAI models and not Anthopic. If I disable Openai it does work on ant...</li><li><a href="https://forum.cursor.com/t/cursor-does-not-send-files-to-claude/43948/6">Cursor does not send files to Claude</a>: Today I randomly lost the ability to share any files via @ with Cursor.  I checked the logs but I don’t see any debug or error information.  I tried new sessions, restarting my machine, and even upgra...</li><li><a href="https://forum.cursor.com/t/o3-mini-support/46324">O3-mini support</a>: OpenAI will drop o3-mini in about 30 minutes, how can we use this model out of the gate? Please give me any hacks to use OpenRouter/direct API calls so I I know what to do when it drops</li><li><a href="https://www.mcpservers.ai/">MCP Servers</a>: Browse the largest library of Model Context Protocol Servers. Share Model Context Protocol Servers you create with others.</li><li><a href="https://www.hailuo.ai/">Hailuo AI - Your Ultimate AI Assistant for Intelligent Solutions</a>: Discover Hailuo AI, your go-to AI assistant that offers advanced solutions across various domains including AI Search, Vision, Voice Chat, and more. Experience fast, accurate information retrieval and...</li><li><a href="https://x.com/imrat/status/1884904074379759872?s=46">Tweet from Imrat (@imrat)</a>: Cursor 0.45.6 - adds MCP support - cant wait to try this!</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: no description found</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1idzrdl/o3_releasing_tomorrow/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/daniel-lxs/mcp-server-starter">GitHub - daniel-lxs/mcp-server-starter</a>: Contribute to daniel-lxs/mcp-server-starter development by creating an account on GitHub.</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://github.com/protectai/modelscan">GitHub - protectai/modelscan: Protection against Model Serialization Attacks</a>: Protection against Model Serialization Attacks. Contribute to protectai/modelscan development by creating an account on GitHub.</li><li><a href="https://github.com/microsoft/BitNet">GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs</a>: Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/supabase-community/mcp-supabase">GitHub - supabase-community/mcp-supabase: A collection of MCP servers that connect LLMs to Supabase</a>: A collection of MCP servers that connect LLMs to Supabase - supabase-community/mcp-supabase</li><li><a href="https://pureinsights.com/blog/2024/1-bit-llms-the-future-of-efficient-ai/">1-Bit LLMs: The Future of Efficient AI? - Pureinsights</a>: This blog explains the initial research on 1-bit llms and their potential for producing AI models that are effective but also efficient.</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>: no description found</li><li><a href="https://openrouter.ai/minimax/minimax-01">MiniMax-01 - API, Providers, Stats</a>: MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 for image understanding. It has 456 billion parameters, with 45. Run MiniMax-01 with API
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1334252639428284528)** (295 messages🔥🔥): 

> `Nous x Solana Event, New Model Releases, Psyche and Distributed Learning, Mistral Small Model Announcement, Community Insights on AI Agents` 


- **Nous x Solana Event Creates Buzz**: The upcoming Nous x Solana event in NYC is heavily attended, leading to numerous requests for attendance approval amid limited capacity.
   - Attendees expressed enthusiasm for potential discussions around infrastructure for distributed training in AI models.
- **Excitement for Model Releases**: Community members are eagerly discussing the new models being released, including Mistral Small, which claims to set a new benchmark among smaller language models.
   - Many participants are hoping for availability and performance comparisons against existing models.
- **Psyche Introduces Distributed Learning Infrastructure**: Nous announced Psyche, a distributed training network for open AI models, which aims to facilitate large-scale Reinforcement Learning (RL) through a modular system.
   - The project received positive feedback for its potential to innovate AI training methodologies.
- **End-user Collaboration and Open Source Developments**: There is an ongoing discussion regarding the open-sourcing of Psyche, with future plans for potential consensus algorithms and other resources.
   - Community desires for better accessibility through GitHub are evident, alongside inquiries about specific channels for Psyche.
- **Discussion Around AI Agents Tied to Nous**: Members expressed interest in knowing whether any AI agents are currently associated with Nous and if there is a list available.
   - The community continues to explore the implications of AI development within the Nous ecosystem.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/allen_ai/status/1884966600039915809">Tweet from Ai2 (@allen_ai)</a>: Here is Tülu 3 405B 🐫 our open-source post-training model that surpasses the performance of DeepSeek-V3! The last member of the Tülu 3 family demonstrates that our recipe, which includes Reinforcemen...</li><li><a href="https://arcprize.org/blog/r1-zero-r1-results-analysis">R1-Zero and R1 Results and Analysis</a>: An analysis of Deepseek's R1</li><li><a href="https://tenor.com/view/popcorn-minions-popcorn-day-laugh-gif-5026739">Minion Spitting Out Popcorn GIF - Popcorn Minions Popcorn Day - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2402.03804">ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs</a>: Sparse computation offers a compelling solution for the inference of Large Language Models (LLMs) in low-resource scenarios by dynamically skipping the computation of inactive neurons. While tradition...</li><li><a href="https://x.com/BlinkDL_AI/status/1884768989743882276">Tweet from BlinkDL (@BlinkDL_AI)</a>: I propose ZeroCoT: a simple method to bootstrap CoT from zero. Let me know what you think 🙂 I will try this using RWKV soon.Quoting BlinkDL (@BlinkDL_AI) Let&#39;s kill attention. RWKV-7 &#34;Goose&#...</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>: no description found</li><li><a href="https://x.com/Teknium1/status/1884740956911718853?t=0NwHRMjFT001dlRoRvAPUw&s=19">Tweet from Teknium (e/λ) (@Teknium1)</a>: @ylecun https://x.com/Teknium1/status/1883955152442515637Quoting Teknium (e/λ) (@Teknium1) Today Nous announced the coming of Psyche - a distributed network and training framework, an infrastructure l...</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">no title found</a>: no description found</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k">open-thoughts/OpenThoughts-114k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/robertcsordas/moe_layer">GitHub - RobertCsordas/moe_layer: sigma-MoE layer</a>: sigma-MoE layer. Contribute to RobertCsordas/moe_layer development by creating an account on GitHub.</li><li><a href="https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf">granite-3.0-language-models/paper.pdf at main · ibm-granite/granite-3.0-language-models</a>: Contribute to ibm-granite/granite-3.0-language-models development by creating an account on GitHub.</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 is now available on Azure AI Foundry and GitHub | Microsoft Azure Blog</a>: DeepSeek R1, available through the model catalog on Microsoft Azure AI Foundry and GitHub, enables businesses to seamlessly integrate advanced AI.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1334647219881640029)** (1 messages): 

> `Autoregressive generation on CLIP embeddings, Multimodal inputs, Stable Diffusion generation` 


- **Exploring Autoregressive Generation on CLIP Embeddings**: A member questioned the feasibility of performing **autoregressive generation** on **CLIP embeddings**, which are used to project multimodal inputs into a single latent space.
   - They noted a lack of information on this method specifically for generation, highlighting its more common use for guidance in **Stable Diffusion** generation.
- **Understanding CLIP and Multimodal Integration**: The discussion revolved around the basics of how **CLIP** effectively integrates various modalities into a unified approach.
   - Despite its applications in guiding models, the conversation pointed to a need for more insights on leveraging CLIP for generative processes.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1334360329152630936)** (2 messages): 

> `China's AI Models, AI Race, Top-tier Models` 


- **China boasts TEN top-tier AI models**: A discussion revealed that **China's** AI landscape includes **TEN top-tier models** trained from scratch, matching or exceeding the capabilities of **Europe's** largest models, including **Mistral**.
   - One member highlighted that *China's only good AI model is not DeepSeek*, showcasing the breadth of development happening outside the US.
- **US AI labs in competitive landscape**: The US is home to only **five major AI labs**—**OpenAI**, **Anthropic**, **Google**, **Meta**, and **xAI**—that are competitive at this scale in the AI arena.
   - This brief underscores that the **AI race is very much on**, with implications on global leadership in AI development.



**Link mentioned**: <a href="https://x.com/deedydas/status/1884786839913111931">Tweet from Deedy (@deedydas)</a>: China&#39;s only good AI model is not DeepSeek.There are TEN top tier models all trained from scratch (equal to or better than Europe / Mistral&#39;s biggest model).The US has only 5 labs—OpenAI, Anth...

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1334269194270670909)** (170 messages🔥🔥): 

> `Reinforcement Learning vs. Deep Learning, DeepSeek developments, Learning strategies in LLMs, Pretraining and fine-tuning frameworks, Educational analogies for LLM training` 


- **Reinforcement Learning's Nuances**: Red_code argued that modern **Reinforcement Learning (RL)** should leverage existing knowledge and enhance reasoning, moving beyond traditional trial-and-error methods.
   - Zickzack countered that while representation learning is not new, RL's unique perspective allows it to address credit assignment and memory more effectively.
- **DeepSeek's Potential**: There was excitement about **DeepSeek**'s recent performance improvements, with discussions focusing on its ability to achieve results efficiently compared to classic models.
   - Red_code noted that their goal is to optimize reasoning and representation learning using the prospective configuration idea.
- **Learning Strategies in LLMs**: Albert_lum shared insights about integrating **prior knowledge** into RL, emphasizing that proper learning strategies could enhance RL capabilities.
   - The conversation highlighted the importance of differentiating between DL and RL, and how both can complement each other.
- **Educational Framework for LLMs**: Erkinalp introduced a framework for understanding LLM training by comparing textbook structures, outlining three major types of information: background, demonstration, and practice problems.
   - He stressed that while LLMs have extensive exposure to the first two types, the incorporation of practice problems represents a new frontier for meaningful learning.
- **Community Engagement and Humor**: Members expressed their enjoyment of the community, with comments about their interest in reading research papers related to AI.
   - Albert_lum added humor by suggesting calling people they dislike a 'segmentation fault', showcasing a light-hearted atmosphere in the discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1884967826215059681?t=VAZyVB2GkDpC_KOy_2mZCQ&s=19">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://x.com/atroyn/status/1884695801773060502">Tweet from anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn)</a>: i am at @aixventureshq for @chrmanning’s talk about deepseek r1. let’s find out what’s going on.</li><li><a href="https://x.com/karpathy/status/1885026028428681698">Tweet from Andrej Karpathy (@karpathy)</a>: We have to take the LLMs to school.When you open any textbook, you&#39;ll see three major types of information:1. Background information / exposition. The meat of the textbook that explains concepts. ...</li><li><a href="https://x.com/karpathy/status/1883941452738355376">Tweet from Andrej Karpathy (@karpathy)</a>: I don&#39;t have too too much to add on top of this earlier post on V3 and I think it applies to R1 too (which is the more recent, thinking equivalent).I will say that Deep Learning has a legendary ra...</li><li><a href="https://youtu.be/lYWIkwvaUIg?t=150">Kristin Bauer - True Blood Season 5 Episode 10: «Gone, Gone, Gone» [Full]</a>: Like: http://facebook.com/kristinbauerfansActress: Kristin Bauer as PamTV Serie: True BloodSeason: number 5Episode: number 10: «Gone, Gone, Gone»**No copyrig...</li><li><a href="https://youtu.be/W_jPWjzzuaw?t=20">Kristin Bauer - True Blood Season 5 Episode 9: « Everybody Wants To Rule The World» [Full]</a>: Like: http://facebook.com/kristinbauerfansActress: Kristin Bauer as PamTV Serie: True BloodSeason: number 5Episode: number 9: « Everybody Wants To Rule The W...</li><li><a href="https://www.nature.com/articles/s41593-023-01514-1">Inferring neural activity before plasticity as a foundation for learning beyond backpropagation - Nature Neuroscience</a>: This paper introduces &#8216;prospective configuration&#8217;, a new principle for learning in neural networks, which differs from backpropagation and is more efficient in learning and more consistent...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1334253123488841738)** (39 messages🔥): 

> `OpenAI Allegations, AI Technology Concerns, Dario Amodei's Blog Post, AI Safety Funding, Daily Paper Discussion` 


- **Allegations Against OpenAI Raises Eyebrows**: Members debated allegations against OpenAI, with one suggesting it resembles a *smear job* rather than constructive criticism. The ongoing discussions point towards a backdrop of a major cyber attack, indicating that **government sentiments** may be influencing public perception.
   - *“OpenAI is up shit creek,”* one member remarked, hinting at a sense of urgency in their legal maneuvering amid scrutiny.
- **Dario Amodei Under Fire**: Dario Amodei was criticized as part of the ongoing discourse, with remarks labeling him as one of the *most obvious frauds* in AI. Commentary related to his recent **$1B** fundraising effort for *AI Safety* was seen as dubious by some members.
   - The sentiment that *“he is less diversified than Scam Altman”* reflects a skepticism towards the intentions behind his actions.
- **Concerns Over AI Coding Quality**: A discussion emerged about the effectiveness of AI in professional software development, with one member asserting that **models are below required quality**. Others echoed that while models like Claude can generate code, they often **over-complicate** tasks and fail to maintain context.
   - *“It’s usually more work to get Claude to output correct code than to type it yourself,”* encapsulates the frustration many feel regarding current AI capabilities.
- **Engagement in Daily Discussions**: A few members reflected on their engagement levels in daily discussions, indicating they often feel well-informed about ongoing topics due to regular participation. One humorous remark suggested that discussions indeed make participants feel like *hackers* in their knowledge base.
   - A new member expressed curiosity about joining future paper reviews, indicating a welcoming atmosphere for novices eager to listen and learn.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.12370">Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models</a>: Scaling the capacity of language models has consistently proven to be a reliable approach for improving performance and unlocking new capabilities. Capacity can be primarily defined by two dimensions:...</li><li><a href="https://tenor.com/view/hair-flip-duhh-gif-26170789">Hair Flip GIF - Hair Flip Duhh - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1334540563667488858)** (7 messages): 

> `PydanticAI, LlamaIndex, LangChain, Model Performance, Future Agent Frameworks` 


- **PydanticAI leads but lags in results**: While exploring **PydanticAI**, users found its API to be the nicest, featuring an internal temperature setting, but the results often yield broken JSON responses.
   - *Inspecting server requests was noted as challenging*, with the best structured outputs obtained from **LlamaIndex** despite PydanticAI's appealing interface.
- **LangChain cautionary tale**: A member warned against **LangChain**, describing its *silly pipe syntax* which complicates troubleshooting, especially when issues arise.
   - In contrast, LlamaIndex was recommended for better performance with minimal hassle and less data loss.
- **Struggles with low-end models**: Another member emphasized the **challenges faced** when using lower-end models like **Llama3.2**, finding it difficult to extract context and output structured data.
   - They noted observing server-side outputs helped them refine prompts and improve model interactions.
- **Logfire UI performance issues**: There were complaints regarding **high CPU/GPU usage** of the Logfire UI when idle in the browser, significantly dropping when the tab was closed.
   - This observation highlighted the potential inefficiency of the UI and the impact on system resources.
- **Wishlist for agent frameworks improvements**: A wishlist was proposed for future frameworks, including the ability to **inspect network traffic** and access metadata about models and usage metrics.
   - Key suggestions included a model pool mechanism for smooth transitions between models and the ability to measure response quality against various prompts.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1334291066131710044)** (64 messages🔥🔥): 

> `DeepSeek IP Controversy, EU AI Strategy Reactions, Mistral Small 3 Launch, Tülu 3 405B Release, Multi-Language Training Challenges` 


- **OpenAI accuses DeepSeek of IP theft**: In a recent controversy, OpenAI and AI Czar David Sacks accused [DeepSeek](https://www.youtube.com/watch?v=hpwoGjpYygI) of stealing their technology to train its new R1 model, causing significant backlash.
   - The situation raises questions about the ownership and ethical use of AI technologies in a rapidly evolving market.
- **Debate over EU businesses using AI**: The EU Commission revealed that only **13.5%** of EU businesses currently utilize AI, prompting calls for a new AI strategy to enhance adoption across sectors.
   - Members expressed skepticism, arguing that improving AI development should take priority rather than merely increasing utilization.
- **Mistral Small 3 launches with impressive specs**: [Mistral Small 3](https://mistral.ai/news/mistral-small-3/) is a latency-optimized 24B-parameter model that delivers competitive performance and efficiency, reportedly achieving over **81% accuracy** on the MMLU benchmark.
   - The model is designed for local deployment and outperforms larger competitors like Llama 3.3 70B while being **3x faster** on the same hardware.
- **Tülu 3 405B outshines competitors**: The release of [Tülu 3 405B](https://allenai.org/blog/tulu-3-405B) showcases advancements in open-weight models, achieving competitive performance against DeepSeek v3 and GPT-4o.
   - The implementation of their **Reinforcement Learning from Verifiable Rewards (RLVR)** framework has led to significant improvements in model performance.
- **Challenges in multi-language AI training**: Discussions surfaced around the low adoption of AI in Europe, highlighting that conversations about training models in multiple languages often lead to debates on GDPR challenges.
   - There are conflicting views on whether focusing on a few major languages is sufficient for AI development, with some advocating for broader language support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/fedora-tipshat-mlady-melady-athiest-gif-7191305">Fedora Tipshat GIF - Fedora Tipshat Mlady - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak">Wiz Research Uncovers Exposed DeepSeek Database Leaking Sensitive Information, Including Chat History | Wiz Blog</a>: A publicly accessible database belonging to DeepSeek allowed full control over database operations, including the ability to access internal data. The exposure includes over a million lines of log str...</li><li><a href="https://www.wheresyoured.at/deep-impact/">Deep Impact</a>: Soundtrack: The Hives — Hate To Say I Told You SoIn the last week or so, but especially over the weekend, the entire generative AI industry has been thrown into chaos.This won’t be a lengthy, technica...</li><li><a href="https://x.com/EU_Commission/status/1884635063054106770">Tweet from European Commission (@EU_Commission)</a>: &#34;Only 13.5% of EU businesses are using AI.This must change.This year we will launch a broad AI Strategy for our continent, including an ‘Apply AI&#39; initiative to drive industrial adoption of Ar...</li><li><a href="https://allenai.org/blog/tulu-3-405B">Scaling the Tülu 3 post-training recipes to surpass the performance of DeepSeek V3  | Ai2</a>: Introducing Tülu 3 405B, the first application of fully open post-training recipes to the largest open-weight models.</li><li><a href="https://slator.com/meta-rolls-out-multimodal-llama-3-2-but-not-in-europe/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=hpwoGjpYygI">DeepSeek stole our tech... says OpenAI</a>: Build better apps with PostHog https://posthog.com/fireshipOpenAI and AI Czar David Sacks accuse DeepSeek of stealing their IP to train their new R1 model, c...</li><li><a href="https://www.reddit.com/r/ABoringDystopia/comments/1ht7fft/used_meta_ai_to_edit_a_selfie_now_instagram_is/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1334275589800202290)** (65 messages🔥🔥): 

> `Tülu 3 405B Launch, Mistral Small 3 Announcement, DeepSeek Database Exposure, OpenAI Presentation in Washington, SoftBank Investment Talks with OpenAI` 


- **Tülu 3 405B Launch Surprises**: The [launch of Tülu 3 405B](https://allenai.org/blog/tulu-3-405B) showcases the scalability and effectiveness of open post-training recipes, achieving superior performance compared to both **Deepseek v3** and **GPT-4o**.
   - *Wow nature is healing* was the sentiment shared, reflecting excitement over the innovation coming from the Tülu team.
- **Mistral Small 3 Promises Efficiency**: Mistral announced the release of **Mistral Small 3**, a 24B-parameter model designed for local deployment, achieving state-of-the-art performance at low latency.
   - Prominent features include being highly **knowledge-dense** and effective for a wide range of generative AI tasks, making it ideal for deployments even on consumer-grade hardware.
- **Sensitive Data Leakage from DeepSeek**: [Wiz Research](https://x.com/wiz_io/status/1884707816935391703?s=61) uncovered that a publicly accessible database from DeepSeek exposed sensitive user data, including secret keys and chat logs.
   - Concerns over the implications for privacy and data security have prompted discussions about the **control measures** needed for AI platforms.
- **OpenAI Presents New Tech in DC**: Sam Altman and Kevin Weil presented new technology to the U.S. administration in Washington, with expectations of significant reactions from the demonstration.
   - Prior presentations from OpenAI have historically stirred considerable interest, indicating that this event could follow suit.
- **SoftBank's Interested Investment**: Reports surfaced that **SoftBank** is negotiating to invest an additional **$15-25 billion** directly into OpenAI alongside their prior commitments.
   - This move signifies growing interest and confidence in AI ventures at a significant scale amidst a competitive landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1884967826215059681">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://www.theguardian.com/technology/2025/jan/29/deepseek-blocked-some-app-stores-italy-questions-data-use">DeepSeek blocked from some app stores in Italy amid questions on data use</a>: Italian and Irish regulators want answers on how data harvested by chatbot could be used by Chinese government</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-30/deepseek-s-ai-restricted-by-hundreds-of-companies">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-30/deepseek-s-ai-restricted-by-hundreds-of-companies-within-days">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/hamishivi/status/1884990994883768721">Tweet from Hamish Ivison (@hamishivi)</a>: This was a fun side effort with lots of help from everyone on the Tulu 3 team. Special shoutouts to @vwxyzjn  (who did a lot on the training+infra side) and @ljvmiranda  (who helped with DPO data gene...</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>: no description found</li><li><a href="https://x.com/wiz_io/status/1884707819737223591?s=61">Tweet from Wiz (@wiz_io)</a>: This meant anyone could access logs containing actual chat messages, internal secrets, service data, and potentially exfiltrate data along with escalating privileges within the server.</li><li><a href="https://x.com/techikansh/status/1884961297709572170">Tweet from Techikansh (@techikansh)</a>: Theo, did you have o3-mini under embargo?</li><li><a href="https://x.com/AndrewCurran_/status/1884950312525725736">Tweet from Andrew Curran (@AndrewCurran_)</a>: Sam Altman and Kevin Weil are in Washington this morning giving a presentation to the new administration. According to Axios they are also demoing new technology that will be released in Q1. The last ...</li><li><a href="https://x.com/hamishivi/status/1884990987757596832">Tweet from Hamish Ivison (@hamishivi)</a>: li&#39;l holiday project from the tulu team :)Scaling up the Tulu recipe to 405B works pretty well! We mainly see this as confirmation that open-instruct scales to large-scale training -- more excitin...</li><li><a href="https://x.com/firstadopter/status/1884794211091759444">Tweet from tae kim (@firstadopter)</a>: FT: “SoftBank is in talks to invest $15-25bn directly into OpenAI on top of its commitment of more than $15bn to Stargate, according to multiple people with direct knowledge of the negotiations.”</li><li><a href="https://x.com/wiz_io/status/1884707816935391703?s=61">Tweet from Wiz (@wiz_io)</a>: BREAKING: Internal #DeepSeek database publicly exposed 🚨Wiz Research has discovered &#34;DeepLeak&#34; - a publicly accessible ClickHouse database belonging to DeepSeek, exposing highly sensitive inf...</li><li><a href="https://x.com/btibor91/status/1884756371058634762">Tweet from Tibor Blaho (@btibor91)</a>: Updated GPT-4o model in ChatGPT confirmed - &#34;Updates to GPT-4o in ChatGPT (January 29, 2025)&#34; - more up-to-date knowledge (June 2024), deeper understanding and analysis of image uploads, smart...</li><li><a href="https://allenai.org/blog/tulu-3-405B">Scaling the Tülu 3 post-training recipes to surpass the performance of DeepSeek V3  | Ai2</a>: Introducing Tülu 3 405B, the first application of fully open post-training recipes to the largest open-weight models.</li><li><a href="https://x.com/Mobius_Labs/status/1885010791344062704">Tweet from Mobius Labs (@Mobius_Labs)</a>: We’ve improved DeepSeek R1 distilled models using logits distillation—delivering +4-14% gains on GSM8K  while only spending $3-18 per training run! 🚀 Now available on Hugging Face-e—run them efficien...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655?mx=2">Tweet from Jiayi Pan (@jiayi_pirate)</a>: We reproduced DeepSeek R1-Zero in the CountDown game, and it just works Through RL, the 3B base LM develops self-verification and search abilities all on its own You can experience the Ahah moment you...</li><li><a href="https://x.com/AndrewCurran_/status/1884971368531587573">Tweet from Andrew Curran (@AndrewCurran_)</a>: Here we go.Quoting Andrew Curran (@AndrewCurran_) Sam Altman and Kevin Weil are in Washington this morning giving a presentation to the new administration. According to Axios they are also demoing new...</li><li><a href="https://x.com/altryne/status/1884778839009796411">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Zuck highlights from the earnings call: - LLama 4 & LLama 4 mini (done with pre-training)- Confirms reasoning LLaMas! - Llama 4 will be natively multimodal -- it&#39;s an omni-model -- and it will hav...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1334276178210590852)** (28 messages🔥): 

> `Meta's Legal Challenges, V3 Licensing Issues, Concerns about Model Deployment, Impact of Licenses on AI Development` 


- **Meta faces anxiety over licensing**: Members expressed concern about **Meta's vulnerabilities** regarding copyright claims with their Llama model, especially if they deploy **DeepSeek** instead.
   - *A member noted that DeepSeek might be motivated to challenge Meta legally if plausible causes under licensing are found.*
- **V3 License is Not MIT**: There's confusion about V3's licensing; it is noted to be a restrictive license, likened to an 'almost OSS' framework that could restrict freedoms.
   - *It was pointed out that to have a legally clean version of V3, duplication is necessary, which is cumbersome and raises concerns.*
- **Legal Clauses create Liability**: Discussion highlighted that **'do no evil' clauses** beyond MIT/Apache are problematic as they may create open-ended legal liabilities.
   - *One member humorously mentioned the JSON license causing issues due to complicated legal language, reflecting on the unpredictable nature of legal clauses.*
- **Censorship Risks with V3**: Concerns were raised that releasing a **V3 finetune** could lead to unintended violations, especially regarding sensitive discussions like **Tiananmen Square**.
   - *A member stressed that even if a violation is only a crime in one country, the licensing agreements could lead to litigation regardless of location.*
- **Licenses are Insane**: Several members shared their feelings about licenses being a stressful yet fascinating subject, underlining their complexity in the AI landscape.
   - *One noted that licenses could invoke a perpetual state of liability, making compliance nearly impossible.*


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1334271624660389961)** (26 messages🔥): 

> `DeepSeek R1 Launch, Speculations on Model Performance, Quantization in GPT-4, Updates on Tulu 3 Paper, Emerging Reasoning Models` 


- **DeepSeek R1 launches on Azure AI Foundry**: DeepSeek R1 is now available in the model catalog on [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry) and GitHub, expanding the portfolio to over **1,800 models** including various AI types.
   - *As stated in the blog*, this launch enables businesses to integrate advanced AI seamlessly while ensuring security and responsible AI commitments.
- **Concerns over model reliability**: There are ongoing speculations regarding whether OpenAI's models generate results reliably, particularly related to **lower precision quantization**.
   - Some members related fluctuating quality in **GPT-4** output to issues with quantization that may have harmed the model's performance.
- **Tulu 3 Paper Update Excitement**: The Tulu 3 paper was recently updated, stirring excitement within the community as members observed this type of responsiveness.
   - *One user noted*, 'the arXiv -> media pipeline is so wild to witness,' reflecting on the rapid dissemination of information.
- **Emerging Reasoning Models Technical Discussions**: Canadians are preparing to enter the **smol reasonoor game**, exploring the integration of tool use and RAG for reasoning models which is deemed notable.
   - However, the technical details of their implementation within reasoning processes remain unclear, causing some frustration among the developers.
- **Speculation on FP8 Support**: There are rumors circulating that **Ascend 910c** may not have native FP8 support, raising questions regarding the future of model training capabilities.
   - This speculation has been a topic of discussion, with community members sharing their thoughts on performance implications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/johnschulman2/status/1884740922744983715">Tweet from John Schulman (@johnschulman2)</a>: I think the term was coined (or popularized) by @bobmcgrewai in the early days of ChatGPT. The team doing ChatGPT fine-tuning was previously called the RL team for historical reasons, and Bob suggeste...</li><li><a href="https://x.com/1vnzh/status/1884899043047887243">Tweet from Ivan Zhang (@1vnzh)</a>: chat is this true help a brother celebrate CNY away from home</li><li><a href="https://x.com/amir/status/1885012737614635280">Tweet from Amir Efrati (@amir)</a>: y&#39;all can now literally make your own o1-like reasoning model basically for free.</li><li><a href="https://fxtwitter.com/erykbanatt/status/1884857074833584269">Tweet from Eryk (@erykbanatt)</a>: noticed the tulu 3 paper was just updated 👀</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 is now available on Azure AI Foundry and GitHub | Microsoft Azure Blog</a>: DeepSeek R1, available through the model catalog on Microsoft Azure AI Foundry and GitHub, enables businesses to seamlessly integrate advanced AI.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1334275598637338624)** (9 messages🔥): 

> `Teortaxes commentary, Deepseek R1 training leak, AME(R1)CA version, Mistral Small 3 architecture, Data visualization passion` 


- **Teortaxes' Self-Description with a Twist**: A member humorously remarked that **Teortaxes** seems to mostly describe himself, but noted that the **Žižek voice** makes everything better forever.
   - *“The Zizek voice makes everything good forever.”*
- **Leaked Deepseek R1 Training Footage**: [Aidenybai's tweet](https://x.com/aidenybai/status/1884826039723114901) revealed a **leaked video** showcasing the training of **Deepseek R1**.
   - The community expressed enthusiasm over the insights shared in the leak.
- **Introducing AME(R1)CA for American Values**: [Tylercosg's tweet](https://x.com/tylercosg/status/1884747401744855467) introduced **AME(R1)CA**, a version of **Deepseek R1** aimed at aligning with **American values** and distancing from CCP influences.
   - Their tagline promised a solution for those concerned about the influence of the **CCP**.
- **Optimizing Mistral Small 3 for Latency**: [Dchaplot's tweet](https://x.com/dchaplot/status/1884975429561487815) highlighted that the **Mistral Small 3 architecture** is specifically optimized for latency.
   - The chat discussed **Mistral's** unusual axis choices in this context.
- **Passion for Data Visualization**: One member expressed that **data visualization** is their passion, in response to the ongoing discussion about Mistral.
   - *“Data visualization is my passion.”*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dchaplot/status/1884975429561487815">Tweet from Devendra Chaplot (@dchaplot)</a>: Mistral Small 3 architecture is optimized for latency.2/N</li><li><a href="https://x.com/aidenybai/status/1884826039723114901">Tweet from Aiden Bai (@aidenybai)</a>: leaked video of deepseek r1 training</li><li><a href="https://x.com/tylercosg/status/1884747401744855467">Tweet from Tyler Cosgrove (@tylercosg)</a>: worried about the influence of the CCP but still want to use deepseek r1? worry no more!say hello to AME(R1)CAa version of r1 that steers toward american values and away from those pesky chinese commu...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1334316467998429194)** (11 messages🔥): 

> `Tulu3 data preparation, verl vs GRPOTrainer, open-instruct implementation, HF GRPO limitations, LoRA support in open-instruct` 


- **Best practices for Tulu3 data preparation**: A member inquired about special considerations for data preparation with **Tulu3** to optimize the LLM for RLHF after post-training.
   - Another member suggested focusing on **domain of interest**, **evals setup**, and ensuring **stable generation for pref data**.
- **Comparing verl and Huggingface's GRPOTrainer**: A member expressed interest in the practical use of **verl** and **Huggingface's GRPOTrainer** in earnest, questioning if either is superior.
   - They are currently using verl but find it has some **rough edges**, prompting them to evaluate whether to invest further or seek better alternatives.
- **Clarifying GRPO limitations**: A discussion highlighted that **HF GRPO** only supports **1 grad step per update**, lacking the clipping logic inherent in **PPO**.
   - Members debated the implications of this limitation, with one member referencing the **TRL code** for clarification.
- **LoRA support in open-instruct**: A member queried about the prioritization of **LoRA support** for open-instruct implementations, speculating against it based on current training methods.
   - There was acknowledgment of the curiosity about whether LoRA support would be considered, reflecting on its relevance to upcoming training runs.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1334263494211665972)** (65 messages🔥🔥): 

> `DeepSeek Math Paper, Mixture-of-Experts (MoE), Multi Token Prediction (MTP), DeepSeek v3 Architecture, Inferences and Experts Balancing` 


- **DeepSeek Paper Marks RL Breakthrough**: The **DeepSeek math paper** is *widely* regarded as a significant **reinforcement learning (RL) breakthrough** and introduces **GRPO** in v2, although the v2 paper is mostly recognized for **MLA**.
   - Discussion points to the importance of this work within the overall context of current RL advancements.
- **MTP Gains Attention for Speculative Decoding**: **MTP** is highlighted as a key aspect of DeepSeek v3, where it predicts **2 tokens** with an **85-90% acceptance rate**, which many frameworks have overlooked.
   - Members expressed curiosity about its role at both training and inference times, particularly in how it relates to regularization.
- **MoE Innovations in DeepSeek V3**: DeepSeek v3 adopts **sigmoid gating** instead of softmax, allowing experts to operate in parallel without competing directly, while also introducing **dropless load balancing**.
   - This architecture involves an additional general layer alongside experts, shifting the perspective on how multipurpose experts function within the model.
- **Exploring Load Balancing in Experts**: Members discussed the challenges of balancing **auxiliary losses** used for expert balancing in the v3 framework, questioning the practical implementation details.
   - The conversation highlighted confusion over how these components affect model performance and whether they truly enhance inference speeds.
- **Conversations about AI's Evolution**: A member shared insights on the **rapid evolution** of AI and its accompanying pressures on researchers, reflected in their jittery behaviors.
   - Context was given around the societal dynamics influencing AI development, emphasizing the urgency felt within this field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.06066">DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models</a>: In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters. However, conventional MoE architectures lik...</li><li><a href="https://www.hyperdimensional.co/p/novus-ordo-seclorum">Novus Ordo Seclorum</a>: Reflections on DeepSeek</li><li><a href="https://ghost.oxen.ai/no-hype-deepseek-r1-reading-list/">No Hype DeepSeek-R1 Reading List</a>: DeepSeek-R1 is a big step forward in the open model ecosystem for AI with their latest model competing with OpenAI&#x27;s o1 on a variety of metrics. There is a lot of hype, and a lot of noise around ...</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py#L898">modeling_deepseek.py · deepseek-ai/deepseek-moe-16b-base at main</a>: no description found</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">Mixture-of-Experts (MoE) LLMs</a>: Understanding models like DeepSeek, Grok, and Mixtral from the ground up...</li><li><a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v3.py">vllm/vllm/model_executor/models/deepseek_v3.py at main · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1334279230447554666)** (2 messages): 

> `Science Phrasing, Opinion on OAI, Metaphors in AI Discourse` 


- **Science by Smell Critique**: A member expressed that some aspects of AI discussion feel *too much like science by smell*, implying a lack of rigor in certain evaluations.
   - This perspective suggests a desire for clearer, more concrete metrics rather than vague assessments.
- **Bitter Pills and Umami in OAI**: Another member remarked that OAI presents *maximum bitter pills*, contrasting it with *verifiers as umami*, which suggests a flavorful component in an otherwise tough landscape.
   - This metaphor highlights the duality of difficult truths and rewarding insights within the OpenAI environment.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1334256165944692848)** (13 messages🔥): 

> `Training Techniques in AI Models, Concerns about Data Sources, Deepseek Speculations, OpenAI Output Usage, Tulu Dataset in Training` 


- **Distillation's Role in SOTA Performance**: Discussion arose regarding the possibility that **distillation** techniques contribute to the **SOTA** results seen in V3, leading to questions about the efficacy of methods like **MLA/8 bit training** without using distillation data.
   - One member speculated that if performance is linked to distillation, then the training strategies employed for base models might need reevaluation.
- **Perplexity Numbers Indicate Strong Training**: It was noted that perplexity numbers from a large-scale dataset appear strong, suggesting that the **training** performed is effective.
   - Members expressed skepticism that such good results could be achieved without a solid training foundation.
- **Speculations on Deepseek's Methodologies**: There were mixed feelings about whether **Deepseek** used **ChatGPT** for data filtering, with members noting that stylistic similarities could suggest a significant distillation process, which seems unsubstantiated.
   - Despite the theories, it was suggested that using OpenAI outputs could be a moderate likelihood.
- **Cautious Sentiments Regarding Deepseek**: Participants displayed a **distrustful** attitude towards Deepseek, expressing concerns that assumptions about their capabilities may stem from **financial** anxieties related to competition.
   - Some theories posited that unrelated functionalities could be influencing the use of OpenAI endpoints.
- **Interest in Tulu Dataset's Influence**: A member expressed enthusiasm at the prospect of **Tulu data** being utilized in the **SFT** phase of training, indicating its value in the community.
   - Others acknowledged **ShareGPT4V** as a noteworthy dataset in the open-source **VLM** landscape, calling it a classic reference.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1334354896220323904)** (31 messages🔥): 

> `OpenAI's training ethics, RL methods and tool usage, Pythia language model sampling, Concerns over model performance, Tool dependency in LLMs` 


- **OpenAI's Ethical Quandary with Deepseek**: It's considered wild that OpenAI highlights issues related to **Deepseek training**, especially given their history of using data from those they aim to place out of jobs.
   - Members expressed skepticism about OpenAI's legal claims and motivations to be seen as competent in a competitive field.
- **Exploring RL Method Benefits for LLM Tool Use**: There’s a realization that using **reinforcement learning (RL)** could minimize the need for extensive datasets by explaining tools briefly, allowing models to learn independently.
   - Concerns were raised about maintaining balance so LLMs do not become excessively reliant on specific tools.
- **Pythia Model's Sampling Probability**: Discussion revolves around the probability of sampling a trained **Pythia language model** from a Gaussian distribution, with acknowledgment of local volume estimation.
   - The concept of focusing on a 'neighborhood' around networks exhibiting specific behaviors was emphasized to refine the analysis.
- **Performance Discrepancies in Distilled Models**: Members noted that despite extensive training and resources, other open-source models haven't matched the downstream performance of **GPT-4o** distillations.
   - There was speculation about the involvement of post-training methodologies like those of **Alpaca** in enhancing model capabilities.
- **Dangers of Tool Dependency in AI**: Community members discussed the potential risks of LLMs becoming overly reliant on certain tools for problem-solving, suggesting random tool availability could be a good strategy.
   - Thoughts were shared on how intelligent models could learn when to apply tools effectively while still retaining fundamental problem-solving skills.



**Link mentioned**: <a href="https://www.overleaf.com/read/krhxtvkxjywb#416acf">Overleaf, Online LaTeX Editor</a>: An online LaTeX editor that’s easy to use. No installation, real-time collaboration, version control, hundreds of LaTeX templates, and more.

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1334253288597753887)** (178 messages🔥🔥): 

> `Hyperfitting in LLMs, Critique Fine-Tuning, Backdoor Detection, Sampling Neural Networks, Generalization vs Memorization` 


- **Hyperfitting enhances LLM text generation**: Recent discussions highlighted that **hyperfitting** on a small dataset can significantly improve **open-ended text generation** capabilities of LLMs, counterintuitive to conventional wisdom.
   - For example, a model's human preference score climbed from **4.9%** to **34.3%**, putting it on par with larger models despite potential overfitting.
- **Introducing Critique Fine-Tuning**: Critique Fine-Tuning (CFT) encourages models to learn from and critique noisy responses instead of only imitating correct ones, yielding consistent improvements in performance.
   - The method, validated on six math benchmarks, showed a **4-10%** improvement over traditional supervised fine-tuning.
- **Concerns on backdoor detection implications**: The ARC backdoor paper suggests that undetectably backdoored models can closely resemble their regular counterparts, leading to potential loss mismatches as models grow larger.
   - This raises questions about the effectiveness of loss functions in differentiating between backdoored and standard models.
- **Sampling techniques in neural networks**: Discussion around a proposed general-purpose **Absolute Unit NN** architecture examined how overall performance could be compromised due to scaling challenges.
   - Critics raised concerns about the practicality of this approach, particularly in terms of generalization versus memorization.
- **Evaluating CE-loss as a training metric**: There was consensus that using **cross-entropy loss** (CE-loss) as a training metric for LLM ability might not be adequate for measuring real-world performance.
   - Participants questioned why this metric remains in use, highlighting a lack of meaningful alternatives to assess model capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BlinkDL_AI/status/1884768989743882276">Tweet from BlinkDL (@BlinkDL_AI)</a>: I propose ZeroCoT: a simple method to bootstrap CoT from zero. Let me know what you think 🙂 I will try this using RWKV soon.Quoting BlinkDL (@BlinkDL_AI) Let&#39;s kill attention. RWKV-7 &#34;Goose&#...</li><li><a href="https://openreview.net/forum?id=Ij9ilPh36h">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for...</a>: This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it...</li><li><a href="https://arxiv.org/abs/2501.17703">Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate</a>: Supervised Fine-Tuning (SFT) is commonly used to train language models to imitate annotated responses for given instructions. In this paper, we challenge this paradigm and propose Critique Fine-Tuning...</li><li><a href="https://www.overleaf.com/read/krhxtvkxjywb#416acf">Overleaf, Online LaTeX Editor</a>: An online LaTeX editor that’s easy to use. No installation, real-time collaboration, version control, hundreds of LaTeX templates, and more.</li><li><a href="https://arxiv.org/abs/2409.03077">Backdoor defense, learnability and obfuscation</a>: We introduce a formal notion of defendability against backdoors using a game between an attacker and a defender. In this game, the attacker modifies a function to behave differently on a particular in...</li><li><a href="https://arxiv.org/abs/2405.14722">DAPE: Data-Adaptive Positional Encoding for Length Extrapolation</a>: Positional encoding plays a crucial role in transformers, significantly impacting model performance and length generalization. Prior research has introduced absolute positional encoding (APE) and rela...</li><li><a href="https://arxiv.org/abs/2406.11235">QTIP: Quantization with Trellises and Incoherence Processing</a>: Post-training quantization (PTQ) reduces the memory footprint of LLMs by quantizing weights to low-precision datatypes. Since LLM inference is usually memory-bound, PTQ methods can improve inference t...</li><li><a href="https://arxiv.org/abs/2306.16830">Sampling weights of deep neural networks</a>: We introduce a probability distribution, combined with an efficient sampling algorithm, for weights and biases of fully-connected neural networks. In a supervised learning context, no iterative optimi...</li><li><a href="https://youtu.be/1GCf29FPM4k?si=osypqodwU_B1QmXD">The LONGEST time - Numberphile</a>: A paper by Don Page claimed to use the longest finite time ever calculated by a physicist - it&#39;s the time it will take the Universe to reset itself!?!More li...</li><li><a href="https://gwern.net/aunn">Absolute Unit NNs: Regression-Based MLPs for Everything · Gwern.net</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1334384668706996245)** (2 messages): 

> `DeepSpeed training issues, Intermediate dimension adjustments for gated MLPs, Llama2 config parameters` 


- **Deleting torch_extensions to fix training issues**: A user suggested deleting the `torch_extensions` directory from the cache folder to resolve a training issue where loading the model prevents the training from starting, referencing [this issue](https://github.com/microsoft/DeepSpeed/issues/2816).
   - This simple fix reportedly worked, indicating a potential solution for similar problems.
- **Setting Intermediate Dimensions in Gated MLPs**: One theory for configuring models with gated MLPs is to set the intermediate dimension to **3x** the desired value and then reset it during export to avoid issues with Hugging Face exports.
   - This workaround worked for two models tested, although the user acknowledges that further checks may be needed.
- **Llama2 Configuration Value Clarification**: The user noted that the **32768** value in the Llama2 config is unexplained and not divisible by **3**, which causes it to adjust to **11008** when considerations about the gated configurations are applied.
   - This insight is based on a reference to the [Llama2 config](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26) and the user is open to corrections on this understanding.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/microsoft/DeepSpeed/issues/2816">Training stuck after loading the model?  · Issue #2816 · microsoft/DeepSpeed</a>: Issue: Training doesn&#39;t begin after loading the model. DS_REPORT (base) ext_abdul.waheed@p4-r69-a:~$ nvcc --version nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2020 NVIDIA Corporation...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26)">gpt-neox/configs/llama2/7B.yml at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1334323384485417051)** (1 messages): 

> `DeepSeek R1 Distill Qwen 32B, DeepSeek R1 Distill Qwen 14B` 


- **Introducing DeepSeek R1 Distill Qwen 32B**: The new model [DeepSeek R1 Distill Qwen 32B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b) delivers lightweight performance similar to the larger **R1 Llama 70b Distill**, priced at **$0.7/M** for input and output.
   - Interested users can request access to the model via the [Discord channel](https://discord.gg/fVyRaUDgxW).
- **Launch of DeepSeek R1 Distill Qwen 14B**: The [DeepSeek R1 Distill Qwen 14B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b) is now available, promising smaller size and faster processing while scoring **69.7 on AIME 2024**.
   - This model is priced at **$0.75/M** for both input and output, and can also be accessed through the [Discord](https://discord.gg/fVyRaUDgxW).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b>)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1334573785751490611)** (6 messages): 

> `Subconscious AI's capabilities, Beamlit's platform features, Discord engagement` 


- **Subconscious AI Transforming Decision-Making**: Subconscious AI is revolutionizing **decision-making** through **advanced AI-driven research**, market simulation, and causal inference modeling, as noted on their [website](https://www.subconscious.ai).
   - They highlighted that their platform helps businesses and policymakers gain deep insights into consumer behavior and market trends, emphasizing the **guaranteed human-level reliability** of their causal models.
- **Beamlit Aims to Accelerate Generative AI Development**: Mathis, co-founder of [Beamlit](https://beamlit.com), shared that their platform allows developers to **ship AI agents** up to **10× faster** using a simple command interface akin to Vercel for AI Agents.
   - They launched a **free public alpha version**, inviting users to provide feedback and explore features like integrated Github workflows and observability tools.
- **Community Engagement on Discord**: A member expressed interest in Subconscious AI and joined their **Discord** for more information.
   - This highlights an ongoing trend of community-oriented conversations aimed at fostering deeper connections between emerging AI technologies and potential users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.subconscious.ai">subconscious.ai</a>: no description found</li><li><a href="https://beamlit.com:">Beamlit</a>: no description found</li><li><a href="https://docs.beamlit.com/Get-started#quickstart">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1334257353628979240)** (180 messages🔥🔥): 

> `OpenRouter Pricing Concerns, DeepSeek R1 Model Limitations, Google AI Studio Rate Limits, Provider Issues and Downtimes, New Model Announcements` 


- **OpenRouter pricing sparks debate**: Users are questioning why OpenRouter charges **5%** for forwarding API requests, with one suggesting it feels too high given the service provided.
   - *You'll have to take that one up with Stripe*, another user quipped, hinting at potential underlying fees.
- **DeepSeek R1 generating issues with context window**: Several users reported issues with the **DeepSeek R1** model, including trouble retrieving responses when generation timed out due to exceeding context limits.
   - One user confirmed that to view reasoning with the model, the `include_reasoning` parameter needs to be passed in the API request.
- **Frequent rate limit errors with Google AI Studio**: Users have experienced **429 RESOURCE_EXHAUSTED** errors while querying **Gemini** models in Google AI Studio, indicating exhausted quotas.
   - The rate limits are imposed by Google, and users are encouraged to plug in their own keys for improved throughput.
- **Provider statuses fluctuate with downtimes**: Some users reported ongoing **404 errors** with OpenRouter's API, particularly when trying to access the chat completions endpoint.
   - The outages are attributed to varying provider capacities, with **Nebius** and **Avian** being highlighted for their inconsistent service.
- **Upcoming AI model releases spark excitement**: Users discussed announcements regarding new AI models like **Mistral's Small 3** and **Tülu 3**, showcasing increased performance in various capacities.
   - The community eagerly anticipates the integration of new models into OpenRouter as they promise significant capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/allen_ai/status/1884966600039915809">Tweet from Ai2 (@allen_ai)</a>: Here is Tülu 3 405B 🐫 our open-source post-training model that surpasses the performance of DeepSeek-V3! The last member of the Tülu 3 family demonstrates that our recipe, which includes Reinforcemen...</li><li><a href="https://x.com/MistralAI/status/1884968836606136636">Tweet from Mistral AI (@MistralAI)</a>: Introducing Small 3, our most efficient and versatile model yet! Pre-trained and instructed version, Apache 2.0, 24B, 81% MMLU, 150 tok/s. No synthetic data so great base for anything reasoning - happ...</li><li><a href="https://openrouter.ai/docs/">Quick Start | OpenRouter</a>: Start building with OpenRouter</li><li><a href="https://openrouter.ai/docs/quick-start">Quick Start | OpenRouter</a>: Start building with OpenRouter</li><li><a href="https://x.com/risphereeditor/status/1885041914191192573">Tweet from Risphere (@risphereeditor)</a>: Fireworks AI is now the fastest DeepSeek provider in the US.DeepSeek-V3 and and DeepSeek-R1 now run at 30 tokens per second.Congrats to the @FireworksAI_HQ team!</li><li><a href="https://openrouter.ai/docs/integrations#automatic-fallback">Integrations | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://www.reddit.com/r/bing/comments/110eagl/the_customer_s">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/docs/parameters#include-reasoning">Parameters | OpenRouter</a>: Configure parameters for requests</li><li><a href="https://openrouter.ai/google/gemini-flash-1.5-8b">Gemini Flash 1.5 8B - API, Providers, Stats</a>: Gemini Flash 1.5 8B is optimized for speed and efficiency, offering enhanced performance in small prompt tasks like chat, transcription, and translation. Run Gemini Flash 1.5 8B with API</li><li><a href="https://www.reddit.com/r/bing/comments/110eagl/the_customer_service_of_the_new_bing_chat_is/#lightbox)">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1334579954968563772)** (1 messages): 

> `Bolt binary asset generation, Token savings, External assets utilization` 


- **Bolt stops generating binary assets**: Bolt now avoids generating binary assets, resulting in significant **token and time savings**, as well as enhanced output quality.
   - This change allows more efficient processing and improves overall performance.
- **Significant token savings achieved**: The latest change in Bolt has saved **hundreds of thousands of tokens** by optimizing how assets are utilized.
   - This enhancement makes operations **orders of magnitude faster**, streamlining the entire process and improving user experience.
- **Leveraging external assets**: Bolt's agent now uses external assets instead of creating them from scratch, leading to more efficient token usage.
   - Members expressed excitement about this strategic shift, which improves the operational speed and quality of outcomes.



**Link mentioned**: <a href="https://x.com/boltdotnew/status/1885019780840653183">Tweet from bolt.new (@boltdotnew)</a>: More tokens savings landed!Bolt&#39;s agent leverages external assets now instead of allowing the LLM to create new ones from scratch.This saves hundreds of thousands of tokens—and is orders of magnit...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1334324876600676353)** (10 messages🔥): 

> `Trailing Zeroes Issue with Bolt, File Update Laziness, Employer Signup Form Update, Community Use Cases for System Prompt, Supabase Signup Error Troubleshooting` 


- **Bolt struggles with trailing zeroes**: Users expressed frustration when trying to enter numbers like **2.7980** for CBTO, as Bolt auto-formats them incorrectly. Despite requests for Bolt to display data exactly as entered, it fails to do so.
   - One member sought tips for managing this auto-formatting annoyance while sharing an image for context.
- **Need to Fix File Update Laziness**: Concerns were raised about persistent issues where users need the rest of the file to remain unchanged during updates. One member experienced a recurring error with syntax, indicating a need to address laziness in execution.
   - A user commented on improvements post-update, noting that while some issues with laziness have lessened, there’s still a way to go.
- **Updating the Employer Signup Form**: A member identified the need to include **First Name** and **Last Name** fields in the Employer signup form, which are currently missing. They emphasized the importance of proper data mapping to ensure smooth integration into the user profile.
   - Suggestions for addressing this gap included confirming matching file names and views especially when multiple updates are made.
- **Exploring Community Use Cases for System Prompt**: Interest was expressed in learning how the community leverages the new **Project and Global System Prompt**. Currently, one member uses it for Bolt to update the changelog, but is eager to hear other inspiring applications.
   - Another member advised sharing specific files and ensuring correct views to generate productive results while troubleshooting.
- **Troubleshooting Supabase Signup Errors**: Ongoing issues surrounding a **Supabase request failure** were highlighted, with a user encountering a **500** status error while signing up. They suggested creating a dedicated troubleshooting group to facilitate discussions on application-specific errors.
   - One member recommended utilizing AI tools to get advice by sharing error details, code, and relevant screenshots to help resolve issues more effectively.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1334263240020070420)** (170 messages🔥🔥): 

> `Supabase integration issues, Token usage concerns, Forked project challenges, CORS issue with Supabase functions, SEO meta data handling in React` 


- **Supabase integration issues after project fork**: Users are experiencing problems connecting their forked projects to Supabase, with the .env file not copying over, resulting in errors and unavailability of user dashboards.
   - Participants noted that until the issue is resolved, it's recommended to use local storage for data handling during development to avoid burning tokens.
- **Token usage and subscription confusion**: There is confusion regarding if tokens reset daily or monthly and how unused tokens are managed, with users clarifying that monthly subscriptions do not roll over unused tokens.
   - Several users expressed concerns over high token burn rates, especially when issues arise that require multiple prompts.
- **Challenges with forked projects**: Users are facing difficulties in re-establishing the Supabase connection after forking projects, with suggestions to copy the .env file manually for proper integration.
   - Creating GitHub issues to track known problems was recommended, as well as the importance of handling project backups appropriately.
- **CORS issue when calling Supabase functions**: A user reported encountering CORS errors while trying to call a Supabase function from the front-end application, hindering their progress.
   - Participants advised that API calls should be made from either a Node backend with Relay Request or through an Edge function to avoid such issues.
- **SEO meta data handling in React apps**: A user is seeking advice on how to implement server-side SEO meta data for different pages in a React application, noting that the usual methods are not effective.
   - There was discussion about using alternatives, as the default helmet approach does not seem to be fetching the right metadata for social media sharing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zp1v56uxy8rdx5ypatb0ockcb9tr6a-oci3--5173--5ab5ceac.local-credentialless.webcontainer-api.io'">no title found</a>: no description found</li><li><a href="https://tenor.com/view/frozen-freezing-cold-shivering-glace-gif-5390133">Frozen Freezing GIF - Frozen Freezing Cold - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://boltsync.mystify.tech/">BoltSync - GitHub Repository Management with Bolt</a>: Modify your GitHub repositories with Bolt Prompts &amp; sync changes back to GitHub with BoltSync. Streamline your development workflow with AI-powered repository management.</li><li><a href="https://x.com/boltdotnew/status/1843668731681267801">Tweet from bolt.new (@boltdotnew)</a>: You can now open public repos in bolt․new 🙌How? For any GitHub URL, just put &#34;http://bolt.new&#34; in front of it!(Release notes below!)</li><li><a href="https://github.com/stackblitz/bolt.new/issues">stackblitz/bolt.new</a>: Prompt, run, edit, and deploy full-stack web applications - stackblitz/bolt.new</li><li><a href="https://showmeyourbolt.io/">Show Me Your Bolt</a>: no description found</li><li><a href="https://boltnew.dev/apps">Bolt.new Builders Hub</a>: no description found</li><li><a href="https://imbuiltwithai.com/">Share Your AI Projects - I'm Built With AI</a>: no description found</li><li><a href="https://youtu.be/tlu5e0TxSzo?si=cCaDQFroJ8_1MNwT&t=77">How to upload files and folders to GitHub: GitHub for beginners</a>: Uploading your project to GitHub doesn&#39;t have to be complicated. In this video, we&#39;ll show you two easy methods to get your files and folders into a GitHub r...</li><li><a href="http://bolt.new/github.com/strapi/strapi">bolt.new</a>: no description found</li><li><a href="https://lucide.dev/icons/">Lucide Icons</a>: Beautiful &amp; consistent icon toolkit made by the community.
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1334253510270779493)** (178 messages🔥🔥): 

> `ComfyUI Performance and Features, Hardware Discussions for AI Workloads, Reactor Tool for Face Swapping, Stable Diffusion Lora Training, Availability of New GPUs` 


- **ComfyUI's Manual Control for Inpainting**: Users discussed the manual processes involved in inpainting and controlnet integrations in ComfyUI, highlighting the flexibility required for specific adjustments.
   - A user expressed their preference for manual controls to leverage the model's capabilities better rather than relying solely on automated methods.
- **Hardware Specifications for Stable Diffusion**: Conversations revolved around GPU specifications for running Stable Diffusion effectively, with users sharing experiences about the capabilities of various models like the 3080 and 3090.
   - One user discussed their experience using the Intel Arc A770 LE and its comparable performance to the 3060/3060TI in gaming and AI tasks.
- **Reactor Tool Removal and Alternatives**: A user inquired about the removal of the Reactor tool, noting that it was taken down due to a lack of an NSFW filter, though it was later re-uploaded with safeguards.
   - Links were shared to the updated version of Reactor, which was made available for auto1111 and ComfyUI users, enabling face swap functionalities.
- **Training Loras for Stable Diffusion**: Users discussed the process for training Loras for Stable Diffusion and the importance of integrating styles while ensuring specific features match.
   - One user sought clarification on workflows that involve combining specific faces and style references, highlighting their recent challenges.
- **Availability of New GPUs**: The rapid sell-out of the new 5090 GPUs sparked discussions around market demand and availability, with some users expressing disappointment at the limited supply.
   - Conversations included opinions on financing options for tech purchases and general market frustrations over the difficulty accessing new hardware.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://streamable.com/d3ww4l">Watch inpaint3432342 (online-video-cutter.com) | Streamable</a>: no description found</li><li><a href="https://tenor.com/view/wall-gif-24534315">Wall GIF - Wall - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://streamable.com/5edusn">Watch inpaint 23432432 (online-video-cutter.com) | Streamable</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=pvxUHpf1pxQ">Sweet Jazz &amp; Ocean Sounds at Beachside Cafe Space ~ Positive Bossa Nova Music to Elevate your Mood</a>: Sweet Jazz &amp; Ocean Sounds at Beachside Cafe Space ~ Positive Bossa Nova Music to Elevate your MoodEscape to a dreamy coastal retreat with our lastest collect...</li><li><a href="https://github.com/Gourieff/sd-webui-reactor-sfw">GitHub - Gourieff/sd-webui-reactor-sfw: (SFW Friendly) Fast and Simple Face Swap Extension for StableDiffusion WebUI (A1111, SD.Next, Cagliostro)</a>: (SFW Friendly) Fast and Simple Face Swap Extension for StableDiffusion WebUI (A1111, SD.Next, Cagliostro) - Gourieff/sd-webui-reactor-sfw</li><li><a href="https://github.com/Gourieff/comfyui-reactor">GitHub - Gourieff/ComfyUI-ReActor: Fast and Simple Face Swap Extension Node for ComfyUI (SFW)</a>: Fast and Simple Face Swap Extension Node for ComfyUI (SFW) - Gourieff/ComfyUI-ReActor
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1334373376462032956)** (1 messages): 

> `Decompression Time, Loading Weights from Disk` 


- **Curiosity about Decompression vs Direct Loading**: A member expressed interest in understanding how much time the **decompression** process would take compared to just **loading from disk**.
   - They questioned if loading directly from disk would be more efficient than the decomposition method.
- **Performance Comparison Inquiry**: The same member's inquiry also suggests a need to evaluate the performance difference between **loading weights directly** and the **decompression process**.
   - This highlights a broader interest in optimizing model loading times for better efficiency.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1334633334818476182)** (1 messages): 

> `Triton Tensor Indexing, Using tl.gather, InterpreterError` 


- **Triton Unable to Index Tensor Columns**: A user attempted to extract a single column from a tensor using `x[:, 0]` but encountered an **InterpreterError** stating `unsupported tensor index: 0`.
   - *This highlights a limitation in Triton's tensor indexing capabilities.*
- **Efficiency Concerns with tl.gather**: The user considered using `tl.gather` with an index tensor set to all zeros as a workaround to extract the column.
   - However, they expressed concern about the efficiency of this approach compared to direct indexing.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1334300650166550629)** (18 messages🔥): 

> `Blackwell architecture features, sm_X features compatibility, Performance comparisons: RTX 5090 vs RTX 4090, PTX ISA documentation, Tensor Operations discussion` 


- **Blackwell's sm_120a features explained**: Members discussed that the `a` designation in architectures like **sm_120a** indicates features that won't receive future support, making it crucial for those needing both forward compatibility and specific features.
   - The **sm_90a** was the first to introduce this distinction, now seen with **Blackwell** in consumer platforms.
- **sm_X architecture compatibility**: It's noted that **sm_120** implies greater compute capability than **sm_100**, but the 'a' variants can omit certain features from future support.
   - The architectural discussions led to insights on differences between **sm_90a** and other iterations which do not guarantee a super-set of features.
- **RTX 5090 performance vs RTX 4090**: A member questioned the performance disparity, noting that **FP4 with FP32** on **RTX 5090** is approximately **5x** faster than **FP8 on RTX 4090**, yet certain other benchmarks suggest only a **2x** advantage.
   - Concerns were raised about potential inaccuracies in NVIDIA's documentation regarding performance claims, pointing to past discrepancies.
- **Notable resources on PTX ISA**: Discussion highlighted the **PTX ISA documentation** as a valuable resource, particularly for understanding architecture-specific features like sm_100a and sm_101a.
   - Members pointed out that the documentation provides crucial insights on instructions and architectural capabilities.
- **Tensor Operations and RTX Architecture**: Members discussed the lack of certain tensor instructions, noting that **Blackwell** introduces tensor functionalities that previous architectures like **RTX 5 series** do not have.
   - Specifically, innovations in tensor memory and operations such as **tcgen05** were highlighted as significant advancements in the latest architecture.



**Link mentioned**: <a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md">cutlass/media/docs/blackwell_functionality.md at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1334312304283746424)** (2 messages): 

> `PyTorch 2.6 release, FP16 support on X86, Deprecating Conda, Manylinux 2.28 build platform` 


- **PyTorch 2.6 Released with Exciting Features**: We're thrilled to announce the release of **PyTorch® 2.6** featuring enhancements such as `torch.compile` compatibility with **Python 3.13** and the new performance knob `torch.compiler.set_stance`.
   - This release also includes **FP16 support on X86 CPUs**, enriching the capability for performance-sensitive applications.
- **Conda Support Deprecation Announcement**: With the release of **PyTorch 2.6**, the decision has been made to stop publishing updates on **Conda**; details are available in the [deprecation announcement](https://github.com/pytorch/pytorch/issues/138506).
   - Users are encouraged to transition to alternative installation methods as this marks a shift in distribution strategy.
- **New Build Platform Utilized with PyTorch**: The experimental Linux binaries in this release come with **CUDA 12.6.3** and utilize the **Manylinux 2.28 build platform**, ensuring compatibility across various systems.
   - For those interested in building from source, the binaries are configured with **CXX11_ABI=1**, allowing for improved integration.
- **Community Excitement for PyTorch 2.6**: Community members expressed enthusiasm regarding the new features in **PyTorch 2.6**, with one user stating they are 'so hyped about this!!!'.
   - The excitement reflects a strong anticipation for the capabilities this new version will bring to their workflows.



**Link mentioned**: <a href="https://pytorch.org/blog/pytorch2-6/">PyTorch 2.6 Release Blog</a>: We are excited to announce the release of PyTorch® 2.6 (release notes)! This release features multiple improvements for PT2: torch.compile can now be used with Python 3.13; new performance-related kno...

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1334567811585933393)** (1 messages): 

> `GPU Kernel Engineers, GPU Compiler Engineers, Next Gen ML Compiler, Job Openings` 


- **Hiring GPU Kernel and Compiler Engineers**: We're seeking **GPU kernel** and **GPU compiler engineers**, offering **good pay** and **equity grants**.
   - The project aims to build a **next-gen ML compiler** that integrates **AI into the compilation flow**, backed by notable industry figures.
- **Exciting Opportunity in AI Compilation**: The team is looking for expertise in **Triton**, **CUDA**, and **HIP** as they design a cutting-edge solution for ML applications.
   - For more details, visit the job posting at [Mako Dev](https://jobs.mako-dev.com/GPU-Kernel-Engineer-144546aebc36805f9ba3f0b27aafa492), though note that the website is undergoing updates.



**Link mentioned**: <a href="https://jobs.mako-dev.com/GPU-Kernel-Engineer-144546aebc36805f9ba3f0b27aafa492">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1334417723081490485)** (6 messages): 

> `C++ versions, CUDA compatibility` 


- **Choosing the Right C++ Version**: Most discussions indicate that **C++20** is a good starting point for development, particularly when using libraries like **Cutlass** and **Thundekittens** that may require newer standards.
   - However, one member mentions using **C++17** for targeting **Windows**, and another indicates that **C++26** with reflection is preferred on **Linux**.
- **Possible Issues with C++20 and CUDA**: There are concerns that using **C++20** can lead to complications if you require older **CUDA** versions, like **CUDA 11.8** which remains compatible with **PyTorch**.
   - This highlights the importance of aligning your C++ version with the libraries and frameworks you intend to use.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1334451783241826386)** (7 messages): 

> `RTX 5090 Availability, Homemade Meal Creations, Novelty Plates` 


- **Inquiry on RTX 5090 Sales in Europe**: A member inquired about the availability of the **RTX 5090**, noting that NVIDIA's website indicates that sales have not yet started in Europe.
   - *Could anyone from Germany or other European country buy it?*
- **Delicious Meal Description**: A member shared a detailed description of a homemade meal, including **salmon patties, fried potatoes**, and a **homemade waffle** with Greek yogurt.
   - The post even included an image that sparked discussion about its visual similarities to an **egg**.
- **Visual Misinterpretation of Meal**: In response to the meal description, a member humorously noted that at first glance, the meal looked like a **giant egg** in the photo.
   - Another member agreed that canned peaches made it appear even more egg-like.
- **Discussion on Novelty Plate**: A member referenced a **novelty plate** in connection with the meal discussion, specifically mentioning **Tuberculosis Sanatorium 96**.
   - The original poster confirmed the plate's novelty nature, adding an intriguing layer to the meal presentation.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1334371446679670825)** (1 messages): 

> `In-Person Events, Discord Channel Updates` 


- **Plans for More In-Person Events**: A member expressed their intention to host **more in-person events** for the server this year and will provide updates in this channel.
   - They reinforced their commitment to fostering **community engagement** through these events.
- **Discord Channel Notifications**: A member shared a link to a Discord message indicating a forgotten **notification** about one of the channels.
   - This shows the importance of keeping track of channel updates and discussions.


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1334566904433737808)** (4 messages): 

> `ROOK blog post, Progress updates, Modding projects for WoW` 


- **ROOK: A New Horizon in Chess AI**: A new blog post introduces **ROOK (Reasoning Over Organized Knowledge)**, a suite of language models designed to tackle strategic reasoning in chess, moving beyond traditional search algorithms.
   - The project includes three transformer models: **ROOK-CLF**, **ROOK-LM**, and **RookWorld-LM**, aimed at capturing chess knowledge in a human-like manner. [Read the full details here](https://laion.ai/notes/rook/).
- **Long-awaited User Check-in**: A member expressed excitement in reconnecting with another by asking about their current projects or potential breaks.
   - The other member humorously acknowledged the passage of time since they last heard from each other, welcoming the light-hearted banter.
- **Modding Projects for WoW**: It's noted that one member might be involved in **modding projects** for World of Warcraft (WoW), highlighting their creative endeavors.
   - The community seems to admire this member's commitment and talent in engaging with the gaming world.



**Link mentioned**: <a href="https://laion.ai/notes/rook/">ROOK: Reasoning Over Organized Knowledge | LAION</a>: &lt;p&gt;The field of artificial intelligence has long used strategic reasoning tasks as benchmarks for measuring and advancing AI capabilities. Chess, with its in...

  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1334305197161185383)** (1 messages): 

> `Implementing new kernel languages, Low-precision kernels, Future benefits of learning` 


- **Questioning Implementation Time for Kernel Languages**: A member inquired whether it would be a waste of time to implement new kernel languages at this point.
   - They pointed out the **obvious future benefits** of learning a new kernel language for low-precision kernels.
- **Exploring Low-Precision Kernels Purposes**: The discussion highlighted the significance of low-precision kernels in reducing computational overhead and enhancing efficiency.
   - A participant emphasized that adopting a new kernel language can lead to **improved performance** in specific applications.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1334298082497658880)** (15 messages🔥): 

> `Mistral AIx Game Jam results, Parental Control Game, Voice Command Features, Flash Attention Implementation, Llama3-8B R1 Model Improvements` 


- **Mistral AIx Game Jam earns bronze!**: The team secured **#2 at the Mistral AIx 🤗 Game Jam** and aims to win the Community Award. They encourage everyone to [try the game](https://huggingface.co/spaces/Mistral-AI-Game-Jam/ParentalControl) and provide feedback.
   - The game features a mix of **AI** and **Game Development**, using Mistral AI, Godot, and more to create an engaging experience.
- **Game project embraces horror elements**: The game emphasizes survival, requiring players to manage a chaotic environment while keeping a baby safe during a video call. Players can engage with **voice commands** to interact with the baby, leading to amusing outcomes.
   - The developers opted for a **horror vibe** to reflect the stress of parenthood, prompting humor and engagement among players.
- **Flash Attention implementation in CUDA**: A user shared their first CUDA project on [GitHub](https://github.com/akshat-sj/flashattention) showcasing implementations of **flash attention** in raw CUDA. They expressed hope for community feedback on their work.
   - The project features a captured image and details about contributing to the flash attention development, demonstrating their progress and learning in CUDA programming.
- **Optimized Llama3-8B model launched**: The team released a new **Llama3-8B R1** re-distilled model, detailing their cost-effective approach that achieved up to **14% performance gains** on the GSM8K benchmark. The model is available on Hugging Face, promoting efficient runs with HQQ.
   - Their announcement included a link to [a blogpost](https://mobiusml.github.io/r1_redistill_blogpost/) discussing the details of their success while only spending between $3-$18 per training run.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Mobius_Labs/status/1885010791344062704">Tweet from Mobius Labs (@Mobius_Labs)</a>: We’ve improved DeepSeek R1 distilled models using logits distillation—delivering +4-14% gains on GSM8K  while only spending $3-18 per training run! 🚀 Now available on Hugging Face-e—run them efficien...</li><li><a href="https://github.com/akshat-sj/flashattention">GitHub - akshat-sj/flashattention: flash attention in raw cuda</a>: flash attention in raw cuda. Contribute to akshat-sj/flashattention development by creating an account on GitHub.</li><li><a href="https://x.com/amtellezfdez)">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1334366946690601025)** (3 messages): 

> `CUDA versions and TK kernels, Support for Nvidia P100 GPU` 


- **CUDA Versions Significantly Impact TK Kernel Performance**: A member noted that their testing with **CUDA 12.4** for Flash Attention Hopper resulted in only **550 tflops**, significantly lower than the **600 tflops** expected, raising concerns about performance disparities across CUDA versions.
   - They questioned if this was normal, mentioning that **CUDNN sdpa** could reach **590 tflops** in similar settings.
- **Onboarding Support for Nvidia P100 GPU**: A user expressed interest in contributing to **Thunderkittens** by adding support for the **Nvidia P100 GPU**, aiming to enable usage on Google Colab.
   - They invited other members to reach out via DM for onboarding assistance.


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1334258466054869137)** (87 messages🔥🔥): 

> `Reasoning Gym Datasets, Game of Life Challenges, Collaborative Problem-Solving, Codenames Game Mechanics, Murder Mystery Environment` 


- **Increase in Available Datasets in Reasoning Gym**: The Reasoning Gym now boasts **33** datasets, with a new simple dataset gallery established in the [GITHUB repository](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md). This marks significant progress in providing diverse reasoning challenges for reinforcement learning.
   - Contributors are encouraged to submit new datasets and ideas to expand the scope of the platform further.
- **Proposed Interactive Games for Reasoning Tasks**: A discussion around expanding RL environments included ideas for **collaborative problem solving** and **multi-agent negotiation** tasks, allowing for complex scenarios requiring LLM interactions. Suggested scenarios like **team-based coding** aim to foster coordination between multiple agents.
   - These suggestions are aimed at enhancing the capabilities of the Reasoning Gym, bringing in multifaceted challenges that require deeper reasoning and social interaction.
- **Innovative Game of Life Reasoning Challenge**: A new challenge was proposed involving Conway's **Game of Life**, where the model predicts the evolution of an initial random configuration. This task was inspired by the idea of leveraging LLMs for **explanatory reasoning** challenges.
   - The challenge includes determining if a given board setup results in halting or non-halted conditions based on defined rules.
- **Integrating Codenames Mechanics into Reasoning Tasks**: The game **Codenames** was discussed as a potential task where LLMs give hints based on selected words to strategize their responses. This can highlight how models can operate on both sides using shared cognitive associations.
   - The discussion reflects ongoing efforts to leverage existing games to create engaging and meaningful reasoning environments.
- **Murder Mystery as a Multi-Turn Environment**: The implementation of a **murder mystery** environment was considered, allowing for interaction without the need for a dungeon master. This setup focuses on logic-based elimination and could lead to further exploration of multi-turn agent interactions.
   - The potential use of dynamic interaction frameworks could greatly enhance problem-solving scenarios in such games.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Category:Logic_puzzles">Category:Logic puzzles - Wikipedia</a>: no description found</li><li><a href="https://x.com/karpathy/status/1884676486713737258">Tweet from Andrej Karpathy (@karpathy)</a>: For friends of open source: imo the highest leverage thing you can do is help construct a high diversity of RL environments that help elicit LLM cognitive strategies. To build a gym of sorts. This is ...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/24">Add WordLadder game dataset (word golf) · Issue #24 · open-thought/reasoning-gym</a>: Create a word ladder game dataset class with configuration and unit tests. Include simple example in the question to define the response format. On github are many word ladder implementations, e.g....</li><li><a href="https://en.wikipedia.org/wiki/Recreational_mathematics">Recreational mathematics - Wikipedia</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/List_of_recreational_number_theory_topics">List of recreational number theory topics - Wikipedia</a>: no description found</li><li><a href="https://vimeo.com/1022776731?autoplay=1&muted=1&stream_id=Y2xpcHN8MjI4Mjc5MjI0fGlkOmRlc2N8eyJyZW1vdmVfdm9kX3RpdGxlcyI6ZmFsc2V9">OARC - C3PO demo 1</a>: This is &amp;quot;OARC - C3PO demo 1&amp;quot; by Leonardo Borch on Vimeo, the home for high quality videos and the people who love them.</li><li><a href="https://vimeo.com/1036385433?autoplay=1&muted=1&stream_id=Y2xpcHN8MjI4Mjc5MjI0fGlkOmRlc2N8eyJyZW1vdmVfdm9kX3RpdGxlcyI6ZmFsc2V9">OARC in chat snake render, html</a>: This is &amp;quot;OARC in chat snake render, html&amp;quot; by Leonardo Borch on Vimeo, the home for high quality videos and the people who love them.</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/38d64649f525363f4525db02d27a993ed8fbd72b/reasoning_gym/cognition/rubiks_cube.py#L38">reasoning-gym/reasoning_gym/cognition/rubiks_cube.py at 38d64649f525363f4525db02d27a993ed8fbd72b · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/23">Adds &quot;Quantum Lock&quot; Puzzle by Miserlou · Pull Request #23 · open-thought/reasoning-gym</a>: In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but th...</li><li><a href="https://youtu.be/JheGL6uSF-4?si=EE1aKCt4C3MiYxXM">I Made a Graph of Wikipedia... This Is What I Found</a>: Code for all my videos: https://github.com/sponsors/adumb-codes/Get the graph as a poster: https://adumb.store/Twitter: https://twitter.com/adumb_codesA deep...</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>: A comprehensive repository of reasoning tasks for LLMs (and beyond) - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/28">Add BF (Brainf*ck) Challenges by Miserlou · Pull Request #28 · open-thought/reasoning-gym</a>: Adds the first &amp;#39;code&amp;#39; challenge: BF!This is a code execution challenge, not a code generation challenge.There are three difficulty levels. Level 1 is a simple print string. Level 2 use...</li><li><a href="https://github.com/ironman5366/ai-murder-mystery-hackathon">GitHub - ironman5366/ai-murder-mystery-hackathon: The game is afoot</a>: The game is afoot. Contribute to ironman5366/ai-murder-mystery-hackathon development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/mlabonne/agentic-datagen">The Rise of Agentic Data Generation</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md">reasoning-gym/GALLERY.md at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/30">Add Conway&#39;s Game of Life Simulations by Miserlou · Pull Request #30 · open-thought/reasoning-gym</a>: (Worked off of the BF branch, sorry about that. Can cherry pick if you like!)Adds Conway&amp;#39;s Game of Life in a configurable way:    config = GameOfLifeConfig(        seed=42,         size=1, ...</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef: 🍲Agent Chef🥘 is my robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and clean their fine-tuning data, eliminating data poisoning and low-quality knowledge bases. Additionally, it will provide templates, and frameworks.</a>: 🍲Agent Chef🥘 is my robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and ....</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1334303351793520751)** (74 messages🔥🔥): 

> `DeepSeek models, Running models with GPT4All, Integrating Ollama with GPT4All, Local document management, AI education tools` 


- **DeepSeek models performance**: Members discussed the upcoming release of DeepSeek, expressing anticipation for its performance with math and LaTeX support.
   - Some noted that using DeepSeek for complex tasks may require managing context size effectively due to VRAM constraints.
- **Integration of GPT4All and Ollama**: Users confirmed that it is possible to connect GPT4All to Ollama by running it as a server and using the OpenAI API within GPT4All.
   - There were inquiries about documentation for this integration, with some members successfully finding relevant resources.
- **Loading remote LLMs in GPT4All**: Discussion included steps on how to load remote LLMs into the GPT4All GUI, with suggestions to ensure proper setup for API keys.
   - Members proposed clearer documentation to aid new users in accessing remote models effectively.
- **Developing AI education tools**: One member shared their initiative to build an AI-driven education tool for kids in Africa, emphasizing offline accessibility and localized content.
   - They plan to use lightweight AI models and a collection of curated resources to facilitate self-learning without the need for internet access.
- **Model quantization differences**: A member sought clarification on the naming conventions of models, specifically the difference between those with and without '-I1-' in their names.
   - No definitive answers were found, indicating a need for better transparency or documentation on model specifications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html:">GPT4All</a>: GPT4All Docs - run LLMs efficiently on your hardware</li><li><a href="https://hastebin.skyra.pw/tucewahaca.kotlin">Hastebin</a>: no description found</li><li><a href="https://emmanuelsibanda.hashnode.dev/funda-ai-building-a-laptop-powered-by-ai-to-help-students-in-africa-learn">Funda AI - building a laptop powered by AI to help students in Africa learn</a>: FundAI provides AI-powered laptops to help African students learn, focusing on exams, logical thinking, and tech skills without internet reliance</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Local-API-Server">Local API Server</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Que">Home</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://anythingllm.com/desktop">Download AnythingLLM for Desktop</a>: Download the ultimate &quot;all in one&quot; chatbot that allows you to use any LLM, embedder, and vector database all in a single application that runs on your desktop. 100% privately.</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3440">Support for deekseek thinking in the gui. by manyoso · Pull Request #3440 · nomic-ai/gpt4all</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1334296931471593503)** (47 messages🔥): 

> `MCP Server Integration, Self-Hosted Web Clients, Cursor MCP Support, Environment Variables for MCP, Function Calling Issues` 


- **Cursor Adds MCP Support with Limitations**: Members expressed excitement over Cursor adding MCP support, although there's currently a limitation on adding environment variables.
   - One suggested using syntax like `FOO=bar npx some-server` to set variables, pointing to potential workarounds.
- **Self-Hosted Web Client Takes Center Stage**: A user shared insights on their self-hosted web client that manages multiple MCP servers and agents, allowing automatic hand-offs.
   - This approach promises seamless operation whether locally or in the cloud, showcasing flexibility in hosting.
- **Discussion on Function Calling in MCP**: Members discussed having trouble with an 8b model that reportedly struggles with function calling and tool usage.
   - Interest was noted in ensuring better integration and understanding of MCP among users, particularly on platforms like Reddit.
- **Dynamic Agent Prompts Absent for MCP**: A member stated that while dynamic agent prompts are not yet implemented, system configuration can be defined simply via prompts.
   - Thus, users can customize agent behavior without complex setups, potentially increasing usability.
- **Config Structure Comparison for MCP vs LSP**: Concerns were raised about MCP not utilizing the same configuration structure as Language Server Protocol (LSP), which allows the server to request config from the client.
   - This disparity in structure was viewed as a limitation in the current MCP implementation.



**Link mentioned**: <a href="https://www.gnu.org/software/coreutils/manual/html_node/env-invocation.html">env invocation (GNU Coreutils 9.6)</a>: no description found

  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1334266918034735306)** (12 messages🔥): 

> `Hataraku SDK Proposal, TypeScript CLI Development, Collaborative Development, User Testing Feedback` 


- **Hataraku Project Gains Momentum on ShowHN**: A project called **Hataraku** is trending **#1** on ShowHN, prompting discussions and support requests from the community on Hacker News.
   - Participants are encouraged to contribute ideas and engage in broader discussions regarding the project.
- **Moonlife's TypeScript CLI Under Development**: Moonlife is actively developing a **TypeScript** version of the Hataraku project and has begun work on a repository, indicating progress.
   - The CLI functionality is already operational, but further abstraction is needed to refine the tool.
- **Collaboration on Hataraku’s TypeScript Implementation**: Saqadri offers to collaborate with Moonlife, particularly in refining the CLI or discussing potential improvements for the TypeScript version.
   - Moonlife confirms they have forked existing code to leverage necessary infrastructure for development.
- **Interface Development in Final Stages**: Moonlife indicates that creating the interface is the last significant step, having progressed well with the core functionality.
   - Feedback is sought from others in the community, with an invitation for direct messaging to share insights.
- **User Testing and Feedback Opportunities**: Neil expresses interest in testing the new interface, highlighting their experience as a user with complex workflows to provide useful feedback.
   - This inquiry reflects ongoing community involvement in ensuring the usability of the evolving Hataraku project.



**Link mentioned**: <a href="https://github.com/turlockmike/hataraku/blob/main/docs/sdk-proposal.md">hataraku/docs/sdk-proposal.md at main · turlockmike/hataraku</a>: An autonomous coding agent and SDK for building AI-powered development tools - turlockmike/hataraku

  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1334657995463856169)** (1 messages): 

> `NotebookLM Usability Study, User Experience Feedback` 


- **NotebookLM seeks user feedback**: NotebookLM UXR is organizing remote chat sessions to hear about users' first experiences with the product and how they currently use it. Participants will receive **$75** (or equivalent) as a thank you for their insights.
   - Interested individuals can fill out the [screener form](https://forms.gle/HJmCwNepsfPSdC7g7) to apply for one of these 60-minute sessions scheduled for **February 6th, 2025**.
- **Upcoming usability study details**: Participants need a high-speed Internet connection, an active Gmail account, and a device with video and audio capabilities for the usability study. This study is focused on gathering feedback for future product enhancements, emphasizing the importance of user needs.



**Link mentioned**: <a href="https://forms.gle/HJmCwNepsfPSdC7g7">Participate in an upcoming Google UXR study!</a>: Hello,I’m contacting you with a short questionnaire to verify your eligibility for an upcoming usability study with Google. This study is an opportunity to provide feedback on something that&#39;s cur...

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1334451854028832789)** (5 messages): 

> `Using AI for learning, NotebookLM Audio Overview, DeepSeek R1, Transcription for understanding, Explaining concepts in different terms` 


- **AI transforms trading course content**: A user shared how they converted trading course videos to audio, transcribed them using AI, and utilized [NotebookLM](https://link.to.notebooklm) to clarify complex topics for peers.
   - One memorable approach was using League of Legends terminology to explain the concept of **Big Players**, demonstrating AI's versatility in framing information.
- **NotebookLM dissects Executive Order in record time**: AI is noted for its efficiency in summarizing complex content, as evidenced by a review within **24 hours** of a new Executive Order focusing on public education privacy.
   - Listeners are directed to a detailed [YouTube video](https://youtu.be/8RFYmgYn7P4?si=r9k0LVu_hOksnA4i) for an objective overview of the Executive Order's implications.
- **NotebookLM Podcast breaks down DeepSeek R1**: The NotebookLM Podcast tackled **DeepSeek R1**, explaining its features like **GRPO** and **Mixture of Experts** in simple terms to make the complex AI technology accessible.
   - Listeners can engage with the [full discussion here](https://youtube.com/watch?v=zVDmKv3hWzk) which includes benchmarking analyses and a quick demo.
- **Conversations in Audio Overview not recorded**: A query arose regarding the persistence of conversations held in Interactive Mode during the Audio Overview, confirming they are not saved in the downloadable recordings.
   - This highlights the limitations of the current design in capturing user interactions during dynamic discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/watch?v=zVDmKv3hWzk">Is DeepSeek R1 the New ChatGPT Killer? NotebookLM Explains! 🔥</a>: 🚀 DeepSeek V3 and R1 Explained Using NotebookLM!DeepSeek R1 is making shockwaves in the AI world, and today, we’re breaking it all down using NotebookLM! Th...</li><li><a href="https://youtu.be/8RFYmgYn7P4?si=r9k0LVu_hOksnA4i">Objective NotebookLM review of an Executive Order focused on public education privacy &amp; patriotism</a>: A new Executive Order signed by President Trump focuses on &quot;Ending radical indoctrination in k-12 schooling.&quot; Google&#39;s NotebookLM Audio Overview tool reviewe...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1334278115119071374)** (52 messages🔥): 

> `NotebookLM Features and Performance, Audio Generation Feedback, Gemini Updates, User Experience Issues, Podcast Insights` 


- **NotebookLM slow generation times**: Users report varied experiences with generation times after clicking the 'study guide' button, with estimates ranging from **10 to 30 minutes** depending on the number of sources involved.
   - Some users have found that even a single source can take unexpectedly long, raising concerns about consistency in performance.
- **Audio Overviews struggling in other languages**: A trainer reported poor performance of Audio Overviews when tested with **Korean** and **Japanese**, indicating issues in multilingual support.
   - Participants noted difficulties and expressed desire for improved functionality in these languages, querying others for their experiences.
- **Gemini 2.0 Flash update causing glitches**: After updates to **Gemini 2.0 Flash**, users experienced temporary glitches, leading to discussions on its impact on performance.
   - The update is believed to have contributed to issues some users faced, although functionality resumed thereafter.
- **Seeking stricter source utilization rules**: Some users are exploring ways to restrict responses strictly to uploaded sources, seeking more definitive directives for the NotebookLM.
   - Feedback suggests that while users can add prompts for better source compliance, the output sometimes incorporates external references, which complicates the expected binary response.
- **Podcast featuring NotebookLM insights**: A podcast featuring NotebookLM's founding engineer provides insights into the platform's history and growth, generating interest among users.
   - Listeners expressed curiosity about future features but noted a lack of specific details shared during the conversation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Cr7J2PLo2fw">A Conversation with NotebookLM&#39;s Founding Engineer</a>: Google&#39;s NotebookLM has become one of the most compelling AI tools for working with text. In this conversation, Adam Bignell, the project&#39;s founding engineer...</li><li><a href="https://youtube.com/watch?v=zVDmKv3hWzk">Is DeepSeek R1 the New ChatGPT Killer? NotebookLM Explains! 🔥</a>: 🚀 DeepSeek V3 and R1 Explained Using NotebookLM!DeepSeek R1 is making shockwaves in the AI world, and today, we’re breaking it all down using NotebookLM! Th...</li><li><a href="https://www.tiktok.com/t/ZT22DHefp/">TikTok - Make Your Day</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1334270047979442309)** (1 messages): 

> `Branch Changes, Pull Requests` 


- **Branch Changes are Completed**: The **branch changes** are now complete, and all outstanding pull requests have been successfully **retargeted**.
   - Team members are encouraged to **reach out** with any questions regarding these updates.
- **Update on Pull Requests**: All open **pull requests** have been retargeted in line with the recent branch changes.
   - This adjustment aims to streamline the workflow and facilitate smoother integrations.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1334338934632747019)** (48 messages🔥): 

> `NeoVim LSP integration, Mojo 1.0 discussions, Backwards compatibility concerns, Reflection in Mojo, Benchmarking Mojo performance` 


- **Integrating Mojo LSP with NeoVim**: Members discussed how to add Mojo LSP support to NeoVim, referencing the [nvim/lspconfig GitHub](https://github.com/neovim/nvim-lspconfig).
   - However, some reported that the solutions proposed were not working as intended.
- **Defining Mojo 1.0: Speed vs. Stability**: Clattner emphasized the need for a meaningful Mojo 1.0, characterizing it as a language ideal for fast execution and GPU utilization.
   - Discussion highlighted the tension between achieving immediate usability and ensuring long-term stability and compatibility.
- **Concerns Over Backwards Compatibility**: Members voiced concerns regarding the lack of backwards compatibility, which could hinder adoption of new versions due to potential breaking changes.
   - The overall consensus stressed that ensuring compatibility with legacy libraries is essential for a thriving ecosystem.
- **Importance of Reflection in Mojo**: There was a debate about whether reflection features should be included in Mojo 1.0, given their importance for use cases like data serialization.
   - Concerns were raised about how lacking reflection could affect usability, but it was noted that some reflection capabilities are currently implemented.
- **Benchmarking Mojo's Performance**: Members discussed the necessity of benchmarking Mojo on larger compute clusters to evaluate its performance effectively.
   - The idea was that ensuring robust performance on high-memory machines would simplify development for users with smaller configurations.



**Link mentioned**: <a href="https://youtu.be/9ag0fPMmYPQ),">Mojo🔥: a deep dive on ownership with Chris Lattner</a>: Learn everything you need to know about ownership in Mojo, a deep dive with Modular CEO Chris LattnerIf you have any questions make sure to join our friendly...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1334295061328035841)** (38 messages🔥): 

> `Mistral Small 3, DeepSeek Database Leak, Riffusion's New Model, OpenAI API Latency Monitoring, ElevenLabs Series C Funding` 


- **Mistral Small 3 Launches with Impressive Specs**: Mistral AI announced **Mistral Small 3**, a 24B-parameter model with **81% accuracy on MMLU** and **150 tokens/sec** performance, now available under Apache 2.0 license.
   - Notable improvements include fewer layers and a significant upgrade to vocabulary size, with detailed comparisons against other models shared across social media.
- **DeepSeek Faces Major Database Exposure**: A public ClickHouse database belonging to **DeepSeek** was discovered, exposing sensitive internal data including chat history and secret keys.
   - The issue was responsibly disclosed and quickly secured after being highlighted by **Wiz Research**, raising concerns about data security in the AI industry.
- **Riffusion Introduces FUZZ, a Generative Music Model**: Riffusion unveiled **FUZZ**, a generative model aimed at producing **high-quality music**, which they are offering for free as long as GPU resources last.
   - The announcement highlights the continued development and capabilities of generative music models, indicating active innovation in this space.
- **Monitoring OpenAI API Latency Discussed**: Concerns about potential **increased latency** in OpenAI APIs prompted discussions on third-party monitoring solutions like **OpenRouter** and **Artificial Analysis**.
   - While preliminary checks show normal latency, community members exchange insights on available tools to better gauge API performance over time.
- **ElevenLabs Raises $180M in Series C**: ElevenLabs secured a **$180M Series C** round led by **a16z & ICONIQ**, emphasizing their commitment to enhancing AI capabilities.
   - This significant funding round signals strong investor confidence in AI voice technologies and the potential market impact.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/allen_ai/status/1884966600039915809">Tweet from Ai2 (@allen_ai)</a>: Here is Tülu 3 405B 🐫 our open-source post-training model that surpasses the performance of DeepSeek-V3! The last member of the Tülu 3 family demonstrates that our recipe, which includes Reinforcemen...</li><li><a href="https://block.github.io/goose/">codename goose | codename goose</a>: Your open source AI agent, automating engineering tasks seamlessly.</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://x.com/dchaplot/status/1884975434519245021">Tweet from Devendra Chaplot (@dchaplot)</a>: Performance of Mistral Small 3 Instruct modelDownload on HF:https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-25014/N</li><li><a href="https://x.com/eliebakouch/status/1884979232280813856">Tweet from elie (@eliebakouch)</a>: New mistral (not so) small, rope -&gt; 100M for 32k context length - no swa this time. For reference Qwen 1M context length have a rope base of 10M. I&#39;m curious</li><li><a href="https://artificialanalysis.ai/providers/openai">OpenAI - Quality, Performance &amp; Price Analysis | Artificial Analysis</a>: Analysis of OpenAI&#x27;s models across key metrics including quality, price, output speed, latency, context window &amp; more.</li><li><a href="https://x.com/riffusionai/status/18849">Tweet from ok, i'll byte 🇺🇦 (@cw)</a>: Shortest flight I&#39;ve ever experienced: 20 min from Santorini to Mykonos. Cell phones were allowed to function.</li><li><a href="https://x.com/stochasticchasm/status/1885005454369272002">Tweet from stochasm (@stochasticchasm)</a>: The new mistral small has a narrower model dim, double the ff dim, and 16 less layers than the previous mistral small? That’s like a ff_dim/model_dim ratio of 6! That’s so weird</li><li><a href="https://x.com/espadrine/status/1885004488206856638">Tweet from Thaddée Tyl (@espadrine)</a>: Mistral Small 2 to 3 changes:• From 55 to 40 layers for latency• From 33K to 131K vocabulary• From 6K to 5K embedding• From 48 to 32 attention heads• 10x rope_theta• SYSTEM_PROMPT tokenhttps://mistral...</li><li><a href="https://x.com/sophiamyang/status/1884970987441316268">Tweet from Sophia Yang, Ph.D. (@sophiamyang)</a>: 🚀 Announcing @MistralAI Small 3, our most efficient and versatile model yet! ✅ 24B parameters✅ 81% MMLU✅ 150 tokens/sec✅ Apache 2.0 license✅ Pre-trained & instructed (No synthetic data – ideal for re...</li><li><a href="https://x.com/kagigz/status/1884670976656630059">Tweet from Katia Gil Guzman (@kagigz)</a>: At OpenAI DevDay, we introduced the Realtime API with a demo of an interactive solar system you can navigate with your voice.A lot of you have asked how it was built, so we&#39;ve open-sourced it—with...</li><li><a href="https://x.com/TheRealAdamG/status/1884971520348283217">Tweet from Adam.GPT (@TheRealAdamG)</a>: https://help.openai.com/en/articles/6825453-chatgpt-release-notes#h_caaeddc37eChatGPT got some nice, incremental updates yesterday.    Shavings make a pile.</li><li><a href="https://x.com/riffusionai/status/1884984941081198954?s=46">Tweet from Riffusion (@riffusionai)</a>: Introducing FUZZ — a generative music model like no other.  Personalized, full-length, high-quality, and infinite. We’re making this instrument free for as long as our GPUs survive.   The best of FUZZ...</li><li><a href="https://x.com/matistanis/status/1885011065018163224">Tweet from Mati Staniszewski (@matistanis)</a>: Today, a new chapter begins for ElevenLabs - we closed our $180M Series C co-led by a16z & ICONIQ to give every AI agent a voice.</li><li><a href="https://x.com/iScienceLuvr/status/1884736091619537346">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Anyone who thinks DeepSeek just came out of nowhere should see this graph.For each model on this graph, weights, code, and detailed papers were released.This is a team with a strong track record and h...</li><li><a href="https://x.com/alibaba_qwen/status/1884809286288810231?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: Announcing Qwen2.5-VL Cookbooks! 🧑‍🍳A collection of notebooks showcasing use cases of Qwen2.5-VL, include local model and API. Examples include Compute use, Spatial Understanding,  Document Parsing,...</li><li><a href="https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak">Wiz Research Uncovers Exposed DeepSeek Database Leaking Sensitive Information, Including Chat History | Wiz Blog</a>: A publicly accessible database belonging to DeepSeek allowed full control over database operations, including the ability to access internal data. The exposure includes over a million lines of log str...</li><li><a href="https://x.com/altryne/status/1884778839009796411?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Zuck highlights from the earnings call: - LLama 4 & LLama 4 mini (done with pre-training)- Confirms reasoning LLaMas! - Llama 4 will be natively multimodal -- it&#39;s an omni-model -- and it will hav...</li><li><a href="https://openrouter.ai/openai/o1-preview">o1-preview - API, Providers, Stats</a>: The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.The o1 models are optimized for math, science, programming, and other STEM-related tasks...</li><li><a href="https://x.com/mistralai/status/1884967826215059681?s=46">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://github.com/openai/openai-realtime-solar-system">GitHub - openai/openai-realtime-solar-system: Demo showing how to use the OpenAI Realtime API to navigate a 3D scene via tool calling</a>: Demo showing how to use the OpenAI Realtime API to navigate a 3D scene via tool calling - openai/openai-realtime-solar-system</li><li><a href="https://archive.is/KiSYM">OpenAI says it has evidence China&#x2019;s DeepSeek used its model to train &#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1334255715757461595)** (8 messages🔥): 

> `Tracks information, Sign up responses, Quiz 1 release, LLM Agents Quiz Repo, Certificate updates` 


- **Tracks Information Awaited**: Members expressed curiosity about the **tracks** being discussed in both **application** and **research** contexts, with promises of more information to come from organizers.
   - *Stay tuned!* for updates regarding the content of these tracks.
- **Sign Up Confirmation Delays**: Several participants reported that they filled out the Google Forms sign-up sheet but have yet to receive any responses about their status.
   - They are eager for updates, especially in relation to pursuing **PhD** opportunities.
- **Quiz 1 Availability**: A member inquired about the release of **Quiz 1**, which has been confirmed to be on the course website under the syllabus section.
   - Details regarding the **first course certifications** are still pending, with members advised to wait for future updates.
- **Seeking Previous Quiz Answers**: A participant requested access to a repository of answers for quizzes from the previous **LLM Agent course**.
   - A link to a [Google Document with quizzes archive](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing) was shared, but a note indicated the browser version is outdated.
- **Certificates Not Yet Released**: It was confirmed that **certificates** for the course have not been released yet, and more information is expected soon.
   - Members are encouraged to stay informed as specific requirements for the current semester's certifications will be unveiled later.



**Link mentioned**: <a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing">Quizzes Archive - LLM Agents MOOC</a>: NOTE: The correct answers are in the black boxes (black text on black background). Highlight the box with your cursor to reveal the correct answer (or copy the text into a new browser if it’s hard to ...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1334334453803384883)** (5 messages): 

> `Lecture Uploads, Berkeley Policies on Accessibility, Lecture Access via Website` 


- **Delay in Uploading First Lecture**: A member requested the upload of the **1st lecture** today, suggesting it only requires **5 minutes** of work from the team.
   - Another member noted that the process involves significant edits and **captioning** due to **Berkeley's policies**.
- **Berkeley's Accessibility Requirements**: Concerns were raised regarding the release of videos without full **accessibility accommodations**, such as captions.
   - Team members emphasized the importance of patience as they work through these requirements for public release.
- **Accessing Lecture Recordings Online**: Members were reminded that the lecture recording is available for viewing on the [website](https://llmagents-learning.org/sp25), accessible via the livestream link.
   - It was clarified that while the edited version isn't public yet due to ongoing captioning, it remains viewable through the provided link.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1334599628737351691)** (2 messages): 

> `AI Agent Workshop, LlamaIndex on BlueSky` 


- **Mastering AI Agents Workshop**: Join @seldo's comprehensive workshop to learn how to build advanced **AI agents** and **multi-agent systems** with **LlamaIndex**! The workshop will cover **AgentWorkflow** and the fundamentals of creating robust multi-agent frameworks, providing hands-on experience from [here](https://t.co/UKIClalkKG).
   - Participants will dive into **Workflows**, the essential building blocks necessary for enhanced agent capabilities, ensuring a deep understanding of multi-agent system architecture.
- **LlamaIndex lands on BlueSky**: LlamaIndex has officially joined **BlueSky**! Stay connected and follow their journey as they explore new opportunities on this emerging platform: [link](https://t.co/GK4L8Sb2N6).
   - Engage with the community and discover interesting discussions happening on **BlueSky** about AI developments and innovations.



**Link mentioned**: <a href="https://t.co/GK4L8Sb2N6">LlamaIndex (@llamaindex.bsky.social)</a>: The framework for connecting LLMs to your data.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1334570774157328517)** (10 messages🔥): 

> `LlamaIndex support for o1, O1 streaming issues, OpenAI model capabilities` 


- **LlamaIndex claims support for o1**: A user questioned why **LlamaIndex** only supports `o1-preview` but flagged that `o1` is indeed supported after an update with `pip install -U llama-index-llms-openai`.
   - However, it's noted by some that full functionality might not be available.
- **O1 model lacks streaming support**: Concerns were raised that the **o1** model does not have proper support for streaming, which was successful with `o1-preview` and **o1-mini**.
   - Error messages indicated an unsupported value for streaming in `o1`, leading to further discussions.
- **Research reveals OpenAI's limitations**: After further research, it was concluded that **OpenAI** has not fully developed the capabilities of the **o1** model.
   - A relevant [community discussion](https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043?utm_source=chatgpt.com#:~:text=Streaming%20of%20the,for%20this%20model.) highlighted these limitations.
- **Weird support experiences with o1**: Members commented on the **weird** support experiences related to the **o1** model from **OpenAI**, pointing out many features being unsupported.
   - This has led to confusion and frustration among users trying to leverage the new model.



**Link mentioned**: <a href="https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043?utm_source=chatgpt.com#:~:text=Streaming%20of%20the,for%20this%20model.">Streaming support for o1 (o1-2024-12-17) (resulting in 400 &quot;Unsupported value&quot;)</a>: Hello, it appears that streaming support was added for o1-preview and o1-mini (see announcement OpenAI o1 streaming now available + API access for tiers 1–5).  I confirm that   both work for me.  Howe...

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1334349996170154066)** (11 messages🔥): 

> `NVIDIA GPUs and Hypervisors, Interconnecting Tiny Boxes, VRAM Sharing Techniques, Performance of Tiny Boxes, Physical Server Choices for LLMs` 


- **Testing Needed for GPU Setup**: A member expressed the importance of **testing** regarding configurations with multiple NVIDIA GPUs when using the **P2P patch**.
   - They inquired whether others are utilizing a **hypervisor like Proxmox** or opting for baremetal installations due to limitations on IOMMU.
- **Curiosity on Tiny Box Interconnectivity**: A member pondered how many **Tiny Boxes** can be interconnected and queried about sharing **VRAM** between them while discussing the achievable inference performance.
   - Another noted the lack of a **seamless method** to share VRAM but suggested using a fast **NIC/connectx** card for network-based inference which could scale nicely.
- **Inference Performance Estimates**: Estimations indicated that if a model could handle **15 tokens/sec**, theoretically it could serve **100 requests** at slightly lower speeds (14 tok/sec each) when scaled.
   - This highlights potential performance characteristics of distributed requests under well-defined conditions.
- **Exploration of MLX for Tiny Boxes**: Discussion about using **MLX** to aggregate the capabilities of Tiny Boxes led to some confusion about its specific role in this context.
   - The reference to **Apple Silicon tensor libraries** indicated some mixed interpretations of **MLX**'s applicability in their setup.
- **Seeking Recommendations for Physical Servers**: A member expressed interest in purchasing a **physical server** to host LLMs locally for enterprise use, seeking advice on ideal choices.
   - This indicates a growing interest in self-hosting solutions for large scale models in enterprise settings.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1334411118583484416)** (1 messages): 

> `Sample Code for Blocked/Fused Programs, Tensor Operations` 


- **Looking for Blocked/Fused Code Samples**: A user inquired if there is any **good sample code** available for implementing **blocked/fused programs** in tinygrad.
   - They specifically requested examples demonstrating how to **load/write tensor blocks** for conducting operations efficiently.
- **Discussion on Tensor Block Operations**: The conversation revolved around how to efficiently perform operations by **processing tensors block-by-block** in tinygrad.
   - Members highlighted the importance of **fusing operations** to enhance performance and minimize resource usage.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1334549568259489926)** (3 messages): 

> `AI Emotional Response, Humanizing AI, Perception of AI Models` 


- **Users feel AI models are cold**: A user expressed that **AI models** appear to be somewhat cold in their interactions.
   - This sentiment caused others to joke about the need for a *blanket* to warm them up.
- **Machines don't need warmth**: Another member pointed out that **AI models** are machines and ultimately don't require any warmth or human emotions.
   - This comment furthered the lighthearted discussion about perceiving AI as more emotional entities.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1334344851092930580)** (1 messages): 

> `Support Tickets, Discord Channel Communication` 


- **Support Ticket Created**: A member created a [support ticket](https://discord.com/channels/954421988141711382/1334344003994386474/1334344003994386474) for assistance, ensuring that the issue is noted and tracked.
   - This reinforces the importance of keeping communication clear in Discord channels for efficient problem-solving.
- **Follow-up Communication Importance**: Following up on support tickets is crucial to maintaining clear communication and resolving issues efficiently.
   - Members discussed best practices for ensuring that support channels remain active and responsive.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1334297971516248124)** (8 messages🔥): 

> `command-r7b, Command R model, distillation frameworks` 


- **User struggles with command-r7b and distillation frameworks**: A member expressed difficulties getting **command-r7b** to respond to **distillation frameworks** like synthetic data generation, inquiring for suggestions about using **ollama**.
   - This indicates potential gaps in the existing support for the integration of various **frameworks** with the command-r7b model.
- **Insight on Command R capabilities**: In a follow-up, the bot provided an overview of **Command R**, detailing its characteristics as a large language model optimized for conversational tasks and retrieval-augmented generation.
   - Command R features a **128,000-token context length**, supports tool use for complex workflows, and has enhancements aimed at improving decision-making and data analysis in the upcoming release.
- **Resources for further learning**: The bot included links to additional reading on the **models** overview, specific details about **Command R**, and its retrieval-augmented generation capabilities.
   - These resources can provide members with deeper insights into the functionality and performance of Command R: [Models Overview](https://docs.cohere.com/v1/docs/models), [The Command R Model](https://docs.cohere.com/v1/docs/command-r), [Command R Changelog](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1334320254872059985)** (6 messages): 

> `Adding proxy to dspy.LM adapter, Supported LLMs in DSPy, Setting litellm client with http_client, Documentation references, LiteLLM model support` 


- **Adding Proxy to dspy.LM Adapter**: A user inquired about how to add a **proxy** to the `dspy.LM` adapter, referencing its addition in a [GitHub PR](https://github.com/stanfordnlp/dspy/pull/1331). The function was previously implemented in the deprecated `gpt3.py` module, raising concerns about compatibility.
   - Another user mentioned that they can't use **dspy 2.6** due to proxy requirements for their hosted endpoints.
- **Supported LLMs in DSPy**: A newcomer asked which **LLMs** are supported in DSPy, prompting a member to share a link to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) detailing various model providers.
   - The documentation lists support for **OpenAI**, **Azure**, and **VertexAI** models, among others.
- **Setting litellm client with http_client**: One user expressed difficulty in finding information about setting a **litellm client** with `http_client` using **SSL context** in the DSPy parameters. They mentioned that this setting isn't specified in the available documentation.
   - Discussion continued with references to specific lines in the [dspy/lm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53) file, highlighting the framework details.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1331">http client added to GPT3 by mjaliz · Pull Request #1331 · stanfordnlp/dspy</a>: Set http_client on openai to support http_proxy</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53">dspy/dspy/clients/lm.py at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—language models - stanfordnlp/dspy
</li>
</ul>

</div>
  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1334251809111543951)** (6 messages): 

> `Axolotl for KTO, New Mistral model, User tasks and feature requests, Winter semester calendar, Mistral AI open source commitment` 


- **Facing Challenges with Axolotl for KTO**: *It's going to be in tough luck if we can't use Axolotl for KTO*, highlighting urgency around the integration of Axolotl.
   - Members expressed concern over the feasibility, with one asking if the tasks could be completed and offering to help review.
- **Excitement for New Mistral Model Release**: A member shared excitement over the announcement of the new [Mistral-Small-24B-Base-2501 model](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501), boasting **24B parameters** and ranking high for small LLMs.
   - It was noted that there will be additional commercial models for specialized capabilities, emphasizing **Mistral AI's commitment to open source**.
- **Uncertainty about Mistral Model Performance**: When asked if the new Mistral model works, a member admitted, *I haven't trained in a while so I don't know*.
   - This indicates a lack of recent hands-on experience with the models, opening up a dialogue about user experiences.
- **Busy Winter Semester Calendar**: A member cited a busy schedule for the winter semester, saying, *Sorry for the winter semester this year my calendar looks very stuffed*.
   - This might affect their availability for collaborative tasks in the upcoming months.



**Link mentioned**: <a href="https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501">mistralai/Mistral-Small-24B-Base-2501 · Hugging Face</a>: no description found

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1334599850293067837)** (3 messages): 

> `Farm Friend, Cliche Reviews` 


- **Inquiry About Farm Friend**: A member expressed their fondness for **Farm Friend**, noting that it was enjoyed last year but seems missing now.
   - There’s community curiosity regarding the current status of the project.
- **Meme Analysis and Cliché Reviews**: Another member humorously commented on the **cliché reviews** within the community, eliciting a light-hearted reaction.
   - An image was shared that likely illustrates this sentiment, reinforcing the playful tone of the discussion.
- **Clarifying Meaning of 01**: A member clarified their earlier message regarding '01', specifying that it did not pertain to **OpenAI**.
   - This comment suggests the conversation may have included misunderstandings or miscommunications.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1334468225999573012)** (2 messages): 

> `DCP Checkpointing, Config Settings` 


- **DCP Checkpointing Status in Configs**: A member raised a question about whether **DCP checkpointing** is enabled in any of the current configs.
   - Another member noted that checkpointing is not currently enabled but can be activated if **enable_async_checkpointing=True** in the config, albeit only for `full_finetune_distributed` at this time.
- **Integration of Checkpointing with Full Finetuning**: The feature of **checkpointing** is indicated to be integrated primarily into `full_finetune_distributed` configurations only.
   - This means that even with async checkpointing enabled, its functionality may not be available across all configurations, limiting its use.


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1334303344071675985)** (2 messages): 

> `img2vid tools, ltxv` 


- **Best local img2vid tool discussed**: A user inquired about the best **img2vid** tools for local use currently available.
   - Another member expressed a preference for **ltxv**, suggesting it as a potential top choice.
- **User preference for ltxv**: The preference for **ltxv** was shared as a notable mention for img2vid applications.
   - This indicates growing interest in local tools that provide effective video generation capabilities.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1334275103957057691)** (1 messages): 

> `MLOps Workshop, Feature Store on Databricks, Q&A Session, Data Engineering, Geospatial Analytics` 


- **MLOps Workshop on Databricks is Live!**: Join our founder, **Simba Khadder**, for a hands-on demo in the 'MLOps Workshop: Building a Feature Store on Databricks' on **January 30th at 8 AM PT**.
   - The workshop covers building and deploying **production-grade feature pipelines** on Databricks, so don't miss the opportunity to sign up [here](https://buff.ly/40Ej4Z6)!
- **Real-World Use Cases and Best Practices**: Simba will guide participants on fully utilizing **Databricks** and **Unity Catalog**, discussing the best practices for setting up a feature store.
   - There will be a **Q&A** session towards the end, allowing attendees to engage directly with the topics presented.
- **Free Event for AI/ML Enthusiasts**: This workshop is designed for **Data Engineers**, **Data Scientists**, and **Machine Learning Engineers**, welcoming anyone interested in AI and ML.
   - The event is **free of charge**, making it accessible for anyone looking to enhance their skills in the field.
- **Upcoming Geospatial Analytics Event**: Mark your calendars for **Geospatial Analytics with Databricks** on **January 30, 2025 at 1:00 PM EST**.
   - This is another free opportunity to engage with advanced analytics topics, with registration available on [Eventbrite](https://www.eventbrite.com/e/doi-geospatial-analytics-with-databricks-tickets-1111902653769?aff=erelexpmlt).



**Link mentioned**: <a href="https://buff.ly/40Ej4Z6">MLOps Workshop: Building a Feature Store on Databricks</a>: Join our 1-hr webinar with Featureform&#39;s founder to learn how to empower your data by using Featureform and Databricks!

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/)** (1 messages): 

glitchglitchglitch: what do we need to do to make the bfcl data hf datasets compliant?
  

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

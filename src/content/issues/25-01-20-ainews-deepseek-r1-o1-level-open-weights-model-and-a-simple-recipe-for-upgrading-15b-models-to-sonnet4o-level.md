---
id: d7e2899a-fcf4-4b0b-a5aa-beb34e412da2
title: >-
  DeepSeek R1: o1-level open weights model and a simple recipe for upgrading
  1.5B models to Sonnet/4o level
date: '2025-01-21T07:50:24.815688Z'
original_slug: ainews-deepseek-r1-o1-level-open-weights-model
description: >-
  **DeepSeek** has released the **DeepSeek-R1** suite, including full-size 671B
  MoE models and multiple distilled versions, achieving reasoning performance
  surpassing **OpenAI-o1** and rivaling **GPT-4o** on benchmarks like *AIME
  2024*. The release is notable for its open MIT license, significant cost
  efficiency (up to *50x* cheaper than o1), and a novel reinforcement learning
  training pipeline (GRPO) that outperforms traditional RL and distillation
  methods.
tags:
  - deepseek
  - openai
  - ollama
  - gpt
  - alpaca
  - deepseek-r1
  - deepseek-r1-zero
  - deepseek-r1-distill-qwen-1.5b
  - deepseek-r1-distill-qwen-7b
  - deepseek-v3
  - openai-o1
  - gpt-4o
  - llama-3.1-8b
  - llama-3.3-70b
  - reinforcement-learning
  - model-distillation
  - reasoning
  - open-source-models
  - large-language-models
  - math-benchmarks
  - cost-efficiency
  - multi-expert-models
  - training-pipelines
companies:
  - deepseek
  - openai
  - ollama
  - gpt
  - alpaca
models:
  - deepseek-r1
  - deepseek-r1-zero
  - deepseek-r1-distill-qwen-1.5b
  - deepseek-r1-distill-qwen-7b
  - deepseek-v3
  - openai-o1
  - gpt-4o
  - llama-3.1-8b
  - llama-3.3-70b
topics:
  - reinforcement-learning
  - model-distillation
  - reasoning
  - open-source-models
  - large-language-models
  - math-benchmarks
  - cost-efficiency
  - multi-expert-models
  - training-pipelines
---


<!-- buttondown-editor-mode: plaintext -->**GRPO is all you need.**

> AI News for 1/17/2025-1/20/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **8019** messages) for you. Estimated reading time saved (at 200wpm): **910 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We knew that we'd get an open weights release of DeepSeek at some point, and DeepSeek is already well known for their papers and [V3 was the top open model in the world](https://www.latent.space/p/baseten), but all our AI sources could not take their eyes off the DeepSeek R1 release today.

![image.png](https://assets.buttondown.email/images/82e28b15-aa19-4a86-8947-b5a6930b919d.png?w=960&fit=max)


R1's performance which turned out to be leaps and bounds above [DeepSeek V3 from literally 3 weeks ago](https://buttondown.com/ainews/archive/ainews-deepseek-v3-671b-finegrained-moe-trained/):

![image.png](https://assets.buttondown.email/images/b70431e4-c259-4163-af25-e4f1d14b5b4e.png?w=960&fit=max)
![image.png](https://assets.buttondown.email/images/a5b3346d-a72b-424f-a33c-70918db2b2cb.png?w=960&fit=max)

When we say "R1", it's ambiguous. DeepSeek actually dropped 8 R1 models - 2 "full" models, and 6 distillations on open models:

- from Qwen 2.5: finetuned with 800k samples curated with DeepSeek-R1, in 1.5B, 7B, 14B, and 32B
- from Llama 3.1 8B Base: DeepSeek-R1-Distill-Llama-8B 
- from Llama3.3-70B-Instruct: DeepSeek-R1-Distill-Llama-70B
- and **DeepSeek-R1 and DeepSeek-R1-Zero**, the full-size, 671B MoE models similar to [DeepSeek V3](https://www.latent.space/p/baseten). Surprisingly, [MIT licensed](https://x.com/deepseek_ai/status/1881318138937233664?s=46) rather than custom licenses, including explicit OK for finetuning and distillation

Other notables from the launch:

- **Pricing** (per million tokens): 14 cents input (cache hit), 55 cents input (cache miss), and 219 cents output. This compares to o1 at 750 cents input (cache hit), 1500 cents input (cache miss), 6000 cents output. **That's 27x-50x cheaper than o1.**
- [solves every problem from the o1 blogpost](https://x.com/mrsiipa/status/1881330071874813963). [every one](https://x.com/nrehiew_/status/1881453058556870934?s=46).
- [can run the distilled models on ollama](https://simonwillison.net/2025/Jan/20/deepseek-r1/)
- can write [manim code](https://x.com/christiancooper/status/1881335734256492605) really well


Surprises from [the paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf):

- The process was:
  1. [V3 Base → R1 Zero](https://x.com/casper_hansen_/status/1881404608591085817) (using GRPO - aka reward for correctness and style outcomes - no fancy PRM/MCTS/RMs)
  2. [R1 Zero → R1 Finetuned Cold Start](https://x.com/casper_hansen_/status/1881404611401236745) (distil long CoT samples from R1 Zero)
  3. [R1 Cold Start → R1 Reasoner with RL](https://x.com/casper_hansen_/status/1881404614190506188) (focus on language consistency - to produce readable reasoning)
  4. [R1 Reasoning → R1 Finetuned-Reasoner](https://x.com/casper_hansen_/status/1881404617235509711) (Generate 600k: multi-response sampling and only keep correct samples (using prev rules) and using V3 as a judge: filter out mixed languages, long paragraphs, and code)
  5. [R1 Instruct-Reasoner → R1 Aligned](https://x.com/casper_hansen_/status/1881404619362013294) (Balance reasoning with helpfulness and harmlessness using GRPO)
- [Visualized](https://x.com/SirrahChan/status/1881488738473357753): ![image.png](https://assets.buttondown.email/images/c2e152cb-1ae5-4c2e-88fe-39d6b0fb411b.png?w=960&fit=max)
- Supervised data, Process reward models, and [MCTS](https://x.com/lu_sichu/status/1881348105586855962) did -NOT- work 
![image.png](https://assets.buttondown.email/images/d7de6974-4c88-40dd-94fe-408c9251306c.png?w=960&fit=max)
- but they do use [GRPO from DeepSeekMath](https://arxiv.org/abs/2402.03300) ([challenged by the DPO author](https://x.com/rm_rafailov/status/1881350883252085000)) as "the RL framework to improve model performance in reasoning" where reasoning (like [in-context back-tracking](https://x.com/paul_cal/status/1881324020592963939)) "naturally emerged" after "thousands of RL steps" - [not quite](https://x.com/cto_junior/status/1881319502861967635) the famous o1 scaling plot, but a close cousin. ![image.png](https://assets.buttondown.email/images/53113db9-e27c-4f57-9a0e-3c2ddf68d842.png?w=960&fit=max)
- using ["aha moments"](https://x.com/teortaxesTex/status/1881317131561922640) as pivot tokens, often [mixing languages in a reader unfriendly way](https://x.com/teortaxesTex/status/1881329351125549144)
- R1 [began training less than a month after the o1 announcement](https://x.com/teortaxesTex/status/1881298065967239183)
- R1 distillations were [remarkably effective](https://x.com/nrehiew_/status/1881330794549182853), giving us [this insane quote](https://x.com/reach_vb/status/1881319500089634954): "DeepSeek-R1-Distill-Qwen-**1.5B outperforms GPT-4o and Claude-3.5-Sonnet** on math benchmarks with 28.9% on AIME and 83.9% on MATH.", and this is [without even pushing the distillation to their limits](https://x.com/teortaxesTex/status/1881331287010550119). 
- This is [more effective than just RL-tuning a small model](https://x.com/DimitrisPapail/status/1881341537499619822): "[reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models.](https://x.com/qtnx_/status/1881330757001502991)" aka "total SFT victory"





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

**DeepSeek-R1 Model Developments**

- **DeepSeek-R1 Releases and Updates**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1881318130334814301) announced the release of **DeepSeek-R1**, an open-source reasoning model with performance on par with **OpenAI-o1**. The release includes a technical report and distilled smaller models, empowering the open-source community. [@cwolferesearch](https://twitter.com/cwolferesearch/status/1881362098141446598) highlighted that **reinforcement learning fine-tuning** is less effective compared to **model distillation**, marking the start of the **Alpaca era** for reasoning models.

**Benchmarking and Performance Comparisons**

- **DeepSeek-R1 vs OpenAI-o1**: [@_philschmid](https://twitter.com/_philschmid/status/1881423639741960416) summarized evaluations showing **DeepSeek-R1** achieving **79.8%** on **AIME 2024** compared to **OpenAI-o1's 79.2%**. Additionally, [@ollama](https://twitter.com/ollama/status/1881427522002506009) noted that **R1-Distill-Qwen-7B** surpasses larger proprietary models like **GPT-4o** on reasoning benchmarks.

**Reinforcement Learning in LLM Training**

- **RL-Based Model Training**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1881362098141446598) emphasized that **pure reinforcement learning** can endow **LLMs** with strong reasoning abilities without extensive supervised fine-tuning. [@Philschmid](https://twitter.com/_philschmid/status/1881420703721009192) detailed the five-stage **RL training pipeline** of **DeepSeek-R1**, showcasing significant performance improvements in **math**, **code**, and **reasoning tasks**.

**Open-Source Models and Distillation**

- **Model Distillation and Open-Source Availability**: [@_akhaliq](https://twitter.com/_akhaliq/status/1881386796266946743) announced that **DeepSeek’s distilled models**, such as **R1-Distill-Qwen-7B**, outperform non-reasoning models like **GPT-4o-0513**. [@reach_vb](https://twitter.com/reach_vb/status/1881412831306002897) highlighted the community benefits from **DeepSeek’s open and distilled models**, making advanced reasoning capabilities accessible on **consumer hardware**.

**AI Research Papers and Technical Insights**

- **Insights from Research Papers**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1881211041247359146) shared insights from the **LongProc** benchmark, revealing that out of **17 LCLMs**, **open-weight models** struggle beyond **2K tokens**, while **closed-source models** like **GPT-4o** degrade at **8K tokens**. [@_philschmid](https://twitter.com/_philschmid/status/1881423639741960416) discussed the **DeepSeek-R1 paper’s** findings on how **reinforcement learning** enhances model reasoning without relying on complex reward models.

**Memes/Humor**

- **Humorous Takes on AI and Technology**: [@swyx](https://twitter.com/swyx/status/1881172252781343133) shared a humorous [xkcd](https://xkcd.com/979/) comic, while [@qtnx_](https://twitter.com/qtnx_/status/1881312367667191933) expressed frustration in a lighthearted manner about game launches and prompt engineering.

- **Satirical Comments on AI Hype**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1881427907832340548) humorously commented on overly optimistic AI expectations, emphasizing the perpetual nature of humorous content regardless of technological advancements.

- **Playful Interactions**: [@jmdagdelen](https://twitter.com/jmdagdelen/status/1881441833190150583) responded playfully to AI discussions, adding a touch of humor to technical conversations.

- **Unexpected Humor in Technical Discussions**: [@evan4life](https://twitter.com/evan4life/status/1881371234567890123) shared a funny anecdote about AI model behaviors, blending technical insights with humor.

- **Lighthearted AI Jokes**: [@sama](https://twitter.com/sama/status/1881258443669172470) humorously downplayed AGI development timelines, reflecting the community's playful skepticism.

- **Funny AI-Related Memes**: [@thegregyang](https://twitter.com/TheGregYang/status/1881111771517497616) tweeted a situational meme about workplace scenarios, adding levity to AI-focused discussions.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek-R1 Distilled Models Showcase Exceptional SOTA Performance**

- **[Deepseek just uploaded 6 distilled verions of R1 + R1 "full" now available on their website.](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)** ([Score: 790, Comments: 226](https://reddit.com/r/LocalLLaMA/comments/1i5or1y/deepseek_just_uploaded_6_distilled_verions_of_r1/)): **Deepseek** has released six distilled versions of **R1** models along with the **R1 "full"** model, now accessible on their website.
  - **Deepseek's Strategy and Licensing**: Commenters praise **Deepseek** for releasing finetunes of competitor models and supporting the local LLM community, noting the strategic aspect of this release. The models, including **DeepSeek-R1-Distill-Qwen-32B**, are released under the **MIT License**, allowing commercial use and modifications, which is seen as a significant move in the open-source community.
  - **Model Performance and Availability**: The **DeepSeek-R1-Distill-Qwen-32B** model reportedly outperforms other models like **OpenAI-o1-mini** in benchmarks, achieving state-of-the-art results for dense models. Users are eagerly awaiting the availability of **GGUF** versions for larger models like **32B** and **70B**, with links to these models being shared on platforms like **Hugging Face**.
  - **Community Reactions and Technical Insights**: Users express excitement about the model's capabilities and performance, with some noting the verbosity of the distilled models and the potential for further improvement through reinforcement learning. There is also a discussion about the practical implications of these models in real-world applications, with some users sharing their testing experiences and results.


- **DeepSeek-R1-Distill-Qwen-32B is straight SOTA, delivering more than GPT4o-level LLM for local use without any limits or restrictions!** ([Score: 247, Comments: 85](https://reddit.com/r/LocalLLaMA/comments/1i5s2yd/deepseekr1distillqwen32b_is_straight_sota/)): **DeepSeek-R1-Distill-Qwen-32B** is establishing itself as the state-of-the-art (SOTA) model, surpassing **GPT-4** level LLMs for local use without restrictions. The model's distillation, especially its fusion with **Qwen-32B**, achieves significant benchmark improvements, making it ideal for users with less VRAM and outperforming the **LLama-70B** distill.
  - **Distillation and Benchmarks**: **DeepSeek-R1-Distill-Qwen-32B**'s performance is highlighted by its entrance into the **Pareto frontier** with a score of **36/48** on a benchmark [without quantization](https://oobabooga.github.io/benchmark.html), showcasing its efficiency and competitive edge in local use models.
  - **Model Comparisons and Features**: There is a discussion about the superiority of **LLama 3.1 8B** and **Qwen 2.5 14B** distillations, which reportedly outperform **QWQ** and include "thinking tags," enhancing reasoning capabilities.
  - **Software and Tools**: Recent updates and support for these models are available, including **PR #11310** for distilled versions, and the requirement for the latest **LM Studio 0.3.7** to support **DeepSeek R1**.


- **Deepseek-R1 and Deepseek-R1-zero repo is preparing to launch？** ([Score: 51, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1i5jlsr/deepseekr1_and_deepseekr1zero_repo_is_preparing/)): **DeepSeek-R1** and **DeepSeek-R1-Zero** models are anticipated for release on Hugging Face, as indicated by the provided links. The user expresses eagerness for the launch, hoping it will occur today.
  - **DeepSeek-R1 Zero** is already available for download if users have sufficient storage capacity. The same applies to **DeepSeek-R1**.


**Theme 2. DeepSeek-R1 Models Outprice OpenAI's High-Cost Tokens**

- **Deepseek R1 = $2.19/M tok output vs o1 $60/M tok. Insane** ([Score: 155, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1i5piy1/deepseek_r1_219m_tok_output_vs_o1_60m_tok_insane/)): **Deepseek R1** offers a pricing of **$2.19 per million tokens output**, which is significantly lower compared to **o1's $60 per million tokens**. The post author is interested in real-world applications and particularly in comparisons related to **code generation**.
  - **Deepseek R1 Pricing and Performance**: The discussion highlights that **Deepseek R1** offers a competitive pricing of **$2.19 per million tokens**, significantly lower than **o1's $60 per million tokens**. Users noted that the **R1 model** has shown impressive performance improvements over its previous versions, particularly the **35B and 70B parameter models** which perform comparably or better than **o1-mini**.
  - **Model Transparency and Cost Factors**: There is a lack of transparency from **OpenAI** regarding their model's architecture and token usage, making replication challenging. Some comments suggest that **OpenAI's pricing** might not solely be based on greed, but rather on the costs associated with R&D and operational expenses, with skepticism around **Sam Altman's** claims about their financial losses.
  - **Access and Implementation**: Users inquired about accessing and testing **Deepseek R1**, with references to the [Deepseek API documentation](https://api-docs.deepseek.com/quick_start/pricing) for more information. The **"deepthink"** feature was mentioned as a way to utilize the R1 model, with updates noted on their website and app.


- **Deepseek-R1 officially release** ([Score: 60, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1i5p9dk/deepseekr1_officially_release/)): **DeepSeek-R1**, released under the **MIT License**, offers open-sourced model weights and an API for chain-of-thought outputs, claiming performance parity with OpenAI o1 in tasks like mathematics and coding. The release includes two 660B models and six smaller distilled models, with the 32B and 70B models matching OpenAI o1-mini's capabilities. The API pricing is **1 RMB per million input tokens (cache hit)** and **16 RMB per million output tokens**, with detailed guidelines available in the official documentation.
  - **DeepSeek-R1**'s pricing in USD can be found in the official documentation at [DeepSeek Pricing](https://api-docs.deepseek.com/quick_start/pricing), providing clarity on the cost structure for those interested in comparing it with other models.


- **[DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)** ([Score: 58, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1i5pepa/deepseekr1_paper/)): The **DeepSeek-R1 Paper** introduces an API that emphasizes cost-efficient token usage.
  - **Self-evolution of DeepSeek-R1-Zero**: The self-evolution process showcases how **reinforcement learning (RL)** can autonomously enhance a model's reasoning capabilities. This process is observed without the influence of supervised fine-tuning, allowing the model to naturally develop sophisticated behaviors like reflection and exploration through extended test-time computation.
  - **Emergence of sophisticated behaviors**: As DeepSeek-R1-Zero's test-time computation increases, it spontaneously develops advanced behaviors, such as revisiting and reevaluating previous steps. These behaviors emerge from the model's interaction with the RL environment and significantly improve its efficiency and accuracy in solving complex tasks.
  - **"Aha Moment" phenomenon**: During training, DeepSeek-R1-Zero experiences an "aha moment," where it autonomously learns to allocate more thinking time to problems, enhancing its reasoning abilities. This phenomenon highlights the potential of RL to foster unexpected problem-solving strategies, emphasizing the power of RL to achieve new levels of intelligence in AI systems.


**Theme 3. DeepSeek-R1 Embraces Full MIT License for Models**

- **[o1 performance at ~1/50th the cost.. and Open Source!! WTF let's goo!!](https://www.reddit.com/gallery/1i5pbb3)** ([Score: 668, Comments: 237](https://reddit.com/r/LocalLLaMA/comments/1i5pbb3/o1_performance_at_150th_the_cost_and_open_source/)): **DeepSeek R1** and **R1 Zero** have been released with an open-license, offering **o1 performance** at approximately **1/50th the cost**, and they are open-source.
  - **DeepSeek's Open-Source and Pricing Concerns**: There is significant discussion about DeepSeek's open-source claims, with some users questioning the availability of model details like code and datasets. Concerns about pricing are raised, particularly regarding token costs being double for DeepSeek V3 and comparisons to OpenAI's pricing, with some users noting that high prices may prevent system overload.
  - **Model Performance and Comparisons**: Users highlight the impressive performance of DeepSeek models, noting the increase from 32 billion to 600 billion parameters. Comparisons are made with other models like **Qwen 32B** and **Llama 7-8B**, with some users claiming these models outperform others like 4o and Claude Sonnet.
  - **Censorship and Geopolitical Implications**: There is a robust debate on the influence of political censorship in AI models, with discussions on how Chinese companies like DeepSeek may embed CCP values in their models. Comparisons are drawn with American companies that also apply their own "guardrails," reflecting political and cultural biases.


- **[DeepSeek-R1 and distilled benchmarks color coded](https://www.reddit.com/gallery/1i5q6b9)** ([Score: 288, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1i5q6b9/deepseekr1_and_distilled_benchmarks_color_coded/)): **DeepSeek R1** licensing explicitly allows for **model distillation**, which can be beneficial for creating efficient AI models. The post mentions **distilled benchmarks** that are color-coded, suggesting a visual method for evaluating performance metrics.
  - The **DeepSeek R1** models, particularly the **1.5B** and **7B** versions, are noted for outperforming larger models like **GPT-4o** and **Claude 3.5 Sonnet** on coding benchmarks, raising skepticism and curiosity about their performance in non-coding benchmarks such as **MMLU** and **DROP**. Users express surprise at these results, questioning the generalization of improvements beyond math and coding tasks.
  - **DeepSeek-R1-Distill-Qwen-14B** is highlighted for its efficiency, being on par with **o1-mini** while offering significantly cheaper pricing for input/output tokens. The **32B** and **70B** models further outperform **o1-mini**, with the **32B** model being 43x to 75x cheaper, making them attractive for both local and commercial use.
  - Concerns are raised about the training data for distilled models, which rely heavily on **Supervised Fine-Tuning (SFT)** data without Reinforcement Learning (RL), although some users clarify that the development pipeline does include two RL stages. There is skepticism about the accuracy of the 1.5B model's benchmarks, with some suggesting further testing to validate these claims.


- **[Deepseek R1 / R1 Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1)** ([Score: 349, Comments: 105](https://reddit.com/r/LocalLLaMA/comments/1i5jh1u/deepseek_r1_r1_zero/)): **DeepSeek** has expanded its licensing to commercial use under the **MIT License**. The post mentions **DeepSeek R1** and **R1 Zero**, but no further details are provided.
  - **DeepSeek R1 Zero** is speculated to be a large model with around **600B to 700B parameters**, as discussed by users like **BlueSwordM** and **Few_Painter_5588**. This model size suggests significant resource requirements, with estimates of needing **1.8TB RAM** to host, indicating its potential computational intensity.
  - Discussions around **DeepSeek R1 Zero** also touch on its architecture, with **De-Alf** noting it shares the same architecture as other **R1** models, suggesting a common framework among them. The release on **Hugging Face** is mentioned, with some users expressing confusion over the model's size and role, such as being a "teacher" or "judge" model.
  - The release of **DeepSeek R1 Zero** under the **MIT License** was praised for its openness, with users like **Ambitious_Subject108** appreciating the decision not to restrict it behind an API. The community also noted the release of multiple distillations, providing flexibility for various hardware specifications.


**Theme 4. DeepSeek-R1 Distilled Models Revolutionize Precision Benchmarks**

- **[Epyc 7532/dual MI50](https://www.reddit.com/gallery/1i5bj66)** ([Score: 68, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1i5bj66/epyc_7532dual_mi50/)): An engineer built an **Epyc 7532 server** with **dual MI50 GPUs** purchased for $110 each from eBay, running on **256 GB of Micron 3200 RAM** and housed in a **Thermaltake W200 case**. Despite cooling challenges with the MI50s reaching over 80°C, the setup runs **ollama** and **open webui** on Ubuntu, achieving approximately **5t/s** with **Phi4** performing well and **qwen 32b** being slower.
  - **Cooling Challenges**: **Evening_Ad6637** shared insights on improving cooling efficiency by addressing airflow issues and using aluminum materials, achieving up to **10°C** lower temperatures compared to standard cooling systems. They recommend ensuring direct contact between aluminum components and the GPU heat sink for better heat dissipation.
  - **Hardware Compatibility and Use**: **Psychological_Ear393** discussed the compatibility of **Radeon VII** and **MI50** GPUs with **ROCm**, noting that while both are deprecated, they still function with the latest drivers. They also mentioned that the **W200 case** is notably large, accommodating the setup effectively.
  - **Fan and Airflow Considerations**: **No-Statement-0001** suggested using turbine-style fans to enhance static pressure and improve airflow through the dense fins of server GPUs, as regular fans may struggle with this task.


- **[o1 thought for 12 minutes 35 sec, r1 thought for 5 minutes and 9 seconds. Both got a correct answer. Both in two tries. They are the first two models that have done it correctly.](https://i.redd.it/g4tvkorg56ee1.png)** ([Score: 104, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1i5t1be/o1_thought_for_12_minutes_35_sec_r1_thought_for_5/)): **DeepSeek R1** and **o1 models** achieved correct answers in a complex mathematical problem within **two tries**, with **o1** taking **12 minutes 35 seconds** and **R1** taking **5 minutes 9 seconds**. The problem involved counting elements like wolves and hares, and highlighted a logical error when the count of wolves became negative, stressing the importance of non-negative variables in calculations.
  - **Problem-Solving Insights**: The discussion delves into the reasoning behind the puzzle, emphasizing the importance of logical reasoning in AI models. **Charuru** provides a detailed breakdown of the problem-solving process, identifying key observations like the reduction of total animal count by one per move, the impossibility of odd final totals, and the stable coexistence of at most one species.
  - **Model Performance Variability**: **No_Training9444** and others discuss the variability in model performance, with some models like **Deepseek R1** and **o1-pro** successfully solving the problem, while other models like **gemini-exp-1206** struggled. **StevenSamAI** notes that repeated trials may yield correct answers, indicating variability in model output.
  - **Community Engagement**: The community actively engages with the problem, sharing attempts and outcomes. **Echo9Zulu-** questions the purpose of such riddles in testing AI, while **DeltaSqueezer** and others express interest in solving the puzzle themselves, highlighting the blend of fun and technical challenge these problems present.


- **Deepseek-R1 GGUFs + All distilled 2 to 16bit GGUFs + 2bit MoE GGUFs** ([Score: 101, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1i5s74x/deepseekr1_ggufs_all_distilled_2_to_16bit_ggufs/)): **Deepseek-R1** models have been uploaded in various **quantization formats** including 2 to 16-bit GGUFs, with a **Q2_K_L 200GB quant** specifically for **large R1 MoE** and R1 Zero models. The models are available on [Hugging Face](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-677cf5cfd7df8b7815fc723c) and include **4-bit dynamic quant** versions for higher accuracy, with instructions for running the models using **llama.cpp** provided on the [Unsloth blog](http://unsloth.ai/blog/deepseek-r1).
  - **Dynamic Quantization and Compatibility Issues**: Users discuss the use of **Q4_K_M** for optimal performance and explore alternatives to **bitsandbytes** for dynamic quantization compatible with **llama.cpp**. There are issues with **LM Studio** not supporting the latest **llama.cpp** updates, causing errors when loading models like **R1 Gguf**.
  - **Model Upload Delays and Availability**: The **Qwen 32b gguf** model faced a temporary 404 error during upload, but was subsequently made available on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUFApologies). Other models are still in the process of being uploaded, with the team working overnight to ensure availability.
  - **Community Appreciation and Feedback**: The community expresses gratitude for the ongoing work and rapid updates from the **Unsloth** team, acknowledging their dedication and responsiveness to user feedback and issues.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. DeepSeek-R1 Launches Open-Source Model at Hardware Cost**

- **[It just happened! DeepSeek-R1 is here!](https://x.com/deepseek_ai/status/1881318130334814301)** ([Score: 250, Comments: 103](https://reddit.com/r/OpenAI/comments/1i5pr7q/it_just_happened_deepseekr1_is_here/)): **DeepSeek-R1** is a new model that requires substantial **GPU** resources, suggesting high computational demand. It is described as an **open model**, indicating its availability for public use and potential for community contributions or modifications.
  - **DeepSeek-R1 Hardware Requirements**: While some users initially believed DeepSeek-R1 required high-end hardware, distillated versions can run on a single **RTX 3090** and even lower **VRAM** cards, allowing for more accessible use for those with consumer-grade GPUs.
  - **Open Source vs. Proprietary Models**: There is a discussion on the openness of DeepSeek-R1 compared to proprietary models like **ChatGPT** and **Claude**, emphasizing the ability to run DeepSeek locally, albeit requiring significant hardware investment, which contrasts with the data collection concerns associated with proprietary APIs.
  - **AI Model Development and Expectations**: The simplicity of DeepSeek's training process, involving standard policy optimization with rewards, raises questions about why such effective methods weren't discovered earlier, highlighting the ongoing evolution and expectations in the AI field for models to improve reasoning and inference capabilities.


**Theme 2. AI Autonomy in Job Applications with Browser-Use Tool**

- **[AI agent applying for jobs on its own](https://v.redd.it/gvchr63e96ee1)** ([Score: 200, Comments: 46](https://reddit.com/r/OpenAI/comments/1i5tk3n/ai_agent_applying_for_jobs_on_its_own/)): The post discusses an **AI agent** that autonomously applies for jobs using **GitHub**. Specific details about the implementation or effectiveness of this AI agent are not provided in the text, as the post body is empty and relies on a video for further information.
  - **Automation and Externalities**: Users express concern over the implications of automating job applications, with comments highlighting the increased volume of applications and the resulting need for employers to use automation for screening. The discussion emphasizes that while AI can apply to thousands of jobs, it may lead to more spam and inefficiencies in the job market.
  - **AI Application Effectiveness**: A journalist's test of AI job application services revealed that applying to thousands of jobs can yield interviews, though with a low success rate per application. The conversation suggests that while AI can scale job applications, it may produce inaccuracies, such as fabricating qualifications, and the overall effectiveness is questioned.
  - **Potential Countermeasures**: Users predict that as AI agents apply for jobs, recruiters may develop strategies like honeypotting to identify AI-generated applications. There is also speculation about AI agents eventually managing remote work, raising ethical and practical questions about AI's role in the job market.


**Theme 3. Critique of OpenAI's Marketing and AGI Promises**

- **[He himself built the hype but it got out of control](https://i.redd.it/6o428kog24ee1.png)** ([Score: 1243, Comments: 135](https://reddit.com/r/OpenAI/comments/1i5luka/he_himself_built_the_hype_but_it_got_out_of/)): **Sam Altman** addresses the excessive **hype** surrounding **OpenAI** on Twitter, clarifying that **artificial general intelligence (AGI)** will not be deployed next month as it has not been built yet. He advises followers to **temper their expectations**, despite exciting developments, as per his tweet dated **January 20, 2025**, with **26.9K views**.
  - Discussions highlight skepticism about **Sam Altman's** statements, with users expressing frustration over perceived inconsistencies and hype management, particularly regarding the timeline for **AGI**. Some users interpret his messaging as strategic, possibly to manage expectations and regulatory scrutiny.
  - Users debate the **singularity community's** response, often mocking their optimistic timelines for AGI, and suggesting that forums like **r/singularity** and **r/openai** are increasingly indistinguishable due to shared unrealistic expectations.
  - Several comments reflect on **Altman's** past statements and the hype surrounding **OpenAI**, with some suggesting that his recent tweets aim to temper market expectations and prevent overvaluation based on speculative AGI timelines.


- **OpenAI’s Marketing Circus: Stop Falling for Their Sci-Fi Hype** ([Score: 357, Comments: 214](https://reddit.com/r/OpenAI/comments/1i5a7ss/openais_marketing_circus_stop_falling_for_their/)): OpenAI's marketing tactics are criticized for promoting unrealistic expectations about **AGI** and **PhD-level super-agents**, suggesting these advancements are imminent. The post argues that **LLMs** lack advanced reasoning skills without specialized training and cautions against believing in overhyped promises, emphasizing the need for improved media literacy.
  - Discussions highlight skepticism towards **OpenAI's marketing** tactics, with some users arguing that the company's claims about **AGI** and **PhD-level super-agents** are exaggerated and not reflective of current capabilities. **Sam Altman** is noted for delivering ambitious statements that are met with both cynicism and anticipation.
  - Users debate the capabilities of **LLMs**, with some asserting that current models like **o1** and **o3** are already performing tasks better than average humans, while others argue that these models still lack common sense and reliability. The conversation touches on the **reasoning abilities** of LLMs, with comparisons to toddlers and discussions on their impressive, yet limited, problem-solving skills.
  - The community expresses a divide between the perceived hype and the actual utility of AI models, with some users advocating for a more realistic understanding of AI capabilities. There is a call for skepticism towards **media representations** of AI advancements, emphasizing the need for practical experience and direct usage of the models to assess their real-world applicability.


**Theme 4. Criticism of Perplexity AI's Reliability and Bias Concerns**

- **[People REALLY need to stop using Perplexity AI](https://i.redd.it/vp3tn1u0g5ee1.png)** ([Score: 220, Comments: 137](https://reddit.com/r/OpenAI/comments/1i5py7o/people_really_need_to_stop_using_perplexity_ai/)): **Perplexity AI's CEO, Aravind Srinivas,** proposes developing an alternative to Wikipedia due to perceived bias, encouraging collaboration through Perplexity APIs. His tweet from **January 14, 2025**, has attracted significant attention with **820.7K views, 593 likes,** and **315 retweets.**
  - Discussions highlight the **bias in Wikipedia**, particularly concerning contentious topics like the **Israel/Palestine conflict**. Commenters argue that Wikipedia's crowd-sourced nature leads to activist-driven content, with some suggesting that a corporate alternative could be more biased due to profit motives.
  - Many commenters express skepticism about **Perplexity AI's intentions**, suggesting the company's proposal might cater to right-wing perspectives under the guise of being "uncensored." Concerns are raised about the feasibility of creating a truly unbiased platform, given that all information sources inherently carry some bias.
  - The idea of **alternative information sources** is debated, with some supporting the diversification of sources to avoid single-narrative dominance, while others worry about the potential for increased bias and misinformation. The conversation reflects broader concerns about the role of technology and AI in shaping public discourse and knowledge repositories.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. Open-Source LLM Rivalries**

- [**DeepSeek R1 Roars Past OpenAI’s o1**](https://huggingface.co/deepseek-ai/DeepSeek-R1): This 671B-parameter model matches o1’s reasoning benchmarks at 4% of the cost and arrives under an MIT license for free commercial use. Its distilled variants (1.5B to 70B) also impress math enthusiasts with high scores on MATH-500 and AIME.  
- [**Kimi k1.5 Slams GPT-4o in a 128k-Token Duel**](https://x.com/Kimi_ai_/status/1881332472748851259): The new “k1.5” orchestrates multi-modal tasks, reportedly outperforming GPT-4o and Claude Sonnet 3.5 by up to +550% in code and math. Users point to its chain-of-thought synergy as it breezes past difficult benchmarks.  
- [**Liquid LFM-7B Dares to Defy Transformers**](https://www.liquid.ai/lfm-7b): Liquid AI touts LFM-7B, a non-transformer design with superior throughput on 7B scale. It boldly claims best-in-class English, Arabic, and Japanese support under a license-based model distribution.  

**Theme 2. Code & Agentic Tools**

- [**Windsurf Wave 2 Surfs with Cascade & Autogenerated Memories**](https://codeium.com/blog/windsurf-wave-2): The new Windurf editor integrates robust web search, doc search, and performance boosts for broader coding teams. Users praise its single global chat approach, though some bemoan sluggish performance under large-file contexts.  
- [**Cursor Stumbles in Sluggish Showdown**](https://forum.cursor.com/): Devs complain about 3-minute delays, code deletion mishaps, and “flow actions” slowing them down. Many threaten to jump ship for faster AI editors like Windsurf or Gemini.  
- [**Aider 0.72.0 Scores with DeepSeek R1**](https://aider.chat/docs/leaderboards/): Aider’s latest release welcomes “--model r1” to unify code generation across Kotlin and Docker enhancements. Users love that Aider wrote “52% of the new code,” proving it’s a double-edged coding partner.  

**Theme 3. RL & Reasoning Power-Ups**

- [**GRPO Simplifies PPO for DeepSeek**](https://x.com/natolambert/status/1881380809153847711): “*Group Relative Policy Optimization (GRPO) is just PPO minus the value function,*” claims Nathan Lambert. By relying on Monte Carlo advantage, DeepSeek R1 emerges with advanced math and code solutions.  
- [**Google’s Mind Evolution Outsmarts Sequential Revision**](https://x.com/_akhaliq/status/1881182840857178146): It achieves 98% success on planning benchmarks with Gemini 1.5 Pro by systematically refining solutions. Observers see it as a new apex for solver-free performance.  
- [**rStar-Math Gambles on MCTS**](https://arxiv.org/abs/2501.04519): It trains small LLMs to surpass big models on tricky math tasks without distilling from GPT-4. The paper shows that token-level Monte Carlo Tree Search can transform modest-scale LLMs into powerhouse reasoners.  

**Theme 4. HPC & Hardware High Jinks**

- [**M2 Ultras Tag-Team DeepSeek 671B**](https://x.com/seo_leaders/status/1881462202831614085): One dev claims near real-time speeds using two M2 Ultras at 3-bit quantization. Enthusiasts debate if the hardware cost justifies the bragging rights for local monstrous LLM runs.  
- [**GPU vs CPU Smackdown**](https://discord.com/): Some argue GPU’s parallelization demolishes CPU for big arrays, though data transfer can bottleneck returns. Others say for small tasks, CPU can be just as quick without the overhead.  
- [**KV Cache Quantization Boosts LM Studio**](https://lmstudio.ai/blog/lmstudio-v0.3.7): Llama.cpp engine v1.9.2 brings memory-friendly inference with 3-bit to 4-bit quantization. Speed freaks applaud the throughput gains on consumer-grade hardware.  

**Theme 5. Partnerships & Policy Kerfuffles**

- [**Microsoft’s $13B OpenAI Bet Spooks the FTC**](https://slashdot.org/story/25/01/17/1958200/microsoft-openai-partnership-raises-antitrust-concerns-ftc-says): Regulators worry about “locked-in” AI partnerships and fear startup competition may suffer. Lina Khan warns that dominating cloud plus AI resources spells trouble for newer contenders.  
- [**FrontierMath Funding Cloaked in NDA**](https://lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=veedfswdCYKZEhptz): It emerges that OpenAI quietly bankrolled the math dataset, leaving many contributors clueless. Critics slam the hush-hush arrangement for hindering transparency.  
- [**TikTok Merger Talk Tangles with Perplexity**](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA): Perplexity upset pro subscribers and then pivoted with big expansions—rumor says it even eyed merging with TikTok. Skeptics question if any synergy exists beyond a flashy headline.  

---

# PART 1: High level Discord summaries




## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 2 & Cascade Upgrades**: The **Windsurf Wave 2** release introduced **Cascade** web and docs search, **autogenerated memories**, and performance enhancements, as noted in [the official blog](https://codeium.com/blog/windsurf-wave-2).
   - Users cited smoother operation in **Cascade**, referencing [status.codeium.com](https://status.codeium.com) and pointing to better reliability for broader teams.
- **Deepseek R1 Rocks 671B Parameters**: The new **Deepseek R1** model boasts **671 billion** parameters, reportedly surpassing other offerings, with [@TheXeophon](https://x.com/TheXeophon/status/1881443117787984265) highlighting its strong test scores.
   - Community members debated integrating it into **Windsurf**, wanting to see further evaluation and clarity around data usage.
- **Performance & Error Woes in Windsurf**: Many users reported **incomplete envelope** errors, slow typing, and lag after version **1.2.1**, particularly with large files.
   - They pointed out frustrations with **flow actions** and **cascading edits**, saying these issues heavily reduced productivity.
- **API Keys & Pro Plan Gripes**: Developers voiced concerns about **Windsurf’s** stance on personal API keys, limiting usage for chat functions and advanced integrations.
   - Some Pro plan subscribers felt shortchanged, comparing Windsurf to other IDEs that freely allow user-owned APIs.
- **Cascade History & Long Chat Issues**: A single global list of **Cascade** chats caused confusion for users seeking project-specific organization.
   - They also complained that extended sessions in **Windsurf** become sluggish, forcing frequent resets and repeated context explanations.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Overhauls Model, Ruffles Pro Feathers**: Following a switch to an **in-house** model, users criticized **Perplexity** for weaker outputs and canceled **Pro** subscriptions, citing a lack of dynamic responses ([Perplexity Status](https://status.perplexity.com)).
   - Others demanded swift fixes and more transparency, referencing the platform's valuation and urging timely improvements.
- **Ithy & Co. Challenge Perplexity’s Reign**: A wave of new AI tools, including **Ithy** and open-source projects like [Perplexica](https://github.com/ItzCrazyKns/Perplexica), gained traction among developers seeking alternatives.
   - Community members said these tools offer broader features, with some predicting that open-source platforms could soon rival closed solutions.
- **DeepSeek-R1 Gears Up in Perplexity**: Perplexity announced plans to integrate **DeepSeek-R1** for advanced reasoning tasks, noting [a tweet from Aravind Srinivas](https://x.com/AravSrinivas/status/1881458694266953934).
   - Users anticipate restored functionality and sharper context handling, hoping for improved synergy with the search interface.
- **Perplexity Pounces on Read.cv**: Perplexity **acquired** Read.cv, aiming to boost its AI-driven insights for professional networking ([details here](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA)).
   - Participants expect stronger user profiles and data-driven matching, fueling speculation about future expansions in the platform’s suite.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 Shines on Benchmarks**: DeepSeek R1 scored **57%** on the aider polyglot benchmark, placing just behind O1’s **62%**, as shown in [this tweet](https://x.com/paulgauthier/status/1881428345973608901).
   - Its open-source approach at [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) drew interest for potential Cursor integration, with some users referencing [DeepSeek’s reasoning model docs](https://api-docs.deepseek.com/guides/reasoning_model) for advanced workflows.
- **Cursor’s Sluggish Woes Spark Debate**: Multiple developers reported **3-minute delays** and slow agent responses in real-world usage, fueling frustration with Cursor’s performance.
   - Some threatened to switch to faster AI editors like **Windsurf** or **Gemini**, while a [Notion entry](https://half-single-ecd.notion.site/Experimental-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4) circulated for fresh prompting ideas.
- **Agent Functionality Face-Off**: Community members highlighted **Cursor**’s hiccups with large files and code deletions, contrasting it with GitHub Copilot and [Cline in a 240k token battle](https://www.youtube.com/watch?v=AtuB7p-JU8Y).
   - Some insisted on better documentation, while others cited a [tweet from Moritz Kremb](https://x.com/moritzkremb/status/1880628661931634700) showcasing single-command best practices.
- **Community Pushes for Cursor Updates**: Calls to include **DeepSeek R1** and other advanced models surfaced to address performance complaints.
   - Developers looked to the [Cursor Forum](https://forum.cursor.com/) for upcoming patches and direct lines of feedback on new releases.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek's Distillation Delivers**: The **DeepSeek-R1** model garnered attention for its robust **distillation results**, showcased in [DeepSeek-R1 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1), with hints of expanded **reasoning** capabilities using **RL** approaches.
   - Contributors brainstormed synergy between **Qwen** and open-source fine-tuning endeavors, suggesting future optimizations for **complex tasks**.
- **Liquid AI's Licenses & LFM-7B**: Liquid AI introduced the **LFM-7B** with a **recurrent** design, touting superior throughput at 7B scale in [their official link](https://www.liquid.ai/lfm-7b).
   - They revealed a **license-based distribution** model and highlighted **English**, **Arabic**, and **Japanese** support for local and budget-limited deployments.
- **Sparsity Speeds & MOEs vs Dense**: Participants compared **MOEs** to **dense models** using a geometric mean trick to match parameter sizes, eyeing a **3-4x** latency advantage.
   - They referenced [NVIDIA's structured sparsity blog](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines) to underscore **2:1** GPU efficiency, albeit with similar memory demands.
- **Google's Mind Evolution Mastery**: Google showcased **Mind Evolution** as outperforming **Best-of-N** and **Sequential Revision**, achieving **98%** success on planning benchmarks with **Gemini 1.5 Pro**.
   - A shared [tweet example](https://x.com/_akhaliq/status/1881182840857178146) highlighted **solver-free** performance gains compared to older inference strategies.
- **CNN Collab for Climate Yields**: A project titled **'Developing a Convolutional Neural Network to Evaluate the Impact of Climate Change on Global Agricultural Yields'** seeks experts in ML and climate science by **January 25**.
   - Prospective collaborators can DM for details on constructing an integrated **CNN** framework to analyze **geospatial data** and **yield factors**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek Revelations & Quantization Quips**: Unsloth announced that all **DeepSeek R1** models, including GGUF and quantized versions, are now on [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1), offering Llama and Qwen distills with improved accessibility.
   - Community members praised **dynamic 4-bit** approaches, referencing a post by **@ggerganov**, highlighting less VRAM use without sacrificing accuracy.
- **Fine-Tuning Feats with Qwen and Phi**: Community members tested **Qwen** and **Phi-4** with various training parameters, noticing underfitting issues on **Phi-4** possibly linked to heavier instruction tuning.
   - They also explored using **Alpaca format** on Qwen2.5, pointing to the [Unsloth documentation](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama) for chat template solutions.
- **Chatterbox Chats & Synthetic Sets**: The new **Chatterbox** dataset builder introduced multi-turn management with features like token counting and Docker-compose, shared in a [GitHub repo](https://github.com/invisietch/Chatterbox).
   - Developers proposed generating **synthetic datasets** in bulk using webworkers or a CLI, aiming for improved multi-turn conversation flows.
- **Sky-T1 Takes Off**: The **Sky-T1-32B** model from the NovaSky team at UC Berkeley scored highly in coding and math, trained on 17K data from Qwen2.5-32B-Instruct in 19 hours on 8 **H100** GPUs.
   - Enthusiasts praised its speed under **DeepSpeed Zero-3 Offload**, indicating it nearly matches **o1-preview** performance.
- **Cohere For AI LLM Research Cohort Calls**: The **Cohere For AI** initiative will run an **LLM Research Cohort** focusing on multilingual long-context tasks, kicking off with a call on January 10th.
   - Participants will practice advanced NLP strategies, referencing a [tweet from @cataluna84](https://x.com/cataluna84/status/1877689686639992872) about combining large-scale teacher models with smaller student models.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV7 Rides High with 'Goose'**: The **RWKV7** release, affectionately dubbed 'Goose,' sparked enthusiasm in the community, with [BlinkDL](https://x.com/BlinkDL_AI/status/1876849157920452781) showcasing strong generative capabilities beyond older models. It notably integrates channel-wise decay and learning-rate tweaks, resulting in solid performance according to user tests.
   - Members compared **RWKV7** to Gated DeltaNet, highlighting new design features that keep this **gen7 RNN** ahead of prior iterations. They also debated memory decay strategies and layering to further sharpen **RWKV7**'s edge.
- **DeepSeek R1 Takes On AIME and MATH-500**: The newly introduced **DeepSeek R1** model outperforms **GPT-4o** and **Claude Sonnet 3.5** in tasks like [AIME](https://x.com/kimi_ai_/status/1881332472748851259) and **MATH-500**, demonstrating coping with extended contexts up to *128k tokens*. Community comparisons suggest improved 'cold start' performance, attributed to robust training strategies.
   - Discussions touched on tackling gradient spikes using strategies from [SPAM: Spike-Aware Adam](https://arxiv.org/abs/2501.06842), hinting that **DeepSeek R1** effectively avoids permanent damage. Users viewed these improvements as promising, while some voiced doubts about fully relying on 'R1 Zero' results without more replication.
- **Qwen2.5 Stumbles Despite Official Scores**: Many tested **Qwen2.5** on gsm8k and observed only ~60% accuracy, diverging from the [official blog’s](https://qwenlm.github.io/blog/qwen2.5-llm/) claim of 73% for the instruct variant. Confusion arose around parsing differences and few-shot formatting details.
   - Some suggested incorporating the same question/answer format used by [QwenLM/Qwen](https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py) plus a “step by step” style to realign results. They reported minor score gains to **66%**, underlining how prompting tactics can sway final outcomes.
- **MoE Hype and Hesitations**: The community praised **Mixture of Experts** models for their efficiency, with references like [Hugging Face’s MoE blog](https://huggingface.co/blog/moe) spurring adoption. Some expressed caution around training stability, underscoring the complexities of sharding and gating strategies.
   - Debates centered on whether **MoE** offers enough practical advantage without advanced tuning to handle potential training volatility. Supporters view it as a promising avenue, while others stressed that sustained experimentation is key.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek’s Daring Drive**: DeepSeek-R1 soared beyond expectations, scoring near-**OpenAI-o1** performance under an **MIT license**, with extra detail in [DeepSeek-R1 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1).
   - Skeptics questioned the **R1 Zero** findings, but others praised **Group Relative Policy Optimization (GRPO)** as a cleaner **PPO** alternative, referencing [GRPO clarifications](https://x.com/natolambert/status/1881380809153847711).
- **Kimi’s Kinetic Kick in RL**: The **Kimi 1.5** paper highlights new RL methods like reward shaping and advanced infrastructure, shared in [Kimi-k1.5 on GitHub](https://github.com/MoonshotAI/Kimi-k1.5).
   - Enthusiasts predict these techniques will bolster synergy between **reinforcement learning** frameworks and chain-of-thought reasoning, signifying a leap forward for agentic models.
- **Molmo’s Multimodal Might**: **Molmo AI** gained traction as a robust VLM, claiming superior performance on detection and text tasks, showcased at [molmo.org](https://molmo.org/).
   - Although some misclassifications surfaced, many see its cross-domain flexibility as a serious contender against models like **GPT-4V**.
- **Cursor Clobbers Devin in Coding Duel**: Teams quickly dropped **Devin** for **Cursor**, citing underwhelming code completions, amid rumors Devin tapped **gpt-4o** for coding tasks instead of stronger alternatives like **Claude**.
   - The shift sparked debates on whether AI groups systematically overestimate emergent agent solutions, echoing points from [Tyler Cowen’s interview](https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5).
- **SOP-Agents Steal the Show**: The [SOP-Agents framework](https://arxiv.org/abs/2501.09316) proposes **Standard Operational Procedures** for large language models, refining multistep planning.
   - Developers anticipate blending it with **Chain of Thought** and RL to enhance the clarity of high-level decision graphs.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.72.0 Achieves New Heights**: The fresh Aider v0.72.0 release brings **DeepSeek R1** support with shortcuts `--model r1` and Kotlin syntax integration, alongside file-writing enhancements using `--line-endings`.
   - Community members cited multiple bugfixes (including **permissions issues** in Docker images) and noted that **Aider** wrote 52% of the new code.
- **DeepSeek R1 Sparks Mixed Reactions**: Some users praised [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) for cheaper alternatives to OpenAI's o1, hitting **57%** on [Aider coding benchmarks](https://aider.chat/docs/leaderboards/).
   - Others reported subpar outcomes in basic tasks, suggesting pairing it with more reliable editing models for improved consistency.
- **Kimi k1.5 KOs GPT-4o**: The new **Kimi k1.5** model reportedly outperforms GPT-4o and Claude Sonnet 3.5 in multi-modal benchmarks, with context scaling up to **128k tokens**.
   - Users highlighted especially strong results on MATH-500 and AIME, fueling optimism for refined reasoning capabilities.
- **AI Data Privacy Draws Concern**: Participants referenced [Fireworks AI Docs](https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security) while describing corporate transparency differences in data usage.
   - They questioned which providers handle user data responsibly, pointing to unclear policies among larger AI vendors.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt.new Banishes White Screens**: After the recent [Tweet from bolt.new](https://x.com/boltdotnew/status/1881442318110347291), **Bolt.new** addresses the notorious white screen and ensures precise template selection from the first prompt.
   - Eager testers report a smoother flow, noting a direct fix to previous frustrations and guaranteeing a more efficient start.
- **Error Loops Gobble Tokens**: Users faced continuous loops leading to severe token consumption—one developer burned through **30 million tokens**—particularly in scenarios involving user permissions.
   - They concluded a complete reset was the only path, with community members urging more robust debugging for **complex functionalities**.
- **RLS Tangles in Supabase**: Developers wrestled with recurring **RLS violations** while implementing booking features in **Supabase**, spurring repeated policy failures.
   - One user recommended referencing [Supabase Docs](https://supabase.com/docs/guides/functions/examples/stripe-webhooks) for sample policies, reducing repeated misconfigurations.
- **Stripe or PayPal? Payment Talk**: Community members debated **Stripe** versus simpler alternatives like **PayPal** for car detailing payments, especially for less technical users.
   - Some pointed to [Supabase's guide on Stripe Webhooks](https://supabase.com/docs/guides/functions/examples/stripe-webhooks), while others recommended WordPress-based solutions for a quicker setup.
- **Pro Plan Eases Token Constraints**: Curious newcomers asked about token usage under the **Pro plan**, discovering the daily limit disappears and usage depends heavily on user skill and optional features like diffs.
   - This approach reassures more advanced developers they can push Bolt without worrying about daily caps or unexpected token exhaustion.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.7 & DeepSeek R1: The Tag-Team Triumph**: The new **LM Studio 0.3.7** includes support for the advanced **DeepSeek R1** model and integrates llama.cpp engine v1.9.2, as outlined in [LM Studio's update](https://lmstudio.ai/blog/lmstudio-v0.3.7).
   - Community members praised the open source approach, referencing the [DeepSeek_R1.pdf](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) for its robust reasoning capabilities with <think> tags.
- **KV Cache Quantization Fuels Efficiency**: The **KV Cache quantization** feature for **llama.cpp** (v1.9.0+) aims to enhance performance by reducing memory usage, as seen in [LM Studio 0.3.7](https://lmstudio.ai/blog/lmstudio-v0.3.7).
   - Users reported faster throughput in large language models, noting that **3-bit** quantization often hits an optimal balance of speed and accuracy.
- **File Attachments Stay Local in LM Studio**: Users questioned whether uploading files in **LM Studio** would send data elsewhere, and were assured the content stays on their machine for local context retrieval.
   - They tested multi-file uploads for domain-specific tasks, confirming offline-only usage without compromising data control.
- **GPUs Under Scrutiny: 4090 vs. Budget Boards**: Membership discussions weighed a $200 GPU against high-end boards like the **4090**, referencing [tech specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889) for large-scale AI tasks.
   - Most agreed bigger memory is a game-changer for massive models, delivering improved throughput for data-driven workloads.
- **Distributed Inference with M2 Ultras: Speed or Splurge?**: An [Andrew C tweet](https://x.com/seo_leaders/status/1881462202831614085) showcased **DeepSeek R1** 671B running on two M2 Ultras, leveraging 3-bit quantization for near real-time speeds.
   - However, participants remained cautious about hardware costs, citing bandwidth constraints and the risk of diminishing returns.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 Distills and Dominates**: The [DeepSeek R1 release](https://x.com/deepseek_ai/status/1881318138937233664) arrived under an **MIT license**, matching **OpenAI o1** performance in math, code, and reasoning tasks.
   - A [distilled variant](https://x.com/reach_vb/status/1881315419086291213) outran **GPT-4o** in AIME and MATH benchmarks, sparking excitement about expanded open-source offerings.
- **OpenAI’s Operator Surfaces in Leaked Docs**: Recent leaks exposed **OpenAI**’s new **Operator** (or Computer Use Agent) project, fueling speculation of an imminent launch.
   - Observers compared it against **Claude 3.5**, referencing details from the [Operator system leak](https://x.com/kimmonismus/status/1881287794544550018).
- **Liquid Foundation Model LFM-7B Sets Sail**: The [LFM-7B model](https://www.liquid.ai/lfm-7b) from **Liquid AI** claims top-tier multilingual capabilities with a non-transformer design.
   - Engineers applauded its low memory footprint for enterprise use, contrasting it with large transformer-based approaches.
- **DeepSeek v3 & SGLang Fuel Mission Critical Inference**: A [Latent.Space podcast](https://www.latent.space/p/baseten) spotlighted **DeepSeek v3** and **SGLang** for advanced workflow requirements in “Mission Critical Inference.”
   - Guests discussed strategies for scaling beyond a single GPU and teased further **DeepSeek** improvements, rousing interest among performance-focused developers.
- **Kimi k1.5 Surprises with O1-Level Performance**: The [Kimi k1.5 model](https://x.com/Kimi_ai_/status/1881332472748851259) reached **o1-level** benchmarks, outperforming **GPT-4o** and **Claude 3.5** in math and code tasks.
   - Reported **+550%** gains on LiveCodeBench spurred debate on how smaller architectures are closing the gap with bigger contenders.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek R1 Takes On OpenAI's o1**: DeepSeek introduced its **R1** model on [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1) with performance that compares well to **OpenAI's o1**, priced at **$0.55/M tokens** (4% of the cost).
   - Community members praised the model’s open-source **MIT license** and strong utility, citing [DeepSeek's tweet](https://x.com/deepseek_ai/status/1881318130334814301) for more details.
- **Censorship-Free Angle Stirs Debate**: **DeepSeek R1** is described as censorship-free on [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737), though some users note it retains filtering components.
   - Others suggest that additional finetuning could broaden its scope, anticipating stronger performance without extra constraints.
- **Llama Endpoints Drop Free Tier**: OpenRouter revealed plans to discontinue **free Llama** endpoints by the month’s end because of changes from **Samba Nova**.
   - A **Standard variant** will replace them at a higher price, surprising many users.
- **OpenAI Model Rate Limits Clarified**: Users confirmed **OpenAI’s paid tiers** carry no daily request cap, but free tiers limit activity to **200 calls per day**.
   - Some overcame these restrictions by attaching their own API keys, reducing usage headaches.
- **Reasoning & Web Search Support in Flux**: Community members asked how to access `reasoning_content` from **DeepSeek R1**, with **OpenRouter** expected to add that feature soon.
   - Others hoped for wider availability of the **Web Search API**, which is currently locked to the chatroom interface.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Photorealistic Flourish with LoRA**: In a discussion about generating lifelike images with **Stable Diffusion** 3.5, participants explored **LoRA** strategies to mitigate a plasticky look, referencing the [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for advanced controls.
   - One user insisted that mixing high-res samples with various **resolutions** yields more realistic outputs, citing [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) for enhanced prompt customization.
- **Cloudy E-commerce Deployments**: A user questioned the feasibility of deploying a text-to-image model on **Google Cloud** for E-commerce, referencing [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) as a starting point for pre-trained solutions.
   - Others weighed whether the **Google Cloud Marketplace** or a **custom Docker** setup would be more efficient, concluding that pre-trained models can greatly reduce setup times.
- **LoRA Resolution Rumble**: Community members debated training **LoRA** solely at 1024×1024, pointing to the [Prompt Syntax docs](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md) for more nuanced control.
   - A group emphasized diverse resolution input so **LoRA** can handle varied image qualities without producing strange artifacts.
- **Background-Editing Tangles**: Users encountered slower performance and flawed background layers, attributing them to **denoising** misconfigurations in **Stable Diffusion** pipelines.
   - They recommended manual fine-tuning via **GIMP** or specialized AI solutions, noting improved results with features from [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md).



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Podcasts & Personality Swaps**: One user introduced a new **GLP-1** themed podcast, exploring host voice changes with a [proposed tool](https://illuminate.google.com/home?pli=1), but current solutions might not properly support it.
   - Another user pointed out random voice role switches can cause confusion, responding that many *podcast generation tools* struggle with stable speaker assignments.
- **Gemini Gains & NotebookLM in Class**: One user described a **Gemini Advanced Deep Research** workflow for generating thorough audio overviews, advising direct sourcing to reduce data loss.
   - Another user debated single vs. multiple notebook usage for an **econ course**, preferring a topic-based approach to maintain consistent organization.
- **Subscriptions & Simple Setups**: Several users compared **Google One AI Premium** with **Google Workspace** for **NotebookLM Plus** access, noting that both provide the needed model features.
   - Users concluded that **Google One** is easier to manage without the complexities of **Workspace** membership.
- **Big Bytes & OCR Ordeals**: One user struggled uploading audio files near **100MB**, suspecting they'd exceed the total **200MB** limit if combined with existing data.
   - Another user highlighted **OCR** problems with non-copyable PDFs, calling for improved **NotebookLM** scanning support.
- **Multi-language Moves & Newcomer Hellos**: Several users expressed interest in **multi-language** podcast support, hoping for official expansions beyond **English** soon.
   - New members introduced themselves, noting *language barriers* and encouraging sharper questions to keep discussions concise.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Bumpy MCP Server Implementation**: Users flagged inconsistent prompt usage across multiple [MCP servers](https://github.com/modelcontextprotocol/servers), leading to confusion about correct specs.
   - Some implementations only fetch resources, ignoring official guidelines, sparking calls for stricter adherence to documentation.
- **Roo Cline Charms with Agentic Twist**: [Roo Cline](https://www.pulsemcp.com/clients) impressed devs by auto-approving commands, giving a nearly hands-free experience with R1 servers.
   - Many praised its helpful VSCode plugin integration as a simpler alternative to bigger clients like **Claude Desktop**.
- **Claude Hits Rate Limit Speed Bumps**: Frequent **Claude** rate limits frustrated testers, restricting context length and message frequency.
   - Some requested better usage tracking in **Claude Desktop**, hoping for clearer thresholds and fewer abrupt halts.
- **Figma MCP Seeks Courageous Coders**: [Figma MCP](https://github.com/nmfisher/figma_mcp) launched as an early prototype, inviting devs to shape its future.
   - *'This is very early/rough, so would appreciate any contributors!'* said one member, asking for new ideas.
- **AI Logic Calculator Sparks Curiosity**: [MCP Logic Calculator](https://github.com/angrysky56/mcp-logic) leverages Prover9/Mace4 in Python to handle logic tasks on Windows systems.
   - Another member suggested pairing it with memory MCP for robust classification, fueling interest in advanced logic workflows.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GPU Gains & CPU Pains**: In a conversation about HPC usage, participants concluded that large arrays often benefit from **GPU** parallelization, though data transfer can cause slowdowns.
   - Some participants described the operation as *trivially parallel*, implying that **CPU** approaches can remain competitive for smaller tasks.
- **Microsoft’s Mega-Bet on OpenAI**: The **$13 billion investment** from Microsoft triggered antitrust warnings, with the **FTC** stressing that cloud dominance might leak into the AI marketplace.
   - FTC Chair **Lina Khan** cautioned that locked-in partnerships could hamper startups from tapping crucial **AI** resources.
- **FrontierMath Funding Fallout**: Community members questioned **OpenAI’s** involvement in **FrontierMath** after discovering a concealed funding arrangement, raising transparency issues.
   - Some claimed that **Epoch** was subject to tough **NDA** terms, leaving many contributors oblivious to OpenAI’s role in financing.
- **Lightning and TPA: Speedy Synthesis**: An integration of **Lightning Attention** and [Tensor Product Attention](https://arxiv.org/abs/2501.06425) yielded about a **3x speed** gain during testing in a toy model.
   - Users credited linearization for enabling big tensor operations in attention, highlighting a major performance leap over prior methods.
- **rStar-Math Surprises with MCTS**: The paper [rStar-Math](https://arxiv.org/abs/2501.04519) presented how small LLMs can surpass bigger models through **Monte Carlo Tree Search** for advanced math tasks.
   - Its authors advocated minimal reliance on human data, detailing a method that uses three distinct training strategies to boost problem-solving.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Konkani Collaboration Gains Steam**: A user aims to build a model for **Konkani** with potential university endorsement, hoping to advance cross-lingual NLP.
   - They noted industry partnerships are essential for expansion and practical adoption of the project.
- **Command-R Conundrum**: Engineers discovered **command-r** references an older model to avoid **breaking changes** for existing users.
   - They proposed official **aliases** with a 'latest' tag to keep releases consistent while enabling new versions on demand.
- **Cohere’s Math Mix-Ups**: Users saw **Cohere** incorrectly compute 18 months as 27 weeks, forcing them to validate results manually.
   - They highlighted that most **LLMs** share this limitation, suggesting lower temperature or separate calculators as solutions.
- **Code Calls and Tool Tactics**: Developers outlined how **Cohere** can invoke external tools by letting the LLM decide when to use specified components.
   - They noted minimal official mention of **AGI**, but emphasized structured prompts and model-driven execution for code generation workflows.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Spring MOOC Gains Momentum**: One member asked about confirmation for the **MOOC course** starting this **January**, highlighting expected **LLM Agents** coverage.
   - They also referenced the **mailing list** starting next week, suggesting more **course timeline** details will be shared soon.
- **Mailing List Kicks Off Soon**: Community members confirmed the **spring course mailing list** will launch next week, addressing open questions about official registration.
   - They anticipate further **course timeline** updates once the list goes live, advising prospective participants to watch for the announcement.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Document to Podcast blueprint on the mic**: A dedicated team introduced the **Document to Podcast blueprint**, a flexible approach for turning textual content into audio using open source solutions.
   - They announced a live session where participants can ask questions, share feedback, and explore how to incorporate this blueprint into their own projects.
- **Blueprints supercharge open source synergy**: Attendees were urged to join the event and connect with fellow open source enthusiasts, promising new collaboration on future projects.
   - They emphasized hitting an 'Interested' button to join the community conversation, fueling new possibilities for deeper open source exchange.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1329913370354385019)** (1 messages): 

> `Windsurf Wave 2 features, Cascade web and doc search, Cascade autogenerated memories, Performance improvements, Status updates` 


- **Windsurf Wave 2 Launches with Major Features**: Windsurf Wave 2 introduces new features like **Cascade** which can now search the web and documentation via automatic detection or user commands.
   - *Cascade* also retains context across conversations through autogenerated memories, enhancing user experience and interaction.
- **Improvements to Cascade and Performance**: The update addresses several **Dev Container issues** while enhancing the overall performance of **Cascade**.
   - These improvements aim to deliver a smoother experience for users interacting with the bot.
- **Cascade Web and Docs Search functionalities**: Users can now trigger web searches automatically, via **URL** input, or using `@web` and `@docs` commands with **Cascade**.
   - These new functionalities allow retrieval of information from various documentation sites and public resources to improve assistance.
- **Windsurf System Status Updated**: The current status of **Windsurf/Codeium** is operational, with no major incidents reported recently, affirming system reliability.
   - Users are encouraged to check the status at [status.codeium.com](https://status.codeium.com) for real-time updates.
- **Stay Updated with Wave 2 Resources**: To explore more about Windurf Wave 2, users can read the complete [announcement on the blog](https://codeium.com/blog/windsurf-wave-2) and view the associated video on X.
   - Further details can be found in the [changelog](https://www.codeium.com/changelog) highlighting all new features and updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.codeium.com">Codeium Status</a>: no description found</li><li><a href="https://codeium.com/blog/windsurf-wave-2">Windsurf Wave 2</a>: Introducing Wave 2, our second batch of updates to the Windsurf Editor.</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384">Tweet from Windsurf (@windsurf_ai)</a>: Wave 2 is here. Included in this update: 🌐Web Search🧠Autogenerated Memories💼Enterprise Ready... and many more!</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1329911461530566786)** (226 messages🔥🔥): 

> `Windsurf Error Messages, Deepseek R1 Release, Codeium Features, User Support Issues, API Key Usage in Windsurf` 


- **Frequent Errors in Windsurf**: Users have reported persistent errors in Windsurf, particularly the message 'Error Protocol error: incomplete envelope: unexpected EOF', leading to frustration in functionality.
   - Others have faced issues with the application not responding to user actions and experiencing difficulties when submitting tokens during registration.
- **Deepseek R1 Surpasses Expectations**: The recently released Deepseek R1 has created buzz by reportedly outperforming OpenAI's models with a staggering **671 billion parameters** and competitive pricing.
   - Users commented on its potential to be integrated into Windsurf, praising its superior benchmark results over existing models.
- **Codeium Features and Limitations**: A discussion arose regarding the limitations of Codeium in JetBrains, particularly the lack of support for the Supercomplete feature, which is currently exclusive to VS Code and Windsurf.
   - Users with Pro plans expressed concerns about not receiving all promised features and faced challenges when attempting to resolve these issues.
- **User Support Challenges**: Several users sought help regarding login issues, persistent error messages, and functionality problems in Windsurf, emphasizing the need for effective user support.
   - Community members shared troubleshooting steps but also indicated frustration with the lack of feedback from direct support channels.
- **Discussion on API Key Usage in Windsurf**: A conversation emerged about Windsurf's business model, specifically its restriction on using personal API keys for chat functions, causing concern among users seeking flexibility.
   - Users compared this to other IDEs that allow personal API integrations, expressing worry about Windsurf's competitive longevity in the market.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open-vsx.org/">Open VSX Registry</a>: no description found</li><li><a href="https://codeium.com/supercomplete">Supercomplete | Windsurf Editor and Codeium extensions</a>: Codeium Supercomplete is able to predict your next intent, regardless of your cursor position. Whether you want an insertion, deletion, or edit, Supercomplete has you covered.</li><li><a href="https://x.com/abacaj/status/1881339724545286257?t=Ydr1EHyPeUj8nYlGTPN1gQ&s=19">Tweet from anton (@abacaj)</a>: China releasing a MIT license model that is on par with o1 and 30x cheaper was not on my bingo card</li><li><a href="https://x.com/itsPaulAi/status/1881329522949447886?t=4igrKlZqJ-rlvMDJvR8yOw&s=19">Tweet from Paul Couvert (@itsPaulAi)</a>: Woow a fully open source reasoning model on par with OpenAI o1 just released Deepseek R1 even outperforms Claude 3.5 Sonnet and o1-mini in almost all benchmarks.You can already use it for free (see be...</li><li><a href="https://codeium.com/s">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265?t=CWcMfDus2ULxJQS6VnnQRA&s=19">Tweet from Xeophon (@TheXeophon)</a>: I am shocked by R1 on my personal bench. This is the full eval set, it completely crushes the competition and is a whole league on its own, even surpassing o1-preview (which is omitted from the graph ...</li><li><a href="https://x.com/TheXeophon/status/1881442133376454694?t=kcwBO9GpmTX5zzXVtA63gA&s=19">Tweet from Xeophon (@TheXeophon)</a>: holyshitwhatthefuckr1 beats o1-preview on my bench</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/profile>">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/plan">Plan Settings</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://www.reddit.com/r/synology/comments/pq0411/cant_mount_network_drive_in_windows_explorer/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Exafunction">Exafunction</a>: Exafunction has 38 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1329913700831854677)** (577 messages🔥🔥🔥): 

> `Windsurf Performance Issues, Deepseek R1 Discussion, Cascade History Management, User Experience with Long Chats, AI Integration with Development Tools` 


- **Windsurf Performance Issues**: Users reported significant performance degradation in Windsurf after version 1.2.1, with problems including slow typing and lag in handling large files.
   - Several users expressed frustration over features like flow actions and cascading edits, which have become cumbersome, leading to a decline in usability.
- **Deepseek R1 Discussion**: Deepseek R1 has been mentioned as a potentially superior model compared to existing solutions like Claude, with some users eager for its integration into Windsurf.
   - The conversation highlighted the need for thorough evaluation and testing before widespread adoption, as well as concerns regarding privacy and data use.
- **Cascade History Management**: There is ongoing discussion about the lack of workspace-specific Cascade history, with users advocating for features that offer better organization of chat histories per project.
   - A user pointed out the single global list of chats, expressing interest in implementation details and roadmap for future updates.
- **User Experience with Long Chats**: Multiple users noted that long chats lead to a decline in responsiveness and functionality within Windsurf, with advice given to start new conversations to mitigate issues.
   - This has led to frustrations regarding the necessity of repeating context and re-explaining problems to Cascade.
- **AI Integration with Development Tools**: Discussions on the potential for AI tools, like Windsurf, to automate connections with databases and provide proactive integration features were brought up.
   - Users shared ideas about how making AI more contextually aware of their development environments could improve user experience significantly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cursorlist.com>">no title found</a>: no description found</li><li><a href="https://discordapp.com/channels/1027685395649015980/1306163501286293515/1330602494958501979">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=810cc140-6780-4bfb-a2f9-906c5d0fdd64">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://docs.codeium.com/getstarted/overview">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://swiftylaun.ch/">SwiftyLaunch</a>: iOS App Generator that has all you need to ensure a swifty launch on the App Store.</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=84d2a993-d3b9-43c1-a847-3d44cfe5c6ce">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://www.semafor.com/article/01/15/2025/replit-ceo-on-ai-breakthroughs-we-dont-care-about-professional-coders-anymore">Replit CEO on AI breakthroughs: ‘We don’t care about professional coders anymore’ | Semafor</a>: Amjad Masad talks about their new AI developments that will allow anyone to code naturally.</li><li><a href="https://vpn.net/">VPN.net – Hamachi by LogMeIn</a>: no description found</li><li><a href="https://tenor.com/view/it%27s-pretty-massive-michael-kupris-become-the-knight-it%27s-very-big-it%27s">no title found</a>: no description found</li><li><a href="https://tenor.com/view/ninja-fortnite-reaction-ninja-low-taper-fade-gif-1784137995500051652">Ninja Fortnite GIF - Ninja Fortnite Reaction - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/itsPaulAi/status/1881329522949447886?t=4igrKlZqJ-rlvMDJvR8yOw&s=19">Tweet from Paul Couvert (@itsPaulAi)</a>: Woow a fully open source reasoning model on par with OpenAI o1 just released Deepseek R1 even outperforms Claude 3.5 Sonnet and o1-mini in almost all benchmarks.You can already use it for free (see be...</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/contact/enterprise">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://www.codacy.com/">Codacy - Code Quality and Security for Developers</a>: Build clean, secure code efficiently and fearlessly with Codacy Platform. </li><li><a href="https://docs.replit.com/category/quickstarts">Quickstarts | Replit Docs</a>: no description found</li><li><a href="https://docs.codeium.com/getstarte">Welcome to Codeium - Codeium Docs</a>: no description found</li><li><a href="https://tenor.com/view/it%27s-pretty-massive-michael-kupris-become-the-knight-it%27s-very-big-it%27s-so-huge-gif-11424078658491848897">It&#039;S Pretty Massive Michael Kupris GIF - It&#039;s pretty massive Michael Kupris Become The Knight - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://codeium.canny.io/feature-requests/p/supercomplete-repeatedly-suggests-to-reindent-the-entire-file-with-the-same-inde">Supercomplete repeatedly suggests to reindent the entire file with the same indentation that the file already has | Feature Requests | Codeium</a>: Ever since the recent update (v 1.2.1, Jan. 17, 2025), Supercomplete weirdly suggests to indent the entire file using the same indentation that the file</li><li><a href="https://x.com/kimmonismus/status/1881092191277457784?t=ctHXhAKdCjvqpl6kq0jTRQ&s=19">Tweet from Chubby♨️ (@kimmonismus)</a>: Dario Amodei (CEO Anthropic) Interview incoming. Finally an announcement for their new model?Quoting Joanna Stern (@JoannaStern) Calling all Claude users. What&#39;s at the top of your wish list? Send...</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">Auto commit message | Feature Requests | Codeium</a>: Generate Commit Messages from Committed File Context</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1329920524796690484)** (624 messages🔥🔥🔥): 

> `Perplexity's Model Changes, User Feedback and Issues, New AI Tools and Alternatives, DeepSeek-R1 Integration, User Interactions and Community Support` 


- **Perplexity's Model Changes Raise Concerns**: Users have expressed dissatisfaction with recent updates to Perplexity, noting the in-house model's lack of dynamic responses and context understanding after disabling third-party LLMs.
   - Many users are frustrated as they feel they are not getting their money's worth from the Pro subscription, and expect improvements soon.
- **Feedback Highlights User Issues**: Community members highlighted billing issues, slow support responses, and generic outputs from Perplexity, leading to cancellations of subscriptions from dissatisfied users.
   - There are calls for better transparency and quicker fixes to maintain customer trust, especially given the platform's valuation.
- **Emergence of New AI Tools and Alternatives**: Several users discussed alternatives like Ithy and complexity extensions that are seen as potentially better solutions for their needs compared to Perplexity.
   - There is a growing interest in leveraging open-source models and tools for improved results and flexibility in their projects.
- **DeepSeek-R1 Integration Promised**: Insightful discussions shared that Perplexity may soon integrate DeepSeek-R1 to enhance advanced reasoning capabilities within its services.
   - Users are eager for this adjustment, which they believe could restore some functionality and improve experience on the platform.
- **Vibrant User Interactions and Support**: The community remains lively, with users sharing advice on troubleshooting, using different AI tools, and supporting each other in navigating recent changes.
   - Feedback about tech advancements and strategies for integrating coding skills into career development indicate a motivated user base interested in continuous learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ithy.com/article/ayaneo-kun-specs-2025-rxuu0hb1">AYANEO Kun Specifications and Processor Chip in 2025</a>: no description found</li><li><a href="https://x.com/testingcatalog/status/1881399907032191334?s=61">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Perplexity may add DeepSeek R1 to their model offering 👀Quoting Aravind Srinivas (@AravSrinivas) @jaseempaloth Yeah</li><li><a href="https://x.com/AravSrinivas/status/1881458694266953934">Tweet from Aravind Srinivas (@AravSrinivas)</a>: You can try DeepSeek-R1 on http://labs.perplexity.ai. We&#39;ll try to bring it up on core Perplexity in the context of advanced reasoning pro searches pretty soon.</li><li><a href="https://aistudio.google.com/prompts/new_chat">no title found</a>: no description found</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI - ItzCrazyKns/Perplexica</li><li><a href="https://ayaneo.com/product/AYANEO-KUN.html">AYANEO KUN - AYANEO</a>: no description found</li><li><a href="https://www.cnbc.com/2025/01/18/perplexity-ai-makes-a-bid-to-merge-with-tiktok-us.html">Perplexity AI makes a bid to merge with TikTok U.S.</a>: Perplexity AI submitted a bid on Saturday to TikTok parent ByteDance, proposing that Perplexity merge with TikTok U.S., CNBC has learned. 
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1330021479391170620)** (24 messages🔥): 

> `RedNote App, FBI Malware Uninstallation, Gaia Sky Scan Co., Perplexity AI Acquisition, ISO27001 and NIS2 Controls` 


- **RedNote App Booms in the US**: The **RedNote App** has seen significant growth in the US, sparking interest among users and developers alike.
   - Further details on its features and user engagement can be found in a [YouTube video](https://www.youtube.com/embed/qsj299D8oLM).
- **FBI Hacked Computers to Uninstall Malware**: Reports are surfacing that the **FBI** has been actively hacking into computers to remove malware to protect users.
   - This unusual move aims to ensure safety among compromised systems but has raised questions regarding privacy.
- **Gaia Sky Scan Company Updates**: The **Gaia Sky Scan Co.** has released new developments that are making waves in the tech community.
   - Details regarding their latest projects and innovations were shared, indicating their growing influence in the market.
- **Perplexity Acquires Read.cv**: Perplexity has officially **acquired Read.cv**, enhancing its capabilities in the AI landscape.
   - Further insights about this acquisition can be found in the [detailed report](https://www.perplexity.ai/page/perplexity-acquired-read-cv-JvwSvLwpQTuyUb.5mf23VA).
- **Overlapping Controls in ISO27001 and NIS2**: A discussion on **overlapping controls** in **ISO27001** and **NIS2** highlighted important compliance overlaps.
   - Participants expressed interest in strategies to streamline implementations of these controls.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/qsj299D8oLM">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/b1eND15ci5A">YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1329906974552494112)** (3 messages): 

> `CrewAI models, Litellm monkey fix, Unnecessary pings` 


- **CrewAI Models Fail to Resolve Issues**: A user reported that they tried all three of the **CrewAI models** without success in fixing a persistent problem.
   - They noted that the **CrewAI documentation** lacked mention of the issue, and another user experienced the same problem with the **o1 model**.
- **Discovery of a Monkey Fix for Litellm**: The user found a **monkey fix** that successfully removes the stop parameters from **Litellm** before making a call, addressing their issue temporarily.
   - This workaround was shared in response to ongoing frustrations with the existing models.
- **Ping Etiquette Reminder**: A user reminded another member to avoid unnecessary pings, asking how they could assist instead.
   - This exchange highlights ongoing concerns about communication etiquette within the group.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1329903374862258187)** (588 messages🔥🔥🔥): 

> `Cursor Performance Issues, DeepSeek R1, Agent Functionality Comparison, Slow Request Concerns, GitHub Integrations` 


- **Cursor experiences slow requests**: Users are expressing frustration over slow requests, particularly with the agent functionalities, noting instances of 3-minute delays even in previously responsive environments.
   - Customer dissatisfaction has been attributed to the perceived lack of value being provided in terms of speed and performance compared to competitors like Windsurf and Gemini.
- **DeepSeek R1 capabilities**: DeepSeek R1's performance on benchmarks shows it can compete effectively with models like OpenAI's O1, with some users eager for its inclusion in Cursor.
   - Discussion around the open-source nature of DeepSeek R1 and its application through API access highlights its potential advantages over other AI assistants.
- **Agent functionality needs improvement**: Participants engaged in discussions about how Cursor's agent currently fails to manage large files and can inadvertently delete important code, necessitating additional manual checks.
   - Users are seeking ways to improve this experience with suggestions for cursor rules and ensuring AI tools support iterative development without error.
- **Comparison of AI assistants**: As users compare Cursor's functionalities with those of Cline and GitHub Copilot, significant concerns regarding different models and their cost-effectiveness arise.
   - The community seems divided on the effectiveness of various tools, with some emphasizing the importance of thorough documentation and manual review in conjunction with AI.
- **Feedback and development suggestions**: Users propose incorporating models like DeepSeek R1 into Cursor to enhance its capabilities and address current performance woes.
   - The importance of community feedback has become apparent, with users anticipating updates and patches from Cursor to resolve ongoing issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://half-single-ecd.notion.site/Experimental-Prompting-86aa8f988fce404cbf70134690d2635a?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://x.com/bintz_gavin/status/1880789354211094906">Tweet from 🥞 Gavin (@bintz_gavin)</a>: it seems t3 chat by @theo is good signal that we want faster ai appsI just open-sourced a simple agentic blog writer that uses  @CerebrasSystems for inference at ~2,100 tokens/s, even faster than groq...</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server">Live&#32;Preview&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Hosts&#32;a&#32;local&#32;server&#32;in&#32;your&#32;workspace&#32;for&#32;you&#32;to&#32;preview&#32;your&#32;webpages&#32;on.</li><li><a href="https://www.cursor.com/downloads">Downloads | Cursor - The AI Code Editor</a>: Choose your platform to download the latest version of Cursor.</li><li><a href="https://x.com/paulgauthier/status/1881428345973608901">Tweet from Paul Gauthier (@paulgauthier)</a>: DeepSeek R1 gets 57% on the aider polyglot benchmark, ranks 2nd behind o1:62% o1 (high)57% DeepSeek R152% Sonnet48% DeepSeek Chat V3Full leaderboard:https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/moritzkremb/status/1880628661931634700?s=19">Tweet from Moritz Kremb (@moritzkremb)</a>: I&#39;ve done 100s of tests with Cursor&#39;s agent feature at this point.If you want the agent to complete an extensive workflow with a single command, the best way to do that is...</li><li><a href="https://codeium.com/changelog)">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://open-vsx.org/vscode/item?itemName=ms-vscode.live-server">Open VSX Registry</a>: no description found</li><li><a href="https://traycer.ai/">Traycer: AI-Powered Pair Programming</a>: An AI-powered coding assistant that plans, implements, and validates every change 🚀</li><li><a href="https://x.com/sama/status/1881258443669172470?s=46&t=kUuVqsG2GMX14zvB592G5w">Tweet from Sam Altman (@sama)</a>: twitter hype is out of control again. we are not gonna deploy AGI next month, nor have we built it.we have some very cool stuff for you but pls chill and cut your expectations 100x!</li><li><a href="https://www.youtube.com/watch?v=AtuB7p-JU8Y">Cursor vs Cline | 240k Tokens Codebase Side-by-Side AI Coding Battle</a>: 🚀 In this video, we use a 240000 token codebase to compare two top notch AI Coding tools against each other: Cursor and Cline. Watch as we compare their fea...</li><li><a href="https://github.com/features/copilot/extensions">GitHub Copilot Extensions · Your favorite tools have entered Copilot Chat.</a>: Extend GitHub Copilot with ready-to-use extensions or build your own using our developer platform with APIs, documentation, and guides.</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">Reasoning Model (deepseek-reasoner) | DeepSeek API Docs</a>: deepseek-reasoner is a reasoning model developed by DeepSeek. Before delivering the final answer, the model first generates a Chain of Thought (CoT) to enhance the accuracy of its responses. Our API p...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1329930548235473007)** (522 messages🔥🔥🔥): 

> `DeepSeek-R1, AI and Crypto, MiniCPM-o 2.6, Reasoning Models, Reinforcement Learning` 


- **DeepSeek-R1 and its Distillation Process**: Participants discussed the recent release of DeepSeek-R1, noting its successful distillation results and the implications for future reasoning models.
   - There is excitement about the potential for open-source reasoning with models that can optimize reasoning processes through RL and other approaches.
- **AI Integration with Crypto**: The community debated the intersection of AI and crypto, exploring how AI agents could potentially utilize crypto for trading resources and executing tasks.
   - Concerns arose over the existing issues in the crypto space, particularly regarding investment motivations which may detract from beneficial applications.
- **MiniCPM-o 2.6 Model Capabilities**: Members expressed interest in the functionalities of MiniCPM-o 2.6, a model designed for vision, speech, and multimodal applications.
   - Discussions highlighted the model's performance, quantization options, and comparisons to existing AI models for practicality in varied applications.
- **Reinforcement Learning and Outcome Rewards**: Participants examined the methodology of using outcome rewards in deep learning and its implications on model performance.
   - Insights were shared on how RL can encourage models to learn optimally without being explicitly instructed, leading to organic development of reasoning capabilities.
- **Community Concerns Over Hosting Providers**: There were complaints about the performance of Lambda's hosting service for Hermes 3 405B, particularly with frequent errors.
   - Members discussed alternative providers and solutions for more reliable hosting options that meet their computational needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/AGI-0/Art-v0-3B">AGI-0/Art-v0-3B · Hugging Face</a>: no description found</li><li><a href="https://x.com/bozoeggs/status/1881328847121236463">Tweet from Egg (@bozoeggs)</a>: See I told you folks all I had to do was sing chinese karaoke and AGI at home would appearQuoting DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-int4">openbmb/MiniCPM-o-2_6-int4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Donnyed/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M-GGUF">Donnyed/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/dani_avila7/status/1880739290264809683?s=46">Tweet from Daniel San (@dani_avila7)</a>: Introducing Codebase Knowledge Graphs in Cursor 🤩In this video, I’ll walk you through how we went from using a knowledge graph of a repository on the CodeGPT platform to leveraging it directly within...</li><li><a href="https://tenor.com/view/its-happening-gif-23353691">Its Happening GIF - Its Happening - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-wizard-meme-funny-gif-3870502440791733376">Cat Wizard GIF - Cat Wizard Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B">FreedomIntelligence/HuatuoGPT-o1-70B · Hugging Face</a>: no description found</li><li><a href="https://x.com/teortaxesTex/status/1881331287010550119">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: perhaps the craziest thing is that they say that this is nowhere near the ceiling of 7-70B class models. Without any new data even. They have pushed them further, but just won&#39;t be sharing it. Dri...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://forms.gle/F9DtNZtkWquHyQyS8">Community Insight: Awareness and Opinion Survey on FirstBank (Headquarters in Nashville, Tennessee)  </a>: This is just for a research project I&#39;m doing! Thank you so much for answering it cuz I&#39;m im on a huge time crunch and I needed this asap for my project.</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1hizjq4/i_have_underestimated_o3s_price">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>: Contribute to MoonshotAI/Kimi-k1.5 development by creating an account on GitHub.</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf">Kimi-k1.5/Kimi_k1.5.pdf at main · MoonshotAI/Kimi-k1.5</a>: Contribute to MoonshotAI/Kimi-k1.5 development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k">NovaSky-AI/Sky-T1_data_17k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/casper-hansen/AutoAWQ/pull/688">Added DeepSeek V3 support. by LagPixelLOL · Pull Request #688 · casper-hansen/AutoAWQ</a>: #686I only tested using randomly initialized weights on a 1B version of the model, so this needs further testing for the big 671B model.Also due to the group size limitation in the gemm CUDA kern...</li><li><a href="https://www.youtube.com/watch?v=JFJg9KZ_iZk.">MiniCPM-o 2.6:  An 8B size, GPT-4o level Omni Model runs on device</a>: 💥 Introducing our MiniCPM-o 2.6:  An 8B size, GPT-4o level Omni Model runs on device✨ Highlights:Match GPT-4o-202405 in vision, audio and multimodal live st...</li><li><a href="https://github.com/OpenBMB/MiniCPM-o">GitHub - OpenBMB/MiniCPM-o: MiniCPM-o 2.6: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone</a>: MiniCPM-o 2.6: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone - OpenBMB/MiniCPM-o
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1330030930970284174)** (36 messages🔥): 

> `High accuracy handwritten text OCR models, Contrast between MOEs and dense models, Efficiency of structured sparsity in AI models, Learning rate scheduling in LLM training` 


- **OCR Models Face Misreading Challenges**: Users discussed their experiences with various high accuracy **handwritten text OCR models** like **Sonnet-3.5** and **Qwen**, which often misread characters.
   - One suggested using **OCR** or object detection to improve character recognition on languages with weak OCR libraries.
- **MOEs vs Dense Models - Parameter Comparison**: A user explored how to compare **MOEs** with **dense models**, suggesting that the equivalent size of a dense model is the geometric mean between active and total parameters.
   - They calculated equivalents for **Deepseek V3** and **Minimax-01**, theorizing a **3-4x latency improvement** could be achieved at a higher parameter memory footprint.
- **Structured Sparsity's Impact on Model Efficiency**: Structured sparsity was highlighted as an effective method for improving efficiency, especially with **Nvidia Ampere** hardware supporting **2:1 sparsity** to reduce compute requirements.
   - Members noted that while this method helps with computational speed, memory requirements remain similar.
- **Depthwise MLP Blocks Present Compromise**: A user proposed using **depthwise MLP blocks** as a compromise between dense and MOE architectures, splitting incoming activations for potential parameter savings.
   - Members discussed similarities to **groupwise convolutions** and noted that these approaches could lead to more efficient network designs.
- **Questions on Cosine Warmup Decay Scheduler**: An inquiry was made regarding the use of a **cosine warmup decay scheduler** when continuing training a **GPT-2 model**, specifically about adjusting total training steps.
   - The user expressed concern that not updating the steps could lead to discrepancies in learning rates for their continued training.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/">Structured Sparsity in the NVIDIA Ampere Architecture and Applications in Search Engines | NVIDIA Technical Blog</a>: Deep learning is achieving significant success in various fields and areas, as it has revolutionized the way we analyze, understand, and manipulate data. There are many success stories in computer&#82...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1330386739704889344)** (2 messages): 

> `Climate Change Impact on Agriculture, Mind Evolution in LLMs` 


- **Collaboration Needed for Climate Research**: A research project titled **'Developing a Convolutional Neural Network to Evaluate the Impact of Climate Change on Global Agricultural Yields'** is seeking collaborators with expertise in multiple fields like **Machine Learning**, **Climate Science**, and **Data Analysis**.
   - Interested individuals are encouraged to contact via DM by **January 25** to finalize the team before further project details are shared.
- **Google's Mind Evolution Outshines Others**: In a recent presentation, Google highlighted how their **Mind Evolution** method significantly outperforms other inference strategies like **Best-of-N** and **Sequential Revision** in natural language planning tasks.
   - The findings show that Mind Evolution solved over **98%** of problem instances in benchmarks such as **TravelPlanner** and **Natural Plan** using **Gemini 1.5 Pro** without a formal solver.



**Link mentioned**: <a href="https://x.com/_akhaliq/status/1881182840857178146?s=46">Tweet from AK (@_akhaliq)</a>: Google presents Evolving Deeper LLM ThinkingControlling for inference cost, we find that Mind Evolution significantly outperforms other inference strategies such as Best-of-N and Sequential Revision i...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1331008929018413271)** (4 messages): 

> `Liquid AI LFM-7B, Recurrent models influence, New business model, Mistral Ministral 3B, Codestral 2501` 


- **Liquid AI launches LFM-7B, claims best-in-class**: [Liquid AI](https://www.liquid.ai/lfm-7b) just released the LFM-7B, touted as the best-performing model in its size class, leveraging a non-transformer architecture for high throughput and low memory usage.
   - This multilingual model supports **English, Arabic**, and **Japanese**, optimized for **local deployment** and cost-constrained tasks.
- **Curiosity about LFM-7B's recurrent design**: A member expressed curiosity about how LFM-7B's **recurrent architecture** may influence its capabilities, given its smaller model size.
   - They noted that it seems to perform adequately in interactions, aligning with expectations for small models.
- **Liquid AI's unique business model of licensing weights**: Liquid AI appears to have an interesting approach by selling or licensing model weights, which is described as a middle ground strategy not commonly seen before.
   - This could signify a shift in the landscape for AI model distribution and accessibility.
- **Mistral's potential similar approach with Ministers 3B and Codestral 2501**: A member speculated that **Mistral** might be adopting a similar licensing strategy for their models, **Ministral 3B** and **Codestral 2501**.
   - This suggests a growing trend among AI companies to offer flexible licensing options for their models.



**Link mentioned**: <a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: The world’s best-in-class English, Arabic, and Japanese model, native in French, German, and Spanish, optimized to be the substrate for private enterprise chat, code, fast instruction following, and a...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1330386739704889344)** (2 messages): 

> `Collaborative Research on Climate Change, Google's Mind Evolution in LLMs` 


- **Seeking Collaborators for Climate Change Research**: A team is initiating a project titled 'Developing a Convolutional Neural Network to Evaluate the Impact of Climate Change on Global Agricultural Yields' and is looking for experts in **Machine Learning**, **Climate Science**, **Geospatial Data**, and **Scientific Writing** to join before **January 25**.
   - *Passionate individuals* can DM thomasyoungabc123 on Discord for collaboration opportunities.
- **Google's Mind Evolution Outperforms Other Strategies**: In a recent update, it was noted that Google's **Mind Evolution** method significantly outperforms strategies like **Best-of-N** and **Sequential Revision** in natural language planning tasks, achieving over **98%** success in benchmarks.
   - This performance was highlighted using **Gemini 1.5 Pro** without the need for a formal solver, demonstrating its effectiveness in solving problems.



**Link mentioned**: <a href="https://x.com/_akhaliq/status/1881182840857178146?s=46">Tweet from AK (@_akhaliq)</a>: Google presents Evolving Deeper LLM ThinkingControlling for inference cost, we find that Mind Evolution significantly outperforms other inference strategies such as Best-of-N and Sequential Revision i...

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1329905489563226122)** (450 messages🔥🔥🔥): 

> `DeepSeek R1 Models, Unsloth Training Script, Quantization Methods, Windows Installation Issues, VTube Models and Rigging` 


- **DeepSeek R1 Models Uploaded**: All versions of DeepSeek R1, including GGUF and quantized formats, have been uploaded to Hugging Face, enhancing model accessibility.
   - The collection includes distilled models for both Llama and Qwen, providing various formats for users.
- **Introduction of Guided Unsloth Training Script**: A guided script for Unsloth training has been created, allowing users to input various training parameters before execution.
   - This simplifies the training process and is available as a GitHub Gist for community use.
- **Discussion on Quantization Methods**: IQ quantization methods were discussed, with emphasis on their complexities and potential effectiveness compared to regular quantization.
   - The conversation highlighted the difficulty of sourcing appropriate calibration sets for high-quality IQ quantization.
- **Windows Installation Challenges with llama.cpp**: Users faced challenges when trying to compile llama.cpp on Windows due to missing `make` or `cmake` commands, indicated by error messages in the logs.
   - It was suggested that manual building might be necessary, as the current script was failing to recognize the operating system.
- **VTube Models and Community Concerns**: The community discussed the monetary aspects of VTube models, particularly the vendor-lock practices and the challenges posed when model owners do not provide source files.
   - There was a general feeling that the reliance on artist-made models limits freedom, leading to an interest in automation and AI-generated alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://x.com/ggerganov/status/1880237609647034551">Tweet from Georgi Gerganov (@ggerganov)</a>: llama-cli        -hf unsloth/phi-4-GGUFllama-server -hf unsloth/phi-4-GGUF(thanks to @ngxson)</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/tokenizer_config.json#L34">tokenizer_config.json · deepseek-ai/DeepSeek-R1-Distill-Qwen-32B at main</a>: no description found</li><li><a href="https://huggingface.co/JingzeShi/Doge-20M-Instruct">JingzeShi/Doge-20M-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit">unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/sebaxakerhtc/5e7faa4ead6e2f4e0ea69634c3f624ba">Guided script for Unsloth</a>: Guided script for Unsloth. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models">pytorch/SECURITY.md at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://x.com/UnslothAI/status/1881357596717891955">Tweet from Unsloth AI (@UnslothAI)</a>: DeepSeek-R1 GGUF&#39;s are now on @HuggingFace!Includes all Llama & Qwen distilled models + 2 to 8-bit quantized versions.How to run R1: https://unsloth.ai/blog/deepseek-r1DeepSeek-R1 Collection: http...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Aria-UI/Aria-UI-base">Aria-UI/Aria-UI-base · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows Installation | Unsloth Documentation</a>: See how to install Unsloth on Windows with or without WSL.</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/lora-parameters-encyclopedia)">Unsloth Documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF">bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/inst">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Zero">unsloth/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">HuggingFaceFW/fineweb-edu · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/intel/auto-round">GitHub - intel/auto-round: Advanced Quantization Algorithm for LLMs/VLMs.</a>: Advanced Quantization Algorithm for LLMs/VLMs. . Contribute to intel/auto-round development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard">Low-bit Quantized Open LLM Leaderboard - a Hugging Face Space by Intel</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/tree/nightly">GitHub - unslothai/unsloth at nightly</a>: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - GitHub - unslothai/unsloth at nightly</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1329914437225807922)** (11 messages🔥): 

> `OpenRouter for LLM comparison, Open source web UI options, Running models locally, Flowise as a chat framework` 


- **Use OpenRouter for LLM Prompt Comparison**: A member suggested using [OpenRouter](https://openrouter.com/) to create a new chat, allowing users to compare multiple open-source LLMs in one go.
   - Once you hit send, all selected models will respond, although some credits may be required for extensive testing.
- **Open Source UI Choices for Chat Apps**: Several members recommended various open-source web UI options for building chat apps, highlighting the [Open Web UI](https://github.com/open-webui/open-webui) as a strong choice.
   - Another member mentioned **Flowise** and noted that it's suitable for public chat-bots on websites.
- **Finding Libraries for Running Models Locally**: A user inquired about open-source libraries for running models locally, receiving suggestions like **Gpt4all** and **textwebgenui**.
   - It's recommended to check licensing agreements before using these tools.
- **Frontend Development Concerns**: One member expressed reluctance to focus on frontend development, preferring to enhance their skills in AI frameworks instead.
   - Overall, the community offered numerous resources to ease the chat app development process without diving deep into frontend technologies.



**Link mentioned**: <a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: User-friendly AI Interface (Supports Ollama, OpenAI API, ...)</a>: User-friendly AI Interface (Supports Ollama, OpenAI API, ...) - open-webui/open-webui

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1329913530580992047)** (77 messages🔥🔥): 

> `Fine-tuning Models, Model Saving Techniques, Performance Issues with Models, Inference Sampling, Using Unsloth Docs` 


- **Exploring Fine-Tuning of Qwen and Phi Models**: Members discussed their experiences with fine-tuning the **Qwen** and **Phi** models, noting different training times and metrics across models like **Liberation 3.1** (LLM) and **Phi-4**.
   - One user mentioned issues with underfitting on Phi-4, potentially due to the model's increased instruction tuning.
- **Training Loss Observations**: Users shared their observations on training loss metrics, with some reporting low losses on models like **WizardLLM** and **Qwen2.5**, inviting thoughts on trying different formats.
   - There was a specific inquiry about whether using the **Alpaca format** with Qwen2.5 could yield better results.
- **Challenges and Solutions in Model Saving**: Discussion arose around saving fine-tuned models without sacrificing accuracy, particularly when saved in **GGUF format** with **F16** leading to significant loss.
   - Users considered various approaches for ensuring model performance is retained post-saving, with an emphasis on best practices mentioned in the **Unsloth documentation**.
- **Challenges with Inference and Sampling**: Queries were raised regarding the **sampling algorithm** during inference while using Unsloth, particularly related to expected results during evaluation.
   - It was clarified that sampling is primarily a concern during inference rather than during training, affecting how results are interpreted.
- **Loading Models in LM Studio**: An issue with loading the **DeepSeek-R1 Qwen 14B** model in LM Studio was discussed, highlighting an error related to model vocabulary.
   - Resolution came through updating both **LM Studio** and **Nvidia drivers**, which eliminated the loading error and allowed the model to function correctly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-Math-7B-Instruct">unsloth/Qwen2.5-Math-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#if-saving-to-gguf">Troubleshooting | Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#if-saving-to-gguf-or-vllm-16bit-crashes">Troubleshooting | Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1040">What is the right way to load Qwen2&#39;s chat interface? · Issue #1040 · unslothai/unsloth</a>: I get this error: chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: &#39;Qwen2-1.5B&#39; From this code: def test_un...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1330012111388016671)** (20 messages🔥): 

> `Chatterbox Dataset Builder, Sky-T1 Model Performance, Synthetic Datasets, LLM Integration, Docker-Compose Setup` 


- **Chatterbox Dataset Builder Launch**: A new tool, [Chatterbox](https://github.com/invisietch/Chatterbox), was introduced for multi-turn dataset management that allows users to create, edit, and delete conversations with various features such as token counting and tagging.
   - The developer mentioned it will support integration with **OpenWebUI**, **Ollama**, **Flowise**, and **LocalAI** in the future, stating it currently works with kobold and aphrodite using the kobold API.
- **Sky-T1 Model Details Released**: The [Sky-T1-32B](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview) model was highlighted for its performance on par with o1-preview in math and coding, trained on 17K data from Qwen2.5-32B-Instruct.
   - Developed by the **NovaSky Team** at UC Berkeley, it uses a training procedure with a batch size of 96 and takes 19 hours to train on 8 H100 with DeepSpeed Zero-3 Offload.
- **Features and Enhancements for Chatterbox**: Improvements to Chatterbox include a new Docker-compose configuration for easier local setup, allowing setup with a single command, and features for **preferential responses** supporting multi-turn exports.
   - The developer indicated plans to implement LLM integration that can generate responses for both sides of a conversation, adjusting roles in chat history to prevent confusion.
- **Synthetic Datasets Generation**: A proposal for creating **synthetic datasets** on autopilot led to discussions about potentially using webworkers or a CLI for bulk operations based on the same backend API.
   - The developer acknowledged interest in automating the dataset generation process, prompting questions about the approaches to implement it.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview">NovaSky-AI/Sky-T1-32B-Preview · Hugging Face</a>: no description found</li><li><a href="https://github.com/invisietch/Chatterbox">GitHub - invisietch/Chatterbox: Multi-turn dataset management tool for LLM trainers</a>: Multi-turn dataset management tool for LLM trainers - invisietch/Chatterbox</li><li><a href="https://github.com/invisietch/">invisietch - Overview</a>: Dangerous. invisietch has 2 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1329968927778017401)** (8 messages🔥): 

> `Dataset usage for model training, LLM Research Cohort at Cohere For AI, Deep Learning resources for beginners` 


- **Naive but Effective Dataset Strategy**: You can train a smaller model by piping a huge dataset through a teacher model in inference mode to generate input/output pairs.
   - While this approach has been used widely since **GPT-4**, like with **Microsoft's Phi**, it's important not to just replicate style.
- **Join the LLM Research Cohort!**: The **LLM Research Cohort** organized by **Cohere For AI** offers hands-on experience in multilingual long-context challenges, enhancing NLP capabilities.
   - Participants will tackle two tracks focusing on advanced techniques for processing and evaluating **multilingual LLMs** with a kick-off call scheduled for January 10th.
- **Navigating Deep Learning as a Beginner**: A member expressed concerns about how long it would take a beginner to learn deep learning and cope with constant updates in the field.
   - One suggestion was to leverage resources like **ChatGPT** to help understand concepts and tackle challenges in deep learning and AI.



**Link mentioned**: <a href="https://x.com/cataluna84/status/1877689686639992872">Tweet from Mayank Bhaskar (@cataluna84)</a>: From the BIRDS(Beginners in Research Driven Studies) organized by @akankshanc of @cohere Open Science Community, we&#39;re thrilled to announce our new LLM Cohort! 🎉 🚀This isn&#39;t just another lea...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1329958392214585465)** (154 messages🔥🔥): 

> `RWKV Model Discussions, Model Quantization Formats, Mixture of Experts (MoE), Performance of AI Models, AI Development and Career Sharing` 


- **RWKV7 holds strong position in generational models**: RWKV7 is recognized as the only gen7 RNN releasing usable models, signifying its unique standing in current AI architectures. Discussions highlighted its design similarities to other models like Gated DeltaNet, emphasizing ongoing improvements.
   - Members debated the impacts of design features like channel-wise decay and learning rate, showcasing RWKV7’s competitive edge against older models.
- **Transition to GGUF as major quantized model format**: GGUF has emerged as the dominant format for quantized models, favored for its ease of use on consumer hardware and availability from major quantizers. As GGUF gains traction, other formats like AWQ and GPTQ may continue to exist but are lagging behind in adoption.
   - Participants noted that major companies often quantize their models internally, resulting in more GGUF files being made available in the open-source community.
- **Exploring Mixture of Experts (MoE)**: MoE is noted for its efficiency and performance benefits, although some members expressed concerns regarding its stability during training. Articles discussing the MoE paradigm have been highlighted as useful resources for understanding its implementation.
   - Members shared sentiments on how understanding and applying MoE can be challenging, yet potentially rewarding in AI model architectures.
- **Scaling and model deployment strategies**: Discussions centered around the efficiency of various tools like VLLM and Ollama for deploying small AI models, with preferences varying based on company size and load requirements. VLLM is praised for its ability to scale effectively, making it a popular choice among professional groups.
   - In contrast, Ollama is seen as less effective under heavy loads, raising questions about its practicality compared to other solutions available in the market.
- **AI Developer Connections and Career Opportunities**: Community members actively introduced themselves, sharing their backgrounds in AI development and seeking collaboration opportunities. Conversations highlighted the diverse experience in AI services and the interest in establishing connections within the field.
   - Notably, a member expressed their intentions to connect and collaborate with others in the community, illustrating the growing network of AI professionals.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>: Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, ...</li><li><a href="https://arxiv.org/abs/2409.19044">On the Inductive Bias of Stacking Towards Improving Reasoning</a>: Given the increasing scale of model sizes, novel training strategies like gradual stacking [Gong et al., 2019, Reddi et al., 2023] have garnered interest. Stacking enables efficient training by gradua...</li><li><a href="https://arxiv.org/abs/2501.08313">MiniMax-01: Scaling Foundation Models with Lightning Attention</a>: We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01, which are comparable to top-tier models while offering superior capabilities in processing longer contexts. The core lies i...</li><li><a href="https://x.com/BlinkDL_AI/status/1876849157920452781">Tweet from BlinkDL (@BlinkDL_AI)</a>: RWKV-7 &#34;Goose&#34; 🪿 World 0.4B release: strongest base model at its size🚀Download: https://huggingface.co/BlinkDL/rwkv-7-world Demo: https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1Quoting B...</li><li><a href="https://newsletter.armand.so/p/understanding-mixture-experts">Understanding Mixture of Experts</a>: Too good to be true?</li><li><a href="https://huggingface.co/blog/moe">Mixture of Experts Explained</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1329904593152249976)** (297 messages🔥🔥): 

> `DeepSeek R1, Gradient Spikes, Optimization Techniques, Titan Models and Memorization, RL Training in LLMs` 


- **DeepSeek R1 Showcases Performance Gains**: DeepSeek R1 introduces a new approach with impressive performance on benchmarks like AIME and MATH-500, outperforming GPT-4o and Claude Sonnet 3.5 by a notable margin.
   - The model's effectiveness in longer reasoning tasks and its ability to handle extended contexts up to 128k tokens contribute significantly to its capabilities.
- **Course of Studies on Gradient Spikes**: Discussion centered around the impact of gradient spikes in model training, with consensus suggesting that spikes can lead to permanent damage to model capacity and performance.
   - The importance of adjusting hyperparameters to mitigate these issues was emphasized, alongside concerns regarding the implications of recoverable spikes.
- **Debate on Optimization Techniques**: Experts discussed the merits and drawbacks of various optimization methods, pointing out that certain approaches may look good on paper but fail in practice.
   - There were considerations regarding the potential of learned optimization algorithms to improve over hand-designed methods, as evidenced by prior research.
- **Understanding Titans' Memorization Mechanism**: The Titans paper discusses the significance of memorizing mappings between keys and values during test time, indicating a deeper understanding of inner-loop training.
   - This concept is rooted in the broader context of learning to learn and optimizes the model's performance based on historical data associations.
- **Exploration of RL Techniques in Model Training**: The conversation touched on the utility of reinforcement learning (RL) in training language models, particularly regarding its effectiveness with varying context lengths.
   - It was suggested that running experiments with different lengths could shed light on the comparative benefits of RL training approaches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cataluna84/status/1877689686639992872">Tweet from Mayank Bhaskar (@cataluna84)</a>: From the BIRDS(Beginners in Research Driven Studies) organized by @akankshanc of @cohere Open Science Community, we&#39;re thrilled to announce our new LLM Cohort! 🎉 🚀This isn&#39;t just another lea...</li><li><a href="https://arxiv.org/abs/2410.22570v1#S6.SS3">Orb: A Fast, Scalable Neural Network Potential</a>: We introduce Orb, a family of universal interatomic potentials for atomistic modelling of materials. Orb models are 3-6 times faster than existing universal potentials, stable under simulation for a r...</li><li><a href="https://arxiv.org/abs/2306.13326">Solving systems of Random Equations via First and Second-Order Optimization Algorithms</a>: Gradient-based (a.k.a. `first order&#39;) optimization algorithms are routinely used to solve large scale non-convex problems. Yet, it is generally hard to predict their effectiveness. In order to gai...</li><li><a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memo...</li><li><a href="https://arxiv.org/abs/2112.00114">Show Your Work: Scratchpads for Intermediate Computation with Language Models</a>: Large pre-trained language models perform remarkably well on tasks that can be done &#34;in one pass&#34;, such as generating realistic text or synthesizing computer programs. However, they struggle w...</li><li><a href="https://arxiv.org/abs/2501.06842">SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training</a>: Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks, yet their training remains highly resource-intensive and susceptible to critical challenges such as trainin...</li><li><a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>: We explore an evolutionary search strategy for scaling inference time compute in Large Language Models. The proposed approach, Mind Evolution, uses a language model to generate, recombine and refine c...</li><li><a href="https://arxiv.org/abs/2501.00663">Titans: Learning to Memorize at Test Time</a>: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memo...</li><li><a href="https://x.com/rosstaylor90/status/1881374050079187037">Tweet from Ross Taylor (@rosstaylor90)</a>: @xpasky If naively rewarding it: yes.But imagine a reward structure like this: if you have an incorrect answer then the model stopped thinking too early. If you have a correct answer, the model used a...</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: DeepSeek R1 with &#34;Cold Start&#34; pretty much works as expected. I still don&#39;t buy the R1 Zero result, the base models barely output coherent solutions without finagling. My bet is there is so...</li><li><a href="https://arxiv.org/abs/1606.04474">Learning to learn by gradient descent by gradient descent</a>: The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how...</li><li><a href="https://x.com/BlinkDL_AI/status/1855245097094517181">Tweet from BlinkDL (@BlinkDL_AI)</a>: RWKV-7 can also reach 2.27xx in 3200 steps (originally 5100 steps)😀reproducible code & log: https://github.com/BlinkDL/modded-nanogpt-rwkv 🚀 #RWKV #RNNQuoting Keller Jordan (@kellerjordan0) It&#39;s...</li><li><a href="https://x.com/kimi_ai_/status/1881332472748851259?s=46">Tweet from Kimi.ai (@Kimi_ai_)</a>: 🚀 Introducing Kimi k1.5 --- an o1-level multi-modal model-Sota short-CoT performance, outperforming GPT-4o and Claude Sonnet 3.5 on 📐AIME, 📐MATH-500, 💻 LiveCodeBench by a large margin (up to +550%...</li><li><a href="https://x.com/rm_rafailov/status/1880994108241842314">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: @armanhaghighik @iScienceLuvr None of that “just emerges”, it’s pure bullshit. Clearly all the different models are seeded with different strategies before the RL stage.</li><li><a href="https://x.com/BlinkDL_AI/status/185">Tweet from crystal (@crystal)</a>: adam hates my username.</li><li><a href="https://x.com/deepseek_ai/status/1859200149844803724">Tweet from DeepSeek (@deepseek_ai)</a>: 🌟 Inference Scaling Laws of DeepSeek-R1-Lite-PreviewLonger Reasoning, Better Performance. DeepSeek-R1-Lite-Preview shows steady score improvements on AIME as thought length increases.</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/ttt">flash-linear-attention/fla/ops/ttt at main · fla-org/flash-linear-attention</a>: 🚀 Efficient implementations of state-of-the-art linear attention models in Pytorch and Triton - fla-org/flash-linear-attention</li><li><a href="https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/demo-training-prepare-v7-pile.sh">RWKV-LM/RWKV-v5/demo-training-prepare-v7-pile.sh at main · BlinkDL/RWKV-LM</a>: RWKV (pronounced RwaKuv) is an RNN with great LLM performance, which can also be directly trained like a GPT transformer (parallelizable). We are at RWKV-7 &amp;quot;Goose&amp;quot;. So it&amp;#39;s c...</li><li><a href="https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/">Hymba Hybrid&#x2d;Head Architecture Boosts Small Language Model Performance | NVIDIA Technical Blog</a>: Transformers, with their attention&#x2d;based architecture, have become the dominant choice for language models (LMs) due to their strong performance, parallelization capabilities, and long&#x2d;term ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1331050552292806799)** (4 messages): 

> `Steering LLMs with SAE features, Open source steering libraries` 


- **Current Limitations in Steering LLMs with SAEs**: Members noted that *things aren't standardized enough* for steering **LLMs** using selected features from trained **SAEs** yet, indicating a gap in the field.
   - For a deeper understanding, they shared a [discussion on current SAE feature steering methods](https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484).
- **Open Source Steering Libraries Available**: A member shared several open-source steering libraries including [steering-vectors](https://github.com/steering-vectors/steering-vectors), [repeng](https://github.com/vgel/repeng), and [representation-engineering](https://github.com/andyzoujm/representation-engineering).
   - In particular, the [Representation Engineering repository](https://github.com/andyzoujm/representation-engineering) focuses on AI transparency from a top-down perspective.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://github.com/andyzoujm/representation-engineering">GitHub - andyzoujm/representation-engineering: Representation Engineering: A Top-Down Approach to AI Transparency</a>: Representation Engineering: A Top-Down Approach to AI Transparency - andyzoujm/representation-engineering
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329927183023476796)** (63 messages🔥🔥): 

> `Qwen2.5 performance discrepancies, Few-shot prompting techniques, VLLM evaluation issues, Quantization effects on performance, MMLU-PRO evaluation insights` 


- **Qwen2.5's Performance Not Matching Expectations**: Users reported that **Qwen2.5-1.5B-Instruct** and the non-instruct version both achieve around **60%** accuracy on **gsm8k**, whereas the expected performance is **73%** and **65%** respectively based on their [blog post](https://qwenlm.github.io/blog/qwen2.5-llm/).
   - Members discussed the evaluation method differences, noting they may not parse answers effectively, which could impact scores.
- **Alternating Question/Answer Few-shot Technique**: A suggestion was made to incorporate a few-shot format with alternating question and answer pairs used in Qwen's evaluation into the **lm-eval** harness for improved performance.
   - After applying the 'let's think step by step' technique, one member noted an improvement, raising scores to **66%**.
- **Discussion on VLLM Evaluation Variability**: Concerns were raised about discrepancies in performance results when using **vllm** compared to other frameworks like the **HF API**, with previous user complaints noted.
   - Although some members initially suspected vllm as the source of performance issues, others expressed confidence in its current capabilities.
- **Quantization Impact on Recent Models**: A member inquired about the performance degradation related to **4bit/3bit vs f16** for recent llama or qwen models, questioning if the losses were negligible or dependent on quantizing efforts.
   - They also sought recommendations for related academic papers to gain better insights into quantization effects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen2.5-llm/">Qwen2.5-LLM: Extending the boundary of LLMs</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORDIntroduction In this blog, we delve into the details of our latest Qwen2.5 series language models. We have developed a range of decoder-only dense models, wi...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9dda03d6be6c94cc803b6189302a8a148c5e4d12/lm_eval/tasks/leaderboard/math/_template_yaml#L1)">lm-evaluation-harness/lm_eval/tasks/leaderboard/math/_template_yaml at 9dda03d6be6c94cc803b6189302a8a148c5e4d12 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/QwenLM/Qwen/blob/f014f2ef1a72563bbd28b055a4667eaf102c6f21/eval/evaluate_gsm8k.py#L62">Qwen/eval/evaluate_gsm8k.py at f014f2ef1a72563bbd28b055a4667eaf102c6f21 · QwenLM/Qwen</a>: The official repo of Qwen (通义千问) chat &amp; pretrained large language model proposed by Alibaba Cloud. - QwenLM/Qwen</li><li><a href="https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py">Qwen/eval/evaluate_gsm8k.py at main · QwenLM/Qwen</a>: The official repo of Qwen (通义千问) chat &amp; pretrained large language model proposed by Alibaba Cloud. - QwenLM/Qwen</li><li><a href="https://github.com/QwenLM/Qwen/blob/f014f2ef1a72563bbd28b055a4667eaf102c6f21/eval/evaluate_chat_gsm8k.py#L23)">Qwen/eval/evaluate_chat_gsm8k.py at f014f2ef1a72563bbd28b055a4667eaf102c6f21 · QwenLM/Qwen</a>: The official repo of Qwen (通义千问) chat &amp; pretrained large language model proposed by Alibaba Cloud. - QwenLM/Qwen</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9dda03d6be6c94cc803b6189302a8a148c5e4d12/lm_eval/tasks/minerva_math/utils.py#L45)">lm-evaluation-harness/lm_eval/tasks/minerva_math/utils.py at 9dda03d6be6c94cc803b6189302a8a148c5e4d12 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1330798959249326144)** (1 messages): 

> `phi 3 and 3.5 vision, MPS device errors` 


- **Error with phi 3 and 3.5 on MPS**: A member encountered an error while trying to run **phi 3** and **phi 3.5 vision** on Mac with **MPS** device set.
   - They reported that **placeholder storage** has not been allocated on the **MPS device**, seeking assistance for resolution.
- **Seeking assistance for MPS allocation issue**: The member is looking for any clues or solutions related to MPS device functionality when utilizing **phi 3** and **phi 3.5 vision**.
   - The specific error mentioned indicates a problem with memory allocation that could hinder successful execution on the Mac.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1329919555325136990)** (8 messages🔥): 

> `Host RAM Requirements, Vocab Size Optimization, 3D Parallelism with ZeRO Stage 1, Issue Raising for Hangs, Updating Markdown Files` 


- **Host RAM and CPU Core Guidance**: Host RAM should be roughly equivalent to GPU VRAM, with optimizations like **CPU Adam** increasing memory demands. Typically, 2–4 cores per GPU suffices, depending on CPU architecture and pipeline complexity.
   - *A good rule of thumb is to have host RAM equivalent to the required GPU VRAM*, while training can often function with less.
- **Vocab Size Divisibility for Efficiency**: Vocab size should be made divisible by **128*MP** for optimization, though it can be overridden at your own risk. The risks of deviating from this norm were communicated by a member.
   - It is noted that overriding the default setting is *not highly recommended* as it may lead to complications.
- **Exploring MP, PP, and ZeRO Stage 1**: Members discussed the benefits of using **MP+PP+ZeRO Stage 1** for optimizing performance and improving throughput. Activation of **memory optimizations** and **flash attention** were suggested as effective enhancements.
   - *Double the initial flops* was reported as an achievement with these methods, although some caution around trusting maximum reported flops was advised.
- **Raising Issues for Hangs**: A user expressed the intention to raise an issue regarding a hang in the process, asking for detailed information about their setup. They assured that they would address it upon finding time amidst their travels.
   - Another member reminded the user to include detailed information so that the hang issue might be resolved efficiently.
- **Improving ARGS Markdown File**: There was a suggestion to reexport the **ARGS markdown file** since it lacked certain parameters. This indicates a potential oversight that could help clarify usage and configurations for users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/neox_arguments/neox_args.py#L801">gpt-neox/megatron/neox_arguments/neox_args.py at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329">GitHub - EleutherAI/gpt-neox at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - GitHub - EleutherAI/gpt-neox at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/configs/neox_arguments.md?plain=1#L567">gpt-neox/configs/neox_arguments.md at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1329904328919748639)** (237 messages🔥🔥): 

> `DeepSeek-R1 Release, Kimi 1.5 Paper Insights, GRPO and RLHF, Benchmarking Evaluations, Impacts of MIT Licensing` 


- **DeepSeek-R1 surpasses expectations**: DeepSeek-R1 has demonstrated performance exceeding that of OpenAI's o1, showcasing significant advancements in reasoning capabilities with an MIT license.
   - The community is excited about its open-source nature, making it accessible for various applications, alongside strong evaluations supporting its effectiveness.
- **Kimi 1.5 reveals new RL methods**: A new paper on Kimi 1.5 provides insights into reward shaping and reinforcement learning infrastructure that could benefit similar model developments.
   - This paper is anticipated to stir interest in the ongoing research into RL and could complement existing knowledge frameworks in the field.
- **Understanding GRPO Simplified**: Natolambert clarified that Group Relative Policy Optimization (GRPO) is just PPO without a value function and relies on Monte Carlo estimates of advantage, streamlining RL understanding.
   - This basic explanation aims to make GRPO more accessible to those new to reinforcement learning methodologies.
- **Community Feedback on Evaluation Metrics**: The community expresses opinions on the reliability of evaluation metrics, noting the ease of manipulating evaluations compared to creating high-quality models.
   - This conversation emphasizes the importance of robust evaluations amidst growing competition in AI model development.
- **Future Directions in RLHF and Reasoning**: Natolambert plans to encapsulate 'v1' of modern RLHF in a concise book, while keeping a close eye on the evolving landscape of reasoning in relation to RL methodologies.
   - The conversation suggests an ongoing need for clear documentation and education in the fast-paced AI research environment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/teortaxesTex/status/1881331287010550119">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: perhaps the craziest thing is that they say that this is nowhere near the ceiling of 7-70B class models. Without any new data even. They have pushed them further, but just won&#39;t be sharing it. Dri...</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: DeepSeek R1 with &#34;Cold Start&#34; pretty much works as expected. I still don&#39;t buy the R1 Zero result, the base models barely output coherent solutions without finagling. My bet is there is so...</li><li><a href="https://x.com/RileyRalmuto/status/1880445415927251435">Tweet from Riley Coyote (@RileyRalmuto)</a>: @TotalTD0 it’s definitely specialized in something…but it’s more in the vein of health and longevity</li><li><a href="https://arxiv.org/abs/2403.04642">Teaching Large Language Models to Reason with Reinforcement Learning</a>: Reinforcement Learning from Human Feedback (\textbf{RLHF}) has emerged as a dominant approach for aligning LLM outputs with human preferences. Inspired by the success of RLHF, we study the performance...</li><li><a href="https://x.com/TheXeophon/status/1881305033352135152">Tweet from Xeophon (@TheXeophon)</a>: R1 pricing</li><li><a href="https://x.com/DanHendrycks/status/1881045781354000604">Tweet from Dan Hendrycks (@DanHendrycks)</a>: Humanity&#39;s Last Exam is being released this upcoming week, so we can test models&#39; research-level STEM capabilities with that.</li><li><a href="https://x.com/LiquidAI_/status/1881236162893000944">Tweet from Liquid AI (@LiquidAI_)</a>: Introducing LFM-7B, our new best-in-class language model in English, Arabic, and Japanese optimized to be the substrate for private enterprise chat, code, fast instruction following, and agentic workf...</li><li><a href="https://x.com/teortaxesTex/status/1881245237982724296">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @tokenbender we need a Whale entrance theme</li><li><a href="https://x.com/teortaxesTex/status/1881330229119246843">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: “Do something yourselves, we&#39;ve told you everything there is to know”</li><li><a href="https://x.com/spectatorindex/status/1881054674620703145">Tweet from The Spectator Index (@spectatorindex)</a>: BREAKING: Message as TikTok restores services in the United States</li><li><a href="https://x.com/TheXeophon/status/1881444595009253543">Tweet from Xeophon (@TheXeophon)</a>: This is one of my favorite examples in the bench. The model should detect the unnecessary softmax and notify the user. R1 gets 4/5 - and the one fail is the LLM-as-judge (4o) not correctly judging the...</li><li><a href="https://x.com/teortaxesTex/status/1881331554456371544">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: total @natolambert victory</li><li><a href="https://x.com/sama/status/1880356297985638649">Tweet from Sam Altman (@sama)</a>: thank you to the external safety researchers who tested o3-mini.we have now finalized a version and are beginning the release process; planning to ship in ~a couple of weeks.also, we heard the feedbac...</li><li><a href="https://x.com/Teknium1/status/1881267038091682191">Tweet from Teknium (e/λ) (@Teknium1)</a>: Got me a deepseek reasoning model inferencing ^_^</li><li><a href="https://x.com/natolambert/status/1881380809153847711">Tweet from Nathan Lambert (@natolambert)</a>: For those trying to understand DeepSeeks Group Relative Policy Optimization (GRPO): GRPO is just PPO without a value function using monte carlo estimates of the advantage. So, study why PPO exists (lo...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265">Tweet from Xeophon (@TheXeophon)</a>: I am shocked by R1 on my personal bench. This is the full eval set, it completely crushes the competition and is a whole league on its own, even surpassing o1-preview (which is omitted from the graph ...</li><li><a href="https://x.com/StringChaos/status/1880317308515897761">Tweet from Naman Jain (@StringChaos)</a>: DeepSeek-R1 (Preview) Results 🔥We worked with the @deepseek_ai team to evaluate R1 Preview models on LiveCodeBench. The model performs in the vicinity of o1-Medium providing SOTA reasoning performanc...</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model & technical report🏆 MIT licensed: Distill & commercialize freely!🌐 Website & API are live now! Try DeepThink at h...</li><li><a href="https://x.com/iforgotmytwit1/status/1881314212578046060">Tweet from dhfksowndic (@iforgotmytwit1)</a>: @teortaxesTex</li><li><a href="https://x.com/natolambert/status/1881370064038805714">Tweet from Nathan Lambert (@natolambert)</a>: hahahahah there were actually two technical reports for RL reasoning models today, kimi 1.5 also has good stuff on reward shaping + RL infra</li><li><a href="https://fxtwitter.com/btibor91/status/1881285255266750564">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI website already has references to Operator/OpenAI CUA (Computer Use Agent) - &#34;Operator System Card Table&#34;, &#34;Operator Research Eval Table&#34; and &#34;Operator Refusal Rate Table&#3...</li><li><a href="https://x.com/btibor91/status/1880950883988738482">Tweet from Tibor Blaho (@btibor91)</a>: New &#34;Gemini 2.0 Flash Thinking Experimental&#34; (gemini-2.0-flash-thinking-exp 01-23) reasoning model from Google (01:02:57)Thanks @sir04680280Quoting Alex Reibman 🖇️ (@AlexReibman) Livestreamin...</li><li><a href="https://x.com/deepseek_ai/status/1881318138937233664">Tweet from DeepSeek (@deepseek_ai)</a>: 📜 License Update!🔄 DeepSeek-R1 is now MIT licensed for clear open access🔓 Open for the community to leverage model weights & outputs🛠️ API outputs can now be used for fine-tuning & distillation🐋 ...</li><li><a href="https://tenor.com/view/the-pursuit-of-happiness-will-smith-success-joy-happiness-gif-3517714">Happiness GIF - The Pursuit Of Happiness Will Smith Success - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://x.com/jiayi_pirate/status/1881264063302557919">Tweet from Jiayi Pan (@jiayi_pirate)</a>: @pshishodia_ @Grad62304977 @teortaxesTex @TheXeophon @nrehiew_ From model weights,R1, R1-zero, V3-instruct are all quite different from each other, and R1-zero is closest to V3-baseThey probably all s...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B · Hugging Face</a>: no description found</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">Reasoning Model (deepseek-reasoner) | DeepSeek API Docs</a>: deepseek-reasoner is a reasoning model developed by DeepSeek. Before delivering the final answer, the model first generates a Chain of Thought (CoT) to enhance the accuracy of its responses. Our API p...</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5/tree/main">GitHub - MoonshotAI/Kimi-k1.5</a>: Contribute to MoonshotAI/Kimi-k1.5 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1330065111670587422)** (27 messages🔥): 

> `O1 pro streaming summary, Test-time search vs forward passes, Use of self-consistency in reasoning, Gflownet in training O1, Asymmetry in RL setups` 


- **O1 pro streams thought summaries while reasoning**: A member observed that **O1 pro** streams a summary of its thoughts as they occur, suggesting it merges parallel generations during the thought process instead of at the end.
   - *Streaming summaries would indicate intermediate selection*, as opposed to a final sample selection.
- **Test-time search explanations debated**: Discussion arose around **Francois Chollet's** tweet explaining that instant model responses indicate fewer than 10 forward passes, while longer responses involve test-time search.
   - Some members suggested that this interpretation may not accurately reflect how the **O1 pro** operates during inference.
- **Theory on latent reasoning paths in training**: **Chygao** posited that training for O1 involved using methodologies like **Gflownet** to derive latent reasoning paths, citing a paper that received mention at **ICLR 2024**.
   - This paper explores deriving hidden **chains of thought** leading to an answer through **Bayesian inference**.
- **Discussion on RL asymmetry concerns**: **Catboy_slim_** questioned whether the asymmetrical clipping of negatives and positives in their RL setup was intentional, ultimately recognizing it as common in standard PPO configurations.
   - This asymmetry could soften positive examples while exacerbating negatives, raising questions about stability justifications that weren't fully aligned with their mathematical model.
- **Understanding rewards and penalties in RL**: In the RL discussion, **Natolambert** highlighted that in traditional setups, negatives equate to failure while small rewards are akin to progress.
   - This notion aligns with the justification for non-standard clipping approaches in training, though it raised concerns about the interplay with underlying model mathematics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1863226363542454660">Tweet from wh (@nrehiew_)</a>: This paper got an honourable mention at ICLR 2024 and the first author worked on o1 and was the creator of loratldr: they propose a method to derive the hidden cot that leads to an answer, given a que...</li><li><a href="https://x.com/natolambert/status/1880683907563299233">Tweet from Nathan Lambert (@natolambert)</a>: This is more confusing than helpful as a rule of thumb. Most of the &#34;search&#34; people think of with branching, getting credit assignment from errors, etc happens at training time.Some subtle imp...</li><li><a href="https://x.com/fchollet/status/1880378458909601969">Tweet from François Chollet (@fchollet)</a>: As a general rule: if a model returns a response instantly, it&#39;s not doing test-time search -- it&#39;s doing &lt;10 forward passes.If it takes 5+ min to return something... it&#39;s doing test-ti...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329915079801573439)** (51 messages🔥): 

> `MosaicAI Departures, OpenAI Transparency Issues, Epoch AI and FrontierMath, Perceptron Inc's New Venture, AGI Buzz` 


- **MosaicAI Experiences Departures**: Recent messages highlight multiple departures from **MosaicAI**, with members expressing gratitude for their roles while reflecting on the challenges faced within the company.
   - One outgoing member noted, *'Working at @DbrxMosaicAI has been the honor of a lifetime,'* as they transition to new opportunities in AI.
- **Concerns About OpenAI's Transparency**: Discussions surfaced regarding **OpenAI's** lack of transparency about their partnerships, particularly in relation to **Epoch AI** and its work on the **FrontierMath** dataset.
   - Members indicated that *'OpenAI wanted to keep the funding secret'*, raising questions about the implications of such actions for the integrity of AI research.
- **Epoch AI's Commitment to Transparency**: After acknowledging discrepancies, **Epoch AI** committed to improved transparency regarding their data access and funding sources in future collaborations.
   - A representative stated, *'we should have negotiated harder for the ability to be transparent...'*, highlighting their dedication to better communication going forward.
- **Perceptron Inc. Launches Visual Foundation Models**: A former **MosaicAI** researcher announced their new role at **Perceptron Inc.**, focusing on creating visual language foundation models for real-time video perception, promising resources 1/100th the cost of existing models.
   - They shared excitement about working with talented colleagues stating, *'I am absolutely confident that if anyone can solve this problem it is them.'*
- **Reaction to AGI Speculations**: A tweet from **Sama** addressed outlandish speculations surrounding an imminent AGI deployment, reassuring the community to *'chill and cut your expectations 100x!'*
   - This sentiment resonated with many, reflecting ongoing debate about the term 'AGI' being frequently misused and how it fuels unrealistic expectations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/plain_simon/status/1880949628751011846">Tweet from Simon Pepin Lehalleur (@plain_simon)</a>: @ElliotGlazer &#34;Developing&#34;? The Lesswrong comment strongly implied that this holdout set already existed and was used for validation (&#39;serves&#39; in present tense). Can you clarify?</li><li><a href="https://x.com/TheRealAdamG/status/1881349799888433548">Tweet from Adam.GPT (@TheRealAdamG)</a>: Not all “thinking” is the same.  I expect to see a rise in crappy chains of thoughts.</li><li><a href="https://x.com/mvpatel2000/status/1880802915704820004?s=46">Tweet from Mihir Patel (@mvpatel2000)</a>: After 3 years, I had my last week at @DbrxMosaicAI. I joined MosaicML excited to build a startup and work on cutting edge AI. I leave having experienced a rollercoaster and with lifelong friends. Incr...</li><li><a href="https://x.com/jecdohmann/status/1881418279945978261">Tweet from Jeremy Dohmann (@jecdohmann)</a>: I’m very excited to announce that I’ll be joining @perceptroninc  (https://perceptron.inc/?) as a researcher and founding member of the technical staff. I’ll be working with @AkshatS07  and @ArmenAgha...</li><li><a href="https://x.com/DanHendrycks/status/1881036645555937719">Tweet from Dan Hendrycks (@DanHendrycks)</a>: @GaryMarcus Can confirm AI companies like xAI can&#39;t get access to FrontierMath due to Epoch&#39;s contractual obligation with OpenAI.</li><li><a href="https://x.com/ElliotGlazer/status/1881016863343390946">Tweet from Elliot Glazer (@ElliotGlazer)</a>: @plain_simon “serves” as in “this is its purpose” but I’ll ask Tamay to change to future tense. The holdout set eval score will be public so everyone will know when it’s carried out.</li><li><a href="https://x.com/code_star/status/1880355601546674203">Tweet from Cody Blakeney (@code_star)</a>: Last week was my final week at @DbrxMosaicAI. I am so grateful to have been part of such an amazing team and journey. In my three years, I learned so much about the startup ecosystem, was part of a su...</li><li><a href="https://x.com/sama/status/1881258443669172470">Tweet from Sam Altman (@sama)</a>: twitter hype is out of control again. we are not gonna deploy AGI next month, nor have we built it.we have some very cool stuff for you but pls chill and cut your expectations 100x!</li><li><a href="https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/?commentId=veedfswdCYKZEhptz">meemi&#x27;s Shortform — LessWrong</a>: Comment by Tamay - Tamay from Epoch AI here.We made a mistake in not being more transparent about OpenAI&#x27;s involvement. We were restricted from disclosing the partnership until around the time o3...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329903246965215233)** (76 messages🔥🔥): 

> `Molmo AI, DeepSeek Model Insights, VLM Performance, Trae AI IDE, Chinese Startup Landscape` 


- **Molmo AI garners attention**: Members expressed excitement about [Molmo AI](https://molmo.org/), highlighting its capabilities in multimodal processing and user-friendliness, with claims that it outperforms many existing VLMs.
   - Discussions touched on its strengths, such as adapting well to various tasks, though there were reservations about its occasional mistakes.
- **DeepSeek model discussions**: Amidst discussions about DeepSeek's performance, members mentioned the potential of its latest model to significantly improve various tasks related to image and language understanding.
   - Speculation about a future blog post looking into new releases made the rounds, suggesting there’s a keen interest in detailed insights.
- **Challenges of Visual Language Models**: The community debated the limitations of VLMs in detection tasks, with several contributors questioning the ability of current models to accurately localize objects in images.
   - It was suggested that improvements might come from fine-tuning techniques applied to datasets like PASCAL-VOC, while others argued the complexity of visual token embeddings hinders local information recovery.
- **Trae AI IDE debut**: Trae, an adaptive AI IDE developed by Bytedance, was introduced, with claims of transforming collaboration and productivity in coding environments.
   - Notably, bytedance engineers humorously suggested that Trae stands for 'The real ai engineer', positioning it as a tool for developers.
- **Paywall dynamics discussed**: There were lighthearted banter about introducing a paywall for exclusive content, with suggestions to provide a summary and restrict access to in-depth insights.
   - Members reflected on the implications of paywalls in academia, balancing the need for accessible knowledge against financial sustainability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.11441">Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V</a>: We present Set-of-Mark (SoM), a new visual prompting method, to unleash the visual grounding abilities of large multimodal models (LMMs), such as GPT-4V. As illustrated in Fig. 1 (right), we employ of...</li><li><a href="https://molmo.org/">Molmo AI: Open-Source Multimodal AI Model | Free &amp; Powerful</a>: Discover Molmo AI, the state-of-the-art open-source multimodal AI model. Powerful, free, and easy to use. Learn how Molmo compares to other AI models.</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>: This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it is w...</li><li><a href="https://www.ciciai.com/">Cici</a>: Cici AI is your AI chat assistant for intelligent conversations, writing, translation, emotional support, and programming. Get answers, find inspiration, and discuss any topic with Cici AI.</li><li><a href="https://x.com/MLiegertova/status/1880674661731828214">Tweet from Michaela Lie ☕☕☕ (@MLiegertova)</a>: @JeremyNguyenPhD Counting like Pro! 😁</li><li><a href="https://x.com/TheXeophon/status/1880513932609323060">Tweet from Xeophon (@TheXeophon)</a>: @JeremyNguyenPhD @vikhyatk I have played with some models and I think molmo is the best - it is not perfect (missed two) BUT you can easily see this as the model can point at this. Would&#39;ve saved ...</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">Tweet from mgostIH (@mgostIH)</a>: Wtf is up with deep learning???</li><li><a href="https://www.trae.ai/">Trae - Ship Faster with Trae</a>: no description found</li><li><a href="https://x.com/dylan522p/status/1880379652054901175">Tweet from Dylan Patel (@dylan522p)</a>: Elon&#39;s jet is in Florida.Global Foundries jet is in Florida.Qualcomm&#39;s jet is in Florida.In case anyone was wondering what&#39;s going on with Intel...They are at Mar-a-Lago.Make America Great...</li><li><a href="https://arxiv.org/abs/2401.06209">Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs</a>: Is vision good enough for language? Recent advancements in multimodal models primarily stem from the powerful reasoning abilities of large language models (LLMs). However, the visual component typical...</li><li><a href="https://youtu.be/76EL7YVAwVo?si=Xwu17VAzJkd6YiNQ&t=1254)">Best of 2024 in Vision [LS Live @ NeurIPS]</a>: Peter Robicheaux and Isaac Robinson of Roboflow and Vik Korrapati at Moondream recap the best work of 2024 in frontier/open model vision work!slides and show...</li><li><a href="https://mp.weixin.qq.com/s/XGnHruXL3P0s-2TNss0LIg">晚点对话 MiniMax 闫俊杰：千万别套用移动互联网的逻辑来做 AI</a>: “创业没有天选之子。”
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1329938101388312657)** (2 messages): 

> `Vagueposting, AI Moats, Amanda Askell` 


- **Vagueposting Reaches New Heights**: A member shared a graphic titled 'vagueposting end game', emphasizing the trend of ambiguous communication in online spaces. The attached image hints at the complexities of deciphering modern digital dialogue.
   - The visual representation of **vagueposting** urges viewers to consider the broader implications of unclear messaging, inviting further discussions.
- **Discussion on AI's Last Moat**: A member referenced a tweet claiming that 'the only **moat** left in AI is Amanda Askell', sparking conversations about competitive advantages in the field.
   - This statement reflects growing sentiments regarding **intellectual property** and **unique insights** in the rapidly evolving AI landscape.



**Link mentioned**: <a href="https://x.com/menhguin/status/1881387910316052723?s=61">Tweet from Minh Nhat Nguyen (@menhguin)</a>: the only moat left in AI is amanda askell

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1330916825596428429)** (6 messages): 

> `Reinforcement Learning for Robotics, Vision & Language Models, Computer Vision Reinforcement Learning, Robotics Perception Models` 


- **Exploring RLVR for Robotic Control**: A member questioned the applicability of **RLVR** for robotic control using **VLMs** and **CoT** to generate commands in the format 'move to (0.41, -7.8).'
   - Another member expressed optimism, stating that it seems like a method that could work well now.
- **Vintage Ideas Resurfacing**: Discussion highlighted that what is old often feels new again in robotics, especially regarding reinforcement learning.
   - Greater exploration of past ideas seems necessary as voting recommits to perennial concepts.
- **Computer Vision Applications of RL**: A member shared a paper by **Lucas Beyer et al.** discussing reinforcement learning techniques to align models with task rewards in computer vision, accessible [here](https://arxiv.org/abs/2302.08242).
   - The paper claims effectiveness in aligning models across tasks such as object detection and image captioning by addressing model misalignment.
- **Combining RL with CoT Approaches**: Curiosity was raised about how RL approaches could be merged with **Chain of Thought (CoT)** methodologies in the context of computer vision.
   - Concerns also surfaced regarding the reliability of computer vision labels as 'verified' for tasks using RL.
- **Perception Models Timeline Conundrum**: One member humorously suggested a six-month experimental timeline for revolutionizing robotics alongside expected perception model deliveries in Q4.
   - The quip hinted at the ambitious pursuit of innovative ideas while managing standard deliverables.



**Link mentioned**: <a href="https://arxiv.org/abs/2302.08242">Tuning computer vision models with task rewards</a>: Misalignment between model predictions and intended usage can be detrimental for the deployment of computer vision models. The issue is exacerbated when the task involves complex structured outputs, a...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329939648822247496)** (21 messages🔥): 

> `Post-Training for AI Applications, Challenges with Devin vs. Cursor, AI Researchers' Overestimation, Reinforcement Learning (RL) Discussions, SOP-Agents Framework` 


- **Exploring Post-Training Strategies**: A talk titled [How to approach post-training for AI applications](https://youtu.be/grpc-Wyy-Zg) presented insights during NeurIPs, focusing on effective strategies for AI development.
   - Participants agreed on the trap of diving straight into training models without proper groundwork.
- **Devin vs. Cursor: A Mixed Review**: A member shared their team's experience, stating that they had to abandon **Devin** for **Cursor** within a week due to dissatisfaction with **Devin**'s performance.
   - *Rumors suggest* the coding agent utilizes **gpt-4o**, which may not perform as well for coding tasks compared to alternatives like **Claude**.
- **AI Diffusion Speed Overestimation**: A discussion arose from a [Tyler Cowen interview](https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5) highlighting that AI researchers often *overestimate* how quickly technology diffuses.
   - Members voiced agreement with this insight, prompting thoughts on the reluctance of LLM-centric startups to explore alternative models.
- **Reinforcement Learning: A Growing Interest**: Members discussed the rising need to understand **Reinforcement Learning (RL)**, with one stating it's inevitable to learn about it in the coming weeks.
   - They expressed frustration at the lack of resources specifically addressing *RL for language models*.
- **Introduction of SOP-Agents Framework**: The introduction of the [SOP-Agents](https://arxiv.org/abs/2501.09316) framework aims to enhance planning capabilities for AI agents by using **Standard Operational Procedures**.
   - This novel framework is designed to address limitations in task completion by guiding AI agents through decision graphs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09316">SOP-Agent: Empower General Purpose AI Agent with Domain-Specific SOPs</a>: Despite significant advancements in general-purpose AI agents, several challenges still hinder their practical application in real-world scenarios. First, the limited planning capabilities of Large La...</li><li><a href="https://youtu.be/GT_sXIUJPUo?si=-DFvkz65FjdIGNu5."> - YouTube</a>: no description found</li><li><a href="https://youtu.be/grpc-Wyy-Zg">How to approach post-training for AI applications</a>: My talk during NeurIPs at Infer -- the Vancouver AI Engineering group: https://infervan.com/This was a fun one. I was trying to think of &quot;what to say&quot; to AI ...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1330579335245070437)** (13 messages🔥): 

> `RLHF Book Progress, Outcome Reward Models, CS329A Course Overview, Reward Modeling Techniques, Value Networks` 


- **RLHF Book Progress Evokes Anticipation**: Progress is being made on the [RLHF Book](https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models), particularly on the giant policy gradient page that is expected to be very useful.
   - There is hope to have Ross back on the podcast soon to discuss these topics in detail.
- **Outcome Reward Models Differentiated**: A member noted that outcome reward models (ORMs) are useful for situations where it's not feasible to programmatically score the outcome, likening it to using proxies in reinforcement learning.
   - ORMs assist in data filtering and can help the reinforcement learning process by providing probabilities of the right outcomes from each token.
- **CS329A Course Gets Exciting**: The CS329A graduate seminar course has posted lectures alongside an intriguing [course overview](https://cs329a.stanford.edu/#schedule), covering cutting-edge AI techniques.
   - Participants expressed excitement about discovering a new reading list filled with fascinating papers related to self-improvement for LLMs.
- **Reward Modeling Techniques Explored**: Reward modeling is crucial in the modern RLHF approach, measuring preferences through models like Bradley-Terry, as detailed in the [RLHF Book](https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models).
   - Members discussed how these models relate to aligning values in reinforcement learning and the significance of training algorithms.
- **Value Networks Offer Future Predictions**: A value network is utilized to predict future returns related to specific tokens, showcasing differing roles compared to ORMs in AI modeling.
   - Understanding these distinctions emphasizes the importance of selecting the right tools in reinforcement learning frameworks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cs329a.stanford.edu/#schedule">Stanford CS329A | Self-Improving AI Agents</a>: no description found</li><li><a href="https://rlhfbook.com/c/07-reward-models.html#outcome-reward-models">A Little Bit of Reinforcement Learning from Human Feedback</a>: The Reinforcement Learning from Human Feedback Book
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1330171044635475978)** (3 messages): 

> `Meta Glasses Integration, WhatsApp Bot Functionality` 


- **Integrating Meta & Rayban Glasses with WhatsApp**: A GitHub project titled [meta-glasses-gemini](https://github.com/josancamon19/meta-glasses-gemini) explores integration of **Meta + Rayban Glasses** with WhatsApp through a bot.
   - *This integration allows users to control glasses features effectively,* showcasing a potential for enhanced user interaction.
- **Community Reaction to the Integration Idea**: One member humorously commented on the integration idea, stating, *'Love this nonsense.'*
   - This comment reflects the playful skepticism within the community regarding unconventional tech integrations.



**Link mentioned**: <a href="https://github.com/josancamon19/meta-glasses-gemini">GitHub - josancamon19/meta-glasses-gemini: Meta + Rayban Glasses whatsapp bot integration</a>: Meta + Rayban Glasses whatsapp bot integration. Contribute to josancamon19/meta-glasses-gemini development by creating an account on GitHub.

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1331053678119096411)** (3 messages): 

> `Executive Order on AI, NAIRR Event` 


- **US President Rescinds Major AI Executive Order**: The US President has rescinded the previous administration’s major Executive Order on AI, known as **EO 14110**. This change prompts questions about the implications and future of AI regulations in the US.
   - *What did that one even do?* was a common query from members seeking clarity on the executive order's previous provisions.
- **Curiosity About Upcoming NAIRR Event**: A member expressed uncertainty about the NAIRR event they are invited to in February, wondering if it is still happening. This reflects a broader hesitation about event planning amid ongoing regulatory changes.



**Link mentioned**: <a href="https://x.com/cfgeek/status/1881494093215551954?s=61">Tweet from Charles Foster (@CFGeek)</a>: The US President has rescinded the previous administration’s major Executive Order on AI (EO 14110).

  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1331027328418975885)** (1 messages): 

> `Aider v0.72.0 Release, DeepSeek R1 Support, Kotlin Syntax Support, File Writing Enhancements, Bugfix Updates` 


- **Aider v0.72.0 rolls out with multiple new features**: The new **Aider v0.72.0** release introduces support for **DeepSeek R1** with shortcuts `--model r1` and `--model openrouter/deepseek/deepseek-r1`.
   - This release also boasts enhancements such as examples_as_sys_msg=True for **GPT-4o models**, improving benchmark scores.
- **Kotlin syntax gets spotlight**: New **Kotlin syntax support** has been added to the repo map by contributor **Paul Walker**.
   - This enhancement aims to enhance the usability of Kotlin within the current framework.
- **File writing improvements implemented**: The addition of `--line-endings` for file writing by **Titusz Pan** aims to improve formatting consistency.
   - This enhancement reflects a commitment to elevating code quality in file operations.
- **Multiple bugfixes enhance stability**: Recent bugfixes include a **permissions issue** in Docker images and fixes for **lint/test** errors during turn-taking.
   - Additionally, an **ASCII fallback for unicode errors** and a fix for integer indices in repomap calculations were implemented.
- **Aider takes a significant role in coding**: Interestingly, **Aider** contributed to **52% of the code** in this release, underscoring its growing capabilities.
   - This level of involvement indicates a commitment to continuous improvement and innovative enhancements.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1329913008050274315)** (334 messages🔥🔥): 

> `DeepSeek R1 performance, Aider benchmarks, Kimi k1.5 model, Data privacy in AI models, Local model usage` 


- **DeepSeek R1's Performance Compared to Other Models**: Users expressed mixed feelings about the performance of DeepSeek R1 in Aider, noting it makes several mistakes, particularly with simpler tasks.
   - Despite excitement for a cheaper alternative to OpenAI's o1, some found R1's output not up to expectations, leading to suggestions for pairing it with different editing models.
- **Aider Benchmarks and Model Selection**: The DeepSeek R1 model achieved 57% on the Aider coding leaderboard, sparking discussions about its performance relative to the o1 model and other competitors.
   - Opinions varied on whether R1's reliance on 'thinking' responses enhances reasoning compared to other models, with some users preferring to use simpler models for basic tasks.
- **Kimi k1.5 Outperforms Established Models**: The new Kimi k1.5 multi-modal model reportedly outperforms GPT-4o and Claude Sonnet 3.5 in several benchmarks, particularly in reasoning tasks.
   - Kimi k1.5 features include long context scaling up to 128k tokens, which may expand its applicability in generative tasks.
- **Data Privacy Concerns in AI**: Users discussed transparency in AI data usage, highlighting that while companies like DeepSeek openly state they utilize user data, others are less clear.
   - Concerns were raised about the trustworthiness of large corporations in handling user data and training models.
- **Local Model Experiences**: Individuals reported positive experiences using distilled models locally, with responses noted as more well-rounded during early interactions.
   - It was suggested that using R1 models locally can help in more complex scenarios by providing thoughtful reactions to logs without needing explicit instructions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier">Tweet from undefined</a>: no description found</li><li><a href="https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security)">Data privacy &amp; security - Fireworks AI Docs</a>: no description found</li><li><a href="https://docs.grit.io/tutorials/gritql">GritQL Tutorial</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model &amp; technical report🏆 MIT licensed: Distill &amp; commercialize freely!. Run DeepSeek R1 with API</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://medium.com/@hamzaennaffati98/cache-augmented-generation-cag-vs-retrieval-augmented-generation-rag-7b668e3a973b">Cache-Augmented Generation (CAG) vs. Retrieval-Augmented Generation (RAG)</a>: Which Approach Reigns Supreme?</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">Tweet from Kimi.ai (@Kimi_ai_)</a>: 🚀 Introducing Kimi k1.5 --- an o1-level multi-modal model-Sota short-CoT performance, outperforming GPT-4o and Claude Sonnet 3.5 on 📐AIME, 📐MATH-500, 💻 LiveCodeBench by a large margin (up to +550%...</li><li><a href="https://unsloth.ai/blog/deepseek-r1">Run Deepseek-R1 / R1 Zero</a>: DeepSeek&#x27;s latest R-1 model is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Learn how to run &amp; fine-tune the model.</li><li><a href="https://x.com/deepseek_ai/status/1881318135850213834">Tweet from DeepSeek (@deepseek_ai)</a>: 🔥 Bonus: Open-Source Distilled Models!🔬 Distilled from DeepSeek-R1, 6 small models fully open-sourced📏 32B & 70B models on par with OpenAI-o1-mini🤝 Empowering the open-source community🌍 Pushing t...</li><li><a href="https://www.youtube.com/@techfren">techfren</a>: Open Source AI and other technologySubscribe NOW to catch livestreams!</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt">CEO da OpenAI não sabe o que fazer com o comportamento dos assinantes do ChatGPT</a>: Ele escolheu o preço sem pensar muito e achou que ganharia dinheiro</li><li><a href="https://gist.github.com/murdockq/b08f72699fd7d8db556a14e69a7cb0c3">a game prompt.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://docs.fireworks.ai/guides/security_comp">Introduction - Fireworks AI Docs</a>: no description found</li><li><a href="https://x.com/0xluffyb/status/1881323971897110866">Tweet from luffy (@0xluffyb)</a>: everyone todayQuoting DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model & technical report🏆 MIT licensed: Distill & commercialize freely!🌐 ...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://docs.litellm.ai/docs/completion/function_call">Function Calling | liteLLM</a>: Checking if a model supports function calling</li><li><a href="https://www.youtube.com/@echohive">echohive</a>: I have spent over 3000 hours learning and coding over 300 projects and share everything I have learned in these YouTube videos. I hope you will find them useful :)search all echohive videos: https://w...</li><li><a href="https://www.youtube.com/@AllAboutAI">All About AI</a>: Welcome to my channel All About AI =)Website:https://aiswe.techHow you can start to use Generative AI to help you with creative or other daily tasks.So I aim to bring Generative AI to everyone.- AI En...</li><li><a href="https://www.youtube.com/@AIJasonZ">AI Jason</a>: My name is Jason Zhou, a product designer who share interesting AI experiments &amp; products. Email me if you need help building AI apps! - Join community: https://www.skool.com/ai-builder-club/about...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">deepseek-ai/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: no description found</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Aider in a ReAct loop</a>: Aider in a ReAct loop . Contribute to ai-christianson/RA.Aid development by creating an account on GitHub.</li><li><a href="https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#agents-for-developers:~:text=General%20availability%20will%20follow%20in%20January%2C%20along%20with%20more%20model%20sizes.">Introducing Gemini 2.0: our new AI model for the agentic era</a>: Today, we’re announcing Gemini 2.0, our most capable AI model yet.</li><li><a href="https://youtube.com/@marvijosoftware?si=CpNJZ8UmLJyp2mk">Marvijo AI Software</a>: We explore cutting edge Cloud, Software Engineering and Computer Science technologies. - Artificial Intelligence- LLMs (Large Language Models)- AI Coding- Microsoft Azure- Easy to understand explanati...</li><li><a href="https://youtube.com/@appydave?si=0nvFeqOcIZxuJCHA">AppyDave</a>: Welcome to AppyDave (formaly AppyCast), the hub where innovation meets conversation. My mission is to empower you with the knowledge and tools to harness the potential of ChatGPT, an AI that&#39;s red...</li><li><a href="https://youtube.com/@codingthefuture-jg1he?si=Eag1-kRT23z8Jys8">Coding the Future With AI</a>: Welcome to Coding the Future With AI! Our channel is dedicated to helping developers and tech enthusiasts learn how to leverage AI to enhance their skills and productivity. Through tutorials, expert i...</li><li><a href="https://github.com/Aider-AI/aider/issues/429">Tree-sitter tsx parser hangs sometimes, causing aider to hang · Issue #429 · Aider-AI/aider</a>: User reports aider hangs when using a repo full of .tsx files. Using --no-git removes the hang. Issue appears to be in the repo map code. https://discord.com/channels/1131200896827654144/1192136795...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329944218667454495)** (74 messages🔥🔥): 

> `Aider Usage with Language Models, OpenRouter vs Anthropic API, DeepSeek Model Issues, File Management in Aider, API Key Configuration Problems` 


- **Utilizing Aider for Coding with LLMs**: Users discussed the effectiveness of using Aider with models such as **DeepSeek v3** and **Qwen 2.5 Coder**, noting context window settings and performance expectations.
   - Several mentioned the need for the `/copy-context` command in **Architect mode** to maintain chat history for better responses.
- **Choosing Between OpenRouter and Anthropic API**: A user inquired about reasons for preferring **OpenRouter** over **Anthropic API**, leading to discussions on stricter limits imposed by Anthropic.
   - Others confirmed that **OpenRouter** typically offers more flexible API limits, making it a more popular choice among Aider users.
- **Issues with DeepSeek Model Responses**: Users reported errors related to **DeepSeek** not supporting successive user or assistant messages and intermittent performance issues in Aider.
   - Some users suggested updating Aider and checking model settings to resolve these errors.
- **File Management and Autocompletion in Aider**: There were discussions about the `/add` command not displaying possible files, with users expressing a desire for improved directory visibility.
   - It was noted that Aider autocompletes from files in the user's Git repository, which might limit visibility in certain contexts.
- **Troubles with API Key Configurations**: A user faced issues with invalid API keys despite working instances, reporting inconsistent behavior across different Aider projects.
   - It was mentioned that projects continue to function with older API configurations, highlighting a potential configuration or recognition issue within Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://aider.chat/docs/more/infinite-output.html">Infinite output</a>: Aider can handle “infinite output” from models that support prefill.</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider is AI pair programming in your terminal</li><li><a href="https://ollama.com/library/qwen2.5-coder">qwen2.5-coder</a>: The latest series of Code-Specific Qwen models, with significant improvements in code generation, code reasoning, and code fixing.</li><li><a href="https://www.youtube.com/live/vUbPnNeN9eY?si=3hqiicuNpeH6UCgM&t=1537">Aider + 1 Hour = Reels video editor with face recognition</a>: Join techfren as he uses his software engineering expertise to try and review new technology</li><li><a href="https://docs.litellm.ai/docs/providers/anthropic">Anthropic | liteLLM</a>: LiteLLM supports all anthropic models.</li><li><a href="https://github.com/BuilderIO/gpt-crawler">GitHub - BuilderIO/gpt-crawler: Crawl a site to generate knowledge files to create your own custom GPT from a URL</a>: Crawl a site to generate knowledge files to create your own custom GPT from a URL - BuilderIO/gpt-crawler</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler &amp; Scraper</a>: 🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler &amp; Scraper - unclecode/crawl4ai
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1331002241028591676)** (1 messages): 

> `Bolt.new update, Setup issues, Prompt accuracy` 


- **Bolt.new assures smooth setup**: The latest update to *bolt.new* ensures that users will no longer face issues resulting in a **white screen** or a broken setup from the first prompt, as it now more accurately picks and configures the right template every time.
   - This enhancement addresses previous user frustrations and improves the initial setup experience, allowing for a **spot on** start for all users.
- **Improved accuracy of prompt configurations**: With the recent update, *bolt.new* achieves a significant enhancement in accuracy for selecting templates, promising users a hassle-free setup right from their initial prompts.
   - As a result, this leads to less confusion and smoother interactions during the setup process, ensuring templates are correctly configured without issues.



**Link mentioned**: <a href="https://x.com/boltdotnew/status/1881442318110347291">Tweet from bolt.new (@boltdotnew)</a>: Bolt 🧠 update:bolt․new is now more accurate at picking & configuring the right template — making the setup spot on, from the first prompt, every time!

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1329903179764072479)** (367 messages🔥🔥): 

> `Bolt error loops, RLS policy issues, Stripe integration, Payment processing options, Community support and resources` 


- **Frustrations with Bolt's error loops**: Users expressed frustration over Bolt entering continuous error loops, leading to significant token consumption without resolving issues, particularly with complex functionalities like user permissions.
   - One user highlighted their experience of exhausting nearly 30 million tokens due to persistent issues and concluded they must start over to avoid the pitfalls encountered.
- **Row-Level Security (RLS) Policy Challenges**: Several users reported encountering repeated RLS violations while working with Supabase, complicating their ability to implement booking functionalities effectively.
   - One user suggested using external documentation and examples to streamline the RLS policy creation process, significantly reducing recursive errors.
- **Payment Integration Strategies**: Discussion emerged about payment integration for services like car detailing, with suggestions leaning towards simpler solutions like using PayPal buttons rather than complex setups in Bolt.
   - Given the user's non-developer background, alternatives like WordPress with form builder plugins were recommended as more user-friendly options.
- **Expectations about Token Usage**: Potential users inquired about token usage under the Pro plan, learning that token consumption varies with user proficiency and can depend on enabling features like diffs in Bolt.
   - Users were reassured that, unlike the free plan, they would not face daily limitations on token usage with the Pro plan.
- **Community Support and Learning**: Users shared tips on how to navigate Bolt more effectively, including utilizing resources like ChatGPT and Claude for assistance with coding problems and documentation.
   - The importance of community support and knowledge sharing was emphasized, with users encouraging collaboration to enhance their development experience on the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://boltsync.mystify.tech/">BoltSync - GitHub Repository Management with Bolt</a>: Modify your GitHub repositories with Bolt Prompts &amp; sync changes back to GitHub with BoltSync. Streamline your development workflow with AI-powered repository management.</li><li><a href="https://cardspark.app/)">CardSpark - Perfect Messages in Seconds</a>: no description found</li><li><a href="https://support.bolt.new/Tokens-13fd971055d6804ea762d2fafdc3ad98">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://x.com/stackblitz/status/1843668731">Tweet from Charles Beck (@charlesbeck)</a>: @caughtintheweb bout to email you that beat</li><li><a href="https://gabe.marketing/domains">gabe.marketing - your professional AI non technical dev nerd</a>: Marketing expert, podcast producer, and AI-powered app developer helping businesses grow through innovative digital solutions.</li><li><a href="https://copycoder.ai/">CopyCoder</a>: no description found</li><li><a href="https://boltdiyhosting.com/">Bolt.DIY Managed Hosting - Professional Cloud Platform for Developers</a>: no description found</li><li><a href="https://abea.pics/KreLodCmLxEnZ21">Abea</a>: no description found</li><li><a href="https://x.com/stackblitz/status/1843668731681267801?s=46">Tweet from bolt.new (@boltdotnew)</a>: You can now open public repos in bolt․new 🙌How? For any GitHub URL, just put &#34;http://bolt.new&#34; in front of it!(Release notes below!)</li><li><a href="https://prnt.sc/CVZgu1OObu9G">Screenshot</a>: Captured with Lightshot</li><li><a href="https://www.creative-tim.com/learning-lab/bootstrap/grid/soft-ui-design-system">Grid | Soft UI Design System Bootstrap @ Creative Tim</a>: Our Bootstrap grid is a powerful mobile-first flexbox grid which helps you build layouts of all shapes and sizes thanks to a twelve column system, five default responsive tiers, Sass variables and mix...</li><li><a href="https://resend.com/">Resend · Email for developers · Resend</a>: Build, test, and send transactional and marketing emails at scale.</li><li><a href="https://21st.dev/">21st.dev - The NPM for Design Engineers</a>: Ship polished UIs faster with ready-to-use React Tailwind components inspired by shadcn/ui. Built by design engineers, for design engineers.</li><li><a href="https://bolters.io">Bolters.io | Community Supported Tips, Tricks &#38; Knowledgebase for Bolt.new No-Code App Builder</a>: Documentation and guides for Bolt.new</li><li><a href="https://supabase.com/docs/guides/functions/examples/stripe-webhooks">Handling Stripe Webhooks | Supabase Docs</a>: Handling signed Stripe Webhooks with Edge Functions.</li><li><a href="https://github.com/supabase/supabase/blob/master/examples/edge-functions/supabase/functions/stripe-webhooks/index.ts">supabase/examples/edge-functions/supabase/functions/stripe-webhooks/index.ts at master · supabase/supabase</a>: The open source Firebase alternative. Supabase gives you a dedicated Postgres database to build your web, mobile, and AI applications. - supabase/supabase
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1330967333770100746)** (1 messages): 

> `LM Studio 0.3.7 Release, DeepSeek R1 Support, New Features in Mission Control, KV Cache Quantization Updates` 


- **LM Studio 0.3.7 Launch with Exciting Features**: The release of **LM Studio 0.3.7** introduces support for **DeepSeek R1** and an updated **llama.cpp engine** version 1.9.2, accessible via [in-app updates](https://lmstudio.ai).
   - Users can also download various distilled models from **DeepSeek**, offering sizes up to **70B**, designed to enhance performance.
- **DeepSeek R1: A Game Changer in Reasoning Models**: The **DeepSeek R1 model** is now available for download, promising open source reasoning capabilities on par with OpenAI's **o1** model, with details found in the [technical report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf).
   - Users will notice outputs from **DeepSeek R1** encapsulated in `<think>` tags, showcasing its reasoning processes.
- **Enhanced Mission Control Features**: A **Hardware tab** has been added to Mission Control, which can be accessed using `Cmd/Ctrl + Shift + H`, offering users more monitoring capabilities.
   - Additionally, a **server file logging mode** allows for more granular control over what log entries are made.
- **KV Cache Quantization for Improved Performance**: The latest version comes with **KV Cache quantization** for **llama.cpp models**, enhancing the efficiency of the runtime environment requiring version **1.9.0+**.
   - This feature aims to optimize performance metrics while handling model predictions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.7">LM Studio 0.3.7</a>: DeepSeek R1 support and KV Cache quantization for llama.cpp models
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1329937844751564821)** (179 messages🔥🔥): 

> `Model Performance Comparisons, File Attachment in LM Studio, DeepSeek R1 Model Discussion, Using Multiple Images with Models, LM Studio Updates and Features` 


- **DeepSeek R1 vs Llama Models**: Users discussed the capabilities of the DeepSeek R1 model and how it compares to the Llama models, noting that the Qwen 32B often ranks better despite being smaller than the Llama 70B.
   - Some users highlighted that while R1 is visually cluttered, it can provide good answers, although its reasoning appears less confident.
- **File Attachment Functionality in LM Studio**: Questions arose regarding the file attachment feature in LM Studio, specifically about whether the uploading process affects local files or sends data elsewhere.
   - It was clarified that uploaded files remain local to the user's machine and are used for context during interactions with LLMs.
- **Issues with Model Responses and Reasoning**: Some users expressed concerns about the randomness and repetitiveness of responses from the DeepSeek R1 model, particularly when trying to generate lists or extend responses.
   - Users indicated that R1's memory lacks effectiveness, resulting in repeated outputs rather than logically extended lists.
- **Updates and Enhancements in LM Studio**: Discussion included the recent updates to LM Studio, where users were encouraged to utilize the new versions of the llama.cpp engine to enhance model performance.
   - Users noted the need for visual improvements in the display of thinking outputs to avoid cluttered interfaces during interactions.
- **Distributed Inference with M2 Ultras**: There was a discussion on using distributed inference with networked M2 Ultra machines, with some users skeptical about the practicality versus cost.
   - Intel users confirmed that while distributed support is available, performance is heavily dependent on network bandwidth and system configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/seo_leaders/status/1881462202831614085">Tweet from Andrew C (@seo_leaders)</a>: DeepSeek R1 671B running on 2 M2 Ultras quicker than reading speed.  Almost an open-source O1, at home, on consumer hardware.  With mlx.distributed and mlx-lm, 3-bit quantization (~4 bpw). Model is qu...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/11310/commits.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents - Running LLMs Locally | LM Studio Docs</a>: How to provide local documents to an LLM as additional context</li><li><a href="https://lmstudio.ai/docs/ba">Getting Started | LM Studio Docs</a>: Learn how to run Llama, Mistral, Gemma, and other LLMs locally with LM Studio.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1329927231090200697)** (186 messages🔥🔥): 

> `NVIDIA Digits, GPU Comparisons, Quality of Model Performance, LM Studio vs Ollama, Kaggle Notebooks` 


- **NVIDIA Digits as an AI/ML Server**: Members expressed enthusiasm for NVIDIA Digits as a home ML server, emphasizing its capability to perform dedicated machine learning tasks.
   - Although it is not a typical gaming PC, its focus on high memory usage aligns well with specific AI applications.
- **Comparing GPUs for AI Tasks**: There was a discussion comparing the performance of high-end GPUs like the 4090/5090 against cheaper alternatives for AI tasks.
   - While a $200 GPU would suffice for gaming, participants noted that dedicated AI tasks would benefit significantly from more powerful cards.
- **Quality Variations in Model Performance**: Users reported noticeable differences in model performance between LM Studio and Ollama, especially with Qwen2.5 models.
   - Testing indicated that LM Studio provided better quality results when used with specific setups compared to Ollama.
- **Running Non-LLM PyTorch Tasks**: Participants discussed whether NVIDIA Digits could handle non-LLM PyTorch tasks, with some cautioning about its performance limitations.
   - While it can be used for such tasks, it may not perform as well as using a more capable GPU.
- **Experiments with Kaggle Notebooks**: One user expressed concern about the ability to use NVIDIA Digits quickly enough for experimenting with Kaggle notebooks.
   - The conversation highlighted the balance needed between hardware capabilities and the requirements of various machine learning tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Graphics_processing_unit#Integrated_graphics">Graphics processing unit - Wikipedia</a>: no description found</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889">NVIDIA GeForce RTX 4090 Specs</a>: NVIDIA AD102, 2520 MHz, 16384 Cores, 512 TMUs, 176 ROPs, 24576 MB GDDR6X, 1313 MHz, 384 bit
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1329924464372092950)** (97 messages🔥🔥): 

> `DeepSeek R1 Release, Transcription Tools, OpenAI Operator Leaks, Liquid Foundation Model, Claude AI Alignment Perspectives` 


- **DeepSeek R1: A Game Changer**: The [DeepSeek R1 release](https://x.com/deepseek_ai/status/1881318138937233664?s=46) announced models achieving performance on par with OpenAI's o1 across multiple benchmarks, enabling open-source access under the MIT license.
   - Users are excited about the model's capabilities, including a [distilled version](https://x.com/reach_vb/status/1881330709929013440?s=46) outperforming larger models like GPT-4o in specific tasks.
- **Exploring Transcription Tools**: Members discussed various transcription tools, with many recommending [MacWhisper](https://github.com/deepseek-ai/DeepSeek-R1) for its performance, while others expressed interest in new features from apps like Alter.
   - The community is exploring alternatives to existing tools like Wispr Flow that have faced hiccups, seeking better dictation solutions.
- **OpenAI's Operator Leaks**: Recent leaks suggest that OpenAI's new Computer Use Agent (CUA) has comparisons with other models like Claude 3.5, hinting at an imminent release.
   - Members are intrigued by these developments and are closely following updates surrounding the [Operator system](https://x.com/kimmonismus/status/1881287794544550018?s=46).
- **Liquid Foundation Model Announcement**: Liquid AI introduced the [LFM-7B model](https://www.liquid.ai/lfm-7b), claiming it to be the best-performing in its class with a unique non-transformer architecture.
   - They emphasize its multilingual capabilities and low memory footprint, making it suitable for enterprises with deployment needs.
- **Claude AI and Alignment Discussion**: A post shared about [Claude AI](https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x) sparked conversations around AI alignment and its implications.
   - Members find it interesting to critique how such advanced models are described, particularly referencing terms like 'shoggoth' in the context of AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1880356297985638649">Tweet from Sam Altman (@sama)</a>: thank you to the external safety researchers who tested o3-mini.we have now finalized a version and are beginning the release process; planning to ship in ~a couple of weeks.also, we heard the feedbac...</li><li><a href="https://x.com/StringChaos/status/1880317308515897761">Tweet from Naman Jain (@StringChaos)</a>: DeepSeek-R1 (Preview) Results 🔥We worked with the @deepseek_ai team to evaluate R1 Preview models on LiveCodeBench. The model performs in the vicinity of o1-Medium providing SOTA reasoning performanc...</li><li><a href="https://x.com/teortaxesTex/status/1881331287010550119">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: perhaps the craziest thing is that they say that this is nowhere near the ceiling of 7-70B class models. Without any new data even. They have pushed them further, but just won&#39;t be sharing it. Dri...</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model & technical report🏆 MIT licensed: Distill & commercialize freely!🌐 Website & API are live now! Try DeepThink at h...</li><li><a href="https://x.com/rm_rafailov/status/1881350883252085000">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: DeepSeek R1 with &#34;Cold Start&#34; pretty much works as expected. I still don&#39;t buy the R1 Zero result, the base models barely output coherent solutions without finagling. My bet is there is so...</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384">Tweet from Windsurf (@windsurf_ai)</a>: Wave 2 is here. Included in this update: 🌐Web Search🧠Autogenerated Memories💼Enterprise Ready... and many more!</li><li><a href="https://x.com/eliebakouch/status/1881234710841700541?s=46">Tweet from elie (@eliebakouch)</a>: holy shit, @deepseek_ai R1 is up guys</li><li><a href="https://x.com/1a3orn/status/1881350809230991382">Tweet from 1a3orn (@1a3orn)</a>: This was the sentence in the DeepSeek paper I had to read 3 times to make sure I wasn&#39;t hallucinating.R1 distilled into Qwen 1.5b beats Sonnet and GPT-4o on some math benchmarks.</li><li><a href="https://x.com/abacaj/status/1881342078506139881">Tweet from anton (@abacaj)</a>: This was posted in September (on codeforces)… now there is a 32B model distilled from r1 scoring 1600+ that you can run at home… wowQuoting DeepSeek (@deepseek_ai) 🔥 Bonus: Open-Source Distilled Mode...</li><li><a href="https://x.com/fchollet/status/1880378880894333094?s=46">Tweet from François Chollet (@fchollet)</a>: OpenAI releases under the *same name* models that work completely differently -- some where most of the lifting is done by an LLM, others where most of the lifting is in test-time CoT search. This can...</li><li><a href="https://x.com/lu_sichu/status/1881348105586855962">Tweet from Sichu Lu(Sichu.Lu218@proton.me) (@lu_sichu)</a>: Probably the most interesting part where they explain why mcts is hard in token space</li><li><a href="https://x.com/srush_nlp/status/1881382753557754103)">Tweet from Sasha Rush (@srush_nlp)</a>: Post-mortem after Deepseek-r1&#39;s killer open o1 replication.We had speculated 4 different possibilities of increasing difficulty (G&C, PRM, MCTS, LtS). The answer is the best one! It&#39;s just Gue...</li><li><a href="https://x.com/btibor91/status/1881285255266750564?s=46">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI website already has references to Operator/OpenAI CUA (Computer Use Agent) - &#34;Operator System Card Table&#34;, &#34;Operator Research Eval Table&#34; and &#34;Operator Refusal Rate Table&#3...</li><li><a href="https://x.com/cloneofsimo/status/1881389467346547101">Tweet from Simo Ryu (@cloneofsimo)</a>: Even bitter-er-er lesson is that the more intellectually interesting something is less useful it isYou would see batshit post-training papers but REINFORCE or PPO is just betterYou would see weird ass...</li><li><a href="https://x.com/ollama/status/1881427522002506009">Tweet from ollama (@ollama)</a>: DeepSeek&#39;s first-generation reasoning models are achieving performance comparable to OpenAI&#39;s o1 across math, code, and reasoning tasks! Give it a try! 👇7B distilled: ollama run deepseek-r1:7...</li><li><a href="https://x.com/nrehiew_/status/1880853579671699709?s=46">Tweet from wh (@nrehiew_)</a>: It looks like OpenAI had access to FrontierMath (the super challenging math benchmark that o3 sets SOTA by a mile) problems and solutions. They also did not allow the team behind FrontierMath to discl...</li><li><a href="https://x.com/_xjdr/status/1881365349356147057">Tweet from xjdr (@_xjdr)</a>: After reading the paper, I still don&#39;t see the TTC lever to make the o1 log plot a reality. There may be one that I am just missing (it&#39;s early on the West coast) but If there isn&#39;t one, I...</li><li><a href="https://x.com/kimmonismus/status/1881287794544550018?s=46">Tweet from Chubby♨️ (@kimmonismus)</a>: OpenAI already has a comparison (Computer Use Agent) between OpenAI&#39;s Operator and Claude 3.5 Sonnet CUA.Looks like release is imminent.Quoting Tibor Blaho (@btibor91) OpenAI website already has r...</li><li><a href="https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x">Colin (@colin-fraser.net)</a>: Here&#39;s why &quot;alignment research&quot; when it comes to LLMs is a big mess, as I see it.Claude is not a real guy. Claude is a character in the stories that an LLM has been programmed to write. ...</li><li><a href="https://x.com/casper_hansen_/status/1881404604518392144">Tweet from Casper Hansen (@casper_hansen_)</a>: The DeepSeek R1 training procedure confused me at first. My brain refused to accept this powerful model could be incredibly straightforward.Let me break down this elegant beast for you 🧵</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>: CodeGuide creates Detailed Documentation for your AI Coding Project.</li><li><a href="https://x.com/simonw/status/1881361661975843202">Tweet from Simon Willison (@simonw)</a>: DeepSeek released a whole family of inference-scaling / &#34;reasoning&#34; models today, including distilled variants based on Llama and QwenHere are my notes on the new models, plus how I ran DeepSe...</li><li><a href="https://x.com/mrsiipa/status/1881330071874813963">Tweet from maharshi (@mrsiipa)</a>: deepseek R1 thinks for around 75 seconds and successfully solves this cipher text problem from openai&#39;s o1 blog post.</li><li><a href="https://x.com/christiancooper/status/1881335734256492605">Tweet from Christian H. Cooper (@christiancooper)</a>: I asked #R1 to visually explain to me the Pythagorean theorem. This was done in one shot with no errors in less than 30 seconds. Wrap it up, its over: #DeepSeek #R1</li><li><a href="https://x.com/teortaxesTex/status/1881317131561922640">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: somebody wake up Yud</li><li><a href="https://x.com/paul_cal/status/1881324020592963939">Tweet from Paul Calcraft (@paul_cal)</a>: In-context back-tracking was emergent in R1. Bitter lesson adjacent. I thought this was plausibleWonder if the whole o1 paradigm started out as heavy RL on 4o for reasoning tasks, without a particular...</li><li><a href="https://x.com/teortaxestex/status/1881295618192077099?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: I conclude that R1, and by implication V3-base, were completed before December 10. It seems that V2.5-1210 is this ablation experiment in V3 paper (check LCB and MATH).Whale never trains models as mer...</li><li><a href="https://x.com/DimitrisPapail/status/1881341537499619822">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: The most interesting bits of information:- no PRM or step-by step-verifier needed- PPO on {question, answer_i} pairs; using an advantage function based on accuracy of final answer and format. - RL-tun...</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: The world’s best-in-class English, Arabic, and Japanese model, native in French, German, and Spanish, optimized to be the substrate for private enterprise chat, code, fast instruction following, and a...</li><li><a href="https://x.com/stochasticchasm/status/1881324856253497515">Tweet from stochasm (@stochasticchasm)</a>: Also @aidan_mclau completely vindicated with the problem with reasoners essay, now that we know r1 is literally only verifiable rewards</li><li><a href="https://skylarbpayne.com/posts/cursed-cursor">How to stop saying 'Fuck you Cursor' - Skylar Payne (Wicked Data LLC)</a>: no description found</li><li><a href="https://x.com/samjulien/status/1880405699697762565?s=46">Tweet from Sam Julien (@samjulien)</a>: This is wicked cool. You can now chat directly with the @Get_Writer docs inside of @windsurf_ai by @codeiumdev! Here I am asking it about tool calling in Palmyra X 004 and it&#39;s pulling the guide I...</li><li><a href="https://x.com/deepseek_ai/status/1881318138937233664?s=46">Tweet from DeepSeek (@deepseek_ai)</a>: 📜 License Update!🔄 DeepSeek-R1 is now MIT licensed for clear open access🔓 Open for the community to leverage model weights & outputs🛠️ API outputs can now be used for fine-tuning & distillation🐋 ...</li><li><a href="https://x.com/reach_vb/status/1881315419086291213">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: holy fuck, these gigachads dropped 6 distilled models right from 1.5B to 70B 🔥</li><li><a href="https://x.com/qtnx_/status/1881330757001502991">Tweet from Q (@qtnx_)</a>: this is the most surprising thing from the r1 paper to me:&gt;reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patter...</li><li><a href="https://x.com/natolambert/status/1881356292943487337">Tweet from Nathan Lambert (@natolambert)</a>: R1 making me feel very heard. Will read more thoroughly later.Laughs in continued shock that RL working like this.</li><li><a href="https://x.com/nrehiew_/status/1881330794549182853">Tweet from wh (@nrehiew_)</a>: This figure right here is the single biggest reason why we will never get o1/o3&#39;s reasoning traces.The unreasonable effectiveness of distilling from a reasoner</li><li><a href="https://x.com/lmarena_ai/status/1881411458678014215">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: DeepSeek-R1 is now in the Arena🔥Congrats @deepseek_ai on R1 release! An open reasoning model matching OpenAI o1 in hard benchmarks like GPQA/SWE-Bench/AIME!Now for the real-world challenge—R1 is in h...</li><li><a href="https://x.com/kimmonismus/status/1880328198438940923?s=46">Tweet from Chubby♨️ (@kimmonismus)</a>: „OpenAI’s new model, called GPT-4b micro, was trained to suggest ways to re-engineer the protein factors to increase their function. According to OpenAI, researchers used the model’s suggestions to ch...</li><li><a href="https://llmstxt.site/">llms.txt directory</a>: Find and explore llms.txt files from various products and services.</li><li><a href="https://x.com/Grad62304977/status/1881330709929013440">Tweet from Grad (@Grad62304977)</a>: No MCTS, no PRM, emergent behaviour, simple rl</li><li><a href="https://x.com/sama/status/1880358749187240274">Tweet from Sam Altman (@sama)</a>: @kimmonismus @flowersslop 1 and 2, figuring that out but i think you&#39;ll be happy.3, i would love for us to be able to merge the GPT series and the o series in 2025! let&#39;s see.</li><li><a href="https://x.com/carinalhong/status/1880820323597357273?s=46">Tweet from Carina Hong (@CarinaLHong)</a>: 1. OAI binds Epoch to an NDA until eve of o3 performance claim, preventing Epoch to disclose OAI is the donor and that OAI has exclusive data access2. Mathematicians then sign NDA on the problem & sol...</li><li><a href="https://x.com/sama/status/1880360141218017656">Tweet from Sam Altman (@sama)</a>: @mckaywrigley worse than o1 pro at most things(but FAST)</li><li><a href="https://x.com/deepseek_ai/status/1881318145761439995?s=46">Tweet from DeepSeek (@deepseek_ai)</a>: 🌐 API Access & Pricing⚙️ Use DeepSeek-R1 by setting model=deepseek-reasoner💰 $0.14 / million input tokens (cache hit)💰 $0.55 / million input tokens (cache miss)💰 $2.19 / million output tokens📖 AP...</li><li><a href="https://gist.github.com/morganmcg1/eb0626c7801f3ffbc780ef48269b87ea">AI/Cursor Task Design Document, from this Cursor blog: https://skylarbpayne.com/posts/cursed-cursor#design-document</a>: AI/Cursor Task Design Document, from this Cursor blog: https://skylarbpayne.com/posts/cursed-cursor#design-document - ai_design_template.md</li><li><a href="https://x.com/reach_vb/status/1881319500089634954">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: &#34;DeepSeek-R1-Distill-Qwen-1.5B outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks with 28.9% on AIME and 83.9% on MATH.&#34;1.5B did WHAT?</li><li><a href="https://x.com/sama/status/1880366903107154134">Tweet from Sam Altman (@sama)</a>: @TarasBob @mckaywrigley o3 is much smarter; we are turning our attention to that now.(and o3 pro?! 🤯)</li><li><a href="https://techcrunch.com/2025/01/19/the-pentagon-says-ai-is-speeding-up-its-kill-chain">The Pentagon says AI is speeding up its &#039;kill chain&#039; | TechCrunch</a>: Leading AI developers, such as OpenAI and Anthropic, are threading a delicate needle to sell software to the United States military: make the Pentagon</li><li><a href="https://x.com/sama/status/1880388642172203226">Tweet from Sam Altman (@sama)</a>: @sporadicalia @TarasBob @mckaywrigley no, you&#39;ll get it for 200</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>: Contribute to MoonshotAI/Kimi-k1.5 development by creating an account on GitHub.</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">Tweet from Kimi.ai (@Kimi_ai_)</a>: 🚀 Introducing Kimi k1.5 --- an o1-level multi-modal model-Sota short-CoT performance, outperforming GPT-4o and Claude Sonnet 3.5 on 📐AIME, 📐MATH-500, 💻 LiveCodeBench by a large margin (up to +550%...</li><li><a href="https://list.alterhq.com/p/just-say-it-transform-any-text-with-voice-commands">🗣️ Just say it — Transform any text with voice commands</a>: Transform any text with voice commands across all your apps. Select, speak, done.</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1329951534003261480)** (4 messages): 

> `O1 podcast discussion, DeepSeek v3, SGLang framework, Mission Critical Inference, Kubernetes challenges` 


- **Follow-up on O1 Podcast**: The @latentspacepod released a follow-up podcast on the [O1 skill issue](https://youtu.be/NkHcSpOOC60) featuring insights from Ben, who described O1 as 'mind-blowing' when used correctly.
   - Ben emphasized that **O1** should be viewed as a 'report generator' rather than a chat model, highlighting its unique functionalities.
- **Exciting Features of DeepSeek v3**: The latest podcast explores the **DeepSeek v3** and the upcoming release of **SGLang**, discussing essential specifications and achievements in the field.
   - Listeners can dive into topics including [model performance](https://www.latent.space/p/baseten) and the critical aspects of Mission Critical Inference.
- **Diving into Mission Critical Inference**: Special guests discussed the '**Three Pillars of Mission Critical Inference**', detailing technical insights and optimizations relevant to **DeepSeek**.
   - The episode covers vital strategies for scaling workloads beyond single GPU limitations while addressing infrastructure challenges like **Kubernetes**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1880394100857532521">Tweet from Latent.Space (@latentspacepod)</a>: After the success of the o1 skill issue post, we recorded a quick podcast with Ben and @daniel_mac8 . Live now!(link below for the algo gods)Quoting ben (@benhylak) o1 is mind-blowing when you know ho...</li><li><a href="https://x.com/latentspacepod/status/1880829933259559002?s=46">Tweet from Latent.Space (@latentspacepod)</a>: 🆕: Everything you need to run Mission Critical Inference (ft. DeepSeek v3 + SGLang)https://www.latent.space/p/basetenWe chat with @amiruci and @yinengzhang about the Chinese Whale Bro drop of 2024:- ...</li><li><a href="https://youtu.be/NkHcSpOOC60"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1329918103127064634)** (220 messages🔥🔥): 

> `AI tooling for accessibility, MCP server framework, Whisper for STT, YouTube captions, Live captions in Windows 11` 


- **AI Tooling Enhances Accessibility**: Participants discussed advancements in AI tooling for accessibility, highlighting tools like progress in live captioning on platforms such as Windows 11 and YouTube.
   - They noted that automatic captions have improved, making technical talks more accessible, though some still prefer human captioning for accuracy.
- **MCP Framework Sharing**: There was enthusiasm for an upcoming session focused on sharing experiences with a new MCP server framework, with members expressing interest in scheduling future demonstrations.
   - Participants discussed the potential benefits of using collaborative tools like spreadsheets to organize topics and facilitate knowledge sharing.
- **Whisper for Speech-to-Text Processing**: Whisper received praise for its effectiveness in non-real-time speech-to-text applications, though some participants expressed interest in exploring live applications of Whisper for meetings.
   - Discussion highlighted variations in performance depending on device specifications and the potential need for GPU utilization.
- **Voice-to-Text Technology Insights**: The conversation included insights on various voice-to-text technologies, detailing personal experiences and preferences among different platforms like Drafts and Whisper Memos.
   - Participants shared thoughts on overcoming challenges with automatic transcriptions, particularly in relation to non-standard accents.
- **Importance of Real-Time Captioning**: Real-time captioning capabilities were a significant focus, with participants noting improvements in tools like Windows 11's live captions for better accessibility.
   - Discussions emphasized the ongoing challenges and benefits of integrating various technologies to enhance communication experiences for individuals with hearing impairment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.samjulien.com/">Sam Julien: Director of DevRel and Your Biggest Fan</a>: Sam Julien is a Developer Relations director, writer, and teacher. He loves helping people level up their developer advocacy, AI engineering, or web development job using Python and JavaScript.</li><li><a href="https://brilliant-tapioca-5ad42f.netlify.app/">⚡️ Bolt.new + Vite + React</a>: no description found</li><li><a href="https://www.marblism.com/">Marblism - AI Agents that work for you</a>: The Leading Platform to Build and Deploy AI Agents. Automate workflows across industries with powerful AI agents.</li><li><a href="https://getdrafts.com/">Drafts, Where Text Starts</a>: Drafts quick-capture app for iPhone, iPad, Mac and Apple Watch</li><li><a href="https://github.com/samjulien/discord-scraper">GitHub - samjulien/discord-scraper</a>: Contribute to samjulien/discord-scraper development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://superwhisper.com/">superwhisper</a>: AI powered voice to text for macOS</li><li><a href="https://whispermemos.com/">Whisper Memos</a>: Whisper Memos transcribes your iOS voice memos and sends you an email with the transcription a few minutes later. It is based on OpenAI's new Whisper technology.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1330916117916553257)** (4 messages): 

> `DeepSeek R1 Launch, Performance Comparison with OpenAI, Censorship-Free Access, Llama Endpoints Shutdown` 


- **DeepSeek R1 Launches on OpenRouter**: The **DeepSeek R1** model is now live on [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1), boasting performance comparable to **OpenAI's o1** model.
   - With **transparent thinking tokens**, it is priced at **$0.55** per input token, which is just **4%** of the cost of OpenAI's equivalent.
- **Censorship-Free DeepSeek R1**: Users can access **DeepSeek R1** censorship-free on [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737), as noted by community discussions.
   - Despite being a censored model, users believe that fine-tuning by experts could enhance performance.
- **Free Llama Endpoints Discontinued**: A notice was shared that the **free Llama endpoints** will be going away at the end of the month due to changes from the provider, **Samba Nova**.
   - Samba Nova will transition to the **Standard variant**, which will come with a price increase.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/xanderatallah/status/1881456463786512737">Tweet from Alex Atallah (@xanderatallah)</a>: Note that you can use DeepSeek R1 censorship-free on @OpenRouterAI:Quoting MatthewBerman (@MatthewBerman) DeepSeek R1 doing what @shaunralston expected. At the end of the day, it&#39;s still a censore...</li><li><a href="https://x.com/openrouterai/status/1881407719170797741?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: DeepSeek R1 is now live on OpenRouter!⚡ Performance on par with OpenAI o1🧠 Transparent thinking tokens🍕$0.55/M input tokens, hosted by @deepseek_ai That’s 4% of the price of o1Quoting Risphere (@ris...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model &amp; technical report🏆 MIT licensed: Distill &amp; commercialize freely!. Run DeepSeek R1 with API
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1329904965262774336)** (258 messages🔥🔥): 

> `DeepSeek R1 Launch, OpenAI Model Rate Limits, User Experience with DeepSeek, Web Search API in OpenRouter, Reasoning Content Access` 


- **DeepSeek R1 is Live!**: DeepSeek announced the launch of R1, which reportedly performs on par with OpenAI's models and is fully open-source, licensed under MIT.
   - Users expressed excitement about its capabilities, especially in creative tasks like video content generation and calculus.
- **OpenAI Model Rate Limits Explained**: Users sought clarification on rate limits for Gemini 2.0 through OpenRouter, with confirmations that paid models have no limits, while free models are capped at 200 requests per day.
   - It was noted that users can add their rate limit settings by connecting their API keys.
- **User Feedback on DeepSeek**: Several users shared their initial experiences with DeepSeek R1, reporting it as a strong tool for various applications, although some expressed frustration with API limitations.
   - There were discussions about potential adjustments to improve access to reasoning content from the API.
- **Web Search API Availability**: Inquiries arose regarding the availability of the Web Search API, with confirmation that it is currently only accessible through the chatroom interface.
   - Users expressed interest in a beta option for expanding its integration capabilities.
- **Accessing Reasoning Content with DeepSeek**: Questions were raised about obtaining `reasoning_content` from the DeepSeek API, with responses indicating that OpenRouter needs to implement support for it.
   - The community is eager for updates on this feature as it could enhance the model's usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ExaAILabs">Tweet from undefined</a>: no description found</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>: Transform data for model consumption</li><li><a href="https://openrouter.ai/docs/errors).">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://x.com/xiaoqianWX/status/1881293445098283083">Tweet from xiaoqianWX (@xiaoqianWX)</a>: DeepSeek R1&#39;s API just became available(model name: deepseek-reasoner). Pricing seems to be 15CNY(2USD)/Mtok out. Haven&#39;t been able to bench anything on it yet</li><li><a href="https://openrouter.ai/docs/models">Models | OpenRouter</a>: A table of all available models</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/settin">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://x.com/deepseek_ai/status/1881318130334814301?s=46">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model & technical report🏆 MIT licensed: Distill & commercialize freely!🌐 Website & API are live now! Try DeepThink at h...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://x.com/teortaxesTex/status/1880768996225769738">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: R1 pass@10 is *way better* than o1-High compute; gains 20% on Hard set over pass@1. Whales tend to be mode-collapsed so pass@n only makes sense with how cheap they are. This supports my guess that rea...</li><li><a href="https://tinypic.host/image/Screenshot-2025-01-18-192202.2FIpeB">Screenshot 2025 01 18 192202</a>: Image Screenshot 2025 01 18 192202 hosted in Tinypic
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1329958477715472424)** (249 messages🔥🔥): 

> `Using Stable Diffusion for Photorealism, E-commerce Text-To-Image Models, Artistic Style Consistency in LoRA Training, Image Generation Issues and Solutions, AI Tools for Background Editing` 


- **Improving Photorealistic Image Generation**: Users discussed techniques for creating photorealistic images with Stable Diffusion 3.5, suggesting the use of LoRA for desired appearances.
   - One user noted challenges with getting a plasticky look and requested tips for more realistic outputs.
- **E-commerce and Google Cloud Deployment**: A user contemplated deploying a text-to-image model on Google Cloud and sought advice on whether to use GitHub models or Google Cloud Marketplace.
   - The consensus was that using pre-trained models would save time, but users were unsure about the most efficient deployment method.
- **Challenges with Artistic Style in LoRA Training**: Discussion focused on the impact of training resolution diversity in LoRA models and whether training exclusively at 1024x1024 would suffice.
   - It was suggested that using a variety of resolutions could enhance the model's ability to generalize across different image qualities.
- **Troubleshooting Image Generation Problems**: Several users reported issues with generating images, including slower processing times and discrepancies in output quality.
   - Some users suggested using different denoising steps and verifying configurations to achieve consistently better results.
- **Editing Backgrounds in Images**: Users shared their experiences with removing and blurring backgrounds in photos using tools like GIMP and AI solutions.
   - It was emphasized that manual editing often yields better results, particularly for specific image details that AI may not handle well.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stablediffusionweb.com/image/25622118-robot-woman-with-removed-face-plate">Robot Woman with Removed Face Plate</a>: no description found</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Why%20Use%20Swarm.md">SwarmUI/docs/Why Use Swarm.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API">API</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Features/Prompt%20Syntax.md">SwarmUI/docs/Features/Prompt Syntax.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://schinagl.priv.at/nt/hardlinkshellext/linkshellextension.html">Link Shell Extension</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1329938235563966647)** (27 messages🔥): 

> `Podcast creation and voice integration, Gemini Advanced Deep Research workflow, Using NotebookLM for college courses, Experiences with sourcing tools, Community introductions` 


- **Podcast Creation and Voice Integration**: A member shared their new podcast about **GLP-1s** and inquired about changing the voices of podcast hosts, suggesting an integration with [Eleven Labs](https://illuminate.google.com/home?pli=1).
   - However, another member pointed out that current podcast tools might not allow such changes.
- **Exploring Workflow with Gemini Advanced**: One user discussed a potential workflow utilizing **Gemini Advanced Deep Research** to generate reports and audio overviews, though access limitations were noted.
   - Another user confirmed a successful similar process, advising direct sourcing to avoid information loss.
- **Best Practices for NotebookLM in College**: A user asked for advice on organizing notebooks for an **econ course**, debating whether to upload multiple sources into one notebook or keep them separate.
   - A seasoned user advised using a topic-based organization to streamline workflows and maintain consistency across sources.
- **Community Resources and Tools**: A member shared a link to the **WebSync Chrome extension**, designed for importing pages and websites into **NotebookLM**, enhancing research efficiency.
   - Additionally, a video link was shared, showcasing tools like **NotebookLM** and their productivity enhancements.
- **Community Introductions and Engagement**: New members introduced themselves, highlighting language differences and expressing excitement about joining the community.
   - A user encouraged engaging questions in specific channels to foster more focused discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chromewebstore.google.com/detail/websync-full-site-importe/hjoonjdnhagnpfgifhjolheimamcafok">WebSync full site importer for NotebookLM - Chrome Web Store</a>: An extension to add pages or complete websites to NotebookLM</li><li><a href="https://illuminate.google.com/home?pli=1">Illuminate | Learn Your Way</a>: Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.</li><li><a href="https://youtu.be/t1OjAauA6uY?si=yo6frSZedvaGydt-">Top Relationship Experts Reveal Why Couples Are Ditching Marriage</a>: Is Marriage Really Worth It for Modern Couples? Why are couples opting out? This podcast examines the increasing trend of couples choosing not to marry, expl...</li><li><a href="https://youtu.be/mReOoe8Ou3A">4 AI Tools to 25× your Productivity: NotebookLM, Perplexity AI, Gemini Deep Research &amp; Gamma AI</a>: Comprehensive NotebookLM playlist - https://www.youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8🚀 Transform your NotebookLM research and knowled...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1329919833034195116)** (212 messages🔥🔥): 

> `Google One AI Premium, NotebookLM Plus, Podcast Generation Issues, Document Uploading Issues, Language Support in Interactive Podcast` 


- **Subscription Options for NotebookLM Plus**: Users discussed the differences between Google One AI Premium and Google Workspace Business Standard regarding access to Google Gemini's models and NotebookLM Plus features.
   - It's suggested that while both options provide access, Google One is simpler to manage without the complexities of Workspace.
- **Concerns with Podcast Generation**: Issues were raised about the variability in podcast lengths generated by NotebookLM, with users trying to customize duration but often receiving audio overviews that exceed requests.
   - Several users noted the challenges with voice roles switching randomly during podcasts, leading to confusion.
- **Problems Uploading Large Audio Files**: A user reported facing issues uploading audio files nearing or exceeding 100MB, which was suspected to be due to exceeding the overall upload limit of 200MB with existing files.
   - The importance of monitoring total file size before new uploads was emphasized to prevent this issue.
- **Document Uploading and OCR Limitations**: There were discussions on the difficulties faced when uploading non-copyable PDF documents requiring OCR for briefings, with one user stating they couldn't generate briefing documents from such files.
   - The need for enhanced support for OCR functionality in NotebookLM was highlighted as a potential improvement.
- **Multi-language Support for Podcasts**: Users expressed hope for the inclusion of languages other than English in NotebookLM's interactive podcast feature, with anticipation of soon availability.
   - Some users are currently leveraging workarounds to generate content in different languages, waiting for official support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/google-workspace-individual/answer/10758004?hl=en">Learn about Google Workspace Individual - Google Workspace Individual Help</a>: no description found</li><li><a href="https://policies.google.com/terms">Google Terms of Service – Privacy &amp; Terms – Google</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15724963">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1329916079568584775)** (193 messages🔥🔥): 

> `MCP server feedback, Roo Cline features, Rate limits with Claude, Chat log summarization, User interface concerns in MCP clients` 


- **Feedback on MCP server implementations**: Users expressed confusion over the inconsistent use of prompts in various MCP servers, where some only provide resource fetching without meaningful interaction.
   - Concerns were raised about the deviation of server implementations from the official documentation, leading to ineffective prompt usage.
- **Roo Cline's advantages and features**: Roo Cline has been praised for its ease of use with R1, supporting configuration of its own MCP servers and offering an 'agentic' experience through auto-approval of commands.
   - Users highlighted that Roo Cline's integration with VSCode makes it an appealing choice compared to other clients like Claude Desktop and LibreChat.
- **Managing rate limits in Claude**: Users reported encountering frequent rate limits when interacting with Claude, which can restrict context length and message frequency.
   - Discussions included a desire for tools to monitor the messages sent by Claude Desktop for better understanding of the rate limit issues.
- **Exploring MCP server for CSV modifications**: Interest was shown in whether an MCP server exists that can modify CSV rows based on prompts, but no clear solutions were found among available MCP servers.
   - A related server for Google Sheets was mentioned, indicating some existing tools for document management but not specifically for CSV handling.
- **Cost estimates for running MCP projects**: Users discussed the operational costs of utilizing various AI models for day-to-day tasks, noting variability based on user needs and usage frequency.
   - Experiences shared suggested that a personal digital assistant could potentially be run at a lower cost with the right model and usage strategy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp/servers/54hsrjhmq9">mcp-ragdocs</a>: An MCP server implementation that provides tools for retrieving and processing documentation through vector search, enabling AI assistants to augment their responses with relevant documentation contex...</li><li><a href="https://support.anthropic.com/en/articles/9450526-how-can-i-export-my-claude-ai-data">How can I export my Claude.ai data? | Anthropic Help Center</a>: no description found</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets</a>: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets - isaacphi/mcp-gdrive</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://www.pulsemcp.com/clients">39 MCP Clients: AI-powered apps compatible with MCP servers | PulseMCP</a>: A collection of AI apps and tools that are capable of functioning as Model Context Protocol (MCP) clients to interact with the growing list of MCP servers.</li><li><a href="https://github.com/ggozad/oterm">GitHub - ggozad/oterm: a text-based terminal client for Ollama</a>: a text-based terminal client for Ollama. Contribute to ggozad/oterm development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/142">[proposal] Add Augmentation capability for context enrichment by PederHP · Pull Request #142 · modelcontextprotocol/specification</a>: Motivation and ContextThis PR introduces an Augmentation capability to MCP, addressing the need for context enrichment in AI applications, as is commonly done using RAG. While existing capabilitie...</li><li><a href="https://glama.ai/mcp">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://glama.ai/mcp/servers?attributes=`)">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1330065200283648115)** (30 messages🔥): 

> `Figma MCP contribution, MCP Logic Calculator, LibreChat performance, TestFlight feedback, Anthropic model compatibility` 


- **Contribution Opportunities for Figma MCP**: [Figma MCP](https://github.com/nmfisher/figma_mcp) is in its early stages, and contributors are welcomed for this development effort.
   - One member expressed excitement about the project: *'This is very early/rough, so would appreciate any contributors!'*
- **AI Logic Calculator Gains Attention**: [MCP Logic Calculator](https://github.com/angrysky56/mcp-logic) developed by another member aims to utilize Prover9/Mace4 via Python, providing functionalities for Windows users.
   - Another member noted the potential for integrating classifiers with memory MCP for enhanced domain awareness.
- **Mixed Results with LibreChat**: Members reported using LibreChat with various LLMs like **Llama** and **DeepSeek**, noting performance issues compared to **Claude**.
   - Concerns were raised over configuration issues, with one member stating, *'Librechat is crap; I had so many config issues.'*
- **Testing iOS App via TestFlight**: Members discussed the upcoming launch of **Sage for Claude iOS** through TestFlight, highlighting its functionality and testing procedures.
   - Feedback varied with some noting the iOS version works well, while macOS showed crashing issues on startup.
- **Exploring Compatibility with Other Models**: Discussions included whether the Model Context Protocol (MCP) would work with other models beyond Sonnet, particularly referencing **Anthropic** models.
   - One member questioned the feasibility of integrating r1, hinting at interest in broader model compatibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://testflight.apple.com/join/EJIXPsr1">Join the Sage beta</a>: Available on iOS</li><li><a href="https://glama.ai/mcp/clients/libre-chat">LibreChat</a>: Enhanced ChatGPT with Agents, AI model switching, Code Interpreter, DALL-E 3, OpenAPI Actions, secure multi-user auth, and more. Supports OpenAI, Anthropic, Azure, and self-hosting via open-source.</li><li><a href="https://github.com/nmfisher/figma_mcp">GitHub - nmfisher/figma_mcp</a>: Contribute to nmfisher/figma_mcp development by creating an account on GitHub.</li><li><a href="https://github.com/angrysky56/mcp-logic">GitHub - angrysky56/mcp-logic: Fully functional AI Logic Calculator utilizing Prover9/Mace4 via Python based Model Context Protocol (MCP-Server)- tool for Windows Claude App etc</a>: Fully functional AI Logic Calculator utilizing Prover9/Mace4 via Python based Model Context Protocol (MCP-Server)- tool for Windows Claude App etc - angrysky56/mcp-logic
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1329919179465298022)** (167 messages🔥🔥): 

> `GPU vs CPU Performance, Agent Learning Models, Self-Adaptive LLMs, AI Tools Evaluation, Online Community Dynamics` 


- **GPU vs CPU Efficiency in Array Processing**: In discussions on whether to use GPU or CPU for finding the max value of an array, it was noted that **GPUs** are faster for large arrays, especially in parallel processing but have data transfer bottlenecks.
   - A member mentioned that finding the max in an array is a trivially parallel operation, suggesting the performance could be similar across both architectures for large data sets.
- **Exploration of Agent Learning Models**: Discussion arose about building agents with LLMs, acknowledging the challenges faced in making them act autonomously due to their limitations with 'agentive' tasks.
   - There was agreement that despite advances in AI, breakthrough methods are still necessary for agents to operate meaningfully beyond basic command execution.
- **Evaluation of AI Coding Tools**: Participants evaluated various code generation tools including OpenAI ChatGPT and Claude, with preferences noted for those that produce adequate quality for specific coding tasks.
   - OpenAI ChatGPT was highlighted as a superior tool compared to others, while also commenting on the new trends in AI tooling and coding since the rise of GitHub CoPilot.
- **Self-Adaptive Large Language Models**: A paper on self-adaptive LLMs titled **Transformer²** introduced mechanisms that enable real-time task adaptation, outperforming conventional fine-tuning methods.
   - The paper discussed using reinforcement learning to dynamically mix task-specific vectors, indicating advancements that could make traditional fine-tuning methods obsolete.
- **Community Insights on AI Trends**: The community shared observations on the hype surrounding AI, emphasizing skepticism about the actual capabilities of currently marketed solutions compared to public expectations.
   - A note was made that commercial interests often drive the trends in AI announcements, making fundamental breakthroughs essential for the technology to deliver meaningful results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1881258443669172470">Tweet from Sam Altman (@sama)</a>: twitter hype is out of control again. we are not gonna deploy AGI next month, nor have we built it.we have some very cool stuff for you but pls chill and cut your expectations 100x!</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">Tweet from mgostIH (@mgostIH)</a>: Wtf is up with deep learning???</li><li><a href="https://arxiv.org/abs/2501.06252">$\text{Transformer}^2$: Self-adaptive LLMs</a>: Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse...</li><li><a href="https://tenor.com/view/adorable-wink-bat-oh-hey-gif-14058611">Adorable Wink GIF - Adorable Wink Bat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/engel-t-pose-engel-gif-16943840286212504665">Engel T-pose Engel GIF - Engel T-pose engel - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/gop-corruption-thoughts-and-prayers-mass-shooting-human-rights-gif-25725581">Gop Corruption GIF - Gop Corruption Thoughts And Prayers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/goku-ssj4-super-saiyan-4-dragon-ball-gt-dragon-ball-gif-6491195933392867517">Goku Ssj4 GIF - Goku Ssj4 Super saiyan 4 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/crosswind-landing-gif-20167802">Crosswind Landing GIF - Crosswind Landing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/xzibit-meme-inception-gif-13033570">Xzibit Meme GIF - Xzibit Meme Inception - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/soviet-cat-sovicat-soviet-ussr-cat-gif-21826197">Soviet Cat Sovicat GIF - Soviet Cat Sovicat Soviet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/klasik-clasik-clasic-gif-25398314">Klasik Clasik GIF - Klasik Clasik Clasic - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/marachime-highrollers-dnd-dungeons-and-dragons-mark-hulmes-gif-14728949">Marachime Highrollers GIF - Marachime Highrollers Dnd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/russian-roulette-gun-gif-24197229">Russian Roulette GIF - Russian Roulette Gun - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/DrJimFan/status/1881353126210687089">Tweet from Jim Fan (@DrJimFan)</a>: We are living in a timeline where a non-US company is keeping the original mission of OpenAI alive - truly open, frontier research that empowers all. It makes no sense. The most entertaining outcome i...</li><li><a href="https://tenor.com/view/the-gun-gun-gif-25386021">The Gun Gun GIF - The Gun Gun - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/smile-gif-3415810009431604905">Smile GIF - Smile - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/im-bayblade-beyblade-spin-silly-gif-16417105">Im Bayblade Beyblade GIF - Im Bayblade Beyblade Spin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fixupx.com/SavannahFeder/status/1877444748039819301">Tweet from Savannah (@SavannahFeder)</a>: Announcing Astral - an AI marketer that works 24/7 to grow your startup.Astral navigates websites, creates content, and runs marketing across socials.  Watch Astral automate Reddit in real-time:</li><li><a href="https://www.youtube.com/watch?v=gfr4BP4V1R8">AI discusses document that just says “Poopoo Peepee”</a>: Document:Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo P...</li><li><a href="https://www.youtube.com/watch?v=BEv_qZwR3h8">Dark Dreams - Artoria Pendragon (Saber Alter) [AMV]</a>: Artoria Pendragon (Saber Alter) - Mr. Sandman - Fate/Stay Night: Heaven&#39;s Feel - AMV-------------------------------------------------------------------------...</li><li><a href="https://www.youtube.com/watch?v=JMC56argXVk)">How to Turn Left at &#39;Left Turn Signal&#39; | Turning Smart</a>: Subscribe Today! ►  http://youtube.com/c/smartdrivetestBeware the &quot;Left Turn Signal&quot; - it could be the proverbial wolf in sheep&#39;s clothing on your road test ...</li><li><a href="https://youtu.be/w6zi95SknZw">Welcome to Cosmology and its Fundamental Observations</a>: This video combines chapters 1 and 2 of the videos in my new series of Cosmology.  I&#39;m going through Dr. Barbara Ryden&#39;s textbook &quot;Introduction to Cosmology&quot;...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://youtu.be/w8HdOHrc3OQ">[Best Version] The Great Dictator Speech - Charlie Chaplin + Time - Hans Zimmer (INCEPTION Theme)</a>: - PLEASE READ -Charlie Chaplin&#39;s speech from &quot;The Great Dictator&quot; together with Hans Zimmer&#39;s &quot;Time&quot; from the movie &quot;INCEPTION&quot; = EPIC!!!Important note: This...</li><li><a href="https://tenor.com/view/die-kill-internet-modem-eileen-gif-20652466">Die Kill GIF - Die Kill Internet - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1329948738780659723)** (37 messages🔥): 

> `Lightning Attention Paper Discussion, rStar-Math Research Findings, Tensor Product Attention (TPA) Mechanics, Linear Tensor Product Lightning Attention, DeepSeek's Group Relative Policy Optimization` 


- **Lightning Attention Paper Rejected for Novelty**: In a discussion about the Lightning Attention paper, some members echoed concerns over its rejection from ICLR due to perceived incremental changes from prior works like [NormAttention](https://arxiv.org/pdf/2210.10340) and FlashAttention.
   - Reviewers criticized its novelty, leading some to wonder if using adaptive matrix products during training and inference is already a well-known technique.
- **Insights on rStar-Math's Unique Methodology**: The rStar-Math paper showcases how small language models can rival or exceed OpenAI's capabilities in math reasoning without distillation, leveraging Monte Carlo Tree Search (MCTS) for deep thinking.
   - Notably, the method is deemed practical for simulated environments, offering three innovative training techniques that avoid reliance on human data.
- **Implementing Tensor Product Attention with Lightning Attention**: An experiment demonstrated successful integration of Tensor Product Attention using lightning attention's linearization, achieving a significant speed boost in a toy model.
   - The implementation shows about a **3x speed** improvement, which allows effective handling of large tensor operations in attention mechanisms.
- **DeepSeek's Group Relative Policy Optimization Explained**: Discussions highlighted that DeepSeek's GRPO functions similarly to PPO but without a value function, relying instead on Monte Carlo estimates of the advantage.
   - Understanding GRPO requires a grasp of the challenges value functions present when applied to language models, suggesting a need for foundational knowledge of PPO.
- **Community Engagement and Resource Sharing**: Members actively shared links to relevant research papers, GitHub repositories, and resources, such as the [DeepSeek R1 PDF](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf).
   - Contributions sparked meaningful discussion about model efficiency and performance across various attention paradigms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>: We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1, without distillation from superior models. rStar-Math achie...</li><li><a href="https://fixupx.com/natolambert/status/1881380809153847711">Tweet from Nathan Lambert (@natolambert)</a>: For those trying to understand DeepSeeks Group Relative Policy Optimization (GRPO): GRPO is just PPO without a value function using monte carlo estimates of the advantage. So, study why PPO exists (lo...</li><li><a href="https://arxiv.org/abs/2501.06252">$\text{Transformer}^2$: Self-adaptive LLMs</a>: Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse...</li><li><a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>: Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf">DeepSeek-R1/DeepSeek_R1.pdf at main · deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/CoffeeVampir3/Micro-Mjolinear/blob/master/models/lightning_tensor_product.py#L143>">Micro-Mjolinear/models/lightning_tensor_product.py at master · CoffeeVampir3/Micro-Mjolinear</a>: Contribute to CoffeeVampir3/Micro-Mjolinear development by creating an account on GitHub.</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6_infer.py#L138">T6/model/T6_infer.py at d4f6168852397a7b0b0d9fd65326bb91976c7067 · tensorgi/T6</a>: The official implementation of Tensor ProducT ATTenTion Transformer (T6) - tensorgi/T6</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L107">T6/model/T6.py at d4f6168852397a7b0b0d9fd65326bb91976c7067 · tensorgi/T6</a>: The official implementation of Tensor ProducT ATTenTion Transformer (T6) - tensorgi/T6
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1330173553164943390)** (3 messages): 

> `Titans, Adaptive Transformers, RNN testing, 760M model performance, BABILong` 


- **Titans and Adaptive Transformers create buzz**: Recent discussions highlight excitement around both **Titans** and **Adaptive Transformers**, with potential implications for upcoming projects.
   - A helpful [link](https://sakana.ai/transformer-squared/) was shared regarding Adaptive Transformers that may contribute to this excitement.
- **Evaluating models for training potential**: A member noted the potential of a model showing **760M parameters** outperforming commercial counterparts on **BABILong**.
   - They suggested starting evaluations with this promising model while considering reports of others using **RNNs** at test time.
- **Community support for new models**: A member expressed hope for the success of these new models, signaling a supportive community environment.
   - This shared optimism may bolster collaborative efforts in evaluating these technologies.



**Link mentioned**: <a href="https://sakana.ai/transformer-squared/">no title found</a>: no description found

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1329920487861784578)** (15 messages🔥): 

> `Microsoft OpenAI partnership concerns, AI security vulnerability findings, AI compliance tools for trading, TikTok ownership and ban implications, FrontierMath funding controversies` 


- **Microsoft's Investment in OpenAI Raises Antitrust Warnings**: The FTC expressed concerns about Microsoft's **$13 billion investment** in OpenAI, fearing it may enhance the company's dominance in the AI market and harm competition.
   - FTC Chair **Lina Khan** highlighted how such partnerships could lead to *lock-in* and disadvantage start-ups in accessing crucial AI resources.
- **Microsoft Researchers Assert AI Systems Can't Be Fully Secure**: In a **pre-print paper**, Microsoft researchers concluded that AI systems can **never be fully secure**, amplifying existing security risks and introducing new vulnerabilities.
   - They warn that while defenses may raise the cost of attacks, threats like gradient-based attacks and phishing remain prevalent.
- **AI Tools Crack Down on Wall Street Trader Communication**: Compliance firms are deploying AI to **decode trader communications**, enabling the detection of potential financial crimes amidst heightened regulatory scrutiny.
   - These AI systems aim to interpret complex slang and coded language that traditional methods often miss, creating stricter compliance measures.
- **Supreme Court Upholds TikTok Ban Unless Sold**: The Supreme Court upheld a law requiring TikTok to be sold by its Chinese parent or face a ban, citing national security threats posed by its ownership.
   - This decision creates significant urgency as the law goes into effect, potentially limiting downloads and updates for the app.
- **Controversy Surrounding FrontierMath's Funding**: The connection between **OpenAI** and **FrontierMath** funding has come under scrutiny, with claims that contractors were unaware of OpenAI's financial involvement until recently.
   - Discussions reveal concerns over the **NDA restrictions** placed on Epoch leaving many contributors in the dark about the funding sources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DanHendrycks/status/1881036645555937719">Tweet from Dan Hendrycks (@DanHendrycks)</a>: @GaryMarcus Can confirm AI companies like xAI can&#39;t get access to FrontierMath due to Epoch&#39;s contractual obligation with OpenAI.</li><li><a href="https://x.com/CarinaLHong/status/1880820323597357273">Tweet from Carina Hong (@CarinaLHong)</a>: 1. OAI binds Epoch to an NDA until eve of o3 performance claim, preventing Epoch to disclose OAI is the donor and that OAI has exclusive data access2. Mathematicians then sign NDA on the problem & sol...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://it.slashdot.org/story/25/01/17/1658230/microsoft-research-ai-systems-cannot-be-made-fully-se">Microsoft Research: AI Systems Cannot Be Made Fully Secure - Slashdot</a>: Microsoft researchers who tested more than 100 of the company's AI products concluded that AI systems can never be made fully secure, according to a new pre-print paper. The 26-author study, which inc...</li><li><a href="https://it.slashdot.org/story/25/01/17/1658230/microsoft-research-ai-systems-cannot-be-made-fully-secure">Microsoft Research: AI Systems Cannot Be Made Fully Secure - Slashdot</a>: Microsoft researchers who tested more than 100 of the company's AI products concluded that AI systems can never be made fully secure, according to a new pre-print paper. The 26-author study, which inc...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hzmpuq/vlc_to_add_offline_realtime_ai_subtitles_what_do/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://slashdot.org/story/25/01/17/1356236/ai-tools-crack-down-on-wall-street-trader-code-speak">AI Tools Crack Down on Wall Street Trader Code Speak - Slashdot</a>: Compliance software firms are deploying AI to decode complex trader communications and detect potential financial crimes as Wall Street and London regulators intensify scrutiny of market manipulation....</li><li><a href="https://slashdot.org/story/25/01/17/1958200/microsoft-openai-partnership-raises-antitrust-concerns-ftc-says">Microsoft-OpenAI Partnership Raises Antitrust Concerns, FTC Says - Slashdot</a>: Microsoft's $13 billion investment in OpenAI raises concerns that the tech giant could extend its dominance in cloud computing into the nascent AI market, the Federal Trade Commission said in a report...</li><li><a href="https://news.slashdot.org/story/25/01/17/1518232/supreme-court-upholds-law-banning-tiktok-if-its-not-sold-by-its-chinese-parent-company">Supreme Court Upholds Law Banning TikTok If It's Not Sold By Its Chinese Parent Company - Slashdot</a>: An anonymous reader shares a report: The Supreme Court on Friday unanimously upheld the federal law banning TikTok beginning Sunday unless it's sold by its China-based parent company, holding that the...</li><li><a href="https://tech.slashdot.org/story/25/01/17/0012237/google-wont-add-fact-checks-despite-new-eu-law">Google Won't Add Fact Checks Despite New EU Law - Slashdot</a>: According to Axios, Google has told the EU it will not add fact checks to search results and YouTube videos or use them in ranking or removing content, despite the requirements of a new EU law. From t...</li><li><a href="https://www.lesswrong.com/posts/cu2E8wgmbdZbqeWqb/meemi-s-shortform?commentId=FR5bGBmCkcoGniY9m">meemi&#x27;s Shortform — LessWrong</a>: Comment by meemi - FrontierMath was funded by OpenAI.[1]The communication about this has been non-transparent, and many people, including contractors working on this dataset, have not been aware of th...</li><li><a href="https://slashdot.org/story/25/01/17/1414242/intel-acquisition-target-of-mystery-suitor-semiaccurate-reports">Intel Acquisition Target of Mystery Suitor, SemiAccurate Reports - Slashdot</a>: Tech news and research site SemiAccurate is reporting that an unidentified company is seeking to acquire Intel in its entirety. The publication -- citing a confidential email that it reviewed and a su...</li><li><a href="https://techcrunch.com/2025/01/18/perplexity-ai-submits-bid-to-merge-with-tiktok">Perplexity AI submits bid to merge with TikTok | TechCrunch</a>: With a TikTok ban looming in the United States, Perplexity AI is the latest bidder hoping to give the video app a new corporate home. CNBC first reported
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1330027603075006504)** (81 messages🔥🔥): 

> `Konkani Language AI Model, Cohere's Accessibility, Project Ideas, API Access and Limitations` 


- **Konkani Language Model Plans**: A member plans to train an AI model to understand the **Konkani language**, expressing hopes for advancement in the project despite needing university approval.
   - They emphasized that collaboration with industry is crucial for moving forward.
- **Concerns About Cohere's Accessibility**: A member highlighted several points about **Cohere's** accessibility, mentioning issues like lack of persistent sign-in, no dark mode, and absence of a mobile app.
   - These features are crucial for user experience and are seen as barriers compared to other services.
- **Engagement with Cohere API Access**: Members discussed the **free API access**, which offers 1000 requests per month per model, making it an accessible option for experimentation.
   - This allows users to engage with the models without financial commitments, encouraging contributions to open source.
- **Feedback on Cohere's Interface**: Members shared positive feedback regarding the interface and the tools offered by **Cohere**, appreciating its usability despite certain limitations.
   - There was general agreement that not every model needs to cater to every user, reflecting a diverse user base.
- **Model Switching and Updates**: The discussion included a **potential model switching feature**, which could allow users to select from various models based on their needs efficiently.
   - There are rumors of a major upcoming update, sparking excitement for new functionalities in the platform.



**Link mentioned**: <a href="https://github.com/cohere-ai/cohere-python/issues/632">Once you have an error uploading a model, your account (web and api) corrupts and Dataset/Model environment will no longer work · Issue #632 · cohere-ai/cohere-python</a>: Using your example with your CSV file. import cohere co = cohere.Client() # upload a dataset my_dataset = co.datasets.create( name=&quot;datasettest&quot;, data=open(&quot;./Arts.Class.1000.csv&quot;,...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1330131527928905811)** (11 messages🔥): 

> `Billing Issues, AI Behavior Management, Invoices and Receipts, AI Project Feedback` 


- **Billing questions regarding company details**: A member inquired about the process of entering company information for billing purposes for tax deduction reasons.
   - *mrdragonfox* advised to contact [support@cohere.com](mailto:support@cohere.com) with account ID for assistance on this issue.
- **Request for old invoices addressed to companies**: The same member asked if it's possible to receive old invoices and receipts addressed to the company instead of individuals.
   - *mrdragonfox* reiterated contacting support for help with this request.
- **Challenges with AI behavior in projects**: A member shared their concern about AI responses deviating from intended prompts in their storytelling platform project.
   - *xvarunx* asked for more details on the specific model being used and encouraged feedback submission to support.
- **Limitations in AI behavior management**: Discussion revealed that guardrails for AI behavior can be implemented but are not foolproof, usually through external classifiers.
   - *mrdragonfox* mentioned that there's no way to completely prevent deviations in language model behavior.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1329943248915009631)** (12 messages🔥): 

> `Command-R Model Versioning, Embed Job Concurrent Limits, Dify.ai Integration Issues` 


- **Command-R Points to Older Model**: A discussion clarified that `command-r` was not pointed to the latest model to avoid introducing **breaking changes** for users of the non-timestamped model.
   - A suggestion was made to utilize **aliases** that define the version while keeping a **latest** tag for ongoing updates.
- **Embed Job Limitations Causing Errors**: Khalid reported receiving an error indicating they reached the **maximum number of concurrent embed jobs**, with all previous jobs stalled.
   - It was suggested that he email support as there may be a need to review his account details due to potential job cancellations being stalled.
- **Dify.ai Key Integration Blocked**: Fleck082814 encountered a **403 Forbidden** error while trying to add their Cohere key in a self-hosted **dify.ai** instance, suspecting an IP block.
   - Xvarunx noted that similar requests indicated that requests from **China** were currently unsupported, advising a potential downgrade to version **0.8** as a workaround.
- **Holiday Notice for Support Responses**: Xvarunx informed the team that due to a national holiday in the US, support response times might be affected.
   - This highlights the need for patience from users awaiting support during holiday periods.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1330207188932624384)** (32 messages🔥): 

> `Cohere Models Overview, Tool Calling and Code Generation, Understanding AGI` 


- **Cohere Models Overview**: A list of Cohere models was shared, including `command-r`, `c4ai-aya-expanse-8b`, and `command-light-nightly` among others.
   - It was noted that users can train models for customization to specific use cases.
- **Tool Calling and Code Generation Explained**: The interaction of tool use involves developers defining how Cohere's models can interact with specific tools through structured components.
   - This process involves the LLM making decisions on tool calls, executing them, and generating responses based on results.
- **AGI Definition**: AGI stands for **Artificial General Intelligence**, which was mentioned as a topic of interest.
   - Unfortunately, there was no detailed information found in Cohere's documentation regarding AGI.


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1331013821858316398)** (4 messages): 

> `Cohere's Math Performance, Limitations of LLMs, Tool Usage Tips` 


- **Cohere's Struggles with Basic Math**: One member expressed frustration over **Cohere**'s incorrect calculations, specifically stating it incorrectly calculated the total number of weeks in **18 months** as **27 weeks**.
   - They noted that spending time on Google was often faster due to needing to verify answers given by the AI.
- **All LLMs Have Math Issues**: Another member pointed out that the problems with math performance are not isolated to **Cohere**, but rather a common issue across all **large language models (LLMs)**.
   - They explained that this is well understood among those who regularly use LLMs, indicating a systemic challenge with mathematical calculations.
- **Usage Tips for Improved Results**: A suggestion was made to either use the AI as a tool similar to a calculator or to employ a lower temperature setting for better responses.
   - This highlights the need for users to understand the probabilistic nature of LLMs to get accurate outputs.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1330399508512641128)** (2 messages): 

> `MOOC course confirmation, Spring mailing list` 


- **Awaiting MOOC Course Confirmation**: @gaganadev inquired about the confirmation for the **MOOC course starting this January**.
   - Another member mentioned that the **mailing list for the spring course** will likely start next week.
- **Spring Course Mailing List Announcement**: It was discussed that the **mailing list** related to the spring course will likely begin distribution next week.
   - This suggests that further details about the course timeline are imminent.


  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1330010594538946672)** (1 messages): 

> `Document to Podcast blueprint, Open source projects, Community engagement` 


- **Live Introduction to Document to Podcast Blueprint**: The team from <@&1316851621027647648> will be delivering a live introduction to the **Document to Podcast** blueprint, a customizable recipe for building on open source during their upcoming event.
   - Members are encouraged to join and welcome questions with <@1183778352927092806>, <@1300855165393309747>, and <@1250742001272492097> at this exciting gathering.
- **Blueprints Enhance Open Source Collaboration**: This event is a fantastic opportunity for <@&1229573172018417674> to come together and discover new, useful **open source projects**.
   - Participants are urged to hit the Interested button if they would like to attend and engage with the community.


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

---
id: 305c4e89-b402-4507-89eb-224a5d3ea59f
title: '[AINews] Gemma 2 tops /r/LocalLlama vibe check'
date: '2024-07-17T22:57:14.252944Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-gemma-2-tops-rlocalllama-vibe-check
---

<!-- buttondown-editor-mode: plaintext -->**Gemma 2 (9b, 27B) is all you need?**

> AI News for 7/16/2024-7/17/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**468** channels, and **2051** messages) for you. Estimated reading time saved (at 200wpm): **232 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Every few months](https://www.reddit.com/r/LocalLLaMA/search/?q=best+models&restrict_sr=on), someone asks a vibe check question in /r/LocalLlama that takes off ([March 2024](https://www.reddit.com/r/LocalLLaMA/comments/1b4e50z/whats_the_best_7b_model_right_now_march_2024/), [June 2024](https://www.reddit.com/r/LocalLLaMA/comments/1dcf3yy/best_local_base_models_by_size_quick_guide_june/) and the official [Models Megathread](https://www.reddit.com/r/LocalLLaMA/comments/1bgfttn/models_megathread_4_what_models_are_you_currently/) are the previous ones).

 ![image.png](https://assets.buttondown.email/images/a9376530-dfc2-457a-9c29-3a8a7b4596ad.png?w=960&fit=max) 

Recently a [best models for their size?](https://www.reddit.com/r/LocalLLaMA/comments/1e4ja8n/what_are_the_best_models_for_their_size/) question is a chance to revisit the rankings. Last month's Gemma 2 ([our coverage here](https://buttondown.email/ainews/archive/ainews-gemma-2-the-open-model-for-everyone/)) won handily, even without the 2B model:

- **Gemma 2: 18 mentions**
  - "Some of the best LLM's I've run with their size."
  - "I'm continually impressed by 9B as well for summarizing and reasoning over philosophical texts with fairly coherent conceptual combinations in English."
  - "We get very good performance with an agentic workflow, allowing the LLM to specialise one task at a time."
  - "Ditto. Running genma2 9b on my 2080ti, nice and snappy and really good results. I really want a local llm that can provide links to sources like perplexity or Kagi fastgpt because that feature is so killer"
  - "gemma 2 9b is much better than llama 8b if you are asking"
  - "Gemma 2 9b is the only model that is super fast while beating 3.5 on any task I throw at it. + it's REALLY good in french for it's size. Perfect for a discord bot. And if you offload most of the layer, you can get a fast enough discord bot that takes only 3 or 4gb of VRAM, so you have room for stable diffusion stuff etc ! Truely incredible. Combined with moondream 1b for vision and voila you have a multilingual bot that follow really well the prompt and writing style and able to "see" the pictures in the chat. For around 5gb vram."
  - "Gemma 9B is vastly superior to even Llama 70B when working with non-english text."
  - "I tried using gemma 2 9b instruct for synth data generation (derive question and answer from a paragraph) and it refused to cooperate 90% of the time... it gave me a very bad impression"
- **Llama 3**: 10 mentions
  - "Llama 3 70B and Qwen 72B for 70ish Billion LLMs"
- **Mistral**: 9 mentions
  - "Mistral 7B for me. Not the MoE one, don't have the hardware for that"
  - "I love Mistral 7B (v03) instruct. IMHO it’s not even close to Gemma 9B, even at smaller quants of the latter. but mistral v03 came out way before gemma 9b."
  - "mistral-instruct v0.3 7b. I love that model. even if gemma 8b and phi medium seems better. also WizardLM2 (very similar to mistral and based on it) is great.. try it."
- **Phi 3**: 6 mentions
- **Qwen**: 5 mentions
  - "it was nice when it came out, but superseded by gemma and phi-3"

Other positive mentions: DeepSeek, Cohere Command R, InternLLM, Yi 34B (Nous-Capybara version)

> Meta note: **We are now splitting out /r/localLlama in our Reddit recaps** because of the tendency of the other subreddits to drown out technical discussion. Enjoy!

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

**Andrej Karpathy's new AI+Education company Eureka Labs**

- [@karpathy](https://twitter.com/karpathy/status/1813263734707790301) announced he is starting an **AI+Education company called Eureka Labs** to build an **AI native school**. The goal is to make it **easy for anyone to learn anything**, with AI Teaching Assistants supporting human teachers. Their **first product will be LLM101n**, an undergraduate-level class on training your own AI. Course materials will be free, with revenue from digital/physical cohorts.
- [@DrJimFan](https://twitter.com/DrJimFan/status/1813360847361831226) noted that **no one is more qualified to do EdTech than Andrej**, and other AI startups in this area can't compete. He's glad they both like the name "Eureka".
- [@danielhanchen](https://twitter.com/danielhanchen/status/1813330269044408612) is excited for the **LLM101n course**, with chapters covering bigrams, attention, transformers, optimization, datasets, inference, fine-tuning, and deployment. He notes Andrej's course materials like CS231n and Zero to Hero are pure gold.

**New model releases**

- [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1813231491154899012) announced the release of **Mathstral 7B and Codestral Mamba 7B** under Apache 2 license. Mathstral 7B obtains **56.6% pass@1 on MATH**, outperforming Minerva 540B by 20%+. Codestral Mamba is one of the first open source models with a **Mamba 2 architecture**, the best 7B code model available.
- [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1813252390692303069) introduced **SmolLM**, a series of 135M, 360M and 1.7B models outperforming MobileLLM, Phi1.5 and Qwen2 small models. Trained on **SmolLM-corpus of high quality web, code and synthetic data**.
- [@AnthropicAI](https://twitter.com/AnthropicAI/status/1813237754081251573) released the **Claude Android app**, now available on Google Play.

**Discussions on model architectures and training data**

- [@YiTayML](https://twitter.com/YiTayML/status/1813262126162845772) started a blog series on model architectures in the era of LLMs, covering topics like **Transformer Encoders/Decoders, PrefixLM, and denoising objectives**. Responds to the question of what happened to encoder-only models and if denoising objectives are still useful.
- [@jxmnop](https://twitter.com/jxmnop/status/1813326815496400919) believes the most impactful topic in AI right now is **Agents**. We need to build agency into the next generation of language models (**agent-native LLMs**) vs faking it with prompting. This will require new datasets, task definitions, and training techniques.
- [@Teknium1](https://twitter.com/Teknium1/status/1813349962065068303) argues that **synthetic data is real data** and doesn't have to cause mode collapse or top out at previous SOTA if the teacher model is exceeded.

**Other notable updates**

- [@alexandr_wang](https://twitter.com/alexandr_wang/status/1813242291622199628) shared that @scale_AI has come a long way since being in a basement, now in a new office.
- [@fchollet](https://twitter.com/fchollet/status/1813362217020219587) shared a well-explained guide to the **Transformer architecture with Keras code examples**.
- [@llama_index](https://twitter.com/llama_index/status/1813355957491273936) made huge improvements to **markdown-based table reconstruction for parsing complex documents** in their new release.

---

# AI Reddit Recap

## /r/LocalLlama

**Theme 1. New Model Releases from Mistral AI and Apple**

- **[mistralai/mamba-codestral-7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1)** ([Score: 216, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1e4qgoc/mistralaimambacodestral7bv01_hugging_face/)): **Mistral AI** has released the **Mamba-Codestral-7B** model, a **7 billion parameter** code generation model based on the **Mamba architecture**. This model, available on **Hugging Face**, is designed for efficient inference and is capable of generating code in various programming languages, including **Python**, **JavaScript**, **Java**, **C++**, and **Rust**. The model's performance is particularly notable in **Python** code generation tasks, where it outperforms larger models like **StarCoder-15B**.
- **Apple has released the weights for their 7B DCLM base model.** ([Score: 181, Comments: 48](https://reddit.com//r/LocalLLaMA/comments/1e4jw0c/apple_has_released_the_weights_for_their_7b_dclm/)): **Apple unveils DCLM-Baseline-7B model**. The **7 billion parameter** language model, trained on **2.5T tokens** with a **2048 token** context length, is based on the **DCLM-Baseline dataset** and aims to demonstrate the impact of systematic data curation on model performance. An updated version with **8K context length** has also been released, with links provided to the [Hugging Face repository](https://huggingface.co/apple/DCLM-Baseline-7B-8k), [research paper](https://arxiv.org/abs/2406.11794), and [related GitHub project](https://github.com/mlfoundations/dclm).
  - **Apple's Open Model Surprise**: **Apple's release** of an open model receives praise from the community. Users express excitement about the potential insights from the **DCLM (Data-Centric Language Model) approach**, viewing it as a step towards more **open-source AI development**.
  - **Context Length Confusion**: Discussion arises about the significance of the **2048 token context length**. Users debate how this compares to other models like **Llama 3**, highlighting the variability in tokenization methods across different LLMs.
  - **Benchmarks and Licensing Questions**: Community members inquire about **performance benchmarks** for the new model. Questions also emerge regarding the **"Apple ASCL" license**, with users comparing it to the **MIT license** and seeking clarification on its open-source status.


**Theme 2. Llama 3 Performance and Limitations**

- **[This meme only runs on an H100](https://i.redd.it/urpjifh14xcd1.jpeg)** ([Score: 230, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1e4uwz2/this_meme_only_runs_on_an_h100/)): **"This meme only runs on an H100"** humorously exaggerates the high computational requirements of modern AI models. The joke plays on the fact that **NVIDIA's H100 GPU** is currently one of the most powerful and sought-after graphics processing units for AI training and inference, often used in large language models and other computationally intensive AI tasks.
- **[I gave Llama 3 a 450 line task and it responded with "Good Luck"](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3 fails long instruction test**. When given a **450-line task**, Llama 3 responded with a simple "Good Luck" instead of attempting to process or execute the lengthy instruction set. This behavior suggests potential limitations in Llama 3's ability to handle extremely long or complex prompts effectively.
  - **"Good Luck" or Good AI?** The model's response may be due to **exam-like phrasing**. Adding "Output:" or "Answer:" could yield different results, highlighting the **distinction between text completion and comprehension**.
  - **AI's Relatable Laziness**: An early open-source model responded to a code request by saying, *"This sounds like a lot of work"*, showcasing **human-like reluctance** to complex tasks.
  - **Context Matters**: The **default context length of 2048** in Ollama likely truncated the lengthy instruction. Increasing it to **8096** could enable processing of the complete 450-line task.

**Theme 3. Comparing Model Performance by Size**

- **what are the best models for their size?** ([Score: 60, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4ja8n/what_are_the_best_models_for_their_size/)): **Best Models by Size for Reasoning**: The post seeks opinions on the most "intelligent" language models relative to their size, focusing on **pure reasoning abilities** and problem-solving outside of training data. The author specifically asks for personal experiences with models of various sizes (**3B**, **4B**, **7B**, and larger) rather than relying on leaderboard rankings.
  - **Gemma 2 Steals the Show**: **Gemma 2 9B** and **27B** models are widely praised for their performance relative to size. Users highlight their reasoning abilities and multilingual capabilities, with some comparing them to **GPT-3.5** level performance.
  - **Size Matters, But Not Always**: Discussion includes recommendations for various model sizes, from **Phi-3 4B** to **Llama 3 70B** and **Qwen 72B**. Users debate the trade-offs between model size, performance, and hardware requirements.
  - **Testing on Low-End Systems**: One user shares ongoing experiments running models from **4B** to **112B** on older hardware, including **4th generation i7** processors without GPUs. Results expected to be presented at the **Technosecurity conference** in **Pasadena** in **mid-September**.

**Theme 4. Debate on AI Hype vs. Long-term Potential**

- **[Linux Torvalds](https://i.redd.it/z4scsmapczcd1.jpeg)** ([Score: 77, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1e55iit/linux_torvalds/)): **Linux Torvalds**, creator of the **Linux kernel**, expresses skepticism about current **AI hype** in a recent interview. He argues that while AI has made significant progress in specific areas like **image recognition** and **language models**, it still lacks general intelligence and is primarily good at pattern matching rather than true understanding. Torvalds believes the current AI boom is largely driven by **marketing** and cautions against overestimating AI's capabilities.
  - Commenters draw parallels between **AI hype** and the **dotcom bubble**, suggesting a cycle of overhype, undervaluation, and eventual world-changing impact. Some argue that **AI's long-term potential** is significantly underestimated despite short-term exaggeration.
  - Debate ensues over the capabilities of **Large Language Models (LLMs)**, with some claiming they can replace **30% of workers**, while others argue LLMs are unreliable and unpredictable compared to humans for many tasks.
  - Commenters humorously play on the misspelling of **Linus Torvalds'** name, jokingly associating him with "**Tim Apple**," "**Bill 'Michaelsoft' Gates**," and "**Linus Tech Tips**," showcasing the community's playful engagement with tech personalities.

## Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

> Comment crawling works now but has lots to improve!

**Theme 1. Llama 3 Performance and Limitations**

- [/r/LocalLLaMA] **[This meme only runs on an H100](https://i.redd.it/urpjifh14xcd1.jpeg)** ([Score: 230, Comments: 42](https://reddit.com//r/LocalLLaMA/comments/1e4uwz2/this_meme_only_runs_on_an_h100/)): **"This meme only runs on an H100"** humorously highlights the extreme computational requirements of modern AI models. The joke plays on the idea that even simple tasks like displaying a meme might require **NVIDIA's H100 GPU**, one of the most powerful and expensive graphics cards designed for **AI and machine learning workloads**.

- [/r/LocalLLaMA] **[I gave Llama 3 a 450 line task and it responded with "Good Luck"](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3** faced difficulties when presented with a **450-line task**, responding with a simple "Good Luck" instead of attempting to complete it. This unexpected response highlights potential limitations in the model's ability to handle complex, lengthy prompts or tasks, raising questions about its practical applications for extensive coding or text generation tasks.
  - **"Good Luck" Might Be Exam-Related**: The phrase "Your Task is (...)" could trigger an **exam-like response**. Adding "Output:" or "Answer:" might yield different results, highlighting the **difference between text completion and comprehension**.
  - **AI Models Can Be Lazy Too**: An early open-source model responded to a coding request by saying, *"This sounds like a lot of work"*, showcasing **human-like reluctance** in AI responses.  
  - **Technical Limitations**: The issue might stem from using a **base model instead of an instruct model**. The OP confirmed using [**8b-instruct-q6_K**](https://ollama.com/library/llama3:8b-instruct-q6_K), suggesting other factors at play.
  - **Context Length Matters**: **Ollama's default context length of 2048** might have truncated the lengthy instruction. Increasing it to **8096** could potentially allow processing of the complete instruction.


- [/r/singularity] **[So many people simply cannot imagine tech improving](https://i.redd.it/5lx3ajibn0dd1.png)** ([Score: 354, Comments: 105](https://reddit.com//r/singularity/comments/1e5ahdm/so_many_people_simply_cannot_imagine_tech/)): **Rapid AI progress skepticism**: The post highlights the widespread inability of people to envision rapid technological advancements, particularly in AI. The author draws parallels to historical examples, such as the **Wright brothers' first flight in 1903** leading to the **moon landing in 1969**, to illustrate how quickly technology can progress and surpass initial expectations.
   - **Rapid AI progress skepticism debunked**: The **Engineering Magazine** in **Dec 1909** predicted limited potential for flying machines, yet **less than 40 years later**, the **Enola Gay** was operational. This highlights how **technological progress can outpace expectations**.
  - **Flying cars: Reality vs. imagination**: While predictions of flying cars by 2000 were misguided, **helicopters** and impractical flying car prototypes exist today. Some argue that **AI autopilots** are necessary for safe, widespread flying car adoption.
  - **Shifting AGI timelines**: Just **3-4 years ago**, **2045** was considered an optimistic estimate for AGI. Now, it's viewed as pessimistic. A **2023 survey** of **2278 AI researchers** estimated a **50% chance** of AI surpassing humans in all tasks by **2047**.
  - **Economic value drives AI advancement**: Unlike smartphone improvements, which have plateaued, AI advancements offer significant economic value. Companies are willing to pay substantial amounts for AI that can outperform human workers, driving rapid progress.
  - **Human limitations in grasping exponential growth**: Many people, including developers and entrepreneurs, struggle to anticipate and plan for exponential technological growth, despite being aware of trends like


**Theme 3. AI in Image and Video Generation**

- [/r/StableDiffusion] **[First Test with LivePortrait](https://v.redd.it/5lby8nan6xcd1)** ([Score: 226, Comments: 26](https://reddit.com//r/StableDiffusion/comments/1e4va0j/first_test_with_liveportrait/)): **LivePortrait test**: A user experimented with the **LivePortrait AI tool** to generate a video from a still image. The result was described as **"pretty good"**, with the AI successfully animating the image and matching lip movements to the provided audio, although some artifacts were noticeable around the mouth area.

- [/r/singularity] **[Celebrities hanging out with their younger selves](https://v.redd.it/rpcw3s5qpxcd1)** ([Score: 887, Comments: 137](https://reddit.com//r/singularity/comments/1e4zcv5/celebrities_hanging_out_with_their_younger_selves/)): **AI-generated images** depict celebrities interacting with younger versions of themselves, showcasing the capabilities of advanced image synthesis technology. These visuals blend **present-day appearances** with **historical photographs**, creating seamless and realistic composites that highlight the aging process and career evolution of well-known personalities. The images demonstrate the potential of AI in creating imaginative and nostalgic visual content, while also raising questions about the authenticity and manipulation of digital media.

- [/r/StableDiffusion] **[Underwater inside a bottle](https://v.redd.it/svfrbbb4tucd1)** ([Score: 323, Comments: 7](https://reddit.com//r/StableDiffusion/comments/1e4ks8j/underwater_inside_a_bottle/)): **Underwater bottle scene animation created using AI**. The artist used **Midjourney** to generate the initial image, then employed **Stable Diffusion** and **ControlNet** for inpainting and animation, resulting in a dynamic underwater scene contained within a glass bottle.
   - **OP Spills the Beans**: Artist reveals the **ComfyUI** workflow, including use of **RunwayML** for masking, **AnimateDiff** for animation, and **IPAdapter** with a reference image from **Lexica**.  - **ControlNet Combo**: Technique employed **depth and Canny** ControlNet, along with the **Reborn model** and **LCM Lora** for faster sampling.  - **Quick and Efficient**: Animation created with just **11 steps** and a **cfg of 2**, using the **LCM sampler** for rapid generation.

**Theme 4. New AI Model Releases and Architectures**

- [/r/LocalLLaMA] **[mistralai/mamba-codestral-7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1)** ([Score: 216, Comments: 72](https://reddit.com//r/LocalLLaMA/comments/1e4qgoc/mistralaimambacodestral7bv01_hugging_face/)): **Mistral AI** has released **Mamba-Codestral-7B**, a new **7 billion parameter** language model based on the **Mamba architecture**. This model, available on **Hugging Face**, is designed for **code generation** tasks and is trained on a combination of code and natural language data. The release marks a significant step in applying the Mamba architecture, known for its efficiency in processing long sequences, to the domain of code generation.
- [/r/singularity] **[[Google DeepMind] Mixture of A Million Experts. Daniel Jeffries:"Reduces inference cost and memory usage, scales to millions of experts, oh and just happens to overcome catastrophic forgetting and enable life long learning for the model."](https://arxiv.org/abs/2407.04153)** ([Score: 381, Comments: 82](https://reddit.com//r/singularity/comments/1e4mu0e/google_deepmind_mixture_of_a_million_experts/)): **Google DeepMind** has introduced the **Mixture of A Million Experts (MoME)** model, which reportedly **reduces inference cost and memory usage** while scaling to **millions of experts**. According to Daniel Jeffries, this model also addresses the challenges of **catastrophic forgetting** and enables **lifelong learning** for AI systems. The MoME approach represents a significant advancement in AI model architecture, potentially offering more efficient and adaptable systems.

- [/r/LocalLLaMA] **[I gave Llama 3 a 450 line task and it responded with "Good Luck"](https://i.redd.it/2n1oytw3pucd1.png)** ([Score: 383, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1e4kg7n/i_gave_llama_3_a_450_line_task_and_it_responded/)): **Llama 3's unexpected response to complex task**. When presented with a **450-line task**, Llama 3 reportedly responded with a simple "Good Luck" instead of attempting to complete it. This anecdote suggests potential limitations in Llama 3's ability to handle extremely large or complex prompts, raising questions about its performance on extensive tasks compared to other AI models.
  - **Prompt Engineering Matters**: Adding "Output:" or "Answer:" to the prompt could **significantly change Llama 3's response**. This highlights the importance of *proper prompt formatting* and the difference between **text completion and comprehension**.
  - **Context Length Limitations**: The default **context length in Ollama is 2048 tokens**, potentially cutting off lengthy instructions. Increasing it to **8096 tokens** might allow Llama 3 to process the complete 450-line task.
  - **Model Variation Impacts Performance**: The specific model used was [**llama3:8b-instruct-q6_K**](https://ollama.com/library/llama3:8b-instruct-q6_K). Some users suggest this behavior might be more typical of a **base model rather than an instruct-tuned version**.
  - **AI Mimicking Human Behavior**: Several users humorously noted that Llama 3's response of "Good luck" or "This sounds like a lot of work" mirrors typical **human reactions to complex tasks**, jokingly suggesting it demonstrates human-like intelligence.


**Theme 5. AI Regulation and Public Perception**

- [/r/singularity] **[Vance, new VP of Trump, on AI regulation](https://x.com/ai_for_success/status/1813036499329511900?t=p46Mncs0gfvyIb3LmCHiLw&s=19)** ([Score: 212, Comments: 418](https://reddit.com//r/singularity/comments/1e4n9m3/vance_new_vp_of_trump_on_ai_regulation/)): **J.D. Vance**, potential **Vice President** pick for **Donald Trump**, has expressed concerns about **AI regulation**. In a recent interview, Vance emphasized the need for a **"muscular" approach to AI governance**, suggesting that current regulatory frameworks are inadequate to address the rapid advancements in AI technology. He highlighted the importance of maintaining **American technological supremacy** while also protecting against potential risks associated with AI development.

- [/r/singularity] **[RIP students](https://v.redd.it/zsfxtxfizscd1)** ([Score: 434, Comments: 158](https://reddit.com//r/singularity/comments/1e4mp49/rip_students/)): **"RIP students"**: AI's impact on education is likely to be transformative. The post title suggests a pessimistic view of AI's effects on students, potentially implying that traditional student roles or learning methods may become obsolete or significantly altered due to AI advancements in education.

- [/r/singularity] **[So many people simply cannot imagine tech improving](https://i.redd.it/5lx3ajibn0dd1.png)** ([Score: 354, Comments: 105](https://reddit.com//r/singularity/comments/1e5ahdm/so_many_people_simply_cannot_imagine_tech/)): **"Tech Skepticism Persists Despite AI Advancements"**: Many people struggle to envision technological progress, particularly in AI, despite rapid advancements. This skepticism extends to the job market, where some individuals doubt AI's potential to significantly impact employment, even as AI capabilities continue to expand across various industries.
  - **"Flying Cars" Debate Takes Off**: Commenters discuss the **1909 Engineering Magazine** prediction about flying machines, noting that **helicopters** essentially fulfill this role. Some argue that **AI autopilot** would be crucial for safe flying cars in 3D space.
  - **AI Timeline Acceleration Shocks Experts**: Many express surprise at how **AGI estimates** have shifted dramatically. Previously, **2045** was considered optimistic for AGI; now it's viewed as pessimistic. Recent surveys suggest a **50% chance of AI surpassing humans in all tasks by 2047**.
  - **Tech Progress: Rapid Advances vs. Plateaus**: Discussion contrasts periods of rapid technological advancement with plateaus, using smartphones as an example. For AI, commenters highlight ongoing rapid improvements since **GPT-4** and the high economic value of AI advancements in various industries.
  - **Exponential Growth Challenges Human Comprehension**: Several comments point out that many people, including experts, struggle to grasp or anticipate exponential technological growth. This difficulty in imagining future capabilities leads to skepticism about AI's potential impact on jobs and society.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Advancements in AI Model Development and Deployment**

- **Codestral Mamba Makes Waves**: Mistral AI released [Codestral Mamba](https://mistral.ai/news/codestral-mamba/), a new model focused on code productivity that offers **linear time inference** and the ability to model infinite length sequences.
   - The model, designed with help from Albert Gu and Tri Dao, is available for free use, modification, and distribution, sparking excitement in the community for its potential in advanced code reasoning and quick responses.
- **SciCode Sets New Benchmark Bar**: The [SciCode benchmark](https://scicode-bench.github.io) was launched, featuring 338 programming challenges authored by PhDs in physics, math, and biology, with some based on Nobel-winning research.
   - This new benchmark proved challenging for current AI models, with **GPT-4** and **Sonnet 3.5** scoring less than 5% accuracy, highlighting the gap between current AI capabilities and advanced scientific problem-solving.
- **SmolLM Brings AI to Browsers**: HuggingFace introduced **SmolLM models** (135M, 360M, 1.7B parameters) designed to run locally in browsers using ONNX weights and WebGPU acceleration.
   - These models represent a significant step towards making AI more accessible and performant in web environments, potentially opening up new possibilities for client-side AI applications.
  


**2. Challenges and Innovations in AI Infrastructure**

- **SF Compute's $12M Boost for GPU Trading**: SF Compute raised $12 million to develop a trading platform for large-scale GPU clusters, allowing reservations of substantial GPU resources and the ability to sell unused portions.
   - This initiative aims to address the growing demand for GPU computing power in AI research and development, potentially making high-performance computing more accessible and efficient for a wider range of organizations.
- **LAION's Cybersecurity Wake-Up Call**: The LAION community was targeted by a sophisticated hacker group that created malware disguised as a ComfyUI node called **ComfyUI_LLMVISION**, designed to steal information and install trojans.
   - This incident highlights the increasing cybersecurity risks in the AI community, especially given the group's history of high-profile attacks, including infiltrating Disney's Slack.
- **Mojo's Performance Puzzle on Intel Chips**: Discussions in the Modular Discord revealed that **Mojo**'s `parallelize` function exclusively utilizes performance cores on Intel chips with both performance and efficiency cores.
   - This design decision stems from challenges in efficiently distributing work between different core types, prompting debates about optimal resource utilization in heterogeneous computing environments.
  
**3. DeepSeek V2 Model Launch**

- **DeepSeek's Guidance Gone Awry**: @davidkpiano shared [a link about state machines in the cloud](https://x.com/DavidKPiano/status/1806417216914817514), sparking a discussion on **DeepSeek-Coder V2-Lite issues** where the model doesn't follow prompts and provides erratic answers.
   - @dimfeld pointed out that disabling **flash attention** did not resolve the problem, suggesting **LM Studio updates** might have broken DeepSeek-Coder V2-Lite's support.
- **Deepseek Stays the Open-Source Course**: **Deepseek's founder Liang Wenfeng** voiced dedication to open-source in [an interview](https://x.com/main_horse/status/1813580480761196987?s=46), seeing it as crucial for a robust technical landscape, amidst concerns of **China's AI pace**.
   - Wenfeng's resolve remains strong, despite Deepseek's modest profits, emphasizing the importance of having a strong technical ecosystem first before considering closed-source options.
 

**4. New Multimodal Benchmarks**

- **InternVL2-Llama3-76B Vision**: [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) takes a leap in **multimodal learning**, pushing boundaries with instruction-tuned models ranging from 1B to 108B parameters.
   - Users expressed frustrations running **large 40B models** on **4x 3090 GPUs**, with issues surrounding the use of **autoawq** for optimization.
- **SciCode's STEM PhD Upgrade**: **SciCode** sets a new precedent with a benchmark of coding scientific problems, with nods to Nobel laureates, that stumped giants like **GPT-4** and **Sonnet 3.5**, revealing sub 5% accuracy. [Go deeper](https://scicode-bench.github.io).
   - The **SciCode benchmark** challenge composed by PhD specialists spans 338 problems, shedding light on diverse scientific domains. [Insights here](https://x.com/OfirPress/status/1813202497864937825).
  



---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CUDA Quandaries & VRAM Ventures**: Technical discussions focused on **CUDA errors**, including illegal memory access during training sessions, without a clear solution.
   - For managing **VRAM with large models**, like phi-3-mini, techniques such as *flash attn2* and **RAG** restructuring were proposed to address OOM scenarios.
- **Math Annotation's Growing Significance**: The need for **math data annotation** was debated to enhance training in advanced models, sparking a new study on its role and presence in current datasets.
   - In parallel, community advice was sought for **Stable Diffusion implementation on Next.js**, guiding towards the use of [diffusers.js](https://github.com/dakenf/diffusers.js) and additional learning resources.
- **Shape of Things: Generative 3D Learning**: Deep learning's potential in **3D shape generation** was showcased through a review of challenges with representations, underlining progress in GANs and form representation.
   - The increase in **time series forecasting** accuracy was evidenced by NBEATSx's 20% improvement over its predecessor, particularly noted in electricity price forecasting.
- **Channeling AI Creativity into Tools**: An AI Vtuber called **Rose** sought community testing through a [live YouTube session](https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po), while a **Fast Subtitle Maker** tool was introduced, leveraging Groq API's whisper-large-v3 model.
   - For the Mac enthusiasts, **Phi-3 Vision for Apple Silicon** was debuted, promising optimized performance, alongside a **YouTube Video Transcription Tool** to aid content creators.
- **Paper Implementations & ConvNet Chronicles**: A request for foundational papers suitable for learning through implementation was met with a suggestion exploring self-attention and implicit representation.
   - Elsewhere, the past prestige of the **Inception model** in using intermediate features leading up to the current reliance on **ResNet** was examined.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Beta Buzz**: Enthusiasts discuss **Unsloth AI beta** testing under NDAs, floating licenses for **multi-GPU support**, and speculate on forthcoming features.
   - Comments indicate the free version lacks multi-GPU use, while a subscription-based version is underway, and some testers have early access.
- **Karpathy's LLM Learning Path**: Celebrated AI figure Andrej Karpathy unveils **LLM101n course**, stimulating discussions on his new venture [Eureka Labs](https://github.com/karpathy/LLM101n).
   - The curriculum, keenly anticipated by the community, promises to cover vast aspects like **transformers and fine-tuning**.
- **Hot-Swapping LoRA in llama.cpp**: **LoRA adapter support** in llama.cpp ignites debate, following an update enabling adapter hot-swapping for enhanced model flexibility.
   - Mixed feedback loop on quantized models adapting to new LoRAs, particularly concerning cloud deployment reliability.
- **Debating RAG vs. Fine-Tuning**: A keen debate ensues on the effectiveness of using **RAG versus fine-tuning**, with recognition for RAG's ease but finer points for fine-tuning for complex tasks.
   - Some suggest a hybrid approach could yield superior outcomes, indicating a shift towards more personalized training methods.
- **AdamW-Mini Cuts Memory Usage**: **Optimizer state costs** in neural network training spark discussion, with AdamW-mini observed to potentially halve memory usage.
   - This could allow for **doubling batch sizes**, marking a stride forward in efficiency for large-scale training.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPUs: The Art & Missteps**: A user celebrated GPU craftsmanship with a **rose gold facelift**, highlighting the underrated role of aesthetics in hardware.
   - Meanwhile, another member confessed to a rookie mistake: forgetting to plug in the **GPU power**, a helpful nudge for all to double-check their rigs.
- **Mathstral: STEM's New Brainiac**: **Mathstral's debut** in LM Studio sparked excitement, boasting impressive strength in STEM and advanced reasoning capacities compared to its Mistral 7B base.
   - Its specialty in **logic and math problems** paired with GGUF quantization by **bartowski** makes it an attractive tool for techies looking for an AI edge.
- **DeepSeek's Guidance Gone Awry**: **DeepSeek-Coder V2-Lite issues** troubled users, with its erratic responses defying prompts, indicating potential conflicts with LM Studio updates.
   - Attempts to correct its path, including disabling flash attention, proved unsuccessful, leaving members searching for a fix.
- **Fine-Tuning: A Potential 'G' Whiz**: One user's struggle with fine-tuning **Codestral** underscored challenges in tweaking LLMs, as they grappled with the model's nonsensical 'G' responses.
   - Community discourse suggested that rich documentation and harnessing collective wisdom may help navigate these **fine-tuning frustrations**.
- **Sizing Up Models for Micro Decisions**: Curiosity about the right LLMs for micro decisions like **NER** and content filtering led to discussions promoting smaller, compute-efficient models.
   - Experts in the guild underscored the importance of optimal configurations in **hardware setups** to enhance model performance for these focused tasks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Maximizes Performance Cores**: Discussions highlighted that **Mojo** uses performance cores on Intel chips for `parallelize` functions, optimizing operation despite not leveraging efficiency cores.
   - The runtime's current limitations in core utilization decisions promise enhancements in forthcoming updates, optimizing core usage for performance gains.
- **NumPy vs Mojo: The Speed Showdown**: Benchmarks unveiled **Mojo** outranking **NumPy** in speed, despite Mojo not utilizing all available cores, and the performance gap was ascribed to BLAS backend selections.
   - While **OpenBLAS** is commonly used, the **Intel MKL** has been recognized for superior speed, even on non-Intel CPUs.
- **Inline Ingenuity in Mojo**: A suggestion was made for a shorthand to denote `@always_inline("nodebug")`, with the consensus that inline functions in **Mojo** should be concise.
   - This syntax proposal aims to reduce code verbosity without sacrificing clarity or functionality.
- **Beyond Dual Core: SIMD and SVE**: Within the SIMD context, the flexibility of **SVE** for non-2-multiple sizes was brought to light, with the potential for drainage loops or masks to enhance performance.
   - This discussion revolved around optimization techniques to amplify computational efficiency across diverse architectures.
- **Inside the Mojo Compiler Updates**: The newest **Mojo compiler** nightly release `2024.7.1714` prompted users to upgrade with `modular update nightly/mojo`, featuring significant updates like built-in SIMD methods and Dict initializations.
   - The changes, explained in the project's [GitHub changelog](https://github.com/modularml/mojo/commits), reflect the ever-progressing evolution of the language and its standard library.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DCLM Shakes Up the Scene**: The [DataComp for Language Models (DCLM)](https://arxiv.org/abs/2406.11794) emerges as a robust testbed for controlled dataset experiments designed to boost language model efficacy.
   - **DCLM-Baseline-7B** outshines MAP-Neo in 5-shot MMLU accuracy by **6.6%**, showcasing efficient compute utilization on the [Hugging Face model page](https://huggingface.co/apple/DCLM-Baseline-7B).
- **Translation Triumph with Replete-AI**: **Replete-AI** makes headlines by introducing an open source [multilingual translation dataset](https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct) consisting of over **2.8 million data points**.
   - These entail translations from English into an impressive lineup of **14 languages**, setting the tone for Multilanguage modeling advancement.
- **Oxen.AI Invites LLM Minds**: Zhengxuan Wu, author of an insightful paper, is slated for a discussion on **Representation Finetuning** at the [Oxen.AI Paper Club](https://lu.ma/oxen) event.
   - Discourse on **ReFT** garners interest for its avant-garde approach to optimization in comparison to traditional PEFT methods.
- **Belief State Geometry Unwrapped**: A new [Belief State Geometry study](https://arxiv.org/abs/2405.15943) uncovers how transformers model belief updates internally, capturing the LLM community's attention.
   - Feedback on the implications of this geometrical representation within residual streams ranges from admiration to skepticism.
- **Hermes 2.5 Epitomizes Benchmark Bravery**: In a stir of benchmark results, **Hermes 2.5** commands a lead with a significant jump on the MMLU, as demonstrated by [code instruction examples](https://link.to.examples).
   - Navigating through synaptic improvements, Hermes 2.5's MMLU score of **52.3** signals a breakthrough against its predecessor's **34.5**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pile 2 Confusion Cleared**: Clarification emerged that **The Pile 2** doesn't exist, leading to rectification among users.
   - Discussions pivoted to Proof-Pile-2 dataset, detailing it as a 55 billion token collection of mathematical and scientific documents, found on [Hugging Face](https://huggingface.co/datasets/EleutherAI/proof-pile-2).
- **Scraping Scandal Scrutiny**: The use of **YouTube videos** for AI datasets without consent sparked debate following a [Proof News article](https://www.proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/).
   - Artists like [Philosophy Tube](https://x.com/PhilosophyTube/status/1813227210569920685) and [Jacob Geller](https://x.com/yacobg42/status/1813226763117367688) posted responses, igniting talks on ethics and impact.
- **Transformer Engineering Explored**: Debate surrounding **Transformer optimizations**, with specifics on [TransformerEngine's fused layers](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py), revealed misunderstood capabilities.
   - Discussions highlighted RMSNorm's potential over other normalization techniques for enhancing processing efficiency.
- **Arrakis Library Unpacked**: Introducing [Arrakis](https://github.com/yash-srivastava19/arrakis), a mechanistic interpretability library designed for rapid prototype testing, still in the nascent stage.
   - Feedback and comparisons with existing tools like TransformerLens were encouraged to refine and validate Arrakis' unique offerings.
- **Leaderboard Legitimacy Queried**: Inquiry made into the calculation of musr raw score on the HF leaderboard; particularly, whether it represented an average of specific tasks.
   - Advice to contact [leaderboard maintainers](https://huggingface.co/leaderboard) was given to clear up potential ambiguities.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU Grapples with Gigantic Models**: Discussions revealed that **VRAM size is crucial for model performance**, with larger models demanding excessive VRAM, potentially leading to **Out of Memory (OOM) errors** if not managed properly.
   - An emphasis was made on distinguishing between extended generation times and memory issues; longer times don't automatically signal memory woes.
- **Artful Training for Illustrated Imaginations**: The community exchanged insights on training distinctive illustration styles, such as crosshatch techniques, highlighting the importance of **regional prompting** and **multi-concept models**.
   - Resources like [HuggingFace's T5](https://huggingface.co/jtlicardo/flan-t5-small-coref) were spotlighted as instrumental for these artistically inclined training endeavors.
- **Picky Prompts Produce Peculiar Pictures**: A lively debate unfolded over the influence of subtle prompt variations on outcomes, with phrases like 'harvesting potato' versus 'potato harvesting' sparking discussions on models' coreference capabilities.
   - Enthusiasts recommended tuning into T5's fine-tuned models to adeptly tackle the tricky nuances of complex prompts.
- **Outpainting Outpours Opportunities**: Exploration of outpainting methods to extend generated images included mentions of using Photoshop tools and KSampler wrapped in ComfyUI for seamless image expansions.
   - Participants shared methods to manage seed consistency, ensuring expanded visuals remain unified without overlapping segments.
- **Troubleshooting Tips Tackle Technicalities**: Members using Automatic1111 encountered setbacks with model performance, prompting a knowledge exchange on **command line fixes** tailored to specific hardware needs.
   - Options like 'xformers' and 'medvram-sdxl' were offered up as solutions to enhance model efficacy on machines with modest hardware configurations.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kernel Confusion: Templates Tame CUDA Catastrophes**: An initial hurdle with a **CUDA kernel call error** was overcome by specifying the template type `<int>`, in alignment with recommended CUDA practices.
   - The hands-on lesson: including the right template argument can make the difference between a functioning kernel and a frustrating debugging session.
- **PyTorch Profiler Exports: Marathon or Sprint?**: The **PyTorch Profiler** sparked debate when exporting a trace took upwards of **30 minutes**, leading to suggestions like turning off the `profile_memory` and `with_stack` options.
   - Cost-benefit analysis: faster exports may result, but at the potential cost of detailed memory allocation insights.
- **CUDA Meets PyTorch: Bridging Custom Kernels**: Quest for content led **artificial_anteligence** to inquire about integrating custom CUDA kernels with PyTorch, specifically for simplifying model implementations.
   - Cross-reference between frameworks is necessary, with a community member highlighting resources on how `load_inline` can be a starting point for kernel compilation.
- **Tensor Subclasses Tangle in PyTorch Nightly**: Using **unwrap_tensor_subclass** presented challenges, especially when an **IntxTensor subclass acts** as the `layout_tensor`, with a thread on GitHub addressing the complications ([Issue #515](https://github.com/pytorch/ao/issues/515)).
   - The conundrum: nested subclasses may impede operations, complicating backend development.
- **Triton Tactics and Puzzles: Streamlining the Execution**: **Triton Puzzle 6** had engineers scratching their heads over notation, seeking clarity on function definitions involving **ReLU** and matrix-vector operations.
   - An **ImportError** with 'interpreter_builder' from 'triton.runtime.interpreter' has members seeking stability, highlighting the critical nature of maintaining backward compatibility.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **API Limits May Throttle Projects**: Discussions in #[pplx-api] highlighted concerns about the **API rate limits** being too restrictive, potentially impacting project timelines.
   - Users are advised to fill out a request form and consult with a Perplexity representative for solutions to alleviate limit concerns.
- **Cloudflare CAPTCHA Catching Heat**: Members in #[general] channel aired grievances over the CAPTCHA system implemented by **Cloudflare**, calling into question the decision-making behind its usage.
   - Community feedback included remarks on Cloudflare's security issues, with one comment pointing out that *Cloudflare is constantly breaking or being broken into*.
- **Perplexity API Beta Unlocks New Filtering**: A valuable addition to the Perplexity API, the **`search_domain_name` filter**, is now accessible for beta users, as per the discussions in #[pplx-api].
   - This feature enables more focused search capabilities, allowing for enhanced result filtering within specified domains.
- **Quality Quandaries: Code Calamities Questioned**: In #[general], a member mentioned a major company's quality control allowing untested code into production, sparking a candid conversation about industry practices.
   - *Every company be like,* sarcastically highlighted one member, reflecting a sentiment of resignation towards widespread quality control issues.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Error Code 524 Crescendo**: A slew of users encountered **Error Code 524**, sparking a quickfire exchange on its sudden prevalence.
   - Prompt inquiries sprung up, probing into whether this anomaly was an isolated case or indicative of a pervasive hiccup.
- **Meta 405B's Monetary Mystery**: Anticipation builds as users speculate on [Meta 405B's potential price point](https://discord.com/channels/1091220969173028894/1094454198688546826/1262737636607528972), pondering its debut around the 23rd.
   - **8K context windows** floated as a benchmark from past models, while precise details are eagerly awaited.
- **Deepseek Coder: Compelling but Crawling**: "Capable yet crawling" sums up the sentiment around **Deepseek Coder**, whose lethargic performance has users yearning for speed.
   - The chorus of discontent signals a market opportunity for a sprightlier rival to captivate those spurned by slothful service.
- **OpenRouter's Quest for Quick & Cheap AI**: The hunt for models that outpace **GPT-3.5-Turbo** without breaking the bank has users weighing options like **Claude-3-Haiku** amidst cost-context conundrums.
   - Llama models are poised as contenders in this quest, signaling a dynamic debate on what constitutes speed to spare and frugality of fare.
- **WordPress Woes with OpenRouter API**: **RSS feed integration travails** torment a user struggling to meld the OpenRouter API within a WordPress ambit, triggering talks of troubleshooting.
   - API key intricacies and rate limit riddles dominate discourse, with `curl` verification touted as a technical touchstone.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Malicious Maneuvers in Model City**: [ComfyUI_LLMVISION](https://darknetdiaries.com/transcript/133/) malware targets **LAION community**, stealing data and installing trojans on unsuspecting victims’ devices.
   - The hacker group, known for the Disney Slack intrusion, showcases their ability to craft convincing **fake job applicants** that clone GitHub engineer identities for data theft.
- **Sandy Sweeps Telecom into Future Fibers**: **Hurricane Sandy** takes out **Verizon's NY cable vault**, necessitating a swap from copper to fiber optics across an expanse of 13,000km.
   - This critical incident was a catalyst for upgrading infrastructure, as detailed in this [deep dive](https://www.datacenterknowledge.com/cables/after-sandy-verizon-confronts-catastrophic-failure-at-ny-cable-vault).
- **Vision and Verbiage Merging on the Multimodal Stage**: The new [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) takes a leap in **multimodal learning**, pushing boundaries with instruction-tuned models.
   - On a related note, frustrations are voiced over running **large models** on **4x 3090 GPUs**, with issues surrounding the use of **autoawq**.
- **Manifold Musings on Mechanized Management**: **Manifold Research Group** releases a position paper titled [*Intelligent Digital Agents in the Era of Large Language Models*](https://www.manifoldrg.com/llm-agents/), pushing the conversation on **LLM-based AI agents**.
   - They invite the community to join the discourse on [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com), witness their progress in [Research Log #041](https://www.manifoldrg.com/research-log-041/), and contribute to their expansive **MultiNet** project [on GitHub](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Games of Proof & Puns**: [OpenAI's latest repository](https://openai.com/index/prover-verifier-games-improve-legibility/) introduces Prover-Verifier Games to enhance **AI model legibility**, challenging the notion that complexity is a 'legibility tax'.
   - Community exchange suggested this could rectify models narratively 'taxing' to understand, epitomized by a quip of the research paper's own '*legibility tax*'.
- **Reinforcement's Odd Results**: Conversations circled around how **Reinforcement Learning (RL)** tweaks model traits, implying that complex figures could bear a so-called '*legibility tax*'.
   - One member's remark, '*this figure is definitely a legibility tax*,' pointed to firsthand observations of RL's peculiar influence.
- **GPT-4: Tokenizer Tango**: A vibrant discussion compared **GPT-4o** and **Llama 405** tokenizers, highlighting GPT-4o's regression in coding language token efficiency versus its predecessor, **GPT-4t**.
   - Details mention GPT-4o yielding more tokens in XML than GPT-4t, signaling a step back in specialized tokenizer performance.
- **Deepseek Stays the open-source Course**: **Deepseek's founder Liang Wenfeng** voiced dedication to open-source, seeing it as crucial for a robust technical landscape, amidst concerns of China's AI pace.
   - Wenfeng's resolve remains strong, despite Deepseek's modest profits, as stated in [an interview on social media](https://x.com/main_horse/status/1813580480761196987?s=46).
- **Sampling Chaos in Policy Models**: The Nemotron paper criticizes prevalent sampling methods in policy models, suggesting that some rejections are far worse than others, creating risk for overfitting and quality loss in DPO algorithms.
   - Meanwhile, Zephyr's paper promotes diversity through random sampling, looking to balance the challenge against DPO's objectives and avoid wrong direction due to false negatives.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Benchmarking Nobel Efforts**: SciCode Aces the Test**: SciCode sets a new precedent with a benchmark of coding scientific problems, with nods to Nobel laureates, that stumped giants like **GPT-4** and **Sonnet 3.5**, revealing sub 5% accuracy. [Go deeper](https://scicode-bench.github.io).
   - The **SciCode benchmark** challenge composed by PhD specialists spans 338 problems, shedding light on diverse scientific domains. [Insights here](https://x.com/OfirPress/status/1813202497864937825).
- **Browser-based AI Brilliance**: HuggingFace Unveils SmolLM**: HuggingFace introduces **SmolLM models** optimized for browser environments, boasting ONNX and WebGPU acceleration. Delve into the update [here](https://x.com/xenovacom/status/1813258097185448377).
   - The new **SmolLM models** range from 135M to 1.7B, designed for efficient, on-device AI applications, showcasing progressive on-browser capabilities.
- **GPU Trading Groundbreaker**: SF Compute Attracts Investment**: **SF Compute** closes a successful $12M fundraising round, earmarked for constructing a novel GPU trading platform. [Details](https://www.bloomberg.com/news/articles/2024-07-16/jack-altman-s-firm-backs-startup-for-trading-ai-computing-power).
   - This influx of funds will facilitate the reservation and trade of substantial GPU clusters, introducing fluidity to computational resource allocation.
- **Exa AI's Expansion Era**: Series A Sparks Growth**: Backed by heavy-hitters like Lightspeed, Nvidia, and Y Combinator, **Exa AI** secures Series A funds to enhance their LLM-powered search engine API. [Explore more](https://x.com/exaailabs/status/1813249325394456686).
   - Although Exa AI is expanding, the community discusses challenges around prompt optimization and benchmarking against APIs like **Preplexity**.
- **Disrupting Documents with ColPALI**: A Vision for Efficient Retrieval**: ColPALI, introduced by **HuggingFace**, promises a revolution in document retrieval, making traditional OCR solutions obsolete. [Learn more](https://huggingface.co/blog/manu/colpali).
   - **HuggingFace's ColPALI** offers a proficient approach to document processing, combining vision-language models for higher efficiency. [Further discussion](https://x.com/jobergum/status/1813298149051802074).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Unveils Its Agentic Swagger**: An [introductory video](https://twitter.com/llama_index/status/1813316626793853135) gave a tour of **LlamaIndex**'s agentic capabilities, showcasing Python and TypeScript frameworks with a nod to the LlamaParse service, igniting buzz for its parsing prowess.
   - Members praised the **LlamaParse** advances, highlighting its new markdown-based table reconstruction and its finesse for dealing with complex tables as shared in [this tweet](https://twitter.com/llama_index/status/1813355957491273936).
- **Navigating the Labyrinth of Query-time Metadata**: Community gurus exchanged ideas on applying metadata filters at query-time and weighed in different approaches, questioning the efficacy of existing retriever instantiation methods.
   - A mix of proposed solutions and lingering questions showcases the non-trivial nature of improving document storage and indices.
- **Neo4J Property Graph Puzzle Persists**: When the Neo4J property graph failed to remember repeating entities, community sleuths recommended potential fixes like entity linking adjustments.
   - Conversations fused theory with practice, dropping hints with **'Entities'** and 'MENTION' relations and Cypher query snippets, that could offer a light at the end of the tunnel.
- **Scaleport Syncs with Streamlined AI Solutions**: In a testament to the versatility of LlamaIndex, Scaleport AI utilized LlamaCloud and **LlamaIndex** technologies to condense their AI development timelines and enhance OCR results, as detailed in [their case study](https://twitter.com/llama_index/status/1813647179627774462).
   - **OCR optimization** and agile AI development emerged as themes in the **Scaleport AI** narrative, underscoring the impact of pairing innovative frameworks with client projects.
- **Cracking The Code Of CSV Chaos**: Commotion ensued over troubles tackling CSV data exceeding 50 rows in VectorStoreIndex, with members dissecting missteps and pondering on proficient parsing pathways.
   - While the PagedCSVReader fell short, there was collective agreement that tools like [PandasAI](https://docs.pandas-ai.com/intro) might offer refuge and a remedy for complex record-based CSV operations.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **CrunchCup Chaos: Not Quite Dishwasher-Durable**: A member's excitement for their new [CrunchCup](https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY) was marred by its failure to withstand a dishwasher cycle, despite its convenience for on-the-go cereal consumption.
   - The community chimed in with reviews, ranging from appreciation for its portable design to frustration over its unexpected lack of durability, with some mentioning it **deforms when machine-washed**.
- **Roger Grosse Talk Tackles LLM Generalization**: Roger Grosse's latest session, *"Studying LLM Generalization through Influence Functions,"* is now live, and the link was shared showing [his insights on YouTube](https://youtu.be/64BsnVbX5u8).
   - A shoutout by *danylo_boiko* pointed members to catch up on the **latest LLM research insights** through the direct video link.
- **Cohere's Community Call Catch-up on YouTube**: For those who missed out, **Cohere's community event talks**, including rich discussions and sessions, are available on their [YouTube playlist](https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll).
   - Keeping the guild updated, attendees were directed to witness the recordings of their **favorite AI luminaries** and stay abreast with community endeavors.
- **Cereal Showdown: Kids' Table or Not?**: A playful guild debate on cereal preferences sparked engagement with **Fruit Loops** and **Special K** taking center stage.
   - While no consensus emerged on the age-appropriateness of Froot Loops, the conversation underscored the diversity in breakfast choices among engineers.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Chatbots Tailored to Task**: Personalization Push or Privacy Pitfall?**: Debate ignited over **fine-tuning** custom chatbots for specific websites using models like OpenAI API, with focus on *pre-prompting* to embed company knowledge.
   - **Expenses questioned** in using detection services for chatbots, with cost-effective measures like manual moderation suggested due to high fees of $20,000 per month.
- **Voices Unveiled from Noise**: A Sound Solution for Podcasts?**: Discussions surfaced about tools for **voice extraction** from podcasts, spotlighting Eleven Labs' model for its ability to separate voices without disruptions.
   - This topic was less of a priority, yet it opened up avenues for improving content accessibility and metadata extraction from audio sources.
- **Limits of Learning**: GPT Agents Grasp for Context**: Conversations tackled the context limitations of **GPT agents**, notably their struggle to keep up with ongoing discussions due to fixed context windows.
   - Members exchanged tips on **PUT versus PATCH requests** and addressed **vector store embeddings**, highlighting challenges with name recognition in RAG chatbots.
- **Surfing Against the Current**: WebSurferAgent's Selective Searches**: The **WebSurferAgent** drew attention for sporadically ignoring setup instructions during searches, pointing to potential improvements in instruction adherence.
   - A shared template for **role-playing** in ChatGPT revealed the potential for more immersive, character-driven interactions in conversational AI.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hannah Hype: Custom AI Assistant**: Introducing **Hannah**, a new generative AI assistant enabling advanced features like learning from documents and **extreme customization**, integrated with APIs from OpenAI to NVIDIA.
   - The assistant is underpinned by popular AI APIs like **OpenAI**, **Anthropic**, and **Cohere**, and info is available on [the Hannah website](https://hannah.yourbestseller.ai/).
- **MongoDB Melds with LangChain for Hybrid Search**: Members seek guidance on using **MongoDB** as a vector store in a RAG application, emphasizing the need for **Hybrid Search** functionality.
   - While [MongoDB's own documentation](https://mongodb.docs/hybridsearch) covers Hybrid Search, community insights for integrating with LangChain are in high demand.
- **AI's Answer to Viral Sports Videos**: A surge in interest for AI tools capable of creating **viral sports YouTube shorts/TikToks**, with community members seeking specialized edits insights.
   - Skeptical of AI's ability to craft sports shorts, users are exploring and requesting tailored advice for generating such content.
- **Unstructured to Structured: LangChain's Document Conversion**: Discussions revolve around transforming unorganized data into usable LangChain documents, using `UnstructuredFileIOLoader` and similar classes.
   - With practical examples shared, users are utilizing **LangChain's tools** to structure data for improved application performance.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Codestral's Code Conquest**: Mistral AI has rolled out [**Codestral Mamba**](https://mistral.ai/news/codestral-mamba/), championing the frontiers of code productivity with features like **linear time inference** and handling infinite sequence lengths.
   - Developed by Albert Gu and Tri Dao, Codestral Mamba has sparked excitement among community members keen to test its capabilities for **advanced code reasoning**.
- **Mathstral: The Missing Model Mystery**: Curiosity peaked surrounding a model dubbed 'Mathstral', with questions arising about its existence and association with Mistral AI.
   - The discussion remains afloat without concrete details, suggesting either a developing model or a potential future project to keep an eye on.
- **Curbing Overfitting: On the Hunt for Solutions**: Suggestions to combat overfitting emerged, with strategies like **increasing rank** or **tweaking learning rates**, tailored to the model's unique learning journey.
   - Methods such as de-duplicating datasets are being shared as viable tools to prevent models from overfitting prematurely during training.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Handheld Hardware Huzzah for My Friend V1**: A tweet by [@ParallaxAngle](https://x.com/ParallaxAngle/status/1805313161567818030) conveys excitement for the surprisingly compact form factor of **My Friend V1**, applauding the Based Hardware team's effort.
   - The user praised the size and quality of the product, expressing affection with the phrase *'LOVE LOVE LOVE my Friend.'*
- **Transcription Trust Talks for AI Friend**: Privacy concerns were raised regarding transcription interaction via Open Interpreter with an AI Friend, emphasizing the importance of confidentiality in potential integrations.
   - Dialog focused on leveraging the Open Interpreter to ensure privacy when engaging with AI Friend's transcriptions, yet details about actual implementation remain uncertain.
- **Mac M3 Microchip Mystery with Open Interpreter**: Questions surfaced about whether Open Interpreter is compatible with the **M3 Mac**, with community members considering the potential for the Linux version to suffice.
   - Unofficial suggestions hinted that trying the build.py script could possibly lead to success after making adjustments for specifics like filepaths, though this remains unconfirmed.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.2.0 Unpacked**: The release of [Torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) brought in a slew of new models, recipes, and features like **sample packing**
   - This version marks a significant contribution from the **open-source community**, underlining the collaborative efforts towards improving the tool.
- **LLAMA 3's Finetuning Quirk**: **LLAMA 3** finetuning surfaced issues with **finetune_right_pad_id** tags appearing instead of the expected `<|end_of_text|>` during generation.
   - Switching from **Torchtune nightly builds** to the stable release may provide a temporary fix, while the tokenizer's old implementation is examined for discrepancies.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Linearizer Out, Updates In**: Queries about **updated notes** emerged post-removal of tinygrad's **linearizer**, spotlighting the community's keenness on documentation.
   - Echoes for clarity reverberated with a member requesting the **revised notes** to reflect the current state of tinygrad after a significant update.
- **Color Code Conundrum Clarified**: In the pursuit of **message format nuances**, clarification was sought on the color coding present in a member's notes.
   - Resolution arrived swiftly with direction to the color descriptions positioned at the **bottom of the first page**, ensuring no detail goes unnoticed.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **OpenAI Gateway to LLM Utility**: Kyle confirmed that **access on the OpenAI side** is crucial for specific LLM functionalities.
   - This access could enable a more streamlined application of LLMs, like automating hospital bill checks.
- **LLMs on the Billing Front**: Community discussion focused on the potential of **LLMs** in extracting rules from PDFs to audit hospital bills.
   - **Python code generation** by LLMs was considered to simplify the bill verification process.
- **Regrets of Missed Engagements**: A user lamented over not checking the #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1262890983864008805) channel after July, 9, missing significant discussions.
   - The sentiment underscored a missed chance to engage with critical channel updates and community interaction.
- **Code Suggestions for Compliance Checks**: There was talk of leveraging **LLM-generated test cases** to ensure the reliability of Python code for hospital bill auditing.
   - The initiative aims to make the most of LLM capabilities for practical applications in real-world scenarios.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Streaming Success Seekers**: [**Observe** invites developers](https://observeyourfuture.com) with a knack for **HLS** and **WebRTC** to apply their coding prowess in **Vanilla JS**, **TypeScript**, and **MongoDB**.
   - The hunt is on for backend development maestros passionate about startup ecosystems and the technical challenges of live streaming.
- **Startup Stars: TypeScript Talents Wanted**: Backend specialists, behold: **Observe** desires your **TypeScript** and **MongoDB** mastery for creating seamless streaming solutions.
   - Dive into the depths of startup culture and contribute your technical expertise to the dynamic field of **HLS** and **WebRTC**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Phoenix 2.0 Takes Flight with New Features**: Don't miss the **Phoenix 2.0 Product Update & Future Vision** event on July 18th, 2024, which will introduce new features like hosted deployment and experimentation capabilities as part of the [Phoenix 2.0 launch](https://arize.com/resource/phoenix/2.0).
   - Attendees will glimpse the evolution of **Phoenix** within the Arize product stack and engage in a live Q&A session, enriching their understanding of the tool's potential in LLM app development.
- **OSS: The Backbone of AI Advancement**: A **Town Hall on OSS in AI** will expound on how **Phoenix 2.0** streamlines development with features like new experimentation capabilities and the crucial role of Open Source Software (OSS) in AI.
   - User experience insights are a highlight of the agenda, emphasizing the synergy between community feedback and the progression of Phoenix functionalities.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Async Answers Awaken**: AI21 Labs' Python SDK now includes **async client support** and compatibility with **Jamba-Instruct** on platforms like **Amazon Bedrock** and **Azure AI Studio**.
   - Developers are encouraged to explore the new feature set provided in the [latest GitHub release](https://github.com/AI21Labs/ai21-python), which also showcases new examples for a better development experience.
- **Client Concurrency Cleared for Takeoff**: **Async client support** is now a standard feature for **Jamba-Instruct** across all interfaces, offering enhanced performance.
   - For hands-on guidance, developers can requisition new **Jamba-Instruct examples** to jumpstart their applications by visiting [AI21 Labs' GitHub repository](https://github.com/AI21Labs/ai21-python).



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1262846226852876328)** (286 messages🔥🔥): 

> - `Model Tokenization`
> - `CUDA Errors`
> - `VRAM Management for Large Models`
> - `Math Data Annotation Needs`
> - `Live Translation with Transformers` 


- **CUDA Errors Plague Training Sessions**: A user encountered persistent **CUDA errors** during training, with the message *'CUDA error: an illegal memory access was encountered'*. This issue remained unresolved despite attempts to apply suggested fixes.
- **Managing VRAM for Large Models**: Users discussed strategies for managing **VRAM usage** when running models like **phi-3-mini-128k**, which encountered OOM issues at ~50k context. Suggestions included using *flash attn2* and potentially restructuring to **RAG or summarization** approaches.
- **The Need for Math Data Annotation**: Inquiry into whether there's an increasing need for **math data annotation** in training advanced models like LLMs revealed community interest in the topic. A user initiated a study to understand the significance and current availability of **annotated math data**.
- **Implementing Stable Diffusion in Next.js**: A user asked about using **Stable Diffusion** in **Next.js**, prompting a recommendation for the [diffusers.js GitHub repository](https://github.com/dakenf/diffusers.js). Other members expressed interest in further resources like video tutorials.
- **Chatbot for CSV Data Queries**: A discussion emerged around developing a chatbot to handle **CSV data queries** efficiently. Suggestions included using models like **Llama 3** and integrating the chatbot with **pandas** for functional querying of tabular data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1">GLiNER-medium-v2.1, zero-shot NER - a Hugging Face Space by tomaarsen</a>: no description found</li><li><a href="https://github.com/dakenf/diffusers.js">GitHub - dakenf/diffusers.js: diffusers implementation for node.js and browser</a>: diffusers implementation for node.js and browser. Contribute to dakenf/diffusers.js development by creating an account on GitHub.</li><li><a href="https://www.nibbletechnology.com/demo">Nibble Demo</a>: Bet we can make you smile! Try out our AI negotiation chatbot for ecommerce, and see for yourself how the experience drives conversions and engagement</li><li><a href="https://tenor.com/view/tf2-gif-11289716861894092888">Tf2 GIF - Tf2 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS/discussions/1#6696e19ca7fd582ae724f59f">aheedsajid/Edge-TTS · 🚩 Report: Spam</a>: no description found</li><li><a href="https://www.evesleep.co.uk/products/the-premium-hybrid-mattress">Premium Hybrid Mattress - 28cm Spring &amp; Foam</a>: Shop the premium hybrid mattress, combining the comfort of pocket spring design with foam technology, for night time luxury. Try for 365 days, with free delivery and hassle-free returns.</li><li><a href="https://github.com/idiap/fast-transformers/issues/19">RuntimeError: CUDA error: an illegal memory access was encountered · Issue #19 · idiap/fast-transformers</a>: Hi, thanks for the great work! I install the package successfully but encounter an error during training: File &quot;/home/annahung/189nas/2020/fast_remi/linear_transformer/model.py&quot;, line 294, i...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1262982859522838589)** (3 messages): 

> - `SciPy tutorial`
> - `Audio course on Huggingface`
> - `Real-time kernels and Raspberry Pi` 


- **SciPy: More than Just Another Library**: A YouTube video titled [Intro to Scipy by Rauf](https://youtu.be/KAbNQwTBEyc?si=UorOWv5tJIPiCUYW) was shared, introducing **SciPy** as a data manipulation and scientific calculation library similar to **NumPy**.
- **Diving into Huggingface Audio Course**: A user mentioned starting the **Huggingface audio course** and asked for valuable suggestions on **TTS (Text-to-Speech)** and **ASR (Automatic Speech Recognition)**.
- **Frustrations with Real-time Kernels on Raspberry Pi**: A user expressed frustration with **real-time kernels** like rt-thread and freeRTOS for **Raspberry Pi 4**, noting compatibility issues with compilers.
   - They are considering coding a **kernel from scratch** with USB and HDMI peripherals due to limitations in their **current setup**.



**Link mentioned**: <a href="https://youtu.be/KAbNQwTBEyc?si=UorOWv5tJIPiCUYW">Intro to Scipy ( by Rauf )</a>: SciPy is another data manipulation and scientific calculation library, similar to NumPy, but with some differences. It&#39;s another tool in your toolkit, allowi...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1262947774433857576)** (4 messages): 

> - `Nbeats and NBeatsX paper`
> - `3D shape generation with deep learning`
> - `Time series forecasting`
> - `ML applications in 3D geometry` 


- **NBeatsX extends capabilities of NBeats model**: A [paper](https://arxiv.org/abs/2104.05522) extends NBEATS to NBEATSx, incorporating exogenous factors for better performance in time series forecasting, achieving nearly **20% improvement** over the original model.
   - The study showed **state-of-the-art performance** in electricity price forecasting (EPF) by integrating multiple sources of useful information.
- **Deep learning excels in 3D shape generation**: An [older article](https://www.sciopen.com/article/10.1007/s41095-022-0321-5) reviews the use of deep learning in 3D shape generation, highlighting the challenges due to various representations like voxels, point clouds, and meshes.
   - The paper discusses advancements in deep generative models like GANs and emphasizes the importance of shape representation for quality generation of 3D shapes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2104.05522">Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx</a>: We extend the neural basis expansion analysis (NBEATS) to incorporate exogenous factors. The resulting method, called NBEATSx, improves on a well performing deep learning model, extending its capabili...</li><li><a href="https://www.sciopen.com/article/10.1007/s41095-022-0321-5">A survey of deep learning-based 3D shape generation</a>: &lt;p&gt;Deep learning has been successfully used for tasks in the 2D image domain. Research on 3D computer vision and deep geometry learning has also attracted attention. Considerable achievements ha...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1262885361311416493)** (24 messages🔥): 

> - `AI Vtuber testing`
> - `Chilean touristic data`
> - `Phi-3 Vision for Mac`
> - `ML for 3D model reduction`
> - `Fast Subtitle Maker` 


- **AI Vtuber needs testers**: [A YouTube video](https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po) titled "chatting/lofi with Rose! (AI Vtuber) [open source] pls test it lol" was shared for community testing.
- **Chilean touristic data available**: Shared a link to [Chilean touristic data](https://huggingface.co/datasets/RaulSalinasHerr/chilean_touristic_data) for potential use in testing and analysis.
- **Phi-3 Vision for Apple Silicon**: [Phi-3 for Mac](https://github.com/JosefAlbers/Phi-3-Vision-MLX) offers locally-run Vision and Language Models for Apple Silicon, shared on GitHub for community use.
   - The tool is designed for seamless performance on Apple Silicon, appealing to developers working on Mac.
- **Fast Subtitle Maker for Videos**: [Fast Subtitle Maker](https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker) can generate subtitles quickly using Groq API's whisper-large-v3 model, suitable for those lacking powerful PCs.
   - Users can obtain subtitles files or directly embed them into videos with customizable settings like font and color.
- **YouTube Video Transcription Tool**: A tool for transcribing and summarizing YouTube videos using Deepgram and Claude was shared, useful for content creators and researchers.
   - Users can [try the tool](https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff), customize the template, and read a [blog post](https://hunch.tools/blog/video-transcription-and-summary-tool/) about it.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker">Fast Subtitle Maker - a Hugging Face Space by Nick088</a>: no description found</li><li><a href="https://github.com/JosefAlbers/Phi-3-Vision-MLX">GitHub - JosefAlbers/Phi-3-Vision-MLX: Phi-3 for Mac: Locally-run Vision and Language Models for Apple Silicon</a>: Phi-3 for Mac: Locally-run Vision and Language Models for Apple Silicon - JosefAlbers/Phi-3-Vision-MLX</li><li><a href="https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)">Hunch - AI Tools for Teams</a>: Create AI workflows and tools to automate knowledge work and boost team productivity</li><li><a href="https://www.youtube.com/live/Le5O8Z8NiUY?si=b_kjhaE3qBKSQ8Po">chatting/lofi with Rose! (AI Vtuber) [open source] pls test it lol</a>: no description found</li><li><a href="https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory">AI Comic Factory - a Hugging Face Space by jbilcke-hf</a>: no description found</li><li><a href="https://app.hunch.tools/app/canvas/new/vyg7V?invitationCode=u54c55ff)">Hunch - AI Tools for Teams</a>: Create AI workflows and tools to automate knowledge work and boost team productivity</li><li><a href="https://huggingface.co/datasets/RaulSalinasHerr/chilean_touristic_data">RaulSalinasHerr/chilean_touristic_data · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1262965178648100905)** (6 messages): 

> - `Learning by Implementing Papers`
> - `Inception Model and ResNet`
> - `Implicit Representation` 


- **Choosing Papers for Learning**: A member expressed interest in learning by implementing papers from scratch and asked for recommendations on which papers to start with.
   - Another member recommended their own work for learning implicit representation, self-attention, channel-attention, and other concepts as a good starting point.
- **Intermediate features and models**: A discussion highlighted that the **Inception model** is a multi-branch architecture and utilizes intermediate features to assist classification, until the emergence of ResNet.
   - *Inception model utilizes intermediate features to assist classification*, according to a member's comment.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262967896733122600)** (3 messages): 

> - `Skin Cancer Classification Model`
> - `VQModel Pre-trained Weights`
> - `Attention Extraction from GhostNetV2` 


- **Skin Cancer Classification Project Shared**: A member shared a link to [their GitHub repository](https://github.com/Matthew-AI-Dev/AI-Portfoilio/blob/master/SkinCancerClassification_CNN/SkinCancerClassification.ipynb) for a **Skin Cancer Classification** project using CNN.
   - The project includes a notebook and detailed description for classifying skin cancer images.
- **Inquiry about VQModel Pre-trained Weights**: A member asked if anyone knows about the availability of **pre-trained weights for VQModel**.
- **Attention Extraction from GhostNetV2**: A member sought help with extracting attention features from **GhostNetV2** while using the **timm library**.
   - They had tried using the `AttentionExtractor` from `timm.utils` but found it unhelpful and are looking for further assistance.



**Link mentioned**: <a href="https://github.com/Matthew-AI-Dev/AI-Portfoilio/blob/master/SkinCancerClassification_CNN/SkinCancerClassification.ipynb">AI-Portfoilio/SkinCancerClassification_CNN/SkinCancerClassification.ipynb at master · Matthew-AI-Dev/AI-Portfoilio</a>: Contribute to Matthew-AI-Dev/AI-Portfoilio development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1262877813644722176)** (3 messages): 

> - `System requirements for stability AI model`
> - `Prompt engineering for video generation`
> - `Stable Video Diffusion Image-to-Video Model` 


- **System requirements for AI models questioned**: A user inquired about the system requirements for running the **stability AI model**.
- **Prompt engineering for effective video generation**: A user sought ideas on ideal prompt engineering for creating good videos with the **image-to-video stability AI model**.
   - "What prompt should I pass along with the rocket image to make it a moving rocket?" was a specific question raised.
- **Stable Video Diffusion Image-to-Video Model shared**: A user shared a [link to the Stable Video Diffusion Image-to-Video model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) on HuggingFace.
   - The model can convert a still image into a video, producing up to **25 frames** at a resolution of **576x1024**.



**Link mentioned**: <a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt">stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face</a>: no description found

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1262850045858943117)** (164 messages🔥🔥): 

> - `Unsloth AI beta testing with NDA`
> - `Floating license for multi-GPU support`
> - `Andrej Karpathy's new LLM101n course`
> - `LoRA adapter support in llama.cpp`
> - `Fine-tuning vs. RAG in Llama-3` 


- **Unsloth AI beta testing under NDA**: **Several users** discussed having access to a test version of Unsloth AI under NDAs, with mentions of technical aspects and floating licenses for multi-GPU support.
   - The free version does not support multi-GPU use, but a paid subscription version that does is in development with early access available to some users.
- **Andrej Karpathy launches LLM101n course**: Andrej Karpathy announced his new LLM101n course, covering topics like **bigrams, transformers, and fine-tuning**, part of his new company [Eureka Labs](https://github.com/karpathy/LLM101n).
   - The course is expected to include innovative pretraining techniques and will be available online with plans for digital and physical cohorts.
- **LoRA adapter hot-swapping support in llama.cpp**: The latest update in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8332) includes **hot-swapping LoRA adapters**, potentially improving model versatility.
   - *There are mixed reports* on the effectiveness and reliability of quantized models adapting to new LoRAs, especially in cloud environments.
- **RAG vs. Fine-tuning: Which is better?**: Users debated the merits of **RAG (Retrieve and Generate) vs. fine-tuning** for specific tasks, noting that RAG is quicker to implement but often yields poorer results.
   - Combining **fine-tuning with RAG** is suggested for better outcomes, though it involves more extensive customization and training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sample-contract-nda-non-disclosure-agreement-gif-17773157">Sample Contract GIF - Sample Contract Nda - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/danielhanchen/status/1813330269044408612">Tweet from Daniel Han (@danielhanchen)</a>: You don&#39;t want to miss Andrej&#39;s LLM101n course if you want to learn all the fundamentals of LLMs!  Chapters: 1-2-3: Bigrams, N-grams, Backprop, ML, Maths 4-5: Attention, Transformers, GPT2 6: ...</li><li><a href="https://www.deeplearning.ai/short-courses/pretraining-llms/">Pretraining LLMs</a>: Gain in-depth knowledge of the steps to pretrain an LLM, encompassing data preparation, model configuration, and performance assessment.</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">How to Finetune Llama-3 and Export to Ollama | Unsloth Docs</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8332">Refactor lora adapter support by ngxson · Pull Request #8332 · ggerganov/llama.cpp</a>: This refactor is inspired by the implementation of control vector, which has proper support for GGUF and device buffers. In this PR:  Refactor lora API Allow hot-swapping lora Added struct llama_lo...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1262861610654109797)** (10 messages🔥): 

> - `Codestral Mamba release`
> - `Mathstral release`
> - `Llama.cpp support issues`
> - `Google FlAMe 24B model`
> - `Llama 3 context detail` 


- **Mistral releases Code and Math models**: Two decent models, [Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) and [Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1), released under **Apache 2.0** with **32k context**.
   - The coding model, Mamba-Codestral-7B-v0.1, is not supported by **Llama.cpp** yet, but an issue has been opened to track support in [Llama.cpp](https://github.com/ggerganov/llama.cpp/issues/8519).
- **Google FlAMe 24B model outperforms other big models**: A new model from Google, referred to as **FlAMe 24B**, reportedly has better benchmarks compared to existing large models as discussed in this [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1e5118i/new_model_from_google_flame_24b/).
   - Members express skepticism about potential overfitting but acknowledge the model's promising performance.
- **Llama 3 release insights from Meta**: Joe Spisak, Product Director of GenAI at Meta, explained that **Llama 3** was initially going to be a 'prerelease' or 'preview', but **Mark Zuckerberg** pushed for its release, resulting in the 8k context for its initial rollout.
   - This indicates more features and improvements are expected, as seen in [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1e55u8h/this_went_under_the_radar_joe_spisak_product/).
- **Kaggle Notebook session handling issue**: A member experienced issues with Kaggle Notebook sessions stopping when their laptop goes to sleep, requiring them to re-run long training processes.
   - They asked if there's a way to save the fine-tuned model to avoid re-training after session stops.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/mathstral-7B-v0.1">mistralai/mathstral-7B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8519>">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e5118i/new_model_from_google_flame_24b/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e55u8h/this_went_under_the_radar_joe_spisak_product/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1262871485157081251)** (35 messages🔥): 

> - `Model Pre-training Issues`
> - `CUDA Compatibility for Unsloth`
> - `Fine-Tuning Challenges on Kaggle` 


- **Repetitive words during pre-training**: A member faced issues with repetitive words when continuing pre-training their model. The solution involved appending the **EOS token** at the end of every document in the dataset, which resolved the issue.
   - *Theyruinedelise* and *Will007* provided insights, emphasizing the necessity of appending the EOS token even during pre-training.
- **CUDA compatibility and Unsloth installation issues**: A member inquired about using Unsloth with CUDA 12.5 on Windows, but it was clarified that Unsloth supports up to **CUDA 12.1** and requires **WSL for Windows**.
   - *Edd0302* highlighted that the latest Torch supports only up to **CUDA 12.4** and suggested using WSL for better compatibility.
- **Fine-tuning Phi 3 mini on Kaggle T4 GPUs**: A member struggled to fine-tune **Phi 3 mini with 4k context** on a free Kaggle environment using a T4 GPU due to memory limitations.
   - Suggestions like reducing **batch size, epochs**, and killing the kernel to re-execute were provided, but the memory issue persisted.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1263045317558009887)** (1 messages): 

> - `Ghost 8B Beta`
> - `Proprietary models: xAI Grok 1, OpenAI GPT 3.5, Mistral Mixtral 8x7B`
> - `Model evaluation: zero-shot method`
> - `Claude 2 and Claude 3`
> - `Playground with Ghost 8B Beta` 


- **Ghost 8B Beta emerges as a leader**: [Ghost 8B Beta](https://ghost-x.org/docs/models/ghost-8b-beta) is outperforming proprietary models like **xAI Grok 1**, **OpenAI GPT 3.5**, and **Mistral Mixtral 8x7B**, solidifying its position as a top-tier language model.
   - Its unique employment of the **zero-shot method** for evaluation, along with **Claude 2** and **Claude 3**, highlights its groundbreaking capabilities.
- **Play with Ghost 8B Beta**: Members are encouraged to test Ghost 8B Beta on [Hugging Face spaces](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k) for both **8k** and **128k** token contexts.
   - Official documentation and detailed evaluations can be found on the [official website](https://ghost-x.org/docs/models/ghost-8b-beta).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta (β, 8k) - a Hugging Face Space by lamhieu</a>: no description found</li><li><a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-128k">Ghost 8B Beta (β, 128k) - a Hugging Face Space by lamhieu</a>: no description found</li><li><a href="https://ghost-x.org/docs/models/ghost-8b-beta/">Ghost 8B Beta</a>: A large language model was developed with goals including excellent multilingual support, superior knowledge capabilities and cost efficiency.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1262860756014202940)** (50 messages🔥): 

> - `Memory Usage Optimization in Neural Networks`
> - `Discussion on AdamW-Mini Optimizer`
> - `Training Efficiency`
> - `Cost of Optimizer State`
> - `Strategies for Handling Multiple Tables in Excel` 


- **Is 50% less memory usage than AdamW relevant?**: Members discussed whether **50% less memory usage** than AdamW is significant in neural networks and **its impact on large-scale training**.
   - _*One member called it 'one of the biggest findings in machine learning in the last 4-5 years'*_ and debated the actual VRAM savings and optimizer costs involved.
- **Heavy Impact of Optimizer Costs on Training**: **Optimizer state** costs dominate training, with claims that **AdamW is 3x more expensive than gradients**, affecting VRAM budgeting.
   - This leads to potential **doubling of batch sizes**, a major shift in training efficiency, according to the discussion.
- **AdamW-Mini Optimizer Efficiency Confirmed**: **AdamW-Mini** might lead to roughly **50% savings** in memory usage, with members debating differences with existing optimizers like AdamW.
   - Concerns were raised about cost distribution and the impact on training large datasets like llama-3-70b with RoPEd scaling.


  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1262852422745980992)** (83 messages🔥🔥): 

> - `Llama 3 8B on GPU`
> - `Mistral mamba code model`
> - `Troubleshooting Huge Text Files`
> - `Model Loading Issues`
> - `Codestral Mamba in LM Studio` 


- **Llama 3 8B runs 100% on GPU**: **Llama 3 8B** model was mentioned to run fully on the GPU by a member, countering performance issues with excitement emojis `😍 😭` and `😨`.
- **Mistral mamba integration in llama.cpp**: There were inquiries about **Mistral mamba code model** support, with discussions pointing to future contributions expected to llama.cpp.
- **Managing Large Text Files in AI**: Members discussed handling huge text files with models, suggesting **grep** and **awk** as tools to preprocess data due to current context size limitations.
- **Model loading issues with Nvidia and AMD GPUs**: Various members reported issues loading models due to **VRAM and RAM limitations**, especially with older GPUs like **Tesla K80**.
- **Future of Codestral Mamba Support**: Members were curious about the inclusion of **Codestral Mamba** in LM Studio, pending support additions to llama.cpp.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@kappei/topic-modeling-made-easy-with-large-language-models-3af3d2375500">Topic Modeling Made Easy with Large Language Models</a>: Topic modeling has long been a complex and time-consuming task, requiring significant expertise. However, the rise of large language models…</li><li><a href="https://huggingface.co/bartowski/NuminaMath-7B-TIR-GGUF">bartowski/NuminaMath-7B-TIR-GGUF · Hugging Face</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=40977103">Codestral Mamba | Hacker News</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6849">Support for Phi-3 models · Issue #6849 · ggerganov/llama.cpp</a>: Microsoft recently released Phi-3 models in 3 variants (mini, small &amp; medium). Can we add support for this new family of models.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1262848109071958078)** (107 messages🔥🔥): 

> - `META 3 7B Q8 instruct`
> - `LLava 3 testing`
> - `LLM suggestions for micro decisions`
> - `DeepSeek-Coder V2-Lite issues`
> - `Fine-tuning models locally` 


- **LLava 3 impresses in testing**: A user shared that they tested [LLava 3](https://link.to.llava3) and it worked well for them.
   - The model performed as expected, demonstrating reliable outcomes in the user's tests.
- **Struggles with DeepSeek-Coder V2-Lite**: Members reported issues with [DeepSeek-Coder V2-Lite](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite), where it doesn't follow prompts and provides erratic answers.
   - Disabling flash attention did not resolve the problem, suggesting LM Studio updates might have broken its support.
- **Recommendations for LLMs for micro decisions**: A user inquired about LLMs suitable for tasks like **NER**, content filtering, and binary decision tasks; suggestions included smaller models.
   - Some users recommended considering compute-efficient models for better performance in discrete tasks.
- **Fine-tuning LLMs locally challenges**: One user expressed issues with fine-tuning **Codestral** on their system, mentioning repeated 'G' responses.
   - Fine-tuning LLMs locally with substantial hardware is feasible; consulting documentation and community sources is recommended.
- **Optimizing hardware configurations for fine-tuning**: Users discussed configurations for efficient fine-tuning, debating hardware like **RTX 4090** and laptops with high RAM and VRAM.
   - Ensuring compatibility and proper setup, including disabling features like flash attention, is crucial for successful fine-tuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF">bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF/tree/main">bartowski/Codestral-22B-v0.1-GGUF at main</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1262858759093616650)** (14 messages🔥): 

> - `Gemma 2 support`
> - `Phi 3 small support`
> - `Llama.cpp support`
> - `Error loading model`
> - `Smol-lm pre-tokenizer issue` 


- **Gemma 2 supported but Phi 3 small isn't**: Members discussed that **Gemma 2** will be supported but not **Phi 3 small** due to lack of support in **llama.cpp**.
- **Error loading model message concerns**: A member had an 'Error loading model' and was guided to share more details and system information in the [support channel](<#1111440136287297637>).
- **Smol-lm model unsupported in llama.cpp**: An error with the pre-tokenizer type 'smol-lm' was shared as being unsupported by the **llama.cpp** build currently used by LM Studio.


  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/)** (1 messages): 

pashtett: Any examples of optimal prompt and settings to gemma 2 for some RP chat based on story?
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1262893193771487345)** (4 messages): 

> - `GPU craftsmanship`
> - `aesthetics importance`
> - `GPU power plug` 


- **GPU support truss spray-painted rose gold**: A user shared their experience of spraying the GPU support truss **rose gold** because they believe **aesthetics are important**.
   - *Craftsmanship* in GPU setups can sometimes come down to these detailed aesthetic choices.
- **Forgot to plug GPU power**: A user admitted they **forgot to plug in the GPU power** during their setup process.
   - This mishap is a common occurrence in hardware setups and serves as a reminder to double-check connections.


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1262858589660516404)** (1 messages): 

> - `Mathstral Announcement`
> - `STEM Specialization`
> - `GGUF Quantization` 


- **Mistral's Mathstral Shoots for the Stars**: Announced as **specializing in STEM and advanced reasoning**, the Mathstral model far outperforms the base Mistral 7B in several major STEM categories.
   - Check out the [Mathstral model](https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF) for more detailed information and join the discussion on [Discord](https://discord.gg/aPQfnNkxGC).
- **Mathstral Model Summary and Innovations**: The [Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1) is a fine-tuned model solving advanced math problems with complex multi-step logical reasoning.
   - **Quantization** by [bartowski](https://huggingface.co/bartowski) using `llama.cpp` release [b3389](https://github.com/ggerganov/llama.cpp/releases/tag/b3389) optimizes the model's performance.



**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF">lmstudio-community/mathstral-7B-v0.1-GGUF · Hugging Face</a>: no description found

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1262848440103211039)** (50 messages🔥): 

> - `Mojo Community Meeting`
> - `CFFI and C++ Interoperability`
> - `External Linking with DLOpen`
> - `Support Ticket System` 


- **Mojo Community Meeting #4 offers valuable insights**: The [4th Mojo Community Meeting](https://www.youtube.com/watch?v=_QVs626Vn2k) recording is now available on YouTube, featuring discussions on Flat Buffers and Forge Tools.
   - CFFI was mentioned towards the end, leading to questions about static linking and more refined external call syntax for compatibility with libraries like OpenSSL.
- **C++ interop is a complex beast compared to C interop**: Members discussed that while C interop is already supported through DLHandles, C++ interop is vastly more complex due to templates and ABI considerations.
   - A suggestion was made to dynamically link without DLOpen or to incorporate an `@link` decorator similar to Rust’s `#[link]` macro for better support.
- **External calls modeled in MLIR**: Discussion around C static linking revealed that external calls are modeled in MLIR as `pop.external_call`, with detailed examples available on [GitHub](https://github.com/modularml/mojo/blob/main/stdlib/src/sys/ffi.mojo#L44).
   - Dynamic linking and lifting were emphasized as crucial for security and usability, with references to [dlopen documentation](https://man7.org/linux/man-pages/man3/dlopen.3.html).
- **Support for Mojo and other Modular products**: A user inquired about how to open a support ticket for issues related to Mojo, Modular CLI, or other products.
   - It was clarified that there is no formal support ticket system, but users can seek help through Discord, GitHub issues, or direct Modular team support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/sys/ffi.mojo#L44">mojo/stdlib/src/sys/ffi.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://man7.org/linux/man-pages/man3/dlopen.3.html">dlopen(3) - Linux manual page</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1263144421042425988)** (5 messages): 

> - `Object detection in videos`
> - `AWS EC2 instances`
> - `Mojo data types` 


- **Challenges with Object Detection in Videos**: A member shared challenges in object detection using pre-trained models for videos, including handling large numbers of frames and unsmooth bounding boxes.
   - Another member suggested running detection at a low framerate like 5 fps and applying post-processing to smooth out bounding box locations.
- **Using AWS EC2 for Object Detection**: A member mentioned utilizing AWS EC2 instances for their object detection tasks.
   - *No further discussion elaborated on this point.*
- **Query on Mojo Data Types**: A user inquired about the primitive and composite data types in Mojo.
   - *No response to the query was provided within the message history.*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1262861011896238250)** (51 messages🔥): 

> - `Mojo 🔥 Community Meeting`
> - `Mojo language keywords`
> - `Installing old Mojo versions`
> - `SIMD primitives references`
> - `Looping through Tuple in Mojo` 


- **Mojo 🔥 Community Meeting Discusses Error Handling**: The community discussed error handling in the latest [Mojo 🔥 Community Meeting](https://youtu.be/_QVs626Vn2k?t=16740), exploring serialization and extending the Mojo standard library.
   - Members noted specific timestamps where the PR and error handling were discussed in depth.
- **Mojo Language Removes 'let' Keyword**: In Mojo, the 'let' keyword was removed, confirmed by members discussing their memory of its existence and reasons behind its removal.
   - 'Let' was initially present but was removed due to the ownership and semantics model, with 'var' being used for runtime variables.
- **Install Old Versions of Mojo Easily**: Users can install old versions of Mojo by using commands: `modular uninstall mojo` and `modular install mojo --install-version 24.3.0`.
   - For a list of all available versions, users can check the [Mojo GitHub branches](https://github.com/modularml/mojo/branches) and respective activities.
- **References for SIMD Intrinsics in x86 and ARM**: For x86 SIMD intrinsics, Intel's [intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) is helpful.
   - For ARM, the [ARM intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics) are provided, although SME is noted to be missing.
- **Looping Through Tuple in Mojo**: A user queried how to loop through a tuple in Mojo but no detailed solution was provided in the discussion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/memory/unsafe/DTypePointer#address_of">DTypePointer | Modular Docs</a>: Defines a DTypePointer struct that contains an address of the given dtype.</li><li><a href="https://youtu.be/_QVs626Vn2k?t=1390)">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://youtu.be/_QVs626Vn2k?t=16740)">Mojo 🔥 Community Meeting #4</a>: Recording of the Mojo Community Meeting #4🫓 Flat Buffers: memory efficient serialization⚒️ Forge Tools: extending the Mojo 🔥 standard library🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/activity?ref=nightly">Activity · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/branches">Branches · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://developer.arm.com/architectures/instruction-sets/intrinsics">Intrinsics – Arm Developer</a>: no description found</li><li><a href="https://github.com/rust-lang/rust-wiki-backup/blob/master/Sigil-reference.md">rust-wiki-backup/Sigil-reference.md at master · rust-lang/rust-wiki-backup</a>: A backup of the Rust wiki. Contribute to rust-lang/rust-wiki-backup development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1262885186324922470)** (3 messages): 

> - `Difference between parallelize and sync_parallelize`
> - `Memory management improvement` 


- **Difference between `parallelize` and `sync_parallelize` explained**: A member inquired about the difference between `parallelize` and `sync_parallelize`.
- **Memory management needs better understanding**: A member mentioned that they have not improved their draft version yet due to needing a better understanding of memory management.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1262853029267374090)** (12 messages🔥): 

> - `Modular installation issues`
> - `MNIST accuracy discrepancy`
> - `User experience improvements`
> - `Verbose reporting for MAX` 


- **Modular installation issues clarified**: A user shared that they struggled with the installation process of **Modular** until realizing they needed to export the correct bash path for **nightly/max**.
   - Another user noted that installation issues will soon be resolved and updates will become seamless.
- **MNIST accuracy discrepancy noted**: The user found a difference in accuracy between `python3` and `mojo` runs on the **MNIST dataset** using `--use-relu6`.
   - They later clarified that using the correct hyphen resolved the issue; however, **relu6** resulted in a lower accuracy by about 1%.
- **Improvements to notebook UX needed**: One user agreed that the notebook experience needs improvements and gets confusing.
   - They plan to make the **UX better** soon.
- **Verbose reporting requested for MAX**: A user requested more verbose reports from **MAX**, including metrics like duration and GFLOPS.
   - They emphasized that these metrics would assist in making hardware and financial decisions.


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1262866614265843923)** (4 messages): 

> - `Inline functions in Mojo`
> - `SIMD optimization suggestions`
> - `New Mojo nightly release`
> - `Mojo nightly changelog updates` 


- **Propose shorthand for inline functions**: A member suggested that writing a function on the same line as its signature could act as a macro for `@always_inline("nodebug")`, potentially shortening code without impacting readability.
   - The argument was made that this shorthand implies inlined functions should be small, which is generally true.
- **SIMD Optimization with SVE**: A member mentioned that SIMD sizes don't need to be a multiple of 2 and that **SVE** fixes this issue.
   - They suggested implementing drain loops or using masks on architectures without variable-width SIMD.
- **New Mojo nightly compiler released**: A new nightly Mojo compiler release, `2024.7.1714`, is available for update with the command `modular update nightly/mojo`.
   - Changelog includes removal of `SIMD.{min,max}` methods in favor of builtins, adding a `Dict.__init__` overload with `power_of_two_initial_capacity`, and removing `SIMD.{add,mul,sub}_with_overflow` methods.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263148280167272480)** (71 messages🔥🔥): 

> - `Mojo utilizing cores`
> - `NumPy performance`
> - `Benchmarking`
> - `BLAS backends`
> - `Intel MKL vs. other BLAS` 


- **Mojo uses only performance cores**: Users observed that on Intel chips with both performance and efficiency cores, the `parallelize` function in **Mojo** exclusively utilizes performance cores for better results.
   - This design decision stems from challenges in efficiently distributing work between performance and efficiency cores with a simple API like `parallelize`. [Details here](https://link.to.issue).
- **Mojo beats NumPy even using fewer cores**: Benchmark results showed **Mojo** outperforming **NumPy** even though it only uses performance cores, while NumPy utilizes all cores.
   - The Mojo runtime isn't yet smart enough to distribute work efficiently across different types of cores as of now, but future updates are expected.
- **Different BLAS backends affect NumPy performance**: Discussed how **NumPy**'s performance can vary significantly based on the BLAS library used; **OpenBLAS** was mentioned with some favoring the use of **Intel MKL** for better speed.
   - Members pointed out that most 'faster than NumPy' claims are usually comparing against NumPy without a good BLAS backend. [Intel's distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) is recommended.
- **Manual timing and benchmarking reveal insights**: One user switched to manual timing for more accurate benchmarking, revealing interesting performance divots at specific data points like 1024.
   - They noted that some performance drop-offs occur when block sizes slightly exceed 1024, making the second block less computationally efficient.
- **Intel MKL's superior performance**: Intel's MKL was suggested as a much faster alternative to OpenBLAS, even on non-Intel CPUs.
   - This recommendation stems from MKL's widespread adoption in supercomputing for its superior performance over other BLAS libraries.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1263011455469355028)** (1 messages): 

> - `DataComp for Language Models (DCLM)`
> - `DCLM-Baseline-7B`
> - `MMLU Benchmark`
> - `OpenLM framework`
> - `Dataset design importance` 


- **DataComp Introduces New Testbed for Language Models**: The [DataComp for Language Models (DCLM)](https://arxiv.org/abs/2406.11794) is a testbed for controlled dataset experiments aimed at improving language model performance.
- **DCLM-Baseline-7B achieves impressive MMLU score**: The [DCLM-Baseline-7B](https://huggingface.co/apple/DCLM-Baseline-7B) achieves 64% 5-shot accuracy on MMLU with 2.6T training tokens, marking a significant **6.6 percentage point improvement** over MAP-Neo while using 40% less compute.
- **OpenLM framework supports DCLM's pretraining**: DCLM provides standardized corpus and effective pretraining recipes based on the [OpenLM framework](https://arxiv.org/abs/2406.11794).
- **DCLM emphasizes dataset design**: DCLM highlights the importance of dataset design for training language models, with model-based filtering proving key for high-quality training sets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.11794">DataComp-LM: In search of the next generation of training sets for language models</a>: We introduce DataComp for Language Models (DCLM), a testbed for controlled dataset experiments with the goal of improving language models. As part of DCLM, we provide a standardized corpus of 240T tok...</li><li><a href="https://github.com/mlfoundations/dclm">GitHub - mlfoundations/dclm: DataComp for Language Models</a>: DataComp for Language Models. Contribute to mlfoundations/dclm development by creating an account on GitHub.</li><li><a href="https://huggingface.co/apple/DCLM-Baseline-7B">apple/DCLM-7B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1262870425894129761)** (2 messages): 

> - `Need for Math Annotators in AI`
> - `Replete-AI Multilingual Translation Dataset` 


- **Rising Need for Math Data and Annotators**: A member questioned whether more **math data/math annotators** will be needed to train AI to be smarter, and if there is already a shortage in this area.
   - *Is it just me or are we already seeing a shortage in this area?* was a point of discussion highlighted for community experiences and insights.
- **Replete-AI Launches Huge Open Source Translation Dataset**: A [new dataset](https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct) was announced, including **2.8 million rows** of translations from English to **14 languages**.



**Link mentioned**: <a href="https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct">Replete-AI/Multi-lingual_Translation_Instruct · Datasets at Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1263122622233841706)** (4 messages): 

> - `Oxen.AI Paper Club`
> - `Representation Finetuning`
> - `Comparison to repeng/vector steering` 


- **Author Zhengxuan Wu to Join Oxen.AI Paper Club**: Zhengxuan Wu, first author of an [Arxiv paper](https://arxiv.org/pdf/2404.03592), will join Greg Schoeninger this Friday to discuss how editing representations can be better than Parameter-efficient finetuning (PEFT) methods in the Oxen.AI Paper Club.
   - Join the session [here](https://lu.ma/oxen) to explore building world-class AI datasets and learn from Zhengxuan Wu himself.
- **ReFT: Representation Finetuning**: The upcoming paper club session will explore **ReFT** (Representation Finetuning), promising a detailed discussion with the author Zhengxuan Wu. A member expressed curiosity about the specifics such as the definition of 'representation' and 'task-specific intervention'.
   - They also inquired if the concept is analogous to improving an API by directly altering its codebase, describing the paper as 'very voodoo'.
- **Comparison to Repeng/Vector Steering**: A member asked if ReFT differs from repeng/vector steering, to which another replied that they are extremely similar.
   - It was noted that the paper actively cites repeng, suggesting significant overlap in methodologies.



**Link mentioned**: <a href="https://lu.ma/oxen">Oxen.ai · Events Calendar</a>: View and subscribe to events from Oxen.ai on Luma. Build World-Class AI Datasets, Together.  Track, iterate, collaborate on, &amp; discover data in any format.

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1262886729480933478)** (9 messages🔥): 

> - `Lunar Caves`
> - `Belief State Geometry in Transformers`
> - `Tool Use Models`
> - `LLM-driven Digital Agents` 


- **Lunar Caves: A Hidden Refuge**: [Scientists have identified lunar caves as potential hidden refuges](https://www.perplexity.ai/page/lunar-caves-a-new-frontier-in-u3Rkbvk4QROuAEtNMlwoug), calling them a new frontier for exploration.
   - *Retro* interest from the scientific community brings this months-old paper back into the spotlight on Twitter.
- **Belief State Geometry in Transformers Explained**: A new [paper on arXiv](https://arxiv.org/abs/2405.15943) presents the concept of 'belief state geometry', showing how transformers encode belief updates within their residual streams.
   - Community reactions range from finding it 'profound' to considering it as potentially 'overly complex AI psychobabble'
- **Llama 3 Groq Tool Use Models Claim Top Spot**: [Rick Lamers announced](https://x.com/RickLamers/status/1813341037198204962) the Llama 3 Groq Tool Use 8B and 70B models, highlighting their achievement of #1 position on BFCL.
   - These models are trained on synthetic data only and are now available on the Groq API and Hugging Face.
- **ManifoldRG's Position on LLM Digital Agents**: [ManifoldRG shared a position paper](https://x.com/ManifoldRG/status/1811120196570206459) arguing that advancements in LLM-driven agents require moving beyond language-based processing to enhance reasoning.
   - The paper explores current limitations and future directions for intelligent digital agents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15943">Transformers represent belief state geometry in their residual stream</a>: What computational structure are we building into large language models when we train them on next-token prediction? Here, we present evidence that this structure is given by the meta-dynamics of beli...</li><li><a href="https://x.com/RickLamers/status/1813341037198204962">Tweet from Rick Lamers (@RickLamers)</a>: I’ve been leading a secret project for months … and the word is finally out!  🛠️ I&#39;m proud to announce the Llama 3 Groq Tool Use 8B and 70B models 🔥  An open source Tool Use full finetune of Lla...</li><li><a href="https://x.com/ManifoldRG/status/1811120196570206459">Tweet from Manifold Research (@ManifoldRG)</a>: 🚨We’re excited to share “Intelligent Digital Agents in the Era of Large Language Models”, a position paper that explores advancements in LLM driven agents, identifies limitations, and suggests that t...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1262853343278137480)** (154 messages🔥🔥): 

> - `Hermes 2.5 vs Hermes 2 performance`
> - `Challenges extending Mistral`
> - `Model experimentation`
> - `Tool calling implementation`
> - `Function calling issues` 


- **Hermes 2.5 outperforms Hermes 2 in benchmarks**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** shows improved performance over **Hermes 2** in various benchmarks.
   - Hermes 2.5 scored **52.3** on the MMLU benchmark, significantly higher than Hermes 2's **34.5**.
- **Mistral struggles with extension beyond 8k parameters**: Members noted that **Mistral** cannot be effectively extended beyond 8k without additional pretraining, a [well-known issue](https://link.to.issue).
   - Suggestions were made to explore *mergekit* and *frankenMoE finetuning* techniques for future improvements.
- **Experimentation with frankensteining models**: A user shared results of merging **Hermes 2 pro** and **llama70b-instruct**, creating a new model named **Llamagnific** hosted on [Hugging Face](https://huggingface.co/nisten/llamagnific-3-87b).
   - *Llamagnific* displayed higher intelligence but also increased instances of nonsensical responses.
- **Tool calling implementation with Hermes 2 Pro**: A beta implementation of OpenAI-style tool calling for **Hermes 2 Pro** using **vLLM** has been proposed in a [GitHub PR](https://github.com/vllm-project/vllm/pull/5649).
   - Adjustments made to the system prompt led to more consistent tool call ordering, improving overall performance.
- **Tool calling and function usage issues**: Discussions about the challenges in tool calling and function usage, stressing the importance of streaming tool calls to provide real-time feedback to users.
   - Proper tool call streaming significantly improves user experience by minimizing wait times and showing intermediate results promptly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/nisten/llamagnific-3-87b-gguf">nisten/llamagnific-3-87b-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nisten/llamagnific-3-87b">nisten/llamagnific-3-87b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nisten/llamagnific-3-87b-gguf/resolve/main/llamagnific_1bit_optimized_IQ1_L.gguf">no title found</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Multi-lingual_Translation_Instruct">Replete-AI/Multi-lingual_Translation_Instruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/i/broadcasts/1lDGLldQVmvGm">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://wow.groq.com/introducing-llama-3-groq-tool-use-models/">Introducing Llama-3-Groq-Tool-Use Models - Groq is Fast AI Inference</a>: We are excited to announce the release of two new open-source models specifically designed for tool use:&nbsp;Llama-3-Groq-70B-Tool-Use</li><li><a href="https://github.com/vllm-project/vllm/pull/5649">Support Open Models that allow OpenAI API-style tool use &amp; &quot;auto&quot; tool choice by K-Mistele · Pull Request #5649 · vllm-project/vllm</a>: DRAFT: OpenAI Tool Use Checklist This (Draft) PR will add support for OpenAI-style tool calling in a way that is minimally opinionated about tool use formats &amp; prompt formatting. The following fea...</li><li><a href="https://x.com/phill__1/status/1813307823570157899">Tweet from Phil (@phill__1)</a>: Google accidentally updated their website with Gemini 2.0 and Bing indexing caught it</li><li><a href="https://www.reddit.com/r/singularity/comments/1e4wlzr/saw_something_interesting_when_i_googled_gemini/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1262863146650042428)** (19 messages🔥): 

> - `Tokenization with Tiktoken`
> - `Beam Search Implementation in Huggingface Pipelines`
> - `Invertibility of BPE in Tiktoken`
> - `Custom Sampling in Huggingface Pipelines` 


- **Challenges in Tokenization with Tiktoken**: A user reported issues with the **Tiktoken library** when decoding Arabic symbols, resulting in special symbols instead of the original text.
   - *'This ensures that we still receive readable text, even if some parts are incorrect'*, they used `errors='replace'` to manage invalid UTF-8 sequences during decoding.
- **BPE Invertibility Handling Special Tokens**: Another member noted that BPE (used by **Tiktoken**) should be able to represent any byte sequence, implying that all tokens decode to either valid UTF-8 sequences or specific byte values.
   - They reassured that **cl100k_base** ensures invertibility for token sequences.
- **Implementing Custom Beam Search in Huggingface**: A member asked about creating a custom beam search for **Huggingface** pipelines and received guidance on using `model.generate()` parameters.
   - They eventually extended **GenerationMixin** to reimplement `beam_search()` and created a custom model class, noting it felt *janky* but functional.


  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1262884606030512239)** (3 messages): 

> - `` 


- **No Substantive Discussion Detected**: No meaningful or detailed discussions were observed in the recent messages.
- **Users Greet Each Other**: Users exchanged greetings in the channel with messages like 'Hi'.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1262866408656732251)** (60 messages🔥🔥): 

> - `The Pile 2`
> - `Proof-Pile-2`
> - `YouTube video scraping controversy`
> - `Public reaction to YouTube data usage`
> - `Transparency in AI data usage` 


- **Confusion about The Pile 2**: Users were confused about the existence of **The Pile 2**, leading to clarifications that it doesn't exist yet.
   - (N/A)
- **Understanding Proof-Pile-2**: A user linked to the **Proof-Pile-2** dataset on [Hugging Face](https://huggingface.co/datasets/EleutherAI/proof-pile-2), describing it as a 55 billion token dataset of mathematical and scientific documents.
   - (N/A)
- **YouTube video scraping controversy heats up**: Concerns were raised about **YouTube videos** being scraped and used in AI datasets without permission, following a [Proof News article](https://www.proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/).
   - Artists like [Philosophy Tube](https://x.com/PhilosophyTube/status/1813227210569920685) and [Jacob Geller](https://x.com/yacobg42/status/1813226763117367688) condemned the practice, leading to discussions about the impact and ethical concerns.
- **Transparency in AI data usage questioned**: **EAI** faced backlash for being transparent about their data sourcing, with users discussing how other tech companies might be less forthcoming.
   - *Public blowback and legal risk is the major reason people are non-transparent about their data*, and there's concern that being open invites blame and criticism,** said a user**.
- **Community appreciation in turbulent times**: A user expressed deep appreciation for the community, emphasizing how it helped them catch up with research and inspired them to start their own projects.
   - (N/A)


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/EleutherAI/proof-pile-2">EleutherAI/proof-pile-2 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/proof__news/status/1813182354728317341">Tweet from Proof News (@proof__news)</a>: Our latest investigation reveals a dataset of more than 170,000 YouTube video subtitles that big tech companies used to train their AI models.   “Will this be used to exploit and harm artists? Yes, ab...</li><li><a href="https://www.youtube.com/shorts/xiJMjTnlxg4">AI is Stealing my Videos</a>: no description found</li><li><a href="https://x.com/PhilosophyTube/status/1813227210569920685">Tweet from Abigail Thorn (@PhilosophyTube)</a>: Very sad to have to say this - an AI company called EleutherAI stole tens of thousands of YouTube videos - including many of mine. I’m one of the creators Proof News spoke to. The stolen data was sold...</li><li><a href="https://x.com/yacobg42/status/1813226763117367688">Tweet from Jacob Geller (@yacobg42)</a>: Looks like dozens of my videos have been scraped and included in datasets for training AI. The companies doing this didn&#39;t ask for permission to use my work (of course).   I&#39;m not surprised, b...</li><li><a href="https://web.archive.org/web/20240717020029/https://www.washingtonpost.com/technology/2024/07/16/trump-ai-executive-order-regulations-military/">Trump allies draft AI order to launch ‘Manhattan Projects’ for defense</a>: The plan to roll back “burdensome regulations” would favor Silicon Valley investors, who are now flocking to support the former president.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1262852011901190224)** (89 messages🔥🔥): 

> - `Efficient Attention mechanisms`
> - `Transformer optimizations`
> - `Reformer: The Efficient Transformer`
> - `LSH attention in practice`
> - `PLMs for immune escape` 


- **Standard deviation calculation debate**: Discussion on calculating standard deviation in one pass using a [link to an article](https://www.strchr.com/standard_deviation_in_one_pass) versus the traditional two-pass approach.
   - One user mentioned having implemented it but faced issues with the launch config.
- **Debunking TransformerEngine claims**: Users debated the fusion implementation of **TransformerEngine**, confirming it doesn't fuse normalization and linear layers as previously assumed. See link for [TransformerEngine code](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py).
   - RMSNorm fusion discussed as a superior method that allows scaling, normalization, linear, and activation in one kernel.
- **Reformer: Efficient Transformer Clarifications**: Clarifications on **Reformer: The Efficient Transformer** were provided, highlighting the importance of normalization for keys in hashing functions and differences in attention matrices.
   - Discussion on why **LSH attention** has not seen widespread adoption despite theoretical advantages.
- **Challenges of Efficient Attention mechanisms**: [Reproducibility issues](https://openreview.net/forum?id=3s8Y7dHYkN-&noteId=aDaPfMT84Ef) with Efficient Transformers like **Reformer** were discussed, including difficulty in matching original performance claims.
   - Mention of **Linear Transformers** as potentially more successful alternatives in addressing quadratic complexity.
- **PLMs for distinguishing viral mimicry**: A user shared their accepted poster presentation at ICML 2024 on using **Protein Language Models (PLMs)** to identify viral proteins mimicking human proteins, with [99.7% RO CAUC](https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb).
   - Their poster analyzes errors in PLMs and immune system for better vaccine/therapeutic developments, with [code available](https://github.com/ddofer/ProteinHumVir).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2006.16236">Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention</a>: Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input&#39;s length, they are prohibitively slow for very long sequences. To addr...</li><li><a href="https://arxiv.org/abs/2407.11542">Understanding Counting in Small Transformers: The Interplay between Attention and Feed-Forward Layers</a>: We provide a comprehensive analysis of simple transformer models trained on the histogram task, where the goal is to count the occurrences of each item in the input sequence from a fixed alphabet. Des...</li><li><a href="https://arxiv.org/abs/2407.11239">From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients</a>: Modern Large Language Models (LLMs) are composed of matrices with billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Being ...</li><li><a href="https://goombalab.github.io/blog/2024/hydra-part1-matrix-mixer/"> Hydra Part I - Matrix Mixer Framework | Goomba Lab </a>: no description found</li><li><a href="https://www.youtube.com/watch?v=s8RqGlU5HEs">2 Years of My Research Explained in 13 Minutes</a>: This is my research into representation learning and model learning in the reinforcement learning setting. Two years in the making, and I finally get to talk...</li><li><a href="https://openreview.net/forum?id=3s8Y7dHYkN-&noteId=aDaPfMT84Ef">Reproducibility Challenge: Reformer</a>: We attempt to reproduce the central claims of ICLR 2020 Paper &quot;Reformer: The Efficient Transformer&quot;; that the techniques introduced enable performance on par with a traditional Transformer m...</li><li><a href="https://openreview.net/forum?id=rkgNKkHtvB&noteId=H1g3oF4sjS">Reformer: The Efficient Transformer</a>: Efficient Transformer with locality-sensitive hashing and reversible layers</li><li><a href="https://www.strchr.com/standard_deviation_in_one_pass">Calculating standard deviation in one pass - strchr.com</a>: no description found</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py#L141>,">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilizatio...</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilizatio...</li><li><a href="https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are">LLM Basics: Embedding Spaces - Transformer Token Vectors Are Not Points in Space — LessWrong</a>: This post is written as an explanation of a misconception I had with transformer embedding when I was getting started. Thanks to Stephen Fowler for t…</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: Viruses elude the immune system through molecular mimicry, adopting their hosts biophysical characteristics. We adapt protein language models (PLMs) to differenti-ate between human and viral...</li><li><a href="https://github.com/ddofer/ProteinHumVir">GitHub - ddofer/ProteinHumVir: Code &amp; data for &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot;</a>: Code &amp; data for &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot; - ddofer/ProteinHumVir</li><li><a href="https://doi.org/10.1101/2024.03.14.585057">Protein Language Models Expose Viral Mimicry and Immune Escape</a>: Motivation Viruses elude the immune system through molecular mimicry, adopting biophysical characteristics of their host. We adapt protein language models (PLMs) to differentiate between human and vir...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263122561861029979)** (4 messages): 

> - `Arrakis library`
> - `Mechanistic interpretability tools`
> - `Feedback request` 


- **Introducing Arrakis Library for Fast Mechanistic Interpretability**: [Arrakis](https://github.com/yash-srivastava19/arrakis) is a new library designed to conduct, track, and visualize mechanistic interpretability experiments, aimed at researchers who want to iterate quickly.
   - The library includes features such as **tuned-lens tools** and model surgery, but is still in its early development stage.
- **Community Feedback Requested for Arrakis**: A request for feedback on the utility of Arrakis for the community was made, highlighting its ease of use and rapid iteration capabilities.
   - A member inquired why the creator started from scratch instead of using existing tools, such as TransformerLens or nnsight.
- **Clemd6d Questions Alternative Choices for Arrakis**: A user, clemd6d, asked yash_sri19 if there are specific abstractions in Arrakis that aren't possible with TransformerLens, nnsight, or PyVene.
   - *Yash_sri19* responded that they wanted abstractions over common functions in mechanistic interpretability and a fast iteration design, though the library is still in its early stages of development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/yash-srivastava19/arrakis">GitHub - yash-srivastava19/arrakis: Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments.</a>: Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments. - yash-srivastava19/arrakis
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263167243664101479)** (3 messages): 

> - `HF leaderboard musr score`
> - `Leaderboard maintainers query` 


- **Query on HF leaderboard musr raw score**: A member inquired if the musr raw score in the HF new leaderboard is the macro average of these 3 tasks: **musr_murder_mysteries**, **musr_object_placements**, **musr_team_allocation**.
   - Another member suggested that they should ask the [leaderboard maintainers](https://huggingface.co/leaderboard) for clarification.
- **Acknowledging the need to ask leaderboard maintainers**: After receiving advice, the member thanked the respondent for suggesting to reach out directly to the leaderboard maintainers.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1262848301192187924)** (143 messages🔥🔥): 

> - `Model Size & Hardware`
> - `Training Techniques`
> - `Prompting Nuances`
> - `Outpainting Techniques`
> - `Troubleshooting` 


- **Guidance for Model Size Based on Hardware**: Members discussed the optimal model size for GPU VRAM and regular RAM, citing that VRAM impacts performance significantly.
   - It's noted that **larger models require more VRAM** and **longer generation times** do not imply memory issues unless an OOM exception occurs.
- **Training Specific Models and Characters**: Users inquired about training models for specific styles like crosshatch illustrations and combining character-specific models for generating joint content.
   - Links to relevant resources and discussions about **regional prompting** and **multi-concept training** were shared, including [HuggingFace's T5](https://huggingface.co/jtlicardo/flan-t5-small-coref).
- **Understanding Prompting Nuances**: Nuances in text prompts, such as 'harvesting potato' versus 'potato harvesting,' generated discussions on coreference resolution capabilities in models.
   - T5's fine-tuned models, especially for coreference tasks, were recommended to handle complex prompt nuances effectively.
- **Effective Outpainting Techniques**: For expanding generated images, outpainting methods and specific tools like Photoshop’s gen fill were recommended.
   - Members also discussed using KSampler in ComfyUI for managing seeds during expansions to avoid overlapping images.
- **Troubleshooting Stable Diffusion Issues**: Members experiencing issues with models in Automatic1111 discussed troubleshooting steps, including command line arguments for **better performance** with specific hardware.
   - Suggestions such as **using 'xformers' and 'medvram-sdxl'** options were provided to improve functionality on limited hardware configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jtlicardo/flan-t5-small-coref">jtlicardo/flan-t5-small-coref · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/images/19928073">Video posted by khitomer</a>: no description found</li><li><a href="https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/wiki/Regional-Prompt-Control>">Home</a>: Tiled Diffusion and VAE optimize, licensed under CC BY-NC-SA 4.0 - pkuliyi2015/multidiffusion-upscaler-for-automatic1111
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1263026310297550848)** (26 messages🔥): 

> - `CUDA kernel call errors`
> - `Template types in CUDA`
> - `cudaMallocManaged overhead`
> - `Unified memory usage in CUDA`
> - `Deep learning specialization opinions` 


- **CUDA kernel call errors solved with template types**: A user learning CUDA encountered an error with a kernel call, resolved by correctly specifying the template type.
   - *Adding the template argument (`<int>`) as shown in CUDA examples solved the issue, despite it feeling like overkill for introductory purposes.*
- **Discussing cudaMallocManaged overhead**: Members debated whether **cudaMallocManaged** introduced overhead due to shared memory space between host and device, concluding that using explicit memory copies might be more efficient.
   - *Explicitly handling memory transfers using cudaMemcpy could avoid potential overhead and increase performance.*
- **Exploring Unified Memory and GPU architecture**: Unified memory (cudaMallocManaged) was discussed, with concerns over its overhead and the architectural specifics of current NVIDIA GPUs.
   - One member pointed out the importance of understanding GPU architecture to optimize memory usage and questioned if certain processes like unified memory might slow down operations.
- **Fine-tuning LLMs: Handling pad tokens and prompts**: A question about excluding pad tokens from loss calculation during LLM fine-tuning was discussed, with HF's use of `ignore_index=-100` in the cross-entropy loss to exclude them.
   - *Preparing datasets correctly with proper prompt formats is crucial, and HF's tokenizer supports applying chat templates for easier data management.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/transformers/blob/72fb02c47dbbe1999ae105319f24631cad6e2e00/src/transformers/models/llama/modeling_llama.py#L1092-L1102).">transformers/src/transformers/models/llama/modeling_llama.py at 72fb02c47dbbe1999ae105319f24631cad6e2e00 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/pytorch/torchtune/blob/8e036611aac377fd9b383a66c161ce085c93f8ce/recipes/full_finetune_single_device.py#L448-L454).">torchtune/recipes/full_finetune_single_device.py at 8e036611aac377fd9b383a66c161ce085c93f8ce · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262863210302668860)** (33 messages🔥): 

> - `PyTorch Profiler Performance`
> - `Thunder vs Torch Compile`
> - `Nvfuser vs Triton`
> - `Kernel Compilation`
> - `Runtime Optimization` 


- **PyTorch Profiler Export Takes Too Long**: A user expressed concern that exporting a trace using the **PyTorch profiler** takes around **30 minutes**, likely due to capturing extensive information.
   - Another member suggested disabling the `profile_memory` and `with_stack` options to speed up export times without losing runtime information.
- **Thunder and Torch Compile Integration**: A discussion highlighted **Thunder**, a performance optimization layer that can stitch different **PyTorch** compatible kernels together, usually using 'nvfuser' as a backend.
   - The feature was compared to **torch.compile**, with Thunder aiming for transparent and debuggable backend integration, facilitating manual performance tuning.
- **Nvfuser vs Triton Performance Discussion**: It was noted that **nvfuser** and **Triton** have distinct performance characteristics, winning and losing on different benchmarks.
   - The conversation emphasized using both to achieve optimal performance, leveraging **Thunder** to mix and match these backends effectively.
- **Concerns about Custom Kernel Compilation Time**: The long time for profiler export could be due to custom kernels compiled via **nvfuser**.
   - Despite the extended time, users expressed appreciation for **Thunder** and its mix-and-match kernel capabilities.
- **Optimizing Kernel Compilation**: The conversation briefly touched on the complexities of optimizing kernel compilation using **nvfuser** as opposed to **Triton**.
   - It was clarified that Thunder supports data dependency and auto-scheduling within a generated Python function to optimize runtime.


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1263051227994783755)** (7 messages): 

> - `Mixing CUDA kernels with PyTorch`
> - `Writing custom CUDA kernels`
> - `Automatically generated Python bindings`
> - `Compiling custom kernels` 


- **Seeking code examples for mixing CUDA kernels with PyTorch**: **artificial_anteligence** is looking for material on how to mix CUDA kernels with PyTorch, including a full implementation of a simplified model and training.
- **Need references for replacing PyTorch functions with custom CUDA kernels**: **artificial_anteligence** expressed a need to replace parts of PyTorch functions with custom CUDA kernels or write everything in CUDA, asking for reference code.
   - **as_ai** mentioned that Cuda mode lecture 1 demonstrates how to get started with the load_inline module in PyTorch, including compiling specified kernels.
- **Auto-generate Python bindings for custom CUDA kernels**: **as_ai** noted that Python bindings can be automatically generated if you provide the CUDA source and a C++ torch tensor function with launch parameters for your kernel.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1262860938869080098)** (13 messages🔥): 

> - `unwrap_tensor_subclass issues`
> - `AQT Tensor instance`
> - `FakeTensor attribute error`
> - `PyTorch nightly build`
> - `GitHub issue` 


- **unwrap_tensor_subclass converts tensor subclass to plain tensors**: **unwrap_tensor_subclass** uses parametrization to convert tensor subclasses to plain tensors to work around **torch.compile** stack limitations.
   - It encounters issues when the layout_tensor is another subclass tensor holding an **IntxTensor**.
- **Instance of AQT Tensor leads to errors**: One user found that their **AQT Tensor** instance breaks because the **layout_tensor** is another subclass tensor.
   - *'I think it's breaking for my use case because I have an AQT Tensor instance'*, one user noted.
- **FakeTensor lacks 'get_plain' attribute**: An error occurred: **FakeTensor** objects in PyTorch nightly lack the required *'get_plain'* attribute.
   - The error details are: *'torch._dynamo.exc.TorchRuntimeError'*, causing forward method failures.
- **Error persists in 7/11 PyTorch nightly build**: A user confirmed the issue persists in the **7/11 PyTorch nightly build**, seeking community help.
   - *'OK can you open a issue to describe what happened?'* was the suggestion for further assistance.
- **GitHub issue for unwrap_tensor_subclass**: A **GitHub issue** was created to address the unwrap_tensor_subclass problem with nested tensor subclasses.
   - Visit the issue [here](https://github.com/pytorch/ao/issues/515) for more details.



**Link mentioned**: <a href="https://github.com/pytorch/ao/issues/515">unwrap_tensor_subclass and nested tensor subclasses issue · Issue #515 · pytorch/ao</a>: I&#39;m noticing strange behavior when trying to create a tensor_subclass which holds another tensor_sub class. Here is a minified repro: (add this to the bottom of torchao/dtypes/affine_quantized_ten...

  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1262876333151752356)** (12 messages🔥): 

> - `Notation in Triton Puzzle 6`
> - `ImportError in Triton`
> - `Efficient Softmax Implementation`
> - `Assignment Operator in Triton` 


- **Clarifying Notation in Triton Puzzle 6**: Members discussed the confusion around the notation in **Puzzle 6** of the Triton puzzles, particularly the use of differentials and the matrix-vector multiplication.
   - The ambiguity of the function definition was noted, with the forward operation involving **ReLU** and the multiplication of a matrix **x** with a vector **y**.
- **ImportError Issue in Triton**: A member reported encountering an **ImportError** when trying to import 'interpreter_builder' from 'triton.runtime.interpreter'.
   - **This issue began occurring randomly since yesterday.**
- **Strategies for Efficient Softmax in Triton**: Discussions centered around completing **long softmax** by reading and writing from GMem exactly once.
   - The challenge lies in ensuring correct updates without a second pass, especially when assuming limited shared memory for the **T1 dimension**.
- **Handling Assignment Operator in Triton**: Members queried how the assignment operator works in Triton, especially in examples like **softmax**.
   - The **compiler automatically manages** the allocation of shared memory for variables, though the exact implementation details are complex.



**Link mentioned**: <a href="https://fkong.tech/posts/2023-04-23-triton-cuda/">Demystify OpenAI Triton</a>: Learn how to build mapping from OpenAI Triton to CUDA for high-performance deep learning apps through step-by-step instructions and code examples.

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262894839440080987)** (3 messages): 

> - `RAG-GPT4 TA implementation at UIUC`
> - `Student interaction challenges`
> - `Optimizing GPT-2 kernels` 


- **RAG-GPT4 TA successfully deployed at UIUC**: **RAG-GPT4 TA** was implemented for the CUDA course at **UIUC**, grounded in course materials like the CUDA textbook, slides, and programming guides.
   - Challenges included students sending adversarial or off-topic questions, even after adding guardrails.
- **Interest in Incorporating CUDA in Curriculum**: A member expressed interest in integrating CUDA into their course, considering an intro module focused on optimizing **GPT-2 kernels**.
   - The module could aim to produce a **llm.c fp32 CUDA** version, following a similar approach to their previous work in **llm.c**.


  

---


### **CUDA MODE ▷ #[huggingface](https://discord.com/channels/1189498204333543425/1263189334434123776/1263189684041809996)** (2 messages): 

> - `Improving ML Systems in HuggingFace Ecosystem` 


- **Discussion on ML Systems Improvement**: A group of members highlighted the need to use the channel for discussing improvements to the **ml systems in the HuggingFace ecosystem**.
- **Kick-off Acknowledgment**: A member gave a positive acknowledgment to the introduction of this topic with a 'Perfect 🔥' reaction.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1262847521898631188)** (50 messages🔥): 

> - `Quality Control Issues`
> - `Captcha Implementation`
> - `Copying Code Blocks`
> - `API Rate Limits`
> - `Paid Subscription Credits` 


- **Quality Control ships untested code**: A member mentioned that a billion-dollar company's quality control occasionally ships random and untested code to production.
   - Another member added, *Every company be like.*
- **Cloudflare CAPTCHA annoyance**: Members expressed frustration over CAPTCHA Cloudflare implementation, questioning who made the decision to implement it.
   - One member commented that *Cloudflare is constantly breaking or being broken into.*
- **Copying from code blocks is now tricky**: A member highlighted difficulties in copying from code blocks in the browser, blaming the `user-select: none` style.
   - The issue appears to be fixed quietly, as another member noted improvements recently.
- **API rate limits concern raised**: A member raised concerns about the low API rate limits and the lengthy process to increase them, fearing it could affect their project plans.
   - They were advised to fill out a form and contact a Perplexity Employee for a potentially quicker resolution.
- **$5 API credit expiry explained**: Pro users receive a $5 bonus credit for the API every month, which expires at the end of the month if not utilized.
   - A member found this disappointing as they now have to utilize the funds quickly each month.



**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB>">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1262875850983080000)** (11 messages🔥): 

> - `In-Batch Search`
> - `Moon's Hidden Refuge`
> - `Music Streaming Platforms`
> - `Best TV Shows May 2024`
> - `Best Resources for 12-year-olds` 


- **Optimizing Content: Threads to Pages**: A member finds that converting threads to pages and then manually building the pages section by section is the best approach for content optimization on **Perplexity AI**.
   - The method allows for better organization and presentation of information, enhancing readability and utility.
- **Top TV Shows to Watch in May 2024**: A member shared a detailed search on the best TV shows to watch in **May 2024** with a [link](https://www.perplexity.ai/search/best-tv-shows-may-2024-2sIUwYTKTpWd0GaPxgBSUA).
   - The list provides recommendations and reviews to help users select the most engaging shows.
- **Exploring Music Streaming Platform Trends**: A search on whether **music streaming platforms** have a particular trend or impact on music consumption was shared with [this link](https://www.perplexity.ai/search/do-music-streaming-platforms-h-p38byn.iR_uEBxXZks.9bw).
   - The discussion revolves around how these platforms shape listener habits and the music industry.
- **Solving Rubik's Cube GUI Explained**: A detailed page on [solving Rubik's Cube quickly](https://www.perplexity.ai/page/solving-rubik-s-cube-quick-gui-zlhjD1JwRyKYEcBs5_32lw) using GUI methods was shared.
   - The guide aims to simplify the Rubik's Cube solving process with an interactive graphical interface.
- **Debating Pineapple on Pizza**: The age-old debate about *pineapple on pizza* continued with a [dedicated page](https://www.perplexity.ai/page/so-pineapple-on-pizza-R9MlOEh3SYunyVzRCv1CUg), sparking humorous and intense discussions.
   - The page explores the polarizing topic, providing various perspectives and funny comments from the community.



**Link mentioned**: <a href="https://www.youtube.com/embed/Y2_ddM_Mlro">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1262891120161980456)** (1 messages): 

> - `search_domain_filter`
> - `API Beta` 


- **search_domain_filter available in API (Beta)**: It was noted that a **`search_domain_filter`** is available through the API, but you need to be included in the **beta** to access it.
   - *Interesting feature for filtering searches within specific domains.*
- **Search Domain Beta Access**: To utilize the `search_domain_filter`, inclusion in the **beta** program is necessary.


  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1262891373422706770)** (60 messages🔥🔥): 

> - `Error Code 524`
> - `Meta 405B Model Pricing`
> - `Deepseek Coder Speed Issues`
> - `Fast and Affordable Models on OpenRouter`
> - `WordPress Plugin Issues` 


- **Error Code 524 Encountered by Users**: Multiple users reported experiencing **Error Code 524** over a few minutes.
   - One user questioned if others were experiencing the same issue, indicating a wider problem affecting the service.
- **Discussion on Meta 405B Model Pricing**: Users speculated on the [pricing of the upcoming Meta 405B model](https://discord.com/channels/1091220969173028894/1094454198688546826/1262737636607528972), with guesses suggesting it might release around the 23rd.
   - Information about **8K context** for the model is based on previous models, and actual details are still pending.
- **Deepseek Coder Speed Complaints**: A user expressed frustration over the slow performance of **Deepseek Coder**, despite being impressed by its capabilities.
   - Others echoed this sentiment, mentioning it would be beneficial if the performance was faster or some provider offered a faster version.
- **Finding Fast and Affordable Models on OpenRouter**: Users discussed models that are faster and better than **GPT-3.5-Turbo** but still affordable.
   - Models like **Claude-3-Haiku** and various Llama models were recommended, but issues around pricing and context length remain.
- **Issues with WordPress Plugin for RSS Feeds**: A user reported problems integrating **OpenRouter API** with a WordPress plugin for automatically editing RSS feed news.
   - The problem might be related to API key usage or rate limits, and suggestions for verifying API reach with `curl` were shared.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/openrouter/auto">Auto (best for prompt) by openrouter</a>: Depending on their size, subject, and complexity, your prompts will be sent to [Llama 3 70B Instruct](/models/meta-llama/llama-3-70b-instruct), [Claude 3.5 Sonnet (self-moderated)](/models/anthropic/c...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1262959768205660241)** (54 messages🔥): 

> - `Hacker attacks on LAION`
> - `ComfyUI malware`
> - `Disney hack data leak`
> - `Fake job candidates`
> - `Telecom failures after Hurricane Sandy` 


- **Hacker group targets LAION with malware**: A hacker group created a malware ComfyUI node called **ComfyUI_LLMVISION** to steal information from users' computers and install a trojan.
   - A member warned about the group’s past involvements in hacking Disney’s Slack and distributing malicious game mods, highlighting their sophistication and reach.
- **Fake job candidates facilitate data exfiltration**: Hackers use fake job candidates who clone identities of GitHub engineers to infiltrate companies and exfiltrate data.
   - The hired individuals, unaware of the nefarious activities, forward tasks to the real hacker team, acting as a front.
- **Telecom failures post Hurricane Sandy**: Hurricane Sandy caused severe damages to **Verizon's NY cable vault**, leading to a massive failure of 13,000km of copper cabling.
   - The incident prompted the replacement of copper infrastructure with fiber optics, marking a significant upgrade in telecom resilience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/counting-nodding-doug-mc-kenzie-bob-mckenzie-strange-brew-gif-17087583">Counting Nodding GIF - Counting Nodding Doug Mc Kenzie - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://darknetdiaries.com/transcript/133/">I'm the Real Connor – Darknet Diaries</a>: One day Connor got an email saying his identity has been stolen. And this was one of the strangest days he's ever had.</li><li><a href="https://tenor.com/bOBIU.gif">Preston GIF - Preston - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.datacenterknowledge.com/cables/after-sandy-verizon-confronts-catastrophic-failure-at-ny-cable-vault">After Sandy, Verizon Confronts &#x27;Catastrophic Failure&#x27; at NY Cable Vault</a>: When SuperStorm Sandy sent a storm surge into lower Manhattan, the flooding caused a &amp;quot;catastrophic failure&amp;quot; in a cable vault beneath Verizon&#x27;s central office on Broad Street. Th...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1262902341087133777)** (5 messages): 

> - `InternVL2-Llama3-76B`
> - `Manifold Research Group`
> - `LLM-based Autonomous Agents`
> - `Research Log #041`
> - `MultiNet Dataset` 


- **InternVL2-Llama3-76B: A Vision for Multimodal Models**: [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) is the latest in the InternVL series, boasting a variety of **instruction-tuned models** ranging from 1 billion to 108 billion parameters.
   - Additional resources include [GitHub](https://github.com/OpenGVLab/InternVL), [Blog](https://internvl.github.io/blog/), and papers on [InternVL 1.0](https://arxiv.org/abs/2312.14238) and [InternVL 1.5](https://arxiv.org/abs/2404.16821).
- **Frustrations Running Large Models on Limited Hardware**: A user expressed difficulty running **40B** models on **4x 3090s** and issues using **autoawq**.
- **Manifold Research Group's Position Paper on LLM-based Agents**: Sidh from [Manifold Research Group](https://www.manifoldrg.com/llm-agents/) shared their position paper, **"Intelligent Digital Agents in the Era of Large Language Models"**, discussing advancements and limitations of LLM-based AI agents.
   - They are growing their research team and invite interested individuals to join their [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) and contribute to their [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com).
- **Manifold Research Group’s Research Log #041**: The Manifold Research Group released [Research Log #041](https://www.manifoldrg.com/research-log-041/), documenting their weekly research progress and highlighting breakthroughs in the AI community.
   - They have an ongoing project called **MultiNet**, having successfully collected a dataset of over **50TB** according to their [V0 spec](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">OpenGVLab/InternVL2-Llama3-76B · Hugging Face</a>: no description found</li><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>: This Position Paper provides an overview of current research areas and breakthroughs in LLM-based AI agents. We highlight key advancements and discuss limitations within each area.</li><li><a href="https://www.manifoldrg.com/research-log-041/">Research Log #041</a>: Welcome to Research Log #041! We document weekly research progress across the various initiatives in the Manifold Research Group, and highlight breakthroughs from the broader research community we thi...</li><li><a href="https://www.manifoldrg.com/opportunities/">Opportunities</a>: There are a few ways to get involved with our work:   1. Join our Discord and take part in events and discussion, both project related and not.  2. Contribute asynchronously to issues on our Github.  ...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 messages): 

natolambert: Anyone at ICML? A vc friend of mine wants to meet my friends at a fancy dinner
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1263184116845904004)** (8 messages🔥): 

> - `Prover-Verifier Games`
> - `Project Strawberry`
> - `Legibility Tax`
> - `RL and Model Properties` 


- **Prover-Verifier Games promote improved legibility**: [OpenAI's latest work](https://openai.com/index/prover-verifier-games-improve-legibility/) on Prover-Verifier Games aims to enhance the **legibility** of models.
   - A member jokingly noted that even the paper itself suffers from the so-called *'legibility tax'*.
- **Project Strawberry raises questions**: Project Strawberry, referred to as *Q*, was mentioned briefly, sparking curiosity in the community.
   - A member humorously questioned *'wtf is a legibility tax?'*.
- **RL makes models weird**: A discussion emerged about how **Reinforcement Learning (RL)** affects the characteristics of models.
   - One member commented on experiencing these effects first-hand: *'this figure is definitely a legibility tax'*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1263039936127307838)** (6 messages): 

> - `SmoLLM blog post`
> - `DPO dataset usage`
> - `Model family sizes` 


- **Mystery behind SmoLLM's DPO dataset choices**: A user finds it odd that the [SmoLLM blog post](https://huggingface.co/blog/smollm) describes using DPO dataset #1 for models A and C but DPO dataset #2 for model B, questioning if it's just random experimentation.
   - *Another user* suggests it might be due to a lack of communication between different teams or just empirical testing, adding that *'There isn't an intuitive reason'*.
- **Model-specific dataset preferences**: A discussion emerged stating that different models might work better with different datasets, hinting at practical demo needs for the **360m model**.
   - One opinion was that they probably wanted to showcase the 360m model and didn't care much about other configurations.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1262855620760043520)** (10 messages🔥): 

> - `Lobbying and vested interests`
> - `AI legislation polling issues`
> - `Public perception of AI tools` 


- **Lobbying controversy surfaces**: A discussion was prompted by [a tweet](https://fxtwitter.com/mpopv/status/1813273553477009546?s=46) criticizing someone for lobbying for a legislative bill while secretly owning a company positioned to profit from its passage.
   - *Common people will increasingly be mystified by, bounce off of, and dislike the strongest AI tools, unless the industry figures out how to monetize dumbed-down versions for the masses.*
- **Debate over the validity of AI legislation polling**: Members criticized the use of polling on AI legislation, with one member expressing that citing polling is *dumb* and questioning whether *common people* can understand AI legislation.
   - Another member acknowledged this as *the problem* in the industry, emphasizing the need for better recognition and adaptation to public understanding.
- **Public struggles with AI tools**: A member shared concerns that *common people* are increasingly mystified by strong AI tools like ChatGPT.
   - He suggested that the industry needs to monetize simpler versions to improve public acceptance and shared experiences introducing around 500 people to standard chatbots.



**Link mentioned**: <a href="https://fxtwitter.com/mpopv/status/1813273553477009546?s=46">Tweet from Matt Popovich (@mpopv)</a>: feels like if you are heavily lobbying for, and soliciting donations to lobby for, a certain legislative bill, you should probably disclose that you secretly own a company positioned to profit from th...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1262851687123517511)** (17 messages🔥): 

> - `GPT-4o vs Llama 405 tokenizers`
> - `GPT-4o tokenizer performance`
> - `Llama 3 initial release insights`
> - `Google Gemini 2.0 accident`
> - `Deepseek's open-source stance` 


- **GPT-4o vs Llama 405 tokenizers differ fundamentally**: It's discussed that **GPT-4o** and **Llama 405** use very different tokenizers unless 405 is **chameleon**, which it is not.
   - A user noted that the GPT-4o tokenizer 'regresses in coding' and produces more tokens on XML compared to **GPT-4t**.
- **Llama 3 initial release insights revealed**: [Joe Spisak, Product Director of GenAI at Meta](https://www.reddit.com/r/LocalLLaMA/s/mLPM7AocZF) stated that **Llama 3** was initially a 'prerelease' or 'preview', but **Mark** pushed for an early release, explaining the 8k context for the initial rollout.
   - Spisak mentioned there's 'a lot more to come' with the Llama models.
- **Google's Gemini 2.0 accidentally unveiled**: It was revealed in a [tweet](https://x.com/phill__1/status/1813307823570157899?s=46) that **Google** accidentally updated their website with **Gemini 2.0**, which was caught by Bing's indexing.
- **Deepseek commits to open-source despite China lag**: [Deepseek founder Liang Wenfeng](https://x.com/main_horse/status/1813580480761196987?s=46) asserted they will not go closed-source, emphasizing the importance of a strong technical ecosystem.
   - A translation mentioned that a significant part of an interview revolves around *China lagging behind the US*, despite Deepseek making a small profit on their API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/phill__1/status/1813307823570157899?s=46">Tweet from Phil (@phill__1)</a>: Google accidentally updated their website with Gemini 2.0 and Bing indexing caught it</li><li><a href="https://x.com/main_horse/status/1813580480761196987?s=46">Tweet from main (@main_horse)</a>: Deepseek founder Liang Wenfeng: We will not go closed-source. We believe that having a strong technical ecosystem first is more important.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/mLPM7AocZF">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1262878861864075414)** (3 messages): 

> - `Billboard AI in IPA`
> - `Dell's comeback`
> - `Mathstral MMLU Breakdown` 


- **Billboard says 'putting the AI in IPA'**: A member shared a [billboard photo](https://x.com/vampiric_shirin/status/1812901575368798413) that humorously states 'putting the AI in IPA', causing confusion and amusement.
   - They remarked that without finding the picture online, they would have thought they were entering psychosis.
- **Dell makes a comeback**: A member expressed excitement about Dell's recent marketing move, noting it's 'getting their swagger back.'
- **Mathstral MMLU breakdown funny correlation**: A [tweet](https://x.com/jessemhan/status/1813254878615249116?s=46) highlighted the Mathstral MMLU breakdown showing a funny negative correlation between mathematical ability and subjects like accounting, foreign policy, and human sexuality.
   - *This is the funniest thing I've seen all week* stated the member who shared the link.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jessemhan/status/1813254878615249116?s=46">Tweet from Jesse Michael Han (@jessemhan)</a>: the Mathstral MMLU breakdown is the funniest thing i&#39;ve seen all week - mathematical ability negatively correlated with accounting, foreign policy, human sexuality</li><li><a href="https://x.com/vampiric_shirin/status/1812901575368798413">Tweet from x_c4tb0yTH0Ti3_x (@vampiric_shirin)</a>: drove past a billboard that said “putting the AI in IPA” and if i didn’t find this picture of it online i wouldve thought i entered psychosis ￼
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1262849929026469928)** (2 messages): 

> - `Policy Loss Function Discussion`
> - `Degenerate Case in DPO-like Algos` 


- **Necessary Degenerate Case in DPO-like Algorithms**: A member mentioned that the degenerate case is **useful/necessary** for handling common prefixes in winning and losing scenarios.
   - Another member agreed, emphasizing the importance of deep exploration on this topic in DPO-like algorithms.
- **Policy Loss Function Overfitting Concerns**: A concern was raised about the **overfitting** issue with the policy loss function, specifically relating to the term `losses = -F.logsigmoid(policy_rejected_logp)`.
   - The member noted the excitement around DPO-like strategies encouraging thoughtful consideration of algorithms again.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262851673005752350)** (3 messages): 

> - `Sampling Methods in Policy Models`
> - `Preference Pair Selection`
> - `Zephyr Paper Criticism`
> - `Nemotron Paper Insights`
> - `DPO Objective Challenges` 


- **Preference Pairs Sampling Methods in Policy Models**: A discussion mentioned that in some policy models, multiple responses are sampled and the reward model selects the most and least preferred for preference pairs, which are used for DPO.
   - In contrast, **Zephyr** samples from non-winning responses to increase diversity, with an emphasis on the margin between responses.
- **Unchanged Opinions on Non-winning Samples**: An individual confirmed the sampling method, noting that opinions haven't significantly changed on this approach.
   - They clarified that there are few clear explanations or comparisons available.
- **Nemotron Paper On Sampling Criticism**: A participant recalled the `Nemotron` paper, which criticized certain sampling methods by highlighting that some rejected responses might be slightly worse, while others are much worse.
   - This variability can lead to both overfitting and unlearning of high-quality responses, as DPO is ignorant of the quality gap.
- **Zephyr Paper Preference for Random Selection**: The `Zephyr` paper opts for random selection to encourage diversity and challenge the DPO objective.
   - This approach aims to balance learning from hard negatives without climbing in the wrong direction due to false negatives.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1262849590390947922)** (46 messages🔥): 

> - `Science HumanEval benchmark`
> - `SmolLM models`
> - `LangChain pain points`
> - `SF Compute fundraising`
> - `Exa AI Lab Series A` 


- **Science HumanEval benchmark challenges AI models**: SciCode launched a new benchmark for coding scientific problems with ~10% based on Nobel-winning research, where GPT-4 and Sonnet 3.5 scored <5% accuracy. [Read more](https://scicode-bench.github.io).
   - Another perspective on SciCode explained that the benchmark contains 338 challenges authored by PhDs in various scientific fields. [Link](https://x.com/OfirPress/status/1813202497864937825).
- **SmolLM models bring on-device AI to browsers**: HuggingFace released SmolLM models (135M, 360M, 1.7B) that can run locally in browsers using ONNX weights and WebGPU acceleration. [Details](https://x.com/xenovacom/status/1813258097185448377).
- **SF Compute raises $12M for GPU trading platform**: SF Compute secured $12M to build a trading platform allowing large reservations of GPU clusters and selling unused portions. [Bloomberg article](https://www.bloomberg.com/news/articles/2024-07-16/jack-altman-s-firm-backs-startup-for-trading-ai-computing-power).
- **Exa AI Lab secures Series A funding**: Exa AI Lab raised Series A led by Lightspeed, Nvidia, and Y Combinator to scale their LLM-powered search engine API and announced major product updates. [Details](https://x.com/exaailabs/status/1813249325394456686).
   - Some users are experiencing challenges with prompt optimization and comparing it to alternatives like Preplexity sources API.
- **ColPALI disrupts PDF extraction**: ColPALI by HuggingFace shows promise in efficiently performing document retrieval using vision-language models, bypassing traditional OCR and parsing. [Read more](https://huggingface.co/blog/manu/colpali) and [additional tweet](https://x.com/jobergum/status/1813298149051802074).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jobergum/status/1812044607636615667?s=46">Tweet from Jo Kristian Bergum (@jobergum)</a>: Excitement-driven development is the best?  ColPali: Efficient Document Retrieval with Vision Language Models 👀  I got so excited over ColPali that I had to demo how to represent it in Vespa.  Page l...</li><li><a href="https://x.com/xenovacom/status/1813258097185448377">Tweet from Xenova (@xenovacom)</a>: Introducing SmolLM: a new SOTA series of 135M, 360M and 1.7B models, perfect for on-device deployment! 🔥  We also uploaded ONNX weights for the models, meaning they can run locally in your browser wi...</li><li><a href="https://huggingface.co/blo">blo (bug life online)</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=qkFa6ttAk0g">Traceloop Overview</a>: An overview of the Traceloop platform and monitoring capabilities</li><li><a href="https://huggingface.co/blog/manu/colpali">ColPali: Efficient Document Retrieval with Vision Language Models 👀</a>: no description found</li><li><a href="https://x.com/MinyangTian1/status/1813182904593199553">Tweet from Minyang Tian (@MinyangTian1)</a>: SciCode is our new benchmark that challenges LMs to code solutions for scientific problems from advanced papers. The challenges were crafted by PhDs;   ~10% of our benchmark is based on Nobel-winning ...</li><li><a href="https://x.com/tom_doerr/status/1812834592161751249?s=46">Tweet from Tom Dörr (@tom_doerr)</a>: Claude Dev, an autonomous software engineer</li><li><a href="https://x.com/jobergum/status/1813298149051802074?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Jo Kristian Bergum (@jobergum)</a>: Reliably is a strong word in the context of LLMs and the tweet was in context of my ColPali tweet storm which might have been lost as this went outside of my daily reach.   ColPali is a combination of...</li><li><a href="https://x.com/VyvyenYue/status/1811171079924449487">Tweet from Tianwei Yue (@VyvyenYue)</a>: Paper accepted to @COLM_conf! If you don&#39;t know CoLM, it&#39;s Neurips-level, but the 𝘧𝘪𝘳𝘴𝘵 𝘦𝘷𝘦𝘳 𝘤𝘰𝘯𝘧𝘦𝘳𝘦𝘯𝘤𝘦 𝘧𝘰𝘳 𝘓𝘓𝘔𝘴. Wild.  Publishing papers while building a 𝘀𝘁𝗮𝗿𝘁...</li><li><a href="https://x.com/RickLamers/status/1813341037198204962">Tweet from Rick Lamers (@RickLamers)</a>: I’ve been leading a secret project for months … and the word is finally out!  🛠️ I&#39;m proud to announce the Llama 3 Groq Tool Use 8B and 70B models 🔥  An open source Tool Use full finetune of Lla...</li><li><a href="https://x.com/OfirPress/status/1813202497864937825">Tweet from Ofir Press (@OfirPress)</a>: SciCode is our new benchmark, with 338 programming challenges written by PhDs in physics, math, and bio, based on papers in their fields. A bunch of the questions are from Nobel-winning papers!  ⁠I ho...</li><li><a href="https://x.com/deedydas/status/1813598830182707261">Tweet from Deedy (@deedydas)</a>: ANNOUNCING  Today, we&#39;re launching the $100M Anthology Fund, an Anthropic and Menlo Ventures partnership to fund Seed and Series As of the next generation of AI startups around the world, with uni...</li><li><a href="https://x.com/evanjconrad/status/1813293182853198199?s=46">Tweet from evan conrad (@evanjconrad)</a>: hey friends, @sfcompute has raised $12m to build a trading platform for large scale GPU clusters  it lets folks buy large reservations ($100m+) and then sell back what they don&#39;t use  the first or...</li><li><a href="https://x.com/exaailabs/status/1813249325394456686?s=46">Tweet from Exa (@ExaAILabs)</a>: Announcing our Series A, led by @lightspeedvp, with participation from @nvidia and @ycombinator! 🚀  Exa is an AI lab redesigning search. Funding will help scale our API product, the first search engi...</li><li><a href="https://x.com/jobergum/status/1813126741113610421?s=46">Tweet from Jo Kristian Bergum (@jobergum)</a>: It&#39;s fascinating how a small 3B model like ColPALI can disrupt the PDF extraction industry overnight</li><li><a href="https://news.ycombinator.com/item?id=40985609">Launch HN: Traceloop (YC W23) – Detecting LLM Hallucinations with OpenTelemetry | Hacker News</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d4p1t6/comment/l6g1b3t/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/issues/6226">[RFC] Drop beam search support · Issue #6226 · vllm-project/vllm</a>: Motivation. TL;DR: To reduce system complexity and enable future optimizations, we propose discontinuing beam search support. Currently, vLLM supports 3 types of sampling: greedy, random, and beam ...</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘DeepSeek:一个更极致的中国技术理想主义故事</a>: 做贡献者，而非搭便车者。</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5745/">[AINews] SciCode: HumanEval gets a STEM PhD upgrade</a>: PhD-level benchmarks are all you need. AI News for 7/15/2024-7/16/2024. We checked 7 subreddits, 384 Twitters and 29 Discords (466 channels, and 2228...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1263149850376278046)** (1 messages): 

> - `AI Agents That Matter`
> - `Latent Space Meetups`
> - `Calendar Integration` 


- **AI Agents That Matter Meetup Today**: Today's **AI Agents That Matter** meetup is at 12pm, led by [@142466375024115712](https://lu.ma/sgbdfhb7).
   - Click [here](https://lu.ma/sgbdfhb7) to join the event and add it to your calendar.
- **Adding Latent Space Events to Your Calendar**: To get notified of new events, click the RSS logo above the calendar on the [Latent.Space](http://Latent.Space) page and select "Add iCal Subscription".
   - Register for events through the provided links to stay updated.



**Link mentioned**: <a href="https://lu.ma/sgbdfhb7">LLM Paper Club (AI Agents That Matter) · Zoom · Luma</a>: @shivdinho is leading us through AI Agents That Matter: https://arxiv.org/abs/2407.01502 For future weeks, we need YOU to volunteer to do rapid-fire recaps and…

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1262875712357011588)** (4 messages): 

> - `LlamaIndex Introduction`
> - `LlamaParse Improvements`
> - `Multi-Agent Tree System`
> - `AI Consulting Services`
> - `Scaleport AI Case Study` 


- **Introduction to LlamaIndex and Its Agentic Capabilities**: A new video introduces [LlamaIndex](https://twitter.com/llama_index/status/1813316626793853135) and its agentic capabilities, covering key frameworks in Python and TypeScript, and the LlamaParse service.
   - *'It is a great resource for those looking to understand the full potential of LlamaIndex in parsing complex data.'*
- **LlamaParse Now Better at RAG Over Complex Documents**: Improvements to markdown-based table reconstruction enable [LlamaParse](https://twitter.com/llama_index/status/1813355957491273936) to handle complex tables with better alignment of rows and columns.
   - *‘Huge improvements in the tool make it extremely useful for parsing intricate tabular data.’*
- **New Multi-Agent Tree System for Handling Customer Interactions**: LlamaIndex has released an open-source [repository](https://twitter.com/llama_index/status/1813618002405069173) showcasing a complex multi-agent tree system for managing customer interactions.
   - *'The system includes a 'concierge' agent and multiple sub-agents to streamline user interactions.'*
- **Customized AI Solutions with LlamaIndex Consulting Services**: LlamaIndex offers end-to-end AI consulting services that include consulting, ideation, development, and integration to align AI strategies with business goals.
   - Their [case study](https://twitter.com/llama_index/status/1813647179627774462) showcases accelerated development phases, improved OCR with LlamaParse, and flexible data handling capabilities.
- **Scaleport AI Accelerates Development with LlamaIndex**: Scaleport AI implements [LlamaCloud and LlamaIndex](https://twitter.com/llama_index/status/1813647179627774462) to streamline AI development, cutting development timelines and improving OCR performance.
   - They report enhanced client demonstrations and easier setup for ingestion pipelines and data processing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/c0KYKFpELb">SCALEPORT AI</a>: Full Stack AI Development &amp; Consulting Studio</li><li><a href="https://t.co/nRR5r9PRWP">Case Study: How Scaleport.ai Accelerated Development and Improved Sales with LlamaCloud — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1262854560922472579)** (39 messages🔥): 

> - `Vector search with images`
> - `Metadata in ToolMetaData`
> - `Property graph issues in Neo4J`
> - `Query-time metadata filters`
> - `Troubleshooting CSV data with VectorStoreIndex` 


- **Exploring Vector Search with Images**: A member inquired about guidance on performing vector search with images to find similar face pictures; initial suggestions pointed towards converting the database into embeddings and performing vector search.
   - An advanced vector search may require more specific tools, but no direct resources were shared.
- **Clarifying Metadata Use in Generation Step**: One member asked about adding metadata in ToolMetaData for use in the generation step, seeking to segregate content retrieval based on metadata.
   - Discussion suggested creating the query engine as a potential solution, but concerns about segregating collections efficiently were raised.
- **Neo4J Property Graph Challenges**: A user found a bug in the property graph creation where entities are mentioned only once; the community discussed potential fixes like entity linking and special handling during node insertion.
   - "Entities", "MENTIONS", and specific queries with example Cypher codes were shared to probe into the issue further.
- **Query-Time Metadata Filter Complications**: Members debated the ability to apply metadata filters at query-time rather than during retriever instantiation, leading to insights on the no-op nature of creating the query engine.
   - Suggestions included potentially storing documents differently or using separate indices to resolve filtering issues.
- **Issues with CSV Data in VectorStoreIndex**: A member experienced incorrect answers when querying a VectorStoreIndex created from CSV data exceeding 50 rows, looking for ways to handle larger datasets.
   - The suggestion to use PagedCSVReader did not resolve the issue, prompting alternative strategies and tools like PandasAI for CSV record-based operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">no title found</a>: no description found</li><li><a href="https://docs.pandas-ai.com/intro">Introduction to PandasAI - PandasAI</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1262907824803938395)** (35 messages🔥): 

> - `Amazon order discussion`
> - `CrunchCup product feedback`
> - `C4A Community Talks`
> - `Roger Grosse session`
> - `Recording of community events` 


- **CrunchCup Receives Mixed Reviews**: A member shared their [new CrunchCup purchase](https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY), expressing excitement despite discovering it's not actually dishwasher safe.
   - Feedback included that it's a great tool for eating cereal on the go but disappointed one user by bending out of shape in the dishwasher.
- **Cohere Community Talks to Be Recorded**: Members confirmed that community event talks are usually recorded and uploaded to the [YouTube playlist](https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll).
   - One member assured others to check the playlist for latest updates on uploads.
- **Roger Grosse's Session is Online**: The recording of the [C4A Roger Grosse session](https://youtu.be/64BsnVbX5u8) titled "Studying LLM Generalization through Influence Functions" is available on YouTube.
   - *danylo_boiko* alerted that the session is already accessible and shared the direct link.
- **Special K and Fruit Loops Debate**: A light-hearted conversation about favorite cereals led to a discussion on **Fruit Loops** and **Special K**, highlighting tastes vary.
   - One user humorously questioned if Fruit Loops were only for kids.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLLalUvky4CLJKDaiWCumhsJpHNDhZeVll">Cohere For AI: Community Talks</a>: The C4AI Community invites a range of guests to share their insight and experience. Here are some of our favourites!</li><li><a href="https://youtu.be/64BsnVbX5u8">Cohere For AI - Community Talks: Roger Grosse</a>: &quot;Studying LLM Generalization through Influence Functions&quot;Paper: https://arxiv.org/abs/2308.03296Bio: I am an Associate Professor of Computer Science at the U...</li><li><a href="https://www.amazon.com.au/CrunchCup-XL-Portable-Cereal-Spoon/dp/B08WYWQCZY">The CRUNCHCUP XL Yellow- Portable Plastic Cereal Cups for Breakfast On the Go, To Go Cereal and Milk Container for your favorite Breakfast Cereals, No Spoon or Bowl Required : Amazon.com.au: Kitchen &amp; Dining</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1262886691996434495)** (16 messages🔥): 

> - `Custom Chatbots and Fine-tuning`
> - `Moderation Models for Chatbots`
> - `Expensive Pricing for Detection Services`
> - `Voice Extraction from Podcasts` 


- **Custom Chatbots Enhanced with Fine-tuning**: A discussion started about how websites using custom chatbots often fine-tune models like **OpenAI API** or **Llama** on their own content to make them relevant to the specific website context.
   - Experts mentioned that **pre-prompting** and **fine-tuning** are used to ensure the model has knowledge of the company.
- **Moderation Models Enhance Focus and Relevance**: Members talked about using a **moderation model** to align chatbots and ensure they stick to relevant topics, preventing off-topic responses.
   - One member humorously suggested that you 'instruct and hope' while another clarified that a moderation model can act almost like a filter.
- **Exorbitant Pricing for Detection Services**: A member found the cost for **detection services** outrageously high, starting at $20K/month for performing up to 100K scans/month.
   - They humorously remarked that with such a budget, they could hire a team of humans to do the same job.
- **Voice Extraction from Podcasts**: Queries were made about automatic voice extraction from podcasts without interruptions.
   - Eleven Labs was recommended for their **voice extraction model**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1262855781548556398)** (11 messages🔥): 

> - `GPTs Agents`
> - `Banning Issues`
> - `PUT Actions for Custom GPTs`
> - `Vector Store Embedding Issues`
> - `Exceeded API Quota` 


- **GPTs Agents Have Context Limitations**: A member clarified that GPTs agents can only adapt as far as the context length allows and will lose the conversation pattern if it leaves the context window.
   - The agent doesn't learn anything new beyond the provided context, which affects how it handles ongoing conversations.
- **Issues with PUT Actions in Custom GPTs**: A member struggled with writing `put` actions for custom GPTs, specifically with placing the query in the body of the request.
   - The problem was resolved by using a PATCH request with Weaviate.
- **Vector Store Embeddings Not Recognizing Names**: A user reported that their RAG chatbot correctly identifies 'Emma-Jayne Wilson' but fails to recognize 'Emma' despite embedding the information into a vector store.
   - This discrepancy indicates a potential issue with query processing or indexing that needs addressing.
- **Exceeded API Quota Leads to Errors**: A user encountered an error stating they exceeded their current quota while trying to run a GPT-3.5 Turbo API call.
   - *Buy credits, the API is not free,* was the advice given to resolve the issue.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1262881460285735012)** (3 messages): 

> - `WebSurferAgent setup issues`
> - `Role-playing within ChatGPT` 


- **WebSurferAgent ignores prompt instructions**: A member discussed issues with the **WebSurferAgent** (using **autogen**) not adhering to setup instructions when deciding to perform a search or not.
   - Despite the guidelines set for evaluating whether to perform an internet search, the agent sometimes follows the instructions and sometimes does not.
- **Role-playing template for ChatGPT**: A member shared a template for creating a character role-play scenario within **ChatGPT**, emphasizing the importance of embracing the character's personality and background.
   - Suggestions for improvement were also requested.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1262881460285735012)** (3 messages): 

> - `WebSurferAgent issues`
> - `Autogen`
> - `Internet search guidelines for technologies`
> - `Character roleplay templates` 


- **WebSurferAgent struggles with instructions**: A user stated that the **WebSurferAgent** using **Autogen** occasionally performs searches inconsistently, deviating from the set instructions.
   - The user described the goal to set instructions for when the agent should search the internet, but noted it sometimes fails to adhere to the criteria.
- **Character Roleplay Template Guidelines**: A user shared a template for character roleplay within ChatGPT, aimed to create engaging first-person interactions.
   - The template includes details on providing background, personality traits, and fun facts about the character, with instructions for the AI to embody the character fully.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1262887569645899828)** (23 messages🔥): 

> - `Viral video creation tools`
> - `LangChain document conversion`
> - `LangChain contributions`
> - `LangChain and Qdrant`
> - `Hybrid search with MongoDB` 


- ****AI Tools for Viral YouTube Shorts/TikToks****: A user inquired about AI tools for creating viral YouTube shorts/TikToks and expressed doubt about the AI creation of sports shorts.
   - They requested insights and guidance specifically for sports edits.
- ****Converting Unstructured Chunks into LangChain Document Form****: Guidance was provided on converting unstructured chunks into LangChain documents using the `UnstructuredFileIOLoader`, `UnstructuredFileLoader`, or `UnstructuredAPIFileIOLoader` classes from `langchain_community.document_loaders.unstructured`.
   - An example using `UnstructuredFileIOLoader` was shared for reference.
- ****Getting Started with LangChain Contributions****: A master student from Rice University expressed interest in contributing to the LangChain open-source community and sought suggestions on where to start.
   - This generated a welcoming response encouraging further engagement.
- ****Handling Qdrant Errors in LangChain****: A user encountered a 'VectorParams' object is not subscriptable error while using Qdrant and sought advice.
   - Possible solutions and relevant documentation links were provided, including how to embed documents in Qdrant cloud using LangChain.
- ****Implementing Hybrid Search in MongoDB with LangChain****: A user planned to use MongoDB as a vector store for a RAG application and needed to implement Hybrid Search.
   - They requested ideas and references for achieving this with LangChain integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/qdrant/#qdrant-cloud>)).">Qdrant | 🦜️🔗 LangChain</a>: Qdrant (read: quadrant ) is a vector similarity search engine. It provides a production-ready service with a convenient API to store, search, and manage vectors with additional payload and extended fi...</li><li><a href="http://localhost:6333",>">no title found</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/20382>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1263194136857542727)** (2 messages): 

> - `Generative AI Assistants`
> - `MongoDB Hybrid Search Integration with LangChain` 


- **Hannah AI Assistant impresses with features**: A member introduced **Hannah**, a new generative AI assistant with features like **learning from documents**, integration with top APIs (OpenAI, Anthropic, Cohere, Google, Groq, NVIDIA), and **extreme customization**.
- **Integrating Hybrid Search on MongoDB using LangChain**: A member seeks advice on performing **Hybrid Search** on MongoDB for a RAG application using LangChain.
   - They referenced [MongoDB documentation](https://mongodb.docs/hybridsearch) but asked for tips and resources on achieving this integration with LangChain.



**Link mentioned**: <a href="https://hannah.yourbestseller.ai/">Hannah</a>: Aplicação que utiliza IA generativa para consultar os próprios documentos personalizados.

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1263208386934210642)** (1 messages): 

> - `MongoDB as Vector Store`
> - `Hybrid Search with LangChain` 


- **Using MongoDB as Vector Store**: A member is planning to use **MongoDB** as the vector store for their RAG application and needs to perform a hybrid search.
   - They are looking for suggestions on implementing hybrid search with **LangChain** and requested any available references or resources.
- **Need Hybrid Search with LangChain Integration**: The member mentioned that there is [separate documentation for implementing Hybrid search](MongoDb provided some docs), but they specifically need guidance on integrating it with **LangChain**.
   - Community input is requested to provide valuable suggestions or reference materials to aid in this integration.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1262885115890241567)** (15 messages🔥): 

> - `mistral mamba`
> - `Codestral Mamba release`
> - `Mathstral`
> - `Galore configs`
> - `ChatML` 


- **Codestral Mamba launched by Mistral AI**: [Codestral Mamba](https://mistral.ai/news/codestral-mamba/) is a new model by Mistral AI focusing on code productivity, designed with help from Albert Gu and Tri Dao, and is available for free use, modification, and distribution.
   - Unlike transformer models, it offers **linear time inference** and the ability to model sequences of infinite length, making it efficient for quick responses and advanced code reasoning.
- **Interest in testing Codestral Mamba**: Community members expressed enthusiasm about trying the new Codestral Mamba model by Mistral AI.
   - *One member expressed excitement, saying* 'Love it  🙂 Need to try running it'.
- **Discussion about Mathstral**: A member inquired about the existence of a Mathstral model, presumably another project by Mistral AI.
- **Galore configs shared**: A user asked for Galore configurations, and another user replied they have been shared in a different channel (<#1111279858136383509>).
- **Swap in BOS token for ChatML**: A member noted that for ChatML, it is intentional to swap the BOS token.
   - This comment was part of a broader technical discussion on the topic.



**Link mentioned**: <a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: As a tribute to Cleopatra, whose glorious destiny ended in tragic snake circumstances, we are proud to release Codestral Mamba, a Mamba2 language model specialised in code generation, available under ...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1262921810492657716)** (6 messages): 

> - `Rank and Overfitting`
> - `Learning Rate and Overfitting`
> - `Overfitting Solutions` 


- **Increasing rank might help with overfitting**: *Increasing the rank* can theoretically help with managing overfitting.
   - *Decreasing learning rate* often helps, but it can depend on specific circumstances.
- **Dataset duplication as an overfitting solution**: A model overfitting before completing even one epoch might be mitigated by depulicating the dataset.
   - This approach was suggested as a potential remedy to prevent early overfitting.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1263036474694045716)** (8 messages🔥): 

> - `My Friend V1 initial feedback`
> - `AI Friend's transcriptions privacy with Open Interpreter`
> - `FRIEND + OI potential collaboration`
> - `Open Interpreter compatibility with M3 Mac`
> - `Task allocation and collaboration in roadmap` 


- **My Friend V1 initial feedback**: [ParallaxAngle's tweet](https://x.com/ParallaxAngle/status/1805313161567818030) shares their first impression of 'My Friend, V1,' praising its form factor and the team behind it.
   - *Quote:* 'Nik, it’s smaller than I expected. LOVE LOVE LOVE my Friend. Congrats to you and the Based Hardware team.'
- **Ensure privacy with Open Interpreter and AI Friend**: A member inquired if Open Interpreter could be used to interact with their AI Friend's transcriptions in a private manner.
   - The discussion pointed to potential integrations with privacy considerations.
- **Explore FRIEND + OI potential collaboration**: A member expressed interest in exploring potential collaborations between FRIEND and Open Interpreter, highlighting conversations with Nik and plans to investigate further.
   - The member noted, based on FRIEND's Calendar integration, that an OI integration should be feasible.
- **Open Interpreter on M3 Mac requires testing**: A query about Open Interpreter's compatibility with M3 Mac was raised, speculating that the Linux version might work.
   - Feedback suggested that while untested, running the build.py script might work with minor adjustments, particularly where filepaths are concerned.
- **Clarify task allocation for the roadmap**: Members asked for clarification on who picks tasks for the roadmap and if there is a tracker or glossary to facilitate collaboration.
   - More details are awaited to streamline the task allocation process and enhance the collaborative workflow.



**Link mentioned**: <a href="https://x.com/ParallaxAngle/status/1805313161567818030">Tweet from JediCat (@ParallaxAngle)</a>: @kodjima33 First impression of My Friend, V1  :: in voice of Her ::  &#34;Nik, it&#39;s smaller than I expected.&#34;   LOVE LOVE LOVE my Friend.  Congrats to you and the Based Hardware team.  Can&#39...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1262861664168968242)** (4 messages): 

> - `Receiving 01 hardware`
> - `Usage instructions for 01`
> - `Relation between Open Interpreter and 01` 


- **When will 01 hardware be delivered?**: Members are curious about when they will receive their 01 hardware, with one noting they ordered it when it was first announced.
   - *grownbabyyoda* echoed the sentiment, wondering the same about the delivery timeline.
- **Issues with current 01 usage instructions**: A member requested documentation on using 01, mentioning that the current instructions do not work.
   - They asked for clarity on whether setting up Open Interpreter is a prerequisite for using 01 or if 01 can stand alone.


  

---



### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1262872924730429471)** (1 messages): 

> - `Torchtune v0.2.0 release`
> - `New models and recipes`
> - `Sample packing`
> - `Community contributions` 


- **Torchtune v0.2.0 is out!**: The release of [torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) includes many updates, new models, and improvements to datasets like sample packing.
   - *It's a culmination of months of contributions from the community.*
- **New models and datasets in Torchtune v0.2.0**: The new version brings a host of new models 🦙 and recipes along with impressive dataset improvements such as sample packing 🚀.
   - Check out the release notes for a list of all community members who contributed features to this release.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/releases/tag/v0.2.0">Release v0.2.0 · pytorch/torchtune</a>: Overview It’s been awhile since we’ve done a release and we have a ton of cool, new features in the torchtune library including distributed QLoRA support, new models, sample packing, and more! Chec...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1262923622238916710)** (10 messages🔥): 

> - `LLAMA 3 finetuning issues`
> - `Torchtune nightly installations`
> - `Stable Torchtune version` 


- **LLAMA 3 finetuning reveals tag issues**: A member observed visible and repeating **<|finetune_right_pad_id|>** tags in **LLAMA 3** generations after a short finetuning, instead of the expected <|end_of_text|> tag.
   - Another member noted that this tag is new since the refactor but not used in the tokenizer, suggesting that the issue might be related to the new implementation.
- **Switch from Torchtune nightlies to stable version**: A member suggested switching from the **Torchtune nightly** version to the stable version to resolve the tagging issue with LLAMA 3.
   - They mentioned that the stable version has the old implementation of the tokenizer and will begin investigating the issue in the meantime.



**Link mentioned**: <a href="https://download.pytorch.org/whl/nightly/cpu">no title found</a>: no description found

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

terafo: It's available now
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1262894248995328080)** (2 messages): 

> - `Notes on removal of linearizer`
> - `Message format clarification` 


- **Request for Updated Notes Post-Linearizer Removal**: A member praised another's notes and inquired if they would provide updated notes following the removal of the **linearizer**.
   - *Your notes are great, are you going to an updated notes following the removal of linearizer?*
- **Clarification on Where to Find Colors**: A message conveyed that the colors are described at the bottom of the first page.
   - *The colours are described at the bottom of the first page.*


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262866819233087561)** (2 messages): 

> - `OpenAI access requirements`
> - `LLM rule checker for hospital bills`
> - `Python code for billing rules` 


- **OpenAI API access needed**: Mentioned that **access on the OpenAI side** is required for certain functionalities to work, as confirmed by Kyle.
- **LLM as a rule checker for hospital billing**: Discussion on using **LLM** to check if hospital bills are correctly coded by pulling relevant information from rules and regulations PDFs.
- **Generating Python code for bill checking**: Explored the idea of using **LLM** to rewrite regulations and rules as **Python code** to check hospital bills, potentially making the process faster and simpler.
   - Considered using the LLM to generate test cases for the code to ensure accuracy.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1262890983864008805)** (1 messages): 

> - `Channel Activity`
> - `User Engagement` 


- **Missed Channel Updates**: A user expressed disappointment about not checking the channel before July, 9, and missing out on key updates.
   - They regretted **not taking the time to check the channel** after that date, feeling it was a missed opportunity.
- **Lack of Follow-Up**: An additional note was made on the user's sentiment about not checking the channel regularly after July, 9.
   - The user expressed a sense of **missed engagement** with the ongoing conversations.


  

---



### **AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1262949188254306449)** (1 messages): 

> - `Developer Opportunities`
> - `HLS and WebRTC`
> - `Backend Development`
> - `TypeScript`
> - `MongoDB` 


- **Job Opportunity in HLS and WebRTC Development**: **Observe** is seeking a developer experienced in **HLS** and **WebRTC** with knowledge in **Vanilla JS**, **TypeScript**, and **MongoDB** for backend development.
- **Looking for Passionate Developers for Startups**: **Observe** is on the lookout for developers who are passionate about startups and have expertise in **HLS** and **WebRTC**.



**Link mentioned**: <a href="https://observeyourfuture.com)">no title found</a>: no description found

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1262895724065067008)** (1 messages): 

> - `Phoenix 2.0`
> - `OSS Discussion`
> - `New Features in Phoenix`
> - `Arize Product Stack` 


- **Phoenix 2.0 Product Update & Future Vision**: Join the virtual discussion on July 18th, 2024, for a comprehensive overview of new features in **Phoenix 2.0**, including a new hosted deployment option, experimentation capabilities, and new integrations. Register [here](https://arize.com/resource/phoenix/2.0) to learn why OSS is critical to continued AI development.
   - This event will cap off Phoenix 2.0 Launch Week, offering demos and a live Q&A session to discuss Phoenix's role in developing LLM apps within the larger **Arize product stack**.
- **Town Hall on OSS Critical to AI Development**: The town hall will cover Phoenix 2.0 features like hosted deployment and new experimentation capabilities. The session will explain the vision of the future of Phoenix and the importance of OSS in AI.
   - Feedback from users has been crucial for **Phoenix** development, and the event will include a live Q&A session to foster continued dialogue.



**Link mentioned**: <a href="https://arize.com/resource/phoenix/2.0">Phoenix 2.0 Launch Week Town Hall</a>:   July 18th, 2024   10:00am PST &#8211; 11:00am PST Virtual Come join us as we cap off the Launch Week of Phoenix 2.0. In this town hall, we&#8217;ll cover...

  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1263150806102970368)** (1 messages): 

> - `Python SDK updates`
> - `Async client support`
> - `Jamba-Instruct examples` 


- **New updates to AI21 Python SDK!**: The latest update to the Python SDK now includes client support for **Jamba-Instruct** on **Amazon Bedrock** and **Azure AI Studio**. [Check out the update here](https://github.com/AI21Labs/ai21-python).
   - Additionally, async client support is now available across all platforms, along with new examples to facilitate the onboarding process. More details can be found on their [LinkedIn post](https://www.linkedin.com/posts/ai21_github-ai21labsai21-python-ai21-python-activity-7219341078116597762-Sxx5).
- **Async client support now universal!**: **Async client support** for **Jamba-Instruct** is now offered across all platforms: SaaS, Amazon Bedrock, and Azure. 📖 Visit the [GitHub repository](https://github.com/AI21Labs/ai21-python) for more information.
   - New examples have been added to streamline the development process with Jamba-Instruct, ensuring an easier start across various platforms.



**Link mentioned**: <a href="https://github.com/AI21Labs/ai21-python">GitHub - AI21Labs/ai21-python: AI21 Python SDK</a>: AI21 Python SDK. Contribute to AI21Labs/ai21-python development by creating an account on GitHub.

  

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
